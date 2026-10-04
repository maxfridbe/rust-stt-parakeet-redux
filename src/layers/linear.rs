use crate::{
    Result,
    tensor::{Matrix, Vector},
    weights::Weights,
};

pub(crate) struct Linear {
    weight: Matrix,
    bias: Option<Vector>,
}

impl Linear {
    pub fn load(
        weights: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            weight: weights.matrix(&format!("{name}.weight"), output, input)?,
            bias: bias
                .then(|| weights.vector(&format!("{name}.bias"), output))
                .transpose()?,
        })
    }

    pub fn from_parts(weight: Matrix, bias: Vector) -> Self {
        Self {
            weight,
            bias: Some(bias),
        }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        let mut output = input.dot(&self.weight.t());
        if let Some(bias) = &self.bias {
            output += bias;
        }
        output
    }
}

/// Keep weights packed between calls. Only a single projection is expanded at a
/// time, avoiding a multi-gigabyte resident copy of the full encoder.
pub(crate) struct PackedLinear {
    packed: Vec<u8>,
    scales: Matrix,
    input: usize,
    output: usize,
    group_size: usize,
}

impl PackedLinear {
    pub fn load(
        weights: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        group_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            packed: weights.bytes(&format!("{name}.qweight"), &[output, input.div_ceil(5)])?,
            scales: weights.matrix(
                &format!("{name}.scales"),
                output,
                input.div_ceil(group_size),
            )?,
            input,
            output,
            group_size,
        })
    }

    fn expand(&self) -> Matrix {
        let mut expanded = Matrix::zeros((self.output, self.input));
        for (row_index, mut row) in expanded.outer_iter_mut().enumerate() {
            let packed_row = &self.packed[row_index * self.input.div_ceil(5)..];
            // A freshly allocated matrix is contiguous. Grouping before decoding
            // removes a variable integer division from every weight, especially
            // valuable on WASM, while preserving groups that cross packed bytes.
            let values = row.as_slice_mut().expect("new matrix rows are contiguous");
            for (group, values) in values.chunks_mut(self.group_size).enumerate() {
                let scale = self.scales[(row_index, group)];
                let start = group * self.group_size;
                for (offset, value) in values.iter_mut().enumerate() {
                    let column = start + offset;
                    *value =
                        TERNARY_DIGITS[usize::from(packed_row[column / 5])][column % 5] * scale;
                }
            }
        }
        expanded
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        input.dot(&self.expand().t())
    }
}

const TERNARY_DIGITS: [[f32; 5]; 243] = {
    let mut table = [[0.0; 5]; 243];
    let mut byte = 0;
    while byte < 243 {
        let mut remaining = byte;
        let mut digit = 0;
        while digit < 5 {
            table[byte][digit] = (remaining % 3) as f32 - 1.0;
            remaining /= 3;
            digit += 1;
        }
        byte += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ternary_digits_cross_scale_groups_and_ignore_row_padding() {
        // Digits [0,1,2,0,1], [2,pad,pad,pad,pad], then a new row.
        let layer = PackedLinear {
            packed: vec![102, 2, 242, 0],
            scales: array![[2., 3., 5.], [7., 11., 13.]],
            input: 6,
            output: 2,
            group_size: 2,
        };
        assert_eq!(
            layer.expand(),
            array![[-2., 0., 3., -3., 0., 5.], [7., 7., 11., 11., 13., -13.]]
        );
        assert_eq!(
            layer.forward(&array![[1., 2., 3., 4., 5., 6.]]),
            array![[25., 85.]]
        );
    }
}
