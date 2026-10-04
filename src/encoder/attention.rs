use ndarray::{Axis, Slice};

use crate::{
    Result,
    config::EncoderConfig,
    layers::PackedLinear,
    tensor::{Matrix, softmax},
    weights::Weights,
};

pub(crate) struct Attention {
    query: PackedLinear,
    key: PackedLinear,
    value: PackedLinear,
    output: PackedLinear,
    relative_key: PackedLinear,
    content_bias: Matrix,
    position_bias: Matrix,
    heads: usize,
}

impl Attention {
    pub fn load(
        weights: &Weights<'_>,
        name: &str,
        config: &EncoderConfig,
        group_size: usize,
    ) -> Result<Self> {
        let width = config.hidden_size;
        let projection = |suffix| {
            PackedLinear::load(
                weights,
                &format!("{name}.{suffix}"),
                width,
                width,
                group_size,
            )
        };
        let heads = config.num_attention_heads;
        Ok(Self {
            query: projection("q_proj")?,
            key: projection("k_proj")?,
            value: projection("v_proj")?,
            output: projection("o_proj")?,
            relative_key: projection("relative_k_proj")?,
            content_bias: weights.matrix(&format!("{name}.bias_u"), heads, width / heads)?,
            position_bias: weights.matrix(&format!("{name}.bias_v"), heads, width / heads)?,
            heads,
        })
    }

    pub fn forward(&self, input: &Matrix, positions: &Matrix) -> Matrix {
        let query = self.query.forward(input);
        let key = self.key.forward(input);
        let value = self.value.forward(input);
        let relative_key = self.relative_key.forward(positions);
        let head_width = input.ncols() / self.heads;
        let mut output = Matrix::zeros(input.raw_dim());
        for head in 0..self.heads {
            let columns = head * head_width..(head + 1) * head_width;
            let queries = query.slice_axis(Axis(1), Slice::from(columns.clone()));
            let content_queries = &queries + &self.content_bias.row(head);
            let position_queries = &queries + &self.position_bias.row(head);
            let mut scores =
                content_queries.dot(&key.slice_axis(Axis(1), Slice::from(columns.clone())).t());
            let relative_scores = position_queries.dot(
                &relative_key
                    .slice_axis(Axis(1), Slice::from(columns.clone()))
                    .t(),
            );
            add_relative_scores(
                &mut scores,
                &relative_scores,
                (head_width as f32).sqrt().recip(),
            );
            for mut row in scores.rows_mut() {
                // Scores are newly allocated in standard contiguous layout.
                if let Some(values) = row.as_slice_mut() {
                    softmax(values);
                }
            }
            let attended = scores.dot(&value.slice_axis(Axis(1), Slice::from(columns.clone())));
            output
                .slice_axis_mut(Axis(1), Slice::from(columns))
                .assign(&attended);
        }
        self.output.forward(&output)
    }
}

fn add_relative_scores(scores: &mut Matrix, relative: &Matrix, scale: f32) {
    let frames = scores.nrows();
    for ((query, key), score) in scores.indexed_iter_mut() {
        // Equivalent to Transformer-XL's relative shift without padding/reshapes.
        *score = (*score + relative[(query, frames - 1 - query + key)]) * scale;
    }
}

pub(crate) fn positional_encoding(frames: usize, width: usize) -> Matrix {
    Matrix::from_shape_fn((2 * frames - 1, width), |(row, column)| {
        let position = frames as f32 - 1.0 - row as f32;
        let frequency = 10000.0_f32.powf(-((column / 2 * 2) as f32) / width as f32);
        let angle = position * frequency;
        if column % 2 == 0 {
            angle.sin()
        } else {
            angle.cos()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn relative_shift_matches_transformer_xl_layout() {
        let mut scores = Matrix::zeros((3, 3));
        let relative = array![
            [0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.],
            [10., 11., 12., 13., 14.]
        ];
        add_relative_scores(&mut scores, &relative, 1.0);
        assert_eq!(scores, array![[2., 3., 4.], [6., 7., 8.], [10., 11., 12.]]);
    }
}
