use crate::{
    Result,
    tensor::{Matrix, Vector},
    weights::Weights,
};

pub(crate) struct LayerNorm {
    weight: Vector,
    bias: Vector,
}

impl LayerNorm {
    pub fn load(weights: &Weights<'_>, name: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            weight: weights.vector(&format!("{name}.weight"), channels)?,
            bias: weights.vector(&format!("{name}.bias"), channels)?,
        })
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        let mut output = input.clone();
        for mut row in output.rows_mut() {
            let mean = row.sum() / row.len() as f32;
            let variance =
                row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / row.len() as f32;
            let inverse_std = (variance + 1e-5).sqrt().recip();
            for ((value, weight), bias) in row.iter_mut().zip(&self.weight).zip(&self.bias) {
                *value = (*value - mean) * inverse_std * weight + bias;
            }
        }
        output
    }
}

pub(crate) struct BatchNorm {
    scale: Vector,
    offset: Vector,
}

impl BatchNorm {
    pub fn load(weights: &Weights<'_>, name: &str, channels: usize) -> Result<Self> {
        let weight = weights.vector(&format!("{name}.weight"), channels)?;
        let bias = weights.vector(&format!("{name}.bias"), channels)?;
        let mean = weights.vector(&format!("{name}.running_mean"), channels)?;
        let variance = weights.vector(&format!("{name}.running_var"), channels)?;
        let scale = weight / variance.mapv(|value| (value + 1e-5).sqrt());
        let offset = bias - &mean * &scale;
        Ok(Self { scale, offset })
    }

    pub fn forward(&self, input: &mut Matrix) {
        *input *= &self.scale;
        *input += &self.offset;
    }
}
