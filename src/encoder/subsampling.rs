use ndarray::Array3;

use crate::{
    Result,
    config::EncoderConfig,
    layers::{Conv2d, Linear},
    tensor::Matrix,
    weights::Weights,
};

pub(crate) struct Subsampling {
    initial: Conv2d,
    stages: Vec<(Conv2d, Conv2d)>,
    projection: Linear,
}

impl Subsampling {
    pub fn load(weights: &Weights<'_>, config: &EncoderConfig) -> Result<Self> {
        let channels = config.subsampling_conv_channels;
        let initial = Conv2d::load(
            weights,
            "encoder.subsampling.layers.0",
            1,
            channels,
            3,
            2,
            false,
        )?;
        let stages = [2, 5]
            .into_iter()
            .map(|index| {
                let depthwise = Conv2d::load(
                    weights,
                    &format!("encoder.subsampling.layers.{index}"),
                    channels,
                    channels,
                    3,
                    2,
                    true,
                )?;
                let pointwise = Conv2d::load(
                    weights,
                    &format!("encoder.subsampling.layers.{}", index + 1),
                    channels,
                    channels,
                    1,
                    1,
                    false,
                )?;
                Ok((depthwise, pointwise))
            })
            .collect::<Result<_>>()?;
        let projection = Linear::load(
            weights,
            "encoder.subsampling.linear",
            channels * (config.num_mel_bins / 8),
            config.hidden_size,
            true,
        )?;
        Ok(Self {
            initial,
            stages,
            projection,
        })
    }

    pub fn forward(&self, features: &Matrix) -> Matrix {
        let input = Array3::from_shape_fn((features.nrows(), features.ncols(), 1), |(t, f, _)| {
            features[(t, f)]
        });
        let mut hidden = self.initial.forward(&input);
        hidden.mapv_inplace(|value| value.max(0.0));
        for (depthwise, pointwise) in &self.stages {
            hidden = pointwise.forward(&depthwise.forward(&hidden));
            hidden.mapv_inplace(|value| value.max(0.0));
        }
        let (time, frequency, channels) = hidden.dim();
        // PyTorch flattens channels before frequency after swapping time/channel.
        let flattened = Matrix::from_shape_fn((time, channels * frequency), |(t, column)| {
            hidden[(t, column % frequency, column / frequency)]
        });
        self.projection.forward(&flattened)
    }
}
