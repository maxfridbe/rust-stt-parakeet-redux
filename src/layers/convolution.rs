use ndarray::{Array3, Array4};

use crate::{
    Result,
    tensor::{Matrix, Vector},
    weights::Weights,
};

/// Channels-last working layout: (time, frequency, channels).
pub(crate) struct Conv2d {
    weight: Array4<f32>,
    bias: Vector,
    stride: usize,
    depthwise: bool,
}

impl Conv2d {
    pub fn load(
        weights: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        depthwise: bool,
    ) -> Result<Self> {
        let shape = [output, if depthwise { 1 } else { input }, kernel, kernel];
        Ok(Self {
            weight: Array4::from_shape_vec(
                shape,
                weights.floats(&format!("{name}.weight"), &shape)?,
            )?,
            bias: weights.vector(&format!("{name}.bias"), output)?,
            stride,
            depthwise,
        })
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (time, frequency, channels) = input.dim();
        let kernel = self.weight.shape()[2];
        let output_time = time.div_ceil(self.stride);
        let output_frequency = frequency.div_ceil(self.stride);
        let output_channels = self.bias.len();
        if kernel == 1 {
            let flattened =
                Matrix::from_shape_fn((time * frequency, channels), |(position, channel)| {
                    input[(position / frequency, position % frequency, channel)]
                });
            let weight = Matrix::from_shape_fn((output_channels, channels), |(out, channel)| {
                self.weight[(out, channel, 0, 0)]
            });
            let projected = flattened.dot(&weight.t()) + &self.bias;
            return Array3::from_shape_fn((time, frequency, output_channels), |(t, f, c)| {
                projected[(t * frequency + f, c)]
            });
        }
        Array3::from_shape_fn(
            (output_time, output_frequency, output_channels),
            |(t, f, out)| self.spatial_sum(input, t, f, out, kernel, channels),
        )
    }

    fn spatial_sum(
        &self,
        input: &Array3<f32>,
        time: usize,
        frequency: usize,
        output: usize,
        kernel: usize,
        channels: usize,
    ) -> f32 {
        let mut sum = self.bias[output];
        let padding = kernel / 2;
        for kt in 0..kernel {
            let Some(t) = (time * self.stride + kt)
                .checked_sub(padding)
                .filter(|&t| t < input.shape()[0])
            else {
                continue;
            };
            for kf in 0..kernel {
                let Some(f) = (frequency * self.stride + kf)
                    .checked_sub(padding)
                    .filter(|&f| f < input.shape()[1])
                else {
                    continue;
                };
                if self.depthwise {
                    sum += input[(t, f, output)] * self.weight[(output, 0, kt, kf)];
                    continue;
                }
                for channel in 0..channels {
                    sum += input[(t, f, channel)] * self.weight[(output, channel, kt, kf)];
                }
            }
        }
        sum
    }
}

pub(crate) struct TemporalConvolution {
    weight: Array3<f32>,
    bias: Vector,
    depthwise: bool,
}

impl TemporalConvolution {
    pub fn load(
        weights: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        depthwise: bool,
        bias: bool,
    ) -> Result<Self> {
        let shape = [output, if depthwise { 1 } else { input }, kernel];
        Ok(Self {
            weight: Array3::from_shape_vec(
                shape,
                weights.floats(&format!("{name}.weight"), &shape)?,
            )?,
            bias: if bias {
                weights.vector(&format!("{name}.bias"), output)?
            } else {
                Vector::zeros(output)
            },
            depthwise,
        })
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        let kernel = self.weight.shape()[2];
        let mut output = Matrix::zeros((input.nrows(), self.bias.len()));
        for (time, mut row) in output.outer_iter_mut().enumerate() {
            row.assign(&self.bias);
            for offset in 0..kernel {
                let Some(source) = (time + offset)
                    .checked_sub(kernel / 2)
                    .filter(|&t| t < input.nrows())
                else {
                    continue;
                };
                if self.depthwise {
                    for (channel, value) in row.iter_mut().enumerate() {
                        *value += input[(source, channel)] * self.weight[(channel, 0, offset)];
                    }
                    continue;
                }
                for (channel, value) in row.iter_mut().enumerate() {
                    for (input_channel, &activation) in input.row(source).iter().enumerate() {
                        *value += activation * self.weight[(channel, input_channel, offset)];
                    }
                }
            }
        }
        output
    }
}
