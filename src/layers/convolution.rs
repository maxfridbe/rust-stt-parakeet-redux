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
            // Channels last lets each spatial tap update adjacent output
            // channels together, which the compiler can vectorize on WASM.
            weight: Array4::from_shape_vec(
                shape,
                weights.floats(&format!("{name}.weight"), &shape)?,
            )?
            .permuted_axes([2, 3, 1, 0])
            .as_standard_layout()
            .into_owned(),
            bias: weights.vector(&format!("{name}.bias"), output)?,
            stride,
            depthwise,
        })
    }

    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let input = input.as_standard_layout();
        let (time, frequency, channels) = input.dim();
        let kernel = self.weight.shape()[0];
        let output_time = time.div_ceil(self.stride);
        let output_frequency = frequency.div_ceil(self.stride);
        let output_channels = self.bias.len();
        if kernel == 1 && self.stride == 1 && !self.depthwise {
            let flattened = input
                .view()
                .into_shape_with_order((time * frequency, channels))
                .expect("contiguous input preserves its element count");
            let weight = self
                .weight
                .view()
                .into_shape_with_order((channels, output_channels))
                .expect("pointwise weights preserve their element count");
            return (flattened.dot(&weight) + &self.bias)
                .into_shape_with_order((time, frequency, output_channels))
                .expect("pointwise output preserves its element count");
        }
        let input = input.as_slice().expect("input layout was normalized");
        let weights = self
            .weight
            .as_slice()
            .expect("loaded weights are contiguous");
        let bias = self.bias.as_slice().expect("loaded bias is contiguous");
        let mut output = Array3::zeros((output_time, output_frequency, output_channels));
        let padding = kernel / 2;
        let tap_size = if self.depthwise {
            output_channels
        } else {
            channels * output_channels
        };
        for (position, row) in output
            .as_slice_mut()
            .expect("new output is contiguous")
            .chunks_exact_mut(output_channels)
            .enumerate()
        {
            row.copy_from_slice(bias);
            let output_t = position / output_frequency * self.stride;
            let output_f = position % output_frequency * self.stride;
            for kt in 0..kernel {
                let Some(t) = (output_t + kt).checked_sub(padding).filter(|&t| t < time) else {
                    continue;
                };
                for kf in 0..kernel {
                    let Some(f) = (output_f + kf)
                        .checked_sub(padding)
                        .filter(|&f| f < frequency)
                    else {
                        continue;
                    };
                    let source = &input[(t * frequency + f) * channels..][..channels];
                    let tap = &weights[(kt * kernel + kf) * tap_size..][..tap_size];
                    accumulate_channels(row, source, tap, self.depthwise);
                }
            }
        }
        output
    }
}

fn accumulate_channels(output: &mut [f32], input: &[f32], weights: &[f32], depthwise: bool) {
    if depthwise {
        for ((value, activation), weight) in output.iter_mut().zip(input).zip(weights) {
            *value += activation * weight;
        }
        return;
    }
    for (activation, weights) in input.iter().zip(weights.chunks_exact(output.len())) {
        for (value, weight) in output.iter_mut().zip(weights) {
            *value += activation * weight;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channelwise_spatial_updates_match_scalar_convolution() {
        for (kernel, stride, depthwise) in [(3, 2, false), (3, 2, true), (1, 1, false)] {
            let input =
                Array3::from_shape_fn((5, 4, 2), |(t, f, c)| (t * 8 + f * 2 + c) as f32 * 0.01);
            let output_channels = if depthwise { 2 } else { 3 };
            let weight = Array4::from_shape_fn(
                (
                    kernel,
                    kernel,
                    if depthwise { 1 } else { 2 },
                    output_channels,
                ),
                |(kt, kf, c, out)| (kt + kf + c + out) as f32 * 0.02 - 0.05,
            );
            let layer = Conv2d {
                weight,
                bias: Vector::from_elem(output_channels, 0.1),
                stride,
                depthwise,
            };
            let expected = Array3::from_shape_fn(
                (
                    5_usize.div_ceil(stride),
                    4_usize.div_ceil(stride),
                    output_channels,
                ),
                |(t, f, out)| {
                    let mut sum = layer.bias[out];
                    for kt in 0..kernel {
                        for kf in 0..kernel {
                            let source_t = (t * stride + kt) as isize - (kernel / 2) as isize;
                            let source_f = (f * stride + kf) as isize - (kernel / 2) as isize;
                            if !(0..5).contains(&source_t) || !(0..4).contains(&source_f) {
                                continue;
                            }
                            for c in 0..if depthwise { 1 } else { 2 } {
                                let channel = if depthwise { out } else { c };
                                sum += input[(source_t as usize, source_f as usize, channel)]
                                    * layer.weight[(kt, kf, c, out)];
                            }
                        }
                    }
                    sum
                },
            );
            let actual = layer.forward(&input);
            assert_eq!(actual.shape(), expected.shape());
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }
}
