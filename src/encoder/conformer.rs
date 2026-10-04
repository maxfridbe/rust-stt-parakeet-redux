use ndarray::{Axis, Slice};

use super::attention::Attention;
use crate::{
    Result,
    config::EncoderConfig,
    layers::{BatchNorm, LayerNorm, PackedLinear, TemporalConvolution},
    tensor::{Matrix, sigmoid, silu},
    weights::Weights,
};

struct FeedForward {
    norm: LayerNorm,
    expand: PackedLinear,
    contract: PackedLinear,
}

impl FeedForward {
    fn load(
        weights: &Weights<'_>,
        name: &str,
        number: usize,
        config: &EncoderConfig,
        group_size: usize,
    ) -> Result<Self> {
        let prefix = format!("{name}.feed_forward{number}");
        Ok(Self {
            norm: LayerNorm::load(
                weights,
                &format!("{name}.norm_feed_forward{number}"),
                config.hidden_size,
            )?,
            expand: PackedLinear::load(
                weights,
                &format!("{prefix}.linear1"),
                config.hidden_size,
                config.intermediate_size,
                group_size,
            )?,
            contract: PackedLinear::load(
                weights,
                &format!("{prefix}.linear2"),
                config.intermediate_size,
                config.hidden_size,
                group_size,
            )?,
        })
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        let mut hidden = self.expand.forward(&self.norm.forward(input));
        silu(&mut hidden);
        input + self.contract.forward(&hidden) * 0.5
    }
}

struct Convolution {
    norm: LayerNorm,
    expand: PackedLinear,
    depthwise: TemporalConvolution,
    batch_norm: BatchNorm,
    contract: PackedLinear,
}

impl Convolution {
    fn load(
        weights: &Weights<'_>,
        name: &str,
        config: &EncoderConfig,
        group_size: usize,
    ) -> Result<Self> {
        let width = config.hidden_size;
        Ok(Self {
            norm: LayerNorm::load(weights, &format!("{name}.norm_conv"), width)?,
            expand: PackedLinear::load(
                weights,
                &format!("{name}.conv.pointwise_conv1"),
                width,
                2 * width,
                group_size,
            )?,
            depthwise: TemporalConvolution::load(
                weights,
                &format!("{name}.conv.depthwise_conv"),
                width,
                width,
                config.conv_kernel_size,
                true,
                false,
            )?,
            batch_norm: BatchNorm::load(weights, &format!("{name}.conv.norm"), width)?,
            contract: PackedLinear::load(
                weights,
                &format!("{name}.conv.pointwise_conv2"),
                width,
                width,
                group_size,
            )?,
        })
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        let expanded = self.expand.forward(&self.norm.forward(input));
        let width = input.ncols();
        let gated = &expanded.slice_axis(Axis(1), Slice::from(..width))
            * &expanded
                .slice_axis(Axis(1), Slice::from(width..))
                .mapv(sigmoid);
        let mut hidden = self.depthwise.forward(&gated);
        self.batch_norm.forward(&mut hidden);
        silu(&mut hidden);
        input + self.contract.forward(&hidden)
    }
}

pub(crate) struct ConformerBlock {
    first_feed_forward: FeedForward,
    attention_norm: LayerNorm,
    attention: Attention,
    convolution: Convolution,
    second_feed_forward: FeedForward,
    output_norm: LayerNorm,
}

impl ConformerBlock {
    pub fn load(
        weights: &Weights<'_>,
        index: usize,
        config: &EncoderConfig,
        group_size: usize,
    ) -> Result<Self> {
        let name = format!("encoder.layers.{index}");
        Ok(Self {
            first_feed_forward: FeedForward::load(weights, &name, 1, config, group_size)?,
            attention_norm: LayerNorm::load(
                weights,
                &format!("{name}.norm_self_att"),
                config.hidden_size,
            )?,
            attention: Attention::load(weights, &format!("{name}.self_attn"), config, group_size)?,
            convolution: Convolution::load(weights, &name, config, group_size)?,
            second_feed_forward: FeedForward::load(weights, &name, 2, config, group_size)?,
            output_norm: LayerNorm::load(weights, &format!("{name}.norm_out"), config.hidden_size)?,
        })
    }

    pub fn forward(&self, input: &Matrix, positions: &Matrix) -> Matrix {
        let mut hidden = self.first_feed_forward.forward(input);
        hidden += &self
            .attention
            .forward(&self.attention_norm.forward(&hidden), positions);
        hidden = self.convolution.forward(&hidden);
        hidden = self.second_feed_forward.forward(&hidden);
        self.output_norm.forward(&hidden)
    }
}
