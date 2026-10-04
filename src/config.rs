use serde::Deserialize;

use crate::{Result, error::invalid};

/// Architecture recorded in the upstream `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub blank_token_id: usize,
    pub decoder_hidden_size: usize,
    pub durations: Vec<usize>,
    pub encoder_config: EncoderConfig,
    pub max_symbols_per_step: usize,
    pub model_type: String,
    pub num_decoder_layers: usize,
    pub vocab_size: usize,
    pub ternary_group_size: usize,
    pub hidden_act: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub num_mel_bins: usize,
    pub conv_kernel_size: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub subsampling_factor: usize,
    pub scale_input: bool,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub convolution_bias: bool,
}

impl ModelConfig {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self> {
        let config: Self = serde_json::from_slice(bytes)?;
        let encoder = &config.encoder_config;
        if config.model_type != "parakeet_tdt"
            || config.hidden_act != "relu"
            || config.blank_token_id >= config.vocab_size
            || config.decoder_hidden_size == 0
            || config.num_decoder_layers == 0
            || config.ternary_group_size == 0
            || config.max_symbols_per_step == 0
            || config.durations.is_empty()
        {
            return Err(invalid("unsupported or invalid TDT configuration"));
        }
        if encoder.hidden_size == 0
            || encoder.hidden_act != "silu"
            || encoder.attention_bias
            || encoder.convolution_bias
            || !encoder.hidden_size.is_multiple_of(2)
            || encoder.num_attention_heads == 0
            || !encoder
                .hidden_size
                .is_multiple_of(encoder.num_attention_heads)
            || encoder.num_key_value_heads != encoder.num_attention_heads
            || encoder.num_hidden_layers == 0
            || encoder.intermediate_size == 0
            || encoder.conv_kernel_size == 0
            || encoder.conv_kernel_size.is_multiple_of(2)
            || encoder.num_mel_bins != 128
            || encoder.subsampling_factor != 8
            || encoder.subsampling_conv_stride != 2
            || encoder.subsampling_conv_kernel_size != 3
            || encoder.subsampling_conv_channels == 0
        {
            return Err(invalid("unsupported or invalid encoder configuration"));
        }
        Ok(config)
    }
}
