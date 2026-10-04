mod attention;
mod conformer;
mod subsampling;

use crate::{ModelConfig, Result, tensor::Matrix, weights::Weights};
use attention::positional_encoding;
use conformer::ConformerBlock;
use subsampling::Subsampling;

pub(crate) struct Encoder {
    subsampling: Subsampling,
    blocks: Vec<ConformerBlock>,
    scale_input: bool,
}

impl Encoder {
    pub fn load(weights: &Weights<'_>, config: &ModelConfig) -> Result<Self> {
        let encoder = &config.encoder_config;
        let blocks = (0..encoder.num_hidden_layers)
            .map(|index| ConformerBlock::load(weights, index, encoder, config.ternary_group_size))
            .collect::<Result<_>>()?;
        Ok(Self {
            subsampling: Subsampling::load(weights, encoder)?,
            blocks,
            scale_input: encoder.scale_input,
        })
    }

    pub fn subsample(&self, features: &Matrix) -> Matrix {
        self.subsampling.forward(features)
    }

    pub fn encode(&self, mut hidden: Matrix) -> Matrix {
        let positions = positional_encoding(hidden.nrows(), hidden.ncols());
        if self.scale_input {
            hidden *= (hidden.ncols() as f32).sqrt();
        }
        for block in &self.blocks {
            hidden = block.forward(&hidden, &positions);
        }
        hidden
    }
}
