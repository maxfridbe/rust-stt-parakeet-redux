//! Portable inference for the packed Moondream Parakeet Redux checkpoint.
//!
//! Supply model files as bytes and mono, 16 kHz PCM as `f32` samples. The core
//! does not access the filesystem, create threads, or require a native ML runtime.

mod config;
mod decoder;
mod encoder;
mod error;
pub mod features;
mod layers;
mod model;
mod tensor;
mod tokenizer;
mod transcript;
mod vad;
mod weights;

#[cfg(feature = "wasm")]
mod wasm;

pub use config::{EncoderConfig, ModelConfig};
pub use error::{Error, Result};
pub use model::{Model, TranscriptionOptions};
pub use transcript::{Segment, Token, Transcript, Word};
