/// Errors encountered while loading a model or transcribing audio.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid model: {0}")]
    InvalidModel(String),
    #[error("invalid audio: {0}")]
    InvalidAudio(String),
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid safetensors: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("invalid tensor shape: {0}")]
    Shape(#[from] ndarray::ShapeError),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidModel(message.into())
}
