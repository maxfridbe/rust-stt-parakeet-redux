use crate::{Model, TranscriptionOptions};
use wasm_bindgen::prelude::*;

/// Owns a loaded model in WebAssembly memory. Run in a Worker: inference is synchronous.
#[wasm_bindgen]
pub struct WasmModel {
    inner: Model,
    audio: Vec<f32>,
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: &[u8],
        manifest: &[u8],
        tokenizer: &[u8],
        weights: &[u8],
    ) -> std::result::Result<Self, JsValue> {
        Model::from_bytes(config, manifest, tokenizer, weights)
            .map(|inner| Self {
                inner,
                audio: Vec::new(),
            })
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    /// Accepts Float32Array mono PCM at 16 kHz and returns transcript JSON.
    pub fn transcribe(&self, samples: &[f32]) -> std::result::Result<String, JsValue> {
        let transcript = self
            .inner
            .transcribe(samples, &TranscriptionOptions::default())
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        serde_json::to_string(&transcript).map_err(|error| JsValue::from_str(&error.to_string()))
    }

    /// Resizes a reusable input buffer and returns its byte offset in WASM memory.
    ///
    /// After this call, create a fresh `Float32Array(memory.buffer, offset, sample_count)`
    /// and fill it before calling `transcribe_prepared`. Do not retain the view across
    /// WASM calls: resizing or memory growth can invalidate it. The model owns the buffer.
    ///
    /// # Errors
    /// Returns an error for fewer than 320 samples or an allocation failure.
    pub fn prepare_audio(&mut self, sample_count: usize) -> std::result::Result<usize, JsValue> {
        if sample_count < 320 {
            return Err(JsValue::from_str("audio must contain at least 320 samples"));
        }
        self.audio
            .try_reserve(sample_count.saturating_sub(self.audio.len()))
            .map_err(|error| {
                JsValue::from_str(&format!("could not allocate audio buffer: {error}"))
            })?;
        self.audio.resize(sample_count, 0.0);
        Ok(self.audio.as_mut_ptr() as usize)
    }

    /// Transcribes the prepared input without copying an array through the JS binding.
    ///
    /// # Errors
    /// Returns an error if input is missing, invalid, or transcription fails.
    pub fn transcribe_prepared(&self) -> std::result::Result<String, JsValue> {
        self.transcribe(&self.audio)
    }
}
