use crate::{Model, TranscriptionOptions};
use wasm_bindgen::prelude::*;

/// Owns a loaded model in WebAssembly memory. Run in a Worker: inference is synchronous.
#[wasm_bindgen]
pub struct WasmModel {
    inner: Model,
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
            .map(|inner| Self { inner })
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
}
