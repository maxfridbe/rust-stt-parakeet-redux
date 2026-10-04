use std::ops::Range;

use crate::{
    Error, ModelConfig, Result, Transcript,
    decoder::Decoder,
    encoder::Encoder,
    error::invalid,
    features::{FeatureExtractor, SAMPLE_RATE},
    tokenizer::Tokenizer,
    vad::{VadHead, next_cut, speech_regions},
    weights::{Weights, validate_manifest},
};

/// Long recordings are cut at VAD pauses, with a hard upper bound per chunk.
#[derive(Debug, Clone)]
pub struct TranscriptionOptions {
    /// Must be between 1 and 30 seconds. Defaults to 30 seconds.
    pub max_segment_seconds: usize,
    /// Use the checkpoint's VAD head to select cuts and skip silent long chunks.
    pub use_vad: bool,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            max_segment_seconds: 30,
            use_vad: true,
        }
    }
}

/// Loaded weights and reusable signal-processing state. Calls reset the decoder,
/// so separate recordings never inherit each other's linguistic context.
pub struct Model {
    config: ModelConfig,
    features: FeatureExtractor,
    encoder: Encoder,
    decoder: Decoder,
    tokenizer: Tokenizer,
    vad: VadHead,
}

impl Model {
    /// Load the four unmodified files from `moondream/parakeet-redux`.
    ///
    /// # Errors
    /// Rejects unsupported configuration, missing tensors, incompatible tensor
    /// shapes/dtypes, malformed packed weights, and incompatible tokenizers.
    pub fn from_bytes(
        config: &[u8],
        manifest: &[u8],
        tokenizer: &[u8],
        safetensors: &[u8],
    ) -> Result<Self> {
        let config = ModelConfig::parse(config)?;
        let weights = Weights::new(safetensors)?;
        let count = validate_manifest(manifest, config.ternary_group_size, &weights)?;
        if count != config.encoder_config.num_hidden_layers * 11 {
            return Err(invalid("ternary module count does not match the encoder"));
        }
        Ok(Self {
            encoder: Encoder::load(&weights, &config)?,
            decoder: Decoder::load(&weights, &config)?,
            tokenizer: Tokenizer::load(tokenizer, config.vocab_size, config.blank_token_id)?,
            vad: VadHead::load(&weights, config.encoder_config.hidden_size)?,
            features: FeatureExtractor::default(),
            config,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Transcribe normalized mono PCM at exactly 16 kHz.
    ///
    /// # Errors
    /// Rejects audio shorter than 20 ms, non-finite samples, invalid options,
    /// or a decoder that exhausts its step budget without consuming the audio.
    pub fn transcribe(
        &self,
        samples: &[f32],
        options: &TranscriptionOptions,
    ) -> Result<Transcript> {
        self.transcribe_with_progress(samples, options, |_, _| {})
    }

    /// As [`Self::transcribe`], reporting `(samples_processed, total_samples)`
    /// after each chunk. The callback runs synchronously on the caller's thread.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::transcribe`].
    pub fn transcribe_with_progress(
        &self,
        samples: &[f32],
        options: &TranscriptionOptions,
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Transcript> {
        if !(1..=30).contains(&options.max_segment_seconds) {
            return Err(Error::InvalidAudio(
                "segment length must be between 1 and 30 seconds".into(),
            ));
        }
        if samples.len() < 320 || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InvalidAudio(
                "provide at least 20 ms of finite mono 16 kHz samples".into(),
            ));
        }
        let cap = options.max_segment_seconds * SAMPLE_RATE;
        let regions = if options.use_vad && samples.len() > cap {
            Some(self.scan_speech(samples)?)
        } else {
            None
        };
        let mut transcript = Transcript {
            duration_seconds: samples.len() as f64 / SAMPLE_RATE as f64,
            ..Transcript::default()
        };
        let mut start = 0;
        while start < samples.len() {
            let end = self.chunk_end(samples.len(), start, cap, regions.as_deref());
            let has_speech = regions.as_ref().is_none_or(|regions| {
                regions
                    .iter()
                    .any(|region| region.start < end && region.end > start)
            });
            if end - start >= 320 && has_speech {
                self.transcribe_chunk(&samples[start..end], start, &mut transcript)?;
            }
            start = end;
            progress(start, samples.len());
        }
        Ok(transcript)
    }

    fn chunk_end(
        &self,
        total: usize,
        start: usize,
        cap: usize,
        regions: Option<&[Range<usize>]>,
    ) -> usize {
        if total - start <= cap {
            return total;
        }
        regions.map_or(start + cap, |regions| next_cut(regions, start, total, cap))
    }

    fn transcribe_chunk(
        &self,
        samples: &[f32],
        offset: usize,
        transcript: &mut Transcript,
    ) -> Result<()> {
        let features = self.features.extract(samples)?;
        let encoded = self.encoder.encode(self.encoder.subsample(&features));
        let emissions = self.decoder.decode(&encoded)?;
        transcript.append(
            &emissions,
            &self.tokenizer,
            offset as f64 / SAMPLE_RATE as f64,
            (offset + samples.len()) as f64 / SAMPLE_RATE as f64,
        );
        Ok(())
    }

    fn scan_speech(&self, samples: &[f32]) -> Result<Vec<Range<usize>>> {
        let mut regions = Vec::new();
        let block_size = 120 * SAMPLE_RATE;
        for (index, block) in samples.chunks(block_size).enumerate() {
            let offset = index * block_size;
            if block.len() < 320 {
                regions.push(offset..offset + block.len());
                continue;
            }
            let features = self.features.extract(block)?;
            let hidden = self.encoder.subsample(&features);
            let probabilities = self.vad.probabilities(&hidden);
            regions.extend(speech_regions(&probabilities).into_iter().map(|region| {
                (offset + region.start * 1280)..(offset + region.end * 1280).min(samples.len())
            }));
        }
        Ok(regions)
    }
}

#[cfg(test)]
mod tests;
