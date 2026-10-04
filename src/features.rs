//! NeMo-compatible, 128-bin log-mel features for mono 16 kHz PCM.

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::{Error, Result, tensor::Matrix};

pub const SAMPLE_RATE: usize = 16_000;
pub const HOP_LENGTH: usize = 160;
const FFT_SIZE: usize = 512;
const WINDOW_LENGTH: usize = 400;
const MEL_BINS: usize = 128;

/// Reusable FFT plan, symmetric Hann window, and Slaney mel filter bank.
pub struct FeatureExtractor {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filters: Matrix,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        let fft = FftPlanner::new().plan_fft_forward(FFT_SIZE);
        let window = (0..WINDOW_LENGTH)
            .map(|index| {
                0.5 - 0.5
                    * (std::f32::consts::TAU * index as f32 / (WINDOW_LENGTH - 1) as f32).cos()
            })
            .collect();
        Self {
            fft,
            window,
            filters: mel_filters(),
        }
    }
}

impl FeatureExtractor {
    /// Extract time-major features with per-channel sample-variance normalization.
    ///
    /// # Errors
    /// Returns an error for non-finite samples or audio shorter than two hops.
    pub fn extract(&self, samples: &[f32]) -> Result<Matrix> {
        if samples.len() < 2 * HOP_LENGTH {
            return Err(Error::InvalidAudio(
                "at least 20 ms of audio is required".into(),
            ));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InvalidAudio("samples must be finite".into()));
        }
        let emphasized: Vec<_> = samples
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                value
                    - index
                        .checked_sub(1)
                        .map_or(0.0, |previous| 0.97 * samples[previous])
            })
            .collect();
        // The centered STFT has one extra frame; NeMo's length mask excludes it.
        let frames = samples.len() / HOP_LENGTH;
        let mut powers = Matrix::zeros((frames, FFT_SIZE / 2 + 1));
        let mut buffer = vec![Complex::default(); FFT_SIZE];
        let mut scratch = vec![Complex::default(); self.fft.get_inplace_scratch_len()];
        for (frame, mut power) in powers.outer_iter_mut().enumerate() {
            buffer.fill(Complex::default());
            self.fill_window(&emphasized, frame, &mut buffer);
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for (value, bin) in power.iter_mut().zip(&buffer) {
                *value = bin.norm_sqr();
            }
        }
        let mut features = powers.dot(&self.filters.t());
        features.mapv_inplace(|value| (value + 2.0_f32.powi(-24)).ln());
        normalize(&mut features);
        Ok(features)
    }

    fn fill_window(&self, samples: &[f32], frame: usize, buffer: &mut [Complex<f32>]) {
        let window_offset = (FFT_SIZE - WINDOW_LENGTH) / 2;
        for (index, &window) in self.window.iter().enumerate() {
            let source = (frame * HOP_LENGTH + index).checked_sub(WINDOW_LENGTH / 2);
            let sample = source
                .and_then(|index| samples.get(index))
                .copied()
                .unwrap_or(0.0);
            buffer[window_offset + index].re = sample * window;
        }
    }
}

fn normalize(features: &mut Matrix) {
    for mut channel in features.columns_mut() {
        let mean = channel.sum() / channel.len() as f32;
        let variance = channel
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / (channel.len() - 1) as f32;
        let denominator = variance.sqrt() + 1e-5;
        channel.mapv_inplace(|value| (value - mean) / denominator);
    }
}

fn mel_to_hertz(mel: f64) -> f64 {
    if mel < 15.0 {
        return mel * (200.0 / 3.0);
    }
    1000.0 * ((mel - 15.0) * (6.4_f64.ln() / 27.0)).exp()
}

fn mel_filters() -> Matrix {
    let maximum = 15.0 + (8000.0_f64 / 1000.0).ln() / (6.4_f64.ln() / 27.0);
    let edges: Vec<_> = (0..MEL_BINS + 2)
        .map(|index| mel_to_hertz(maximum * index as f64 / (MEL_BINS + 1) as f64))
        .collect();
    Matrix::from_shape_fn((MEL_BINS, FFT_SIZE / 2 + 1), |(mel, bin)| {
        let frequency = bin as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
        let rising = (frequency - edges[mel]) / (edges[mel + 1] - edges[mel]);
        let falling = (edges[mel + 2] - frequency) / (edges[mel + 2] - edges[mel + 1]);
        // librosa stores the triangular weights as f32 before applying the norm.
        let triangle = rising.min(falling).max(0.0) as f32;
        (f64::from(triangle) * (2.0 / (edges[mel + 2] - edges[mel]))) as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_finite_and_centered() {
        let features = FeatureExtractor::default()
            .extract(&vec![0.0; 16_000])
            .unwrap();
        assert_eq!(features.dim(), (100, 128));
        assert!(
            features
                .iter()
                .all(|value| value.is_finite() && value.abs() < 1.0)
        );
    }

    #[test]
    fn rejects_short_and_nonfinite_audio() {
        let extractor = FeatureExtractor::default();
        assert!(extractor.extract(&[]).is_err());
        assert!(extractor.extract(&[f32::NAN; 320]).is_err());
    }
}
