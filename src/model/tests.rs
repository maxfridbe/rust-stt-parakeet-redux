use super::*;
use crate::tensor::Matrix;
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Fixture {
    samples: Vec<f32>,
    subsampled: Vec<Vec<f32>>,
    encoded: Vec<Vec<f32>>,
    vad: Vec<f32>,
}

fn assert_close(name: &str, actual: &Matrix, expected: &[Vec<f32>], tolerance: f32) {
    assert_eq!(actual.dim(), (expected.len(), expected[0].len()));
    let maximum = actual
        .iter()
        .zip(expected.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("{name}: maximum absolute error {maximum}");
    assert!(maximum < tolerance, "{name}: {maximum} exceeds {tolerance}");
}

#[test]
#[ignore = "requires downloaded checkpoint; run cargo test --release -- --ignored --nocapture"]
fn checkpoint_matches_pytorch_reference() {
    let directory = PathBuf::from(
        std::env::var_os("PARAKEET_MODEL_DIR").unwrap_or_else(|| "models/parakeet-redux".into()),
    );
    let model = Model::from_bytes(
        &fs::read(directory.join("config.json")).unwrap(),
        &fs::read(directory.join("ternary.json")).unwrap(),
        &fs::read(directory.join("tokenizer.json")).unwrap(),
        &fs::read(directory.join("model.safetensors")).unwrap(),
    )
    .unwrap();
    let fixture: Fixture =
        serde_json::from_str(include_str!("../../tests/fixtures/reference.json")).unwrap();
    let features = model.features.extract(&fixture.samples).unwrap();
    let subsampled = model.encoder.subsample(&features);
    assert_close("subsampler", &subsampled, &fixture.subsampled, 5e-3);
    let vad = model.vad.probabilities(&subsampled);
    for (actual, expected) in vad.iter().zip(&fixture.vad) {
        assert!((actual - expected).abs() < 1e-5);
    }
    let encoded = model.encoder.encode(subsampled);
    assert_close("encoder", &encoded, &fixture.encoded, 2e-5);
    assert!(model.decoder.decode(&encoded).unwrap().is_empty());

    let samples: Vec<_> = include_bytes!("../../tests/fixtures/jfk.pcm")
        .chunks_exact(2)
        .map(|bytes| f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32768.0)
        .collect();
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/jfk.json")).unwrap();
    let transcript = model
        .transcribe(&samples, &TranscriptionOptions::default())
        .unwrap();
    assert_eq!(transcript.text, expected["text"].as_str().unwrap());
    let emissions = expected["emissions"].as_array().unwrap();
    assert_eq!(transcript.tokens.len(), emissions.len());
    for (token, emission) in transcript.tokens.iter().zip(emissions) {
        assert_eq!(token.id, emission["token_id"].as_u64().unwrap() as usize);
        let frame = emission["frame"].as_u64().unwrap();
        let duration = emission["duration"].as_u64().unwrap();
        assert!((token.start - frame as f64 * 0.08).abs() < 1e-9);
        assert!((token.end - (frame + duration) as f64 * 0.08).abs() < 1e-9);
    }
    eprintln!(
        "speech: {} tokens and all timestamps match PyTorch",
        transcript.tokens.len()
    );
}
