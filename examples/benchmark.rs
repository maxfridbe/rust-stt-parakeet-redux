//! Reproducible single-thread CPU measurement using the committed speech fixture.
use parakeet_redux::{Model, TranscriptionOptions};
use serde_json::json;
use std::{error::Error, fs, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let directory = PathBuf::from(
        arguments
            .next()
            .unwrap_or_else(|| "models/parakeet-redux".into()),
    );
    let repetitions: usize = arguments.next().map_or(Ok(5), |value| value.parse())?;
    if repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    let samples: Vec<_> = include_bytes!("../tests/fixtures/jfk.pcm")
        .chunks_exact(2)
        .map(|bytes| f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32768.0)
        .collect();
    let start = Instant::now();
    let model = Model::from_bytes(
        &fs::read(directory.join("config.json"))?,
        &fs::read(directory.join("ternary.json"))?,
        &fs::read(directory.join("tokenizer.json"))?,
        &fs::read(directory.join("model.safetensors"))?,
    )?;
    let load_seconds = start.elapsed().as_secs_f64();
    let options = TranscriptionOptions::default();
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/jfk.json"))?;
    // Warm up allocator and CPU caches separately from timed repetitions.
    let warmup = model.transcribe(&samples, &options)?;
    if warmup.text != expected["text"].as_str().ok_or("missing reference text")? {
        return Err("transcript does not match reference".into());
    }
    let mut timings = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        let result = model.transcribe(&samples, &options)?;
        timings.push(start.elapsed().as_secs_f64());
        if result.text != warmup.text {
            return Err("transcription changed between runs".into());
        }
    }
    let mut sorted = timings.clone();
    sorted.sort_by(f64::total_cmp);
    let median = (sorted[(repetitions - 1) / 2] + sorted[repetitions / 2]) / 2.0;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "audio_seconds": samples.len() as f64 / 16000.0, "tokens": warmup.tokens.len(),
            "load_seconds": load_seconds, "warmup_runs": 1, "runs_seconds": timings,
            "median_seconds": median, "realtime_speed": 11.0 / median, "rtf": median / 11.0,
            "text": warmup.text,
        }))?
    );
    Ok(())
}
