use clap::Parser;
use parakeet_redux::{Model, TranscriptionOptions};
use std::{error::Error, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(
    version,
    about = "Transcribe 16 kHz WAV audio using pure Rust Parakeet Redux inference"
)]
struct Arguments {
    /// Directory containing config.json, ternary.json, tokenizer.json and model.safetensors.
    #[arg(long, default_value = "models/parakeet-redux")]
    model: PathBuf,
    /// Input WAV (16 kHz, PCM integer or float; channels are averaged).
    audio: PathBuf,
    /// Print transcript JSON, including token, word and sentence timestamps.
    #[arg(long)]
    json: bool,
    /// Disable VAD and split at fixed chunk boundaries.
    #[arg(long)]
    no_vad: bool,
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u8).range(1..=30))]
    segment_seconds: u8,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse();
    let samples = read_wav(&arguments.audio)?;
    let start = Instant::now();
    let model = Model::from_bytes(
        &fs::read(arguments.model.join("config.json"))?,
        &fs::read(arguments.model.join("ternary.json"))?,
        &fs::read(arguments.model.join("tokenizer.json"))?,
        &fs::read(arguments.model.join("model.safetensors"))?,
    )?;
    eprintln!(
        "Loaded {} encoder layers in {:.2}s",
        model.config().encoder_config.num_hidden_layers,
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let options = TranscriptionOptions {
        max_segment_seconds: arguments.segment_seconds.into(),
        use_vad: !arguments.no_vad,
    };
    let transcript = model.transcribe_with_progress(&samples, &options, |processed, total| {
        eprintln!(
            "Transcribed {:.2}/{:.2}s",
            processed as f64 / 16000.0,
            total as f64 / 16000.0
        );
    })?;
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "Inference: {elapsed:.2}s, {:.2}x real time",
        transcript.duration_seconds / elapsed
    );
    if arguments.json {
        println!("{}", serde_json::to_string_pretty(&transcript)?);
    } else {
        println!("{}", transcript.text);
    }
    Ok(())
}

fn read_wav(path: &PathBuf) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels == 0 {
        return Err("input must be a 16 kHz WAV with at least one channel".into());
    }
    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let channels = usize::from(spec.channels);
    if interleaved.len() % channels != 0 {
        return Err("incomplete WAV frame".into());
    }
    Ok(interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect())
}
