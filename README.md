# Parakeet Redux in Rust

Pure Rust CPU inference for [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux), with a native CLI and a WebAssembly API. Loads the original checkpoint directly: no conversion, Python runtime, ONNX, LibTorch, or native BLAS dependency.

## Try it now

**[Open the microphone speech lab →](https://maxfridbe.github.io/rust-stt-parakeet-redux/demo/)**

Load the model, allow microphone access, and speak. Live mode processes short phrases while listening; browser speech synthesis echoes words after they are stable across previews or finalized at a pause. Use headphones to avoid microphone feedback. The dashboard shows the waveform, transcript, inference speed, result delay, WASM memory capacity, and run history. You can also upload audio or try the bundled speech sample.

The first load downloads roughly 179 MB directly from Hugging Face. Audio stays in the browser. Voice synthesis uses the selected browser voice; voices marked “local” run on device, while other voices may use a network service. Live previews can change, and already-spoken text cannot be retracted. Latency is measured rather than guaranteed: this is an offline transducer repeatedly evaluated on utterances, not an incremental streaming encoder.

**On a phone:** use the default **Short · earlier text** phrase length and **Automatic** updates. The first phrase is submitted at a brief pause or around three seconds, so processing starts while you continue speaking. Each phrase is processed once. When the worker is busy, continuous speech accumulates into a longer phrase, up to eight seconds, to avoid queuing many tiny calls. Pauses and Stop preserve completed phrases. Choose **Long · more context** for fewer cuts: short phrases can lose context or split words. **Frequent previews** adds repeated processing and is best reserved for faster devices. Below 1× processing speed, continuous speech still outpaces this model; pauses give it time to catch up.

**Speed versus delay:** 0.56× means 10 seconds of audio takes about 17.9 seconds of inference. It excludes collecting the utterance and waiting behind previous work. The **Where the time goes** panel shows time to the first visible text and separates collection, queue, audio preparation, worker dispatch, the WASM call, and result delivery. During processing, the microphone panel shows queued audio seconds and whether the worker has received the job. The WASM call includes input copying and result parsing; it is not just the model's arithmetic. Echo startup has its own status.

To investigate a slow device, run the sample or record speech, then click **Copy diagnostics** below the timing panel and paste the log into an issue or conversation. You can copy while processing is still running. The log includes browser details, phrase settings, cut reasons, time to first text, microphone packet counts, queued and still-collecting audio, input byte counts, and separate WASM allocation, copy, model-call, and result-parsing timings. It retains up to 80 recent events in page memory; nothing is sent automatically, and no audio, transcript text, model URLs, or exception messages are included. If clipboard access fails, a selectable text field appears.

To run the same demo locally:

```sh
sh scripts/download-model.sh
sh scripts/build-container.sh
node scripts/serve-demo.mjs
# Open http://localhost:8080/demo/
```

See [PERFORMANCE.md](PERFORMANCE.md) for measured CPU/WASM speeds and reproduction commands.

The optimized kernels decode packed weights five at a time and store spatial
convolution weights with adjacent output channels, improving SIMD access and
avoiding temporary matrices in 1×1 convolutions. Inference still uses one CPU
worker. Sharing model computation across cores would require a threaded WASM
build and cross-origin isolation; the current demo does not initialize a thread
pool. The diagnostic log identifies this kernel revision in its `model-ready`
event so device comparisons can distinguish it from earlier builds.

The implementation covers log-mel features, convolutional subsampling, all 24 Conformer blocks, packed ternary projections, the two-layer LSTM predictor, greedy token-duration decoding, tokenizer decoding, token/word/sentence timestamps, and the checkpoint's VAD head for long recordings. Model weights stay packed between projection calls; each projection is temporarily expanded to float32 for portable matrix multiplication.

## Run

```sh
sh scripts/download-model.sh
cargo run --release -- --model models/parakeet-redux speech.wav
cargo run --release -- speech.wav --json
```

Input is a **16 kHz WAV** containing integer PCM or float samples. Multichannel WAV is averaged to mono. Convert other formats/sample rates before calling the CLI, for example:

```sh
ffmpeg -i recording.mp3 -ar 16000 -ac 1 speech.wav
```

The core accepts mono `f32` PCM directly. Audio must contain at least 320 samples (20 ms). Recordings longer than 30 seconds are scanned by the model's VAD, cut at pauses, and decoded in independent chunks. `--no-vad` uses fixed cuts; `--segment-seconds` sets a smaller cap. JSON includes the complete text, duration, sentence segments with words, and subword tokens with timestamps. TDT alignments have an 80 ms frame resolution; zero-duration tokens are valid. Sentence boundaries are a punctuation heuristic, not a separate alignment model.

## Build everything with Podman

Only Podman and a POSIX shell are needed on the host:

```sh
sh scripts/build-container.sh
```

The pinned Rust container runs formatting checks, Clippy with warnings denied, tests with and without the CLI feature, and release builds for Linux and `wasm32-unknown-unknown`. It also generates JavaScript and TypeScript bindings. Outputs:

```text
dist/
  parakeet-redux
  parakeet-benchmark
  SHA256SUMS
  demo/                        # microphone lab
  web/
    parakeet_redux.js
    parakeet_redux.d.ts
    parakeet_redux_bg.wasm
    parakeet_redux_bg.wasm.d.ts
  site/                        # GitHub Pages bundle
```

The native binary targets the build host's architecture and glibc 2.36 or later. To run inside the resulting container:

```sh
podman run --rm \
  -v "$PWD/models:/models:ro,Z" \
  -v "$PWD/audio:/audio:ro,Z" \
  localhost/parakeet-redux-build \
  --model /models/parakeet-redux /audio/speech.wav --json
```

[GitHub Actions](.github/workflows/build.yml) invokes the same Podman script on Ubuntu x86-64 and uploads separate native, browser, and checksum artifacts on pushes, pull requests, and manual runs. Manual runs can enable `validate_model` for real-weight regression tests, also inside Podman. Checkpoint downloads are separate from ordinary compilation; model weights are excluded from Git and the container build context. The artifacts include license and attribution files.

## Library

```rust,no_run
use parakeet_redux::{Model, TranscriptionOptions};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let read = |name| std::fs::read(format!("models/parakeet-redux/{name}"));
let model = Model::from_bytes(
    &read("config.json")?,
    &read("ternary.json")?,
    &read("tokenizer.json")?,
    &read("model.safetensors")?,
)?;
let samples = vec![0.0_f32; 16_000]; // Replace with mono 16 kHz PCM.
let transcript = model.transcribe(&samples, &TranscriptionOptions::default())?;
println!("{}", transcript.text);
# Ok(())
# }
```

Disable default features to use only the library. Model loading takes byte slices so callers control storage and downloading. `transcribe_with_progress` reports completed chunks. The inference core uses no filesystem access or background threads.

## WebAssembly

The container produces a complete browser module with WebAssembly SIMD enabled, supported by current mainstream browsers. Use it in a Worker, since inference is synchronous:

```js
import init, { WasmModel } from './web/parakeet_redux.js';
await init();
const read = async name => new Uint8Array(
  await (await fetch(`/models/parakeet-redux/${name}`)).arrayBuffer()
);
const files = await Promise.all(
  ['config.json', 'ternary.json', 'tokenizer.json', 'model.safetensors'].map(read)
);
const model = new WasmModel(...files);
const result = JSON.parse(model.transcribe(samples)); // Float32Array, mono 16 kHz
model.free();
```

Browser callers must supply enough memory for the 178 MB checkpoint, float32 non-ternary weights, and intermediate tensors. The tested 11-second clip uses 394.5 MiB of WASM linear memory capacity, plus JavaScript buffers. The current implementation uses compiler-generated SIMD and prioritizes correctness and portability. Specialized ternary kernels, GPU execution, and further memory/performance tuning are future work; Photon benchmark numbers do not apply to this implementation.

After building and downloading the checkpoint, `node scripts/test-wasm.mjs` runs the generated browser bindings under Node's WebAssembly engine and checks the speech fixture's text, tokens, and timestamps.

The demo uses a reusable Rust-owned audio buffer to time input copying separately:

```js
const wasm = await init();
// model is an existing WasmModel; samples is external Float32Array PCM.
const offset = model.prepare_audio(samples.length);
new Float32Array(wasm.memory.buffer, offset, samples.length).set(samples);
const result = JSON.parse(model.transcribe_prepared());
```

Create the memory view after `prepare_audio` and discard it before another WASM call: allocations can grow memory and invalidate views. Fill the entire prepared buffer before transcribing. Its capacity is reused until `model.free()`. The original `transcribe(samples)` API remains available. The WASM regression test checks both paths, shorter subsequent inputs, memory growth, and invalid audio.

## Structure

| Component | Location |
| --- | --- |
| Public loading/transcription API | `src/model.rs` |
| Config and checkpoint validation | `src/config.rs`, `src/weights.rs` |
| STFT, Slaney filters, normalization | `src/features.rs` |
| Packed/dense projections, convolutions, norms | `src/layers/` |
| Subsampling, relative attention, Conformer blocks | `src/encoder/` |
| LSTM predictor and TDT search | `src/decoder.rs` |
| VAD and pause selection | `src/vad.rs` |
| Text and timestamp assembly | `src/tokenizer.rs`, `src/transcript.rs` |
| Native and browser entry points | `src/bin/`, `src/wasm.rs` |
| Microphone, live previews, browser speech echo, visual metrics | `web-demo/` |

## Validation

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --no-default-features
sh scripts/download-model.sh
cargo test --locked --release --lib -- --ignored --nocapture
```

Ordinary tests do not download weights. The ignored checkpoint test checks real-weight subsampling, all encoder layers, and VAD against committed PyTorch fixtures, then checks the 11-second JFK speech sample against the reference's text, all 39 token IDs, and their timestamps. Set `PARAKEET_MODEL_DIR` to use another local model directory.

`scripts/reference.py` is an independent **validation-only** float32 PyTorch implementation. It is not used during Rust builds or inference. To regenerate the deterministic numerical fixture in an isolated Python environment with `torch`, `numpy`, `librosa`, and `safetensors` installed:

```sh
python scripts/reference.py --output tests/fixtures/reference.json
```

The pinned checkpoint revision is `2bf128600aac4b16946f7ed8372e56117fe5e23b`; the download script verifies every file's SHA-256. This is a portable float32 inference implementation, not a bitwise reproduction of Photon's hardware-specific quantized activation kernels. Broad multilingual WER and long-form accuracy benchmarking remain separate validation work.

## Attribution

Code is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights remain **CC-BY-4.0**, credited to Moondream and the NVIDIA Parakeet base model. The architecture reference is [Hugging Face Transformers Parakeet](https://github.com/huggingface/transformers/blob/0a896aa41bba78f92338db21cc468fe043888657/src/transformers/models/parakeet/modeling_parakeet.py). The published `ternary.json` defines the packed format. VAD and pause behavior were checked against the published Kestrel 0.9.1 runtime. See [fixture provenance](tests/fixtures/README.md) for numerical and audio test inputs.
