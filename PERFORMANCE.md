# Performance

Measured on 2026-10-04 with the pinned Parakeet Redux checkpoint, using the
committed 11-second, 16 kHz JFK speech fixture. These are measurements of this
Rust implementation, not Photon's published performance numbers.

## Optimized kernels

The `grouped-ternary-channels-last-v1` kernel revision reduces repeated packed
weight decoding and reorganizes spatial convolutions for contiguous channel
updates. The pointwise convolutions now reshape views rather than rebuilding
input, weight, and output arrays on every call. The implementation remains safe
Rust and uses the same standard WASM SIMD build flags and one inference worker.

Same-session comparison against commit `3b6bfe9`, on the machine described below,
using the same 11-second speech fixture and one warm-up, with three timed WASM
runs and five timed native runs:

| Runtime | Before median | Optimized median | Less processing time | Optimized audio / wall time |
| --- | ---: | ---: | ---: | ---: |
| Node 22.11.0 WASM | 5.731 s | **4.810 s** | **16.1%** | **2.29×** |
| Firefox 155 desktop WASM | 5.738 s | **5.065 s** | **11.7%** | **2.17×** |
| Native Rust | 2.448 s | **1.921 s** | **21.5%** | **5.73×** |

Grouped weight decoding alone measured 5.058 seconds in Node; the convolution
changes brought the combined median to 4.810 seconds. Optimized Node runs were
4.810, 4.800, and 4.825 seconds. Firefox runs were 5.066, 5.065, and 5.063 seconds.
Raw measurements are in [benchmarks/kernel-optimization/](benchmarks/kernel-optimization/).
The Firefox worker benchmark includes the reusable input copy and JSON parsing;
the Node benchmark uses the original `transcribe(samples)` binding. Each comparison
uses the same API before and after. Browser timings have millisecond precision.
To reproduce the browser measurement, run Try a sample four times, discard the
first run, and take the median of the next three `inferenceMs` values in Copy
diagnostics. Native runs use the existing `parakeet-benchmark` command below.

The optimized Node run used 394.625 MiB of WASM linear memory capacity. Native
real-weight checks still pass the independent subsampler, encoder and VAD
fixtures; all 39 speech tokens and timestamps also match in Firefox. These are
desktop measurements, not a prediction of Android speed. Use Copy diagnostics
on the phone to compare: its `model-ready` event records the kernel revision.

### Parallel execution

Microphone capture, UI work, and model inference already run concurrently.
SIMD performs several arithmetic operations per instruction inside the inference
worker. Multiple CPU cores do not yet share the model's matrix calculations.

Matrix tiles, attention heads and some projections can be parallelized within a
layer, but successive Conformer blocks depend on preceding outputs. A browser
thread pool needs shared WASM memory, a threading-enabled Rust build and
[cross-origin isolation](https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated).
The [wasm-bindgen-rayon setup](https://github.com/RReverser/wasm-bindgen-rayon)
documents the additional toolchain and worker-pool requirements. The hosted demo
currently reports `crossOriginIsolated: false`; this update does not enable
multicore inference. Loading a separate full model per worker would multiply
memory use, so it is not an appropriate default for this phone demo. A future
shared-memory implementation should benchmark small thread counts and retain
the single-worker fallback.

## Initial CPU processing speeds

| Runtime | Timed runs | Median inference | Audio / wall time | RTF |
| --- | ---: | ---: | ---: | ---: |
| Native Rust, container-built release binary | 5 | **2.468 s** | **4.46× real time** | 0.224 |
| WebAssembly scalar, Node 22.11.0 | 3 | 17.763 s | 0.62× real time | 1.615 |
| WebAssembly SIMD, Node 22.11.0 | 3 | **5.788 s** | **1.90× real time** | 0.526 |
| WebAssembly SIMD, desktop Firefox 155 | 3 | 5.77 s | 1.91× real time | 0.525 |

The shipped browser build enables `simd128`. That reduced median WASM inference
time by approximately 67% compared with the scalar build (3.07× faster). No
hand-written unsafe SIMD or native ML runtime is required.

Each measurement includes feature extraction, subsampling, all 24 Conformer
blocks, TDT decoding, and transcript/timestamp assembly. WASM timing also includes
the JS/Rust call, PCM copying, and JSON parsing. Model loading is excluded. Each
process loads the model once, runs one unmeasured warm-up, and then processes the
same clip sequentially. All timed outputs are checked against the reference text;
the WASM benchmark also checks all token IDs.

Native runs: 2.468, 2.478, 2.446, 2.486, 2.455 seconds. WASM SIMD runs: 5.788,
5.805, 5.771 seconds. Raw results are in [benchmarks/](benchmarks/).
Firefox timings came from the demo's worker timer, displayed to two decimal
places: 5.75, 5.77, 5.77 seconds after one 6.00-second warm-up. This was headless
desktop Firefox on the same Ryzen CPU, with a 390-pixel touch viewport; it checks
engine compatibility and layout, **not Android hardware performance**. All four
outputs matched the reference text.

`Audio / wall time = audio seconds / inference seconds`, so higher is faster.
`RTF = inference seconds / audio seconds`, so lower is faster and values below
one mean the clip was processed faster than its duration.

## Machine and build

- AMD Ryzen 7 7700X: 8 physical cores, 16 logical CPUs, x86-64 Linux.
- One inference thread; no Rayon, native BLAS, GPU, or worker pool. The browser
  keeps that single thread in a Worker so the interface remains responsive.
- CPU boost enabled; no CPU affinity or fixed-frequency governor was applied.
- Rust 1.97.1, release optimization, thin LTO, one codegen unit.
- Native build uses the generic Rust target, without `target-cpu=native`.
- WASM build adds `-C target-feature=+simd128`; current mainstream browsers
  support this instruction set. The scalar build remains possible.
- Model revision: `2bf128600aac4b16946f7ed8372e56117fe5e23b`.

These results describe one short English clip with warm filesystem/CPU caches.
They are not a multilingual accuracy benchmark or a long-form throughput claim.
Different CPUs, browsers, silence patterns, and token counts change processing
time. In particular, decoding many tokens per frame costs more than blank-heavy
audio.

## Loading and memory

Native loading, including local file reads, took 0.256 seconds. WASM SIMD loading,
including local Node file reads and copying into WASM, took 0.411 seconds. Browser
download time is separate and depends on the network: the four checkpoint files
total 179,005,408 bytes.

The speech fixture used **394.5 MiB of WASM linear memory capacity**. That value is
allocated addressable WASM memory, not process RSS or total browser memory; JS
download buffers and the browser engine consume additional memory. The native
benchmark's process peak RSS is recorded in
[benchmarks/native-resources.txt](benchmarks/native-resources.txt), including model
loading and the warm-up. Encoder projection weights stay packed between calls;
only one projection at a time is expanded to float32.

The browser lab caps each recording/upload at 30 seconds. The native library can
process longer recordings using VAD segmentation; its VAD scan uses 120-second
blocks and can require substantially more memory than this short-clip benchmark.

## Live microphone and voice echo

The microphone lab stays open while inference runs in a Worker. It finalizes an
utterance after 600 ms of silence and caps a continuous utterance at 8 seconds.
An energy threshold chooses interactive utterance boundaries; this is separate from the model's native
long-recording VAD path.

Only one inference runs at a time. Previews are skipped while any inference or
final job is pending, before copying the growing audio buffer. Final utterances
are preserved. Three update settings control the additional preview work:

- **Automatic** starts with pause-based updates on coarse-pointer/touch devices.
  Measurements at or below RTF 0.3 enable previews; RTF 0.5 or higher disables
  them. The gap between these thresholds prevents rapid switching. When enabled,
  previews wait at least 1.5 seconds and three times the estimated inference cost
  (the greater of the previous inference time and current duration × latest RTF).
- **On pauses** runs only final utterances, including the 8-second cap and Stop.
- **Frequent previews** requests a preview every 1.5 seconds when inference is
  idle. It trades more repeated work for earlier hypotheses.

For an 8-second continuous utterance, frequent previews can submit 1.5, 3, 4.5,
6, 7.5, and 8 seconds of audio: **30.5 seconds across six calls**. Pause mode sends
**8 seconds in one call**, about **74% less submitted audio** in this scheduling
example. This is a deterministic scheduler test, not a measured 74% wall-time
speedup; startup and decoding costs do not scale uniformly with duration. The
waveform also redraws at most 30 times per second while recording and skips
unchanged idle frames and hidden tabs.

A word is eligible for browser speech synthesis once it agrees across two
previews and is followed by another
word with sufficient audio context, or once an utterance is finalized. Already
spoken words cannot be retracted if a later hypothesis changes.

This is repeated offline inference over growing utterances, not a streaming
encoder with cached states. Faster-than-real-time processing of an 11-second clip
does **not** guarantee instant word feedback: short snapshots repeat model work,
confirmation waits for context, and browser voice startup adds latency. The demo
reports result age, inference speed, queue length, and the delay from word
confirmation to the speech API's `onstart` event. Speech synthesis is supplied by
the browser/OS and is not counted as Rust inference time. Use headphones to avoid
feeding the synthesized voice back into the microphone.

For Firefox on Android, use Automatic or On pauses and check the sample's speed
metric. Below 1×, even final-only inference cannot keep up with continuous input.
Pauses provide time to catch up. These scheduling changes do not accelerate an
individual model pass or reduce its weight memory, and no Android device speed
has been measured here.

Chromium 153 was exercised with the sample, a simulated microphone, desktop and
390-pixel mobile layouts, and instrumented browser speech API calls. Headless
testing confirms the speech API is invoked; audible playback and installed
voices depend on the user's browser and operating system.

## Throughput versus perceived delay

At 0.56× speed, an 8-second utterance needs approximately 14.3 seconds to process.
If submission waits for the 8-second cap, its first words appear about 22.3
seconds after speaking starts, even with an empty queue. Subsequent utterances
can wait behind that job. A shorter utterance may have a different throughput
because per-call costs matter. These are calculations, not measured phone timings.

The demo now reports each stage separately:

| Stage | Measurement |
| --- | --- |
| Collecting utterance | Estimated first audio arrival through preview/final submission; includes waiting for a pause |
| Waiting in queue | Submission until the job is selected for processing |
| Preparing audio | Resampling to 16 kHz if needed, up to worker dispatch |
| Sending to worker | Dispatch until the worker starts handling the job |
| WASM call | Input copying, Rust inference, transcript serialization, and JS parsing |
| Delivering result | Worker completion until the main thread receives the result |

The headline speed still measures the WASM call. Sample fetching, file decoding,
hardware microphone latency, rendering, and synthesized speech playback are not
part of that metric. Microphone speech-to-result timing uses the arrival of the
last packet above the energy threshold, so it is approximate, not acoustic
latency. Window and Worker timestamps share `performance.timeOrigin +
performance.now()`; small values can round to zero with browser timer precision.
The live status shows queued audio seconds and elapsed time while processing,
and speech echo no longer overwrites transcription delay.

The microphone now transfers 512-sample packets (32 ms at 16 kHz), down from
2,048 samples (128 ms). Full packets transfer their existing buffer rather than
copying it first. Final partial packets are retained. This reduces packet fill
latency by up to 96 ms; it does not turn the offline model into a streaming
encoder or remove seconds of inference/queue delay.

An instrumented desktop Firefox sample run measured 5,989 ms in the WASM call
and 1 ms elsewhere after submission; worker dispatch and result delivery rounded
to zero. This supports inference as the dominant cost on that desktop run. The
new panel is needed to check whether the same holds on the user's phone.

### Separating the JS-to-WASM input copy

The demo now explicitly allocates/resizes a Rust-owned audio buffer, copies into
its WASM memory view once, and calls `transcribe_prepared()` with no array argument.
This removes the implicit per-call input allocation in the original
`transcribe(samples)` binding and makes the actual copy independently measurable.
The old API is retained. Buffer capacity is reused; views are recreated after
allocation and discarded before inference so WASM memory growth cannot leave a
stale view in the worker.

The timing panel and **Copy diagnostics** log now split the aggregate WASM work
into allocation, input copy, model plus result encoding, and JSON parsing. Model
time includes Rust inference, transcript serialization, and returning its string
through the binding; it excludes input copying. The headline speed still includes
all these stages, so it remains comparable to previous readings. Stage messages
also identify whether a currently running job has reached inference.

One Node 22 desktop regression run on the same CPU measured the 11-second
fixture's **704,000-byte** input at **0.058 ms allocation**, **0.025 ms copy**,
**5,512.932 ms model/encoding**, and **0.040 ms parsing**. This is one diagnostic
run after the original API's fixture check, not a controlled throughput comparison
or an Android measurement. Both APIs returned identical text, all token IDs, and
timestamps. The test also checks shorter subsequent inputs and WASM memory growth.

Copy diagnostics retains at most 80 recent events, including capture packet
counts and arrival span, job stages, queue depth, and per-run timings. It excludes
audio, transcript text, URLs and exception messages. It is kept in page memory
and only exported when the user clicks Copy; denied clipboard access exposes a
manual-copy field. Values near zero can be rounded by browser timer precision.

## Correctness gates

- All 39 token IDs and start/end timestamps on the speech fixture match the
  independent float32 PyTorch reference in native Rust and WASM.
- The deterministic frontend fixture matches within `1e-4` absolute error.
- Full encoder output on that fixture differs by at most `6.33e-8`.
- Subsampling's maximum absolute difference is `0.00455`, on activations with
  magnitudes reaching approximately 3,945; float32 accumulation order differs.
- Real-weight VAD probabilities pass the reference check.

The tests establish those fixtures, not universal numerical equivalence or WER.
Photon hardware-specific activation quantization is not emulated; this port
evaluates the published ternary weights with float32 activations.

## Reproduce

All Rust compilation and packaging can occur inside Podman:

```sh
sh scripts/download-model.sh
sh scripts/build-container.sh
dist/parakeet-benchmark models/parakeet-redux 5
node scripts/benchmark-wasm.mjs 3
node scripts/test-wasm.mjs
```

To measure native peak RSS on Linux:

```sh
/usr/bin/time -v dist/parakeet-benchmark models/parakeet-redux 5
```

Run these benchmarks sequentially, with other CPU-intensive tasks stopped. The
native benchmark can also run inside the artifact container:

```sh
podman run --rm --entrypoint /artifacts/parakeet-benchmark \
  -v "$PWD/models/parakeet-redux:/model:ro,Z" \
  localhost/parakeet-redux-build /model 5
```

To build scalar WASM, run the WASM Cargo command without `RUSTFLAGS`, then rerun
`wasm-bindgen` with the same version used in `Containerfile`. The committed
container build intentionally publishes the faster SIMD variant.
