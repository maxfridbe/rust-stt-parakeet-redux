// One explicit copy into reusable Rust-owned storage. Never retain a view across
// WASM calls, because allocation or inference can grow and detach linear memory.
export function transcribeWithTimings(
  model,
  memory,
  samples,
  reportStage = () => {},
) {
  reportStage("Allocating WASM input");
  const allocationStarted = performance.now();
  const pointer = model.prepare_audio(samples.length);
  const allocationMs = performance.now() - allocationStarted;

  reportStage("Copying audio into WASM");
  const copyStarted = performance.now();
  new Float32Array(memory.buffer, pointer, samples.length).set(samples);
  const copyMs = performance.now() - copyStarted;

  reportStage("Running model");
  const modelStarted = performance.now();
  const json = model.transcribe_prepared();
  const modelMs = performance.now() - modelStarted;

  const parseStarted = performance.now();
  const result = JSON.parse(json);
  const parseMs = performance.now() - parseStarted;
  return {
    result,
    inputTiming: {
      inputBytes: samples.byteLength,
      allocationMs,
      copyMs,
      modelMs,
      parseMs,
    },
  };
}
