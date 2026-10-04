// Run after the container build: node scripts/test-wasm.mjs [model-directory]
// This exercises the actual generated browser bindings under Node's WASM engine.
import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import init, { WasmModel } from "../dist/web/parakeet_redux.js";
import { transcribeWithTimings } from "../web-demo/wasm-inference.js";

const directory = process.argv[2] ?? "models/parakeet-redux";
const wasm = await init({
  module_or_path: await readFile("dist/web/parakeet_redux_bg.wasm"),
});
const files = await Promise.all(
  ["config.json", "ternary.json", "tokenizer.json", "model.safetensors"].map(
    (name) => readFile(`${directory}/${name}`),
  ),
);
const model = new WasmModel(...files);
const pcm = await readFile("tests/fixtures/jfk.pcm");
const samples = Float32Array.from(
  { length: pcm.length / 2 },
  (_, index) => pcm.readInt16LE(index * 2) / 32768,
);
const expected = JSON.parse(await readFile("tests/fixtures/jfk.json", "utf8"));
const start = performance.now();
try {
  const result = JSON.parse(model.transcribe(samples));
  assert.equal(result.text, expected.text);
  assert.equal(result.tokens.length, expected.emissions.length);
  result.tokens.forEach((token, index) => {
    const reference = expected.emissions[index];
    assert.equal(token.id, reference.token_id);
    assert.ok(Math.abs(token.start - reference.frame * 0.08) < 1e-9);
    assert.ok(
      Math.abs(token.end - (reference.frame + reference.duration) * 0.08) <
        1e-9,
    );
  });
  console.log(
    `WASM: ${result.tokens.length} tokens and timestamps match in ${((performance.now() - start) / 1000).toFixed(2)}s`,
  );
  const measured = transcribeWithTimings(model, wasm.memory, samples);
  assert.deepEqual(measured.result, result);
  console.log(
    "Explicit WASM input timings:",
    JSON.stringify(measured.inputTiming),
  );
  assert.throws(() => model.prepare_audio(0), /320 samples/);
  assert.throws(() => model.prepare_audio(0xffffffff), /allocate audio buffer/);

  // Shrinking must update the input length, and memory growth must not leave a
  // retained JS view pointing at the detached buffer from the previous call.
  wasm.memory.grow(1);
  const shortSamples = samples.slice(0, 16000);
  const shortResult = transcribeWithTimings(model, wasm.memory, shortSamples);
  assert.deepEqual(
    shortResult.result,
    JSON.parse(model.transcribe(shortSamples)),
  );
  assert.equal(shortResult.result.duration_seconds, 1);
  const pointer = model.prepare_audio(320);
  new Float32Array(wasm.memory.buffer, pointer, 320).fill(NaN);
  assert.throws(() => model.transcribe_prepared(), /finite/);
  console.log(
    "Reusable input: identical transcript, tokens and timestamps; shrinking, memory growth and invalid input verified.",
  );
} finally {
  model.free();
}
