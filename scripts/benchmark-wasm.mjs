import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import init, { WasmModel } from "../dist/web/parakeet_redux.js";

const repetitions = Number(process.argv[2] ?? 3);
if (!Number.isInteger(repetitions) || repetitions < 1)
  throw new Error("Provide a positive repetition count.");
const wasm = await init({
  module_or_path: await readFile("dist/web/parakeet_redux_bg.wasm"),
});
const start = performance.now();
const files = await Promise.all(
  ["config.json", "ternary.json", "tokenizer.json", "model.safetensors"].map(
    (name) => readFile(`models/parakeet-redux/${name}`),
  ),
);
const model = new WasmModel(...files);
const loadSeconds = (performance.now() - start) / 1000;
const pcm = await readFile("tests/fixtures/jfk.pcm");
const samples = Float32Array.from(
  { length: pcm.length / 2 },
  (_, index) => pcm.readInt16LE(index * 2) / 32768,
);
const expected = JSON.parse(await readFile("tests/fixtures/jfk.json", "utf8"));
const timings = [];
try {
  for (let index = -1; index < repetitions; index++) {
    const start = performance.now();
    const result = JSON.parse(model.transcribe(samples));
    const seconds = (performance.now() - start) / 1000;
    assert.equal(result.text, expected.text);
    assert.deepEqual(
      result.tokens.map((token) => token.id),
      expected.emissions.map((emission) => emission.token_id),
    );
    if (index >= 0) timings.push(seconds);
  }
  const ordered = [...timings].sort((a, b) => a - b);
  const median =
    (ordered[Math.floor((repetitions - 1) / 2)] +
      ordered[Math.floor(repetitions / 2)]) /
    2;
  console.log(
    JSON.stringify(
      {
        audio_seconds: 11,
        load_seconds: loadSeconds,
        warmup_runs: 1,
        runs_seconds: timings,
        median_seconds: median,
        realtime_speed: 11 / median,
        rtf: median / 11,
        wasm_capacity_mib: wasm.memory.buffer.byteLength / 1048576,
        node: process.version,
      },
      null,
      2,
    ),
  );
} finally {
  model.free();
}
