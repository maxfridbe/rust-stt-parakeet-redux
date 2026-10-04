import { test } from "node:test";
import assert from "node:assert/strict";
import { transcribeWithTimings } from "./wasm-inference.js";

test("copies into the current memory after growth and performs no second input copy", () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const samples = Float32Array.from(
    { length: 320 },
    (_, index) => index / 1000,
  );
  const stages = [];
  let calls = 0;
  const model = {
    prepare_audio(length) {
      assert.equal(length, samples.length);
      memory.grow(1);
      return 65536;
    },
    transcribe_prepared(...args) {
      assert.equal(args.length, 0);
      assert.deepEqual(
        new Float32Array(memory.buffer, 65536, samples.length),
        samples,
      );
      calls++;
      memory.grow(1);
      return '{"text":"fixture"}';
    },
  };
  const measured = transcribeWithTimings(model, memory, samples, (stage) =>
    stages.push(stage),
  );
  assert.equal(calls, 1);
  assert.equal(measured.result.text, "fixture");
  assert.equal(measured.inputTiming.inputBytes, 1280);
  for (const name of ["allocationMs", "copyMs", "modelMs", "parseMs"])
    assert.ok(measured.inputTiming[name] >= 0);
  assert.deepEqual(stages, [
    "Allocating WASM input",
    "Copying audio into WASM",
    "Running model",
  ]);
});
