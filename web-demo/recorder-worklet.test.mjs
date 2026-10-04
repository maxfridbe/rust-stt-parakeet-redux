import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { runInNewContext } from "node:vm";

test("microphone transfers 32 ms packets without losing partial tails", async () => {
  const messages = [];
  let Recorder;
  runInNewContext(
    await readFile(new URL("recorder-worklet.js", import.meta.url), "utf8"),
    {
      Float32Array,
      AudioWorkletProcessor: class {
        port = {
          postMessage: (data, transfer) =>
            messages.push(structuredClone(data, { transfer })),
        };
      },
      registerProcessor: (_, processor) => {
        Recorder = processor;
      },
    },
  );
  const recorder = new Recorder();
  for (let i = 0; i < 5; i++) {
    recorder.process([
      [new Float32Array(128).fill(i), new Float32Array(128).fill(i + 2)],
    ]);
  }
  assert.equal(messages.length, 1);
  assert.equal(messages[0].samples.length, 512);
  recorder.port.onmessage({ data: "flush" });
  assert.equal(messages[1].samples.length, 128);
  assert.equal(messages[2].type, "flushed");
  const samples = [...messages[0].samples, ...messages[1].samples];
  for (let i = 0; i < 640; i++)
    assert.equal(samples[i], Math.floor(i / 128) + 1);
});
