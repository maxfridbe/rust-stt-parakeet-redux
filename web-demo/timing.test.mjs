import { test } from "node:test";
import assert from "node:assert/strict";
import { pipelineTiming } from "./timing.js";

test("separates collection, queue, transfer and inference with a shared clock", () => {
  const origin = 1_700_000_000_000;
  const job = {
    audioStartedAt: origin,
    speechEndedAt: origin + 1400,
    enqueuedAt: origin + 2000,
    processingStartedAt: origin + 5000,
    dispatchedAt: origin + 5004,
  };
  // A worker's local clock need not have the window's origin.
  const workerOrigin = origin + 4000;
  const result = {
    startedAt: workerOrigin + 1006,
    finishedAt: workerOrigin + 4606,
    inferenceMs: 3600,
  };
  const timing = pipelineTiming(job, result, origin + 8610);
  assert.deepEqual(timing, {
    collectionMs: 2000,
    speechWaitMs: 600,
    queueMs: 3000,
    preparationMs: 4,
    dispatchMs: 2,
    inferenceMs: 3600,
    deliveryMs: 4,
    submittedToResultMs: 6610,
    speechToResultMs: 7210,
  });
});

test("samples have no microphone timing and rounded clocks cannot produce negative waits", () => {
  const timing = pipelineTiming(
    { enqueuedAt: 10, processingStartedAt: 10, dispatchedAt: 12 },
    { startedAt: 11, finishedAt: 20, inferenceMs: 9 },
    20,
  );
  assert.equal(timing.dispatchMs, 0);
  assert.equal(timing.collectionMs, null);
  assert.equal(timing.speechToResultMs, null);
});
