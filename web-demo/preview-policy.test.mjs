import { test } from "node:test";
import assert from "node:assert/strict";
import { PreviewPolicy } from "./preview-policy.js";
import { LiveSegmenter } from "./live.js";

test("touch devices wait for measured spare capacity before previewing", () => {
  const policy = new PreviewPolicy({ conservative: true });
  assert.equal(policy.allows(4, 4, true), false);
  policy.observe(1, 5);
  assert.equal(policy.allows(4, 4, true), true);
  policy.observe(3, 5);
  assert.equal(policy.allows(4, 4, true), false);
  policy.observe(2, 5);
  assert.equal(policy.allows(4, 4, true), false);
});

test("automatic previews reserve time for final inference", () => {
  const policy = new PreviewPolicy();
  policy.observe(0.5, 2);
  assert.equal(policy.allows(4, 2, true), false);
  assert.equal(policy.allows(4, 3, true), true);
  assert.equal(policy.allows(4, 4, false), false);
});

test("manual settings preserve pause-only and frequent choices", () => {
  const policy = new PreviewPolicy({ conservative: true });
  policy.mode = "frequent";
  assert.equal(policy.allows(2, 1.4, true), false);
  assert.equal(policy.allows(2, 1.5, true), true);
  assert.equal(policy.allows(2, 2, false), false);
  policy.mode = "pauses";
  policy.observe(0.1, 5);
  assert.equal(policy.allows(8, 8, true), false);
});

test("pause mode avoids repeated audio and keeps every final utterance", () => {
  function capture(mode) {
    const policy = new PreviewPolicy();
    policy.mode = mode;
    const jobs = [];
    const segmenter = new LiveSegmenter(
      (job) => jobs.push(job),
      (duration, elapsed) => policy.allows(duration, elapsed, true),
    );
    for (let i = 0; i < 160; i++)
      segmenter.add(new Float32Array(1600).fill(0.1), 16000);
    segmenter.finish();
    return jobs;
  }
  const frequent = capture("frequent");
  const pauses = capture("pauses");
  assert.equal(frequent.length, 12);
  assert.equal(pauses.length, 2);
  assert.deepEqual(
    pauses.map((job) => job.segment),
    [0, 1],
  );
  assert.ok(pauses.every((job) => job.final));
  const samples = (jobs) =>
    jobs.reduce((sum, job) => sum + job.samples.length, 0);
  assert.equal(samples(frequent) / 16000, 61);
  assert.equal(samples(pauses) / 16000, 16);
});

test("busy inference skips previews but never drops final audio", () => {
  const policy = new PreviewPolicy();
  const jobs = [];
  const segmenter = new LiveSegmenter(
    (job) => jobs.push(job),
    (duration, elapsed) => policy.allows(duration, elapsed, false),
  );
  segmenter.add(new Float32Array(32000).fill(0.1), 16000);
  segmenter.add(new Float32Array(10000), 16000);
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].final, true);
  assert.equal(jobs[0].samples.length, 42000);
});
