import { test } from "node:test";
import assert from "node:assert/strict";
import { LiveSegmenter, stableWordCount } from "./live.js";

test("only confirmed complete words are echoed before finalization", () => {
  const before = [
    { word: "Hello", end: 0.5 },
    { word: "world", end: 1.0 },
  ];
  assert.equal(stableWordCount(before, before, false, 2), 1);
  assert.equal(stableWordCount(before, before, true, 2), 2);
  assert.equal(
    stableWordCount(
      before,
      [{ word: "Yellow", end: 0.5 }, before[1]],
      false,
      2,
    ),
    0,
  );
  assert.equal(stableWordCount(before, before, false, 0.6), 0);
});

test("silence finalizes speech and previews retain utterance identity", () => {
  const jobs = [];
  const segmenter = new LiveSegmenter((job) => jobs.push(job));
  segmenter.add(new Float32Array(32000).fill(0.1), 16000);
  segmenter.add(new Float32Array(10000), 16000);
  assert.equal(jobs.length, 2);
  assert.equal(jobs[0].final, false);
  assert.equal(jobs[1].final, true);
  assert.equal(jobs[0].segment, jobs[1].segment);
  assert.equal(jobs[1].samples.length, 42000);
  segmenter.finish();
  assert.equal(jobs.length, 2);
});

test("snapshots retain speech time so pause and queue delays are visible", () => {
  const jobs = [];
  const segmenter = new LiveSegmenter(
    (job) => jobs.push(job),
    () => false,
  );
  segmenter.add(new Float32Array(32000).fill(0.1), 16000, 5000);
  segmenter.add(new Float32Array(10000), 16000, 5625);
  assert.equal(jobs[0].audioStartedAt, 3000);
  assert.equal(jobs[0].speechEndedAt, 5000);
});

test("short phrases start processing during continuous speech without repeating audio", () => {
  const jobs = [];
  const segmenter = new LiveSegmenter(
    (job) => jobs.push(job),
    () => false,
    { maxSeconds: 3, pauseSeconds: 0.35 },
  );
  for (let packet = 0; packet < 250; packet++) {
    segmenter.add(
      new Float32Array(512).fill(packet + 1),
      16000,
      32 * (packet + 1),
    );
    if (packet === 93) assert.equal(jobs.length, 1);
  }
  segmenter.finish();
  assert.deepEqual(
    jobs.map((job) => job.segment),
    [0, 1, 2],
  );
  assert.deepEqual(
    jobs.map((job) => job.boundary),
    ["duration", "duration", "stop"],
  );
  assert.ok(
    jobs.every((job) => job.final && job.samples.length <= 3.032 * 16000),
  );
  const captured = jobs.flatMap((job) => Array.from(job.samples));
  assert.equal(captured.length, 8 * 16000);
  for (let index = 0; index < captured.length; index++)
    assert.equal(captured[index], Math.floor(index / 512) + 1);
});

test("short phrases use brief word gaps near the cap and flush only once", () => {
  const jobs = [];
  const segmenter = new LiveSegmenter(
    (job) => jobs.push(job),
    () => false,
    { maxSeconds: 3, pauseSeconds: 0.35 },
  );
  segmenter.add(new Float32Array(2.3 * 16000).fill(0.1), 16000);
  segmenter.add(new Float32Array(0.1 * 16000), 16000);
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].boundary, "pause");
  assert.equal(jobs[0].samples.length, 2.4 * 16000);
  segmenter.finish();
  segmenter.finish();
  assert.equal(jobs.length, 1);
});

test("short phrases keep a brief hesitation within a phrase, then finalize a pause", () => {
  const jobs = [];
  const segmenter = new LiveSegmenter(
    (job) => jobs.push(job),
    () => false,
    { maxSeconds: 3, pauseSeconds: 0.35 },
  );
  segmenter.add(new Float32Array(16000).fill(0.1), 16000);
  segmenter.add(new Float32Array(1600), 16000);
  assert.equal(jobs.length, 0);
  segmenter.add(new Float32Array(4000), 16000);
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].boundary, "pause");
});

test("busy processing batches continuous speech but preserves the hard cap and Stop", () => {
  const jobs = [];
  let idle = true;
  const segmenter = new LiveSegmenter(
    (job) => jobs.push(job),
    () => false,
    { targetSeconds: 3, maxSeconds: 8, canFinalize: () => idle },
  );
  const second = new Float32Array(16000).fill(0.1);
  for (let index = 0; index < 3; index++) segmenter.add(second, 16000);
  assert.equal(jobs.length, 1);
  idle = false;
  for (let index = 0; index < 4; index++) segmenter.add(second, 16000);
  assert.equal(jobs.length, 1);
  idle = true;
  segmenter.add(second, 16000);
  assert.equal(jobs[1].samples.length, 5 * 16000);
  idle = false;
  for (let index = 0; index < 8; index++) segmenter.add(second, 16000);
  assert.equal(jobs[2].samples.length, 8 * 16000);
  segmenter.add(second, 16000);
  segmenter.finish();
  assert.equal(jobs[3].boundary, "stop");
  assert.equal(
    jobs.reduce((sum, job) => sum + job.samples.length, 0),
    17 * 16000,
  );
});

test("background hiss after louder speech does not conceal a pause", () => {
  const jobs = [];
  const segmenter = new LiveSegmenter(
    (job) => jobs.push(job),
    () => false,
    { maxSeconds: 3, pauseSeconds: 0.35 },
  );
  segmenter.add(new Float32Array(16000).fill(0.3), 16000);
  segmenter.add(new Float32Array(5600).fill(0.01), 16000);
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].boundary, "pause");
  // A loud transient must not raise the threshold enough to discard quiet speech.
  segmenter.add(new Float32Array(16000).fill(0.8), 16000);
  segmenter.add(new Float32Array(8000).fill(0.025), 16000);
  assert.equal(jobs.length, 1);
  segmenter.finish();
  assert.equal(jobs[1].samples.length, 24000);
});
