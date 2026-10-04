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
