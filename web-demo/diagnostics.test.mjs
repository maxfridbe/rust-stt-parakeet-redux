import { test } from "node:test";
import assert from "node:assert/strict";
import { DiagnosticLog } from "./diagnostics.js";

test("copyable diagnostics retain a bounded ordered history across runs", () => {
  const log = new DiagnosticLog({ userAgent: "test-browser" }, 2);
  log.record("submitted", { id: 1 });
  log.record("result", { id: 1, copyMs: 0.2 });
  log.record("submitted", { id: 2 });
  const saved = JSON.parse(log.serialize());
  assert.equal(saved.environment.userAgent, "test-browser");
  assert.equal(saved.events.length, 2);
  assert.equal(saved.events[0].event, "result");
  assert.equal(saved.events[0].copyMs, 0.2);
  assert.equal(saved.events[1].id, 2);
  assert.ok(saved.events[1].elapsedMs >= saved.events[0].elapsedMs);
});
