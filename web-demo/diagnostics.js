// Explicitly supplied numeric/runtime fields only: callers do not pass audio,
// transcripts, voice names, model URLs, or exception messages into this log.
export class DiagnosticLog {
  constructor(environment, limit = 80) {
    this.environment = environment;
    this.limit = limit;
    this.events = [];
    this.startedAt = new Date().toISOString();
    this.start = performance.now();
  }

  record(event, details = {}) {
    this.events.push({
      elapsedMs: performance.now() - this.start,
      event,
      ...details,
    });
    if (this.events.length > this.limit) this.events.shift();
  }

  serialize() {
    return JSON.stringify(
      {
        format: "parakeet-diagnostics-v1",
        runtimeInterface: "reusable-wasm-input-v1",
        startedAt: this.startedAt,
        environment: this.environment,
        events: this.events,
      },
      null,
      2,
    );
  }
}
