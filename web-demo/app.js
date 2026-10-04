import { Microphone, concatenate, decodeFile, resample } from "./audio.js";
import { LiveSegmenter, stableWordCount } from "./live.js";
import { SpeechEcho } from "./speech.js";

const element = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});
const microphone = new Microphone();
const echo = new SpeechEcho(element("voice"), (message) => {
  element("live-status").textContent = message;
});
const state = {
  ready: false,
  recording: false,
  stopping: false,
  preparing: false,
  busy: null,
  pending: [],
  nextId: 0,
  chunks: [],
  finalized: new Map(),
  previews: new Map(),
  spoken: new Map(),
  takes: 0,
  session: 0,
};
let segmenter;
let stopTimer;
let startedAt = 0;
let sampleRate = 16000;

if (!["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)) {
  element("model-url").value =
    "https://huggingface.co/moondream/parakeet-redux/resolve/2bf128600aac4b16946f7ed8372e56117fe5e23b";
}
if (!echo.supported) {
  element("echo-mode").checked = false;
  element("echo-mode").disabled = true;
}

function showError(error) {
  element("error").hidden = false;
  element("error").textContent = error.message ?? String(error);
}
function clearError() {
  element("error").hidden = true;
}
function updateControls() {
  const working = Boolean(
    state.preparing || state.busy || state.pending.length,
  );
  element("record").disabled =
    !state.ready || state.stopping || (!state.recording && working);
  element("sample").disabled =
    !state.ready || state.recording || state.stopping || working;
  element("audio-file").disabled = element("sample").disabled;
  element("live-mode").disabled = state.recording || state.stopping || working;
  element("record").classList.toggle("is-recording", state.recording);
  element("record").innerHTML =
    `<span class="record-dot"></span> ${state.recording ? "Stop recording" : "Start recording"}`;
  element("mic-light").classList.toggle("active", state.recording);
  if (working)
    element("transcript-state").textContent =
      `${state.pending.length ? `${state.pending.length} QUEUED · ` : ""}PROCESSING`;
  else if (state.ready)
    element("transcript-state").textContent = state.recording
      ? "LISTENING"
      : "READY";
}

element("load-model").onclick = () => {
  clearError();
  element("load-model").disabled = true;
  element("model-url").disabled = true;
  element("download-progress").hidden = false;
  element("model-status").textContent = "Fetching model files…";
  worker.postMessage({ type: "load", url: element("model-url").value.trim() });
};

worker.onmessage = ({ data }) => {
  if (data.type === "download") {
    element("download-progress").value = Math.min(1, data.bytes / data.total);
    element("model-status").textContent =
      `Loading ${(data.bytes / 1e6).toFixed(1)} / ${(data.total / 1e6).toFixed(1)} MB`;
  }
  if (data.type === "ready") {
    state.ready = true;
    element("download-progress").hidden = true;
    element("model-status").textContent =
      `Loaded · ${(data.loadMs / 1000).toFixed(2)}s initialization`;
    element("load-model").textContent = "Model ready ✓";
    element("metric-memory").textContent = (data.memoryBytes / 1048576).toFixed(
      0,
    );
    element("runtime-details").textContent =
      `Download ${(data.downloadMs / 1000).toFixed(2)}s · initialization ${(data.loadMs / 1000).toFixed(2)}s · single CPU worker`;
  }
  if (data.type === "result") receiveResult(data);
  if (data.type === "error") {
    showError(new Error(data.message));
    if (!state.ready) {
      element("load-model").disabled = false;
      element("model-url").disabled = false;
    }
    state.busy = null;
    processNext();
  }
  updateControls();
};
worker.onerror = (event) => {
  showError(
    new Error(
      `WASM worker failed: ${event.message}. Reload the page to restart.`,
    ),
  );
  state.ready = false;
  state.pending = [];
  state.busy = null;
  if (state.recording) stopRecording();
  updateControls();
};

function resetSession() {
  state.session++;
  state.chunks = [];
  state.finalized.clear();
  state.previews.clear();
  state.spoken.clear();
  echo.cancel();
  clearError();
  element("transcript").classList.remove("placeholder");
  element("transcript").textContent = "Listening…";
  element("copy").disabled = true;
}

function enqueue(snapshot) {
  snapshot.session = state.session;
  snapshot.capturedAt = performance.now();
  snapshot.id = state.nextId++;
  // Replace obsolete previews for this utterance; preserve every final chunk.
  state.pending = state.pending.filter(
    (job) => job.segment !== snapshot.segment || job.final,
  );
  state.pending.push(snapshot);
  updateControls();
  processNext();
}

async function processNext() {
  if (state.busy || !state.pending.length) return;
  const job = state.pending.shift();
  state.busy = job;
  updateControls();
  try {
    const samples = await resample(job.samples, job.rate);
    if (samples.length < 320)
      throw new Error("Record at least 20 ms of audio.");
    job.dispatchedAt = performance.now();
    worker.postMessage({ type: "transcribe", id: job.id, samples }, [
      samples.buffer,
    ]);
  } catch (error) {
    showError(error);
    state.busy = null;
    processNext();
    updateControls();
  }
}

function receiveResult(data) {
  const job = state.busy;
  if (!job || data.id !== job.id) return;
  const previous = state.previews.get(job.segment)?.words ?? [];
  const words = data.result.segments.flatMap((segment) => segment.words);
  const stable = stableWordCount(
    previous,
    words,
    job.final,
    data.result.duration_seconds,
  );
  const spoken = state.spoken.get(job.segment) ?? 0;
  if (element("echo-mode").checked && stable > spoken) {
    echo.say(
      words
        .slice(spoken, stable)
        .map((word) => word.word)
        .join(" "),
    );
  }
  state.spoken.set(job.segment, Math.max(stable, spoken));
  if (job.final) {
    state.finalized.set(job.segment, data.result.text);
    state.previews.delete(job.segment);
  } else state.previews.set(job.segment, { text: data.result.text, words });
  renderTranscript();
  updateMetrics(data, job);
  state.busy = null;
  processNext();
}

function renderTranscript() {
  const ids = [
    ...new Set([...state.finalized.keys(), ...state.previews.keys()]),
  ].sort((a, b) => a - b);
  const transcript = element("transcript");
  transcript.replaceChildren();
  for (const id of ids) {
    const span = document.createElement("span");
    span.textContent =
      (state.finalized.get(id) ?? state.previews.get(id)?.text ?? "") + " ";
    if (!state.finalized.has(id)) span.className = "provisional";
    transcript.append(span);
  }
  if (!transcript.textContent.trim())
    transcript.textContent = "No speech detected.";
  element("copy").disabled = !ids.length;
}

function updateMetrics(data, job) {
  const seconds = data.inferenceMs / 1000;
  const duration = data.result.duration_seconds;
  const speed = duration / seconds;
  element("metric-audio").textContent = duration.toFixed(2);
  element("metric-inference").textContent = seconds.toFixed(2);
  element("metric-speed").textContent = speed.toFixed(2);
  element("metric-memory").textContent = (data.memoryBytes / 1048576).toFixed(
    0,
  );
  element("token-count").textContent =
    `${data.result.tokens.length} tokens · ${job.final ? "final" : "preview"}`;
  const age = (performance.now() - job.capturedAt) / 1000;
  element("live-status").textContent =
    `${job.final ? "Final" : "Preview"} result ${age.toFixed(2)}s after capture · ${state.pending.length} waiting${speed < 1 ? " · slower than real time" : ""}`;
  element("history").querySelector(".empty-history")?.remove();
  const row = document.createElement("tr");
  for (const value of [
    `${String(++state.takes).padStart(2, "0")} ${job.final ? "final" : "live"}`,
    `${duration.toFixed(2)}s`,
    `${seconds.toFixed(2)}s`,
    `${speed.toFixed(2)}×`,
    (seconds / duration).toFixed(3),
  ]) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.append(cell);
  }
  element("history").prepend(row);
  while (element("history").children.length > 10)
    element("history").lastChild.remove();
}

async function startRecording() {
  resetSession();
  state.stopping = true;
  updateControls();
  segmenter = new LiveSegmenter(enqueue);
  try {
    sampleRate = await microphone.start((samples, rate) => {
      if (element("live-mode").checked) segmenter.add(samples, rate);
      else state.chunks.push(samples);
    });
    state.recording = true;
    startedAt = performance.now();
    element("input-tag").textContent = "MICROPHONE";
    element("sample-rate").textContent =
      `${(sampleRate / 1000).toFixed(0)} kHz capture → 16 kHz model`;
    element("capture-status").textContent = "Listening on this device";
    stopTimer = setTimeout(stopRecording, 30000);
  } catch (error) {
    showError(error);
  } finally {
    state.stopping = false;
    updateControls();
  }
}

async function stopRecording() {
  if (!state.recording || state.stopping) return;
  state.stopping = true;
  state.recording = false;
  clearTimeout(stopTimer);
  updateControls();
  try {
    await microphone.stop();
    if (element("live-mode").checked) segmenter.finish();
    else
      enqueue({
        segment: 0,
        final: true,
        samples: concatenate(state.chunks),
        rate: sampleRate,
      });
    element("capture-status").textContent = "Recording complete";
    if (!state.busy && !state.pending.length) renderTranscript();
  } catch (error) {
    showError(error);
  } finally {
    state.stopping = false;
    updateControls();
  }
}

element("record").onclick = () =>
  state.recording ? stopRecording() : startRecording();
element("echo-mode").onchange = () => {
  if (!element("echo-mode").checked) echo.cancel();
};
element("sample").onclick = async () => {
  resetSession();
  state.preparing = true;
  updateControls();
  try {
    const response = await fetch("jfk.pcm");
    if (!response.ok) throw new Error("Could not load sample audio.");
    const bytes = new DataView(await response.arrayBuffer());
    const samples = Float32Array.from(
      { length: bytes.byteLength / 2 },
      (_, i) => bytes.getInt16(i * 2, true) / 32768,
    );
    element("input-tag").textContent = "11s SPEECH SAMPLE";
    enqueue({ segment: 0, final: true, samples, rate: 16000 });
  } catch (error) {
    showError(error);
  } finally {
    state.preparing = false;
    updateControls();
  }
};
element("audio-file").onchange = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  resetSession();
  state.preparing = true;
  updateControls();
  try {
    const samples = await decodeFile(file);
    if (samples.length > 30 * 16000)
      throw new Error(
        "The browser lab accepts clips up to 30 seconds. Use the CLI for longer recordings.",
      );
    element("input-tag").textContent = "UPLOADED AUDIO";
    enqueue({ segment: 0, final: true, samples, rate: 16000 });
  } catch (error) {
    showError(error);
  } finally {
    state.preparing = false;
    updateControls();
  }
  event.target.value = "";
};
element("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(
      element("transcript").textContent.trim(),
    );
  } catch (error) {
    showError(error);
  }
};

const canvas = element("waveform");
const drawing = canvas.getContext("2d");
const waveform = new Float32Array(512);
function draw() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = devicePixelRatio || 1;
  if (
    canvas.width !== Math.round(width * ratio) ||
    canvas.height !== Math.round(height * ratio)
  ) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  drawing.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawing.clearRect(0, 0, width, height);
  if (microphone.analyser) microphone.analyser.getFloatTimeDomainData(waveform);
  else waveform.fill(0);
  drawing.strokeStyle = state.recording ? "#e96a39" : "#b5c1ac";
  drawing.lineWidth = 1.5;
  drawing.beginPath();
  waveform.forEach((value, index) => {
    const x = (index / (waveform.length - 1)) * width;
    const y = height * 0.43 + value * height * 0.8;
    if (!index) drawing.moveTo(x, y);
    else drawing.lineTo(x, y);
  });
  drawing.stroke();
  if (state.recording) {
    const seconds = Math.min(30, (performance.now() - startedAt) / 1000);
    element("recording-clock").textContent =
      `00:${seconds.toFixed(1).padStart(4, "0")}`;
  }
  requestAnimationFrame(draw);
}
draw();
