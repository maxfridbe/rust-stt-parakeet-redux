import init, { WasmModel } from "../web/parakeet_redux.js";
import { timestamp } from "./timing.js";

let model;
let wasm;
let downloaded = 0;
const estimatedBytes = 179005408;

async function download(url) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(
      `${response.status} loading ${url}. Run scripts/download-model.sh first.`,
    );
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let size = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    size += value.length;
    downloaded += value.length;
    postMessage({ type: "download", bytes: downloaded, total: estimatedBytes });
  }
  const output = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const start = performance.now();
      wasm = await init();
      downloaded = 0;
      const files = [];
      for (const name of [
        "config.json",
        "ternary.json",
        "tokenizer.json",
        "model.safetensors",
      ]) {
        files.push(await download(`${data.url.replace(/\/$/, "")}/${name}`));
      }
      const downloadedAt = performance.now();
      model?.free();
      model = new WasmModel(...files);
      postMessage({
        type: "ready",
        downloadMs: downloadedAt - start,
        loadMs: performance.now() - downloadedAt,
        memoryBytes: wasm.memory.buffer.byteLength,
      });
      return;
    }
    if (data.type === "transcribe") {
      if (!model) throw new Error("Load the model first.");
      const startedAt = timestamp();
      postMessage({ type: "started", id: data.id, startedAt });
      const result = JSON.parse(model.transcribe(data.samples));
      const finishedAt = timestamp();
      postMessage({
        type: "result",
        id: data.id,
        result,
        startedAt,
        finishedAt,
        inferenceMs: finishedAt - startedAt,
        memoryBytes: wasm.memory.buffer.byteLength,
      });
    }
  } catch (error) {
    postMessage({
      type: "error",
      id: data.id,
      message: error.message ?? String(error),
    });
  }
};
