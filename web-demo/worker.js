import init, { WasmModel } from "../web/parakeet_redux.js?input-buffer=1";
import { timestamp } from "./timing.js";
import { transcribeWithTimings } from "./wasm-inference.js";

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
      wasm = await init({
        module_or_path: new URL(
          "../web/parakeet_redux_bg.wasm?input-buffer=1",
          import.meta.url,
        ),
      });
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
      const { result, inputTiming } = transcribeWithTimings(
        model,
        wasm.memory,
        data.samples,
        (stage) => postMessage({ type: "stage", id: data.id, stage }),
      );
      const finishedAt = timestamp();
      postMessage({
        type: "result",
        id: data.id,
        result,
        inputTiming,
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
