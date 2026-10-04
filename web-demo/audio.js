export function concatenate(chunks) {
  const output = new Float32Array(
    chunks.reduce((sum, chunk) => sum + chunk.length, 0),
  );
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

export async function resample(samples, sampleRate) {
  if (sampleRate === 16000) return samples;
  const context = new OfflineAudioContext(
    1,
    Math.ceil((samples.length * 16000) / sampleRate),
    16000,
  );
  const buffer = context.createBuffer(1, samples.length, sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.start();
  return (await context.startRendering()).getChannelData(0).slice();
}

export async function decodeFile(file) {
  const context = new AudioContext();
  try {
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    const mono = new Float32Array(buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const values = buffer.getChannelData(channel);
      for (let index = 0; index < mono.length; index++)
        mono[index] += values[index] / buffer.numberOfChannels;
    }
    return await resample(mono, buffer.sampleRate);
  } finally {
    await context.close();
  }
}

export class Microphone {
  async start(onSamples) {
    if (!navigator.mediaDevices?.getUserMedia)
      throw new Error("Microphone capture requires localhost or HTTPS.");
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    try {
      this.context = new AudioContext({ sampleRate: 16000 });
      await this.context.resume();
      await this.context.audioWorklet.addModule("recorder-worklet.js");
      this.source = this.context.createMediaStreamSource(this.stream);
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 512;
      this.recorder = new AudioWorkletNode(this.context, "recorder");
      this.recorder.port.onmessage = ({ data }) => {
        if (data.type === "audio")
          onSamples(data.samples, this.context.sampleRate);
        if (data.type === "flushed") this.flushed?.();
      };
      this.source.connect(this.analyser);
      this.analyser.connect(this.recorder);
      this.recorder.connect(this.context.destination); // Worklet output is silence.
      return this.context.sampleRate;
    } catch (error) {
      this.stream.getTracks().forEach((track) => track.stop());
      await this.context?.close();
      throw error;
    }
  }

  async stop() {
    if (!this.context) return;
    this.source.disconnect();
    this.stream.getTracks().forEach((track) => track.stop());
    await new Promise((resolve) => {
      const timeout = setTimeout(resolve, 500);
      this.flushed = () => {
        clearTimeout(timeout);
        resolve();
      };
      this.recorder.port.postMessage("flush");
    });
    this.recorder.disconnect();
    await this.context.close();
    this.context = null;
    this.analyser = null;
  }
}
