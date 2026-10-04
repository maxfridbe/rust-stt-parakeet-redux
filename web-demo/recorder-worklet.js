class Recorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(512);
    this.offset = 0;
    this.port.onmessage = ({ data }) => {
      if (data === "flush") {
        this.flush();
        this.port.postMessage({ type: "flushed" });
      }
    };
  }
  flush() {
    if (!this.offset) return;
    const samples =
      this.offset === this.buffer.length
        ? this.buffer
        : this.buffer.slice(0, this.offset);
    this.port.postMessage({ type: "audio", samples }, [samples.buffer]);
    this.buffer = new Float32Array(512);
    this.offset = 0;
  }
  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let index = 0; index < channels[0].length; index++) {
      let sum = 0;
      for (const channel of channels) sum += channel[index];
      this.buffer[this.offset++] = sum / channels.length;
      if (this.offset === this.buffer.length) this.flush();
    }
    return true;
  }
}
registerProcessor("recorder", Recorder);
