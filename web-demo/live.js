import { concatenate } from "./audio.js";

// Each utterance has one final job. The caller can skip previews before we copy
// their audio; final audio is always delivered, even when inference is busy.
export class LiveSegmenter {
  constructor(onSnapshot, canPreview = (_, elapsed) => elapsed >= 1.5) {
    this.onSnapshot = onSnapshot;
    this.canPreview = canPreview;
    this.id = 0;
    this.reset();
  }
  reset() {
    this.chunks = [];
    this.length = 0;
    this.silence = 0;
    this.heardSpeech = false;
    this.lastPreview = 0;
  }
  add(samples, rate) {
    this.rate = rate;
    this.chunks.push(samples);
    this.length += samples.length;
    const rms = Math.sqrt(
      samples.reduce((sum, sample) => sum + sample * sample, 0) /
        samples.length,
    );
    if (rms >= 0.008) {
      this.heardSpeech = true;
      this.silence = 0;
    } else this.silence += samples.length;
    if (!this.heardSpeech && this.length > rate) {
      this.chunks = [samples];
      this.length = samples.length;
      this.lastPreview = 0;
      return;
    }
    if (!this.heardSpeech) return;
    if (
      (this.silence >= 0.6 * rate && this.length >= rate) ||
      this.length >= 8 * rate
    ) {
      this.finish();
      return;
    }
    if (
      this.canPreview(
        this.length / rate,
        (this.length - this.lastPreview) / rate,
      )
    ) {
      this.onSnapshot({
        segment: this.id,
        final: false,
        samples: concatenate(this.chunks),
        rate,
      });
      this.lastPreview = this.length;
    }
  }
  finish() {
    if (this.heardSpeech && this.length >= 320) {
      this.onSnapshot({
        segment: this.id,
        final: true,
        samples: concatenate(this.chunks),
        rate: this.rate,
      });
      this.id++;
    }
    this.reset();
  }
}

export function stableWordCount(previous, current, final, duration) {
  if (final) return current.length;
  let count = 0;
  const normalize = (value) =>
    value.toLocaleLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
  while (count < Math.min(previous.length, current.length - 1)) {
    if (
      normalize(previous[count].word) !== normalize(current[count].word) ||
      current[count].end > duration - 0.24
    )
      break;
    count++;
  }
  return count;
}
