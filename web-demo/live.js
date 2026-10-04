import { concatenate } from "./audio.js";
import { timestamp } from "./timing.js";

// Each utterance has one final job. The caller can skip previews before we copy
// their audio; final audio is always delivered, even when inference is busy.
export class LiveSegmenter {
  constructor(
    onSnapshot,
    canPreview = (_, elapsed) => elapsed >= 1.5,
    {
      maxSeconds = 8,
      targetSeconds = maxSeconds,
      pauseSeconds = 0.6,
      canFinalize = () => true,
    } = {},
  ) {
    this.onSnapshot = onSnapshot;
    this.canPreview = canPreview;
    this.maxSeconds = maxSeconds;
    this.targetSeconds = targetSeconds;
    this.pauseSeconds = pauseSeconds;
    this.canFinalize = canFinalize;
    this.id = 0;
    this.reset();
  }
  reset() {
    this.chunks = [];
    this.length = 0;
    this.silence = 0;
    this.heardSpeech = false;
    this.lastPreview = 0;
    this.audioStartedAt = null;
    this.speechEndedAt = null;
    this.peakRms = 0;
  }
  add(samples, rate, receivedAt = timestamp()) {
    this.rate = rate;
    if (!this.length)
      this.audioStartedAt = receivedAt - (samples.length / rate) * 1000;
    this.chunks.push(samples);
    this.length += samples.length;
    const rms = Math.sqrt(
      samples.reduce((sum, sample) => sum + sample * sample, 0) /
        samples.length,
    );
    this.peakRms = Math.max(
      rms,
      this.peakRms * Math.exp(-samples.length / rate / 2),
    );
    // Follow speech volume so a little background hiss does not hide pauses.
    // The fixed floor still handles quiet microphones; the ceiling limits how
    // much a loud transient can raise the threshold for subsequent speech.
    const speechThreshold = Math.max(0.008, Math.min(0.02, this.peakRms * 0.1));
    if (rms >= speechThreshold) {
      this.heardSpeech = true;
      this.silence = 0;
      this.speechEndedAt = receivedAt;
    } else this.silence += samples.length;
    if (!this.heardSpeech && this.length > rate) {
      this.chunks = [samples];
      this.length = samples.length;
      this.lastPreview = 0;
      this.audioStartedAt = receivedAt - (samples.length / rate) * 1000;
      return;
    }
    if (!this.heardSpeech) return;
    // Near the duration limit, use a brief gap between words when possible.
    // Short phrases are processed once; the next phrase starts with fresh audio.
    const nearLimit = this.length >= (this.targetSeconds - 0.75) * rate;
    const pause =
      this.length >= rate &&
      (this.silence >= this.pauseSeconds * rate ||
        (nearLimit && this.silence >= 0.096 * rate));
    // If inference is occupied, accumulate a longer continuous phrase instead
    // of queuing many tiny calls. Real pauses and Stop always retain their audio.
    const targetReached =
      this.length >= this.targetSeconds * rate && this.canFinalize();
    if (pause || targetReached || this.length >= this.maxSeconds * rate) {
      this.finish(pause ? "pause" : "duration");
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
        audioStartedAt: this.audioStartedAt,
        speechEndedAt: this.speechEndedAt,
      });
      this.lastPreview = this.length;
    }
  }
  finish(boundary = "stop") {
    if (this.heardSpeech && this.length >= 320) {
      this.onSnapshot({
        segment: this.id,
        final: true,
        samples: concatenate(this.chunks),
        rate: this.rate,
        audioStartedAt: this.audioStartedAt,
        speechEndedAt: this.speechEndedAt,
        boundary,
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
