// Reserve most processing time for final utterances. Touch devices start with
// finals only until a real inference measurement establishes spare capacity.
export class PreviewPolicy {
  constructor({ conservative = false, shortPhrases = false } = {}) {
    this.mode = "auto";
    this.shortPhrases = shortPhrases;
    this.pauseOnly = conservative;
    this.seconds = 0;
    this.ratio = 0;
  }

  observe(inferenceSeconds, audioSeconds) {
    if (!(inferenceSeconds > 0 && audioSeconds > 0)) return;
    this.seconds = inferenceSeconds;
    this.ratio = inferenceSeconds / audioSeconds;
    if (this.ratio >= 0.5) this.pauseOnly = true;
    if (this.ratio <= 0.3) this.pauseOnly = false;
  }

  allows(duration, sincePreview, idle) {
    if (!idle || this.mode === "pauses") return false;
    if (this.mode === "frequent") return sincePreview >= 1.5;
    if (this.shortPhrases) return false;
    if (this.pauseOnly) return false;
    const estimatedSeconds = Math.max(this.seconds, duration * this.ratio);
    return sincePreview >= Math.max(1.5, 3 * estimatedSeconds);
  }

  get description() {
    const cadence = this.shortPhrases
      ? "Short phrases: submit at a brief pause or around 3 seconds. While busy, collect longer phrases (up to 8 seconds)."
      : "Long phrases: submit at a pause or after 8 seconds.";
    const capacity =
      this.ratio > 1
        ? " Processing is slower than live speech; pauses let it catch up."
        : " Listening continues while each phrase is processed.";
    if (this.shortPhrases && this.mode !== "frequent")
      return cadence + capacity;
    if (this.mode === "pauses" || (this.mode === "auto" && this.pauseOnly))
      return cadence + capacity;
    if (this.mode === "frequent")
      return "Frequent previews when the processor is free. Uses more battery; final words wait for confirmation.";
    return "Preview frequency adapts to measured speed, leaving processing time for completed utterances.";
  }
}
