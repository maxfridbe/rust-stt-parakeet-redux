export class SpeechEcho {
  constructor(select, report) {
    this.select = select;
    this.report = report;
    this.voices = [];
    this.supported = "speechSynthesis" in window;
    if (!this.supported) return;
    this.refreshVoices();
    speechSynthesis.addEventListener("voiceschanged", () =>
      this.refreshVoices(),
    );
  }
  refreshVoices() {
    const previous = this.select.value;
    this.voices = speechSynthesis.getVoices();
    this.select.replaceChildren(
      new Option("Browser default (may use network)", ""),
    );
    this.voices.forEach((voice, index) =>
      this.select.add(
        new Option(
          `${voice.name} · ${voice.lang}${voice.localService ? " · local" : " · network"}`,
          String(index),
        ),
      ),
    );
    const local = this.voices.findIndex(
      (voice) =>
        voice.localService &&
        voice.lang.startsWith(navigator.language.split("-")[0]),
    );
    this.select.value = previous || (local >= 0 ? String(local) : "");
  }
  say(text) {
    if (!this.supported || !text.trim()) return;
    const utterance = new SpeechSynthesisUtterance(text);
    if (this.select.value !== "")
      utterance.voice = this.voices[Number(this.select.value)];
    utterance.rate = 1.1;
    const queued = performance.now();
    utterance.onstart = () =>
      this.report(
        `Echo started ${Math.round(performance.now() - queued)} ms after word confirmation`,
      );
    utterance.onerror = (event) => {
      if (!["canceled", "interrupted"].includes(event.error))
        this.report(`Speech playback: ${event.error}`);
    };
    speechSynthesis.speak(utterance);
  }
  cancel() {
    if (this.supported) speechSynthesis.cancel();
  }
}
