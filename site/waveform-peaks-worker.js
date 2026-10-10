// Module worker that builds the waveform's min/max peak pyramid off the Audio Inspector UI thread.
// The summary's typed arrays are transferred back; the page keeps the PCM for sub-bucket views.
import { buildPeakLevels, peakSummaryTransferables } from "./waveform-peaks.js";

self.addEventListener("message", (event) => {
  const { id, channels, sampleRate } = event.data ?? {};
  try {
    const summary = buildPeakLevels(channels, sampleRate);
    self.postMessage({ id, ok: true, summary }, peakSummaryTransferables(summary));
  } catch (error) {
    self.postMessage({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
});
