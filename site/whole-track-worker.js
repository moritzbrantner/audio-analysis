// Module worker that runs the Rust/WASM whole-song analysis off the Audio Inspector UI thread.
import { analyzeWholeTrack } from "./whole-track-analysis.js";

self.addEventListener("message", async (event) => {
  const { id, samples, sampleRate, durationSeconds } = event.data ?? {};
  try {
    const result = await analyzeWholeTrack(samples, sampleRate, durationSeconds);
    self.postMessage({ id, ok: true, ...result });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
});
