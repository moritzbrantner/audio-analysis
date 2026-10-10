// Module worker that prepares the full-file PCM and runs the Rust/WASM whole-song analysis off the
// Audio Inspector UI thread.
import { analyzeWholeTrack, prepareWholeTrackSamples } from "./whole-track-analysis.js";

self.addEventListener("message", async (event) => {
  const { id, channels, sourceSampleRate } = event.data ?? {};
  try {
    const { samples, sampleRate } = prepareWholeTrackSamples(channels, sourceSampleRate);
    const result = await analyzeWholeTrack(samples, sampleRate);
    self.postMessage({ id, ok: true, analysisSampleRate: sampleRate, ...result });
  } catch (error) {
    self.postMessage({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
});
