// Module worker that builds the waveform's min/max peak pyramid off the Audio Inspector UI thread.
// The page streams the PCM in bounded chunks (one in flight at a time, acknowledged after each), so
// no complete copy of the file is made; the summary's typed arrays are transferred back. The page
// keeps its own PCM for sub-bucket views.
//
// Protocol: { type: "start", channelCount, length, sampleRate } → { type: "chunk", channels } … each
// answered with { type: "ack" } → { type: "finish" } answered with { type: "done", ok, summary | error }.
import { createPeakBuilder, peakSummaryTransferables } from "./waveform-peaks.js";

let builder = null;

self.addEventListener("message", (event) => {
  const message = event.data ?? {};
  try {
    if (message.type === "start") {
      builder = createPeakBuilder(message.channelCount, message.length, message.sampleRate);
    } else if (message.type === "chunk") {
      if (!builder) throw new Error("peak worker received a chunk before start");
      builder.append(message.channels);
      self.postMessage({ type: "ack" });
    } else if (message.type === "finish") {
      if (!builder) throw new Error("peak worker received finish before start");
      const summary = builder.finish();
      builder = null;
      self.postMessage({ type: "done", ok: true, summary }, peakSummaryTransferables(summary));
    }
  } catch (error) {
    builder = null;
    self.postMessage({ type: "done", ok: false, error: error instanceof Error ? error.message : String(error) });
  }
});
