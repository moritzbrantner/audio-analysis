import { describe, expect, test } from "bun:test";
import { beatOverlayEvents, overlayStatusText, rhythmCoverage } from "../../site/waveform-overlay.js";

describe("waveform beat overlay", () => {
  test("projects center-window beat timestamps onto the full-file waveform", () => {
    const report = {
      source: { durationSeconds: 168 },
      coverage: { rhythm: { startSeconds: 74, sourceDurationSeconds: 20 } },
      rhythm: {
        analysisStartSeconds: 0,
        beats: [
          { index: 1, timestampSeconds: 0.5, strength: 0.8, downbeat: true },
          { index: 2, timestampSeconds: 1.0, strength: 0.5, downbeat: false },
        ],
      },
    };

    expect(beatOverlayEvents(report)).toEqual([
      { index: 1, timeSeconds: 74.5, strength: 0.8, downbeat: true },
      { index: 2, timeSeconds: 75, strength: 0.5, downbeat: false },
    ]);
    expect(rhythmCoverage(report)).toEqual({ startSeconds: 74, endSeconds: 94 });
    expect(overlayStatusText(report)).toBe("2 detected beats from the 20.0 s rhythm window");
  });

  test("does not double-offset absolute rhythm timestamps", () => {
    const report = {
      source: { durationSeconds: 168 },
      coverage: { rhythm: { startSeconds: 74, sourceDurationSeconds: 20 } },
      rhythm: {
        analysisStartSeconds: 74,
        beats: [{ timestampSeconds: 74.5, downbeat: false }],
      },
    };

    expect(beatOverlayEvents(report)[0].timeSeconds).toBe(74.5);
  });

  test("filters unusable or out-of-file beat timestamps", () => {
    const report = {
      source: { durationSeconds: 10 },
      coverage: { rhythm: { startSeconds: 0, sourceDurationSeconds: 10 } },
      rhythm: {
        analysisStartSeconds: 0,
        beats: [
          { timestampSeconds: -1 },
          { timestampSeconds: 2 },
          { timestampSeconds: 11 },
          { timestampSeconds: "not-a-number" },
        ],
      },
    };

    expect(beatOverlayEvents(report).map((event) => event.timeSeconds)).toEqual([2]);
  });
});
