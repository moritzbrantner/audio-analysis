import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import {
  beatOverlayEvents,
  beatOverlayStatusText,
  formatMusicalContext,
  musicalContextAtTime,
  rhythmCoverage,
  sectionOverlaySegments,
  sectionOverlayStatusText,
} from "../../site/waveform-overlay.js";

const overlaySource = readFileSync(new URL("../../site/waveform-overlay.js", import.meta.url), "utf8");

describe("waveform rhythm overlays", () => {
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
    expect(beatOverlayStatusText(report)).toBe("2 detected beats from the 20.0 s rhythm window");
  });

  test("projects structural sections from the bounded rhythm window", () => {
    const report = {
      source: { durationSeconds: 168 },
      coverage: { rhythm: { startSeconds: 74, sourceDurationSeconds: 20 } },
      rhythm: {
        analysisStartSeconds: 0,
        sections: [
          {
            index: 1,
            label: "section-1",
            identity: "A",
            startSeconds: 0,
            endSeconds: 8,
            startBoundaryConfidence: 1,
          },
          {
            index: 2,
            label: "section-2",
            identity: "B",
            startSeconds: 8,
            endSeconds: 20,
            startBoundaryConfidence: 0.72,
          },
        ],
      },
    };

    expect(sectionOverlaySegments(report)).toEqual([
      {
        index: 1,
        startSeconds: 74,
        endSeconds: 82,
        identity: "A",
        label: "section-1",
        boundaryConfidence: 1,
      },
      {
        index: 2,
        startSeconds: 82,
        endSeconds: 94,
        identity: "B",
        label: "section-2",
        boundaryConfidence: 0.72,
      },
    ]);
    expect(sectionOverlayStatusText(report)).toBe("2 detected sections in the 20.0 s rhythm window");
  });

  test("builds musical context from section, bar, beat, and local tempo metadata", () => {
    const report = {
      source: { durationSeconds: 30 },
      coverage: { rhythm: { startSeconds: 5, sourceDurationSeconds: 20 } },
      rhythm: {
        analysisStartSeconds: 0,
        bpm: 128,
        beatsPerBar: 4,
        beats: [
          {
            index: 1,
            timestampSeconds: 4,
            localBpm: 127.8,
            beatInBar: 3,
            barIndex: 17,
            sectionIndex: 2,
            sectionIdentity: "B",
          },
          {
            index: 2,
            timestampSeconds: 4.5,
            localBpm: 128.2,
            beatInBar: 4,
            barIndex: 17,
            sectionIndex: 2,
            sectionIdentity: "B",
          },
        ],
        sections: [
          { index: 1, startSeconds: 0, endSeconds: 2, identity: "A" },
          { index: 2, startSeconds: 2, endSeconds: 8, identity: "B" },
        ],
      },
    };

    const context = musicalContextAtTime(report, 9.2);
    expect(context).toMatchObject({
      timeSeconds: 9.2,
      inCoverage: true,
      sectionIndex: 2,
      sectionIdentity: "B",
      barIndex: 17,
      beatInBar: 3,
      beatsPerBar: 4,
      bpm: 127.8,
    });
    expect(formatMusicalContext(context)).toBe("0:09.2 · Section B · Bar 17 · Beat 3/4 · 127.8 BPM");

    const outside = musicalContextAtTime(report, 2);
    expect(outside?.inCoverage).toBe(false);
    expect(formatMusicalContext(outside)).toBe("0:02.0 · Outside analyzed rhythm window");
  });

  test("does not double-offset absolute rhythm timestamps", () => {
    const report = {
      source: { durationSeconds: 168 },
      coverage: { rhythm: { startSeconds: 74, sourceDurationSeconds: 20 } },
      rhythm: {
        analysisStartSeconds: 74,
        beats: [{ timestampSeconds: 74.5, downbeat: false }],
        sections: [{ index: 1, startSeconds: 74, endSeconds: 82, identity: "A" }],
      },
    };

    expect(beatOverlayEvents(report)[0].timeSeconds).toBe(74.5);
    expect(sectionOverlaySegments(report)[0]).toMatchObject({ startSeconds: 74, endSeconds: 82, identity: "A" });
  });

  test("filters unusable or out-of-file rhythm annotations", () => {
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
        sections: [
          { startSeconds: -3, endSeconds: 2, identity: "A" },
          { startSeconds: 2, endSeconds: 7, identity: "B" },
          { startSeconds: 9, endSeconds: 12, identity: "C" },
          { startSeconds: 8, endSeconds: 8, identity: "ignored" },
        ],
      },
    };

    expect(beatOverlayEvents(report).map((event) => event.timeSeconds)).toEqual([2]);
    expect(sectionOverlaySegments(report).map(({ startSeconds, endSeconds, identity }) => ({ startSeconds, endSeconds, identity }))).toEqual([
      { startSeconds: 0, endSeconds: 2, identity: "A" },
      { startSeconds: 2, endSeconds: 7, identity: "B" },
      { startSeconds: 9, endSeconds: 10, identity: "C" },
    ]);
  });

  test("keeps annotations static instead of adding a second playback renderer", () => {
    expect(overlaySource).not.toContain("requestAnimationFrame");
    expect(overlaySource).not.toContain("waveform-presentation");
    expect(overlaySource).not.toContain("getContext(");
    expect(overlaySource).toContain("waveform-overlay-layer");
    expect(overlaySource).toContain("waveform-section-layer");
    expect(overlaySource).toContain("waveform-beat-layer");
    expect(overlaySource).toContain("waveform-structure-rail");
    expect(overlaySource).toContain("waveform-context-hud");
  });
});
