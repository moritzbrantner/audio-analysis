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
  wholeTrackTimeline,
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

describe("whole-track musical timeline", () => {
  const windowRhythm = {
    analysisStartSeconds: 0,
    beats: [{ index: 1, timestampSeconds: 1, downbeat: true }],
    sections: [{ index: 1, startSeconds: 0, endSeconds: 20, identity: "A" }],
  };

  function wholeTrackReport({ bBpm = 150, bKey = { tonic: "F#", scale: "major" } } = {}) {
    return {
      source: { durationSeconds: 72 },
      coverage: { rhythm: { startSeconds: 26, sourceDurationSeconds: 20 } },
      rhythm: windowRhythm,
      wholeTrack: {
        status: "complete",
        song: {
          schemaVersion: "audio-analysis-song/v1",
          analysisStartSeconds: 0,
          analysisEndSeconds: 72,
          beatsPerBar: 4,
          bpm: 120,
          tempoMap: [
            { timestampSeconds: 2, bpm: 119.8 },
            { timestampSeconds: 12, bpm: 120.2 },
            { timestampSeconds: 30, bpm: bBpm },
            { timestampSeconds: 40, bpm: bBpm },
            { timestampSeconds: 60, bpm: 120 },
          ],
          beats: [
            { index: 1, timestampSeconds: 0.5, downbeat: true, localBpm: 120, barIndex: 1, beatInBar: 1 },
            { index: 2, timestampSeconds: 70, downbeat: false, localBpm: 120, barIndex: 36, beatInBar: 2 },
          ],
          sections: [
            { index: 1, identity: "A", label: "section-1", startSeconds: 0, endSeconds: 24, startBoundaryConfidence: 1 },
            { index: 2, identity: "B", label: "section-2", startSeconds: 24, endSeconds: 48, startBoundaryConfidence: 0.31 },
            { index: 3, identity: "A", label: "section-3", startSeconds: 48, endSeconds: 72, startBoundaryConfidence: 0.29 },
          ],
        },
        key: {
          schemaVersion: "audio-analysis-key-track/v2",
          durationSeconds: 72,
          segments: [
            { startSeconds: 0, endSeconds: 24, confidence: 0.6, key: { tonic: "C", scale: "major" } },
            { startSeconds: 24, endSeconds: 48, confidence: 0.5, key: bKey },
            { startSeconds: 48, endSeconds: 72, confidence: 0.6, key: { tonic: "C", scale: "major" } },
          ],
        },
      },
    };
  }

  test("uses the Rust whole-song sections and beats across the complete file", () => {
    const report = wholeTrackReport();
    expect(wholeTrackTimeline(report)).not.toBeNull();
    expect(rhythmCoverage(report)).toBeNull();
    expect(beatOverlayEvents(report).map((event) => event.timeSeconds)).toEqual([0.5, 70]);
    const sections = sectionOverlaySegments(report);
    expect(sections.map(({ identity, startSeconds, endSeconds }) => [identity, startSeconds, endSeconds])).toEqual([
      ["A", 0, 24],
      ["B", 24, 48],
      ["A", 48, 72],
    ]);
    expect(sectionOverlayStatusText(report)).toBe("3 detected sections across the whole track");
    expect(beatOverlayStatusText(report)).toBe("2 detected beats across the whole track");
    expect(formatMusicalContext(musicalContextAtTime(report, 71))).toBe("1:11.0 · Section A · Bar 36 · Beat 2/4 · 120.0 BPM");
  });

  test("extends the final region over the resampling remainder of the decoded file", () => {
    const report = wholeTrackReport();
    report.source.durationSeconds = 72.00004;
    const sections = sectionOverlaySegments(report);
    expect(sections.at(-1).endSeconds).toBe(72.00004);
    expect(musicalContextAtTime(report, 72.00004)?.sectionIdentity).toBe("A");
  });

  test("annotates sustained tempo and key changes on every neighbouring region", () => {
    const sections = sectionOverlaySegments(wholeTrackReport());
    expect(sections.map(({ localBpm, key, tempoChanged, keyChanged }) => [localBpm, key?.label, tempoChanged, keyChanged])).toEqual([
      [120, "C major", true, true],
      [150, "F# major", true, true],
      [120, "C major", true, true],
    ]);
  });

  test("does not annotate estimator jitter", () => {
    const sections = sectionOverlaySegments(wholeTrackReport({ bBpm: 120.5, bKey: { tonic: "C", scale: "major" } }));
    expect(sections).toHaveLength(3);
    expect(sections.every((section) => !section.tempoChanged && !section.keyChanged)).toBe(true);
  });

  test("does not present the rhythm window as structure while the whole track is analyzed", () => {
    const report = { ...wholeTrackReport(), wholeTrack: { status: "pending" } };
    expect(sectionOverlaySegments(report)).toEqual([]);
    expect(sectionOverlayStatusText(report)).toBe("Analyzing whole-track sections");
    expect(beatOverlayEvents(report).map((event) => event.timeSeconds)).toEqual([27]);
  });

  test("falls back to the labelled rhythm window when whole-track analysis failed", () => {
    const report = { ...wholeTrackReport(), wholeTrack: { status: "failed", error: "boom" } };
    expect(sectionOverlaySegments(report)).toHaveLength(1);
    expect(sectionOverlayStatusText(report)).toBe("1 detected section in the 20.0 s rhythm window");
  });
});
