import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { buildFrequencyOverview } from "../../site/frequency-overview.js";

const songSource = readFileSync(new URL("../../site/song-analysis.js", import.meta.url), "utf8");
const songHtml = readFileSync(new URL("../../site/song-analysis.html", import.meta.url), "utf8");
const capabilities = JSON.parse(
  readFileSync(new URL("../../site/analysis-capabilities.json", import.meta.url), "utf8"),
);

describe("DJ frequency overview", () => {
  test("separates low and high frequency emphasis deterministically", () => {
    const sampleRate = 16_000;
    const seconds = 1;
    const low = new Float32Array(sampleRate * seconds);
    const high = new Float32Array(sampleRate * seconds);
    for (let index = 0; index < low.length; index += 1) {
      low[index] = Math.sin((2 * Math.PI * 90 * index) / sampleRate) * 0.8;
      high[index] = Math.sin((2 * Math.PI * 4_000 * index) / sampleRate) * 0.8;
    }

    const lowOverview = buildFrequencyOverview(low, sampleRate, 128);
    const highOverview = buildFrequencyOverview(high, sampleRate, 128);
    const lowShares = lowOverview.bins.map((bin) => bin.lowShare);
    const highShares = highOverview.bins.map((bin) => bin.highShare);

    expect(lowOverview.schemaVersion).toBe("audio-analysis-frequency-overview/v1");
    expect(lowOverview.bins).toHaveLength(128);
    expect(Math.max(...lowShares)).toBeGreaterThan(0.5);
    expect(Math.max(...highShares)).toBeGreaterThan(0.5);
  });

  test("keeps visualization separate from Rust-owned analysis truth", () => {
    expect(songSource).toContain("buildFrequencyOverview(samples, analysisRate, OVERVIEW_BIN_COUNT)");
    expect(songSource).toContain("drawFrequencyOverview(context, width, height, state.frequencyOverview)");
    expect(songSource).toContain('authority: "presentation-only"');
    expect(songSource).toContain("drawBeatMarkers(context, width, height, duration, state.analysis.beats)");
    expect(songSource).toContain("drawSectionBoundaries(context, width, height, duration, state.analysis.sections)");
    expect(songHtml).toContain("low-frequency energy");
    expect(capabilities.coverage.wholeSongOverview).toContain("presentation-only browser rendering");
    expect(capabilities.outputs.songOverviewSchema).toBe("audio-analysis-frequency-overview/v1");
  });
});
