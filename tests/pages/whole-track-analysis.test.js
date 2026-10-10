import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { keyBarBoundaries, prepareWholeTrackSamples } from "../../site/whole-track-analysis.js";

const appSource = readFileSync(new URL("../../site/app.js", import.meta.url), "utf8");

describe("whole-track analysis preparation", () => {
  test("keeps key bar boundaries inside the analyzed sample duration", () => {
    // 44.1 kHz → 16 kHz floors the sample count, so the analyzed duration is shorter than decoded.
    const decodedFrames = 44_100 * 10 + 7;
    const { samples, sampleRate } = prepareWholeTrackSamples([new Float32Array(decodedFrames)], 44_100);
    const analyzed = samples.length / sampleRate;
    expect(analyzed).toBeLessThan(decodedFrames / 44_100);

    const boundaries = keyBarBoundaries([0.5, 4, 8, decodedFrames / 44_100], analyzed);
    expect(boundaries[0]).toBe(0);
    expect(boundaries.at(-1)).toBe(analyzed);
    expect(boundaries.every((value) => value >= 0 && value <= analyzed)).toBe(true);
    expect(boundaries.every((value, index) => index === 0 || value > boundaries[index - 1])).toBe(true);
  });

  test("mixes channels to mono and resamples to the analysis rate", () => {
    const left = Float32Array.from({ length: 48_000 }, () => 0.5);
    const right = Float32Array.from({ length: 48_000 }, () => -0.1);
    const { samples, sampleRate } = prepareWholeTrackSamples([left, right], 48_000);
    expect(sampleRate).toBe(16_000);
    expect(samples).toBeInstanceOf(Float32Array);
    expect(samples.length).toBe(16_000);
    expect(samples[100]).toBeCloseTo(0.2, 6);
  });

  test("prepares full-file PCM in the worker instead of on the UI thread", () => {
    expect(appSource).not.toContain("mixAndResampleMono");
    expect(appSource).toContain("sourceSampleRate: buffer.sampleRate");
  });
});
