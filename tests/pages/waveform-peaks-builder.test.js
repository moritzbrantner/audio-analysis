// The peak worker receives the PCM in bounded chunks (#140); the incremental builder must produce the
// same pyramid as a one-pass build, whatever the chunk boundaries.
import { describe, expect, test } from "bun:test";
import { buildPeakLevels, createPeakBuilder, viewportPeaks, buildPeakPyramid } from "../../site/waveform-peaks.js";

function signal(length, seed) {
  const samples = new Float32Array(length);
  let state = seed;
  for (let index = 0; index < length; index += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    samples[index] = state / 2_147_483_648 - 1;
  }
  return samples;
}

describe("incremental peak builder", () => {
  test("chunked appends equal a one-pass build for mono and stereo", () => {
    const length = 10_007;
    for (const channels of [[signal(length, 1)], [signal(length, 2), signal(length, 3)]]) {
      const expected = buildPeakLevels(channels, 8_000);
      for (const chunk of [1, 31, 32, 33, 1_000, 4_096]) {
        const builder = createPeakBuilder(channels.length, length, 8_000);
        for (let offset = 0; offset < length; offset += chunk) {
          builder.append(channels.map((channel) => channel.subarray(offset, Math.min(length, offset + chunk))));
        }
        const summary = builder.finish();
        expect(summary.length).toBe(expected.length);
        expect(summary.levels.length).toBe(expected.levels.length);
        summary.levels.forEach((level, index) => {
          expect(Array.from(level.min)).toEqual(Array.from(expected.levels[index].min));
          expect(Array.from(level.max)).toEqual(Array.from(expected.levels[index].max));
        });
      }
    }
  });

  test("finish refuses an incomplete stream", () => {
    const builder = createPeakBuilder(1, 100, 8_000);
    builder.append([new Float32Array(60)]);
    expect(() => builder.finish()).toThrow();
  });

  test("the last column of a direct-PCM view reaches the view's final frame", () => {
    const samples = new Float32Array(8_000);
    samples[7_999] = 0.8;
    const pyramid = buildPeakPyramid([samples], 8_000);
    // 3 columns over 10 frames: fractional frames per column, raw path.
    const { max } = viewportPeaks(pyramid, 7_990 / 8_000, 1, 3);
    expect(max[2]).toBeCloseTo(0.8, 6);
  });
});
