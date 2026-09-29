import { beforeAll, expect, test } from "bun:test";

import init, { analyzeAudioSamples, mixToMono, planAudioFrames } from "./index.js";

beforeAll(async () => {
  await init();
});

test("analyzes audio samples through packaged wasm bindings", () => {
  const sampleRate = 48_000;
  const samples = Float32Array.from({ length: 4096 }, (_, index) =>
    Math.sin((index * 440 * Math.PI * 2) / sampleRate),
  );

  const analysis = analyzeAudioSamples(samples, { sampleRate, fftSize: 2048 });

  expect(analysis.sampleRate).toBe(sampleRate);
  expect(analysis.sampleCount).toBe(4096);
  expect(analysis.peak).toBeGreaterThan(0.99);
  expect(analysis.dominantFrequencyHz).toBeGreaterThan(420);
  expect(analysis.pitch.frequencyHz).toBeGreaterThan(420);
  expect(analysis.pitch.noteName).toBe("A4");
});

test("mixes stereo samples and plans frames", () => {
  expect(mixToMono([1, -1, 0.5, 0.25], 2, "average")).toEqual([0, 0.375]);
  expect(planAudioFrames(10, 4, 3)).toEqual({
    frameSize: 4,
    hopSize: 3,
    sampleCount: 10,
    frameCount: 3,
    starts: [0, 3, 6],
  });
});
