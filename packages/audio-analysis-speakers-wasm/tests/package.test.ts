import { expect, test } from "bun:test";

test("audio-analysis-speakers-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
  expect(typeof entry.browserDiarizationCapabilities).toBe("function");
  expect(typeof entry.supportsBrowserDiarization).toBe("function");
  expect(typeof entry.diarizeAudioBlob).toBe("function");
  expect(typeof entry.diarizeAudioSamples).toBe("function");
  expect(typeof entry.assignBrowserSpeakersToSegments).toBe("function");
});

test("browser diarization returns local anonymous speaker segments", async () => {
  const { diarizeAudioSamples } = await import("../index.js");
  const sampleRateHz = 16_000;
  const samples = new Float32Array(sampleRateHz * 3);
  for (let index = 0; index < samples.length; index += 1) {
    const seconds = index / sampleRateHz;
    const frequency = seconds < 1.5 ? 180 : 740;
    samples[index] = Math.sin(2 * Math.PI * frequency * seconds) * 0.3;
  }

  const result = diarizeAudioSamples(samples, {
    windowSeconds: 0.75,
    hopSeconds: 0.5,
    vadThreshold: 0.01,
    clusterThreshold: 0.12,
  });

  expect(result.runtime).toBe("audio-analysis-browser-spectral");
  expect(result.modelId).toBe("spectral-speaker-baseline");
  expect(result.segments.length).toBeGreaterThan(0);
  expect(result.segments.every((segment) => segment.speaker.startsWith("speaker_"))).toBe(true);
});

test("browser transcript assignment uses greatest overlap", async () => {
  const { assignBrowserSpeakersToSegments } = await import("../index.js");
  const assigned = assignBrowserSpeakersToSegments(
    [
      { index: 0, startSeconds: 0, endSeconds: 1, text: "hello", speaker: null },
      { index: 1, startSeconds: 1, endSeconds: 2, text: "there", speaker: null },
    ],
    {
      accepted: true,
      operation: "diarize",
      modelId: "fixture",
      runtime: "fixture",
      speakerCount: 2,
      segments: [
        { speaker: "speaker_0", startSeconds: 0, endSeconds: 1.1, score: 1 },
        { speaker: "speaker_1", startSeconds: 0.9, endSeconds: 2, score: 1 },
      ],
      attributes: {},
      diagnostics: [],
    },
  );

  expect(assigned.map((segment) => segment.speaker)).toEqual(["speaker_0", "speaker_1"]);
});
