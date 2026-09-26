import { expect, test } from "bun:test";

test("audio-analysis-speakers-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
  expect(typeof entry.browserDiarizationCapabilities).toBe("function");
  expect(typeof entry.supportsBrowserDiarization).toBe("function");
  expect(typeof entry.diarizeBrowserAudioSamples).toBe("function");
  expect(typeof entry.assignBrowserDiarizationToTranscript).toBe("function");
});

test("browser diarization capability stays local and bounded", async () => {
  const entry = await import("../index.js");
  const capabilities = entry.browserDiarizationCapabilities();

  expect(capabilities.input).toEqual({
    sampleRateHz: 16_000,
    channels: 1,
    sampleFormat: "f32",
  });
  expect(capabilities.chunking.windowSeconds).toBe(10);
  expect(capabilities.clustering.metric).toBe("cosine");
  expect(capabilities.clustering.defaultThreshold).toBeGreaterThan(0);
  expect(capabilities.clustering.defaultThreshold).toBeLessThan(1);
  expect(capabilities.backends).toEqual({
    segmentation: "wasm-q8",
    speakerEmbedding: "wasm-q8",
  });
  expect(capabilities.features.diarization).toBe(true);
  expect(capabilities.features.overlapAwareSegmentation).toBe(true);
  expect(capabilities.features.globalSpeakerClustering).toBe(true);
  expect(capabilities.features.transcriptAssignment).toBe(true);
  expect(capabilities.fallbacks.server).toBe(false);
  expect(capabilities.fallbacks.python).toBe(false);
});

test("browser diarization assigns transcript segments by majority overlap", async () => {
  const entry = await import("../index.js");
  const transcript = {
    text: "alpha beta",
    segments: [
      { index: 0, startSeconds: 0, endSeconds: 1.2, text: "alpha", speaker: null },
      { index: 1, startSeconds: 1.2, endSeconds: 2.4, text: "beta", speaker: null },
    ],
  };
  const diarization = {
    accepted: true,
    operation: "diarize",
    modelId: "fixture",
    runtime: "fixture",
    speakerCount: 2,
    diagnostics: [],
    segments: [
      { speaker: "speaker_00", startSeconds: 0, endSeconds: 1, score: 0.9 },
      { speaker: "speaker_01", startSeconds: 1, endSeconds: 2.4, score: 0.8 },
    ],
  };

  const assigned = entry.assignBrowserDiarizationToTranscript(transcript, diarization);

  expect(assigned.segments[0].speaker).toBe("speaker_00");
  expect(assigned.segments[1].speaker).toBe("speaker_01");
  expect(transcript.segments[0].speaker).toBeNull();
});

test("browser diarization assignment preserves untimed segments", async () => {
  const entry = await import("../index.js");
  const transcript = {
    segments: [{ index: 0, startSeconds: null, endSeconds: null, text: "untimed" }],
  };
  const diarization = {
    segments: [{ speaker: "speaker_00", startSeconds: 0, endSeconds: 1, score: 1 }],
  };

  const assigned = entry.assignBrowserDiarizationToTranscript(transcript, diarization);
  expect(assigned.segments[0]).toEqual(transcript.segments[0]);
});
