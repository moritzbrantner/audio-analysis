import { expect, test } from "bun:test";

test("audio-analysis-transcription-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
  expect(typeof entry.browserTranscriptionCapabilities).toBe("function");
  expect(typeof entry.supportsBrowserTranscription).toBe("function");
  expect(typeof entry.transcribeAudioBlob).toBe("function");
  expect(typeof entry.transcribeAudioSamples).toBe("function");
});

test("browser transcription capabilities stay WebGPU-only and bounded", async () => {
  const entry = await import("../index.js");
  const capabilities = entry.browserTranscriptionCapabilities();

  expect(capabilities.requiredAcceleration).toBe("webgpu");
  expect(capabilities.input.sampleRateHz).toBe(16_000);
  expect(capabilities.input.channels).toBe(1);
  expect(capabilities.features.transcription).toBe(true);
  expect(capabilities.features.timedSegments).toBe(true);
  expect(capabilities.features.alignment).toBe(false);
  expect(capabilities.features.diarization).toBe(false);
  expect(capabilities.features.translation).toBe(false);
  expect(capabilities.fallbacks.server).toBe(false);
  expect(capabilities.fallbacks.python).toBe(false);
  expect(capabilities.fallbacks.cpu).toBe(false);
});

test("browser output normalization preserves timed transcription segments", async () => {
  const entry = await import("../index.js");
  const result = entry.normalizeBrowserTranscriptionOutput(
    {
      text: "hello world",
      chunks: [
        { text: " hello", timestamp: [0, 0.5] },
        { text: " world", timestamp: [0.5, 1.25] },
      ],
    },
    { durationSeconds: 1.25, source: "fixture" },
  );

  expect(result.text).toBe("hello world");
  expect(result.source).toBe("fixture");
  expect(result.segments).toHaveLength(2);
  expect(result.segments[0]).toMatchObject({ startSeconds: 0, endSeconds: 0.5, text: "hello" });
  expect(result.segments[1]).toMatchObject({ startSeconds: 0.5, endSeconds: 1.25, text: "world" });
});
