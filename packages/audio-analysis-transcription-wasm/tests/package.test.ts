import { expect, test } from "bun:test";

test("audio-analysis-transcription-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
  expect(typeof entry.browserTranscriptionCapabilities).toBe("function");
  expect(typeof entry.browserTranscriptionWindowPlan).toBe("function");
  expect(typeof entry.stitchBrowserTranscriptionWindow).toBe("function");
  expect(typeof entry.supportsBrowserTranscription).toBe("function");
  expect(typeof entry.transcribeAudioBlob).toBe("function");
  expect(typeof entry.transcribeAudioSamples).toBe("function");
  expect(typeof entry.createBrowserTranscriptionSession).toBe("function");
});

test("browser transcription capabilities stay WebGPU-only and bounded", async () => {
  const entry = await import("../index.js");
  const capabilities = entry.browserTranscriptionCapabilities();

  expect(capabilities.requiredAcceleration).toBe("webgpu");
  expect(capabilities.input.sampleRateHz).toBe(16_000);
  expect(capabilities.input.channels).toBe(1);
  expect(capabilities.features.transcription).toBe(true);
  expect(capabilities.features.timedSegments).toBe(true);
  expect(capabilities.features.boundedPcmStreaming).toBe(true);
  expect(capabilities.features.alignment).toBe(false);
  expect(capabilities.features.diarization).toBe(false);
  expect(capabilities.features.translation).toBe(false);
  expect(capabilities.streaming.windowSeconds).toBe(29);
  expect(capabilities.streaming.strideSeconds).toBe(5);
  expect(capabilities.streaming.maxBufferedSeconds).toBe(58);
  expect(capabilities.fallbacks.server).toBe(false);
  expect(capabilities.fallbacks.python).toBe(false);
  expect(capabilities.fallbacks.cpu).toBe(false);
});

test("bounded transcription plan uses deterministic 29 second windows", async () => {
  const entry = await import("../index.js");
  const plan = entry.browserTranscriptionWindowPlan();

  expect(plan).toEqual({
    sampleRateHz: 16_000,
    windowSeconds: 29,
    strideSeconds: 5,
    stepSeconds: 24,
    maxBufferedSeconds: 58,
    windowSamples: 464_000,
    strideSamples: 80_000,
    stepSamples: 384_000,
    maxBufferedSamples: 928_000,
  });
  expect(() => entry.browserTranscriptionWindowPlan({ windowSeconds: 30 })).toThrow();
  expect(() => entry.browserTranscriptionWindowPlan({ windowSeconds: 5, strideSeconds: 5 })).toThrow();
  expect(() => entry.browserTranscriptionWindowPlan({ maxBufferedSeconds: 10 })).toThrow();
});

test("bounded transcription plan preserves an explicit zero stride", async () => {
  const entry = await import("../index.js");
  const plan = entry.browserTranscriptionWindowPlan({
    windowSeconds: 10,
    strideSeconds: 0,
    maxBufferedSeconds: 10,
  });

  expect(plan.windowSamples).toBe(160_000);
  expect(plan.strideSamples).toBe(0);
  expect(plan.stepSamples).toBe(160_000);
  expect(plan.stepSeconds).toBe(10);
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

test("browser output normalization offsets bounded windows onto the global timeline", async () => {
  const entry = await import("../index.js");
  const result = entry.normalizeBrowserTranscriptionOutput(
    {
      text: "next window",
      chunks: [{ text: " next window", timestamp: [1, 3.5] }],
    },
    { durationSeconds: 5, offsetSeconds: 24, source: "fixture" },
  );

  expect(result.segments[0]).toMatchObject({
    startSeconds: 25,
    endSeconds: 27.5,
    text: "next window",
  });
});

test("bounded window stitching defers overlap and rejects already committed segments", async () => {
  const entry = await import("../index.js");
  const first = entry.stitchBrowserTranscriptionWindow(
    [
      { text: "committed", startSeconds: 4, endSeconds: 8 },
      { text: "deferred", startSeconds: 23, endSeconds: 26 },
    ],
    { commitUntilSeconds: 24 },
  );

  expect(first.committedThroughSeconds).toBe(24);
  expect(first.segments.map((segment) => segment.text)).toEqual(["committed"]);

  const second = entry.stitchBrowserTranscriptionWindow(
    [
      { text: "old", startSeconds: 20, endSeconds: 23 },
      { text: "deferred", startSeconds: 23, endSeconds: 26 },
      { text: "new", startSeconds: 27, endSeconds: 30 },
    ],
    {
      committedThroughSeconds: first.committedThroughSeconds,
      commitUntilSeconds: 48,
      startIndex: first.segments.length,
    },
  );

  expect(second.committedThroughSeconds).toBe(48);
  expect(second.segments.map((segment) => segment.text)).toEqual(["deferred", "new"]);
  expect(second.segments.map((segment) => segment.index)).toEqual([1, 2]);
});

test("bounded session rejects acquisition that exceeds its PCM backlog before inference", async () => {
  const entry = await import("../index.js");
  const session = entry.createBrowserTranscriptionSession({
    windowSeconds: 1,
    strideSeconds: 0,
    maxBufferedSeconds: 1,
  });

  const tooMuchPcm = new Float32Array(16_001);
  try {
    await session.push(tooMuchPcm);
    throw new Error("Expected bounded backpressure rejection.");
  } catch (error) {
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("BrowserTranscriptionBackpressureError");
  }
});

test("empty bounded session flushes deterministically without loading a model", async () => {
  const entry = await import("../index.js");
  const session = entry.createBrowserTranscriptionSession({ source: "empty-fixture" });
  const result = await session.flush();

  expect(session.closed).toBe(true);
  expect(session.bufferedSeconds).toBe(0);
  expect(result.source).toBe("empty-fixture");
  expect(result.text).toBe("");
  expect(result.segments).toEqual([]);
});
