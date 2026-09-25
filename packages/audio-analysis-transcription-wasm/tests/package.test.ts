import { expect, test } from "bun:test";

test("audio-analysis-transcription-wasm package exports stable entrypoints", async () => {
  const entry = await import("../index.js");
  expect(typeof entry.init).toBe("function");
  expect(typeof entry.packageSurface).toBe("function");
  expect(typeof entry.runOperation).toBe("function");
  expect(typeof entry.browserTranscriptionModels).toBe("function");
  expect(typeof entry.browserTranscriptionCapabilities).toBe("function");
  expect(typeof entry.browserTranscriptionWindowPlan).toBe("function");
  expect(typeof entry.stitchBrowserTranscriptionWindow).toBe("function");
  expect(typeof entry.supportsBrowserTranscription).toBe("function");
  expect(typeof entry.transcribeAudioBlob).toBe("function");
  expect(typeof entry.transcribeAudioSamples).toBe("function");
  expect(typeof entry.createBrowserPcmResampler).toBe("function");
  expect(typeof entry.createBrowserDecodedAudioTranscriptionSession).toBe("function");
  expect(typeof entry.createBrowserTranscriptionSession).toBe("function");
  expect(typeof entry.createBrowserMediaStreamTranscriptionSession).toBe("function");
});

test("browser transcription capabilities stay WebGPU-only and bounded", async () => {
  const entry = await import("../index.js");
  const capabilities = entry.browserTranscriptionCapabilities();

  expect(capabilities.requiredAcceleration).toBe("webgpu");
  expect(capabilities.modelId).toBe("onnx-community/whisper-tiny");
  expect(capabilities.models.map((model) => model.id)).toEqual([
    "onnx-community/whisper-tiny",
    "onnx-community/whisper-base",
    "onnx-community/whisper-small",
  ]);
  expect(capabilities.modelLifecycle).toEqual({
    maxIdleResidentModels: 1,
    eviction: "dispose-superseded",
  });
  expect(capabilities.input.sampleRateHz).toBe(16_000);
  expect(capabilities.input.channels).toBe(1);
  expect(capabilities.input.acceptedSources).toContain("caller-acquired MediaStream");
  expect(capabilities.features.transcription).toBe(true);
  expect(capabilities.features.timedSegments).toBe(true);
  expect(capabilities.features.boundedPcmStreaming).toBe(true);
  expect(capabilities.features.decodedAudioAdapter).toBe(true);
  expect(capabilities.features.mediaStreamAdapter).toBe(true);
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

test("MediaStream adapter rejects invalid caller acquisition before browser runtime setup", async () => {
  const entry = await import("../index.js");

  await expect(entry.createBrowserMediaStreamTranscriptionSession(null)).rejects.toThrow(
    "caller-acquired MediaStream",
  );
  await expect(
    entry.createBrowserMediaStreamTranscriptionSession({ getAudioTracks: () => [] }),
  ).rejects.toThrow("contains no audio track");
});


test("browser transcription model catalog is curated and defensive", async () => {
  const entry = await import("../index.js");
  const first = entry.browserTranscriptionModels();
  const second = entry.browserTranscriptionModels();

  expect(first.map((model) => model.label)).toEqual([
    "Whisper Tiny",
    "Whisper Base",
    "Whisper Small",
  ]);
  expect(first).not.toBe(second);
  first[0].label = "mutated";
  expect(entry.browserTranscriptionModels()[0].label).toBe("Whisper Tiny");
});

test("empty bounded sessions preserve the selected model without loading it", async () => {
  const entry = await import("../index.js");
  const session = entry.createBrowserTranscriptionSession({
    source: "empty-base-fixture",
    modelId: "onnx-community/whisper-base",
  });
  const result = await session.flush();

  expect(result.attributes.modelId).toBe("onnx-community/whisper-base");
  expect(result.segments).toEqual([]);
});

test("browser transcription rejects model ids outside the curated catalog", async () => {
  const entry = await import("../index.js");

  expect(() =>
    entry.createBrowserTranscriptionSession({
      modelId: "some-owner/arbitrary-whisper",
    }),
  ).toThrow("Unsupported browser transcription model");
});

test("normalized browser output records the selected model", async () => {
  const entry = await import("../index.js");
  const result = entry.normalizeBrowserTranscriptionOutput(
    {
      text: "selected model",
      chunks: [{ text: " selected model", timestamp: [0, 1] }],
    },
    {
      durationSeconds: 1,
      source: "fixture",
      modelId: "onnx-community/whisper-small",
    },
  );

  expect(result.attributes.modelId).toBe("onnx-community/whisper-small");
  expect(result.segments[0].attributes.modelId).toBe("onnx-community/whisper-small");
});


test("browser PCM resampler preserves continuity across decoded frame boundaries", async () => {
  const entry = await import("../index.js");
  const resampler = entry.createBrowserPcmResampler(48_000);

  const first = resampler.push([
    Float32Array.from({ length: 480 }, (_, index) => index / 480),
    Float32Array.from({ length: 480 }, (_, index) => index / 240),
  ]);
  const second = resampler.push([
    Float32Array.from({ length: 480 }, (_, index) => (480 + index) / 480),
    Float32Array.from({ length: 480 }, (_, index) => (480 + index) / 240),
  ]);

  expect(first.length).toBe(160);
  expect(second.length).toBe(160);
  expect(first[0]).toBeCloseTo(0, 6);
  expect(first[159]).toBeCloseTo((477 / 480 + 477 / 240) / 2, 5);
  expect(second[0]).toBeCloseTo((480 / 480 + 480 / 240) / 2, 5);
  expect(second[159]).toBeCloseTo((957 / 480 + 957 / 240) / 2, 5);
  expect(resampler.outputSampleRateHz).toBe(16_000);
});

test("browser PCM resampler handles non-integer source ratios without resetting phase", async () => {
  const entry = await import("../index.js");
  const resampler = entry.createBrowserPcmResampler(44_100);

  const first = resampler.push([new Float32Array(441).fill(0.25)]);
  const second = resampler.push([new Float32Array(441).fill(0.25)]);

  expect(first.length + second.length).toBe(320);
  expect([...first, ...second].every((sample) => Math.abs(sample - 0.25) < 1e-6)).toBe(true);
});
