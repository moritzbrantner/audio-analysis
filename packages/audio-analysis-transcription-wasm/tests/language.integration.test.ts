import { expect, mock, test } from "bun:test";

test("explicit languages reach each Whisper request without replacing automatic detection", async () => {
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu: { requestAdapter: async () => ({}) } },
  });
  const requests: Record<string, unknown>[] = [];
  let modelLoads = 0;
  mock.module("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1", () => ({
    env: {},
    pipeline: async () => {
      modelLoads += 1;
      return Object.assign(
        async (_samples: Float32Array, options: Record<string, unknown>) => {
          requests.push(options);
          return {
            text: "A model result.",
            chunks: [{ text: "A model result.", timestamp: [0, 1] }],
          };
        },
        { dispose: async () => {} },
      );
    },
  }));
  try {
    const entry = await import("../index.js");
    const samples = new Float32Array(16_000);
    const french = await entry.transcribeAudioSamples(samples, { language: "fr" });
    const japanese = await entry.transcribeAudioSamples(samples, { language: "ja" });
    const automatic = await entry.transcribeAudioSamples(samples);
    expect(requests.map((request) => request.language)).toEqual(["fr", "ja", undefined]);
    expect(Object.hasOwn(requests[2] ?? {}, "language")).toBe(false);
    expect(french.language).toBe("fr");
    expect(french.segments[0]?.language).toBe("fr");
    expect(japanese.language).toBe("ja");
    expect(automatic.language).toBeNull();
    const session = entry.createBrowserTranscriptionSession({ language: "es" });
    await session.push(samples);
    const streamed = await session.flush();
    expect(requests.at(-1)?.language).toBe("es");
    expect(streamed.language).toBe("es");
    expect(streamed.segments[0]?.language).toBe("es");
    expect(entry.browserTranscriptionLanguages()).toEqual(
      expect.arrayContaining(["fr", "ja", "uk", "ar"]),
    );
    await expect(entry.transcribeAudioSamples(samples, { language: "zz" })).rejects.toThrow(
      "Unsupported browser transcription language",
    );
    expect(requests).toHaveLength(4);
    expect(modelLoads).toBe(1);
  } finally {
    if (originalNavigator) Object.defineProperty(globalThis, "navigator", originalNavigator);
    else Reflect.deleteProperty(globalThis, "navigator");
    mock.restore();
  }
});
