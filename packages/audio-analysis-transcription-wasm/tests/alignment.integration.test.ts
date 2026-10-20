import { expect, test } from "bun:test";
import { createBrowserAlignmentAdapter } from "../alignment";

test("alignment consumes model emissions once and preserves source text and identities", async () => {
  const calls: string[] = [];
  const adapter = createBrowserAlignmentAdapter({
    loadModel: async (model) => {
      calls.push(model.language);
      return {
        vocabulary: { blankId: 0, tokens: ["<pad>", "|", "A", "B"] },
        emit: async (samples) => {
          calls.push(`emit:${samples.length}`);
          return Array.from({ length: 6 }, () => [-0.1, -3, -3, -3]);
        },
        dispose: async () => {},
      };
    },
    alignEmissions: async (request) => {
      calls.push(request.text);
      return [
        {
          segmentIndex: 7,
          wordIndex: 0,
          text: "A",
          startSeconds: 0.1,
          endSeconds: 0.3,
          confidence: 0.8,
        },
        {
          segmentIndex: 7,
          wordIndex: 1,
          text: "B",
          startSeconds: 0.5,
          endSeconds: 0.8,
          confidence: 0.9,
        },
      ];
    },
  });
  const source = {
    text: "A B",
    language: "en",
    source: "fixture",
    attributes: {},
    segments: [
      {
        index: 7,
        startSeconds: 0,
        endSeconds: 1,
        text: "A B",
        language: "en",
        speaker: null,
        confidence: null,
        isFinal: true,
        words: [],
        chars: [],
        attributes: {},
      },
    ],
  };
  const before = structuredClone(source);
  const result = await adapter.align(new Float32Array(16_000), source, { language: "en" });
  expect(calls).toEqual(["en", "emit:16000", "A B"]);
  expect(source).toEqual(before);
  expect(result.text).toBe(source.text);
  expect(result.segments[0]?.index).toBe(7);
  expect(result.segments[0]?.startSeconds).toBe(0.1);
  expect(result.segments[0]?.endSeconds).toBe(0.8);
  expect(result.segments[0]?.words).toHaveLength(2);
  await expect(adapter.align(new Float32Array(16_000), source, { language: "ja" })).rejects.toThrow(
    "alignment language",
  );
  expect(calls).toHaveLength(3);
});
