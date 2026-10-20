import { expect, test } from "bun:test";
import {
  browserCommunityDiarizationCapabilities,
  createBrowserCommunityDiarizationAdapter,
} from "../community.js";

const capability = browserCommunityDiarizationCapabilities();
const bundle = {
  modelId: capability.modelId,
  sourceRevision: capability.sourceRevision,
  artifactSetSha256: capability.artifactSetSha256,
  files: {},
};
function fixture() {
  const counts = { segmentation: 0, embedding: 0, finish: 0, free: 0, models: 0, processors: 0 };
  let fail = false;
  const adapter = createBrowserCommunityDiarizationAdapter({
    async loadInference() {
      counts.models++;
      return {
        async segment(samples) {
          expect(samples.length).toBe(160000);
          counts.segmentation++;
          return new Float32Array(589 * 3);
        },
        async embed(_samples, mask) {
          expect(mask.length).toBe(589);
          counts.embedding++;
          if (fail) {
            throw Error("model execution failed");
          }
          return new Float32Array(256);
        },
        async dispose() {},
      };
    },
    async createProcessor() {
      counts.processors++;
      return {
        pushSegmentation() {
          return [{ localSpeaker: 0, cleanFrames: 589, mask: Array(589).fill(1) }];
        },
        pushEmbedding() {},
        finish() {
          counts.finish++;
          return {
            modelId: capability.modelId,
            segments: [{ speaker: "SPEAKER_00", startSeconds: 0.03, endSeconds: 0.5, score: null }],
            diagnostics: [],
          };
        },
        free() {
          counts.free++;
        },
      };
    },
  });
  return {
    counts,
    adapter,
    failNext() {
      fail = true;
    },
    recover() {
      fail = false;
    },
  };
}

test("real-model boundary keeps tentative counts, source timing and model reuse", async () => {
  const { counts, adapter } = fixture();
  const samples = new Float32Array(16000).fill(0.1);
  const before = samples.slice();
  const result = await adapter.diarize(samples, bundle);
  await adapter.diarize(samples, bundle);
  expect(result.speakerCount).toBe(1);
  expect(result.attributes.countStatus).toBe("tentativeAutomatic");
  expect(result.attributes.countUncertainty).toBe("notReported");
  expect(result.segments[0]?.startSeconds).toBe(0.03);
  expect(result.segments[0]?.score).toBeNull();
  expect(samples).toEqual(before);
  expect(counts).toEqual({
    segmentation: 2,
    embedding: 2,
    finish: 2,
    free: 2,
    models: 1,
    processors: 2,
  });
});

test("browser task cancellation stops before another window and releases the processor", async () => {
  const { counts, adapter } = fixture();
  const controller = new AbortController();
  await expect(
    adapter.diarize(new Float32Array(176000), bundle, {
      signal: controller.signal,
      onProgress: () => {
        setTimeout(() => controller.abort(), 0);
      },
    }),
  ).rejects.toThrow();
  expect(counts.segmentation).toBe(1);
  expect(counts.finish).toBe(0);
  expect(counts.free).toBe(1);
});

test("execution failures produce no result, then a retry uses a fresh processor", async () => {
  const { counts, adapter, failNext, recover } = fixture();
  failNext();
  await expect(adapter.diarize(new Float32Array(16000), bundle)).rejects.toThrow(
    "model execution failed",
  );
  expect(counts.finish).toBe(0);
  expect(counts.free).toBe(1);
  recover();
  expect((await adapter.diarize(new Float32Array(16000), bundle)).speakerCount).toBe(1);
  expect(counts.processors).toBe(2);
  expect(counts.models).toBe(1);
});

test("invalid audio and unsupported bundles fail before loading any model", async () => {
  const { counts, adapter } = fixture();
  await expect(adapter.diarize(new Float32Array([NaN]), bundle)).rejects.toThrow("finite");
  await expect(
    adapter.diarize(new Float32Array(1), { ...bundle, sourceRevision: "unsupported" }),
  ).rejects.toThrow("Unsupported");
  expect(counts.models).toBe(0);
  expect(counts.processors).toBe(0);
});
