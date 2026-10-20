import type * as Ort from "onnxruntime-web";
import { COMMUNITY_PROFILE as PROFILE } from "./community-profile.js";
import type { CommunityBundle } from "./resources.js";

type Span = { speaker: string; startSeconds: number; endSeconds: number; score: number | null };
type ProviderResult = { modelId: string; segments: Span[]; diagnostics: string[] };
type Mask = { localSpeaker: number; cleanFrames: number; mask: number[] };
type Processor = {
  pushSegmentation(scores: Float32Array): Mask[];
  pushEmbedding(chunk: number, speaker: number, values: Float32Array): void;
  finish(): ProviderResult;
  free(): void;
};
type Bindings = {
  default(): Promise<unknown>;
  CommunityDiarization: new (
    transform: Uint8Array,
    model: Uint8Array,
    clustering: Uint8Array,
    sampleCount: number,
  ) => Processor;
};
type Inference = {
  segment(samples: Float32Array): Promise<Float32Array>;
  embed(samples: Float32Array, mask: Float32Array): Promise<Float32Array>;
  dispose(): Promise<void>;
};
export type CommunityDiarizationOptions = {
  signal?: AbortSignal;
  onProgress?: (event: {
    stage: "diarize";
    message: string;
    completedWindows: number;
    totalWindows: number;
  }) => void;
};
export type CommunityDiarizationDependencies = {
  loadInference: (bundle: CommunityBundle) => Promise<Inference>;
  createProcessor: (bundle: CommunityBundle, sampleCount: number) => Promise<Processor>;
};
export type BrowserCommunityDiarizationResult = {
  accepted: true;
  operation: "diarize";
  modelId: string;
  runtime: string;
  speakerCount: number;
  segments: Span[];
  diagnostics: string[];
  attributes: {
    quality: "pyannote-community-1";
    countStatus: "tentativeAutomatic";
    countUncertainty: "notReported";
    sourceRevision: string;
    artifactSetSha256: string;
  };
};
const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
let runtime: Promise<typeof Ort> | undefined;
let bindings: Promise<Bindings> | undefined;

export function browserCommunityDiarizationCapabilities() {
  return {
    runtime: "onnxruntime-web-wasm-pyannote-community-1",
    modelId: PROFILE.manifest.modelId,
    sourceRevision: PROFILE.manifest.source.revision,
    artifactSetSha256: PROFILE.manifest.artifactSetSha256,
    sampleRateHz: 16000,
    maxDurationSeconds: 3600,
    quality: "pyannote-community-1",
    countStatus: "tentativeAutomatic",
    countUncertainty: "notReported",
    modelProvisioning: "licensed-browser-download",
    fallbacks: { server: false, python: false },
  } as const;
}
export async function supportsBrowserCommunityDiarization() {
  return (
    typeof WebAssembly === "object" &&
    typeof fetch === "function" &&
    typeof crypto?.subtle === "object"
  );
}
async function ortRuntime(): Promise<typeof Ort> {
  runtime ??= (async () => {
    const module: typeof Ort = await import(/* @vite-ignore */ `${ORT_BASE}ort.all.bundle.min.mjs`);
    module.env.wasm.wasmPaths = ORT_BASE;
    module.env.wasm.numThreads = 1;
    return module;
  })();
  return runtime;
}
async function wasmBindings(): Promise<Bindings> {
  bindings ??= (async () => {
    const module: Bindings = await import(
      /* @vite-ignore */ new URL("./pkg/audio_analysis_speakers_wasm.js", import.meta.url).href
    );
    await module.default();
    return module;
  })();
  return bindings;
}
function file(bundle: CommunityBundle, name: string) {
  const bytes = bundle.files[name];
  if (!(bytes instanceof Uint8Array)) {
    throw new Error("Required pyannote model resource is missing.");
  }
  return bytes;
}
async function createProcessor(bundle: CommunityBundle, sampleCount: number): Promise<Processor> {
  const wasm = await wasmBindings();
  return new wasm.CommunityDiarization(
    file(bundle, "plda_transform.json"),
    file(bundle, "plda_model.json"),
    file(bundle, "clustering.json"),
    sampleCount,
  );
}
async function modelBytes(bundle: CommunityBundle, name: "segmentation.onnx" | "embedding.onnx") {
  const bytes = Uint8Array.from(file(bundle, name));
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (v) =>
    v.toString(16).padStart(2, "0"),
  ).join("");
  if (hash !== PROFILE.manifest.files[name]) {
    throw new Error("Pyannote model checksum mismatch.");
  }
  return bytes;
}
function output(outputs: Record<string, Ort.Tensor>, name: string, shape: readonly number[]) {
  const tensor = outputs[name];
  if (
    !tensor ||
    tensor.type !== "float32" ||
    !(tensor.data instanceof Float32Array) ||
    tensor.dims.length !== shape.length ||
    tensor.dims.some((value, index) => value !== shape[index]) ||
    tensor.data.some((value) => !Number.isFinite(value))
  ) {
    throw new Error("Invalid pyannote ONNX model output.");
  }
  return Float32Array.from(tensor.data);
}
async function loadInference(bundle: CommunityBundle): Promise<Inference> {
  const ort = await ortRuntime();
  const segmentation = await ort.InferenceSession.create(
    await modelBytes(bundle, "segmentation.onnx"),
    { executionProviders: ["wasm"] },
  );
  let embedding: Ort.InferenceSession;
  try {
    embedding = await ort.InferenceSession.create(await modelBytes(bundle, "embedding.onnx"), {
      executionProviders: ["wasm"],
    });
  } catch (error) {
    await segmentation.release();
    throw error;
  }
  return {
    async segment(samples) {
      return output(
        await segmentation.run({ waveform: new ort.Tensor("float32", samples, [1, 1, 160000]) }),
        "segmentations",
        [1, 589, 3],
      );
    },
    async embed(samples, mask) {
      return output(
        await embedding.run({
          waveform: new ort.Tensor("float32", samples, [1, 1, 160000]),
          masks: new ort.Tensor("float32", mask, [1, 589]),
        }),
        "embeddings",
        [1, 256],
      );
    },
    async dispose() {
      await Promise.all([segmentation.release(), embedding.release()]);
    },
  };
}
function cancelled(options: CommunityDiarizationOptions) {
  options.signal?.throwIfAborted();
}

export function createBrowserCommunityDiarizationAdapter(
  dependencies: Partial<CommunityDiarizationDependencies> = {},
) {
  let resident: Inference | undefined;
  let active = false;
  return {
    async diarize(
      samples: Float32Array,
      bundle: CommunityBundle,
      options: CommunityDiarizationOptions = {},
    ): Promise<BrowserCommunityDiarizationResult> {
      if (active) {
        throw new Error("Browser pyannote diarization is already running.");
      }
      if (
        !(samples instanceof Float32Array) ||
        !samples.length ||
        samples.length > 16000 * 3600 ||
        samples.some((value) => !Number.isFinite(value))
      ) {
        throw new Error("Pyannote requires finite mono 16 kHz PCM, at most one hour.");
      }
      if (
        bundle.modelId !== PROFILE.manifest.modelId ||
        bundle.sourceRevision !== PROFILE.manifest.source.revision ||
        bundle.artifactSetSha256 !== PROFILE.manifest.artifactSetSha256
      ) {
        throw new Error("Unsupported pyannote model bundle.");
      }
      cancelled(options);
      active = true;
      let processor: Processor | undefined;
      try {
        processor = await (dependencies.createProcessor ?? createProcessor)(bundle, samples.length);
        cancelled(options);
        resident ??= await (dependencies.loadInference ?? loadInference)(bundle);
        cancelled(options);
        const totalWindows = 1 + Math.ceil(Math.max(0, samples.length - 160000) / 16000);
        for (let chunk = 0; chunk < totalWindows; chunk++) {
          cancelled(options);
          const window = new Float32Array(160000);
          window.set(samples.subarray(chunk * 16000, chunk * 16000 + 160000));
          const scores = await resident.segment(window);
          cancelled(options);
          const requests = processor.pushSegmentation(scores);
          for (const request of requests) {
            const values = await resident.embed(window, Float32Array.from(request.mask));
            cancelled(options);
            processor.pushEmbedding(chunk, request.localSpeaker, values);
          }
          options.onProgress?.({
            stage: "diarize",
            message: `Diarizing speakers locally · ${chunk + 1}/${totalWindows} windows`,
            completedWindows: chunk + 1,
            totalWindows,
          });
          // CPU WASM can resolve through microtasks continuously. Give input and
          // rendering a task boundary so progress and cancellation stay usable.
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
        }
        cancelled(options);
        const result = processor.finish();
        return {
          accepted: true,
          operation: "diarize",
          modelId: result.modelId,
          runtime: browserCommunityDiarizationCapabilities().runtime,
          speakerCount: new Set(result.segments.map((segment) => segment.speaker)).size,
          segments: result.segments,
          diagnostics: result.diagnostics,
          attributes: {
            quality: "pyannote-community-1",
            countStatus: "tentativeAutomatic",
            countUncertainty: "notReported",
            sourceRevision: bundle.sourceRevision,
            artifactSetSha256: bundle.artifactSetSha256,
          },
        };
      } finally {
        processor?.free();
        active = false;
      }
    },
    async dispose() {
      if (active) {
        throw new Error("Wait for active diarization before releasing models.");
      }
      if (resident) {
        await resident.dispose();
        resident = undefined;
      }
    },
  };
}
const defaultAdapter = createBrowserCommunityDiarizationAdapter();
export const diarizeCommunityAudioSamples = defaultAdapter.diarize;
