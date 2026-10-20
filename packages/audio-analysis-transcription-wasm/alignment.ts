import type * as Ort from "onnxruntime-web";
import type { BrowserTranscriptionResult } from "./index.js";

const RATE = 16_000;
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.all.bundle.min.mjs";
const ORT_ASSETS = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const MODELS = [
  {
    language: "en",
    modelId: "Xenova/wav2vec2-base-960h",
    revision: "a19f851b3d42865797e410752b4c570c871e4825",
    weights: "onnx/model.onnx",
    precision: "float32",
  },
  {
    language: "de",
    modelId: "Rafaelmstori/wav2vec2-base-german-cv9-onnx",
    revision: "23ac47b38595eaeb208415b66c0788bbaaa529c9",
    weights: "model_quantized.onnx",
    precision: "int8",
  },
] as const;
export type BrowserAlignmentModel = (typeof MODELS)[number];
type Vocabulary = { blankId: number; tokens: string[] };
type Word = {
  segmentIndex: number;
  wordIndex: number;
  text: string;
  startSeconds: number;
  endSeconds: number;
  confidence: number | null;
};
type EmissionRequest = {
  vocabulary: Vocabulary;
  emissions: number[][];
  text: string;
  segmentIndex: number;
  startSeconds: number;
  endSeconds: number;
};
export type BrowserAlignmentOptions = {
  language?: string;
  signal?: AbortSignal;
  onProgress?: (event: { stage: "model" | "align"; message: string }) => void;
};
type Model = {
  vocabulary: Vocabulary;
  emit(samples: Float32Array): Promise<number[][]>;
  dispose(): Promise<void>;
};
export type BrowserAlignmentDependencies = {
  loadModel: (model: BrowserAlignmentModel, options: BrowserAlignmentOptions) => Promise<Model>;
  alignEmissions: (request: EmissionRequest) => Promise<Word[]>;
};
type Bindings = {
  default(): Promise<unknown>;
  alignCtcEmissions(request: EmissionRequest): Word[];
};
let bindings: Promise<Bindings> | undefined;
let runtime: Promise<typeof Ort> | undefined;

export function browserAlignmentCapabilities() {
  return {
    runtime: "onnxruntime-web-wasm-ctc",
    models: MODELS.map((model) => ({ ...model })),
    languages: MODELS.map((model) => model.language),
    sampleRateHz: RATE,
    maxSegmentSeconds: 30,
    modelProvisioning: "browser-cache",
    fallbacks: { server: false, python: false },
  };
}
export async function supportsBrowserAlignment() {
  return typeof WebAssembly === "object" && typeof fetch === "function";
}
function checkCancelled(options: BrowserAlignmentOptions) {
  options.signal?.throwIfAborted();
}
function object(raw: unknown): Record<string, unknown> {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw))
    throw new Error("Invalid alignment model metadata.");
  return Object.fromEntries(Object.entries(raw));
}
function index(raw: unknown): number {
  if (typeof raw !== "number" || !Number.isSafeInteger(raw) || raw < 0)
    throw new Error("Invalid alignment vocabulary index.");
  return raw;
}
async function modelFile(
  model: BrowserAlignmentModel,
  file: string,
  options: BrowserAlignmentOptions,
): Promise<ArrayBuffer> {
  checkCancelled(options);
  const response = await fetch(
    `https://huggingface.co/${model.modelId}/resolve/${model.revision}/${file}`,
    {
      credentials: "omit",
      ...(options.signal ? { signal: options.signal } : {}),
    },
  );
  if (!response.ok)
    throw new Error(
      `Alignment model download failed (HTTP ${response.status}); check model access and connectivity.`,
    );
  return response.arrayBuffer();
}
async function loadRuntime(): Promise<typeof Ort> {
  // The remote module is the pinned SDK whose declarations type this boundary.
  runtime ??= import(/* @vite-ignore */ ORT_URL)
    .then((module) => {
      const ort: typeof Ort = module;
      ort.env.wasm.wasmPaths = ORT_ASSETS;
      ort.env.wasm.numThreads = 1;
      return ort;
    })
    .catch((error) => {
      runtime = undefined;
      throw error;
    });
  return runtime;
}
async function alignEmissions(request: EmissionRequest): Promise<Word[]> {
  bindings ??= (async () => {
    const url = new URL("./pkg/audio_analysis_transcription_wasm.js", import.meta.url).href;
    // This generated bridge belongs to this package and is verified by its WASM tests.
    const module: Bindings = await import(/* @vite-ignore */ url);
    await module.default();
    return module;
  })().catch((error) => {
    bindings = undefined;
    throw error;
  });
  return (await bindings).alignCtcEmissions(request);
}
async function loadModel(
  model: BrowserAlignmentModel,
  options: BrowserAlignmentOptions,
): Promise<Model> {
  const decode = (bytes: ArrayBuffer): unknown => JSON.parse(new TextDecoder().decode(bytes));
  const [configuration, preprocessing, dictionary, weights, ort] = await Promise.all([
    modelFile(model, "config.json", options),
    modelFile(model, "preprocessor_config.json", options),
    modelFile(model, "vocab.json", options),
    modelFile(model, model.weights, options),
    loadRuntime(),
  ]);
  checkCancelled(options);
  const config = object(decode(configuration));
  const preprocessor = object(decode(preprocessing));
  const vocab = object(decode(dictionary));
  if (preprocessor.sampling_rate !== RATE)
    throw new Error("Alignment model does not support 16 kHz audio.");
  const vocabulary: Vocabulary = {
    blankId: index(config.pad_token_id),
    tokens: Array(index(config.vocab_size)).fill(""),
  };
  if (vocabulary.blankId >= vocabulary.tokens.length)
    throw new Error("Alignment blank token is outside the model vocabulary.");
  for (const [token, value] of Object.entries(vocab)) {
    const id = index(value);
    if (id >= vocabulary.tokens.length)
      throw new Error("Alignment vocabulary exceeds the model output.");
    vocabulary.tokens[id] = token;
  }
  const session = await ort.InferenceSession.create(weights, { executionProviders: ["wasm"] });
  if (options.signal?.aborted) {
    await session.release();
    checkCancelled(options);
  }
  return {
    vocabulary,
    async emit(samples) {
      const normalized = samples.slice();
      if (preprocessor.do_normalize !== false) {
        const mean = normalized.reduce((sum, value) => sum + value, 0) / normalized.length;
        const variance =
          normalized.reduce((sum, value) => sum + (value - mean) ** 2, 0) / normalized.length;
        for (let i = 0; i < normalized.length; i++)
          normalized[i] = ((normalized[i] ?? 0) - mean) / Math.sqrt(variance + 1e-7);
      }
      const outputs = await session.run({
        input_values: new ort.Tensor("float32", normalized, [1, normalized.length]),
      });
      const logits = outputs.logits;
      if (
        !logits ||
        logits.type !== "float32" ||
        !(logits.data instanceof Float32Array) ||
        logits.dims.length !== 3 ||
        logits.dims[0] !== 1 ||
        logits.dims[2] !== vocabulary.tokens.length
      ) {
        throw new Error("Alignment model returned an unsupported emission shape.");
      }
      const frames = logits.dims[1] ?? 0;
      const width = vocabulary.tokens.length;
      if (frames < 1 || logits.data.length !== frames * width)
        throw new Error("Alignment model returned no complete emissions.");
      const emissions: number[][] = [];
      for (let t = 0; t < frames; t++) {
        const row = logits.data.subarray(t * width, (t + 1) * width);
        const maximum = Math.max(...row);
        const logsum =
          maximum +
          Math.log(
            Array.from(row, (value) => Math.exp(value - maximum)).reduce((a, b) => a + b, 0),
          );
        emissions.push(Array.from(row, (value) => Math.min(0, value - logsum)));
      }
      return emissions;
    },
    dispose: () => session.release(),
  };
}

/** One active alignment operation; one idle model is retained between calls. */
export function createBrowserAlignmentAdapter(
  dependencies: Partial<BrowserAlignmentDependencies> = {},
) {
  const load = dependencies.loadModel ?? loadModel;
  const align = dependencies.alignEmissions ?? alignEmissions;
  let resident: { id: string; model: Model } | undefined;
  let active = false;
  return {
    async align(
      samples: Float32Array,
      source: BrowserTranscriptionResult,
      options: BrowserAlignmentOptions = {},
    ): Promise<BrowserTranscriptionResult> {
      if (active) throw new Error("Browser alignment is already running.");
      const language = options.language?.trim().toLowerCase() || source.language;
      const profile = MODELS.find((model) => model.language === language);
      if (!profile) throw new Error("Choose a supported alignment language: English or German.");
      if (
        !(samples instanceof Float32Array) ||
        !samples.length ||
        samples.some((value) => !Number.isFinite(value))
      )
        throw new Error("Alignment requires finite 16 kHz mono PCM.");
      const duration = samples.length / RATE;
      const ranges = source.segments.map((segment) => {
        const start = segment.startSeconds ?? 0;
        const end = segment.endSeconds ?? duration;
        if (
          !Number.isFinite(start) ||
          !Number.isFinite(end) ||
          start < 0 ||
          end <= start ||
          end > duration + 1e-6 ||
          end - start > 30
        )
          throw new Error(
            "Alignment requires ASR segment ranges of at most 30 seconds within the recording.",
          );
        return { start, end };
      });
      checkCancelled(options);
      active = true;
      try {
        if (!source.segments.some((segment) => segment.text.trim())) return structuredClone(source);
        if (resident?.id !== profile.modelId) {
          if (resident) {
            await resident.model.dispose();
            resident = undefined;
          }
          options.onProgress?.({
            stage: "model",
            message: `Loading ${profile.language} CTC alignment model…`,
          });
          resident = { id: profile.modelId, model: await load(profile, options) };
        }
        checkCancelled(options);
        const model = resident.model;
        const segments = [];
        for (const [i, segment] of source.segments.entries()) {
          if (!segment.text.trim()) {
            segments.push({ ...segment });
            continue;
          }
          const range = ranges[i];
          if (!range) throw new Error("Alignment segment range is missing.");
          options.onProgress?.({
            stage: "align",
            message: `Aligning segment ${i + 1} of ${source.segments.length}…`,
          });
          const pcm = samples.subarray(
            Math.floor(range.start * RATE),
            Math.min(samples.length, Math.ceil(range.end * RATE)),
          );
          const emissions = await model.emit(pcm);
          checkCancelled(options);
          const words = await align({
            vocabulary: model.vocabulary,
            emissions,
            text: segment.text,
            segmentIndex: segment.index,
            startSeconds: range.start,
            endSeconds: range.end,
          });
          checkCancelled(options);
          if (!words.length || words.some((word) => word.segmentIndex !== segment.index))
            throw new Error("Alignment did not return words for the source segment.");
          segments.push({
            ...segment,
            startSeconds: words[0]?.startSeconds ?? segment.startSeconds,
            endSeconds: words.at(-1)?.endSeconds ?? segment.endSeconds,
            words: words.map((word) => ({
              index: word.wordIndex,
              text: word.text,
              startSeconds: word.startSeconds,
              endSeconds: word.endSeconds,
              confidence: word.confidence,
              attributes: {},
            })),
            attributes: { ...segment.attributes, alignment: "ctc-forced-alignment" },
          });
        }
        return {
          ...source,
          segments,
          attributes: {
            ...source.attributes,
            alignment: "ctc-forced-alignment",
            alignmentModel: profile.modelId,
            alignmentModelRevision: profile.revision,
            alignmentRuntime: "onnxruntime-web-wasm",
          },
        };
      } finally {
        active = false;
      }
    },
    async dispose() {
      if (active) throw new Error("Wait for active alignment before releasing its model.");
      if (resident) {
        await resident.model.dispose();
        resident = undefined;
      }
    },
  };
}
const defaultAdapter = createBrowserAlignmentAdapter();
export const alignTranscriptAudioSamples = defaultAdapter.align;
