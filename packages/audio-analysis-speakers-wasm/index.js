const TRANSFORMERS_MODULE_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const BROWSER_SAMPLE_RATE_HZ = 16_000;
const BROWSER_WINDOW_SECONDS = 10;
const BROWSER_WINDOW_SAMPLES = BROWSER_SAMPLE_RATE_HZ * BROWSER_WINDOW_SECONDS;
const MAX_EMBEDDING_SECONDS = 6;
const MIN_EMBEDDING_SECONDS = 0.5;
const DEFAULT_CLUSTER_THRESHOLD = 0.82;
const SEGMENTATION_MODEL_ID = "onnx-community/pyannote-segmentation-3.0";
const SPEAKER_EMBEDDING_MODEL_ID = "Xenova/wavlm-base-plus-sv";
const BROWSER_DIARIZATION_RUNTIME_ID =
  "audio-analysis-transformers-js-browser-diarization";

let wasmModulePromise;
let transformersModulePromise;
let browserDiarizationRuntimePromise;
const browserModelProgressListeners = new Set();

export async function init() {
  const wasmEntry = "./pkg/audio_analysis_speakers_wasm.js";
  wasmModulePromise ??= import(/* @vite-ignore */ wasmEntry).then(async (module) => {
    if (typeof module.default === "function") {
      await module.default();
    }
    return module;
  });
  return wasmModulePromise;
}

export async function packageSurface() {
  const module = await init();
  return module.packageSurface();
}

export async function runOperation(request) {
  const module = await init();
  return module.runOperation(request);
}

export function browserDiarizationCapabilities() {
  return {
    runtime: BROWSER_DIARIZATION_RUNTIME_ID,
    modelProvisioning: "browser-cache",
    segmentationModelId: SEGMENTATION_MODEL_ID,
    speakerEmbeddingModelId: SPEAKER_EMBEDDING_MODEL_ID,
    input: {
      sampleRateHz: BROWSER_SAMPLE_RATE_HZ,
      channels: 1,
      sampleFormat: "f32",
    },
    chunking: {
      windowSeconds: BROWSER_WINDOW_SECONDS,
    },
    clustering: {
      metric: "cosine",
      defaultThreshold: DEFAULT_CLUSTER_THRESHOLD,
    },
    backends: {
      segmentation: "wasm-q8",
      speakerEmbedding: "wasm-q8",
    },
    features: {
      diarization: true,
      overlapAwareSegmentation: true,
      globalSpeakerClustering: true,
      transcriptAssignment: true,
    },
    fallbacks: {
      server: false,
      python: false,
    },
  };
}

export async function supportsBrowserDiarization() {
  return (
    typeof WebAssembly === "object" &&
    typeof Float32Array === "function" &&
    typeof navigator !== "undefined"
  );
}

export async function diarizeBrowserAudioSamples(samples, options = {}) {
  validateBrowserPcm(samples, options);
  if (!(await supportsBrowserDiarization())) {
    throw new Error(
      "Browser speaker diarization requires WebAssembly. No server or Python fallback is used.",
    );
  }

  const threshold = clusterThreshold(options.clusterThreshold);
  const runtime = await acquireBrowserDiarizationRuntime(options);
  const clusters = [];
  const segments = [];
  const windowCount = Math.ceil(samples.length / BROWSER_WINDOW_SAMPLES);

  for (let windowIndex = 0; windowIndex < windowCount; windowIndex += 1) {
    const windowStartSample = windowIndex * BROWSER_WINDOW_SAMPLES;
    const windowEndSample = Math.min(
      samples.length,
      windowStartSample + BROWSER_WINDOW_SAMPLES,
    );
    const windowSamples = samples.subarray(windowStartSample, windowEndSample);
    const windowOffsetSeconds = windowStartSample / BROWSER_SAMPLE_RATE_HZ;
    const localTurns = await segmentBrowserWindow(
      runtime,
      windowSamples,
      windowIndex,
      windowCount,
      options,
    );

    const localAssignments = await assignWindowSpeakers(
      runtime,
      windowSamples,
      localTurns,
      clusters,
      threshold,
      windowIndex,
      windowCount,
      options,
    );

    for (const turn of localTurns) {
      const speaker = localAssignments.get(turn.id);
      if (!speaker) {
        continue;
      }
      segments.push({
        speaker,
        startSeconds: windowOffsetSeconds + turn.start,
        endSeconds: windowOffsetSeconds + turn.end,
        score: turn.confidence,
      });
    }
  }

  const merged = mergeAdjacentSpeakerSegments(segments);
  emitProgress(options, {
    stage: "cluster",
    message: `Speaker diarization finished with ${clusters.length} speaker${clusters.length === 1 ? "" : "s"}.`,
    detail: { speakerCount: clusters.length, segmentCount: merged.length },
  });

  return {
    accepted: true,
    operation: "diarize",
    modelId: `${SEGMENTATION_MODEL_ID}+${SPEAKER_EMBEDDING_MODEL_ID}`,
    runtime: BROWSER_DIARIZATION_RUNTIME_ID,
    speakerCount: clusters.length,
    segments: merged,
    diagnostics: [
      `segmentation=${SEGMENTATION_MODEL_ID}`,
      `speakerEmbedding=${SPEAKER_EMBEDDING_MODEL_ID}`,
      `windowSeconds=${BROWSER_WINDOW_SECONDS}`,
      `clusterThreshold=${threshold}`,
      "serverFallback=false",
      "pythonFallback=false",
    ],
  };
}

export function assignBrowserDiarizationToTranscript(transcript, diarization) {
  if (!transcript || !Array.isArray(transcript.segments)) {
    throw new TypeError("Browser diarization assignment requires transcript segments.");
  }
  if (!diarization || !Array.isArray(diarization.segments)) {
    throw new TypeError("Browser diarization assignment requires diarization segments.");
  }

  return {
    ...transcript,
    segments: transcript.segments.map((segment) => {
      const start = finiteNumberOrNull(segment?.startSeconds);
      const end = finiteNumberOrNull(segment?.endSeconds);
      if (start === null || end === null || end < start) {
        return { ...segment };
      }

      let best = null;
      let bestOverlap = 0;
      for (const speakerSegment of diarization.segments) {
        const speakerStart = finiteNumberOrNull(speakerSegment?.startSeconds);
        const speakerEnd = finiteNumberOrNull(speakerSegment?.endSeconds);
        if (speakerStart === null || speakerEnd === null || speakerEnd < speakerStart) {
          continue;
        }
        const overlap = Math.max(
          0,
          Math.min(end, speakerEnd) - Math.max(start, speakerStart),
        );
        if (
          overlap > bestOverlap ||
          (overlap === bestOverlap &&
            overlap > 0 &&
            compareSpeakerSegments(speakerSegment, best) < 0)
        ) {
          best = speakerSegment;
          bestOverlap = overlap;
        }
      }

      return bestOverlap > 0 && typeof best?.speaker === "string"
        ? { ...segment, speaker: best.speaker }
        : { ...segment };
    }),
  };
}

async function acquireBrowserDiarizationRuntime(options) {
  const listener =
    typeof options.onProgress === "function" ? options.onProgress : null;
  if (listener) {
    browserModelProgressListeners.add(listener);
  }

  try {
    browserDiarizationRuntimePromise ??= loadBrowserDiarizationRuntime();
    return await browserDiarizationRuntimePromise;
  } catch (error) {
    browserDiarizationRuntimePromise = null;
    throw error;
  } finally {
    if (listener) {
      browserModelProgressListeners.delete(listener);
    }
  }
}

async function loadBrowserDiarizationRuntime() {
  const {
    AutoModel,
    AutoModelForAudioFrameClassification,
    AutoProcessor,
  } = await loadTransformers();

  reportModelProgress({
    stage: "model",
    message: "Loading browser speaker-segmentation processor…",
  });
  const segmentationProcessor = await AutoProcessor.from_pretrained(
    SEGMENTATION_MODEL_ID,
    {
      progress_callback: (detail) =>
        reportModelProgress(modelProgress("speaker segmentation", detail)),
    },
  );
  const segmentationModel =
    await AutoModelForAudioFrameClassification.from_pretrained(
      SEGMENTATION_MODEL_ID,
      {
        device: "wasm",
        dtype: "q8",
        progress_callback: (detail) =>
          reportModelProgress(modelProgress("speaker segmentation", detail)),
      },
    );

  reportModelProgress({
    stage: "model",
    message: "Loading browser speaker-embedding processor…",
  });
  const embeddingProcessor = await AutoProcessor.from_pretrained(
    SPEAKER_EMBEDDING_MODEL_ID,
    {
      progress_callback: (detail) =>
        reportModelProgress(modelProgress("speaker embedding", detail)),
    },
  );
  const embeddingModel = await AutoModel.from_pretrained(
    SPEAKER_EMBEDDING_MODEL_ID,
    {
      device: "wasm",
      dtype: "q8",
      progress_callback: (detail) =>
        reportModelProgress(modelProgress("speaker embedding", detail)),
    },
  );

  return {
    segmentationProcessor,
    segmentationModel,
    embeddingProcessor,
    embeddingModel,
  };
}

async function segmentBrowserWindow(
  runtime,
  samples,
  windowIndex,
  windowCount,
  options,
) {
  emitProgress(options, {
    stage: "segment",
    message: `Detecting speaker turns in window ${windowIndex + 1}/${windowCount}…`,
    detail: { windowIndex, windowCount },
  });

  const padded = new Float32Array(BROWSER_WINDOW_SAMPLES);
  padded.set(samples);
  const inputs = await runtime.segmentationProcessor(padded);
  const { logits } = await runtime.segmentationModel(inputs);
  const raw = runtime.segmentationProcessor.post_process_speaker_diarization(
    logits,
    padded.length,
  );
  const turns = Array.isArray(raw?.[0]) ? raw[0] : raw;
  return normalizeLocalTurns(
    Array.isArray(turns) ? turns : [],
    samples.length / BROWSER_SAMPLE_RATE_HZ,
  );
}

async function assignWindowSpeakers(
  runtime,
  samples,
  turns,
  clusters,
  threshold,
  windowIndex,
  windowCount,
  options,
) {
  const grouped = groupTurnsByLocalSpeaker(turns);
  const assignments = new Map();
  const unavailableClusters = new Set();

  for (const [localId, localTurns] of grouped) {
    emitProgress(options, {
      stage: "embed",
      message: `Matching speakers in window ${windowIndex + 1}/${windowCount}…`,
      detail: {
        windowIndex,
        windowCount,
        localSpeaker: localId,
      },
    });

    const speakerSamples = collectSpeakerSamples(samples, localTurns);
    const embedding = await extractSpeakerEmbedding(runtime, speakerSamples);
    const clusterIndex = bestClusterIndex(
      embedding,
      clusters,
      unavailableClusters,
      threshold,
    );
    const resolvedIndex =
      clusterIndex === -1
        ? createSpeakerCluster(clusters, embedding, speakerSamples.length)
        : updateSpeakerCluster(
            clusters,
            clusterIndex,
            embedding,
            speakerSamples.length,
          );

    unavailableClusters.add(resolvedIndex);
    assignments.set(localId, speakerLabel(resolvedIndex));
  }

  return assignments;
}

async function extractSpeakerEmbedding(runtime, samples) {
  const inputs = await runtime.embeddingProcessor(samples);
  const output = await runtime.embeddingModel(inputs);
  const tensor = output?.embeddings ?? output?.logits;
  if (!tensor?.data || tensor.data.length === 0) {
    throw new Error("Browser speaker embedding model returned no embedding.");
  }
  return normalizeVector(Float32Array.from(tensor.data));
}

function groupTurnsByLocalSpeaker(turns) {
  const grouped = new Map();
  for (const turn of turns) {
    const existing = grouped.get(turn.id);
    if (existing) {
      existing.push(turn);
    } else {
      grouped.set(turn.id, [turn]);
    }
  }
  return [...grouped.entries()].sort((left, right) => {
    const leftStart = left[1][0]?.start ?? 0;
    const rightStart = right[1][0]?.start ?? 0;
    return leftStart - rightStart || String(left[0]).localeCompare(String(right[0]));
  });
}

function collectSpeakerSamples(samples, turns) {
  const maximum = Math.round(MAX_EMBEDDING_SECONDS * BROWSER_SAMPLE_RATE_HZ);
  const minimum = Math.round(MIN_EMBEDDING_SECONDS * BROWSER_SAMPLE_RATE_HZ);
  const chunks = [];
  let total = 0;

  for (const turn of turns) {
    if (total >= maximum) {
      break;
    }
    const start = clamp(
      Math.floor(turn.start * BROWSER_SAMPLE_RATE_HZ),
      0,
      samples.length,
    );
    const end = clamp(
      Math.ceil(turn.end * BROWSER_SAMPLE_RATE_HZ),
      start,
      samples.length,
    );
    const count = Math.min(end - start, maximum - total);
    if (count > 0) {
      chunks.push(samples.subarray(start, start + count));
      total += count;
    }
  }

  const output = new Float32Array(Math.max(total, minimum));
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

function bestClusterIndex(embedding, clusters, unavailable, threshold) {
  let bestIndex = -1;
  let bestSimilarity = threshold;
  for (let index = 0; index < clusters.length; index += 1) {
    if (unavailable.has(index)) {
      continue;
    }
    const similarity = cosineSimilarity(embedding, clusters[index].centroid);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function createSpeakerCluster(clusters, embedding, weight) {
  clusters.push({
    centroid: embedding,
    weight: Math.max(1, weight),
  });
  return clusters.length - 1;
}

function updateSpeakerCluster(clusters, index, embedding, weight) {
  const cluster = clusters[index];
  const nextWeight = Math.max(1, weight);
  const totalWeight = cluster.weight + nextWeight;
  const centroid = new Float32Array(cluster.centroid.length);
  for (let dimension = 0; dimension < centroid.length; dimension += 1) {
    centroid[dimension] =
      (cluster.centroid[dimension] * cluster.weight +
        embedding[dimension] * nextWeight) /
      totalWeight;
  }
  cluster.centroid = normalizeVector(centroid);
  cluster.weight = totalWeight;
  return index;
}

function normalizeLocalTurns(turns, durationSeconds) {
  return turns
    .map((turn) => {
      const start = finiteNumberOrNull(turn?.start);
      const end = finiteNumberOrNull(turn?.end);
      if (start === null || end === null) {
        return null;
      }
      const clippedStart = clamp(start, 0, durationSeconds);
      const clippedEnd = clamp(end, clippedStart, durationSeconds);
      if (clippedEnd <= clippedStart) {
        return null;
      }
      return {
        id: String(turn.id ?? "0"),
        start: clippedStart,
        end: clippedEnd,
        confidence: finiteNumberOrNull(turn.confidence),
      };
    })
    .filter(Boolean);
}

function mergeAdjacentSpeakerSegments(segments) {
  const merged = [];
  for (const segment of segments) {
    const previous = merged[merged.length - 1];
    if (
      previous &&
      previous.speaker === segment.speaker &&
      segment.startSeconds - previous.endSeconds <= 0.08
    ) {
      previous.endSeconds = Math.max(previous.endSeconds, segment.endSeconds);
      previous.score = averageOptional(previous.score, segment.score);
    } else {
      merged.push({ ...segment });
    }
  }
  return merged;
}

function compareSpeakerSegments(left, right) {
  if (!right) {
    return -1;
  }
  return (
    (left.startSeconds ?? 0) - (right.startSeconds ?? 0) ||
    String(left.speaker ?? "").localeCompare(String(right.speaker ?? ""))
  );
}

function speakerLabel(index) {
  return `speaker_${String(index).padStart(2, "0")}`;
}

function cosineSimilarity(left, right) {
  if (left.length !== right.length) {
    throw new Error("Speaker embeddings must have the same dimensions.");
  }
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }
  const denominator = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
  return denominator > 0 ? dot / denominator : 0;
}

function normalizeVector(values) {
  let normSquared = 0;
  for (const value of values) {
    normSquared += value * value;
  }
  const norm = Math.sqrt(normSquared);
  if (!(norm > 0)) {
    throw new Error("Browser speaker embedding must contain non-zero finite values.");
  }
  const normalized = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index] / norm;
    if (!Number.isFinite(value)) {
      throw new Error("Browser speaker embedding must contain finite values.");
    }
    normalized[index] = value;
  }
  return normalized;
}

function validateBrowserPcm(samples, options) {
  if (!(samples instanceof Float32Array) || samples.length === 0) {
    throw new TypeError(
      "audio-analysis browser diarization requires a non-empty Float32Array.",
    );
  }
  const sampleRateHz = options.sampleRateHz ?? BROWSER_SAMPLE_RATE_HZ;
  if (sampleRateHz !== BROWSER_SAMPLE_RATE_HZ) {
    throw new RangeError(
      `Browser diarization requires ${BROWSER_SAMPLE_RATE_HZ} Hz mono PCM.`,
    );
  }
  for (const sample of samples) {
    if (!Number.isFinite(sample)) {
      throw new TypeError("Browser diarization PCM samples must be finite.");
    }
  }
}

function clusterThreshold(value) {
  if (value === undefined) {
    return DEFAULT_CLUSTER_THRESHOLD;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value >= 1) {
    throw new RangeError("clusterThreshold must be a finite number between 0 and 1.");
  }
  return value;
}

function averageOptional(left, right) {
  if (Number.isFinite(left) && Number.isFinite(right)) {
    return (left + right) / 2;
  }
  return Number.isFinite(left) ? left : Number.isFinite(right) ? right : null;
}

function finiteNumberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function emitProgress(options, update) {
  if (typeof options.onProgress === "function") {
    options.onProgress(update);
  }
}

function reportModelProgress(update) {
  for (const listener of browserModelProgressListeners) {
    listener(update);
  }
}

function modelProgress(label, detail) {
  if (detail && typeof detail === "object" && detail.status === "progress") {
    return {
      stage: "model",
      message: `Downloading/caching browser ${label} assets…`,
      detail,
    };
  }
  if (detail && typeof detail === "object" && detail.status === "done") {
    return {
      stage: "model",
      message: `Browser ${label} assets ready…`,
      detail,
    };
  }
  return {
    stage: "model",
    message: `Preparing browser ${label} model…`,
    detail,
  };
}

async function loadTransformers() {
  if (!transformersModulePromise) {
    transformersModulePromise = import(
      /* @vite-ignore */ TRANSFORMERS_MODULE_URL
    ).then((module) => {
      module.env.allowLocalModels = false;
      module.env.useBrowserCache = true;
      module.env.useWasmCache = true;
      return module;
    });
  }
  return transformersModulePromise;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
