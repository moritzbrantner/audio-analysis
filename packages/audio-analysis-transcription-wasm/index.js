const DEFAULT_BROWSER_MODEL_ID = "onnx-community/whisper-tiny";
const TRANSFORMERS_MODULE_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const BROWSER_SAMPLE_RATE_HZ = 16_000;
const BROWSER_RUNTIME_ID = "audio-analysis-transformers-js-webgpu";
const DEFAULT_WINDOW_SECONDS = 29;
const DEFAULT_STRIDE_SECONDS = 5;
const DEFAULT_MAX_BUFFERED_SECONDS = 58;
const TIME_EPSILON_SECONDS = 1e-6;

let wasmModulePromise;
let transformersModulePromise;
let transcriberPromise;
let activeModelProgress = null;

export async function init() {
  const wasmEntry = "./pkg/audio_analysis_transcription_wasm.js";
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

export function browserTranscriptionCapabilities() {
  return {
    runtime: BROWSER_RUNTIME_ID,
    requiredAcceleration: "webgpu",
    modelId: DEFAULT_BROWSER_MODEL_ID,
    modelProvisioning: "browser-cache",
    input: {
      sampleRateHz: BROWSER_SAMPLE_RATE_HZ,
      channels: 1,
      sampleFormat: "f32",
      acceptedSources: ["Blob", "Float32Array", "bounded Float32Array stream"],
    },
    features: {
      transcription: true,
      timedSegments: true,
      boundedPcmStreaming: true,
      alignment: false,
      diarization: false,
      translation: false,
    },
    streaming: browserTranscriptionWindowPlan(),
    fallbacks: {
      server: false,
      python: false,
      cpu: false,
    },
  };
}

export function browserTranscriptionWindowPlan(options = {}) {
  const windowSeconds = positiveFiniteOrDefault(options.windowSeconds, DEFAULT_WINDOW_SECONDS);
  const strideSeconds = nonNegativeFiniteOrDefault(options.strideSeconds, DEFAULT_STRIDE_SECONDS);
  const maxBufferedSeconds = positiveFiniteOrDefault(
    options.maxBufferedSeconds,
    DEFAULT_MAX_BUFFERED_SECONDS,
  );

  if (windowSeconds > DEFAULT_WINDOW_SECONDS) {
    throw new RangeError(
      `Browser transcription windows are bounded to ${DEFAULT_WINDOW_SECONDS} seconds; got ${windowSeconds}.`,
    );
  }
  if (strideSeconds >= windowSeconds) {
    throw new RangeError("Browser transcription stride must be smaller than the window.");
  }
  if (maxBufferedSeconds < windowSeconds) {
    throw new RangeError("Browser transcription maxBufferedSeconds must cover at least one window.");
  }

  const windowSamples = secondsToSamples(windowSeconds, 1);
  const strideSamples = secondsToSamples(strideSeconds, 0);
  const stepSamples = windowSamples - strideSamples;
  const maxBufferedSamples = secondsToSamples(maxBufferedSeconds, 1);
  if (stepSamples <= 0) {
    throw new RangeError("Browser transcription window rounding must leave a positive step.");
  }
  if (maxBufferedSamples < windowSamples) {
    throw new RangeError("Browser transcription buffer rounding must cover at least one window.");
  }

  return {
    sampleRateHz: BROWSER_SAMPLE_RATE_HZ,
    windowSeconds,
    strideSeconds,
    stepSeconds: stepSamples / BROWSER_SAMPLE_RATE_HZ,
    maxBufferedSeconds,
    windowSamples,
    strideSamples,
    stepSamples,
    maxBufferedSamples,
  };
}

export function stitchBrowserTranscriptionWindow(segments, options = {}) {
  if (!Array.isArray(segments)) {
    throw new TypeError("Browser transcription stitching requires an array of segments.");
  }

  const committedThroughSeconds = nonNegativeFiniteOrDefault(
    options.committedThroughSeconds,
    0,
  );
  const commitUntilSeconds = nonNegativeFiniteOrInfinity(options.commitUntilSeconds, Infinity);
  const final = options.final === true;
  const startIndex = nonNegativeIntegerOrDefault(options.startIndex, 0);
  const accepted = [];

  for (const segment of segments) {
    if (!segment || typeof segment !== "object") continue;
    const text = String(segment.text ?? "").trim();
    if (!text) continue;

    const startSeconds = finiteOrNull(segment.startSeconds);
    const endSeconds = finiteOrNull(segment.endSeconds);
    if (!final && (endSeconds === null || endSeconds > commitUntilSeconds + TIME_EPSILON_SECONDS)) {
      continue;
    }
    if (endSeconds !== null && endSeconds <= committedThroughSeconds + TIME_EPSILON_SECONDS) {
      continue;
    }

    accepted.push({
      ...segment,
      index: startIndex + accepted.length,
      text,
      startSeconds,
      endSeconds,
    });
  }

  const acceptedEnd = accepted.reduce(
    (latest, segment) => Math.max(latest, segment.endSeconds ?? latest),
    committedThroughSeconds,
  );
  const nextCommittedThroughSeconds = final
    ? acceptedEnd
    : Math.max(committedThroughSeconds, commitUntilSeconds);

  return {
    segments: accepted,
    committedThroughSeconds: nextCommittedThroughSeconds,
  };
}

export async function supportsBrowserTranscription() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return false;
  }
  const adapter = await navigator.gpu.requestAdapter();
  return adapter !== null;
}

export async function transcribeAudioBlob(source, options = {}) {
  if (!source || typeof source.arrayBuffer !== "function") {
    throw new TypeError("audio-analysis browser transcription requires a Blob-like audio source.");
  }
  emitProgress(options, {
    stage: "decode",
    message: "Decoding and resampling audio to 16 kHz mono…",
  });
  const audio = await decodeAndResample(source);
  return transcribeAudioSamples(audio.samples, {
    ...options,
    durationSeconds: audio.durationSeconds,
  });
}

export async function transcribeAudioSamples(samples, options = {}) {
  validatePcmSamples(samples);
  const transcriber = await requireTranscriber(options);

  emitProgress(options, {
    stage: "transcribe",
    message: "Transcribing captured audio locally with audio-analysis WebGPU…",
  });
  const output = await transcriber(samples, {
    chunk_length_s: DEFAULT_WINDOW_SECONDS,
    stride_length_s: DEFAULT_STRIDE_SECONDS,
    return_timestamps: true,
    task: "transcribe",
  });
  return normalizeBrowserTranscriptionOutput(output, {
    durationSeconds:
      finiteOrNull(options.durationSeconds) ?? samples.length / BROWSER_SAMPLE_RATE_HZ,
    source: typeof options.source === "string" ? options.source : "browser-audio",
  });
}

export function createBrowserTranscriptionSession(options = {}) {
  const plan = browserTranscriptionWindowPlan(options);
  const source = typeof options.source === "string" ? options.source : "browser-pcm-stream";
  const queue = createPcmQueue();
  const committedSegments = [];
  let committedThroughSeconds = 0;
  let absoluteStartSample = 0;
  let windowIndex = 0;
  let closed = false;
  let drainPromise = null;
  let transcriberReadyPromise = null;

  async function ensureTranscriber() {
    transcriberReadyPromise ??= requireTranscriber(options);
    return transcriberReadyPromise;
  }

  async function processWindow(final) {
    const length = final ? queue.length : plan.windowSamples;
    if (length <= 0) return [];

    const samples = queue.peek(length);
    const windowStartSeconds = absoluteStartSample / BROWSER_SAMPLE_RATE_HZ;
    const durationSeconds = samples.length / BROWSER_SAMPLE_RATE_HZ;
    const transcriber = await ensureTranscriber();

    emitProgress(options, {
      stage: "transcribe",
      message: final
        ? `Transcribing final bounded audio window ${windowIndex + 1}…`
        : `Transcribing bounded audio window ${windowIndex + 1}…`,
      detail: {
        windowIndex,
        windowStartSeconds,
        durationSeconds,
        bufferedSeconds: queue.length / BROWSER_SAMPLE_RATE_HZ,
      },
    });

    const output = await transcriber(samples, {
      chunk_length_s: plan.windowSeconds,
      stride_length_s: 0,
      return_timestamps: true,
      task: "transcribe",
    });
    const normalized = normalizeBrowserTranscriptionOutput(output, {
      durationSeconds,
      offsetSeconds: windowStartSeconds,
      source,
    });
    const commitUntilSeconds = final ? Infinity : windowStartSeconds + plan.stepSeconds;
    const stitched = stitchBrowserTranscriptionWindow(normalized.segments, {
      committedThroughSeconds,
      commitUntilSeconds,
      final,
      startIndex: committedSegments.length,
    });
    committedThroughSeconds = stitched.committedThroughSeconds;
    committedSegments.push(
      ...stitched.segments.map((segment) => ({
        ...segment,
        attributes: {
          ...segment.attributes,
          windowIndex: String(windowIndex),
        },
      })),
    );

    const consumedSamples = final ? length : plan.stepSamples;
    queue.discard(consumedSamples);
    absoluteStartSample += consumedSamples;
    windowIndex += 1;
    return stitched.segments;
  }

  async function drainAvailableWindows() {
    const emitted = [];
    while (queue.length >= plan.windowSamples) {
      emitted.push(...(await processWindow(false)));
    }
    return emitted;
  }

  async function ensureDrained() {
    while (queue.length >= plan.windowSamples) {
      if (!drainPromise) {
        drainPromise = drainAvailableWindows().finally(() => {
          drainPromise = null;
        });
      }
      await drainPromise;
    }
  }

  async function push(samples) {
    if (closed) {
      throw new Error("Browser transcription session is already closed.");
    }
    validatePcmSamples(samples);
    if (queue.length + samples.length > plan.maxBufferedSamples) {
      const error = new Error(
        `Browser transcription backlog exceeded the bounded ${plan.maxBufferedSeconds}-second buffer. Await push() backpressure or stop capture.`,
      );
      error.name = "BrowserTranscriptionBackpressureError";
      throw error;
    }

    const before = committedSegments.length;
    queue.append(samples);
    await ensureDrained();
    return committedSegments.slice(before);
  }

  async function flush() {
    if (!closed) closed = true;
    await ensureDrained();
    if (queue.length > 0) {
      await processWindow(true);
    }

    return {
      text: committedSegments.map((segment) => segment.text).join(" ").trim(),
      language: null,
      segments: committedSegments.slice(),
      source,
      attributes: {
        acceleration: "webgpu",
        modelId: DEFAULT_BROWSER_MODEL_ID,
        requiredChannels: "1",
        requiredSampleRateHz: String(BROWSER_SAMPLE_RATE_HZ),
        runtime: BROWSER_RUNTIME_ID,
        task: "transcribe",
        streaming: "bounded-pcm",
        windowSeconds: String(plan.windowSeconds),
        strideSeconds: String(plan.strideSeconds),
        maxBufferedSeconds: String(plan.maxBufferedSeconds),
        alignment: "not-run-in-browser",
        diarization: "not-run-in-browser",
      },
    };
  }

  return {
    push,
    flush,
    get bufferedSeconds() {
      return queue.length / BROWSER_SAMPLE_RATE_HZ;
    },
    get closed() {
      return closed;
    },
    plan,
  };
}

export function normalizeBrowserTranscriptionOutput(output, context = {}) {
  const text = String(output?.text ?? "").trim();
  const durationSeconds = finiteOrNull(context.durationSeconds);
  const offsetSeconds = finiteOrNull(context.offsetSeconds) ?? 0;
  const rawChunks = Array.isArray(output?.chunks) ? output.chunks : [];
  const chunks =
    rawChunks.length > 0
      ? rawChunks
      : [{ text, timestamp: text ? [0, durationSeconds] : [null, null] }];
  const segments = chunks
    .map((chunk, index) => {
      const segmentText = String(chunk?.text ?? "").trim();
      const timestamp = Array.isArray(chunk?.timestamp) ? chunk.timestamp : [];
      return {
        index,
        startSeconds: addOffset(finiteOrNull(timestamp[0]), offsetSeconds),
        endSeconds: addOffset(finiteOrNull(timestamp[1]), offsetSeconds),
        text: segmentText,
        language: null,
        speaker: null,
        confidence: null,
        isFinal: true,
        words: [],
        chars: [],
        attributes: {
          modelId: DEFAULT_BROWSER_MODEL_ID,
          runtime: BROWSER_RUNTIME_ID,
          task: "transcribe",
        },
      };
    })
    .filter((segment) => segment.text.length > 0);

  return {
    text: text || segments.map((segment) => segment.text).join(" "),
    language: null,
    segments,
    source: typeof context.source === "string" ? context.source : "browser-audio",
    attributes: {
      acceleration: "webgpu",
      modelId: DEFAULT_BROWSER_MODEL_ID,
      requiredChannels: "1",
      requiredSampleRateHz: String(BROWSER_SAMPLE_RATE_HZ),
      runtime: BROWSER_RUNTIME_ID,
      task: "transcribe",
      alignment: "not-run-in-browser",
      diarization: "not-run-in-browser",
    },
  };
}

async function decodeAndResample(source) {
  const arrayBuffer = await source.arrayBuffer();
  const decodeContext = new AudioContext();
  try {
    const decoded = await decodeContext.decodeAudioData(arrayBuffer.slice(0));
    const outputLength = Math.max(1, Math.ceil(decoded.duration * BROWSER_SAMPLE_RATE_HZ));
    const offline = new OfflineAudioContext(1, outputLength, BROWSER_SAMPLE_RATE_HZ);
    const bufferSource = offline.createBufferSource();
    bufferSource.buffer = decoded;
    bufferSource.connect(offline.destination);
    bufferSource.start(0);
    const rendered = await offline.startRendering();
    const samples = rendered.getChannelData(0).slice();
    return {
      samples,
      durationSeconds: samples.length / BROWSER_SAMPLE_RATE_HZ,
    };
  } finally {
    await decodeContext.close();
  }
}

async function requireTranscriber(options) {
  if (!(await supportsBrowserTranscription())) {
    throw new Error("WebGPU is required for browser transcription. No CPU or server fallback is used.");
  }
  emitProgress(options, {
    stage: "model",
    message: "Loading the audio-analysis Whisper model into the browser cache…",
  });
  activeModelProgress = typeof options.onProgress === "function" ? options.onProgress : null;
  try {
    return await getTranscriber();
  } finally {
    activeModelProgress = null;
  }
}

async function getTranscriber() {
  if (!transcriberPromise) {
    transcriberPromise = loadTransformers().then(({ pipeline }) =>
      pipeline("automatic-speech-recognition", DEFAULT_BROWSER_MODEL_ID, {
        device: "webgpu",
        progress_callback: (info) => {
          if (activeModelProgress) {
            activeModelProgress({
              stage: "model",
              message: modelProgressMessage(info),
              detail: info,
            });
          }
        },
      }),
    );
    transcriberPromise = transcriberPromise.catch((error) => {
      transcriberPromise = null;
      throw error;
    });
  }
  return transcriberPromise;
}

async function loadTransformers() {
  if (!transformersModulePromise) {
    transformersModulePromise = import(/* @vite-ignore */ TRANSFORMERS_MODULE_URL).then((module) => {
      module.env.allowLocalModels = false;
      module.env.useBrowserCache = true;
      module.env.useWasmCache = true;
      return module;
    });
  }
  return transformersModulePromise;
}

function createPcmQueue() {
  const chunks = [];
  let headOffset = 0;
  let totalSamples = 0;

  return {
    append(samples) {
      chunks.push(samples.slice());
      totalSamples += samples.length;
    },
    peek(length) {
      if (!Number.isInteger(length) || length < 0 || length > totalSamples) {
        throw new RangeError("Cannot read beyond the buffered PCM queue.");
      }
      const output = new Float32Array(length);
      let written = 0;
      let chunkIndex = 0;
      let offset = headOffset;
      while (written < length && chunkIndex < chunks.length) {
        const chunk = chunks[chunkIndex];
        const available = chunk.length - offset;
        const count = Math.min(available, length - written);
        output.set(chunk.subarray(offset, offset + count), written);
        written += count;
        chunkIndex += 1;
        offset = 0;
      }
      return output;
    },
    discard(length) {
      if (!Number.isInteger(length) || length < 0 || length > totalSamples) {
        throw new RangeError("Cannot discard beyond the buffered PCM queue.");
      }
      let remaining = length;
      while (remaining > 0 && chunks.length > 0) {
        const head = chunks[0];
        const available = head.length - headOffset;
        if (remaining < available) {
          headOffset += remaining;
          remaining = 0;
        } else {
          remaining -= available;
          chunks.shift();
          headOffset = 0;
        }
      }
      totalSamples -= length;
    },
    get length() {
      return totalSamples;
    },
  };
}

function validatePcmSamples(samples) {
  if (!(samples instanceof Float32Array) || samples.length === 0) {
    throw new TypeError("audio-analysis browser transcription requires non-empty Float32Array PCM.");
  }
  if (samples.some((sample) => !Number.isFinite(sample))) {
    throw new TypeError("audio-analysis browser transcription PCM samples must be finite.");
  }
}

function modelProgressMessage(info) {
  if (info && typeof info === "object" && info.status === "progress") {
    const file = typeof info.file === "string" ? ` (${shortFileName(info.file)})` : "";
    return `Downloading/caching audio-analysis model assets${file}…`;
  }
  if (info && typeof info === "object" && info.status === "done") {
    return "Audio-analysis model assets ready. Preparing WebGPU inference…";
  }
  return "Preparing the audio-analysis WebGPU transcription runtime…";
}

function emitProgress(options, update) {
  if (typeof options.onProgress === "function") {
    options.onProgress(update);
  }
}

function shortFileName(value) {
  const parts = value.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? value;
}

function addOffset(value, offsetSeconds) {
  return value === null ? null : value + offsetSeconds;
}

function secondsToSamples(seconds, minimum) {
  return Math.max(minimum, Math.round(seconds * BROWSER_SAMPLE_RATE_HZ));
}

function positiveFiniteOrDefault(value, fallback) {
  if (value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new RangeError("Expected a positive finite number.");
  }
  return value;
}

function nonNegativeFiniteOrDefault(value, fallback) {
  if (value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new RangeError("Expected a non-negative finite number.");
  }
  return value;
}

function nonNegativeIntegerOrDefault(value, fallback) {
  if (value === undefined) return fallback;
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError("Expected a non-negative integer.");
  }
  return value;
}

function nonNegativeFiniteOrInfinity(value, fallback) {
  if (value === undefined) return fallback;
  if (value === Infinity) return Infinity;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new RangeError("Expected a non-negative finite number or Infinity.");
  }
  return value;
}

function finiteOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
