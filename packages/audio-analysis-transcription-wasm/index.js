const DEFAULT_BROWSER_MODEL_ID = "onnx-community/whisper-tiny";
const TRANSFORMERS_MODULE_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const BROWSER_SAMPLE_RATE_HZ = 16_000;
const BROWSER_RUNTIME_ID = "audio-analysis-transformers-js-webgpu";

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
      acceptedSources: ["Blob", "Float32Array"],
    },
    features: {
      transcription: true,
      timedSegments: true,
      alignment: false,
      diarization: false,
      translation: false,
    },
    fallbacks: {
      server: false,
      python: false,
      cpu: false,
    },
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
  if (!(samples instanceof Float32Array) || samples.length === 0) {
    throw new TypeError("audio-analysis browser transcription requires non-empty Float32Array PCM.");
  }
  if (samples.some((sample) => !Number.isFinite(sample))) {
    throw new TypeError("audio-analysis browser transcription PCM samples must be finite.");
  }
  if (!(await supportsBrowserTranscription())) {
    throw new Error("WebGPU is required for browser transcription. No CPU or server fallback is used.");
  }

  emitProgress(options, {
    stage: "model",
    message: "Loading the audio-analysis Whisper model into the browser cache…",
  });
  activeModelProgress = typeof options.onProgress === "function" ? options.onProgress : null;
  try {
    const transcriber = await getTranscriber();
    emitProgress(options, {
      stage: "transcribe",
      message: "Transcribing captured audio locally with audio-analysis WebGPU…",
    });
    const output = await transcriber(samples, {
      chunk_length_s: 29,
      stride_length_s: 5,
      return_timestamps: true,
      task: "transcribe",
    });
    return normalizeBrowserTranscriptionOutput(output, {
      durationSeconds:
        finiteOrNull(options.durationSeconds) ?? samples.length / BROWSER_SAMPLE_RATE_HZ,
      source: typeof options.source === "string" ? options.source : "browser-audio",
    });
  } finally {
    activeModelProgress = null;
  }
}

export function normalizeBrowserTranscriptionOutput(output, context = {}) {
  const text = String(output?.text ?? "").trim();
  const durationSeconds = finiteOrNull(context.durationSeconds);
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
        startSeconds: finiteOrNull(timestamp[0]),
        endSeconds: finiteOrNull(timestamp[1]),
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

function finiteOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
