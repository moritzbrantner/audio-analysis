const DEFAULT_BROWSER_DECODE_SAMPLE_RATE_HZ = 16_000;

let wasmModulePromise;

export async function init() {
  const wasmEntry = "./pkg/audio_analysis_io_wasm.js";
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

export function browserAudioDecodeCapabilities() {
  return {
    runtime: "web-audio",
    acceptedSources: ["Blob"],
    output: {
      channels: 1,
      sampleFormat: "f32",
      defaultSampleRateHz: DEFAULT_BROWSER_DECODE_SAMPLE_RATE_HZ,
    },
    fallbacks: {
      server: false,
      python: false,
    },
  };
}

export function supportsBrowserAudioDecode() {
  return (
    typeof globalThis.AudioContext === "function"
    && typeof globalThis.OfflineAudioContext === "function"
  );
}

export async function decodeBrowserAudioBlob(source, options = {}) {
  if (!source || typeof source.arrayBuffer !== "function") {
    throw new TypeError("audio-analysis browser decode requires a Blob-like audio source.");
  }

  const sampleRateHz = positiveFiniteSampleRate(
    options.sampleRateHz,
    DEFAULT_BROWSER_DECODE_SAMPLE_RATE_HZ,
  );
  const AudioContextConstructor = globalThis.AudioContext;
  const OfflineAudioContextConstructor = globalThis.OfflineAudioContext;
  if (
    typeof AudioContextConstructor !== "function"
    || typeof OfflineAudioContextConstructor !== "function"
  ) {
    throw new Error(
      "Browser audio decode requires AudioContext and OfflineAudioContext support.",
    );
  }

  const encoded = await source.arrayBuffer();
  const decodeContext = new AudioContextConstructor();
  try {
    const decoded = await decodeContext.decodeAudioData(encoded.slice(0));
    const sourceSampleRateHz = positiveFiniteSampleRate(decoded.sampleRate, null);
    const sourceChannels = positiveInteger(decoded.numberOfChannels, "decoded channel count");
    const sourceDurationSeconds = positiveFinite(decoded.duration, "decoded duration");
    const outputLength = Math.max(1, Math.ceil(sourceDurationSeconds * sampleRateHz));
    const offline = new OfflineAudioContextConstructor(1, outputLength, sampleRateHz);
    const bufferSource = offline.createBufferSource();
    bufferSource.buffer = decoded;
    bufferSource.connect(offline.destination);
    bufferSource.start(0);

    const rendered = await offline.startRendering();
    const samples = rendered.getChannelData(0).slice();
    validateFinitePcm(samples);

    return {
      samples,
      sampleRateHz,
      channels: 1,
      durationSeconds: samples.length / sampleRateHz,
      sourceSampleRateHz,
      sourceChannels,
    };
  } finally {
    await closeAudioContext(decodeContext);
  }
}

function positiveFiniteSampleRate(value, fallback) {
  const candidate = value === undefined ? fallback : value;
  if (
    typeof candidate !== "number"
    || !Number.isFinite(candidate)
    || candidate <= 0
  ) {
    throw new RangeError("Browser audio decode requires a positive finite sample rate.");
  }
  return candidate;
}

function positiveFinite(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`Browser audio decode returned an invalid ${label}.`);
  }
  return value;
}

function positiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Browser audio decode returned an invalid ${label}.`);
  }
  return value;
}

function validateFinitePcm(samples) {
  if (!(samples instanceof Float32Array) || samples.length === 0) {
    throw new Error("Browser audio decode produced no PCM samples.");
  }
  for (const sample of samples) {
    if (!Number.isFinite(sample)) {
      throw new Error("Browser audio decode produced non-finite PCM samples.");
    }
  }
}

async function closeAudioContext(context) {
  if (!context || context.state === "closed" || typeof context.close !== "function") {
    return;
  }
  await context.close();
}
