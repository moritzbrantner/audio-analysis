let wasmModulePromise;

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


const BROWSER_DIARIZATION_SAMPLE_RATE_HZ = 16_000;
const BROWSER_DIARIZATION_RUNTIME_ID = "audio-analysis-browser-spectral";
const BROWSER_DIARIZATION_MODEL_ID = "spectral-speaker-baseline";
const DEFAULT_WINDOW_SECONDS = 1.5;
const DEFAULT_HOP_SECONDS = 0.75;
const DEFAULT_VAD_THRESHOLD = 0.012;
const DEFAULT_CLUSTER_THRESHOLD = 0.22;
const DEFAULT_MAX_SPEAKERS = 8;
const FEATURE_FREQUENCIES_HZ = Object.freeze([120, 180, 260, 380, 560, 820, 1_200, 1_800, 2_600, 3_600]);

export function browserDiarizationCapabilities() {
  return {
    runtime: BROWSER_DIARIZATION_RUNTIME_ID,
    modelId: BROWSER_DIARIZATION_MODEL_ID,
    modelProvisioning: "built-in",
    input: {
      sampleRateHz: BROWSER_DIARIZATION_SAMPLE_RATE_HZ,
      channels: 1,
      sampleFormat: "f32",
      acceptedSources: ["Blob", "Float32Array"],
    },
    features: {
      diarization: true,
      transcriptAssignment: true,
      speakerIdentification: false,
    },
    quality: "deterministic-baseline",
    fallbacks: {
      server: false,
      python: false,
    },
  };
}

export async function supportsBrowserDiarization() {
  return (
    typeof globalThis.AudioContext === "function"
    && typeof globalThis.OfflineAudioContext === "function"
  );
}

export async function diarizeAudioBlob(source, options = {}) {
  if (!source || typeof source.arrayBuffer !== "function") {
    throw new TypeError("audio-analysis browser diarization requires a Blob-like audio source.");
  }
  emitBrowserProgress(options, {
    stage: "decode",
    message: "Decoding and resampling audio for local speaker diarization…",
  });
  const audio = await decodeBrowserAudio(source);
  return diarizeAudioSamples(audio.samples, {
    ...options,
    durationSeconds: audio.durationSeconds,
  });
}

export function diarizeAudioSamples(samples, options = {}) {
  validateBrowserSamples(samples);
  const sampleRateHz = positiveFiniteOrDefault(
    options.sampleRateHz,
    BROWSER_DIARIZATION_SAMPLE_RATE_HZ,
  );
  if (sampleRateHz !== BROWSER_DIARIZATION_SAMPLE_RATE_HZ) {
    throw new RangeError(
      `Browser diarization expects ${BROWSER_DIARIZATION_SAMPLE_RATE_HZ} Hz mono PCM.`,
    );
  }

  const windowSeconds = positiveFiniteOrDefault(options.windowSeconds, DEFAULT_WINDOW_SECONDS);
  const hopSeconds = positiveFiniteOrDefault(options.hopSeconds, DEFAULT_HOP_SECONDS);
  const vadThreshold = nonNegativeFiniteOrDefault(options.vadThreshold, DEFAULT_VAD_THRESHOLD);
  const clusterThreshold = positiveFiniteOrDefault(
    options.clusterThreshold,
    DEFAULT_CLUSTER_THRESHOLD,
  );
  const maxSpeakers = positiveIntegerOrDefault(options.maxSpeakers, DEFAULT_MAX_SPEAKERS);
  if (hopSeconds > windowSeconds) {
    throw new RangeError("Browser diarization hopSeconds must not exceed windowSeconds.");
  }

  const windowSamples = Math.max(1, Math.round(windowSeconds * sampleRateHz));
  const hopSamples = Math.max(1, Math.round(hopSeconds * sampleRateHz));
  const windows = [];
  for (let start = 0; start < samples.length; start += hopSamples) {
    const end = Math.min(samples.length, start + windowSamples);
    if (end - start < Math.min(windowSamples, Math.round(0.25 * sampleRateHz))) {
      break;
    }
    const frame = samples.subarray(start, end);
    const rms = rootMeanSquare(frame);
    if (rms < vadThreshold) {
      continue;
    }
    windows.push({
      start,
      end,
      rms,
      feature: speakerFeature(frame, sampleRateHz),
    });
  }

  emitBrowserProgress(options, {
    stage: "diarize",
    message: "Clustering local speech windows into anonymous speakers…",
    detail: { speechWindows: windows.length },
  });

  const prototypes = [];
  const assigned = windows.map((window) => {
    const assignment = assignPrototype(window.feature, prototypes, clusterThreshold, maxSpeakers);
    return { ...window, speakerIndex: assignment };
  });
  const segments = mergeSpeakerWindows(assigned, sampleRateHz, hopSamples);
  const speakerCount = new Set(segments.map((segment) => segment.speaker)).size;

  return {
    accepted: true,
    operation: "diarize",
    modelId: BROWSER_DIARIZATION_MODEL_ID,
    runtime: BROWSER_DIARIZATION_RUNTIME_ID,
    speakerCount,
    segments,
    attributes: {
      durationSeconds:
        finiteOrNull(options.durationSeconds) ?? samples.length / sampleRateHz,
      quality: "deterministic-baseline",
      sampleRateHz,
      windowSeconds,
      hopSeconds,
      clusterThreshold,
    },
    diagnostics: [
      "Browser diarization uses the deterministic audio-analysis spectral baseline. It is local and suitable for preview/prototyping, not pyannote parity.",
    ],
  };
}

export function assignBrowserSpeakersToSegments(segments, diarization) {
  if (!Array.isArray(segments)) {
    throw new TypeError("Browser speaker assignment requires transcript segments.");
  }
  const speakerSegments = Array.isArray(diarization?.segments) ? diarization.segments : [];
  return segments.map((segment) => {
    const start = finiteOrNull(segment?.startSeconds);
    const end = finiteOrNull(segment?.endSeconds);
    if (start === null || end === null || end < start) {
      return { ...segment, speaker: segment?.speaker ?? null };
    }

    let bestSpeaker = null;
    let bestOverlap = 0;
    for (const speakerSegment of speakerSegments) {
      const speakerStart = finiteOrNull(speakerSegment?.startSeconds);
      const speakerEnd = finiteOrNull(speakerSegment?.endSeconds);
      if (speakerStart === null || speakerEnd === null || speakerEnd < speakerStart) {
        continue;
      }
      const overlap = Math.max(0, Math.min(end, speakerEnd) - Math.max(start, speakerStart));
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestSpeaker = typeof speakerSegment.speaker === "string" ? speakerSegment.speaker : null;
      }
    }
    return { ...segment, speaker: bestSpeaker ?? segment?.speaker ?? null };
  });
}

async function decodeBrowserAudio(source) {
  const arrayBuffer = await source.arrayBuffer();
  const AudioContextConstructor = globalThis.AudioContext;
  const OfflineAudioContextConstructor = globalThis.OfflineAudioContext;
  if (
    typeof AudioContextConstructor !== "function"
    || typeof OfflineAudioContextConstructor !== "function"
  ) {
    throw new Error("Browser AudioContext support is required for local diarization.");
  }
  const decodeContext = new AudioContextConstructor();
  try {
    const decoded = await decodeContext.decodeAudioData(arrayBuffer.slice(0));
    const outputLength = Math.max(
      1,
      Math.ceil(decoded.duration * BROWSER_DIARIZATION_SAMPLE_RATE_HZ),
    );
    const offline = new OfflineAudioContextConstructor(
      1,
      outputLength,
      BROWSER_DIARIZATION_SAMPLE_RATE_HZ,
    );
    const bufferSource = offline.createBufferSource();
    bufferSource.buffer = decoded;
    bufferSource.connect(offline.destination);
    bufferSource.start(0);
    const rendered = await offline.startRendering();
    const samples = rendered.getChannelData(0).slice();
    return {
      samples,
      durationSeconds: samples.length / BROWSER_DIARIZATION_SAMPLE_RATE_HZ,
    };
  } finally {
    await decodeContext.close();
  }
}

function speakerFeature(frame, sampleRateHz) {
  let mean = 0;
  for (const sample of frame) {
    mean += sample;
  }
  mean /= frame.length;

  const centered = new Float32Array(frame.length);
  let energy = 0;
  let zeroCrossings = 0;
  let derivativeEnergy = 0;
  let previous = frame[0] - mean;
  for (let index = 0; index < frame.length; index += 1) {
    const value = frame[index] - mean;
    centered[index] = value;
    energy += value * value;
    if (index > 0) {
      if ((value >= 0) !== (previous >= 0)) {
        zeroCrossings += 1;
      }
      const delta = value - previous;
      derivativeEnergy += delta * delta;
    }
    previous = value;
  }

  const norm = Math.sqrt(energy) || 1;
  const features = [
    Math.sqrt(energy / frame.length),
    zeroCrossings / Math.max(1, frame.length - 1),
    Math.sqrt(derivativeEnergy / Math.max(1, frame.length - 1)),
  ];
  const stride = Math.max(1, Math.floor(frame.length / 6_000));
  for (const frequencyHz of FEATURE_FREQUENCIES_HZ) {
    const omega = (2 * Math.PI * frequencyHz * stride) / sampleRateHz;
    let real = 0;
    let imag = 0;
    let sampleIndex = 0;
    for (let index = 0; index < centered.length; index += stride) {
      const value = centered[index] / norm;
      const phase = omega * sampleIndex;
      real += value * Math.cos(phase);
      imag -= value * Math.sin(phase);
      sampleIndex += 1;
    }
    features.push(Math.log1p(real * real + imag * imag));
  }
  return normalizeVector(features);
}

function assignPrototype(feature, prototypes, threshold, maxSpeakers) {
  let bestIndex = -1;
  let bestDistance = Infinity;
  for (let index = 0; index < prototypes.length; index += 1) {
    const distance = 1 - cosineSimilarity(feature, prototypes[index].feature);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }

  if ((bestIndex === -1 || bestDistance > threshold) && prototypes.length < maxSpeakers) {
    prototypes.push({ feature: feature.slice(), count: 1 });
    return prototypes.length - 1;
  }

  const chosen = Math.max(0, bestIndex);
  const prototype = prototypes[chosen];
  const nextCount = prototype.count + 1;
  prototype.feature = normalizeVector(
    prototype.feature.map(
      (value, index) => (value * prototype.count + feature[index]) / nextCount,
    ),
  );
  prototype.count = nextCount;
  return chosen;
}

function mergeSpeakerWindows(windows, sampleRateHz, hopSamples) {
  const merged = [];
  for (const window of windows) {
    const speaker = `speaker_${window.speakerIndex}`;
    const startSeconds = window.start / sampleRateHz;
    const endSeconds = window.end / sampleRateHz;
    const score = Math.min(1, Math.max(0, window.rms * 8));
    const previous = merged[merged.length - 1];
    if (
      previous
      && previous.speaker === speaker
      && startSeconds <= previous.endSeconds + hopSamples / sampleRateHz
    ) {
      previous.endSeconds = Math.max(previous.endSeconds, endSeconds);
      previous.score = Math.max(previous.score, score);
      continue;
    }
    merged.push({ speaker, startSeconds, endSeconds, score });
  }
  return merged;
}

function rootMeanSquare(frame) {
  let sum = 0;
  for (const sample of frame) {
    sum += sample * sample;
  }
  return Math.sqrt(sum / frame.length);
}

function normalizeVector(values) {
  let normSquared = 0;
  for (const value of values) {
    normSquared += value * value;
  }
  const norm = Math.sqrt(normSquared) || 1;
  return values.map((value) => value / norm);
}

function cosineSimilarity(left, right) {
  let dot = 0;
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    dot += left[index] * right[index];
  }
  return dot;
}

function validateBrowserSamples(samples) {
  if (!(samples instanceof Float32Array) || samples.length === 0) {
    throw new TypeError("audio-analysis browser diarization requires non-empty Float32Array PCM.");
  }
  for (const sample of samples) {
    if (!Number.isFinite(sample)) {
      throw new TypeError("audio-analysis browser diarization PCM samples must be finite.");
    }
  }
}

function emitBrowserProgress(options, update) {
  if (typeof options.onProgress === "function") {
    options.onProgress(update);
  }
}

function positiveFiniteOrDefault(value, fallback) {
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new RangeError("Expected a positive finite number.");
  }
  return value;
}

function nonNegativeFiniteOrDefault(value, fallback) {
  if (value === undefined) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new RangeError("Expected a non-negative finite number.");
  }
  return value;
}

function positiveIntegerOrDefault(value, fallback) {
  if (value === undefined) {
    return fallback;
  }
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError("Expected a positive integer.");
  }
  return value;
}

function finiteOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
