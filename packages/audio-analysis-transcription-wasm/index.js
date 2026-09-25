const BROWSER_TRANSCRIPTION_MODELS = Object.freeze([
  Object.freeze({
    id: "onnx-community/whisper-tiny",
    label: "Whisper Tiny",
    description: "Fastest and lowest-memory browser option.",
  }),
  Object.freeze({
    id: "onnx-community/whisper-base",
    label: "Whisper Base",
    description: "Balanced browser accuracy and resource use.",
  }),
  Object.freeze({
    id: "onnx-community/whisper-small",
    label: "Whisper Small",
    description: "Higher accuracy with a much larger download and memory footprint.",
  }),
]);
const DEFAULT_BROWSER_MODEL_ID = BROWSER_TRANSCRIPTION_MODELS[0].id;
const TRANSFORMERS_MODULE_URL =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const BROWSER_SAMPLE_RATE_HZ = 16_000;
const BROWSER_RUNTIME_ID = "audio-analysis-transformers-js-webgpu";
const DEFAULT_WINDOW_SECONDS = 29;
const DEFAULT_STRIDE_SECONDS = 5;
const DEFAULT_MAX_BUFFERED_SECONDS = 58;
const TIME_EPSILON_SECONDS = 1e-6;
const PCM_CAPTURE_PROCESSOR_NAME = "moenarch-audio-analysis-pcm-capture-v1";
const PCM_CAPTURE_CHUNK_FRAMES = BROWSER_SAMPLE_RATE_HZ;

let wasmModulePromise;
let transformersModulePromise;
const transcriberEntries = new Map();
const activeModelProgress = new Map();
let preferredModelId = DEFAULT_BROWSER_MODEL_ID;

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

export function browserTranscriptionModels() {
  return BROWSER_TRANSCRIPTION_MODELS.map((model) => ({ ...model }));
}

export function browserTranscriptionCapabilities() {
  return {
    runtime: BROWSER_RUNTIME_ID,
    requiredAcceleration: "webgpu",
    modelId: DEFAULT_BROWSER_MODEL_ID,
    models: browserTranscriptionModels(),
    modelProvisioning: "browser-cache",
    modelLifecycle: {
      maxIdleResidentModels: 1,
      eviction: "dispose-superseded",
    },
    input: {
      sampleRateHz: BROWSER_SAMPLE_RATE_HZ,
      channels: 1,
      sampleFormat: "f32",
      acceptedSources: [
        "Blob",
        "Float32Array",
        "bounded Float32Array stream",
        "WebCodecs AudioData stream",
        "caller-acquired MediaStream",
      ],
    },
    features: {
      transcription: true,
      timedSegments: true,
      boundedPcmStreaming: true,
      decodedAudioAdapter: true,
      mediaStreamAdapter: true,
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
  const model = resolveBrowserModel(options.modelId);
  const modelOptions = { ...options, modelId: model.id };
  const lease = await acquireTranscriber(modelOptions);

  try {
    emitProgress(modelOptions, {
      stage: "transcribe",
      message: `Transcribing captured audio locally with ${model.label} on WebGPU…`,
    });
    const output = await lease.transcriber(samples, {
      chunk_length_s: DEFAULT_WINDOW_SECONDS,
      stride_length_s: DEFAULT_STRIDE_SECONDS,
      return_timestamps: true,
      task: "transcribe",
    });
    return normalizeBrowserTranscriptionOutput(output, {
      durationSeconds:
        finiteOrNull(options.durationSeconds) ?? samples.length / BROWSER_SAMPLE_RATE_HZ,
      source: typeof options.source === "string" ? options.source : "browser-audio",
      modelId: model.id,
    });
  } finally {
    await lease.release();
  }
}

export function createBrowserPcmResampler(inputSampleRateHz) {
  if (
    typeof inputSampleRateHz !== "number"
    || !Number.isFinite(inputSampleRateHz)
    || inputSampleRateHz <= 0
  ) {
    throw new RangeError("Browser PCM resampling requires a positive finite input sample rate.");
  }

  const step = inputSampleRateHz / BROWSER_SAMPLE_RATE_HZ;
  let inputCursor = 0;
  let nextOutputPosition = 0;
  let previousSample = null;
  let closed = false;

  function push(channelPlanes) {
    if (closed) {
      throw new Error("Browser PCM resampler is already closed.");
    }
    if (!Array.isArray(channelPlanes) || channelPlanes.length === 0) {
      throw new TypeError("Browser PCM resampling requires at least one channel plane.");
    }
    const frameCount = channelPlanes[0]?.length ?? 0;
    if (frameCount === 0) {
      throw new TypeError("Browser PCM resampling requires non-empty channel planes.");
    }
    for (const plane of channelPlanes) {
      if (!(plane instanceof Float32Array) || plane.length !== frameCount) {
        throw new TypeError(
          "Browser PCM resampling requires equally sized Float32Array channel planes.",
        );
      }
    }

    const mono = new Float32Array(frameCount);
    for (let frame = 0; frame < frameCount; frame += 1) {
      let mixed = 0;
      for (const plane of channelPlanes) {
        const sample = plane[frame];
        if (!Number.isFinite(sample)) {
          throw new TypeError("Browser PCM resampling requires finite samples.");
        }
        mixed += sample;
      }
      mono[frame] = mixed / channelPlanes.length;
    }

    const chunkStart = inputCursor;
    const chunkEnd = chunkStart + frameCount;
    const output = [];

    while (Math.ceil(nextOutputPosition) < chunkEnd) {
      const leftIndex = Math.floor(nextOutputPosition);
      const rightIndex = Math.ceil(nextOutputPosition);
      const left = sourceSample(leftIndex);
      const right = sourceSample(rightIndex);
      if (left === undefined || right === undefined) {
        break;
      }
      const fraction = nextOutputPosition - leftIndex;
      output.push(left + (right - left) * fraction);
      nextOutputPosition += step;
    }

    previousSample = mono[frameCount - 1];
    inputCursor = chunkEnd;
    return Float32Array.from(output);

    function sourceSample(absoluteIndex) {
      if (absoluteIndex === chunkStart - 1) {
        return previousSample ?? undefined;
      }
      if (absoluteIndex < chunkStart || absoluteIndex >= chunkEnd) {
        return undefined;
      }
      return mono[absoluteIndex - chunkStart];
    }
  }

  function close() {
    closed = true;
  }

  return {
    push,
    close,
    get inputSampleRateHz() {
      return inputSampleRateHz;
    },
    get outputSampleRateHz() {
      return BROWSER_SAMPLE_RATE_HZ;
    },
    get closed() {
      return closed;
    },
  };
}

export function createBrowserDecodedAudioTranscriptionSession(options = {}) {
  const transcription = createBrowserTranscriptionSession(options);
  let resampler = null;
  let decodedSampleRateHz = null;
  let closed = false;

  async function push(audioData) {
    if (closed) {
      throw new Error("Browser decoded-audio transcription session is already closed.");
    }
    if (
      !audioData
      || typeof audioData.copyTo !== "function"
      || typeof audioData.close !== "function"
      || !Number.isInteger(audioData.numberOfFrames)
      || audioData.numberOfFrames <= 0
      || !Number.isInteger(audioData.numberOfChannels)
      || audioData.numberOfChannels <= 0
      || typeof audioData.sampleRate !== "number"
      || !Number.isFinite(audioData.sampleRate)
      || audioData.sampleRate <= 0
    ) {
      throw new TypeError("Browser decoded-audio transcription requires a valid AudioData frame.");
    }

    try {
      if (decodedSampleRateHz === null) {
        decodedSampleRateHz = audioData.sampleRate;
        resampler = createBrowserPcmResampler(decodedSampleRateHz);
      } else if (audioData.sampleRate !== decodedSampleRateHz) {
        throw new Error(
          `Decoded audio sample rate changed from ${decodedSampleRateHz} Hz to ${audioData.sampleRate} Hz.`,
        );
      }

      const planes = [];
      for (let channel = 0; channel < audioData.numberOfChannels; channel += 1) {
        const plane = new Float32Array(audioData.numberOfFrames);
        audioData.copyTo(plane, { planeIndex: channel, format: "f32-planar" });
        planes.push(plane);
      }
      const samples = resampler.push(planes);
      if (samples.length === 0) {
        return [];
      }
      return await transcription.push(samples);
    } finally {
      audioData.close();
    }
  }

  async function flush() {
    if (!closed) {
      closed = true;
      resampler?.close();
    }
    return transcription.flush();
  }

  return {
    push,
    flush,
    get bufferedSeconds() {
      return transcription.bufferedSeconds;
    },
    get closed() {
      return closed;
    },
    get decodedSampleRateHz() {
      return decodedSampleRateHz;
    },
    get outputSampleRateHz() {
      return BROWSER_SAMPLE_RATE_HZ;
    },
    plan: transcription.plan,
  };
}

export function createBrowserTranscriptionSession(options = {}) {
  const plan = browserTranscriptionWindowPlan(options);
  const model = resolveBrowserModel(options.modelId);
  const modelOptions = { ...options, modelId: model.id };
  const source = typeof options.source === "string" ? options.source : "browser-pcm-stream";
  const queue = createPcmQueue();
  const committedSegments = [];
  let committedThroughSeconds = 0;
  let absoluteStartSample = 0;
  let windowIndex = 0;
  let closed = false;
  let drainPromise = null;

  async function processWindow(final) {
    const length = final ? queue.length : plan.windowSamples;
    if (length <= 0) return [];

    const samples = queue.peek(length);
    const windowStartSeconds = absoluteStartSample / BROWSER_SAMPLE_RATE_HZ;
    const durationSeconds = samples.length / BROWSER_SAMPLE_RATE_HZ;
    const lease = await acquireTranscriber(modelOptions);
    let output;
    try {
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

      output = await lease.transcriber(samples, {
        chunk_length_s: plan.windowSeconds,
        stride_length_s: 0,
        return_timestamps: true,
        task: "transcribe",
      });
    } finally {
      await lease.release();
    }
    const normalized = normalizeBrowserTranscriptionOutput(output, {
      durationSeconds,
      offsetSeconds: windowStartSeconds,
      source,
      modelId: model.id,
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
        modelId: model.id,
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

export async function createBrowserMediaStreamTranscriptionSession(stream, options = {}) {
  const audioTrack = requireBrowserAudioTrack(stream);
  const AudioContextConstructor = globalThis.AudioContext;
  const AudioWorkletNodeConstructor = globalThis.AudioWorkletNode;
  const MediaStreamConstructor = globalThis.MediaStream;

  if (!(await supportsBrowserTranscription())) {
    throw new Error("WebGPU is required for browser transcription. No CPU or server fallback is used.");
  }
  if (typeof AudioContextConstructor !== "function" || typeof AudioWorkletNodeConstructor !== "function") {
    throw new Error(
      "Browser MediaStream transcription requires AudioContext and AudioWorklet support.",
    );
  }
  if (typeof MediaStreamConstructor !== "function") {
    throw new Error("Browser MediaStream transcription requires the MediaStream API.");
  }

  const session = createBrowserTranscriptionSession(options);
  let audioContext;
  try {
    audioContext = new AudioContextConstructor({ sampleRate: BROWSER_SAMPLE_RATE_HZ });
  } catch (error) {
    throw new Error(
      `Unable to create the required ${BROWSER_SAMPLE_RATE_HZ} Hz browser audio context: ${formatError(error)}`,
    );
  }

  if (audioContext.sampleRate !== BROWSER_SAMPLE_RATE_HZ) {
    await closeAudioContext(audioContext);
    throw new Error(
      `Browser MediaStream transcription requires an exact ${BROWSER_SAMPLE_RATE_HZ} Hz audio context; got ${audioContext.sampleRate} Hz.`,
    );
  }
  if (!audioContext.audioWorklet || typeof audioContext.audioWorklet.addModule !== "function") {
    await closeAudioContext(audioContext);
    throw new Error("Browser MediaStream transcription requires AudioWorklet module support.");
  }

  const workletUrl = createPcmCaptureWorkletUrl();
  try {
    await audioContext.audioWorklet.addModule(workletUrl);
  } catch (error) {
    await closeAudioContext(audioContext);
    throw new Error(`Unable to initialize the browser PCM capture worklet: ${formatError(error)}`);
  } finally {
    URL.revokeObjectURL(workletUrl);
  }

  const captureStream = new MediaStreamConstructor([audioTrack]);
  const sourceNode = audioContext.createMediaStreamSource(captureStream);
  const workletNode = new AudioWorkletNodeConstructor(audioContext, PCM_CAPTURE_PROCESSOR_NAME, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: { chunkFrames: PCM_CAPTURE_CHUNK_FRAMES },
  });
  const silentGain = audioContext.createGain();
  silentGain.gain.value = 0;

  const pendingPushes = new Set();
  let graphConnected = false;
  let closed = false;
  let terminalError = null;
  let finishPromise = null;
  let flushResolver = null;

  function releaseFlushWaiter() {
    if (!flushResolver) return;
    const resolve = flushResolver;
    flushResolver = null;
    resolve();
  }

  function disconnectGraph() {
    if (!graphConnected) return;
    graphConnected = false;
    for (const node of [sourceNode, workletNode, silentGain]) {
      try {
        node.disconnect();
      } catch {
        // The graph is already detached.
      }
    }
  }

  function failCapture(error) {
    if (!terminalError) {
      terminalError = toError(error);
      if (typeof options.onError === "function") {
        try {
          options.onError(terminalError);
        } catch {
          // Consumer error reporting must not replace the capture failure.
        }
      }
    }
    releaseFlushWaiter();
    disconnectGraph();
    void audioContext.suspend().catch(() => {});
  }

  function enqueuePcm(samples) {
    if (closed || terminalError) return;
    if (!(samples instanceof Float32Array)) {
      failCapture(new TypeError("Browser PCM capture worklet returned a non-Float32Array payload."));
      return;
    }

    let pending;
    pending = session
      .push(samples)
      .then((segments) => {
        if (segments.length > 0 && typeof options.onSegments === "function") {
          options.onSegments(segments);
        }
      })
      .catch((error) => {
        failCapture(error);
      })
      .finally(() => {
        pendingPushes.delete(pending);
      });
    pendingPushes.add(pending);
  }

  workletNode.port.onmessage = (event) => {
    const data = event.data;
    if (!data || typeof data !== "object") return;
    if (data.type === "pcm") {
      enqueuePcm(data.samples);
    } else if (data.type === "flushed") {
      releaseFlushWaiter();
    }
  };
  workletNode.addEventListener("processorerror", () => {
    failCapture(new Error("Browser PCM capture worklet stopped unexpectedly."));
  });

  sourceNode.connect(workletNode);
  workletNode.connect(silentGain);
  silentGain.connect(audioContext.destination);
  graphConnected = true;

  try {
    await audioContext.resume();
  } catch (error) {
    failCapture(
      new Error(
        `Unable to start browser audio capture. Start it from a user interaction: ${formatError(error)}`,
      ),
    );
  }
  if (audioContext.state !== "running" && !terminalError) {
    failCapture(
      new Error(
        "Browser audio capture remained suspended. Start transcription from a user interaction.",
      ),
    );
  }
  if (terminalError) {
    await closeAudioContext(audioContext);
    throw terminalError;
  }

  emitProgress(options, {
    stage: "capture",
    message: "Capturing caller-approved audio as bounded 16 kHz mono PCM…",
    detail: {
      sampleRateHz: BROWSER_SAMPLE_RATE_HZ,
      chunkFrames: PCM_CAPTURE_CHUNK_FRAMES,
      maxBufferedSeconds: session.plan.maxBufferedSeconds,
    },
  });

  async function finish() {
    finishPromise ??= (async () => {
      try {
        if (!terminalError && graphConnected) {
          const flushed = new Promise((resolve) => {
            flushResolver = resolve;
          });
          workletNode.port.postMessage({ type: "flush" });
          await flushed;
        }
        disconnectGraph();
        await closeAudioContext(audioContext);
        await Promise.all([...pendingPushes]);
        if (terminalError) throw terminalError;
        return await session.flush();
      } finally {
        closed = true;
        releaseFlushWaiter();
        disconnectGraph();
        await closeAudioContext(audioContext);
      }
    })();
    return finishPromise;
  }

  async function abort(reason) {
    if (closed) return;
    terminalError ??= toError(reason ?? new Error("Browser MediaStream transcription aborted."));
    closed = true;
    releaseFlushWaiter();
    disconnectGraph();
    await closeAudioContext(audioContext);
  }

  return {
    finish,
    abort,
    get bufferedSeconds() {
      return session.bufferedSeconds;
    },
    get closed() {
      return closed;
    },
    get error() {
      return terminalError;
    },
    plan: session.plan,
    sampleRateHz: BROWSER_SAMPLE_RATE_HZ,
  };
}

export function normalizeBrowserTranscriptionOutput(output, context = {}) {
  const model = resolveBrowserModel(context.modelId);
  const text = normalizeBrowserTranscriptText(output?.text);
  const durationSeconds = finiteOrNull(context.durationSeconds);
  const offsetSeconds = finiteOrNull(context.offsetSeconds) ?? 0;
  const rawChunks = Array.isArray(output?.chunks) ? output.chunks : [];
  const chunks =
    rawChunks.length > 0
      ? rawChunks
      : [{ text, timestamp: text ? [0, durationSeconds] : [null, null] }];
  const segments = chunks
    .map((chunk, index) => {
      const segmentText = normalizeBrowserTranscriptText(chunk?.text);
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
          modelId: model.id,
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
      modelId: model.id,
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

function resolveBrowserModel(modelId) {
  const requested =
    typeof modelId === "string" && modelId.trim().length > 0
      ? modelId.trim()
      : DEFAULT_BROWSER_MODEL_ID;
  const model = BROWSER_TRANSCRIPTION_MODELS.find((candidate) => candidate.id === requested);
  if (!model) {
    throw new RangeError(
      `Unsupported browser transcription model "${requested}". Choose one of: ${BROWSER_TRANSCRIPTION_MODELS.map((candidate) => candidate.id).join(", ")}.`,
    );
  }
  return model;
}

async function acquireTranscriber(options) {
  if (!(await supportsBrowserTranscription())) {
    throw new Error("WebGPU is required for browser transcription. No CPU or server fallback is used.");
  }

  const model = resolveBrowserModel(options.modelId);
  preferredModelId = model.id;
  await disposeIdleTranscribers(model.id);

  emitProgress(options, {
    stage: "model",
    message: `Loading ${model.label} into the browser cache…`,
  });

  const onProgress = typeof options.onProgress === "function" ? options.onProgress : null;
  if (onProgress) {
    let listeners = activeModelProgress.get(model.id);
    if (!listeners) {
      listeners = new Set();
      activeModelProgress.set(model.id, listeners);
    }
    listeners.add(onProgress);
  }

  let entry = transcriberEntries.get(model.id);
  if (!entry) {
    entry = {
      activeUses: 0,
      promise: loadTranscriber(model),
    };
    transcriberEntries.set(model.id, entry);
  }
  entry.activeUses += 1;

  try {
    const transcriber = await entry.promise;
    let released = false;
    return {
      transcriber,
      async release() {
        if (released) {
          return;
        }
        released = true;
        entry.activeUses -= 1;
        if (entry.activeUses === 0 && preferredModelId !== model.id) {
          await disposeTranscriberEntry(model.id, entry);
        }
      },
    };
  } catch (error) {
    entry.activeUses -= 1;
    if (transcriberEntries.get(model.id) === entry) {
      transcriberEntries.delete(model.id);
    }
    throw error;
  } finally {
    if (onProgress) {
      const listeners = activeModelProgress.get(model.id);
      listeners?.delete(onProgress);
      if (listeners?.size === 0) {
        activeModelProgress.delete(model.id);
      }
    }
  }
}

function loadTranscriber(model) {
  return loadTransformers().then(({ pipeline }) =>
    pipeline("automatic-speech-recognition", model.id, {
      device: "webgpu",
      progress_callback: (info) => {
        const update = {
          stage: "model",
          message: modelProgressMessage(info, model.label),
          detail: info,
        };
        for (const listener of activeModelProgress.get(model.id) ?? []) {
          listener(update);
        }
      },
    }),
  );
}

async function disposeIdleTranscribers(exceptModelId) {
  for (const [modelId, entry] of transcriberEntries) {
    if (modelId !== exceptModelId && entry.activeUses === 0) {
      await disposeTranscriberEntry(modelId, entry);
    }
  }
}

async function disposeTranscriberEntry(modelId, entry) {
  if (entry.activeUses !== 0 || transcriberEntries.get(modelId) !== entry) {
    return;
  }

  transcriberEntries.delete(modelId);
  let transcriber;
  try {
    transcriber = await entry.promise;
  } catch {
    return;
  }

  if (typeof transcriber.dispose !== "function") {
    throw new Error(`Browser transcription model "${modelId}" cannot release its WebGPU pipeline.`);
  }
  await transcriber.dispose();
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

function createPcmCaptureWorkletUrl() {
  if (typeof Blob !== "function" || typeof URL?.createObjectURL !== "function") {
    throw new Error("Browser MediaStream transcription requires Blob URL support.");
  }

  const source = `
class MoenarchAudioAnalysisPcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const configured = options?.processorOptions?.chunkFrames;
    this.chunkFrames = Number.isInteger(configured) && configured > 0 ? configured : ${PCM_CAPTURE_CHUNK_FRAMES};
    this.buffer = new Float32Array(this.chunkFrames);
    this.offset = 0;
    this.port.onmessage = (event) => {
      if (event.data?.type === "flush") {
        this.flush();
      }
    };
  }

  emitBuffer(buffer) {
    this.port.postMessage({ type: "pcm", samples: buffer }, [buffer.buffer]);
  }

  flush() {
    if (this.offset > 0) {
      this.emitBuffer(this.buffer.slice(0, this.offset));
      this.buffer = new Float32Array(this.chunkFrames);
      this.offset = 0;
    }
    this.port.postMessage({ type: "flushed" });
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels || channels.length === 0 || channels[0].length === 0) {
      return true;
    }

    const frames = channels[0].length;
    for (let frame = 0; frame < frames; frame += 1) {
      let mixed = 0;
      for (let channel = 0; channel < channels.length; channel += 1) {
        mixed += channels[channel][frame] ?? 0;
      }
      this.buffer[this.offset] = mixed / channels.length;
      this.offset += 1;
      if (this.offset === this.chunkFrames) {
        this.emitBuffer(this.buffer);
        this.buffer = new Float32Array(this.chunkFrames);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("${PCM_CAPTURE_PROCESSOR_NAME}", MoenarchAudioAnalysisPcmCapture);
`;
  return URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
}

function requireBrowserAudioTrack(stream) {
  if (!stream || typeof stream.getAudioTracks !== "function") {
    throw new TypeError("Browser MediaStream transcription requires a caller-acquired MediaStream.");
  }
  const track = stream.getAudioTracks()[0];
  if (!track) {
    throw new Error("The caller-acquired MediaStream contains no audio track.");
  }
  if (track.readyState === "ended") {
    throw new Error("The caller-acquired MediaStream audio track has already ended.");
  }
  return track;
}

async function closeAudioContext(audioContext) {
  if (!audioContext || audioContext.state === "closed") return;
  try {
    await audioContext.close();
  } catch {
    // Cleanup must not hide the transcription result or the original failure.
  }
}

function validatePcmSamples(samples) {
  if (!(samples instanceof Float32Array) || samples.length === 0) {
    throw new TypeError("audio-analysis browser transcription requires non-empty Float32Array PCM.");
  }
  if (samples.some((sample) => !Number.isFinite(sample))) {
    throw new TypeError("audio-analysis browser transcription PCM samples must be finite.");
  }
}

function modelProgressMessage(info, modelLabel) {
  if (info && typeof info === "object" && info.status === "progress") {
    const file = typeof info.file === "string" ? ` (${shortFileName(info.file)})` : "";
    return `Downloading/caching ${modelLabel} assets${file}…`;
  }
  if (info && typeof info === "object" && info.status === "done") {
    return `${modelLabel} assets ready. Preparing WebGPU inference…`;
  }
  return `Preparing ${modelLabel} for WebGPU transcription…`;
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

function normalizeBrowserTranscriptText(value) {
  return String(value ?? "")
    .replace(/(?:\.\s*){6,}/g, " ")
    .replace(/(?:…\s*){3,}/g, " ")
    .replace(/\s+/g, " ")
    .trim();
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

function toError(value) {
  if (value instanceof Error) return value;
  return new Error(String(value));
}

function formatError(value) {
  return value instanceof Error ? value.message : String(value);
}
