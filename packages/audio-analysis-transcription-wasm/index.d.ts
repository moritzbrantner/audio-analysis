export type SurfaceRequest = {
  operation: string;
  input: unknown;
};

export type SurfaceOperation = {
  id: string;
  name: string;
  description?: string;
  inputSchema: unknown;
  outputSchema: unknown;
  exampleRequest: unknown;
  wasmSupported: boolean;
  serverSupported: boolean;
};

export type PackageSurface = {
  library: string;
  version: string;
  operations: SurfaceOperation[];
  capabilities: unknown;
};

export type SurfaceResponse = {
  operation: string;
  value: unknown;
  diagnostics: unknown[];
  artifacts: unknown[];
};

export type BrowserTranscriptionProgress = {
  stage: "capture" | "decode" | "model" | "transcribe";
  message: string;
  detail?: unknown;
};

export type BrowserTranscriptionModel = {
  id: string;
  label: string;
  description: string;
};

export type BrowserTranscriptionOptions = {
  source?: string;
  durationSeconds?: number;
  modelId?: string;
  onProgress?: (progress: BrowserTranscriptionProgress) => void;
};

export type BrowserTranscriptionSegment = {
  index: number;
  startSeconds: number | null;
  endSeconds: number | null;
  text: string;
  language: string | null;
  speaker: string | null;
  confidence: number | null;
  isFinal: boolean;
  words: unknown[];
  chars: unknown[];
  attributes: Record<string, string>;
};

export type BrowserTranscriptionResult = {
  text: string;
  language: string | null;
  segments: BrowserTranscriptionSegment[];
  source: string;
  attributes: Record<string, string>;
};

export type BrowserTranscriptionWindowOptions = {
  windowSeconds?: number;
  strideSeconds?: number;
  maxBufferedSeconds?: number;
};

export type BrowserTranscriptionWindowPlan = {
  sampleRateHz: number;
  windowSeconds: number;
  strideSeconds: number;
  stepSeconds: number;
  maxBufferedSeconds: number;
  windowSamples: number;
  strideSamples: number;
  stepSamples: number;
  maxBufferedSamples: number;
};

export type BrowserTranscriptionCapabilities = {
  runtime: string;
  requiredAcceleration: "webgpu";
  modelId: string;
  models: BrowserTranscriptionModel[];
  modelProvisioning: string;
  input: {
    sampleRateHz: number;
    channels: number;
    sampleFormat: string;
    acceptedSources: string[];
  };
  features: {
    transcription: boolean;
    timedSegments: boolean;
    boundedPcmStreaming: boolean;
    mediaStreamAdapter: boolean;
    alignment: boolean;
    diarization: boolean;
    translation: boolean;
  };
  streaming: BrowserTranscriptionWindowPlan;
  fallbacks: {
    server: false;
    python: false;
    cpu: false;
  };
};

export type BrowserTranscriptionSessionOptions =
  BrowserTranscriptionOptions & BrowserTranscriptionWindowOptions;

export type BrowserTranscriptionSession = {
  push(samples: Float32Array): Promise<BrowserTranscriptionSegment[]>;
  flush(): Promise<BrowserTranscriptionResult>;
  readonly bufferedSeconds: number;
  readonly closed: boolean;
  readonly plan: BrowserTranscriptionWindowPlan;
};

export type BrowserMediaStreamTranscriptionOptions =
  BrowserTranscriptionSessionOptions & {
    onSegments?: (segments: BrowserTranscriptionSegment[]) => void;
    onError?: (error: Error) => void;
  };

export type BrowserMediaStreamTranscriptionSession = {
  finish(): Promise<BrowserTranscriptionResult>;
  abort(reason?: unknown): Promise<void>;
  readonly bufferedSeconds: number;
  readonly closed: boolean;
  readonly error: Error | null;
  readonly plan: BrowserTranscriptionWindowPlan;
  readonly sampleRateHz: number;
};

export type BrowserTranscriptionStitchOptions = {
  committedThroughSeconds?: number;
  commitUntilSeconds?: number;
  final?: boolean;
  startIndex?: number;
};

export type BrowserTranscriptionStitchResult = {
  segments: BrowserTranscriptionSegment[];
  committedThroughSeconds: number;
};

export function init(): Promise<unknown>;
export function packageSurface(): Promise<PackageSurface>;
export function runOperation(request: SurfaceRequest): Promise<SurfaceResponse>;
export function browserTranscriptionModels(): BrowserTranscriptionModel[];
export function browserTranscriptionCapabilities(): BrowserTranscriptionCapabilities;
export function browserTranscriptionWindowPlan(
  options?: BrowserTranscriptionWindowOptions,
): BrowserTranscriptionWindowPlan;
export function stitchBrowserTranscriptionWindow(
  segments: BrowserTranscriptionSegment[],
  options?: BrowserTranscriptionStitchOptions,
): BrowserTranscriptionStitchResult;
export function supportsBrowserTranscription(): Promise<boolean>;
export function transcribeAudioBlob(
  source: Blob,
  options?: BrowserTranscriptionOptions,
): Promise<BrowserTranscriptionResult>;
export function transcribeAudioSamples(
  samples: Float32Array,
  options?: BrowserTranscriptionOptions,
): Promise<BrowserTranscriptionResult>;
export function createBrowserTranscriptionSession(
  options?: BrowserTranscriptionSessionOptions,
): BrowserTranscriptionSession;
export function createBrowserMediaStreamTranscriptionSession(
  stream: MediaStream,
  options?: BrowserMediaStreamTranscriptionOptions,
): Promise<BrowserMediaStreamTranscriptionSession>;
export function normalizeBrowserTranscriptionOutput(
  output: unknown,
  context?: {
    durationSeconds?: number;
    offsetSeconds?: number;
    source?: string;
    modelId?: string;
  },
): BrowserTranscriptionResult;
