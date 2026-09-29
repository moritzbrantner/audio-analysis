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

export type BrowserDiarizationProgress = {
  stage: "decode" | "diarize";
  message: string;
  detail?: unknown;
};

export type BrowserDiarizationOptions = {
  sampleRateHz?: number;
  durationSeconds?: number;
  windowSeconds?: number;
  hopSeconds?: number;
  vadThreshold?: number;
  clusterThreshold?: number;
  maxSpeakers?: number;
  onProgress?: (progress: BrowserDiarizationProgress) => void;
};

export type BrowserDiarizationSegment = {
  speaker: string;
  startSeconds: number;
  endSeconds: number;
  score: number;
};

export type BrowserDiarizationResult = {
  accepted: true;
  operation: "diarize";
  modelId: string;
  runtime: string;
  speakerCount: number;
  segments: BrowserDiarizationSegment[];
  attributes: Record<string, unknown>;
  diagnostics: string[];
};

export type BrowserDiarizationCapabilities = {
  runtime: string;
  modelId: string;
  modelProvisioning: "built-in";
  input: {
    sampleRateHz: 16000;
    channels: 1;
    sampleFormat: "f32";
    acceptedSources: string[];
  };
  features: {
    diarization: true;
    transcriptAssignment: true;
    speakerIdentification: false;
  };
  quality: "deterministic-baseline";
  fallbacks: {
    server: false;
    python: false;
  };
};

export function init(): Promise<unknown>;
export function packageSurface(): Promise<PackageSurface>;
export function runOperation(request: SurfaceRequest): Promise<SurfaceResponse>;
export function browserDiarizationCapabilities(): BrowserDiarizationCapabilities;
export function supportsBrowserDiarization(): Promise<boolean>;
export function diarizeAudioBlob(
  source: Blob,
  options?: BrowserDiarizationOptions,
): Promise<BrowserDiarizationResult>;
export function diarizeAudioSamples(
  samples: Float32Array,
  options?: BrowserDiarizationOptions,
): BrowserDiarizationResult;
export function assignBrowserSpeakersToSegments<T extends {
  startSeconds?: number | null;
  endSeconds?: number | null;
  speaker?: string | null;
}>(
  segments: T[],
  diarization: BrowserDiarizationResult,
): Array<T & { speaker: string | null }>;
