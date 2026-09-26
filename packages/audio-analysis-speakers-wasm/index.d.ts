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
  stage: "model" | "segment" | "embed" | "cluster";
  message: string;
  detail?: unknown;
};

export type BrowserDiarizationOptions = {
  sampleRateHz?: 16000;
  clusterThreshold?: number;
  onProgress?: (progress: BrowserDiarizationProgress) => void;
};

export type BrowserDiarizationSegment = {
  speaker: string;
  startSeconds: number;
  endSeconds: number;
  score: number | null;
};

export type BrowserDiarizationResult = {
  accepted: true;
  operation: "diarize";
  modelId: string;
  runtime: string;
  speakerCount: number;
  segments: BrowserDiarizationSegment[];
  diagnostics: string[];
};

export type BrowserDiarizationCapabilities = {
  runtime: string;
  modelProvisioning: "browser-cache";
  segmentationModelId: string;
  speakerEmbeddingModelId: string;
  input: {
    sampleRateHz: 16000;
    channels: 1;
    sampleFormat: "f32";
  };
  chunking: {
    windowSeconds: number;
  };
  clustering: {
    metric: "cosine";
    defaultThreshold: number;
  };
  backends: {
    segmentation: "wasm-q8";
    speakerEmbedding: "wasm-q8";
  };
  features: {
    diarization: true;
    overlapAwareSegmentation: true;
    globalSpeakerClustering: true;
    transcriptAssignment: true;
  };
  fallbacks: {
    server: false;
    python: false;
  };
};

export type BrowserTranscriptSegment = {
  startSeconds: number | null;
  endSeconds: number | null;
  speaker?: string | null;
  [key: string]: unknown;
};

export type BrowserTranscriptContract = {
  segments: BrowserTranscriptSegment[];
  [key: string]: unknown;
};

export function init(): Promise<unknown>;
export function packageSurface(): Promise<PackageSurface>;
export function runOperation(request: SurfaceRequest): Promise<SurfaceResponse>;
export function browserDiarizationCapabilities(): BrowserDiarizationCapabilities;
export function supportsBrowserDiarization(): Promise<boolean>;
export function diarizeBrowserAudioSamples(
  samples: Float32Array,
  options?: BrowserDiarizationOptions,
): Promise<BrowserDiarizationResult>;
export function assignBrowserDiarizationToTranscript<
  T extends BrowserTranscriptContract,
>(transcript: T, diarization: BrowserDiarizationResult): T;
