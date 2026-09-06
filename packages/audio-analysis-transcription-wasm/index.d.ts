export interface SurfaceRequest {
  operation: string;
  input: unknown;
}

export interface SurfaceOperation {
  id: string;
  name: string;
  description?: string;
  inputSchema: unknown;
  outputSchema: unknown;
  exampleRequest: unknown;
  wasmSupported: boolean;
  serverSupported: boolean;
}

export interface PackageSurface {
  library: string;
  version: string;
  operations: SurfaceOperation[];
  capabilities: unknown;
}

export interface SurfaceResponse {
  operation: string;
  value: unknown;
  diagnostics: unknown[];
  artifacts: unknown[];
}

export interface BrowserTranscriptionProgress {
  stage: "decode" | "model" | "transcribe";
  message: string;
  detail?: unknown;
}

export interface BrowserTranscriptionOptions {
  source?: string;
  durationSeconds?: number;
  onProgress?: (progress: BrowserTranscriptionProgress) => void;
}

export interface BrowserTranscriptionSegment {
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
}

export interface BrowserTranscriptionResult {
  text: string;
  language: string | null;
  segments: BrowserTranscriptionSegment[];
  source: string;
  attributes: Record<string, string>;
}

export function init(): Promise<unknown>;
export function packageSurface(): Promise<PackageSurface>;
export function runOperation(request: SurfaceRequest): Promise<SurfaceResponse>;
export function browserTranscriptionCapabilities(): Record<string, unknown>;
export function supportsBrowserTranscription(): Promise<boolean>;
export function transcribeAudioBlob(
  source: Blob,
  options?: BrowserTranscriptionOptions,
): Promise<BrowserTranscriptionResult>;
export function transcribeAudioSamples(
  samples: Float32Array,
  options?: BrowserTranscriptionOptions,
): Promise<BrowserTranscriptionResult>;
export function normalizeBrowserTranscriptionOutput(
  output: unknown,
  context?: { durationSeconds?: number; source?: string },
): BrowserTranscriptionResult;
