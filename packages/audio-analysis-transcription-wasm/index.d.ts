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
  stage: "capture" | "decode" | "model" | "transcribe";
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

export interface BrowserTranscriptionWindowOptions {
  windowSeconds?: number;
  strideSeconds?: number;
  maxBufferedSeconds?: number;
}

export interface BrowserTranscriptionWindowPlan {
  sampleRateHz: number;
  windowSeconds: number;
  strideSeconds: number;
  stepSeconds: number;
  maxBufferedSeconds: number;
  windowSamples: number;
  strideSamples: number;
  stepSamples: number;
  maxBufferedSamples: number;
}

export interface BrowserTranscriptionSessionOptions
  extends BrowserTranscriptionOptions,
    BrowserTranscriptionWindowOptions {}

export interface BrowserTranscriptionSession {
  push(samples: Float32Array): Promise<BrowserTranscriptionSegment[]>;
  flush(): Promise<BrowserTranscriptionResult>;
  readonly bufferedSeconds: number;
  readonly closed: boolean;
  readonly plan: BrowserTranscriptionWindowPlan;
}

export interface BrowserMediaStreamTranscriptionOptions
  extends BrowserTranscriptionSessionOptions {
  onSegments?: (segments: BrowserTranscriptionSegment[]) => void;
  onError?: (error: Error) => void;
}

export interface BrowserMediaStreamTranscriptionSession {
  finish(): Promise<BrowserTranscriptionResult>;
  abort(reason?: unknown): Promise<void>;
  readonly bufferedSeconds: number;
  readonly closed: boolean;
  readonly error: Error | null;
  readonly plan: BrowserTranscriptionWindowPlan;
  readonly sampleRateHz: number;
}

export interface BrowserTranscriptionStitchOptions {
  committedThroughSeconds?: number;
  commitUntilSeconds?: number;
  final?: boolean;
  startIndex?: number;
}

export interface BrowserTranscriptionStitchResult {
  segments: BrowserTranscriptionSegment[];
  committedThroughSeconds: number;
}

export function init(): Promise<unknown>;
export function packageSurface(): Promise<PackageSurface>;
export function runOperation(request: SurfaceRequest): Promise<SurfaceResponse>;
export function browserTranscriptionCapabilities(): Record<string, unknown>;
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
  context?: { durationSeconds?: number; offsetSeconds?: number; source?: string },
): BrowserTranscriptionResult;
