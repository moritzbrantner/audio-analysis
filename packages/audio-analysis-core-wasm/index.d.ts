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

/** Thresholds of `captureMetrics`; omitted fields use the documented defaults. */
export interface CaptureMetricsOptions {
  /** Frame length in seconds, rounded to whole samples per channel. Default 0.02. */
  frameSeconds?: number;
  /** Absolute sample level at or above which a sample counts as clipped. Default 0.999. */
  clipLevel?: number;
  /** Frame RMS strictly below which a frame counts as no input. Default 1e-4 (about -80 dBFS). */
  noInputRms?: number;
  /** Frame RMS strictly above which a frame counts as activity. Default 0.01 (about -40 dBFS). */
  activityRms?: number;
}

export interface CaptureMetrics {
  sampleRate: number;
  channels: number;
  samplesPerChannel: number;
  durationSeconds: number;
  clippedSampleCount: number;
  clippedSampleRatio: number;
  frameSamples: number;
  frameCount: number;
  noInputSeconds: number;
  longestNoInputSeconds: number;
  activitySeconds: number;
  config: Required<CaptureMetricsOptions>;
}

export function init(): Promise<unknown>;
export function packageSurface(): Promise<PackageSurface>;
export function runOperation(request: SurfaceRequest): Promise<SurfaceResponse>;
/** Measures clipping, no-input and activity over interleaved samples normalized to [-1, 1]. */
export function captureMetrics(
  samples: Float32Array | ArrayLike<number>,
  sampleRate: number,
  channels?: number,
  options?: CaptureMetricsOptions,
): Promise<CaptureMetrics>;
