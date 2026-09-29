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

export type BrowserAudioDecodeOptions = {
  sampleRateHz?: number;
};

export type BrowserDecodedAudio = {
  samples: Float32Array;
  sampleRateHz: number;
  channels: 1;
  durationSeconds: number;
  sourceSampleRateHz: number;
  sourceChannels: number;
};

export type BrowserAudioDecodeCapabilities = {
  runtime: "web-audio";
  acceptedSources: ["Blob"];
  output: {
    channels: 1;
    sampleFormat: "f32";
    defaultSampleRateHz: number;
  };
  fallbacks: {
    server: false;
    python: false;
  };
};

export function init(): Promise<unknown>;
export function packageSurface(): Promise<PackageSurface>;
export function runOperation(request: SurfaceRequest): Promise<SurfaceResponse>;
export function browserAudioDecodeCapabilities(): BrowserAudioDecodeCapabilities;
export function supportsBrowserAudioDecode(): boolean;
export function decodeBrowserAudioBlob(
  source: Blob,
  options?: BrowserAudioDecodeOptions,
): Promise<BrowserDecodedAudio>;
