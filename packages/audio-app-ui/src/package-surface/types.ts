import type { ReactNode } from "react";

export type RuntimeMode = "client-wasm" | "overview-server" | "standalone-server";

export type PackageDomain =
  | "text"
  | "audio"
  | "image"
  | "video"
  | "vector"
  | "three-d"
  | "comfyui"
  | "data"
  | "math"
  | "runtime"
  | "jobs"
  | "support"
  | "animation";

export interface SurfaceOperation {
  id: string;
  name: string;
  description?: string;
  curation?: SurfaceOperationCuration;
  inputSchema: unknown;
  outputSchema: unknown;
  landscape?: LandscapeOperationContract;
  exampleRequest: unknown;
  wasmSupported: boolean;
  serverSupported: boolean;
}

export type SurfaceOperationRole = "workflow" | "debug" | "support";

export interface SurfaceOperationCuration {
  role: SurfaceOperationRole;
  primary: boolean;
  sortOrder: number;
}

export interface LandscapeOperationContract {
  function: LandscapeFunction;
}

export interface LandscapeFunction {
  id: string;
  owner: string;
  inputs: LandscapePort[];
  outputs: LandscapePort[];
  stability: "stable" | "experimental" | "internal";
}

export interface LandscapePort {
  name: string;
  typeRef: LandscapeTypeRef;
  required: boolean;
  cardinality: "one" | "optional" | "many";
}

export interface LandscapeTypeRef {
  id: string;
  owner: string;
  rustType?: string | null;
  schemaRef?: string | null;
}

export interface PackageSurface {
  library: string;
  version: string;
  operations: SurfaceOperation[];
  capabilities: unknown;
}

export interface SurfaceRequest {
  operation: string;
  input: unknown;
}

export interface SurfaceResponse {
  operation: string;
  value: unknown;
  diagnostics: unknown[];
  artifacts: unknown[];
}

export interface HealthPayload {
  ok: boolean;
  package: string;
  library: string;
  domain?: string;
  linked?: boolean;
  requiredFeature?: string | null;
}

export type ModelRuntime =
  | "deterministic"
  | "heuristic"
  | "candle"
  | "onnx"
  | "whisper_cpp"
  | "opencv"
  | "comfyui"
  | "external";

export interface ModelCatalogEntry {
  id: string;
  label: string;
  task: string;
  runtime: ModelRuntime;
  supported: boolean;
  loadable: boolean;
  fallback?: string;
  requiredFeature?: string;
  requiredSetup?: string;
  smokeOperation?: string;
  source?: string;
  note?: string;
}

export interface BenchmarkScenario {
  id: string;
  label: string;
  description?: string;
  operation: string;
  input: unknown;
  iterations: number;
  warmupIterations?: number;
  runtimeModes?: RuntimeMode[];
  outputCountPath?: string[];
}

export interface PackageAppPreset {
  id: string;
  label: string;
  operation: string;
  input: unknown;
  description?: string;
}

export interface OperationGroupDefinition {
  id: string;
  label: string;
  description?: string;
  operations: string[];
}

export interface ResultTabDefinition {
  id: string;
  label: string;
  select?: (response: SurfaceResponse) => unknown;
  render?: (response: SurfaceResponse | null) => ReactNode;
}

export interface FileInputSample {
  id: string;
  label: string;
  url: string;
  description?: string;
  missingHint?: string;
  patches?: FileInputPatch[];
}

export interface FileInputPatch {
  targetPath: string[];
  value: unknown;
}

export interface FileInputDefinition {
  id: string;
  label: string;
  accept?: string;
  targetPath: string[];
  encoding?: "data-url" | "text";
  samples?: FileInputSample[];
}

export interface PackageWorkbenchPresentation {
  layout?: "standard" | "focused";
  sidePanels?: {
    runtime?: boolean;
    models?: boolean;
    files?: boolean;
    support?: boolean;
  };
  inputFields?: Record<string, string[]>;
  inputChrome?: "full" | "compact";
  showLandscapeContract?: boolean;
}

export interface PackageAppConfig {
  library: string;
  title: string;
  description: string;
  domain: PackageDomain;
  wasm?: {
    init: () => Promise<unknown>;
    packageSurface: () => Promise<PackageSurface> | PackageSurface;
    runOperation: (request: SurfaceRequest) => Promise<SurfaceResponse> | SurfaceResponse;
  };
  server?: {
    baseUrlEnv?: string;
    scopedRoute: `/api/rust/packages/${string}`;
    standaloneRoute?: "";
  };
  featuredOperations?: string[];
  operationGroups?: OperationGroupDefinition[];
  defaultOperation?: string;
  defaultPresetId?: string;
  defaultRuntime?: RuntimeMode;
  presets?: PackageAppPreset[];
  benchmarkScenarios?: BenchmarkScenario[];
  resultTabs?: ResultTabDefinition[];
  fileInputs?: FileInputDefinition[];
  workbench?: PackageWorkbenchPresentation;
  children?: ReactNode | ((context: PackageSurfaceWorkbenchContext) => ReactNode);
}

export interface PackageSurfaceWorkbenchContext {
  input: unknown;
  inputJson: string;
  response: SurfaceResponse | null;
  selectedOperation: string;
  runtimeMode: RuntimeMode;
  patchInput: (path: string[], value: unknown) => void;
  setInput: (value: unknown) => void;
  setInputJson: (value: string) => void;
}
