import type {
  HealthPayload,
  LandscapeOperationContract,
  LandscapePort,
  ModelCatalogEntry,
  PackageAppConfig,
  PackageSurface,
  RuntimeMode,
  SurfaceOperation,
  SurfaceResponse,
} from "./types";

export function configuredServerBaseUrl(config: PackageAppConfig): string {
  const key = config.server?.baseUrlEnv ?? "VITE_SERVER_URL";
  const env = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env;
  return env?.[key] ?? env?.VITE_SERVER_URL ?? "http://127.0.0.1:3000";
}

export async function initializeWasmSurface(config: PackageAppConfig): Promise<PackageSurface> {
  if (!config.wasm) {
    throw new Error("No WASM runtime is configured for this package.");
  }
  await config.wasm.init();
  return config.wasm.packageSurface();
}

export async function fetchHealth(config: PackageAppConfig, mode: RuntimeMode): Promise<HealthPayload> {
  return fetchPackageJson<HealthPayload>(config, mode, "/health");
}

export async function fetchServerSurface(config: PackageAppConfig, mode: RuntimeMode): Promise<PackageSurface> {
  const metadata = await fetchPackageJson<{ library: string; version?: string; operations?: unknown[]; capabilities?: unknown }>(
    config,
    mode,
    "/api/package",
  );
  return {
    library: metadata.library,
    version: metadata.version ?? "0.1.0",
    operations: normalizeOperations(metadata.operations ?? []),
    capabilities: metadata.capabilities ?? {},
  };
}

export async function fetchModelCatalog(config: PackageAppConfig, mode: RuntimeMode): Promise<ModelCatalogEntry[]> {
  try {
    const models = await fetchPackageJson<unknown[]>(config, mode, "/api/models");
    return normalizeModelCatalog(models);
  } catch {
    return [];
  }
}

export async function runOperation(
  config: PackageAppConfig,
  mode: RuntimeMode,
  operation: string,
  input: unknown,
): Promise<SurfaceResponse> {
  if (mode === "client-wasm") {
    if (!config.wasm) {
      throw new Error("No WASM runtime is configured for this package.");
    }
    return config.wasm.runOperation({ operation, input });
  }
  const response = await fetchPackageRoute(config, mode, "/api/run", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ operation, input }),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Server returned ${response.status}`);
  }
  return response.json() as Promise<SurfaceResponse>;
}

async function fetchPackageJson<T>(config: PackageAppConfig, mode: RuntimeMode, path: string): Promise<T> {
  const response = await fetchPackageRoute(config, mode, path);
  if (!response.ok) {
    throw new Error(`Server returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function fetchPackageRoute(
  config: PackageAppConfig,
  mode: RuntimeMode,
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const serverBaseUrl = configuredServerBaseUrl(config);
  if (mode === "standalone-server") {
    const standaloneRoute = config.server?.standaloneRoute ?? "";
    return fetch(`${serverBaseUrl}${standaloneRoute}${path}`, init);
  }
  const scopedRoute = config.server?.scopedRoute ?? `/api/rust/packages/${config.library}`;
  const scopedResponse = await fetch(`${serverBaseUrl}${scopedRoute}${path}`, init);
  if (scopedResponse.status !== 404) {
    return scopedResponse;
  }
  return fetch(`${serverBaseUrl}${path}`, init);
}

function normalizeOperations(input: unknown[]): PackageSurface["operations"] {
  return input
    .filter((value): value is Record<string, unknown> => Boolean(value) && typeof value === "object")
    .map((value) => ({
      id: String(value.id ?? ""),
      name: String(value.name ?? value.id ?? "Operation"),
      description: typeof value.description === "string" ? value.description : undefined,
      inputSchema: value.inputSchema ?? value.input_schema ?? {},
      outputSchema: value.outputSchema ?? value.output_schema ?? {},
      landscape: normalizeLandscape(value.landscape ?? schemaExtension(value.inputSchema ?? value.input_schema, "xLandscape")),
      exampleRequest: value.exampleRequest ?? value.example_request ?? {},
      wasmSupported: Boolean(value.wasmSupported ?? value.wasm_supported ?? false),
      serverSupported: Boolean(value.serverSupported ?? value.server_supported ?? false),
    }))
    .filter((operation) => operation.id.length > 0);
}

export function landscapeContractForOperation(operation: SurfaceOperation | null): LandscapeOperationContract | null {
  if (!operation) {
    return null;
  }
  return operation.landscape ?? normalizeLandscape(schemaExtension(operation.inputSchema, "xLandscape")) ?? null;
}

function schemaExtension(schema: unknown, key: string): unknown {
  if (!schema || typeof schema !== "object") {
    return undefined;
  }
  return (schema as Record<string, unknown>)[key];
}

function normalizeLandscape(value: unknown): LandscapeOperationContract | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const landscape = value as Record<string, unknown>;
  const fn = landscape.function;
  if (!fn || typeof fn !== "object") {
    return undefined;
  }
  const functionValue = fn as Record<string, unknown>;
  const id = stringField(functionValue.id);
  const owner = stringField(functionValue.owner);
  if (!id || !owner) {
    return undefined;
  }
  return {
    function: {
      id,
      owner,
      inputs: normalizeLandscapePorts(functionValue.inputs),
      outputs: normalizeLandscapePorts(functionValue.outputs),
      stability: normalizeLandscapeStability(functionValue.stability),
    },
  };
}

function normalizeLandscapePorts(value: unknown): LandscapeOperationContract["function"]["inputs"] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((port): port is Record<string, unknown> => Boolean(port) && typeof port === "object")
    .map((port): LandscapePort | null => {
      const typeRef = port.typeRef ?? port.type_ref;
      const normalizedTypeRef = normalizeLandscapeTypeRef(typeRef);
      if (!normalizedTypeRef) {
        return null;
      }
      return {
        name: String(port.name ?? ""),
        typeRef: normalizedTypeRef,
        required: Boolean(port.required ?? true),
        cardinality: normalizeLandscapeCardinality(port.cardinality),
      };
    })
    .filter((port): port is LandscapePort => port !== null && port.name.length > 0);
}

function normalizeLandscapeTypeRef(value: unknown): LandscapeOperationContract["function"]["inputs"][number]["typeRef"] | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const typeRef = value as Record<string, unknown>;
  const id = stringField(typeRef.id);
  const owner = stringField(typeRef.owner);
  if (!id || !owner) {
    return null;
  }
  return {
    id,
    owner,
    rustType: stringField(typeRef.rustType ?? typeRef.rust_type) ?? null,
    schemaRef: stringField(typeRef.schemaRef ?? typeRef.schema_ref) ?? null,
  };
}

function normalizeLandscapeStability(value: unknown): LandscapeOperationContract["function"]["stability"] {
  return value === "experimental" || value === "internal" ? value : "stable";
}

function normalizeLandscapeCardinality(value: unknown): LandscapeOperationContract["function"]["inputs"][number]["cardinality"] {
  return value === "optional" || value === "many" ? value : "one";
}

export function normalizeModelCatalog(input: unknown[]): ModelCatalogEntry[] {
  return input
    .filter((value): value is Record<string, unknown> => Boolean(value) && typeof value === "object")
    .map((value) => ({
      id: String(value.id ?? value.modelId ?? value.model_id ?? ""),
      label: String(value.label ?? value.id ?? value.modelId ?? value.model_id ?? "Model"),
      task: String(value.task ?? "general"),
      runtime: normalizeRuntime(value.runtime),
      supported: Boolean(value.supported ?? false),
      loadable: Boolean(value.loadable ?? value.supported ?? false),
      fallback: stringField(value.fallback),
      requiredFeature: stringField(value.requiredFeature ?? value.required_feature),
      requiredSetup: stringField(value.requiredSetup ?? value.required_setup),
      smokeOperation: stringField(value.smokeOperation ?? value.smoke_operation),
      source: stringField(value.source ?? value.modelId ?? value.model_id),
      note: stringField(value.note),
    }))
    .filter((model) => model.id.length > 0);
}

function normalizeRuntime(value: unknown): ModelCatalogEntry["runtime"] {
  const runtime = String(value ?? "heuristic").replace("-", "_");
  if (
    runtime === "deterministic" ||
    runtime === "heuristic" ||
    runtime === "candle" ||
    runtime === "onnx" ||
    runtime === "whisper_cpp" ||
    runtime === "opencv" ||
    runtime === "comfyui" ||
    runtime === "external"
  ) {
    return runtime;
  }
  return "heuristic";
}

function stringField(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}
