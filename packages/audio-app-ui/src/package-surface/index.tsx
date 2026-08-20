import { useEffect, useMemo, useState } from "react";

import { BenchmarkPanel } from "./BenchmarkPanel";
import { FileInputs } from "./FileInputs";
import { ModelSelector } from "./ModelSelector";
import {
  OperationWorkbench,
  type OperationWorkbenchGroup,
  type OperationWorkbenchScenarioGroup,
} from "./OperationWorkbench";
import { ResultViewer } from "./ResultViewer";
import { builtInVideoFileInput } from "./samples";
import {
  configuredServerBaseUrl,
  fetchHealth,
  fetchModelCatalog,
  fetchServerSurface,
  initializeWasmSurface,
  runOperation,
} from "./runtime";
import type {
  HealthPayload,
  ModelCatalogEntry,
  PackageAppConfig,
  PackageAppPreset,
  OperationGroupDefinition,
  PackageSurfaceWorkbenchContext,
  PackageSurface,
  RuntimeMode,
  SurfaceOperation,
  SurfaceResponse,
  ResultTabDefinition,
  SurfaceOperationRole,
} from "./types";

export * from "./types";
export * from "./runtime";
export * from "./samples";
export { FileInputs } from "./FileInputs";
export { ModelSelector } from "./ModelSelector";
export { OperationWorkbench } from "./OperationWorkbench";
export { ResultViewer } from "./ResultViewer";
export { BenchmarkPanel } from "./BenchmarkPanel";
export { createTextResultTabs } from "./TextResultPanels";
export type { TextOperationPresentation, TextResultTabsConfig } from "./TextResultPanels";

type LoadState = "loading" | "ready" | "error" | "disabled";

function readQuery(key: string): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return new URLSearchParams(window.location.search).get(key);
}

function initialPresetId(config: PackageAppConfig): string | null {
  const presetFromQuery = readQuery("preset");
  if (config.presets?.some((preset) => preset.id === presetFromQuery)) {
    return presetFromQuery;
  }
  return null;
}

function initialOperationId(config: PackageAppConfig): string {
  const presetFromQuery = config.presets?.find((preset) => preset.id === readQuery("preset"));
  const defaultPreset = config.presets?.find((preset) => preset.id === config.defaultPresetId);
  return presetFromQuery?.operation ?? readQuery("operation") ?? defaultPreset?.operation ?? config.defaultOperation ?? "describe";
}

export function PackageSurfaceWorkbench({ config }: { config: PackageAppConfig }) {
  const runtimeFromUrl = readRuntimeMode(config);
  const [runtimeMode, setRuntimeMode] = useState<RuntimeMode>(runtimeFromUrl);
  const [wasmState, setWasmState] = useState<LoadState>(config.wasm ? "loading" : "disabled");
  const [serverState, setServerState] = useState<LoadState>("loading");
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [surface, setSurface] = useState<PackageSurface | null>(null);
  const [models, setModels] = useState<ModelCatalogEntry[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(() => initialPresetId(config));
  const [selectedOperation, setSelectedOperation] = useState(() => initialOperationId(config));
  const [input, setInput] = useState("{}");
  const [response, setResponse] = useState<SurfaceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let selectionInitialized = false;
    const initializeSurfaceSelection = (nextSurface: PackageSurface) => {
      if (selectionInitialized) return;
      selectionInitialized = true;
      initializeSelection(nextSurface, selectedOperation, selectedPresetId, setSelectedOperation, setSelectedPresetId, setInput, config);
    };
    if (config.wasm) {
      initializeWasmSurface(config)
        .then((nextSurface) => {
          if (cancelled) return;
          setSurface((current) => current ?? nextSurface);
          initializeSurfaceSelection(nextSurface);
          setWasmState("ready");
        })
        .catch(() => {
          if (!cancelled) setWasmState("error");
        });
    }

    Promise.all([
      fetchHealth(config, "overview-server"),
      fetchServerSurface(config, "overview-server"),
      fetchModelCatalog(config, "overview-server"),
    ])
      .then(([nextHealth, nextSurface, nextModels]) => {
        if (cancelled) return;
        setHealth(nextHealth);
        setSurface((current) => current ?? nextSurface);
        initializeSurfaceSelection(nextSurface);
        setModels(nextModels);
        setSelectedModel(nextModels[0]?.id ?? "");
        setServerState(nextHealth.ok === false ? "error" : "ready");
      })
      .catch(() => {
        if (!cancelled) setServerState("error");
      });

    return () => {
      cancelled = true;
    };
  }, [config]);

  const operations = useMemo(() => orderedOperations(surface?.operations ?? [], config.featuredOperations), [surface, config.featuredOperations]);
  const operationGroups = useMemo(
    () => groupedOperations(operations, config.operationGroups),
    [operations, config.operationGroups],
  );
  const scenarioGroups = useMemo(
    () => groupedScenarioOptions(operations, operationGroups, config.presets),
    [operations, operationGroups, config.presets],
  );
  const operation = useMemo(
    () => operations.find((candidate) => candidate.id === selectedOperation) ?? operations[0] ?? null,
    [operations, selectedOperation],
  );
  const selectedScenario = selectedPresetId ? presetScenarioValue(selectedPresetId) : operationScenarioValue(selectedOperation);
  const parsedInput = useMemo(() => parseInputOrNull(input), [input]);
  const wasmAvailable = Boolean(config.wasm) && wasmState === "ready";
  const overviewServerAvailable = serverState === "ready";
  const selectedOperationRuntimeSupported = operationSupportsRuntime(operation, runtimeMode);
  const selectedRuntimeAvailable =
    runtimeMode === "client-wasm"
      ? wasmAvailable
      : runtimeMode === "overview-server"
        ? overviewServerAvailable
        : true;
  const runDisabledReason = runtimeDisabledReason(
    runtimeMode,
    wasmAvailable,
    overviewServerAvailable,
    operations.length,
    operation,
  );
  const canRun = selectedRuntimeAvailable && selectedOperationRuntimeSupported && operations.length > 0;

  useEffect(() => {
    if (operation && !selectedOperation) {
      chooseOperation(operation.id);
    }
  }, [operation, selectedOperation]);

  useEffect(() => {
    if (runtimeMode === "client-wasm" && wasmState === "error" && overviewServerAvailable) {
      chooseRuntime("overview-server");
      return;
    }
    if (runtimeMode === "client-wasm" && operation && !operation.wasmSupported && overviewServerAvailable) {
      chooseRuntime("overview-server");
      return;
    }
    if (runtimeMode === "overview-server" && serverState === "error" && wasmAvailable) {
      chooseRuntime("client-wasm");
    }
  }, [operation, overviewServerAvailable, runtimeMode, serverState, wasmAvailable, wasmState]);

  function chooseRuntime(nextMode: RuntimeMode) {
    setRuntimeMode(nextMode);
    writeQuery({ runtime: nextMode });
  }

  function chooseOperation(nextOperation: string) {
    setSelectedOperation(nextOperation);
    setSelectedPresetId(null);
    writeQuery({ operation: nextOperation, preset: null });
    const next = operations.find((candidate) => candidate.id === nextOperation);
    setInput(storedInput(config.library, nextOperation) ?? JSON.stringify(next?.exampleRequest ?? {}, null, 2));
    setResponse(null);
    setError(null);
  }

  function applyPreset(preset: PackageAppPreset) {
    setSelectedOperation(preset.operation);
    setSelectedPresetId(preset.id);
    setInput(JSON.stringify(preset.input, null, 2));
    writeQuery({ operation: preset.operation, preset: preset.id });
    setResponse(null);
    setError(null);
  }

  function chooseScenario(nextScenario: string) {
    if (nextScenario.startsWith("preset:")) {
      const presetId = nextScenario.slice("preset:".length);
      const preset = config.presets?.find((candidate) => candidate.id === presetId);
      if (preset) {
        applyPreset(preset);
      }
      return;
    }

    if (nextScenario.startsWith("operation:")) {
      chooseOperation(nextScenario.slice("operation:".length));
    }
  }

  function patchInput(path: string[], value: unknown) {
    setInput((currentInput) => {
      try {
        const parsed = JSON.parse(currentInput || "{}") as unknown;
        const patched = patchValue(parsed, path, value);
        const nextInput = JSON.stringify(patched, null, 2);
        persistDraftInput(nextInput);
        return nextInput;
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
        return currentInput;
      }
    });
  }

  function setInputValue(value: unknown) {
    const nextInput = JSON.stringify(value, null, 2);
    setInput(nextInput);
    persistDraftInput(nextInput);
  }

  function persistDraftInput(nextInput: string) {
    if (!selectedPresetId) {
      persistInput(config.library, selectedOperation, nextInput);
    }
  }

  async function run() {
    if (!canRun) {
      setError(runDisabledReason ?? "No runnable runtime is available for this package.");
      return;
    }
    setRunning(true);
    setResponse(null);
    setError(null);
    try {
      const payload = JSON.parse(input || "{}");
      persistDraftInput(input);
      const result = await runOperation(config, runtimeMode, selectedOperation, payload);
      setResponse(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Operation failed");
    } finally {
      setRunning(false);
    }
  }

  const childContext: PackageSurfaceWorkbenchContext = {
    input: parsedInput ?? {},
    inputJson: input,
    response,
    selectedOperation,
    runtimeMode,
    patchInput,
    setInput: setInputValue,
    setInputJson: (nextInput) => {
      setInput(nextInput);
      persistDraftInput(nextInput);
    },
  };
  const children = typeof config.children === "function" ? config.children(childContext) : config.children;
  const resultTabs = useMemo(
    () => benchmarkResultTabs(config, runtimeMode),
    [config, runtimeMode],
  );
  const sidePanels = {
    runtime: config.workbench?.sidePanels?.runtime ?? true,
    models: config.workbench?.sidePanels?.models ?? true,
    files: config.workbench?.sidePanels?.files ?? true,
    support: config.workbench?.sidePanels?.support ?? true,
  };
  const hasSidePanel = sidePanels.runtime || sidePanels.models || sidePanels.files || sidePanels.support;
  const layoutClass =
    config.workbench?.layout === "focused" || !hasSidePanel
      ? "mx-auto grid max-w-screen-2xl gap-5 px-5 py-5 xl:grid-cols-[minmax(420px,0.85fr)_minmax(0,1.15fr)]"
      : "mx-auto grid max-w-screen-2xl gap-5 px-5 py-5 xl:grid-cols-[minmax(380px,0.8fr)_minmax(0,1.2fr)_360px]";

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-950">
      <section className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-screen-2xl flex-col gap-4 px-5 py-5 lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">Package workbench</p>
            <h1 className="mt-1 break-words text-2xl font-semibold">{config.title}</h1>
            <p className="mt-2 max-w-4xl text-sm leading-6 text-zinc-600">{config.description}</p>
          </div>
          <RuntimeButtons
            config={config}
            operation={operation}
            runtimeMode={runtimeMode}
            serverState={serverState}
            wasmState={wasmState}
            onRuntimeMode={chooseRuntime}
          />
        </div>
      </section>

      <section className={layoutClass}>
        <div className="space-y-5">
          <OperationWorkbench
            canRun={canRun}
            error={error}
            input={input}
            operation={operation}
            operationGroups={operationGroups}
            operations={operations}
            presets={config.presets}
            running={running}
            runDisabledReason={runDisabledReason}
            selectedOperation={selectedOperation}
            inputChrome={config.workbench?.inputChrome}
            showLandscapeContract={config.workbench?.showLandscapeContract}
            visibleInputFields={config.workbench?.inputFields?.[selectedOperation]}
            onInputChange={(nextInput) => {
              setInput(nextInput);
              persistDraftInput(nextInput);
            }}
            onPreset={applyPreset}
            onRun={() => void run()}
            onSelectScenario={chooseScenario}
            onSelectOperation={chooseOperation}
            scenarioGroups={scenarioGroups}
            selectedScenario={selectedScenario}
          />
          {children}
        </div>
        <ResultViewer response={response} resultTabs={resultTabs} />
        {hasSidePanel ? (
          <aside className="space-y-5">
            {sidePanels.runtime ? (
              <RuntimePanel
                config={config}
                health={health}
                serverState={serverState}
                surface={surface}
                wasmState={wasmState}
              />
            ) : null}
            {sidePanels.models ? <ModelSelector models={models} selectedModel={selectedModel} onSelectModel={setSelectedModel} /> : null}
            {sidePanels.files ? <FileInputs definitions={config.fileInputs ?? defaultFileInputs(config.domain)} onPatch={patchInput} /> : null}
            {sidePanels.support ? <SupportPanel operations={operations} /> : null}
          </aside>
        ) : null}
      </section>
    </main>
  );
}

function benchmarkResultTabs(config: PackageAppConfig, runtimeMode: RuntimeMode): ResultTabDefinition[] | undefined {
  if (!config.benchmarkScenarios?.length) {
    return config.resultTabs;
  }
  return [
    ...(config.resultTabs ?? []),
    {
      id: "benchmarks",
      label: "Benchmarks",
      render: () => <BenchmarkPanel config={config} runtimeMode={runtimeMode} />,
    },
  ];
}

function RuntimeButtons({
  config,
  operation,
  runtimeMode,
  serverState,
  wasmState,
  onRuntimeMode,
}: {
  config: PackageAppConfig;
  operation: SurfaceOperation | null;
  runtimeMode: RuntimeMode;
  serverState: LoadState;
  wasmState: LoadState;
  onRuntimeMode: (mode: RuntimeMode) => void;
}) {
  return (
    <div className="inline-grid overflow-hidden rounded-md border border-zinc-300 bg-white sm:grid-cols-3" role="group" aria-label="Runtime mode">
      <ModeButton
        active={runtimeMode === "client-wasm"}
        disabled={!config.wasm || wasmState === "error" || operation?.wasmSupported === false}
        onClick={() => onRuntimeMode("client-wasm")}
      >
        Client WASM
      </ModeButton>
      <ModeButton
        active={runtimeMode === "overview-server"}
        disabled={serverState === "error" || operation?.serverSupported === false}
        onClick={() => onRuntimeMode("overview-server")}
      >
        Overview Server
      </ModeButton>
      <ModeButton active={runtimeMode === "standalone-server"} disabled={operation?.serverSupported === false} onClick={() => onRuntimeMode("standalone-server")}>
        Standalone Server
      </ModeButton>
    </div>
  );
}

function ModeButton(props: { active: boolean; children: string; disabled?: boolean; onClick: () => void }) {
  return (
    <button
      className={
        props.active
          ? "px-3 py-2 text-sm font-medium bg-zinc-950 text-white"
          : "px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-50"
      }
      disabled={props.disabled}
      type="button"
      onClick={props.onClick}
    >
      {props.children}
    </button>
  );
}

function RuntimePanel({
  config,
  health,
  serverState,
  surface,
  wasmState,
}: {
  config: PackageAppConfig;
  health: HealthPayload | null;
  serverState: LoadState;
  surface: PackageSurface | null;
  wasmState: LoadState;
}) {
  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <h2 className="text-sm font-semibold text-zinc-950">Runtime</h2>
      <dl className="mt-3 grid gap-3 text-sm">
        <StatusRow label="Client WASM" state={wasmState} detail="Runs the generated browser WASM package." />
        <StatusRow label="Overview Server" state={serverState} detail={`Calls ${config.server?.scopedRoute ?? "/api/rust/packages/<library>"}/api/run.`} />
        <DetailRow label="Standalone Server" value="Uses package server routes through the configured base URL." />
        <DetailRow label="Server URL" value={configuredServerBaseUrl(config)} />
        <DetailRow label="Health" value={health?.package ?? "Not loaded"} />
        <DetailRow label="Library" value={surface?.library ?? config.library} />
        <DetailRow label="Operations" value={String(surface?.operations.length ?? 0)} />
        {health?.requiredFeature ? <DetailRow label="Feature" value={health.requiredFeature} /> : null}
      </dl>
    </section>
  );
}

function SupportPanel({ operations }: { operations: SurfaceOperation[] }) {
  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <h2 className="text-sm font-semibold text-zinc-950">Support</h2>
      <ul className="mt-3 space-y-2 font-mono text-xs text-zinc-800">
        {operations.map((candidate) => (
          <li key={candidate.id} className="rounded-md bg-zinc-50 p-2">
            {candidate.id} · WASM {candidate.wasmSupported ? "yes" : "no"} · server {candidate.serverSupported ? "yes" : "no"}
          </li>
        ))}
      </ul>
    </section>
  );
}

function StatusRow({ label, state, detail }: { label: string; state: LoadState; detail?: string }) {
  return (
    <div>
      <DetailRow label={label} value={state === "ready" ? "Ready" : state === "error" ? "Unavailable" : state === "disabled" ? "Disabled" : "Loading"} />
      {detail ? <p className="mt-1 text-xs leading-5 text-zinc-500">{detail}</p> : null}
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-semibold uppercase text-zinc-500">{label}</dt>
      <dd className="mt-1 break-words font-mono text-zinc-900">{value}</dd>
    </div>
  );
}

function runtimeDisabledReason(
  runtimeMode: RuntimeMode,
  wasmAvailable: boolean,
  overviewServerAvailable: boolean,
  operationCount: number,
  operation: SurfaceOperation | null,
): string | undefined {
  if (!wasmAvailable && !overviewServerAvailable && runtimeMode !== "standalone-server") {
    return "No runnable runtime is available for this package.";
  }
  if (operationCount === 0) {
    return "No operations are available for this package.";
  }
  if (runtimeMode === "client-wasm" && !wasmAvailable) {
    return "Client WASM is unavailable. Use Overview Server or build the generated WASM package.";
  }
  if (runtimeMode === "client-wasm" && operation?.wasmSupported === false) {
    return "This operation is server-only. Use Overview Server or Standalone Server.";
  }
  if (runtimeMode === "overview-server" && !overviewServerAvailable) {
    return "Overview Server is unavailable. Start the dev server with bun run dev.";
  }
  if ((runtimeMode === "overview-server" || runtimeMode === "standalone-server") && operation?.serverSupported === false) {
    return "This operation is not supported by the selected server runtime.";
  }
  return undefined;
}

function operationSupportsRuntime(operation: SurfaceOperation | null, runtimeMode: RuntimeMode): boolean {
  if (!operation) {
    return true;
  }
  if (runtimeMode === "client-wasm") {
    return operation.wasmSupported;
  }
  return operation.serverSupported;
}

function orderedOperations(operations: SurfaceOperation[], featured?: string[]): SurfaceOperation[] {
  const originalRank = new Map(operations.map((operation, index) => [operation.id, index]));
  if (featured?.length) {
    const rank = new Map(featured.map((operation, index) => [operation, index]));
    return [...operations].sort(
      (left, right) =>
        (rank.get(left.id) ?? 999) - (rank.get(right.id) ?? 999) ||
        (originalRank.get(left.id) ?? 999) - (originalRank.get(right.id) ?? 999),
    );
  }
  return [...operations].sort(
    (left, right) =>
      roleRank(operationRole(left)) - roleRank(operationRole(right)) ||
      operationSortOrder(left) - operationSortOrder(right) ||
      (originalRank.get(left.id) ?? 999) - (originalRank.get(right.id) ?? 999),
  );
}

function groupedOperations(
  operations: SurfaceOperation[],
  groups?: OperationGroupDefinition[],
): OperationWorkbenchGroup[] | undefined {
  if (operations.length === 0) {
    return undefined;
  }
  if (!groups?.length) {
    return groupedOperationsByCuration(operations);
  }

  const byId = new Map(operations.map((operation) => [operation.id, operation]));
  const claimed = new Set<string>();
  const workbenchGroups = groups
    .map((group) => {
      const groupOperations = group.operations
        .map((operationId) => byId.get(operationId))
        .filter((operation): operation is SurfaceOperation => Boolean(operation));
      for (const operation of groupOperations) {
        claimed.add(operation.id);
      }
      return {
        id: group.id,
        label: group.label,
        description: group.description,
        operations: groupOperations,
      };
    })
    .filter((group) => group.operations.length > 0);

  const remaining = operations.filter((operation) => !claimed.has(operation.id));
  if (remaining.length > 0) {
    workbenchGroups.push({
      id: "other",
      label: "Other",
      description: undefined,
      operations: remaining,
    });
  }

  return workbenchGroups.length > 1 ? workbenchGroups : undefined;
}

function groupedOperationsByCuration(operations: SurfaceOperation[]): OperationWorkbenchGroup[] | undefined {
  const labels: Record<SurfaceOperationRole, string> = {
    workflow: "Workflow",
    support: "Support",
    debug: "Debug",
  };
  const groups: OperationWorkbenchGroup[] = (["workflow", "support", "debug"] as SurfaceOperationRole[])
    .map((role) => ({
      id: role,
      label: labels[role],
      description: undefined,
      operations: operations.filter((operation) => operationRole(operation) === role),
    }))
    .filter((group) => group.operations.length > 0);

  return groups.length > 1 ? groups : undefined;
}

function operationRole(operation: SurfaceOperation): SurfaceOperationRole {
  const role = operation.curation?.role;
  if (role === "workflow" || role === "support" || role === "debug") {
    return role;
  }
  if (operation.id === "describe") {
    return "debug";
  }
  return "workflow";
}

function operationSortOrder(operation: SurfaceOperation): number {
  return typeof operation.curation?.sortOrder === "number" ? operation.curation.sortOrder : operationRole(operation) === "debug" ? 900 : 100;
}

function roleRank(role: SurfaceOperationRole): number {
  return role === "workflow" ? 0 : role === "support" ? 1 : 2;
}

function groupedScenarioOptions(
  operations: SurfaceOperation[],
  operationGroups?: OperationWorkbenchGroup[],
  presets: PackageAppPreset[] = [],
): OperationWorkbenchScenarioGroup[] | undefined {
  if (presets.length === 0 || operations.length === 0) {
    return undefined;
  }

  const groups =
    operationGroups && operationGroups.length > 0
      ? operationGroups
      : [
          {
            id: "scenarios",
            label: "Scenarios",
            description: undefined,
            operations,
          },
        ];
  const operationsWithPresets = new Set(presets.map((preset) => preset.operation));

  return groups
    .map((group) => {
      const groupOperationIds = new Set(group.operations.map((operation) => operation.id));
      const presetOptions = presets
        .filter((preset) => groupOperationIds.has(preset.operation))
        .map((preset) => {
          const operation = group.operations.find((candidate) => candidate.id === preset.operation);
          return {
            value: presetScenarioValue(preset.id),
            kind: "preset" as const,
            label: preset.label,
            description: preset.description,
            operationId: preset.operation,
            operationName: operation?.name ?? preset.operation,
          };
        });
      const rawOperationOptions = group.operations
        .filter((operation) => isRawScenarioOperation(group.id, operation.id, operationsWithPresets))
        .map((operation) => ({
          value: operationScenarioValue(operation.id),
          kind: "operation" as const,
          label: operation.name,
          description: operation.description,
          operationId: operation.id,
          operationName: operation.name,
        }));

      return {
        id: group.id,
        label: group.label,
        description: group.description,
        options: [...presetOptions, ...rawOperationOptions],
      };
    })
    .filter((group) => group.options.length > 0);
}

function isRawScenarioOperation(groupId: string, operationId: string, operationsWithPresets: Set<string>): boolean {
  return groupId === "debug" || groupId === "support" || !operationsWithPresets.has(operationId);
}

function presetScenarioValue(presetId: string): string {
  return `preset:${presetId}`;
}

function operationScenarioValue(operationId: string): string {
  return `operation:${operationId}`;
}

function initializeSelection(
  surface: PackageSurface,
  current: string,
  currentPresetId: string | null,
  setSelectedOperation: (operation: string) => void,
  setSelectedPresetId: (preset: string | null) => void,
  setInput: (input: string) => void,
  config: PackageAppConfig,
) {
  const presetFromQuery = findValidPreset(readQuery("preset"), config.presets, surface.operations);
  if (presetFromQuery) {
    setSelectedOperation(presetFromQuery.operation);
    setSelectedPresetId(presetFromQuery.id);
    setInput(JSON.stringify(presetFromQuery.input, null, 2));
    return;
  }

  const operationFromQuery = findOperation(surface.operations, readQuery("operation"));
  if (operationFromQuery) {
    setSelectedOperation(operationFromQuery.id);
    setSelectedPresetId(null);
    setInput(storedInput(surface.library, operationFromQuery.id) ?? JSON.stringify(operationFromQuery.exampleRequest ?? {}, null, 2));
    return;
  }

  const currentPreset = findValidPreset(currentPresetId, config.presets, surface.operations);
  if (currentPreset) {
    setSelectedOperation(currentPreset.operation);
    setSelectedPresetId(currentPreset.id);
    setInput(JSON.stringify(currentPreset.input, null, 2));
    return;
  }

  const defaultPreset = findValidPreset(config.defaultPresetId, config.presets, surface.operations);
  if (defaultPreset) {
    setSelectedOperation(defaultPreset.operation);
    setSelectedPresetId(defaultPreset.id);
    setInput(JSON.stringify(defaultPreset.input, null, 2));
    return;
  }

  const operation =
    findOperation(surface.operations, config.defaultOperation) ??
    preferredOperation(surface.operations) ??
    findOperation(surface.operations, current) ??
    surface.operations[0];
  if (!operation) {
    return;
  }
  setSelectedOperation(operation.id);
  setSelectedPresetId(null);
  setInput(storedInput(surface.library, operation.id) ?? JSON.stringify(operation.exampleRequest ?? {}, null, 2));
}

function findValidPreset(
  presetId: string | null | undefined,
  presets: PackageAppPreset[] | undefined,
  operations: SurfaceOperation[],
): PackageAppPreset | undefined {
  if (!presetId) {
    return undefined;
  }
  const preset = presets?.find((candidate) => candidate.id === presetId);
  if (!preset || !operations.some((operation) => operation.id === preset.operation)) {
    return undefined;
  }
  return preset;
}

function findOperation(operations: SurfaceOperation[], operationId: string | null | undefined): SurfaceOperation | undefined {
  if (!operationId) {
    return undefined;
  }
  return operations.find((operation) => operation.id === operationId);
}

function preferredOperation(operations: SurfaceOperation[]): SurfaceOperation | undefined {
  return (
    operations.find((operation) => operation.curation?.primary) ??
    operations.find((operation) => operationRole(operation) === "workflow") ??
    operations[0]
  );
}

function readRuntimeMode(config: PackageAppConfig): RuntimeMode {
  const runtime = readQuery("runtime");
  if (runtime === "client-wasm" || runtime === "overview-server" || runtime === "standalone-server") {
    if (runtime === "client-wasm" && !config.wasm) {
      return "overview-server";
    }
    return runtime;
  }
  if (config.defaultRuntime) {
    if (config.defaultRuntime === "client-wasm" && !config.wasm) {
      return "overview-server";
    }
    return config.defaultRuntime;
  }
  return config.wasm ? "client-wasm" : "overview-server";
}

function parseInputOrNull(input: string): unknown | null {
  try {
    return JSON.parse(input || "{}");
  } catch {
    return null;
  }
}

function writeQuery(values: Record<string, string | null>) {
  if (typeof window === "undefined") {
    return;
  }
  const url = new URL(window.location.href);
  for (const [key, value] of Object.entries(values)) {
    if (value == null) {
      url.searchParams.delete(key);
    } else {
      url.searchParams.set(key, value);
    }
  }
  window.history.replaceState({}, "", url);
}

function storageKey(library: string, operation: string): string {
  return `package-workbench:${library}:${operation}`;
}

function storedInput(library: string, operation: string): string | null {
  try {
    return localStorage.getItem(storageKey(library, operation));
  } catch {
    return null;
  }
}

function persistInput(library: string, operation: string, input: string) {
  try {
    localStorage.setItem(storageKey(library, operation), input);
  } catch {
    return;
  }
}

function patchValue(input: unknown, path: string[], value: unknown): unknown {
  const root = input && typeof input === "object" && !Array.isArray(input) ? { ...(input as Record<string, unknown>) } : {};
  let cursor: Record<string, unknown> = root;
  for (const segment of path.slice(0, -1)) {
    const next = cursor[segment];
    const object = next && typeof next === "object" && !Array.isArray(next) ? { ...(next as Record<string, unknown>) } : {};
    cursor[segment] = object;
    cursor = object;
  }
  const last = path[path.length - 1];
  if (last) {
    cursor[last] = value;
  }
  return root;
}

function defaultFileInputs(domain: PackageAppConfig["domain"]) {
  if (domain === "image") {
    return [{ id: "image", label: "Image input", accept: "image/*", targetPath: ["imageDataUrl"] }];
  }
  if (domain === "audio") {
    return [{ id: "audio", label: "Audio input", accept: "audio/*", targetPath: ["audioDataUrl"] }];
  }
  if (domain === "video") {
    return [builtInVideoFileInput()];
  }
  if (domain === "comfyui") {
    return [{ id: "workflow", label: "Workflow JSON", accept: "application/json,.json", targetPath: ["workflow"], encoding: "text" as const }];
  }
  return [];
}
