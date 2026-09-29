import { useMemo, useState } from "react";

import { Button, CopyButton, EmptyState, StatCard } from "../shared/primitives";
import { runOperation } from "./runtime";
import type { BenchmarkScenario, PackageAppConfig, RuntimeMode } from "./types";

interface BenchmarkResult {
  packageName: string;
  scenarioId: string;
  scenarioLabel: string;
  operation: string;
  runtimeMode: RuntimeMode;
  iterations: number;
  warmupIterations: number;
  totalMs: number;
  averageMs: number;
  opsPerSecond: number;
  outputCount: number | null;
  userAgent: string;
  measuredAt: string;
}

export function BenchmarkPanel({
  config,
  runtimeMode,
}: {
  config: PackageAppConfig;
  runtimeMode: RuntimeMode;
}) {
  const scenarios = config.benchmarkScenarios ?? [];
  const runnableScenarios = useMemo(
    () =>
      scenarios.filter((scenario) => {
        if (!scenario.runtimeModes?.length) {
          return true;
        }
        return scenario.runtimeModes.includes(runtimeMode);
      }),
    [runtimeMode, scenarios],
  );
  const [selectedScenarioId, setSelectedScenarioId] = useState(scenarios[0]?.id ?? "");
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const selectedScenario =
    runnableScenarios.find((scenario) => scenario.id === selectedScenarioId) ??
    runnableScenarios[0] ??
    scenarios.find((scenario) => scenario.id === selectedScenarioId) ??
    scenarios[0];
  const selectedScenarioRunnable = Boolean(
    selectedScenario &&
      (!selectedScenario.runtimeModes?.length || selectedScenario.runtimeModes.includes(runtimeMode)),
  );
  const resultJson = JSON.stringify(result ?? {}, null, 2);

  async function runBenchmark() {
    if (!selectedScenario) {
      return;
    }
    if (!selectedScenarioRunnable) {
      setError(`Scenario ${selectedScenario.label} is not configured for ${runtimeMode}.`);
      return;
    }
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const warmupIterations = selectedScenario.warmupIterations ?? 3;
      for (let index = 0; index < warmupIterations; index += 1) {
        await runOperation(config, runtimeMode, selectedScenario.operation, selectedScenario.input);
      }

      let lastValue: unknown = null;
      const start = performance.now();
      for (let index = 0; index < selectedScenario.iterations; index += 1) {
        const response = await runOperation(
          config,
          runtimeMode,
          selectedScenario.operation,
          selectedScenario.input,
        );
        lastValue = response.value;
      }
      const totalMs = performance.now() - start;
      setResult({
        packageName: config.library,
        scenarioId: selectedScenario.id,
        scenarioLabel: selectedScenario.label,
        operation: selectedScenario.operation,
        runtimeMode,
        iterations: selectedScenario.iterations,
        warmupIterations,
        totalMs,
        averageMs: totalMs / selectedScenario.iterations,
        opsPerSecond: (selectedScenario.iterations / totalMs) * 1000,
        outputCount: outputCountAtPath(lastValue, selectedScenario.outputCountPath),
        userAgent: typeof navigator === "undefined" ? "unknown" : navigator.userAgent,
        measuredAt: new Date().toISOString(),
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Benchmark failed");
    } finally {
      setRunning(false);
    }
  }

  if (scenarios.length === 0) {
    return (
      <div className="mt-4">
        <EmptyState>No benchmark scenarios are configured for this package.</EmptyState>
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-5">
      <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
          <div>
            <label className="text-xs font-semibold uppercase text-zinc-500" htmlFor="benchmark-scenario">
              Scenario
            </label>
            <select
              className="mt-2 min-h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm"
              id="benchmark-scenario"
              value={selectedScenario?.id ?? ""}
              onChange={(event) => setSelectedScenarioId(event.target.value)}
            >
              {scenarios.map((scenario) => (
                <option key={scenario.id} value={scenario.id}>
                  {scenario.label}
                </option>
              ))}
            </select>
            {selectedScenario?.description ? (
              <p className="mt-2 text-sm leading-6 text-zinc-600">{selectedScenario.description}</p>
            ) : null}
            <p className="mt-1 font-mono text-xs text-zinc-500">
              {selectedScenario?.operation ?? "operation"} · {runtimeMode}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              className="rounded-md bg-zinc-950 px-3 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
              disabled={running || !selectedScenarioRunnable}
              type="button"
              onClick={() => void runBenchmark()}
            >
              {running ? "Running" : "Run Benchmark"}
            </Button>
            <CopyButton
              className="rounded-md border border-zinc-300 px-3 py-2 text-sm font-semibold"
              value={resultJson}
              variant="outline"
            />
            <Button
              className="rounded-md border border-zinc-300 px-3 py-2 text-sm font-semibold"
              disabled={!result}
              type="button"
              variant="outline"
              onClick={() => downloadJson(resultJson)}
            >
              Export
            </Button>
          </div>
        </div>
        {!selectedScenarioRunnable ? (
          <p className="mt-3 text-sm text-amber-700">This scenario is not configured for the current runtime mode.</p>
        ) : null}
        {error ? <p className="mt-3 text-sm text-rose-700">{error}</p> : null}
      </div>

      {result ? (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            <StatCard label="Total" value={`${result.totalMs.toFixed(2)} ms`} detail="elapsed wall-clock time" />
            <StatCard label="Average" value={`${result.averageMs.toFixed(2)} ms`} detail="per operation" />
            <StatCard label="Throughput" value={result.opsPerSecond.toFixed(2)} detail="ops/sec" />
            <StatCard label="Iterations" value={String(result.iterations)} detail={`${result.warmupIterations} warmup`} />
            <StatCard label="Output Count" value={result.outputCount == null ? "n/a" : String(result.outputCount)} detail="selected path" />
            <StatCard label="Runtime" value={result.runtimeMode} detail={result.packageName} />
          </div>
          <pre className="max-h-[28rem] overflow-auto rounded-md bg-zinc-950 p-4 text-sm leading-6 text-zinc-50">
            {resultJson}
          </pre>
        </>
      ) : (
        <EmptyState>Benchmarks run only when started from this tab.</EmptyState>
      )}
    </div>
  );
}

function outputCountAtPath(value: unknown, path: string[] | undefined): number | null {
  if (!path?.length) {
    return countValue(value);
  }
  let cursor = value;
  for (const segment of path) {
    if (!cursor || typeof cursor !== "object") {
      return null;
    }
    cursor = (cursor as Record<string, unknown>)[segment];
  }
  return countValue(cursor);
}

function countValue(value: unknown): number | null {
  if (Array.isArray(value)) {
    return value.length;
  }
  if (value && typeof value === "object") {
    return Object.keys(value).length;
  }
  if (typeof value === "string") {
    return value.length;
  }
  return null;
}

function downloadJson(text: string) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "package-benchmark.json";
  anchor.click();
  URL.revokeObjectURL(url);
}
