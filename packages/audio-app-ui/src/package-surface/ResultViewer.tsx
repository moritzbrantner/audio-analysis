import { useMemo, useState } from "react";

import { Badge, Button, CopyButton, EmptyState, StatCard } from "../shared/primitives";
import type { ResultTabDefinition, SurfaceResponse } from "./types";

type ResultTab = ResultTabDefinition & { description?: string };

export function ResultViewer({
  response,
  resultTabs = [],
}: {
  response: SurfaceResponse | null;
  resultTabs?: ResultTabDefinition[];
}) {
  const tabs = useMemo<ResultTab[]>(
    () =>
      mergeTabs(
        [
          {
            id: "summary",
            label: "Summary",
            description: "Compact response summary",
            render: (value) => <ResponseSummary response={value} />,
          },
          {
            id: "json",
            label: "JSON",
            description: "Full raw response JSON",
            select: (value: SurfaceResponse) => value,
          },
          {
            id: "diagnostics",
            label: "Diagnostics",
            description: "Warnings, notes, and non-fatal operation messages",
            select: (value: SurfaceResponse) => value.diagnostics,
          },
          {
            id: "artifacts",
            label: "Artifacts",
            description: "Generated outputs and file-like references",
            select: (value: SurfaceResponse) => value.artifacts,
          },
        ],
        resultTabs,
      ),
    [resultTabs],
  );
  const [activeTab, setActiveTab] = useState(tabs[0]?.id ?? "summary");
  const tab = tabs.find((candidate) => candidate.id === activeTab) ?? tabs[0];
  const customRendered = tab?.render ? tab.render(response) : null;
  const selected = response && tab?.select ? tab.select(response) : (response ?? {});
  const rendered = JSON.stringify(selected, null, 2);

  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 pb-3">
        <div className="flex flex-wrap gap-2">
          {tabs.map((candidate) => (
            <Button
              key={candidate.id}
              className={
                activeTab === candidate.id
                  ? "rounded-md bg-zinc-950 px-3 py-2 text-sm font-semibold text-white"
                  : "rounded-md bg-zinc-100 px-3 py-2 text-sm font-semibold text-zinc-700 hover:bg-zinc-200"
              }
              aria-label={candidate.description ? `${candidate.label}: ${candidate.description}` : candidate.label}
              title={candidate.description}
              type="button"
              onClick={() => setActiveTab(candidate.id)}
            >
              {candidate.label}
            </Button>
          ))}
        </div>
        {tab?.render ? null : (
          <div className="flex gap-2">
            <CopyButton
              className="rounded-md border border-zinc-300 px-3 py-2 text-sm font-semibold"
              value={rendered}
              variant="outline"
            />
            <Button className="rounded-md border border-zinc-300 px-3 py-2 text-sm font-semibold" type="button" variant="outline" onClick={() => downloadJson(rendered)}>
              Download
            </Button>
          </div>
        )}
      </div>
      {customRendered ?? (
        <pre className="mt-4 max-h-[42rem] overflow-auto rounded-md bg-zinc-950 p-4 text-sm leading-6 text-zinc-50">
          {rendered}
        </pre>
      )}
    </section>
  );
}

function mergeTabs(defaultTabs: ResultTab[], customTabs: ResultTabDefinition[]): ResultTab[] {
  const customById = new Map(customTabs.map((tab) => [tab.id, tab]));
  const merged = defaultTabs.map((tab) => ({ ...tab, ...customById.get(tab.id) }));
  const defaultIds = new Set(defaultTabs.map((tab) => tab.id));
  return [...merged, ...customTabs.filter((tab) => !defaultIds.has(tab.id))];
}

function ResponseSummary({ response }: { response: SurfaceResponse | null }) {
  if (!response) {
    return (
      <div className="mt-4">
        <EmptyState>Run an operation to see the response summary.</EmptyState>
      </div>
    );
  }

  const value = asObject(response.value);
  const title = stringValue(value.title) ?? response.operation;
  const message = stringValue(value.message);
  const summary = asObject(value.summary);
  const summaryEntries = Object.entries(summary).filter(([, entryValue]) => isDisplayableScalar(entryValue));
  const resultEntries = Object.entries(value)
    .filter(([key]) => !["title", "message", "summary", "operation", "result"].includes(key))
    .slice(0, 8);

  return (
    <div className="mt-4 space-y-5">
      <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase text-zinc-500">Operation result</p>
            <h2 className="mt-1 text-lg font-semibold text-zinc-950">{title}</h2>
            {message ? <p className="mt-2 max-w-3xl text-sm leading-6 text-zinc-600">{message}</p> : null}
          </div>
          <Badge tone={response.diagnostics.length > 0 ? "amber" : "emerald"}>
            {response.diagnostics.length > 0 ? `${response.diagnostics.length} diagnostics` : "No diagnostics"}
          </Badge>
        </div>
      </div>

      {summaryEntries.length > 0 ? (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {summaryEntries.map(([key, entryValue]) => (
            <StatCard key={key} label={humanizeKey(key)} value={formatScalar(entryValue)} detail={key} />
          ))}
        </div>
      ) : null}

      {resultEntries.length > 0 ? (
        <div className="rounded-md border border-zinc-200 bg-white">
          <div className="border-b border-zinc-200 px-4 py-3">
            <h3 className="text-sm font-semibold text-zinc-950">Response contents</h3>
          </div>
          <dl className="divide-y divide-zinc-100">
            {resultEntries.map(([key, entryValue]) => (
              <div key={key} className="grid gap-1 px-4 py-3 sm:grid-cols-[12rem_minmax(0,1fr)]">
                <dt className="text-sm font-medium text-zinc-600">{humanizeKey(key)}</dt>
                <dd className="min-w-0 break-words text-sm text-zinc-950">{summarizeValue(entryValue)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}

      {response.artifacts.length > 0 ? (
        <div className="rounded-md border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-700">
          {response.artifacts.length} artifact{response.artifacts.length === 1 ? "" : "s"} attached.
        </div>
      ) : null}
    </div>
  );
}

function downloadJson(text: string) {
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "surface-response.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function asObject(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function isDisplayableScalar(value: unknown): boolean {
  return value == null || ["string", "number", "boolean"].includes(typeof value);
}

function formatScalar(value: unknown): string {
  if (value == null) {
    return "n/a";
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}

function summarizeValue(value: unknown): string {
  if (Array.isArray(value)) {
    return `${value.length} ${value.length === 1 ? "item" : "items"}`;
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value);
    return `${keys.length} ${keys.length === 1 ? "field" : "fields"}${keys.length > 0 ? `: ${keys.slice(0, 6).join(", ")}` : ""}`;
  }
  return formatScalar(value);
}

function humanizeKey(key: string): string {
  return key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[-_.]/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
