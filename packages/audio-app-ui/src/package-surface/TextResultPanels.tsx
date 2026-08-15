import type { ReactNode } from "react";

import { Badge, EmptyState, StatCard } from "../shared/primitives";
import type { ResultTabDefinition, SurfaceResponse } from "./types";

export interface TextResultTabsConfig {
  library: string;
  primaryOperations: Record<string, TextOperationPresentation>;
}

export interface TextOperationPresentation {
  title: string;
  summaryFields?: string[];
  listFields?: string[];
  objectFields?: string[];
  explanation?: (value: unknown) => string | undefined;
}

export function createTextResultTabs(config: TextResultTabsConfig): ResultTabDefinition[] {
  return [
    {
      id: "summary",
      label: "Summary",
      render: (response) => <TextResultPanel config={config} response={response} />,
    },
  ];
}

function TextResultPanel({
  config,
  response,
}: {
  config: TextResultTabsConfig;
  response: SurfaceResponse | null;
}) {
  if (!response) {
    return (
      <div className="mt-4">
        <EmptyState>Run a text operation to see the formatted result.</EmptyState>
      </div>
    );
  }

  const value = asRecord(response.value);
  const presentation = config.primaryOperations[response.operation];
  const title = stringValue(value.title) ?? presentation?.title ?? response.operation;
  const message = stringValue(value.message);
  const explanation = presentation?.explanation?.(response.value);
  const summary = asRecord(value.summary);
  const summaryEntries = summaryEntriesFor(summary, presentation?.summaryFields);
  const lists = (presentation?.listFields ?? defaultListFields(value))
    .map((field) => ({ field, value: resolveField(value, field) }))
    .filter((entry) => Array.isArray(entry.value) && entry.value.length > 0) as Array<{ field: string; value: unknown[] }>;
  const objects = (presentation?.objectFields ?? defaultObjectFields(value))
    .map((field) => ({ field, value: resolveField(value, field) }))
    .filter((entry) => isRenderableObjectSection(entry.value));

  return (
    <div className="mt-4 space-y-5">
      <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase text-zinc-500">{config.library} result</p>
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

      {explanation ? (
        <section className="rounded-md border border-sky-200 bg-sky-50 p-4">
          <h3 className="text-sm font-semibold text-sky-950">What happened</h3>
          <p className="mt-2 text-sm leading-6 text-sky-900">{explanation}</p>
        </section>
      ) : null}

      {lists.map(({ field, value }) => (
        <ListSection key={field} field={field} values={value} />
      ))}

      {objects.map(({ field, value }) => (
        <ObjectSection key={field} field={field} value={value} />
      ))}

      {summaryEntries.length === 0 && lists.length === 0 && objects.length === 0 ? (
        <ObjectSection field="result" value={asRecord(value.result).result ?? value.result ?? value} />
      ) : null}
    </div>
  );
}

function ListSection({ field, values }: { field: string; values: unknown[] }) {
  return (
    <section className="overflow-hidden rounded-md border border-zinc-200 bg-white">
      <div className="flex items-center justify-between gap-3 border-b border-zinc-200 px-4 py-3">
        <h3 className="text-sm font-semibold text-zinc-950">{humanizeKey(lastPathPart(field))}</h3>
        <Badge>{values.length} {values.length === 1 ? "item" : "items"}</Badge>
      </div>
      <ol className="divide-y divide-zinc-100">
        {values.slice(0, 12).map((entry, index) => (
          <li key={index} className="px-4 py-3 text-sm text-zinc-800">
            {renderCompactValue(entry)}
          </li>
        ))}
      </ol>
      {values.length > 12 ? (
        <p className="border-t border-zinc-100 px-4 py-3 text-xs text-zinc-500">
          Showing 12 of {values.length}. Open JSON for the full response.
        </p>
      ) : null}
    </section>
  );
}

function ObjectSection({ field, value }: { field: string; value: unknown }) {
  const object = asRecord(value);
  const entries = Object.entries(object).filter(([, entryValue]) => entryValue !== undefined);

  if (entries.length === 0) {
    return null;
  }

  return (
    <section className="overflow-hidden rounded-md border border-zinc-200 bg-white">
      <div className="border-b border-zinc-200 px-4 py-3">
        <h3 className="text-sm font-semibold text-zinc-950">{humanizeKey(lastPathPart(field))}</h3>
      </div>
      <dl className="divide-y divide-zinc-100">
        {entries.slice(0, 16).map(([key, entryValue]) => (
          <div key={key} className="grid gap-1 px-4 py-3 sm:grid-cols-[12rem_minmax(0,1fr)]">
            <dt className="text-sm font-medium text-zinc-600">{humanizeKey(key)}</dt>
            <dd className="min-w-0 break-words text-sm text-zinc-950">{renderCompactValue(entryValue)}</dd>
          </div>
        ))}
      </dl>
      {entries.length > 16 ? (
        <p className="border-t border-zinc-100 px-4 py-3 text-xs text-zinc-500">
          Showing 16 of {entries.length} fields. Open JSON for the full response.
        </p>
      ) : null}
    </section>
  );
}

function renderCompactValue(value: unknown): ReactNode {
  if (isDisplayableScalar(value)) {
    return formatScalar(value);
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "No items";
    }
    if (value.every(isDisplayableScalar)) {
      return value.map(formatScalar).join(", ");
    }
    return (
      <div className="space-y-2">
        {value.slice(0, 4).map((entry, index) => (
          <div key={index} className="rounded-md bg-zinc-50 px-3 py-2">
            {renderCompactValue(entry)}
          </div>
        ))}
        {value.length > 4 ? <p className="text-xs text-zinc-500">+{value.length - 4} more</p> : null}
      </div>
    );
  }

  const object = asRecord(value);
  const title = stringValue(object.text) ?? stringValue(object.label) ?? stringValue(object.id) ?? stringValue(object.term);
  const score = numberValue(object.score) ?? numberValue(object.similarity) ?? numberValue(object.probability) ?? numberValue(object.weight);
  const detailEntries = Object.entries(object)
    .filter(([key]) => !["text", "label", "id", "term", "score", "similarity", "probability", "weight"].includes(key))
    .slice(0, 5);

  return (
    <div className="min-w-0">
      {title ? <p className="break-words font-medium text-zinc-950">{title}</p> : null}
      {score != null ? <p className="mt-1 text-xs tabular-nums text-zinc-500">score {formatScalar(score)}</p> : null}
      {detailEntries.length > 0 ? (
        <dl className="mt-2 grid gap-1">
          {detailEntries.map(([key, entryValue]) => (
            <div key={key} className="grid gap-1 sm:grid-cols-[8rem_minmax(0,1fr)]">
              <dt className="text-xs font-medium text-zinc-500">{humanizeKey(key)}</dt>
              <dd className="min-w-0 break-words text-xs text-zinc-700">{summarizeValue(entryValue)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {!title && score == null && detailEntries.length === 0 ? (
        <pre className="max-h-48 overflow-auto rounded-md bg-zinc-950 p-3 text-xs leading-5 text-zinc-50">
          {JSON.stringify(value, null, 2)}
        </pre>
      ) : null}
    </div>
  );
}

function summaryEntriesFor(summary: Record<string, unknown>, fields?: string[]): Array<[string, unknown]> {
  if (fields?.length) {
    return fields
      .map((field): [string, unknown] => [field, resolveField(summary, field)])
      .filter(([, value]) => isDisplayableScalar(value));
  }
  return Object.entries(summary).filter(([, value]) => isDisplayableScalar(value));
}

function defaultListFields(value: Record<string, unknown>): string[] {
  return ["predictions", "answers", "keywords", "results", "segments", "tokens", "chunks", "terms", "entities", "relations", "events"]
    .filter((field) => Array.isArray(resolveField(value, field)));
}

function defaultObjectFields(value: Record<string, unknown>): string[] {
  return ["model", "metadata", "diagnostics", "embedding", "statistics", "language", "summary", "result"]
    .filter((field) => isRenderableObjectSection(resolveField(value, field)));
}

function resolveField(value: Record<string, unknown>, field: string): unknown {
  const candidates = [value, asRecord(value.result), asRecord(asRecord(value.result).result), asRecord(value.summary)];
  for (const candidate of candidates) {
    const resolved = getPath(candidate, field);
    if (resolved !== undefined) {
      return resolved;
    }
  }
  return undefined;
}

function getPath(value: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((current, segment) => {
    const object = asRecord(current);
    return Object.prototype.hasOwnProperty.call(object, segment) ? object[segment] : undefined;
  }, value);
}

function isRenderableObjectSection(value: unknown): boolean {
  return Boolean(value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length > 0);
}

function summarizeValue(value: unknown): string {
  if (isDisplayableScalar(value)) {
    return formatScalar(value);
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "No items";
    }
    if (value.every(isDisplayableScalar)) {
      return value.map(formatScalar).join(", ");
    }
    return `${value.length} ${value.length === 1 ? "item" : "items"}`;
  }
  const keys = Object.keys(asRecord(value));
  return `${keys.length} ${keys.length === 1 ? "field" : "fields"}${keys.length > 0 ? `: ${keys.slice(0, 5).join(", ")}` : ""}`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
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

function humanizeKey(key: string): string {
  return key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[-_.]/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function lastPathPart(value: string): string {
  const parts = value.split(".");
  return parts[parts.length - 1] ?? value;
}
