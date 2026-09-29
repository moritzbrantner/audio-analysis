import type { ModelCatalogEntry } from "./types";

export function ModelSelector({
  models,
  selectedModel,
  onSelectModel,
}: {
  models: ModelCatalogEntry[];
  selectedModel: string;
  onSelectModel: (model: string) => void;
}) {
  if (models.length === 0) {
    return (
      <section className="rounded-md border border-zinc-200 bg-white p-4">
        <h2 className="text-sm font-semibold text-zinc-950">Models</h2>
        <p className="mt-2 text-sm text-zinc-500">No model presets are registered for this package.</p>
      </section>
    );
  }

  const current = models.find((model) => model.id === selectedModel) ?? models[0];

  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-zinc-950">Models</h2>
          <p className="mt-1 text-xs text-zinc-500">{modelStatusMessage(current)}</p>
        </div>
        <ModelStatusBadge model={current} />
      </div>
      <select
        className="mt-3 min-h-10 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm"
        value={current.id}
        onChange={(event) => onSelectModel(event.target.value)}
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label} - {model.task}
          </option>
        ))}
      </select>
      <dl className="mt-3 grid gap-2 text-sm">
        <div>
          <dt className="text-xs font-semibold uppercase text-zinc-500">Runtime</dt>
          <dd className="font-mono text-zinc-900">{current.runtime}</dd>
        </div>
        {current.source ? (
          <div>
            <dt className="text-xs font-semibold uppercase text-zinc-500">Source</dt>
            <dd className="font-mono text-zinc-900">{current.source}</dd>
          </div>
        ) : null}
        {current.fallback ? (
          <div>
            <dt className="text-xs font-semibold uppercase text-zinc-500">Fallback</dt>
            <dd className="font-mono text-zinc-900">{current.fallback}</dd>
          </div>
        ) : null}
        {current.requiredFeature ? (
          <div>
            <dt className="text-xs font-semibold uppercase text-zinc-500">Feature</dt>
            <dd className="font-mono text-zinc-900">{current.requiredFeature}</dd>
          </div>
        ) : null}
        {current.requiredSetup ? (
          <div>
            <dt className="text-xs font-semibold uppercase text-zinc-500">Setup</dt>
            <dd className="font-mono text-zinc-900">{current.requiredSetup}</dd>
          </div>
        ) : null}
        {current.smokeOperation ? (
          <div>
            <dt className="text-xs font-semibold uppercase text-zinc-500">Smoke</dt>
            <dd className="font-mono text-zinc-900">{current.smokeOperation}</dd>
          </div>
        ) : null}
        {current.note ? <dd className="text-xs leading-5 text-zinc-500">{current.note}</dd> : null}
      </dl>
    </section>
  );
}

function modelStatusMessage(model: ModelCatalogEntry): string {
  if (model.loadable) {
    return "This runtime can be used by the selected server/runtime path.";
  }
  if (model.supported) {
    return "Registered and supported, but may require setup before execution.";
  }
  return "Catalog metadata only; this page will use the fallback or deterministic operation.";
}

function ModelStatusBadge({ model }: { model: ModelCatalogEntry }) {
  if (model.loadable) {
    return (
      <span className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-800">
        Loadable
      </span>
    );
  }
  if (model.supported) {
    return (
      <span className="rounded-md border border-sky-200 bg-sky-50 px-2 py-1 text-xs font-semibold text-sky-800">
        Supported
      </span>
    );
  }
  return (
    <span className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-xs font-semibold text-amber-800">
      Reference
    </span>
  );
}
