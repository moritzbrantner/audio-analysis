import { useState } from "react";

import type { FileInputDefinition, FileInputSample } from "./types";

export function FileInputs({
  definitions,
  onPatch,
}: {
  definitions: FileInputDefinition[];
  onPatch: (path: string[], value: unknown) => void;
}) {
  if (definitions.length === 0) {
    return null;
  }

  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <h2 className="text-sm font-semibold text-zinc-950">Files</h2>
      <div className="mt-3 grid gap-3">
        {definitions.map((definition) => (
          <FileInputControl key={definition.id} definition={definition} onPatch={onPatch} />
        ))}
      </div>
    </section>
  );
}

function FileInputControl({
  definition,
  onPatch,
}: {
  definition: FileInputDefinition;
  onPatch: (path: string[], value: unknown) => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const encoding = definition.encoding ?? "data-url";

  async function patchBlob(blob: Blob) {
    setError(null);
    try {
      onPatch(definition.targetPath, await readBlob(blob, encoding));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to read file");
    }
  }

  async function loadSample(sample: FileInputSample) {
    setError(null);
    for (const patch of sample.patches ?? []) {
      onPatch(patch.targetPath, patch.value);
    }
    try {
      const response = await fetch(sample.url);
      if (!response.ok) {
        throw new Error(`Unable to load ${sample.label}: ${response.status}`);
      }
      await patchBlob(await response.blob());
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : `Unable to load ${sample.label}`;
      setError(sample.missingHint ? `${message}. ${sample.missingHint}` : message);
    }
  }

  return (
    <div className="grid gap-2">
      <label className="grid gap-1 text-sm font-medium text-zinc-700">
        {definition.label}
        <input
          className="block w-full text-sm text-zinc-600 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-950 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-white"
          type="file"
          accept={definition.accept}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              void patchBlob(file);
            }
          }}
        />
      </label>
      {definition.samples?.length ? (
        <div className="flex flex-wrap gap-2">
          {definition.samples.map((sample) => (
            <button
              key={sample.id}
              className="rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-xs font-semibold text-zinc-700 hover:bg-zinc-50"
              title={sample.description}
              type="button"
              onClick={() => void loadSample(sample)}
            >
              {sample.label}
            </button>
          ))}
        </div>
      ) : null}
      {error ? (
        <p className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800">
          {error}
        </p>
      ) : null}
    </div>
  );
}

async function readBlob(blob: Blob, encoding: "data-url" | "text"): Promise<string> {
  if (encoding === "text") {
    return typeof blob.text === "function" ? blob.text() : readBlobWithFileReader(blob, "text");
  }

  if (typeof blob.arrayBuffer !== "function") {
    return readBlobWithFileReader(blob, "data-url");
  }

  const bytes = new Uint8Array(await blob.arrayBuffer());
  return `data:${blob.type || "application/octet-stream"};base64,${base64Encode(bytes)}`;
}

function readBlobWithFileReader(blob: Blob, encoding: "data-url" | "text"): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("error", () => reject(reader.error ?? new Error("Unable to read file")));
    reader.addEventListener("load", () => resolve(String(reader.result ?? "")));
    if (encoding === "text") {
      reader.readAsText(blob);
    } else {
      reader.readAsDataURL(blob);
    }
  });
}

function base64Encode(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return btoa(binary);
}
