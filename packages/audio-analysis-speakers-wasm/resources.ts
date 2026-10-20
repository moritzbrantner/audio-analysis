import { gunzipSync, unzipSync, zipSync } from "fflate";
import { parse } from "npyjs";
import { COMMUNITY_PROFILE as PROFILE } from "./community-profile.js";

type ModelName = keyof typeof PROFILE.models;
export type CommunityBundle = {
  modelId: string;
  sourceRevision: string;
  artifactSetSha256: string;
  files: Record<string, Uint8Array<ArrayBuffer>>;
};
export type CommunityResourceOptions = {
  token: string;
  signal?: AbortSignal;
  onProgress?: (event: { stage: "resources"; message: string }) => void;
};

async function digest(bytes: Uint8Array<ArrayBuffer>) {
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
}
async function checked(bytes: Uint8Array<ArrayBuffer>, expected: string) {
  if ((await digest(bytes)) !== expected) {
    throw new Error("Pyannote resource checksum mismatch.");
  }
  return bytes;
}
async function download(
  url: string,
  expected: string,
  options: CommunityResourceOptions,
  authorized = false,
) {
  options.signal?.throwIfAborted();
  let response: Response;
  try {
    response = await fetch(url, {
      credentials: "omit",
      ...(options.signal ? { signal: options.signal } : {}),
      ...(authorized
        ? { headers: { Authorization: `Bearer ${options.token}` }, cache: "no-store" as const }
        : {}),
    });
  } catch {
    options.signal?.throwIfAborted();
    throw new Error("Unable to download pyannote resources. Check your network connection.");
  }
  if (!response.ok) {
    if (authorized && [401, 403].includes(response.status)) {
      throw new Error(
        "Check your HF_TOKEN and accept the pyannote community-1 model terms on Hugging Face.",
      );
    }
    throw new Error(`Pyannote resource download failed (HTTP ${response.status}).`);
  }
  return checked(new Uint8Array(await response.arrayBuffer()), expected);
}
function progress(options: CommunityResourceOptions, message: string) {
  options.signal?.throwIfAborted();
  options.onProgress?.({ stage: "resources", message });
}
function array(archive: Record<string, Uint8Array>, name: string, shape: readonly number[]) {
  const bytes = archive[`${name}.npy`];
  if (!bytes) {
    throw new Error("Missing approved PLDA array.");
  }
  const tensor = parse(Uint8Array.from(bytes).buffer);
  if (
    !(tensor.data instanceof Float32Array || tensor.data instanceof Float64Array) ||
    tensor.fortranOrder ||
    tensor.shape.length !== shape.length ||
    tensor.shape.some((value, index) => value !== shape[index]) ||
    tensor.data.some((value) => !Number.isFinite(value))
  ) {
    throw new Error("Invalid approved PLDA array.");
  }
  return Array.from(tensor.data);
}
function matrix(archive: Record<string, Uint8Array>, name: string, rows: number, columns: number) {
  const values = array(archive, name, [rows, columns]);
  return Array.from({ length: rows }, (_, row) => values.slice(row * columns, (row + 1) * columns));
}
// Preserve the approved converter's Python float representation. Final file
// hashes independently reject any numeric encoding difference for this checkpoint.
function pythonFloat(value: number) {
  if (!Number.isFinite(value)) {
    throw new Error("Invalid PLDA number.");
  }
  if (Object.is(value, -0)) {
    return "-0.0";
  }
  const magnitude = Math.abs(value);
  if (magnitude === 0) {
    return "0.0";
  }
  if (magnitude >= 1e-4 && magnitude < 1e16) {
    const result = value.toString();
    return result.includes(".") ? result : `${result}.0`;
  }
  const [mantissa, rawExponent] = value.toExponential().split("e");
  const exponent = Number(rawExponent);
  return `${mantissa}e${exponent < 0 ? "-" : "+"}${Math.abs(exponent).toString().padStart(2, "0")}`;
}
function vector(values: number[]) {
  return `[${values.map(pythonFloat).join(",")}]`;
}
function rows(values: number[][]) {
  return `[${values.map(vector).join(",")}]`;
}
function encode(value: string) {
  return new TextEncoder().encode(value);
}
function convertPlda(transformBytes: Uint8Array, modelBytes: Uint8Array) {
  const transform = unzipSync(transformBytes);
  const model = unzipSync(modelBytes);
  return {
    "plda_transform.json": encode(
      `{"schemaVersion":1,"inputDimension":256,"outputDimension":128,"mean1":${vector(array(transform, "mean1", [256]))},"mean2":${vector(array(transform, "mean2", [128]))},"lda":${rows(matrix(transform, "lda", 256, 128))}}\n`,
    ),
    "plda_model.json": encode(
      `{"schemaVersion":1,"dimension":128,"mean":${vector(array(model, "mu", [128]))},"transform":${rows(matrix(model, "tr", 128, 128))},"psi":${vector(array(model, "psi", [128]))}}\n`,
    ),
  };
}
async function reconstruct(name: ModelName, options: CommunityResourceOptions) {
  const profile = PROFILE.models[name];
  const donor = PROFILE.donors[profile.donor];
  const weights = await download(donor.url, donor.sha256, options);
  const compressed = Uint8Array.from(atob(profile.templateBase64), (char) => char.charCodeAt(0));
  await checked(compressed, profile.templateSha256);
  const template = Uint8Array.from(gunzipSync(compressed));
  if (template.length !== profile.bytes) {
    throw new Error("Invalid pyannote graph template.");
  }
  for (const part of profile.recipe.parts) {
    if (
      part.targetStart < 0 ||
      part.targetEnd > template.length ||
      part.start < 0 ||
      part.end > weights.length ||
      part.targetEnd - part.targetStart !== part.end - part.start
    ) {
      throw new Error("Invalid pyannote reconstruction span.");
    }
    template.set(weights.subarray(part.start, part.end), part.targetStart);
  }
  options.signal?.throwIfAborted();
  return checked(template, profile.sha256);
}

/** Materializes the approved native/browser bundle. Credentials are used only
 * for the two original gated PLDA requests and are never retained in the result. */
export async function loadCommunityDiarizationResources(
  options: CommunityResourceOptions,
): Promise<CommunityBundle> {
  if (
    typeof options.token !== "string" ||
    !options.token.trim() ||
    /[^\x21-\x7e]/.test(options.token) ||
    options.token.length > 4096
  ) {
    throw new Error("Enter an HF_TOKEN with access to the pyannote community-1 model.");
  }
  const source = PROFILE.manifest.source;
  const inputs = PROFILE.manifest.conversion.inputHashes;
  progress(options, "Checking licensed pyannote resources…");
  const root = `https://huggingface.co/${source.modelId}/resolve/${source.revision}/plda/`;
  const transform = await download(
    `${root}xvec_transform.npz`,
    inputs["plda/xvec_transform.npz"],
    options,
    true,
  );
  const model = await download(`${root}plda.npz`, inputs["plda/plda.npz"], options, true);
  const plda = convertPlda(transform, model);
  await checked(plda["plda_transform.json"], PROFILE.manifest.files["plda_transform.json"]);
  await checked(plda["plda_model.json"], PROFILE.manifest.files["plda_model.json"]);
  const files: CommunityBundle["files"] = { ...plda };
  for (const [name, text] of Object.entries(PROFILE.files)) {
    files[name] = encode(text);
  }
  for (const name of ["segmentation.onnx", "embedding.onnx"] as const) {
    progress(
      options,
      `Preparing pyannote ${name === "segmentation.onnx" ? "segmentation" : "speaker embedding"} model…`,
    );
    files[name] = await reconstruct(name, options);
  }
  progress(options, "Approved pyannote model bundle is ready.");
  return {
    modelId: source.modelId,
    sourceRevision: source.revision,
    artifactSetSha256: PROFILE.manifest.artifactSetSha256,
    files,
  };
}

/** Exports an independently verified, flat model bundle for a native consumer.
 * No credentials or original gated source archives are included. */
export async function archiveCommunityDiarizationResources(
  bundle: CommunityBundle,
): Promise<Uint8Array<ArrayBuffer>> {
  if (
    bundle.modelId !== PROFILE.manifest.modelId ||
    bundle.sourceRevision !== PROFILE.manifest.source.revision ||
    bundle.artifactSetSha256 !== PROFILE.manifest.artifactSetSha256
  ) {
    throw new Error("Unsupported pyannote model bundle.");
  }
  const expected: Record<string, string> = {
    ...PROFILE.manifest.files,
    "pyannote_diarization_manifest.json": await digest(
      encode(PROFILE.files["pyannote_diarization_manifest.json"]),
    ),
  };
  if (Object.keys(bundle.files).length !== Object.keys(expected).length) {
    throw new Error("Incomplete pyannote model bundle.");
  }
  const files: Record<string, Uint8Array<ArrayBuffer>> = {};
  for (const [name, hash] of Object.entries(expected)) {
    const bytes = bundle.files[name];
    if (!(bytes instanceof Uint8Array)) {
      throw new Error("Missing pyannote model resource.");
    }
    files[name] = await checked(Uint8Array.from(bytes), hash);
  }
  return Uint8Array.from(zipSync(files, { level: 0 }));
}
