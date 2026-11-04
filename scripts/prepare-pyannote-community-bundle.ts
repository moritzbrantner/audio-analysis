#!/usr/bin/env bun

/** Reconcile the approved, locally licensed community-1 bundle for native consumers. */
import { createHash } from "node:crypto";
import { readFile, lstat, mkdir, mkdtemp, readdir, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join, parse, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { COMMUNITY_PROFILE } from "../packages/audio-analysis-speakers-wasm/community-profile.ts";

type BundleFiles = Record<string, Uint8Array<ArrayBuffer>>;
type BundleHashes = Readonly<Record<string, string>>;
type BundleStatus = "changed" | "unchanged";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestName = "pyannote_diarization_manifest.json";
const approvedHashes: BundleHashes = {
  ...COMMUNITY_PROFILE.manifest.files,
  [manifestName]: sha256(new TextEncoder().encode(COMMUNITY_PROFILE.files[manifestName])),
};

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function assertSafeDestination(destination: string): void {
  const resolved = resolve(destination);
  if (
    [parse(resolved).root, resolve(homedir()), resolve(process.cwd()), repoRoot].includes(resolved)
  ) {
    throw new Error(`Refusing to use a broad directory as a model bundle: ${resolved}`);
  }
}

function expectedNames(hashes: BundleHashes): string[] {
  const names = Object.keys(hashes).sort();
  if (
    names.length === 0 ||
    names.some((name) => name !== basename(name) || name === "." || name === "..")
  ) {
    throw new Error("Invalid approved bundle file inventory.");
  }
  return names;
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return false;
    }
    throw error;
  }
}

/** Strictly verifies the bundle layout and bytes, rejecting symlinks and extra files. */
export async function verifyBundle(directory: string, hashes: BundleHashes): Promise<void> {
  const stat = await lstat(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new Error("Bundle path must be a real directory.");
  }
  const names = expectedNames(hashes);
  const entries = (await readdir(directory)).sort();
  if (entries.length !== names.length || entries.some((name, index) => name !== names[index])) {
    throw new Error("Bundle file inventory differs from the approved snapshot.");
  }
  for (const name of names) {
    const filePath = join(directory, name);
    const fileStat = await lstat(filePath);
    if (!fileStat.isFile() || fileStat.isSymbolicLink()) {
      throw new Error(`Invalid bundle file type: ${name}`);
    }
    if (sha256(await readFile(filePath)) !== hashes[name]) {
      throw new Error(`Bundle checksum mismatch: ${name}`);
    }
  }
}

/** Reconciles once without ever modifying an existing target, even when incomplete. */
export async function reconcileBundle(
  destination: string,
  hashes: BundleHashes,
  createFiles: () => Promise<BundleFiles>,
): Promise<BundleStatus> {
  const target = resolve(destination);
  assertSafeDestination(target);
  const names = expectedNames(hashes);
  const existing = async (): Promise<boolean> => {
    if (!(await pathExists(target))) {
      return false;
    }
    try {
      await verifyBundle(target, hashes);
    } catch (error) {
      throw new Error(
        `Existing bundle at ${target} is not the approved snapshot. ` +
          `Move it aside manually before retrying. ${(error as Error).message}`,
      );
    }
    return true;
  };
  if (await existing()) {
    return "unchanged";
  }

  const files = await createFiles();
  if (
    Object.keys(files).length !== names.length ||
    names.some((name) => !(files[name] instanceof Uint8Array) || sha256(files[name]) !== hashes[name])
  ) {
    throw new Error("Produced bundle does not match the approved manifest; nothing was installed.");
  }

  await mkdir(dirname(target), { recursive: true, mode: 0o700 });
  const staging = await mkdtemp(join(dirname(target), `.${basename(target)}.staging-`));
  try {
    for (const name of names) {
      await writeFile(join(staging, name), files[name], { flag: "wx", mode: 0o600 });
    }
    await verifyBundle(staging, hashes);
    if (await existing()) {
      return "unchanged";
    }
    try {
      await rename(staging, target);
    } catch (error) {
      if (await existing()) {
        return "unchanged";
      }
      throw error;
    }
    await verifyBundle(target, hashes);
    return "changed";
  } finally {
    // This exact path was created by this invocation. Never remove a user's target.
    await rm(staging, { recursive: true, force: true });
  }
}

export function defaultBundleDirectory(): string {
  const cacheRoot = process.env.XDG_CACHE_HOME?.trim() || join(homedir(), ".cache");
  return resolve(
    cacheRoot,
    "audio-analysis",
    "pyannote-community-1",
    `sha256-${COMMUNITY_PROFILE.manifest.artifactSetSha256}`,
  );
}

function parseOutput(args: string[]): string | null {
  if (args.length === 1 && (args[0] === "--help" || args[0] === "-h")) {
    return null;
  }
  if (args.length === 0) {
    return defaultBundleDirectory();
  }
  if (args.length === 2 && args[0] === "--output" && args[1].trim()) {
    return resolve(args[1]);
  }
  throw new Error("Usage: bun scripts/prepare-pyannote-community-bundle.ts [--output DIRECTORY]");
}

async function installDependencies(): Promise<void> {
  console.error("Checking Bun dependencies with the committed lockfile…");
  const child = Bun.spawn([process.execPath, "install", "--frozen-lockfile"], {
    cwd: repoRoot,
    stdout: "pipe",
    stderr: "inherit",
  });
  const output = await new Response(child.stdout).text();
  if (output.trim()) {
    console.error(output.trim());
  }
  if ((await child.exited) !== 0) {
    throw new Error("Unable to prepare Bun dependencies from bun.lock.");
  }
}

async function main(): Promise<void> {
  const destination = parseOutput(process.argv.slice(2));
  if (destination === null) {
    console.log(
      "Usage: bun scripts/prepare-pyannote-community-bundle.ts [--output DIRECTORY]\n" +
        "On the first run, accept the community-1 Hugging Face terms and set HF_TOKEN.\n" +
        "On a verified cache hit, no token, download, or installation is needed.",
    );
    return;
  }

  let didProvision = false;
  const status = await reconcileBundle(destination, approvedHashes, async () => {
    const token = process.env.HF_TOKEN?.trim();
    if (!token) {
      throw new Error(
        "HF_TOKEN is required for first-time provisioning. Accept the model terms at " +
          "https://huggingface.co/pyannote/speaker-diarization-community-1 " +
          "and create a read token at https://huggingface.co/settings/tokens.",
      );
    }
    await installDependencies();
    const { archiveCommunityDiarizationResources, loadCommunityDiarizationResources } = await import(
      "../packages/audio-analysis-speakers-wasm/resources.ts"
    );
    const bundle = await loadCommunityDiarizationResources({
      token,
      onProgress: ({ message }) => console.error(message),
    });
    // The existing packager independently verifies model identity and every file hash.
    await archiveCommunityDiarizationResources(bundle);
    didProvision = true;
    return bundle.files;
  });
  console.error(`Pyannote community-1: ${status} (${Object.keys(approvedHashes).length} verified files).`);
  if (didProvision) {
    console.error("Licensed model files remain local and are not committed to Git.");
  }
  // Keep stdout machine-readable for: bundle_dir="$(bun scripts/...)"
  console.log(resolve(destination));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    await main();
  } catch (error) {
    console.error(`Pyannote bundle setup failed: ${(error as Error).message}`);
    process.exitCode = 1;
  }
}
