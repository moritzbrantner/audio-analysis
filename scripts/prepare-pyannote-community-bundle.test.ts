import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, readdir, rm, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, test } from "bun:test";
import { reconcileBundle, verifyBundle } from "./prepare-pyannote-community-bundle.ts";

const roots: string[] = [];
const encode = (value: string): Uint8Array<ArrayBuffer> => new TextEncoder().encode(value);
const digest = (value: Uint8Array) => createHash("sha256").update(value).digest("hex");

async function sandbox() {
  const root = await mkdtemp(join(tmpdir(), "audio-analysis-pyannote-bundle-test-"));
  roots.push(root);
  return { root, destination: join(root, "cache", "approved") };
}

afterEach(async () => {
  for (const root of roots.splice(0)) {
    await rm(root, { recursive: true, force: true });
  }
});

test("first run installs once; a verified rerun performs zero mutations or model downloads", async () => {
  const { destination } = await sandbox();
  const content = encode("approved fixture");
  const hashes = { "model.onnx": digest(content) };
  let loaderCalls = 0;
  const createFiles = async () => {
    loaderCalls++;
    return { "model.onnx": content };
  };

  expect(await reconcileBundle(destination, hashes, createFiles)).toBe("changed");
  const before = await stat(join(destination, "model.onnx"));
  expect(await reconcileBundle(destination, hashes, createFiles)).toBe("unchanged");
  const after = await stat(join(destination, "model.onnx"));
  expect(loaderCalls).toBe(1);
  expect(after.mtimeMs).toBe(before.mtimeMs);
  expect(await readFile(join(destination, "model.onnx"))).toEqual(Buffer.from(content));
});

test("offline cache hit does not call resource loading", async () => {
  const { destination } = await sandbox();
  const bytes = encode("verified");
  await mkdir(destination, { recursive: true });
  await writeFile(join(destination, "model.onnx"), bytes);
  expect(
    await reconcileBundle(destination, { "model.onnx": digest(bytes) }, async () => {
      throw new Error("should not load");
    }),
  ).toBe("unchanged");
});

test("corrupted, incomplete, extra-file and symlink bundles fail closed", async () => {
  const bytes = encode("approved");
  const hashes = { "model.onnx": digest(bytes) };
  for (const kind of ["corrupted", "incomplete", "extra", "symlink"]) {
    const { root, destination } = await sandbox();
    await mkdir(destination, { recursive: true });
    if (kind === "corrupted") {
      await writeFile(join(destination, "model.onnx"), "different");
    } else if (kind === "extra") {
      await writeFile(join(destination, "model.onnx"), bytes);
      await writeFile(join(destination, "unapproved.bin"), "unexpected");
    } else if (kind === "symlink") {
      const other = join(root, "other.onnx");
      await writeFile(other, bytes);
      await symlink(other, join(destination, "model.onnx"));
    }
    let called = false;
    await expect(
      reconcileBundle(destination, hashes, async () => {
        called = true;
        return { "model.onnx": bytes };
      }),
    ).rejects.toThrow("Move it aside manually");
    expect(called).toBe(false);
    if (kind === "corrupted") {
      expect(await readFile(join(destination, "model.onnx"), "utf8")).toBe("different");
    }
  }
});

test("failed creation leaves no published directory or staging residue; retry converges", async () => {
  const { destination, root } = await sandbox();
  const bytes = encode("approved");
  const hashes = { "model.onnx": digest(bytes) };
  await expect(
    reconcileBundle(destination, hashes, async () => ({ "model.onnx": encode("bad") })),
  ).rejects.toThrow("nothing was installed");
  await expect(verifyBundle(destination, hashes)).rejects.toThrow();
  expect(await readdir(root)).toEqual([]);

  await expect(
    reconcileBundle(destination, hashes, async () => {
      throw new Error("upstream gated model unavailable");
    }),
  ).rejects.toThrow("upstream gated model unavailable");
  expect(await readdir(root)).toEqual([]);
  expect(
    await reconcileBundle(destination, hashes, async () => ({ "model.onnx": bytes })),
  ).toBe("changed");
  expect((await readdir(join(root, "cache"))).sort()).toEqual(["approved"]);
});

test("refuses broad destination directories and invalid inventory names", async () => {
  const { destination } = await sandbox();
  await expect(reconcileBundle("/", { file: "value" }, async () => ({}))).rejects.toThrow(
    "broad directory",
  );
  await expect(
    reconcileBundle(destination, { "../escape": "value" }, async () => ({})),
  ).rejects.toThrow("Invalid approved bundle file inventory");
});
