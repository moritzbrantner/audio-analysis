import { afterEach, expect, test } from "bun:test";
import {
  archiveCommunityDiarizationResources,
  loadCommunityDiarizationResources,
} from "../resources.js";

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

test("missing or invalid credentials fail before any model request", async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    throw new Error("unexpected request");
  };
  for (const token of ["", "hf_bad\nheader", " ", "a".repeat(4097)]) {
    await expect(loadCommunityDiarizationResources({ token })).rejects.toThrow("HF_TOKEN");
  }
  expect(calls).toBe(0);
});

test("denied licensed access fails before public model downloads without exposing credentials", async () => {
  const calls: Array<{ url: string; authorization: string | null }> = [];
  const token = "hf_private_fixture";
  globalThis.fetch = async (input, options) => {
    calls.push({
      url: String(input),
      authorization: new Headers(options?.headers).get("Authorization"),
    });
    return new Response("untrusted remote body", { status: 403 });
  };
  await expect(loadCommunityDiarizationResources({ token })).rejects.toThrow(
    "accept the pyannote community-1 model terms",
  );
  expect(calls.length).toBe(1);
  expect(new URL(calls[0]!.url).origin).toBe("https://huggingface.co");
  expect(calls[0]!.url).toContain("/pyannote/speaker-diarization-community-1/resolve/");
  expect(calls[0]!.url).not.toContain(token);
  expect(calls[0]!.authorization).toBe(`Bearer ${token}`);
});

test("corrupted licensed resources fail integrity before parsing or further acquisition", async () => {
  let calls = 0;
  const events: unknown[] = [];
  globalThis.fetch = async () => {
    calls++;
    return new Response("corrupt archive");
  };
  await expect(
    loadCommunityDiarizationResources({
      token: "hf_private_fixture",
      onProgress: (event) => events.push(event),
    }),
  ).rejects.toThrow("checksum mismatch");
  expect(calls).toBe(1);
  expect(JSON.stringify(events)).not.toContain("hf_private_fixture");
});

test("native export rejects incomplete or unsupported bundles", async () => {
  await expect(
    archiveCommunityDiarizationResources({
      modelId: "unsupported",
      sourceRevision: "",
      artifactSetSha256: "",
      files: {},
    }),
  ).rejects.toThrow("Unsupported");
});
