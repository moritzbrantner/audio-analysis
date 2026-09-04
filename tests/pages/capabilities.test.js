import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

const manifest = JSON.parse(readFileSync(new URL("../../site/analysis-capabilities.json", import.meta.url), "utf8"));
const appSource = readFileSync(new URL("../../site/app.js", import.meta.url), "utf8");
const html = readFileSync(new URL("../../site/index.html", import.meta.url), "utf8");

const expectedBrowserOperations = {
  "audio-analysis-core": ["audio.levels"],
  "audio-analysis-fourier": [
    "audio.fourier.spectrum",
    "audio.fourier.spectrogram",
    "audio.fourier.features",
  ],
  "audio-analysis-pitch": ["audio.pitch.estimate", "audio.pitch.key"],
  "audio-analysis-rhythm": ["audio.rhythm.analyze"],
};

describe("Audio Inspector public capability contract", () => {
  test("keeps GitHub Pages browser-only and upload-free", () => {
    expect(manifest.schemaVersion).toBe("audio-analysis-pages-capabilities/v1");
    expect(manifest.defaultRuntime).toBe("client-wasm");
    expect(manifest.privacy.githubPages).toBe("browser-only");
    expect(manifest.privacy.automaticUpload).toBe(false);
    expect(manifest.backend.githubPagesEnabled).toBe(false);
    expect(manifest.backend.capabilityProbe).toBe(
      "/api/rust/packages/audio-analysis-transcription/api/package",
    );

    expect(appSource).toContain("if (!baseUrl || isGitHubPages())");
    expect(appSource).toContain("automaticUpload: false");
  });

  test("declares exactly the browser package operations the inspector invokes", () => {
    const declared = Object.fromEntries(
      manifest.browserPackages.map((entry) => [entry.package, entry.operations]),
    );

    expect(declared).toEqual(expectedBrowserOperations);
    for (const operations of Object.values(expectedBrowserOperations)) {
      for (const operation of operations) expect(appSource).toContain(`\"${operation}\"`);
    }
  });

  test("documents bounded analysis coverage instead of claiming full-file model analysis", () => {
    expect(manifest.coverage.fileStatistics).toContain("whole decoded file");
    expect(manifest.coverage.spectralPitchKey).toContain("bounded representative center window");
    expect(manifest.coverage.rhythm).toContain("bounded representative center window");
    expect(appSource).toContain('kind: "representative-center-window"');
    expect(appSource).toContain('kind: "representative-center-window-resampled"');
  });

  test("ships the intended input, examples, export, and provenance surfaces", () => {
    expect(html).toContain('id="file-input"');
    expect(html).toContain('accept="audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.webm"');
    expect(html).toContain('data-example="tone"');
    expect(html).toContain('data-example="clicks"');
    expect(html).toContain('id="export-json"');
    expect(html).toContain('id="provenance"');
    expect(html).toContain('<script type="module" src="./app.js"></script>');
  });
});
