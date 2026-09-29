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
    expect(manifest.coverage.spectralPitchKey).toContain("longer 20 s window");
    expect(manifest.coverage.rhythm).toContain("bounded representative center window");
    expect(manifest.coverage.rhythm).toContain("1024/128 STFT");
    expect(appSource).toContain('kind: "representative-center-window"');
    expect(appSource).toContain('kind: "representative-center-window-resampled"');
  });

  test("keeps musical key and rhythm on the longer high-resolution music window", () => {
    expect(appSource).toContain(
      'safeRun(analyzers.pitch, "audio.pitch.key", {\n          samples: rhythmSamples,\n          sampleRate: rhythmRate,',
    );
    expect(appSource).toContain("fftSize: RHYTHM_FFT_SIZE");
    expect(appSource).toContain("hopSize: RHYTHM_HOP_SIZE");
    expect(appSource).toContain("if (targetRate < sourceRate)");
  });

  test("ships the intended input, examples, export, provenance, and usability surfaces", () => {
    expect(html).toContain('id="file-input"');
    expect(html).toContain('id="choose-file"');
    expect(html).toContain('accept="audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.webm"');
    expect(html).toContain('data-example="tone"');
    expect(html).toContain('data-example="clicks"');
    expect(html).toContain('aria-label="Report sections"');
    expect(html).toContain('id="waveform-readout"');
    expect(html).toContain('id="spectral-readout"');
    expect(html).toContain('id="technical"');
    expect(html).toContain('id="export-json"');
    expect(html).toContain('id="provenance"');
    expect(html).toContain('<script type="module" src="./app.js"></script>');
  });

  test("keeps engineering detail progressively disclosed and visualizations keyboard-operable", () => {
    expect(html).toContain('<details id="technical"');
    expect(html).toContain('id="waveform"');
    expect(html).toContain('role="slider"');
    expect(html).toContain('id="spectral-timeline"');
    expect(html).toContain('tabindex="0"');
    expect(appSource).toContain('event.key === "ArrowLeft"');
    expect(appSource).toContain('event.key === "ArrowRight"');
    expect(appSource).toContain('event.key === "Home"');
    expect(appSource).toContain('event.key === "End"');
    expect(appSource).toContain("state.analysisGeneration");
  });
});
