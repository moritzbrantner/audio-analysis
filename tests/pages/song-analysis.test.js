import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const htmlUrl = new URL("../../site/song-analysis.html", import.meta.url);
const sourceUrl = new URL("../../site/song-analysis.js", import.meta.url);
const indexUrl = new URL("../../site/index.html", import.meta.url);
const capabilitiesUrl = new URL("../../site/analysis-capabilities.json", import.meta.url);
const buildPagesUrl = new URL("../../scripts/build-pages.sh", import.meta.url);
const wasmBindingUrl = new URL(
  "../../crates/bindings/audio-analysis-rhythm-wasm/src/lib.rs",
  import.meta.url,
);
const html = readFileSync(htmlUrl, "utf8");
const source = readFileSync(sourceUrl, "utf8");
const index = readFileSync(indexUrl, "utf8");
const capabilities = JSON.parse(readFileSync(capabilitiesUrl, "utf8"));
const buildPages = readFileSync(buildPagesUrl, "utf8");
const wasmBinding = readFileSync(wasmBindingUrl, "utf8");

describe("whole-song analysis page", () => {
  test("is discoverable from the Audio Inspector", () => {
    expect(index).toContain('href="./song-analysis.html"');
    expect(html).toContain('<script type="module" src="./song-analysis.js"></script>');
  });

  test("runs the bounded whole song through a typed Rust/WASM PCM bridge", () => {
    expect(source).toContain("const MAX_TRACK_SECONDS = 15 * 60;");
    expect(source).toContain("audioBuffer.duration > MAX_TRACK_SECONDS");
    expect(source).toContain("mixAndResample(audioBuffer, analysisRate)");
    expect(source).toContain("new Float32Array(outputLength)");
    expect(source).toContain("analyzer.analyzeTrack(samples, analysisRate, {");
    expect(source).toContain("timeOffsetSeconds: 0");
    expect(source).not.toContain('operation: "audio.rhythm.analyze"');

    expect(wasmBinding).toContain("#[wasm_bindgen(js_name = analyzeTrack)]");
    expect(wasmBinding).toContain("samples: &[f32]");
    expect(wasmBinding).toContain('OperationId::new("audio.rhythm.analyze")');
    expect(buildPages).toContain("export async function analyzeTrack(samples, sampleRate, options = {})");
  });

  test("renders Rust-owned beats and sections on an interactive playback timeline", () => {
    expect(html).toContain('id="song-audio-player"');
    expect(html).toContain('id="song-timeline"');
    expect(html).toContain('role="slider"');
    expect(html).toContain('id="song-timeline-readout"');
    expect(html).toContain("Left/Right moves between beats");

    expect(source).toContain("drawBeatMarkers(context, width, height, duration, state.analysis.beats)");
    expect(source).toContain("drawSectionBoundaries(context, width, height, duration, state.analysis.sections)");
    expect(source).toContain("adjacentAnalysisTime(current, -1, event.shiftKey)");
    expect(source).toContain("adjacentAnalysisTime(current, 1, event.shiftKey)");
    expect(source).toContain("replacePlayerSource(file)");
    expect(source).toContain("nearestBeat(time)");
    expect(source).toContain("sectionAtTime(time)");
  });

  test("exposes a downloadable machine-readable song contract", () => {
    expect(html).toContain('id="song-download-json"');
    expect(source).toContain('schemaVersion = "audio-analysis-song/v1"');
    expect(source).toContain('pcmTransport: "float32array"');
    expect(source).toContain("analysis.sections");
    expect(source).toContain("analysis.beats");
    expect(source).toContain(".song-analysis.json");
    expect(capabilities.coverage.wholeSongRhythm).toContain("complete decoded track up to 15 minutes");
    expect(capabilities.outputs.songAnalysisSchema).toBe("audio-analysis-song/v1");
    expect(capabilities.outputs.songAnalysisEvents).toEqual(["beats", "downbeatEvents", "sections"]);
  });

  test("keeps the new browser module syntactically valid", () => {
    const result = spawnSync("node", ["--check", sourceUrl.pathname], { encoding: "utf8" });
    expect(result.status).toBe(0);
    expect(result.stderr).toBe("");
  });
});
