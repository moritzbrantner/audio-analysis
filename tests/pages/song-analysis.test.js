import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const htmlUrl = new URL("../../site/song-analysis.html", import.meta.url);
const sourceUrl = new URL("../../site/song-analysis.js", import.meta.url);
const indexUrl = new URL("../../site/index.html", import.meta.url);
const capabilitiesUrl = new URL("../../site/analysis-capabilities.json", import.meta.url);
const buildPagesUrl = new URL("../../scripts/build-pages.sh", import.meta.url);
const rhythmWasmBindingUrl = new URL(
  "../../crates/bindings/audio-analysis-rhythm-wasm/src/lib.rs",
  import.meta.url,
);
const pitchWasmBindingUrl = new URL(
  "../../crates/bindings/audio-analysis-pitch-wasm/src/lib.rs",
  import.meta.url,
);
const pitchKeyTimelineUrl = new URL(
  "../../crates/audio/audio-analysis-pitch/src/key/timeline.rs",
  import.meta.url,
);
const html = readFileSync(htmlUrl, "utf8");
const source = readFileSync(sourceUrl, "utf8");
const index = readFileSync(indexUrl, "utf8");
const capabilities = JSON.parse(readFileSync(capabilitiesUrl, "utf8"));
const buildPages = readFileSync(buildPagesUrl, "utf8");
const rhythmWasmBinding = readFileSync(rhythmWasmBindingUrl, "utf8");
const pitchWasmBinding = readFileSync(pitchWasmBindingUrl, "utf8");
const pitchKeyTimeline = readFileSync(pitchKeyTimelineUrl, "utf8");

describe("whole-song analysis page", () => {
  test("is discoverable from the Audio Inspector", () => {
    expect(index).toContain('href="./song-analysis.html"');
    expect(html).toContain('<script type="module" src="./song-analysis.js"></script>');
  });

  test("runs the bounded whole song through typed rhythm and key PCM bridges", () => {
    expect(source).toContain("const MAX_TRACK_SECONDS = 15 * 60;");
    expect(source).toContain("audioBuffer.duration > MAX_TRACK_SECONDS");
    expect(source).toContain("mixAndResample(audioBuffer, analysisRate)");
    expect(source).toContain("new Float32Array(outputLength)");
    expect(source).toContain("analyzer.analyzeTrack(samples, analysisRate, {");
    expect(source).toContain("keyAnalyzer.analyzeTrackKey(samples, analysisRate, {");
    expect(source).toContain("timeOffsetSeconds: 0");
    expect(source).not.toContain('operation: "audio.rhythm.analyze"');

    expect(rhythmWasmBinding).toContain("#[wasm_bindgen(js_name = analyzeTrack)]");
    expect(rhythmWasmBinding).toContain("samples: &[f32]");
    expect(rhythmWasmBinding).toContain('OperationId::new("audio.rhythm.analyze")');
    expect(pitchWasmBinding).toContain("#[wasm_bindgen(js_name = analyzeTrackKey)]");
    expect(pitchWasmBinding).toContain("samples: &[f32]");
    expect(pitchWasmBinding).toContain(
      "analyze_key_track(samples, sample_rate, harmonic_config, timeline_config)",
    );
    expect(pitchWasmBinding).not.toContain("fn key_timeline(");
    expect(pitchKeyTimeline).toContain("pub fn analyze_key_track(");
    expect(buildPages).toContain("export async function analyzeTrack(samples, sampleRate, options = {})");
    expect(buildPages).toContain("export async function analyzeTrackKey(samples, sampleRate, options = {})");
  });

  test("keeps key uncertainty and change windows explicit", () => {
    expect(html).toContain('id="song-key-summary"');
    expect(source).toContain("keyValue.dominant ?? null");
    expect(source).toContain("keyValue.timeline");
    expect(source).toContain("keyWindowAtTime(time)");
    expect(pitchWasmBinding).toContain('"timelineMinConfidence"');
    expect(pitchKeyTimeline).toContain(".filter(|estimate| estimate.confidence >= timeline_config.min_confidence)");
    expect(pitchKeyTimeline).toContain("Uncertain windows are retained with `key: null`");
    expect(capabilities.coverage.wholeSongKey).toContain("complete bounded Float32 PCM track");
    expect(capabilities.outputs.songKeySchema).toBe("audio-analysis-key-track/v1");
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
    expect(source).toContain("analysis.keyTimeline");
    expect(source).toContain(".song-analysis.json");
    expect(capabilities.coverage.wholeSongRhythm).toContain("complete decoded track up to 15 minutes");
    expect(capabilities.outputs.songAnalysisSchema).toBe("audio-analysis-song/v1");
    expect(capabilities.outputs.songAnalysisEvents).toEqual([
      "beats",
      "downbeatEvents",
      "sections",
      "keyTimeline",
    ]);
  });

  test("keeps the browser module syntactically valid", () => {
    const result = spawnSync("node", ["--check", sourceUrl.pathname], { encoding: "utf8" });
    expect(result.status).toBe(0);
    expect(result.stderr).toBe("");
  });
});
