import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const htmlUrl = new URL("../../site/song-analysis.html", import.meta.url);
const sourceUrl = new URL("../../site/song-analysis.js", import.meta.url);
const indexUrl = new URL("../../site/index.html", import.meta.url);
const capabilitiesUrl = new URL("../../site/analysis-capabilities.json", import.meta.url);
const html = readFileSync(htmlUrl, "utf8");
const source = readFileSync(sourceUrl, "utf8");
const index = readFileSync(indexUrl, "utf8");
const capabilities = JSON.parse(readFileSync(capabilitiesUrl, "utf8"));

describe("whole-song analysis page", () => {
  test("is discoverable from the Audio Inspector", () => {
    expect(index).toContain('href="./song-analysis.html"');
    expect(html).toContain('<script type="module" src="./song-analysis.js"></script>');
  });

  test("runs the Rust rhythm operation over an explicit bounded whole song", () => {
    expect(source).toContain("const MAX_TRACK_SECONDS = 15 * 60;");
    expect(source).toContain('operation: "audio.rhythm.analyze"');
    expect(source).toContain("timeOffsetSeconds: 0");
    expect(source).toContain("audioBuffer.duration > MAX_TRACK_SECONDS");
    expect(source).toContain("mixAndResample(audioBuffer, analysisRate)");
  });

  test("exposes a downloadable machine-readable song contract", () => {
    expect(html).toContain('id="song-download-json"');
    expect(source).toContain('schemaVersion = "audio-analysis-song/v1"');
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
