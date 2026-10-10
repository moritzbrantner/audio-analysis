// Acceptance for audio-analysis#138: the main inspector waveform presents the whole decoded file as
// a musical timeline built from the Rust-owned whole-song contracts (`audio-analysis-song/v1`
// sections/beats/tempo map and the `audio-analysis-key-track/*` key segments), instead of the
// bounded representative rhythm window.
//
// Real browser, real Rust/WASM analyzers, deterministic synthetic fixtures. The assertions use the
// rendered inspector (structure rail segments, labels, beat markers, context HUD, audio element)
// and the downloaded JSON report; they do not depend on where the implementation stores the
// whole-track evidence inside the report.
import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { contrastingSong, songDurationSeconds, songWav, steadySong } from "./fixtures/whole-track-songs.mjs";

const baseUrl = process.env.PAGES_E2E_BASE_URL ?? "http://127.0.0.1:4173";
const artifactDir = process.env.PAGES_E2E_ARTIFACT_DIR ?? "";
const ANALYSIS_TIMEOUT_MS = 120_000;
const TIME_TOLERANCE_SECONDS = 0.05;
// Rust places A→B and B→A′ on the downbeats nearest 24 s and 48 s.
const EXPECTED_BOUNDARIES_SECONDS = [24, 48];
const SEMANTIC_SECTION_NAMES = /\b(intro|verse|pre-?chorus|chorus|bridge|drop|breakdown|outro)\b/i;

const browser = await chromium.launch({ headless: true });

try {
  const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.route("http://127.0.0.1:3000/**", (route) =>
    route.fulfill({
      status: 404,
      contentType: "application/json",
      headers: { "Access-Control-Allow-Origin": "*" },
      body: JSON.stringify({ error: "backend intentionally unavailable in browser acceptance test" }),
    }),
  );

  // ---------------------------------------------------------------------------------------------
  // Journey 1: A (120 BPM, C major) → B (150 BPM, F# major) → A again, 72 s.
  // ---------------------------------------------------------------------------------------------
  const contrast = await analyzeSong(page, contrastingSong);
  const { duration, song, keyTrack } = contrast;

  // Fixture guard: the Rust analyzers themselves see A/B/A with the expected sustained changes.
  assert.deepEqual(
    song.sections.map((section) => section.identity),
    ["A", "B", "A"],
    `Rust song contract should detect A/B/A; got ${JSON.stringify(song.sections)}`,
  );
  song.sections.slice(1).forEach((section, index) =>
    assertNear(section.startSeconds, EXPECTED_BOUNDARIES_SECONDS[index], 1, `Rust boundary ${index + 1}`),
  );

  // Whole-file coverage with contiguous regions taken from the Rust sections.
  const regions = await sectionRegions(page);
  assert.equal(
    regions.length,
    song.sections.length,
    `structure rail should render one region per Rust section; got ${JSON.stringify(regions)}`,
  );
  assertNear(regions[0].startSeconds, 0, TIME_TOLERANCE_SECONDS, "first region start");
  assertNear(regions.at(-1).endSeconds, duration, TIME_TOLERANCE_SECONDS, "last region end");
  for (let index = 1; index < regions.length; index += 1) {
    assertNear(regions[index].startSeconds, regions[index - 1].endSeconds, 0.01, `region ${index + 1} contiguity`);
  }
  regions.forEach((region, index) => {
    if (index > 0) {
      assertNear(region.startSeconds, song.sections[index].startSeconds, 0.01, `region ${index + 1} Rust start`);
    }
  });

  // The regions span the full waveform width (not a centered window).
  const stage = await page.locator(".waveform-stage").boundingBox();
  assert.ok(stage && stage.width > 0, "expected waveform stage geometry");
  assert.ok(Math.abs(regions[0].box.x - stage.x) <= 2, "first region should start at the waveform's left edge");
  assert.ok(
    Math.abs(regions.at(-1).box.x + regions.at(-1).box.width - (stage.x + stage.width)) <= 2,
    "last region should end at the waveform's right edge",
  );

  // Stable identities: labels come from Rust; repeated identities look the same.
  assert.equal(regions[0].label, "A");
  assert.equal(regions[1].label, "B");
  assert.match(regions[2].label, /^A(?:[′'’]+)?$/, `repeat of A should be labelled A or A′; got ${regions[2].label}`);
  assert.equal(regions[2].color, regions[0].color, "a repeated identity should keep the same section color");
  assert.notEqual(regions[1].color, regions[0].color, "different identities should be visually distinct");
  for (const region of regions) {
    assert.doesNotMatch(region.text, SEMANTIC_SECTION_NAMES, "no verse/chorus/drop-style semantic labels");
  }

  // Local tempo/key annotations where they materially differ from the neighbouring region,
  // with values taken from the Rust contracts.
  assert.match(regions[1].text, /\b150(?:\.\d+)?\s*BPM\b/i, `B should show its local tempo; got "${regions[1].text}"`);
  assert.match(regions[1].text, /\bF(?:#|♯)\s*(?:major|maj)\b/i, `B should show its local key; got "${regions[1].text}"`);
  for (const index of [0, 2]) {
    assert.match(regions[index].text, /\b120(?:\.\d+)?\s*BPM\b/i, `region ${index + 1} differs from B in tempo`);
    assert.match(regions[index].text, /\bC\s*(?:major|maj)\b/i, `region ${index + 1} differs from B in key`);
  }
  assert.ok(
    keyTrack.segments.some((segment) => segment.key?.tonic === "F#" && segment.key?.scale === "major"),
    "the exported key track should hold the F# major evidence the B annotation shows",
  );

  // Confidence stays available.
  for (const region of regions.slice(1)) {
    assert.match(region.description, /confidence/i, "section boundary confidence should remain available");
  }
  assert.ok(song.sections.every((section) => Number.isFinite(section.startBoundaryConfidence)));
  assert.ok(keyTrack.segments.every((segment) => Number.isFinite(segment.confidence)));

  // The bounded rhythm window is not presented as the structural extent.
  const sectionToggleLabel = (await page.locator("#section-overlay-toggle").getAttribute("aria-label")) ?? "";
  assert.doesNotMatch(sectionToggleLabel, /rhythm window/i, `section status should describe the whole file; got "${sectionToggleLabel}"`);
  assert.match(sectionToggleLabel, /\b3 detected sections\b/i);

  // Beats/downbeats are overlaid across the full file, outside the old center window too.
  const beatRatios = await markerRatios(page, ".waveform-beat-marker");
  const downbeatRatios = await markerRatios(page, ".waveform-beat-marker-downbeat");
  assert.ok(beatRatios.length >= song.beats.length * 0.9, `expected whole-track beat markers; got ${beatRatios.length}`);
  assert.ok(beatRatios.some((ratio) => ratio < 0.1) && beatRatios.some((ratio) => ratio > 0.9), "beats near both file ends");
  assert.ok(
    downbeatRatios.some((ratio) => ratio < 0.1) && downbeatRatios.some((ratio) => ratio > 0.9),
    "downbeats near both file ends",
  );

  // Playback position stays in sync with the whole-track timeline.
  const hud = page.locator("#waveform-context-hud");
  for (const [time, regionIndex, bpm] of [
    [5, 0, 120],
    [30, 1, 150],
    [66, 2, 120],
  ]) {
    await setPlaybackTime(page, time);
    const hudText = (await hud.innerText()).replace(/\s+/g, " ");
    assert.doesNotMatch(hudText, /outside/i, `t=${time}s is inside the analyzed file; HUD: "${hudText}"`);
    assert.match(hudText, new RegExp(`Section ${escapeRegExp(regions[regionIndex].label)}(?=\\s|·|$)`), `HUD section at t=${time}s: "${hudText}"`);
    assert.match(hudText, new RegExp(`\\b${bpm}(?:\\.\\d+)? BPM`), `HUD local tempo at t=${time}s: "${hudText}"`);
    assert.equal(await activeRegionIndex(page), regionIndex, `active region at t=${time}s`);
  }

  // Clicking a section seeks playback to its start.
  await setPlaybackTime(page, 2);
  await page.locator(".waveform-section-segment").nth(regions[2].domIndex).click();
  const seeked = await page.locator("#audio-player").evaluate((element) => element.currentTime);
  assertNear(seeked, regions[2].startSeconds, TIME_TOLERANCE_SECONDS, "seek after clicking the A′ section");

  if (artifactDir) {
    await mkdir(artifactDir, { recursive: true });
    await page.locator("#waveform-section").screenshot({ path: path.join(artifactDir, "whole-track-sections.png") });
  }

  // ---------------------------------------------------------------------------------------------
  // Journey 2: the same arrangement contrast, but tempo (120 vs 120.5 BPM) and key (C major) do not
  // materially change. The timeline must neither fragment nor annotate estimator jitter.
  // ---------------------------------------------------------------------------------------------
  const steady = await analyzeSong(page, steadySong);
  assert.deepEqual(
    steady.song.sections.map((section) => section.identity),
    ["A", "B", "A"],
    `Rust song contract should detect A/B/A for the steady fixture; got ${JSON.stringify(steady.song.sections)}`,
  );
  const steadyRegions = await sectionRegions(page);
  assert.equal(
    steadyRegions.length,
    steady.song.sections.length,
    `small tempo/key fluctuations must not fragment the timeline; got ${JSON.stringify(steadyRegions)}`,
  );
  assertNear(steadyRegions[0].startSeconds, 0, TIME_TOLERANCE_SECONDS, "steady first region start");
  assertNear(steadyRegions.at(-1).endSeconds, steady.duration, TIME_TOLERANCE_SECONDS, "steady last region end");
  for (const region of steadyRegions) {
    assert.doesNotMatch(region.text, /BPM/i, `no tempo annotation without a material change; got "${region.text}"`);
    assert.doesNotMatch(
      region.text,
      /\b(?:major|minor|maj|min)\b/i,
      `no key annotation without a material change; got "${region.text}"`,
    );
  }

  assert.deepEqual(pageErrors, [], `browser page errors:\n${pageErrors.join("\n")}`);
  console.log("Whole-track sections acceptance passed");
} finally {
  await browser.close();
}

async function analyzeSong(page, fixture) {
  const expectedDuration = songDurationSeconds(fixture);
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator("#file-input").setInputFiles({
    name: fixture.name,
    mimeType: "audio/wav",
    buffer: songWav(fixture),
  });
  await page.locator("#report").waitFor({ state: "visible", timeout: ANALYSIS_TIMEOUT_MS });
  await page.waitForFunction(
    (name) => {
      const text = document.querySelector("#raw-json")?.textContent ?? "";
      return text.includes(name);
    },
    fixture.name,
    { timeout: ANALYSIS_TIMEOUT_MS },
  );

  // Whole-track evidence may arrive after the fast representative-window report.
  let report = null;
  let song = null;
  let keyTrack = null;
  const deadline = Date.now() + ANALYSIS_TIMEOUT_MS;
  while (Date.now() < deadline) {
    report = JSON.parse((await page.locator("#raw-json").textContent()) || "{}");
    song = wholeTrackSong(report, expectedDuration);
    keyTrack = wholeTrackKey(report, expectedDuration);
    if (song && keyTrack && (await page.locator(".waveform-section-segment").count()) > 0) break;
    await page.waitForTimeout(250);
  }
  assert.ok(
    song,
    "the inspector report should carry an audio-analysis-song/v1 result covering the whole decoded file " +
      `(${expectedDuration} s); found only: ${JSON.stringify(songContracts(report).map(songExtent))}`,
  );
  assert.ok(keyTrack, "the inspector report should carry a whole-file audio-analysis-key-track result with segments");

  const duration = Number(report.source?.durationSeconds);
  assertNear(duration, expectedDuration, 0.01, "decoded duration");
  await page.waitForFunction(() => {
    const player = document.querySelector("#audio-player");
    return player && Number.isFinite(player.duration) && player.duration > 0;
  });

  // JSON export retains the authoritative section/key/tempo evidence used by the UI.
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#export-json").click();
  const download = await downloadPromise;
  const exported = JSON.parse(await readFile(await download.path(), "utf8"));
  const exportedSong = wholeTrackSong(exported, expectedDuration);
  const exportedKey = wholeTrackKey(exported, expectedDuration);
  assert.ok(exportedSong, "exported JSON should retain the whole-track song/v1 sections, beats, and tempo map");
  assert.ok(exportedKey, "exported JSON should retain the whole-track key segments");
  assert.deepEqual(exportedSong.sections, song.sections);
  assert.ok(Array.isArray(exportedSong.tempoMap) && exportedSong.tempoMap.length > 0, "exported tempo map");
  assert.ok(exportedSong.tempoMap.at(-1).timestampSeconds > expectedDuration * 0.9, "tempo map reaches the file end");

  return { report, duration, song, keyTrack };
}

function songContracts(value, found = []) {
  if (!value || typeof value !== "object") return found;
  if (value.schemaVersion === "audio-analysis-song/v1" && Array.isArray(value.sections)) found.push(value);
  for (const nested of Object.values(value)) songContracts(nested, found);
  return found;
}

function songExtent(song) {
  return { start: song.analysisStartSeconds, end: song.analysisEndSeconds, sections: song.sections?.length };
}

function wholeTrackSong(report, duration) {
  return (
    songContracts(report).find(
      (song) =>
        Number(song.analysisStartSeconds) <= TIME_TOLERANCE_SECONDS &&
        Number(song.analysisEndSeconds) >= duration - 0.1 &&
        song.sections.length > 0 &&
        Number(song.sections.at(-1).endSeconds) >= duration - 0.1,
    ) ?? null
  );
}

function wholeTrackKey(value, duration) {
  if (!value || typeof value !== "object") return null;
  if (
    typeof value.schemaVersion === "string" &&
    value.schemaVersion.startsWith("audio-analysis-key-track/") &&
    Array.isArray(value.segments) &&
    value.segments.length > 0 &&
    Number(value.durationSeconds) >= duration - 0.1
  ) {
    return value;
  }
  for (const nested of Object.values(value)) {
    const found = wholeTrackKey(nested, duration);
    if (found) return found;
  }
  return null;
}

async function sectionRegions(page) {
  const regions = await page.locator(".waveform-section-segment").evaluateAll((elements) =>
    elements.map((element, domIndex) => {
      const box = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      return {
        domIndex,
        startSeconds: Number(element.dataset.sectionStartSeconds),
        endSeconds: Number(element.dataset.sectionEndSeconds),
        label: element.querySelector(".waveform-section-label")?.textContent?.trim() ?? "",
        text: element.innerText.replace(/\s+/g, " ").trim(),
        description: `${element.getAttribute("aria-label") ?? ""} ${element.title ?? ""}`,
        color: `${style.backgroundColor} ${style.backgroundImage}`,
        visible: box.width > 0 && box.height > 0 && style.visibility !== "hidden",
        box: { x: box.x, y: box.y, width: box.width, height: box.height },
      };
    }),
  );
  for (const region of regions) {
    assert.ok(Number.isFinite(region.startSeconds) && Number.isFinite(region.endSeconds), "region timing metadata");
    assert.ok(region.endSeconds > region.startSeconds, "regions must have positive duration");
    assert.ok(region.visible, `region ${region.label} should be visible on the waveform`);
  }
  return regions.sort((left, right) => left.startSeconds - right.startSeconds);
}

async function markerRatios(page, selector) {
  const stage = await page.locator(".waveform-stage").boundingBox();
  return page
    .locator(selector)
    .evaluateAll(
      (elements, stageBox) =>
        elements.map((element) => (element.getBoundingClientRect().x - stageBox.x) / stageBox.width),
      stage,
    );
}

async function activeRegionIndex(page) {
  const regions = await sectionRegions(page);
  const active = await page
    .locator(".waveform-section-segment")
    .evaluateAll((elements) =>
      elements
        .map((element, index) => (element.classList.contains("is-active") || element.getAttribute("aria-current") ? index : -1))
        .filter((index) => index >= 0),
    );
  assert.equal(active.length, 1, `exactly one region should be active; got ${JSON.stringify(active)}`);
  return regions.findIndex((region) => region.domIndex === active[0]);
}

async function setPlaybackTime(page, time) {
  await page.locator("#audio-player").evaluate((element, nextTime) => {
    element.currentTime = nextTime;
    element.dispatchEvent(new Event("timeupdate"));
  }, time);
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => resolve())));
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function assertNear(actual, expected, tolerance, label) {
  assert.ok(
    Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected} ± ${tolerance}, got ${actual}`,
  );
}
