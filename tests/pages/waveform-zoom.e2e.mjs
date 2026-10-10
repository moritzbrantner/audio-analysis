// Acceptance for audio-analysis#140: zoom and pan on the main inspector waveform with a persistent
// whole-file overview.
//
// Real browser, real Rust/WASM analyzers, the deterministic 72 s A→B→A song from #138 (120 BPM,
// 150 BPM in B, section boundaries near 24 s and 48 s). The visible time window is measured from
// the rendered page itself — the beat markers the inspector places over the detailed waveform —
// and every other surface is checked against it: click-to-seek, hover readout, playhead, section
// boundaries and rail, and the overview's viewport indicator.
//
// Contract inferences (recorded in the PR):
// - Zoom controls are buttons named "Zoom in…" / "Zoom out…"; they are the keyboard path.
// - The overview is labelled "Whole-file overview" (or "Full-file overview") and its viewport
//   indicator is a labelled element ("visible window/range/region" or "viewport").
// - A plain mouse wheel over the waveform zooms around the pointer and does not scroll the page;
//   Ctrl+wheel (how Chromium delivers trackpad pinch) zooms too; a two-finger touch spread zooms.
// - Dragging the waveform pans without seeking; a click without drag still seeks.
// - Zoom buttons anchor on the playhead, keeping its on-screen position stable.
// - Deepest zoom shows at most 2 s (beat scale); zooming out ends at exactly the whole file.
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { contrastingSong, songDurationSeconds, songWav } from "./fixtures/whole-track-songs.mjs";

const baseUrl = process.env.PAGES_E2E_BASE_URL ?? "http://127.0.0.1:4173";
const artifactDir = process.env.PAGES_E2E_ARTIFACT_DIR ?? "";
const ANALYSIS_TIMEOUT_MS = 120_000;
const PX_TOLERANCE = 3;
const BEAT_SCALE_SECONDS = 2;
const PLAYHEAD_TIME = 30;

const browser = await chromium.launch({ headless: true });

try {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, hasTouch: true });
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

  const { duration, beats } = await analyzeSong(page);
  const zoomIn = page.getByRole("button", { name: /^zoom in\b/i });
  const zoomOut = page.getByRole("button", { name: /^zoom out\b/i });
  const overview = page.getByLabel(/\b(?:whole|full)[- ]file overview\b/i);
  const indicator = page.getByLabel(/\b(?:visible|detail(?:ed)?) (?:window|range|region)\b|\bviewport\b/i);

  // -------------------------------------------------------------------------------------------
  // Whole-file state: controls, overview and indicator present; the detail view shows the file.
  // -------------------------------------------------------------------------------------------
  await zoomIn.waitFor({ state: "visible", timeout: 10_000 }).catch(() => {});
  assert.equal(await zoomIn.isVisible(), true, 'expected a visible "Zoom in" button for keyboard-accessible zoom');
  assert.equal(await zoomOut.isVisible(), true, 'expected a visible "Zoom out" button for keyboard-accessible zoom');
  assert.equal(await overview.isVisible(), true, "expected a persistent whole-file overview");
  assert.equal(await indicator.isVisible(), true, "expected a visible viewport indicator on the overview");
  const overviewBox = await overview.boundingBox();
  const overviewPixels = await overviewHash(page, overview);

  let view = await visibleWindow(page, beats, duration);
  assertNear(view.startSeconds, 0, pxTime(view, 2), "initial view starts at the file start");
  assertNear(view.endSeconds, duration, pxTime(view, 2), "initial view ends at the file end");
  await assertOverview(page, overview, overviewBox, indicator, view, duration, "initial");

  // -------------------------------------------------------------------------------------------
  // Mouse wheel zooms around the pointer, without scrolling the page.
  // -------------------------------------------------------------------------------------------
  const stage = await page.locator("#waveform").boundingBox();
  const pointerX = stage.x + Math.round(stage.width * 0.3);
  const pointerY = stage.y + Math.round(stage.height * 0.6);
  const pointerTime = timeAtX(view, stage, pointerX);
  const scrollBefore = await page.evaluate(() => window.scrollY);
  await page.mouse.move(pointerX, pointerY);
  for (let tick = 0; tick < 3; tick += 1) {
    await page.mouse.wheel(0, -120);
    await nextFrame(page);
  }
  assert.equal(await page.evaluate(() => window.scrollY), scrollBefore, "wheel over the waveform should not scroll the page");
  let zoomed = await visibleWindow(page, beats, duration);
  assert.ok(span(zoomed) <= span(view) * 0.9, `wheel up should zoom in; span ${span(view)} → ${span(zoomed)}`);
  for (let tick = 0; tick < 20 && span(zoomed) > 30; tick += 1) {
    await page.mouse.wheel(0, -120);
    await nextFrame(page);
    zoomed = await visibleWindow(page, beats, duration);
  }
  assert.ok(span(zoomed) <= 30 && span(zoomed) >= 6, `wheel should reach phrase scale (tens of seconds); got ${span(zoomed)} s`);
  assertNearPx(xAtTime(zoomed, stage, pointerTime), pointerX, "time under the pointer stays fixed during wheel zoom");
  view = zoomed;

  // At phrase scale every surface uses the zoomed mapping.
  await assertAligned(page, view, beats, duration, "after wheel zoom");
  await assertOverview(page, overview, overviewBox, indicator, view, duration, "after wheel zoom");
  await assertHoverReadout(page, view, stage, beats, "after wheel zoom");
  await assertClickSeeks(page, view, stage, beats, duration, "after wheel zoom");
  assert.equal(await overviewHash(page, overview), overviewPixels, "the overview keeps showing the whole file while zoomed");

  // -------------------------------------------------------------------------------------------
  // Ctrl+wheel (trackpad pinch as Chromium reports it) also zooms around the pointer.
  // -------------------------------------------------------------------------------------------
  const pinchX = stage.x + Math.round(stage.width * 0.55);
  const pinchTime = timeAtX(view, stage, pinchX);
  await page.mouse.move(pinchX, pointerY);
  await page.keyboard.down("Control");
  for (let tick = 0; tick < 4; tick += 1) {
    await page.mouse.wheel(0, -40);
    await nextFrame(page);
  }
  await page.keyboard.up("Control");
  assert.equal(await page.evaluate(() => window.visualViewport?.scale ?? 1), 1, "trackpad pinch must not zoom the page");
  zoomed = await visibleWindow(page, beats, duration);
  assert.ok(span(zoomed) < span(view) * 0.97, `Ctrl+wheel should zoom in; span ${span(view)} → ${span(zoomed)}`);
  assertNearPx(xAtTime(zoomed, stage, pinchTime), pinchX, "time under the pointer stays fixed during pinch zoom");
  view = zoomed;

  // -------------------------------------------------------------------------------------------
  // Drag pans the visible window and does not seek; pan is bounded by the file.
  // -------------------------------------------------------------------------------------------
  await setPlaybackTime(page, view.startSeconds + span(view) * 0.5);
  const playbackBeforeDrag = await currentTime(page);
  await drag(page, stage.x + stage.width * 0.6, stage.x + stage.width * 0.4, pointerY);
  let panned = await visibleWindow(page, beats, duration);
  assertNear(span(panned), span(view), pxTime(view, 2), "dragging keeps the zoom level");
  assertNear(panned.startSeconds, view.startSeconds + span(view) * 0.2, pxTime(view, PX_TOLERANCE), "drag pans by the dragged distance");
  assertNear(await currentTime(page), playbackBeforeDrag, 0.01, "a drag pans and must not seek playback");
  view = panned;
  await assertAligned(page, view, beats, duration, "after drag pan");
  await assertOverview(page, overview, overviewBox, indicator, view, duration, "after drag pan");
  await assertClickSeeks(page, view, stage, beats, duration, "after drag pan");

  // Dragging the content leftwards reveals later time until the window meets the file end.
  for (let attempt = 0; attempt < 10; attempt += 1) await drag(page, stage.x + stage.width - 20, stage.x + 20, pointerY);
  panned = await visibleWindow(page, beats, duration);
  assertNear(panned.endSeconds, duration, pxTime(panned, PX_TOLERANCE), "panning stops at the file end");
  assertNear(span(panned), span(view), pxTime(view, 2), "pan clamping keeps the zoom level");
  await assertOverview(page, overview, overviewBox, indicator, panned, duration, "panned to the end");
  // And rightwards until it meets the file start.
  for (let attempt = 0; attempt < 10; attempt += 1) await drag(page, stage.x + 20, stage.x + stage.width - 20, pointerY);
  panned = await visibleWindow(page, beats, duration);
  assertNear(panned.startSeconds, 0, pxTime(panned, PX_TOLERANCE), "panning stops at the file start");
  assertNear(span(panned), span(view), pxTime(view, 2), "pan clamping keeps the zoom level");
  await assertOverview(page, overview, overviewBox, indicator, panned, duration, "panned to the start");
  view = panned;

  // -------------------------------------------------------------------------------------------
  // Touch pinch (two-finger spread) zooms where touch is supported.
  // -------------------------------------------------------------------------------------------
  const centerX = stage.x + stage.width / 2;
  const centerTime = timeAtX(view, stage, centerX);
  await touchPinch(context, page, centerX, pointerY, 60, 240);
  zoomed = await visibleWindow(page, beats, duration);
  assert.ok(span(zoomed) < span(view) * 0.8, `two-finger spread should zoom in; span ${span(view)} → ${span(zoomed)}`);
  assert.ok(
    Math.abs(xAtTime(zoomed, stage, centerTime) - centerX) <= 8,
    `pinch should keep the time between the fingers roughly fixed; ${centerTime}s now at ${xAtTime(zoomed, stage, centerTime)} vs ${centerX}`,
  );
  assert.equal(await page.evaluate(() => window.visualViewport?.scale ?? 1), 1, "touch pinch on the waveform must not zoom the page");

  // -------------------------------------------------------------------------------------------
  // Keyboard zoom controls: back to the whole file, then playhead-anchored zoom to beat scale.
  // -------------------------------------------------------------------------------------------
  await page.mouse.move(5, 5);
  view = await pressUntilStable(page, zoomOut, beats, duration, "Enter");
  assertNear(view.startSeconds, 0, pxTime(view, 2), "Zoom out returns to the file start");
  assertNear(view.endSeconds, duration, pxTime(view, 2), "Zoom out returns to the file end");
  await assertOverview(page, overview, overviewBox, indicator, view, duration, "zoomed out by keyboard");

  await setPlaybackTime(page, PLAYHEAD_TIME);
  const playheadFraction = (PLAYHEAD_TIME - view.startSeconds) / span(view);
  const spans = [span(view)];
  let reachedBeatScale = null;
  for (let press = 0; press < 30 && !reachedBeatScale; press += 1) {
    const before = await visibleWindow(page, beats, duration);
    if (await zoomIn.isDisabled()) {
      reachedBeatScale = before;
      break;
    }
    await zoomIn.focus();
    assert.equal(await zoomIn.evaluate((element) => element === document.activeElement), true, "Zoom in is keyboard focusable");
    await page.keyboard.press(press % 2 ? " " : "Enter");
    const after = await visibleWindow(page, beats, duration);
    assert.ok(after.startSeconds <= PLAYHEAD_TIME && after.endSeconds >= PLAYHEAD_TIME, "the playhead stays in view while zooming");
    assertNearPx(
      xAtTime(after, stage, PLAYHEAD_TIME),
      stage.x + playheadFraction * stage.width,
      `zoom keeps the playhead's on-screen position (span ${span(after).toFixed(2)} s)`,
      Math.max(PX_TOLERANCE, stage.width * 0.01),
    );
    assert.equal(await currentTime(page), PLAYHEAD_TIME, "zooming does not move playback");
    if (Math.abs(span(after) - span(before)) <= pxTime(before, 1)) reachedBeatScale = after;
    else spans.push(span(after));
  }
  assert.ok(reachedBeatScale, `repeated Zoom in should stop at a bounded limit; spans ${spans.map((value) => value.toFixed(2)).join(", ")}`);
  assert.ok(
    spans.some((value) => value >= 8 && value <= 40),
    `zoom steps should pass through phrase scale (tens of seconds); spans ${spans.map((value) => value.toFixed(2)).join(", ")}`,
  );
  view = await visibleWindow(page, beats, duration);
  assert.ok(span(view) <= BEAT_SCALE_SECONDS + 0.05, `deepest zoom should reach beat scale (≤ ${BEAT_SCALE_SECONDS} s); got ${span(view)} s`);

  // Beat scale: overlays, playhead, current time, seek and overview all still agree.
  await assertAligned(page, view, beats, duration, "at beat scale");
  await assertPlayhead(page, view, stage, PLAYHEAD_TIME, "at beat scale");
  await assertOverview(page, overview, overviewBox, indicator, view, duration, "at beat scale");
  assert.equal(Number(await page.locator("#waveform").getAttribute("aria-valuenow")), PLAYHEAD_TIME, "seek slider keeps absolute time");
  const hudText = (await page.locator("#waveform-context-hud").innerText()).replace(/\s+/g, " ");
  assert.match(hudText, /0:30\.0/, `HUD shows the current playback time; got "${hudText}"`);
  assert.match(hudText, /Section B\b/, `HUD shows the section at the playhead; got "${hudText}"`);
  assert.match(hudText, /\b150(?:\.\d+)? BPM/, `HUD shows the local tempo at the playhead; got "${hudText}"`);
  await assertHoverReadout(page, view, stage, beats, "at beat scale");
  if (artifactDir) {
    await mkdir(artifactDir, { recursive: true });
    await page.locator("#waveform-section").screenshot({ path: path.join(artifactDir, "waveform-zoom-beat-scale.png") });
  }
  await assertClickSeeks(page, view, stage, beats, duration, "at beat scale");

  await zoomOut.focus();
  view = await pressUntilStable(page, zoomOut, beats, duration, "Enter");
  assertNear(view.startSeconds, 0, pxTime(view, 2), "keyboard zoom out ends at the whole file (start)");
  assertNear(view.endSeconds, duration, pxTime(view, 2), "keyboard zoom out ends at the whole file (end)");
  assert.equal(await overviewHash(page, overview), overviewPixels, "the overview is unchanged by zoom and pan");
  if (artifactDir) {
    await page.locator("#waveform-section").screenshot({ path: path.join(artifactDir, "waveform-zoom-whole-file.png") });
  }

  assert.deepEqual(pageErrors, [], `browser page errors:\n${pageErrors.join("\n")}`);
  console.log("Waveform zoom/pan/overview acceptance passed");
} finally {
  await browser.close();
}

// ---------------------------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------------------------

async function analyzeSong(page) {
  const expectedDuration = songDurationSeconds(contrastingSong);
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator("#file-input").setInputFiles({
    name: contrastingSong.name,
    mimeType: "audio/wav",
    buffer: songWav(contrastingSong),
  });
  await page.locator("#report").waitFor({ state: "visible", timeout: ANALYSIS_TIMEOUT_MS });
  let report = null;
  let song = null;
  const deadline = Date.now() + ANALYSIS_TIMEOUT_MS;
  while (Date.now() < deadline) {
    report = JSON.parse((await page.locator("#raw-json").textContent()) || "{}");
    song = wholeTrackSong(report, expectedDuration);
    if (song && (await page.locator(".waveform-section-segment").count()) === song.sections.length) break;
    await page.waitForTimeout(250);
  }
  assert.ok(song, "fixture guard: the inspector should carry the whole-track song contract (#138)");
  await page.waitForFunction(() => {
    const player = document.querySelector("#audio-player");
    return player && Number.isFinite(player.duration) && player.duration > 0;
  });
  await page.locator("#waveform").scrollIntoViewIfNeeded();
  const duration = Number(report.source?.durationSeconds);
  assertNear(duration, expectedDuration, 0.01, "decoded duration");
  const beats = new Map(song.beats.map((beat, index) => [String(beat.index ?? index + 1), Number(beat.timestampSeconds)]));
  assert.ok(beats.size > 100, "fixture guard: whole-track beats");
  return { duration, beats };
}

function songContracts(value, found = []) {
  if (!value || typeof value !== "object") return found;
  if (value.schemaVersion === "audio-analysis-song/v1" && Array.isArray(value.sections)) found.push(value);
  for (const nested of Object.values(value)) songContracts(nested, found);
  return found;
}

function wholeTrackSong(report, duration) {
  return (
    songContracts(report).find(
      (song) =>
        Number(song.analysisStartSeconds) <= 0.05 &&
        Number(song.analysisEndSeconds) >= duration - 0.1 &&
        Array.isArray(song.beats) &&
        song.beats.length > 0,
    ) ?? null
  );
}

// ---------------------------------------------------------------------------------------------
// Measuring the visible window from the rendered beat markers
// ---------------------------------------------------------------------------------------------

async function renderedBeatMarkers(page) {
  return page.locator(".waveform-beat-marker").evaluateAll((elements) => {
    const stage = document.querySelector("#waveform").getBoundingClientRect();
    return elements.map((element) => {
      const box = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      let hidden = style.display === "none" || style.visibility === "hidden" || box.width === 0;
      for (let node = element.parentElement; node && !hidden; node = node.parentElement) {
        if (node.hidden || getComputedStyle(node).display === "none") hidden = true;
      }
      const center = box.x + box.width / 2;
      return {
        beatIndex: element.dataset.beatIndex,
        center,
        inStage: !hidden && center >= stage.x - 0.5 && center <= stage.x + stage.width + 0.5,
      };
    });
  });
}

// Least-squares fit of x = stage.x + (t - start) / span * width over the markers inside the stage.
async function visibleWindow(page, beats, duration) {
  await nextFrame(page);
  const stage = await page.locator("#waveform").boundingBox();
  const markers = (await renderedBeatMarkers(page)).filter((marker) => marker.inStage && beats.has(marker.beatIndex));
  assert.ok(markers.length >= 2, `need at least two beat markers on the waveform to measure the view; got ${markers.length}`);
  const points = markers.map((marker) => ({ t: beats.get(marker.beatIndex), x: marker.center - stage.x }));
  const n = points.length;
  const meanT = points.reduce((sum, point) => sum + point.t, 0) / n;
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / n;
  let numerator = 0;
  let denominator = 0;
  for (const point of points) {
    numerator += (point.t - meanT) * (point.x - meanX);
    denominator += (point.t - meanT) ** 2;
  }
  const pxPerSecond = numerator / denominator;
  assert.ok(pxPerSecond > 0, "beat markers should increase left to right");
  const startSeconds = meanT - meanX / pxPerSecond;
  const window = { startSeconds, endSeconds: startSeconds + stage.width / pxPerSecond, width: stage.width };
  assert.ok(window.startSeconds >= -pxTime(window, PX_TOLERANCE), `view should not start before the file: ${JSON.stringify(window)}`);
  assert.ok(window.endSeconds <= duration + pxTime(window, PX_TOLERANCE), `view should not end after the file: ${JSON.stringify(window)}`);
  return window;
}

function span(window) {
  return window.endSeconds - window.startSeconds;
}

function pxTime(window, pixels) {
  return (span(window) / window.width) * pixels;
}

function xAtTime(window, stage, time) {
  return stage.x + ((time - window.startSeconds) / span(window)) * stage.width;
}

function timeAtX(window, stage, x) {
  return window.startSeconds + ((x - stage.x) / stage.width) * span(window);
}

// ---------------------------------------------------------------------------------------------
// Invariants at any zoom level
// ---------------------------------------------------------------------------------------------

async function assertAligned(page, view, beats, duration, label) {
  const stage = await page.locator("#waveform").boundingBox();
  const markers = await renderedBeatMarkers(page);
  let shown = 0;
  for (const marker of markers) {
    if (!beats.has(marker.beatIndex)) continue;
    const time = beats.get(marker.beatIndex);
    const expectedX = xAtTime(view, stage, time);
    const insideView = time >= view.startSeconds + pxTime(view, 1) && time <= view.endSeconds - pxTime(view, 1);
    if (marker.inStage) {
      shown += 1;
      assertNearPx(marker.center, expectedX, `${label}: beat ${marker.beatIndex} (${time.toFixed(3)} s) marker position`);
    } else if (insideView) {
      assert.fail(`${label}: beat ${marker.beatIndex} at ${time.toFixed(3)} s is inside the view but not shown`);
    }
  }
  const expectedBeats = [...beats.values()].filter((time) => time >= view.startSeconds && time <= view.endSeconds).length;
  assert.ok(Math.abs(shown - expectedBeats) <= 2, `${label}: ${shown} beat markers shown for ${expectedBeats} beats in view`);

  const sections = await page.locator(".waveform-section-segment").evaluateAll((elements) =>
    elements.map((element) => {
      const box = element.getBoundingClientRect();
      return {
        index: element.dataset.sectionIndex,
        start: Number(element.dataset.sectionStartSeconds),
        end: Number(element.dataset.sectionEndSeconds),
        left: box.x,
        right: box.x + box.width,
      };
    }),
  );
  assert.ok(sections.length > 0, `${label}: section rail should still be rendered`);
  for (const section of sections) {
    if (section.start > view.startSeconds && section.start < view.endSeconds) {
      assertNearPx(section.left, xAtTime(view, stage, section.start), `${label}: section ${section.index} rail start`);
    } else if (section.start <= view.startSeconds && section.end > view.startSeconds) {
      assert.ok(section.left <= stage.x + PX_TOLERANCE, `${label}: section ${section.index} begins before the view`);
    }
    if (section.end > view.startSeconds && section.end < view.endSeconds) {
      assertNearPx(section.right, xAtTime(view, stage, section.end), `${label}: section ${section.index} rail end`);
    } else if (section.end >= view.endSeconds && section.start < view.endSeconds) {
      assert.ok(section.right >= stage.x + stage.width - PX_TOLERANCE, `${label}: section ${section.index} continues past the view`);
    }
  }

  const boundaries = await page.locator(".waveform-section-boundary").evaluateAll((elements) =>
    elements.map((element) => {
      const box = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      return { index: element.dataset.sectionIndex, x: box.x, hidden: style.display === "none" || box.height === 0 };
    }),
  );
  for (const boundary of boundaries) {
    const section = sections.find((candidate) => candidate.index === boundary.index);
    if (!section || boundary.hidden) continue;
    if (section.start > view.startSeconds && section.start < view.endSeconds) {
      assertNearPx(boundary.x, xAtTime(view, stage, section.start), `${label}: section ${section.index} boundary marker`);
    } else {
      const visibleInStage = boundary.x >= stage.x + 1 && boundary.x <= stage.x + stage.width - 1;
      assert.ok(!visibleInStage, `${label}: boundary at ${section.start}s is outside the view but drawn on the waveform`);
    }
  }
}

async function assertOverview(page, overview, overviewBox, indicator, view, duration, label) {
  assert.equal(await overview.isVisible(), true, `${label}: overview stays visible`);
  const box = await overview.boundingBox();
  assert.ok(
    Math.abs(box.x - overviewBox.x) <= 1 && Math.abs(box.width - overviewBox.width) <= 1,
    `${label}: overview geometry is persistent (not zoomed); was ${JSON.stringify(overviewBox)}, now ${JSON.stringify(box)}`,
  );
  const indicatorBox = await indicator.boundingBox();
  assert.ok(indicatorBox, `${label}: viewport indicator is rendered`);
  const toX = (time) => box.x + (time / duration) * box.width;
  const expectedLeft = toX(view.startSeconds);
  const expectedRight = toX(view.endSeconds);
  const expectedCenter = (expectedLeft + expectedRight) / 2;
  const center = indicatorBox.x + indicatorBox.width / 2;
  assert.ok(
    Math.abs(center - expectedCenter) <= PX_TOLERANCE,
    `${label}: indicator center ${center.toFixed(1)} should mark the visible window center ${expectedCenter.toFixed(1)}`,
  );
  const expectedWidth = expectedRight - expectedLeft;
  const widthMatches =
    Math.abs(indicatorBox.width - expectedWidth) <= PX_TOLERANCE * 2 || (expectedWidth < 8 && indicatorBox.width <= 16);
  assert.ok(widthMatches, `${label}: indicator width ${indicatorBox.width.toFixed(1)} should match the visible span ${expectedWidth.toFixed(1)}`);
}

async function assertClickSeeks(page, view, stage, beats, duration, label) {
  const markers = (await renderedBeatMarkers(page)).filter((marker) => marker.inStage);
  const targets = [markers[Math.floor(markers.length / 3)], markers[Math.floor((markers.length * 2) / 3)]];
  for (const marker of targets) {
    const x = Math.round(marker.center);
    const expected = timeAtX(view, stage, x);
    await page.mouse.click(x, stage.y + stage.height * 0.6);
    await nextFrame(page);
    assertNear(await currentTime(page), expected, pxTime(view, PX_TOLERANCE), `${label}: click at beat ${marker.beatIndex} seeks to its time`);
    const after = await visibleWindow(page, beats, duration);
    assertNear(after.startSeconds, view.startSeconds, pxTime(view, 1.5), `${label}: a click seek does not move the view`);
  }
}

async function assertHoverReadout(page, view, stage, beats, label) {
  const markers = (await renderedBeatMarkers(page)).filter((marker) => marker.inStage);
  const marker = markers[Math.floor(markers.length / 2)];
  const x = Math.round(marker.center);
  const expected = timeAtX(view, stage, x);
  await page.mouse.move(x, stage.y + stage.height * 0.6);
  await nextFrame(page);
  const readout = (await page.locator("#waveform-readout").innerText()).replace(/\s+/g, " ");
  const hud = (await page.locator("#waveform-context-hud").innerText()).replace(/\s+/g, " ");
  const tolerance = 0.06 + pxTime(view, PX_TOLERANCE);
  assertNear(parseClock(readout), expected, tolerance, `${label}: pointer readout "${readout}"`);
  assertNear(parseClock(hud), expected, tolerance, `${label}: context HUD under the pointer "${hud}"`);
  await page.mouse.move(5, 5);
  await nextFrame(page);
}

async function assertPlayhead(page, view, stage, time, label) {
  const expected = xAtTime(view, stage, time) - stage.x;
  const domPlayhead = await page.evaluate(() => {
    const element = document.querySelector("[data-waveform-playhead], .waveform-playhead");
    if (!element) return null;
    const box = element.getBoundingClientRect();
    return box.x + box.width / 2 - document.querySelector("#waveform").getBoundingClientRect().x;
  });
  if (domPlayhead !== null) {
    assertNearPx(domPlayhead, expected, `${label}: playhead position`);
    return;
  }
  // Canvas playhead: a light vertical line drawn above the waveform body.
  const columns = await page.locator("#waveform").evaluate((canvas) => {
    const context = canvas.getContext("2d");
    const scale = canvas.width / canvas.getBoundingClientRect().width;
    const row = Math.round(4 * scale);
    const pixels = context.getImageData(0, row, canvas.width, 1).data;
    const found = [];
    for (let x = 0; x < canvas.width; x += 1) {
      const [r, g, b] = [pixels[x * 4], pixels[x * 4 + 1], pixels[x * 4 + 2]];
      if (r > 190 && g > 190 && b > 190) found.push(x / scale);
    }
    return found;
  });
  assert.ok(
    columns.some((x) => Math.abs(x - expected) <= PX_TOLERANCE),
    `${label}: playhead should be drawn at x≈${expected.toFixed(1)}; light columns at ${JSON.stringify(columns.slice(0, 8))}`,
  );
}

// ---------------------------------------------------------------------------------------------
// Interaction helpers
// ---------------------------------------------------------------------------------------------

async function pressUntilStable(page, button, beats, duration, key) {
  let view = await visibleWindow(page, beats, duration);
  for (let press = 0; press < 30; press += 1) {
    if (await button.isDisabled()) return view;
    await button.focus();
    await page.keyboard.press(key);
    const next = await visibleWindow(page, beats, duration);
    if (Math.abs(span(next) - span(view)) <= pxTime(view, 1)) return next;
    view = next;
  }
  assert.fail("zoom controls should reach a bounded limit within 30 presses");
}

async function drag(page, fromX, toX, y) {
  await page.mouse.move(fromX, y);
  await page.mouse.down();
  await page.mouse.move(fromX + (toX - fromX) / 2, y, { steps: 4 });
  await page.mouse.move(toX, y, { steps: 4 });
  await page.mouse.up();
  await nextFrame(page);
}

async function touchPinch(context, page, centerX, y, startGap, endGap) {
  const session = await context.newCDPSession(page);
  const points = (gap) => [
    { x: centerX - gap / 2, y, id: 1, radiusX: 2, radiusY: 2, force: 1 },
    { x: centerX + gap / 2, y, id: 2, radiusX: 2, radiusY: 2, force: 1 },
  ];
  await session.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: points(startGap) });
  for (let step = 1; step <= 8; step += 1) {
    const gap = startGap + ((endGap - startGap) * step) / 8;
    await session.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: points(gap) });
    await nextFrame(page);
  }
  await session.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  await nextFrame(page);
  await session.detach();
}

async function setPlaybackTime(page, time) {
  await page.locator("#audio-player").evaluate((element, nextTime) => {
    element.currentTime = nextTime;
    element.dispatchEvent(new Event("timeupdate"));
  }, time);
  await nextFrame(page);
}

async function currentTime(page) {
  return page.locator("#audio-player").evaluate((element) => element.currentTime);
}

async function overviewHash(page, overview) {
  return overview.evaluate((element) => {
    const canvas = element instanceof HTMLCanvasElement ? element : element.querySelector("canvas");
    if (!canvas) return element.innerHTML.length;
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hash = 2166136261;
    for (let index = 0; index < data.length; index += 4) {
      hash ^= data[index] + data[index + 1] * 3 + data[index + 2] * 7;
      hash = Math.imul(hash, 16777619);
    }
    return `${canvas.width}x${canvas.height}:${hash >>> 0}`;
  });
}

async function nextFrame(page) {
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

function parseClock(text) {
  const match = /(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)/.exec(text);
  assert.ok(match, `expected a m:ss.s time in "${text}"`);
  return Number(match[1] ?? 0) * 3600 + Number(match[2]) * 60 + Number(match[3]);
}

function assertNear(actual, expected, tolerance, label) {
  assert.ok(
    Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected} ± ${tolerance}, got ${actual}`,
  );
}

function assertNearPx(actual, expected, label, tolerance = PX_TOLERANCE) {
  assert.ok(
    Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: expected x≈${expected.toFixed(1)} ± ${tolerance}px, got ${actual}`,
  );
}
