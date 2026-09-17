import assert from "node:assert/strict";
import { chromium } from "playwright";

const baseUrl = process.env.PAGES_E2E_BASE_URL ?? "http://127.0.0.1:4173";
const browser = await chromium.launch({ headless: true });

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.locator('[data-example="clicks"]').click();
  await page.locator("#report").waitFor({ state: "visible", timeout: 30_000 });
  await page.waitForFunction(
    () =>
      document.querySelectorAll(".waveform-beat-marker").length > 0 &&
      document.querySelectorAll(".waveform-section-segment").length > 0,
    null,
    { timeout: 30_000 },
  );

  const reportText = await page.locator("#raw-json").textContent();
  const report = JSON.parse(reportText ?? "{}");
  assert.ok(Array.isArray(report.rhythm?.beats) && report.rhythm.beats.length > 0, "expected detected beats");
  assert.ok(Array.isArray(report.rhythm?.sections) && report.rhythm.sections.length > 0, "expected detected sections");

  const beatToggle = page.locator("#beat-overlay-toggle");
  const sectionToggle = page.locator("#section-overlay-toggle");
  const overlayLayer = page.locator(".waveform-overlay-layer");
  const beatLayer = page.locator(".waveform-beat-layer");
  const sectionLayer = page.locator(".waveform-section-layer");

  assert.equal(await beatToggle.isChecked(), true, "beat overlay should be enabled by default when beats are available");
  assert.equal(await beatToggle.isEnabled(), true, "beat overlay toggle should be enabled when beats are available");
  assert.match(await beatToggle.getAttribute("aria-label"), /detected beat/i);
  assert.equal(await sectionToggle.isChecked(), true, "section overlay should be enabled by default when sections are available");
  assert.equal(await sectionToggle.isEnabled(), true, "section overlay toggle should be enabled when sections are available");
  assert.match(await sectionToggle.getAttribute("aria-label"), /detected section/i);
  assert.equal(await overlayLayer.isVisible(), true, "waveform annotation layer should be visible by default");
  assert.ok((await page.locator(".waveform-beat-marker").count()) > 0, "expected rendered beat markers");
  assert.ok((await page.locator(".waveform-section-segment").count()) > 0, "expected rendered section bands");
  assert.ok((await page.locator(".waveform-section-label").count()) > 0, "expected section identity labels");
  assert.equal(
    await page.locator("#waveform-presentation").count(),
    0,
    "rhythm overlays must not add a second animated waveform canvas",
  );

  const stageBefore = await page.locator(".waveform-stage").boundingBox();
  assert.ok(stageBefore, "expected stable waveform stage");
  assert.ok(stageBefore.height >= 219 && stageBefore.height <= 221, `expected fixed 220px waveform height; got ${stageBefore.height}`);

  const waveformBeforeToggles = await waveformHash(page, []);
  await beatToggle.uncheck();
  await nextFrame(page);
  assert.equal(await beatLayer.isVisible(), false, "disabling beat overlay should hide beat annotations");
  assert.equal(await sectionLayer.isVisible(), true, "disabling beat overlay should preserve section annotations");
  assert.equal(await overlayLayer.isVisible(), true, "section annotations should keep the overlay layer visible");
  assert.equal(await waveformHash(page, []), waveformBeforeToggles, "toggling beat annotations must not rewrite waveform pixels");

  await sectionToggle.uncheck();
  await nextFrame(page);
  assert.equal(await sectionLayer.isVisible(), false, "disabling section overlay should hide section annotations");
  assert.equal(await overlayLayer.isVisible(), false, "overlay container should hide when all annotations are disabled");
  assert.equal(await waveformHash(page, []), waveformBeforeToggles, "toggling section annotations must not rewrite waveform pixels");

  await beatToggle.check();
  await sectionToggle.check();
  await nextFrame(page);
  assert.equal(await beatLayer.isVisible(), true, "re-enabling beat overlay should restore beat annotations");
  assert.equal(await sectionLayer.isVisible(), true, "re-enabling section overlay should restore section annotations");

  const firstBeatRatioBefore = await markerPositionRatio(page);
  const firstSectionBefore = await sectionPositionRatios(page);
  await page.setViewportSize({ width: 900, height: 900 });
  await nextFrame(page);
  const firstBeatRatioAfter = await markerPositionRatio(page);
  const firstSectionAfter = await sectionPositionRatios(page);
  assert.ok(
    Math.abs(firstBeatRatioAfter - firstBeatRatioBefore) < 0.002,
    `beat marker should keep its normalized time position after resize; before=${firstBeatRatioBefore}, after=${firstBeatRatioAfter}`,
  );
  assert.ok(
    Math.abs(firstSectionAfter.start - firstSectionBefore.start) < 0.002 &&
      Math.abs(firstSectionAfter.end - firstSectionBefore.end) < 0.002,
    `section should keep its normalized time range after resize; before=${JSON.stringify(firstSectionBefore)}, after=${JSON.stringify(firstSectionAfter)}`,
  );

  const duration = report.source?.durationSeconds;
  assert.equal(typeof duration, "number");
  const startTime = 0;
  const laterTime = duration / 2;
  await setPlaybackTime(page, startTime);
  const stableBefore = await waveformHash(page, [startTime, laterTime], duration);

  for (let index = 1; index <= 40; index += 1) {
    await setPlaybackTime(page, (duration * index) / 40);
  }
  await setPlaybackTime(page, laterTime);
  const stableAfter = await waveformHash(page, [startTime, laterTime], duration);
  assert.equal(
    stableAfter,
    stableBefore,
    "waveform pixels outside the moving cursor should remain unchanged across repeated playback redraws",
  );

  const stageAfter = await page.locator(".waveform-stage").boundingBox();
  assert.ok(stageAfter, "expected stable waveform stage after repeated redraws");
  assert.ok(stageAfter.height >= 219 && stageAfter.height <= 221, "waveform height should remain fixed after repeated redraws");

  assert.deepEqual(pageErrors, [], `browser page errors:\n${pageErrors.join("\n")}`);
  console.log("Waveform beat/section overlays and stability regression passed");
} finally {
  await browser.close();
}

async function markerPositionRatio(page) {
  const marker = page.locator(".waveform-beat-marker").first();
  const stage = page.locator(".waveform-stage");
  const markerBox = await marker.boundingBox();
  const stageBox = await stage.boundingBox();
  assert.ok(markerBox && stageBox && stageBox.width > 0, "expected beat marker and waveform stage geometry");
  return (markerBox.x - stageBox.x) / stageBox.width;
}

async function sectionPositionRatios(page) {
  const section = page.locator(".waveform-section-segment").first();
  const stage = page.locator(".waveform-stage");
  const sectionBox = await section.boundingBox();
  const stageBox = await stage.boundingBox();
  assert.ok(sectionBox && stageBox && stageBox.width > 0, "expected section and waveform stage geometry");
  return {
    start: (sectionBox.x - stageBox.x) / stageBox.width,
    end: (sectionBox.x + sectionBox.width - stageBox.x) / stageBox.width,
  };
}

async function setPlaybackTime(page, time) {
  await page.locator("#audio-player").evaluate((element, nextTime) => {
    element.currentTime = nextTime;
    element.dispatchEvent(new Event("timeupdate"));
  }, time);
  await nextFrame(page);
}

async function nextFrame(page) {
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => resolve())));
}

async function waveformHash(page, maskedTimes = [], duration = null) {
  return page.locator("#waveform").evaluate(
    (canvas, { maskedTimes: times, durationSeconds }) => {
      const context = canvas.getContext("2d");
      const { data, width, height } = context.getImageData(0, 0, canvas.width, canvas.height);
      const rect = canvas.getBoundingClientRect();
      const maskColumns = new Set();
      if (Number.isFinite(durationSeconds) && durationSeconds > 0) {
        for (const time of times) {
          const logicalX = (Math.max(0, Math.min(durationSeconds, time)) / durationSeconds) * rect.width;
          const backingX = Math.round((logicalX / Math.max(1, rect.width)) * width);
          for (let delta = -4; delta <= 4; delta += 1) maskColumns.add(backingX + delta);
        }
      }

      let hash = 2166136261;
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          if (maskColumns.has(x)) continue;
          const offset = (y * width + x) * 4;
          for (let channel = 0; channel < 4; channel += 1) {
            hash ^= data[offset + channel];
            hash = Math.imul(hash, 16777619) >>> 0;
          }
        }
      }
      return hash;
    },
    { maskedTimes, durationSeconds: duration },
  );
}
