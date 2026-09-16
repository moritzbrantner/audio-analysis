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
  await page.waitForFunction(() => document.querySelectorAll(".waveform-beat-marker").length > 0, null, {
    timeout: 30_000,
  });

  const reportText = await page.locator("#raw-json").textContent();
  const report = JSON.parse(reportText ?? "{}");
  assert.ok(Array.isArray(report.rhythm?.beats) && report.rhythm.beats.length > 0, "expected detected beats");

  const toggle = page.locator("#beat-overlay-toggle");
  const overlayLayer = page.locator(".waveform-overlay-layer");
  assert.equal(await toggle.isChecked(), true, "beat overlay should be enabled by default when beats are available");
  assert.equal(await toggle.isEnabled(), true, "beat overlay toggle should be enabled when beats are available");
  assert.match(await toggle.getAttribute("aria-label"), /detected beat/i);
  assert.equal(await overlayLayer.isVisible(), true, "beat overlay layer should be visible by default");
  assert.ok((await page.locator(".waveform-beat-marker").count()) > 0, "expected rendered beat markers");
  assert.equal(
    await page.locator("#waveform-presentation").count(),
    0,
    "beat overlay must not add a second animated waveform canvas",
  );

  const stageBefore = await page.locator(".waveform-stage").boundingBox();
  assert.ok(stageBefore, "expected stable waveform stage");
  assert.ok(stageBefore.height >= 219 && stageBefore.height <= 221, `expected fixed 220px waveform height; got ${stageBefore.height}`);

  const waveformBeforeToggle = await waveformHash(page, []);
  await toggle.uncheck();
  await nextFrame(page);
  assert.equal(await overlayLayer.isVisible(), false, "disabling beat overlay should hide only the annotation layer");
  const waveformAfterToggle = await waveformHash(page, []);
  assert.equal(waveformAfterToggle, waveformBeforeToggle, "toggling beat annotations must not rewrite waveform pixels");
  await toggle.check();
  await nextFrame(page);
  assert.equal(await overlayLayer.isVisible(), true, "re-enabling beat overlay should restore annotations");

  const firstMarkerRatioBefore = await markerPositionRatio(page);
  await page.setViewportSize({ width: 900, height: 900 });
  await nextFrame(page);
  const firstMarkerRatioAfter = await markerPositionRatio(page);
  assert.ok(
    Math.abs(firstMarkerRatioAfter - firstMarkerRatioBefore) < 0.002,
    `beat marker should keep its normalized time position after resize; before=${firstMarkerRatioBefore}, after=${firstMarkerRatioAfter}`,
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
  console.log("Waveform beat-overlay and stability regression passed");
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
