import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { chromium } from "playwright";

const baseUrl = process.env.PAGES_E2E_BASE_URL ?? "http://127.0.0.1:4173";
const browser = await chromium.launch({ headless: true });

try {
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  const pageErrors = [];
  const unexpectedRequestFailures = [];

  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("requestfailed", (request) => {
    if (!request.url().startsWith("http://127.0.0.1:3000/")) {
      unexpectedRequestFailures.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText ?? "failed"}`);
    }
  });

  await page.route("http://127.0.0.1:3000/**", async (route) => {
    await route.fulfill({
      status: 404,
      contentType: "application/json",
      headers: { "Access-Control-Allow-Origin": "*" },
      body: JSON.stringify({ error: "backend intentionally unavailable in browser smoke test" }),
    });
  });

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await assertText(page, "h1", "Audio Inspector");

  await page.locator('[data-example="tone"]').click();
  await page.locator("#report").waitFor({ state: "visible", timeout: 30_000 });
  await assertText(page, "#report-title", "440-hz-reference-tone.wav");

  const toneReport = await readRenderedReport(page);
  assert.equal(toneReport.schemaVersion, "audio-analysis-inspector/v1");
  assert.ok(
    Number.isFinite(toneReport.source.sampleRate) && toneReport.source.sampleRate > 0,
    `browser-decoded sample rate should be positive; got ${toneReport.source.sampleRate}`,
  );
  assert.equal(toneReport.source.channels, 1);
  assert.ok(toneReport.source.durationSeconds > 2.9 && toneReport.source.durationSeconds < 3.1);
  assertPackageExecution(toneReport, ["core", "fourier", "pitch"]);
  assertSuccessfulOperation(toneReport.raw?.core, "audio.levels");
  assertSuccessfulOperation(toneReport.raw?.fourier?.spectrum, "audio.fourier.spectrum");
  assertSuccessfulOperation(toneReport.raw?.pitch?.estimate, "audio.pitch.estimate");
  assertInRange(toneReport.spectrum?.dominantFrequencyHz, 430, 450, "tone dominant frequency");
  assertInRange(toneReport.pitch?.frequencyHz, 430, 450, "tone pitch estimate");

  const downloadPromise = page.waitForEvent("download");
  await page.locator("#export-json").click();
  const download = await downloadPromise;
  assert.equal(download.suggestedFilename(), "440-hz-reference-tone.audio-analysis.json");
  const downloadPath = await download.path();
  assert.ok(downloadPath, "expected Playwright to expose the downloaded report path");
  const exportedReport = JSON.parse(await readFile(downloadPath, "utf8"));
  assert.equal(exportedReport.schemaVersion, "audio-analysis-inspector/v1");
  assert.equal(exportedReport.source.name, toneReport.source.name);
  assert.equal(exportedReport.runtime.defaultMode, "client-wasm");

  await page.locator("#choose-another").click();
  await page.locator('[data-example="clicks"]').click();
  await page.locator("#report").waitFor({ state: "visible", timeout: 30_000 });
  await assertText(page, "#report-title", "120-bpm-click-track.wav");

  const clickReport = await readRenderedReport(page);
  assertPackageExecution(clickReport, ["core", "fourier", "rhythm"]);
  assertSuccessfulOperation(clickReport.raw?.rhythm, "audio.rhythm.analyze");
  assertTempoFamily(clickReport.rhythm, 120);
  assert.ok(Array.isArray(clickReport.rhythm?.beats) && clickReport.rhythm.beats.length >= 8, "expected a rendered beat path");

  assert.deepEqual(pageErrors, [], `browser page errors:\n${pageErrors.join("\n")}`);
  assert.deepEqual(
    unexpectedRequestFailures,
    [],
    `unexpected browser request failures:\n${unexpectedRequestFailures.join("\n")}`,
  );

  console.log("Audio Inspector real-browser smoke test passed");
} finally {
  await browser.close();
}

async function readRenderedReport(page) {
  await page.waitForFunction(() => Boolean(document.querySelector("#raw-json")?.textContent?.trim()), null, {
    timeout: 30_000,
  });
  const text = await page.locator("#raw-json").textContent();
  assert.ok(text?.trim(), "expected rendered raw report JSON");
  return JSON.parse(text);
}

function assertPackageExecution(report, packageIds) {
  const packages = new Map((report.runtime?.packages ?? []).map((entry) => [entry.id, entry]));
  for (const packageId of packageIds) {
    const entry = packages.get(packageId);
    assert.ok(entry, `missing ${packageId} package provenance`);
    assert.equal(
      entry.available,
      true,
      `${packageId} WASM package should be available; initialization error: ${entry.error ?? "none reported"}`,
    );
    assert.equal(entry.runtime, "client-wasm");
  }
}

function assertSuccessfulOperation(result, operation) {
  assert.ok(result, `expected ${operation} result`);
  assert.equal(result.operation, operation);
  assert.equal(result.error, undefined, `${operation} should not return an analyzer error`);
  assert.ok(result.value && typeof result.value === "object", `${operation} should return a structured value`);
}

function assertTempoFamily(rhythm, targetBpm) {
  const bpm = rhythm?.bpm;
  assert.equal(typeof bpm, "number", "click-track primary tempo should be numeric");
  assert.ok(Number.isFinite(bpm), "click-track primary tempo should be finite");

  const octaveEquivalent = [bpm / 2, bpm, bpm * 2].some((candidate) => Math.abs(candidate - targetBpm) <= 10);
  assert.ok(
    octaveEquivalent,
    `primary tempo ${bpm} BPM should be half-time, nominal, or double-time equivalent to ${targetBpm} BPM`,
  );

  const candidates = Array.isArray(rhythm?.tempoCandidates) ? rhythm.tempoCandidates : [];
  assert.ok(
    candidates.some((candidate) => typeof candidate?.bpm === "number" && Math.abs(candidate.bpm - targetBpm) <= 10),
    `expected a tempo candidate near ${targetBpm} BPM; got ${JSON.stringify(candidates)}`,
  );
}

async function assertText(page, selector, expected) {
  const text = (await page.locator(selector).textContent())?.trim();
  assert.equal(text, expected);
}

function assertInRange(value, min, max, label) {
  assert.equal(typeof value, "number", `${label} should be numeric`);
  assert.ok(Number.isFinite(value), `${label} should be finite`);
  assert.ok(value >= min && value <= max, `${label} ${value} should be between ${min} and ${max}`);
}
