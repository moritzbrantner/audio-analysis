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
  await assertText(page, "h1", "Understand an audio file in seconds.");
  await page.locator("#choose-file").waitFor({ state: "visible" });
  assert.match(await page.locator('[data-example="tone"]').innerText(), /Expect A4/);
  assert.match(await page.locator('[data-example="clicks"]').innerText(), /half\/double-time ambiguity/);
  assert.equal(await page.locator("#technical").evaluate((element) => element.open), false);

  await page.locator('[data-example="tone"]').click();
  await page.locator("#report").waitFor({ state: "visible", timeout: 30_000 });
  await assertText(page, "#report-title", "440-hz-reference-tone.wav");
  await page.locator(".report-nav").waitFor({ state: "visible" });

  const overviewComesFirst = await page.evaluate(() => {
    const overview = document.querySelector("#overview");
    const waveform = document.querySelector("#waveform-section");
    return Boolean(overview && waveform && (overview.compareDocumentPosition(waveform) & Node.DOCUMENT_POSITION_FOLLOWING));
  });
  assert.equal(overviewComesFirst, true, "overview and findings should precede waveform instrumentation");

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
  assert.match(await page.locator("#findings").innerText(), /440/);

  await page.waitForFunction(() => {
    const player = document.querySelector("#audio-player");
    return player && Number.isFinite(player.duration) && player.duration > 0;
  });

  const waveformBox = await page.locator("#waveform").boundingBox();
  assert.ok(waveformBox, "expected a rendered waveform canvas");
  await page.locator("#waveform").click({
    position: { x: waveformBox.width / 2, y: waveformBox.height / 2 },
  });
  const midpoint = await page.locator("#audio-player").evaluate((element) => element.currentTime);
  assertInRange(midpoint, 1.2, 1.8, "waveform click seek position");
  assert.match(await page.locator("#waveform-readout").innerText(), /sample amplitude/);

  await page.locator("#waveform").focus();
  await page.keyboard.press("Home");
  assertInRange(await page.locator("#audio-player").evaluate((element) => element.currentTime), 0, 0.05, "waveform Home key");
  await page.keyboard.press("End");
  assertInRange(await page.locator("#audio-player").evaluate((element) => element.currentTime), 2.9, 3.1, "waveform End key");

  const spectralBox = await page.locator("#spectral-timeline").boundingBox();
  assert.ok(spectralBox, "expected a rendered spectral timeline");
  await page.locator("#spectral-timeline").hover({
    position: { x: spectralBox.width / 2, y: spectralBox.height / 2 },
  });
  assert.match(await page.locator("#spectral-readout").innerText(), /Frame \d+\/\d+/);
  await page.locator("#spectral-timeline").focus();
  await page.keyboard.press("End");
  assert.match(await page.locator("#spectral-readout").innerText(), /Frame \d+\/\d+/);

  await page.locator('.report-nav a[href="#technical"]').click();
  assert.equal(await page.locator("#technical").evaluate((element) => element.open), true);
  assert.match(await page.locator("#coverage-file").innerText(), /Full file/);
  assert.match(await page.locator("#glossary-title").innerText(), /What these terms mean/);

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
  assertBeatPath(clickReport.rhythm?.beats);
  assert.match(await page.locator("#rhythm-content").innerText(), /BPM/);

  assert.deepEqual(pageErrors, [], `browser page errors:\n${pageErrors.join("\n")}`);
  assert.deepEqual(
    unexpectedRequestFailures,
    [],
    `unexpected browser request failures:\n${unexpectedRequestFailures.join("\n")}`,
  );

  console.log("Audio Inspector real-browser usability smoke test passed");
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

function assertBeatPath(beats) {
  assert.ok(Array.isArray(beats) && beats.length > 0, "expected a non-empty rendered beat path");
  const timestamps = beats.map((beat) => beat?.timestampSeconds);
  assert.ok(
    timestamps.every((timestamp) => typeof timestamp === "number" && Number.isFinite(timestamp) && timestamp >= 0),
    `beat timestamps should be finite and non-negative; got ${JSON.stringify(timestamps)}`,
  );
  assert.ok(
    timestamps.every((timestamp, index) => index === 0 || timestamp > timestamps[index - 1]),
    `beat timestamps should increase strictly; got ${JSON.stringify(timestamps)}`,
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
