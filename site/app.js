const MAX_SURFACE_SAMPLES = 192_000;
const REPRESENTATIVE_SAMPLES = 131_072;
const RHYTHM_SECONDS = 20;
const RHYTHM_RATE = 16_000;
const MAX_STAT_FRAMES = 5_000_000;
const CLIP_THRESHOLD = 0.999;
const NEAR_SILENCE_THRESHOLD = 0.001;

const analyzerDefinitions = {
  core: {
    label: "audio-analysis-core",
    module: "./wasm/audio-analysis-core/index.js",
  },
  fourier: {
    label: "audio-analysis-fourier",
    module: "./wasm/audio-analysis-fourier/index.js",
  },
  pitch: {
    label: "audio-analysis-pitch",
    module: "./wasm/audio-analysis-pitch/index.js",
  },
  rhythm: {
    label: "audio-analysis-rhythm",
    module: "./wasm/audio-analysis-rhythm/index.js",
  },
};

const state = {
  audioBuffer: null,
  currentObjectUrl: null,
  report: null,
  analyzers: null,
  backend: null,
};

const elements = {
  inputPanel: document.querySelector("#input-panel"),
  dropZone: document.querySelector("#drop-zone"),
  fileInput: document.querySelector("#file-input"),
  inputError: document.querySelector("#input-error"),
  loadingPanel: document.querySelector("#loading-panel"),
  loadingTitle: document.querySelector("#loading-title"),
  loadingDetail: document.querySelector("#loading-detail"),
  report: document.querySelector("#report"),
  reportTitle: document.querySelector("#report-title"),
  exportJson: document.querySelector("#export-json"),
  chooseAnother: document.querySelector("#choose-another"),
  audioPlayer: document.querySelector("#audio-player"),
  waveform: document.querySelector("#waveform"),
  spectralTimeline: document.querySelector("#spectral-timeline"),
  statisticsCoverage: document.querySelector("#statistics-coverage"),
  spectralCoverage: document.querySelector("#spectral-coverage"),
  rhythmCoverage: document.querySelector("#rhythm-coverage"),
  findings: document.querySelector("#findings"),
  levelsMetrics: document.querySelector("#levels-metrics"),
  qualityMetrics: document.querySelector("#quality-metrics"),
  spectrumMetrics: document.querySelector("#spectrum-metrics"),
  pitchContent: document.querySelector("#pitch-content"),
  rhythmContent: document.querySelector("#rhythm-content"),
  backendSummary: document.querySelector("#backend-summary"),
  backendBadge: document.querySelector("#backend-badge"),
  backendContent: document.querySelector("#backend-content"),
  provenance: document.querySelector("#provenance"),
  rawJson: document.querySelector("#raw-json"),
  summaryDuration: document.querySelector("#summary-duration"),
  summaryRate: document.querySelector("#summary-rate"),
  summaryChannels: document.querySelector("#summary-channels"),
  summaryPeak: document.querySelector("#summary-peak"),
  summaryRms: document.querySelector("#summary-rms"),
  summarySize: document.querySelector("#summary-size"),
};

wireUi();
const backendInitialization = initializeBackendBoundary();

function wireUi() {
  elements.fileInput.addEventListener("change", () => {
    const file = elements.fileInput.files?.[0];
    if (file) void analyzeFile(file);
  });

  for (const eventName of ["dragenter", "dragover"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.add("is-dragging");
    });
  }

  for (const eventName of ["dragleave", "drop"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove("is-dragging");
    });
  }

  elements.dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) void analyzeFile(file);
  });

  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", async () => {
      clearError();
      const kind = button.getAttribute("data-example");
      const name = button.getAttribute("data-example-name") ?? "Example audio";
      if (!kind) return;
      try {
        const file = createExampleFile(kind, name);
        await analyzeFile(file);
      } catch (error) {
        showError(`Could not create the example: ${errorMessage(error)}`);
      }
    });
  });

  elements.exportJson.addEventListener("click", exportReport);
  elements.chooseAnother.addEventListener("click", resetInspector);

  window.addEventListener("resize", () => {
    if (!state.audioBuffer || !state.report) return;
    drawWaveform(elements.waveform, state.audioBuffer);
    drawSpectralTimeline(elements.spectralTimeline, responseValue(state.report.raw?.fourier?.spectrogram));
  });
}

async function analyzeFile(file) {
  clearError();
  setLoading(true, "Decoding audio…", "The browser is decoding the selected file locally.");
  elements.report.hidden = true;

  try {
    const arrayBuffer = await file.arrayBuffer();
    const context = new AudioContext();
    let decoded;
    try {
      decoded = await context.decodeAudioData(arrayBuffer.slice(0));
    } finally {
      await context.close();
    }

    if (!decoded.length || !decoded.numberOfChannels) {
      throw new Error("The decoded audio buffer is empty.");
    }

    state.audioBuffer = decoded;
    replacePlayerSource(file);

    setLoading(true, "Scanning the signal…", "Computing file-level levels, dynamics, channel, and clipping statistics.");
    const statistics = scanAudioBuffer(decoded);

    setLoading(true, "Running Rust/WASM analyzers…", "Loading core, Fourier, pitch, and rhythm package surfaces.");
    const analyzers = await loadAnalyzers();
    state.analyzers = analyzers;

    const representative = monoCenterWindow(decoded, REPRESENTATIVE_SAMPLES);
    const rhythmWindow = monoCenterWindowBySeconds(decoded, RHYTHM_SECONDS);
    const rhythmSamples = resampleLinear(rhythmWindow.samples, decoded.sampleRate, Math.min(decoded.sampleRate, RHYTHM_RATE));
    const rhythmRate = Math.min(decoded.sampleRate, RHYTHM_RATE);
    const fftSize = chooseFftSize(representative.samples.length);

    const [coreResult, spectrumResult, spectrogramResult, featuresResult, pitchResult, keyResult, rhythmResult] =
      await Promise.all([
        safeRun(analyzers.core, "audio.levels", {
          samples: representative.samples,
          sampleRate: decoded.sampleRate,
          channels: 1,
          previewSamples: 64,
        }),
        safeRun(analyzers.fourier, "audio.fourier.spectrum", {
          samples: representative.samples,
          sampleRate: decoded.sampleRate,
          fftSize,
          maxBins: 64,
        }),
        safeRun(analyzers.fourier, "audio.fourier.spectrogram", {
          samples: representative.samples,
          sampleRate: decoded.sampleRate,
          fftSize,
          hopSize: Math.max(128, fftSize / 4),
          maxFrames: 16,
        }),
        safeRun(analyzers.fourier, "audio.fourier.features", {
          samples: representative.samples,
          sampleRate: decoded.sampleRate,
          fftSize,
          hopSize: Math.max(128, fftSize / 4),
          melBandCount: 24,
        }),
        safeRun(analyzers.pitch, "audio.pitch.estimate", {
          samples: representative.samples,
          sampleRate: decoded.sampleRate,
        }),
        safeRun(analyzers.pitch, "audio.pitch.key", {
          samples: representative.samples,
          sampleRate: decoded.sampleRate,
          profile: "ensemble",
        }),
        safeRun(analyzers.rhythm, "audio.rhythm.analyze", {
          samples: rhythmSamples,
          sampleRate: rhythmRate,
          minBpm: 45,
          maxBpm: 220,
        }),
      ]);

    await backendInitialization;

    const packageProvenance = Object.entries(analyzers).map(([id, analyzer]) => ({
      id,
      library: analyzer.surface?.library ?? analyzerDefinitions[id].label,
      version: analyzer.surface?.version ?? "unknown",
      runtime: "client-wasm",
      available: !analyzer.error,
      error: analyzer.error ?? null,
    }));

    const report = buildReport({
      file,
      buffer: decoded,
      statistics,
      representative,
      rhythmWindow: {
        startSample: rhythmWindow.startSample,
        sourceSampleCount: rhythmWindow.samples.length,
        analysisSampleRate: rhythmRate,
        analysisSampleCount: rhythmSamples.length,
      },
      packageProvenance,
      results: {
        core: coreResult,
        spectrum: spectrumResult,
        spectrogram: spectrogramResult,
        features: featuresResult,
        pitch: pitchResult,
        key: keyResult,
        rhythm: rhythmResult,
      },
    });

    state.report = report;
    renderReport(report, decoded);
  } catch (error) {
    showError(errorMessage(error));
  } finally {
    setLoading(false);
  }
}

async function loadAnalyzers() {
  const entries = await Promise.all(
    Object.entries(analyzerDefinitions).map(async ([id, definition]) => {
      try {
        const module = await import(definition.module);
        await module.init();
        const surface = await module.packageSurface();
        return [id, { ...module, surface, error: null }];
      } catch (error) {
        return [id, { surface: null, runOperation: null, error: errorMessage(error) }];
      }
    }),
  );
  return Object.fromEntries(entries);
}

async function safeRun(analyzer, operation, input) {
  if (!analyzer?.runOperation) {
    return { operation, error: analyzer?.error ?? "WASM analyzer unavailable", value: null };
  }
  try {
    return await analyzer.runOperation({ operation, input });
  } catch (error) {
    return { operation, error: errorMessage(error), value: null };
  }
}

function buildReport({ file, buffer, statistics, representative, rhythmWindow, packageProvenance, results }) {
  const spectralValue = responseValue(results.features);
  const pitchValue = responseValue(results.pitch);
  const keyValue = responseValue(results.key);
  const rhythmValue = responseValue(results.rhythm);
  const findings = buildFindings(statistics, spectralValue, pitchValue, keyValue, rhythmValue);

  return {
    schemaVersion: "audio-analysis-inspector/v1",
    generatedAt: new Date().toISOString(),
    source: {
      name: file.name,
      mediaType: file.type || "unknown",
      byteLength: file.size,
      durationSeconds: buffer.duration,
      sampleRate: buffer.sampleRate,
      channels: buffer.numberOfChannels,
      framesPerChannel: buffer.length,
    },
    runtime: {
      defaultMode: "client-wasm",
      privacy: isGitHubPages()
        ? "Public Pages mode: decoded audio remains in the browser and is not uploaded."
        : "Browser-local analysis remains the default; a configured local backend may expose optional heavier capabilities.",
      backend: state.backend,
      packages: packageProvenance,
    },
    coverage: {
      fileStatistics: statistics.coverage,
      spectralPitchKey: {
        kind: "representative-center-window",
        startSeconds: representative.startSample / buffer.sampleRate,
        durationSeconds: representative.samples.length / buffer.sampleRate,
        sampleRate: buffer.sampleRate,
        sampleCount: representative.samples.length,
        packageSurfaceLimit: MAX_SURFACE_SAMPLES,
      },
      rhythm: {
        kind: "representative-center-window-resampled",
        startSeconds: rhythmWindow.startSample / buffer.sampleRate,
        sourceDurationSeconds: rhythmWindow.sourceSampleCount / buffer.sampleRate,
        analysisSampleRate: rhythmWindow.analysisSampleRate,
        analysisSampleCount: rhythmWindow.analysisSampleCount,
      },
    },
    overview: {
      peak: statistics.peak,
      peakDbfs: amplitudeToDb(statistics.peak),
      rms: statistics.rms,
      rmsDbfs: amplitudeToDb(statistics.rms),
      meanAbsolute: statistics.meanAbsolute,
      dcOffset: statistics.dcOffset,
      crestFactorDb: statistics.rms > 0 ? 20 * Math.log10(statistics.peak / statistics.rms) : null,
    },
    quality: {
      clipThreshold: CLIP_THRESHOLD,
      clippedSampleCount: statistics.clippedSampleCount,
      clippedSamplePercent: statistics.clippedSamplePercent,
      firstClippedTimesSeconds: statistics.firstClippedTimesSeconds,
      nearSilenceThreshold: NEAR_SILENCE_THRESHOLD,
      nearSilentSamplePercent: statistics.nearSilentSamplePercent,
      stereoCorrelation: statistics.stereoCorrelation,
      channelRms: statistics.channelRms,
      stereoBalanceDb: statistics.stereoBalanceDb,
    },
    findings,
    spectrum: responseValue(results.spectrum),
    spectralFeatures: spectralValue,
    spectrogram: responseValue(results.spectrogram),
    pitch: pitchValue,
    musicalKey: keyValue,
    rhythm: rhythmValue,
    raw: {
      core: results.core,
      fourier: {
        spectrum: results.spectrum,
        spectrogram: results.spectrogram,
        features: results.features,
      },
      pitch: {
        estimate: results.pitch,
        key: results.key,
      },
      rhythm: results.rhythm,
    },
  };
}

function scanAudioBuffer(buffer) {
  const stride = Math.max(1, Math.ceil(buffer.length / MAX_STAT_FRAMES));
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) => buffer.getChannelData(index));
  const channelSumsSq = new Array(channels.length).fill(0);
  let peak = 0;
  let sumSq = 0;
  let sumAbs = 0;
  let sum = 0;
  let clipped = 0;
  let nearSilent = 0;
  let visitedFrames = 0;
  const clippedFrames = [];

  let stereo = null;
  if (channels.length === 2) {
    stereo = { sumL: 0, sumR: 0, sumLL: 0, sumRR: 0, sumLR: 0 };
  }

  for (let frame = 0; frame < buffer.length; frame += stride) {
    visitedFrames += 1;
    let frameClipped = false;
    for (let channel = 0; channel < channels.length; channel += 1) {
      const sample = channels[channel][frame] ?? 0;
      const absolute = Math.abs(sample);
      peak = Math.max(peak, absolute);
      sumSq += sample * sample;
      sumAbs += absolute;
      sum += sample;
      channelSumsSq[channel] += sample * sample;
      if (absolute >= CLIP_THRESHOLD) {
        clipped += 1;
        frameClipped = true;
      }
      if (absolute <= NEAR_SILENCE_THRESHOLD) nearSilent += 1;
    }

    if (frameClipped && clippedFrames.length < 8) clippedFrames.push(frame);

    if (stereo) {
      const left = channels[0][frame] ?? 0;
      const right = channels[1][frame] ?? 0;
      stereo.sumL += left;
      stereo.sumR += right;
      stereo.sumLL += left * left;
      stereo.sumRR += right * right;
      stereo.sumLR += left * right;
    }
  }

  const visitedSamples = visitedFrames * channels.length;
  const rms = visitedSamples ? Math.sqrt(sumSq / visitedSamples) : 0;
  const channelRms = channelSumsSq.map((value) => (visitedFrames ? Math.sqrt(value / visitedFrames) : 0));
  const stereoCorrelation = stereo ? correlation(stereo, visitedFrames) : null;
  const stereoBalanceDb = channelRms.length === 2 && channelRms[0] > 0 && channelRms[1] > 0
    ? 20 * Math.log10(channelRms[0] / channelRms[1])
    : null;

  return {
    peak,
    rms,
    meanAbsolute: visitedSamples ? sumAbs / visitedSamples : 0,
    dcOffset: visitedSamples ? sum / visitedSamples : 0,
    clippedSampleCount: clipped,
    clippedSamplePercent: visitedSamples ? (clipped / visitedSamples) * 100 : 0,
    nearSilentSamplePercent: visitedSamples ? (nearSilent / visitedSamples) * 100 : 0,
    firstClippedTimesSeconds: clippedFrames.map((frame) => frame / buffer.sampleRate),
    channelRms,
    stereoCorrelation,
    stereoBalanceDb,
    coverage: {
      kind: stride === 1 ? "exact-whole-file" : "deterministic-stride-sample",
      frameStride: stride,
      visitedFrames,
      totalFrames: buffer.length,
      visitedSampleValues: visitedSamples,
    },
  };
}

function correlation(stereo, count) {
  if (count < 2) return null;
  const covariance = stereo.sumLR - (stereo.sumL * stereo.sumR) / count;
  const varianceL = stereo.sumLL - (stereo.sumL * stereo.sumL) / count;
  const varianceR = stereo.sumRR - (stereo.sumR * stereo.sumR) / count;
  const denominator = Math.sqrt(Math.max(0, varianceL) * Math.max(0, varianceR));
  return denominator > 0 ? Math.max(-1, Math.min(1, covariance / denominator)) : null;
}

function monoCenterWindow(buffer, maxSamples) {
  const count = Math.min(maxSamples, buffer.length);
  const startSample = Math.max(0, Math.floor((buffer.length - count) / 2));
  return {
    startSample,
    samples: mixMono(buffer, startSample, count),
  };
}

function monoCenterWindowBySeconds(buffer, seconds) {
  return monoCenterWindow(buffer, Math.min(buffer.length, Math.floor(buffer.sampleRate * seconds)));
}

function mixMono(buffer, startSample, count) {
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) => buffer.getChannelData(index));
  const output = new Array(count);
  for (let offset = 0; offset < count; offset += 1) {
    let sum = 0;
    for (const channel of channels) sum += channel[startSample + offset] ?? 0;
    output[offset] = sum / channels.length;
  }
  return output;
}

function resampleLinear(samples, sourceRate, targetRate) {
  if (samples.length < 2 || sourceRate === targetRate) return samples;
  const outputLength = Math.max(1, Math.floor((samples.length * targetRate) / sourceRate));
  const output = new Array(outputLength);
  const scale = sourceRate / targetRate;
  for (let index = 0; index < outputLength; index += 1) {
    const source = index * scale;
    const left = Math.min(samples.length - 1, Math.floor(source));
    const right = Math.min(samples.length - 1, left + 1);
    const fraction = source - left;
    output[index] = samples[left] * (1 - fraction) + samples[right] * fraction;
  }
  return output;
}

function chooseFftSize(sampleCount) {
  let size = 2048;
  while (size > sampleCount && size > 256) size /= 2;
  return Math.max(256, size);
}

function buildFindings(stats, spectral, pitch, key, rhythm) {
  const findings = [];

  if (stats.clippedSampleCount > 0) {
    findings.push({
      tone: "warning",
      title: "Possible clipping detected",
      detail: `${formatPercent(stats.clippedSamplePercent)} of scanned sample values reached |${CLIP_THRESHOLD}| or above.`,
    });
  } else {
    findings.push({
      tone: "good",
      title: "No sample-level clipping detected",
      detail: `No scanned sample value reached the ${CLIP_THRESHOLD} clipping threshold.`,
    });
  }

  const dominant = finiteNumber(spectral?.dominantFrequencyHz);
  const centroid = finiteNumber(spectral?.centroidHz);
  if (dominant !== null || centroid !== null) {
    findings.push({
      tone: "neutral",
      title: dominant !== null ? `Dominant frequency near ${formatHz(dominant)}` : "Spectral shape measured",
      detail: centroid !== null
        ? `The representative window has a spectral centroid near ${formatHz(centroid)}.`
        : "Frequency-domain analysis completed on the representative window.",
    });
  }

  const pitchFrequency = finiteNumber(pitch?.frequencyHz);
  const pitchConfidence = finiteNumber(pitch?.confidence);
  if (pitchFrequency !== null && pitchConfidence !== null && pitchConfidence >= 0.55) {
    findings.push({
      tone: "neutral",
      title: `Pitch estimate: ${pitch?.noteName ?? formatHz(pitchFrequency)}`,
      detail: `${formatHz(pitchFrequency)} with ${formatPercent(pitchConfidence * 100)} estimator confidence in the representative window.`,
    });
  }

  if (typeof key?.key === "string") {
    const keyConfidence = finiteNumber(key.confidence);
    findings.push({
      tone: "neutral",
      title: `Estimated musical key: ${key.key}`,
      detail: keyConfidence === null
        ? "Key analysis completed on the representative window."
        : `Key confidence: ${formatPercent(keyConfidence * 100)}.`,
    });
  }

  const bpm = finiteNumber(rhythm?.bpm);
  if (bpm !== null) {
    const confidence = finiteNumber(rhythm?.confidence);
    findings.push({
      tone: "neutral",
      title: `Estimated tempo: ${bpm.toFixed(1)} BPM`,
      detail: confidence === null
        ? "Rhythm analysis completed on a bounded representative window."
        : `Tempo confidence: ${formatPercent(confidence * 100)}; analysis uses a bounded, resampled window.`,
    });
  }

  if (stats.nearSilentSamplePercent >= 50) {
    findings.push({
      tone: "neutral",
      title: "Large near-silent sample share",
      detail: `${formatPercent(stats.nearSilentSamplePercent)} of scanned sample values are at or below |${NEAR_SILENCE_THRESHOLD}|. This is a sample-level measure, not a silence-duration detector.`,
    });
  }

  if (stats.stereoCorrelation !== null && stats.stereoCorrelation < -0.2) {
    findings.push({
      tone: "warning",
      title: "Negative stereo correlation",
      detail: `The scanned stereo correlation is ${stats.stereoCorrelation.toFixed(2)}. Strongly negative values can indicate polarity or phase-sensitive material.`,
    });
  }

  return findings.slice(0, 6);
}

function renderReport(report, buffer) {
  elements.inputPanel.hidden = true;
  elements.report.hidden = false;
  elements.reportTitle.textContent = report.source.name;

  elements.summaryDuration.textContent = formatDuration(report.source.durationSeconds);
  elements.summaryRate.textContent = `${Math.round(report.source.sampleRate).toLocaleString()} Hz`;
  elements.summaryChannels.textContent = String(report.source.channels);
  elements.summaryPeak.textContent = formatDb(report.overview.peakDbfs);
  elements.summaryRms.textContent = formatDb(report.overview.rmsDbfs);
  elements.summarySize.textContent = formatBytes(report.source.byteLength);

  const statsCoverage = report.coverage.fileStatistics;
  elements.statisticsCoverage.textContent = statsCoverage.kind === "exact-whole-file"
    ? "exact whole-file scan"
    : `sampled every ${statsCoverage.frameStride} frames`;
  elements.spectralCoverage.textContent = `${report.coverage.spectralPitchKey.durationSeconds.toFixed(1)} s center window`;
  elements.rhythmCoverage.textContent = `${report.coverage.rhythm.sourceDurationSeconds.toFixed(1)} s @ ${Math.round(report.coverage.rhythm.analysisSampleRate / 1000)} kHz`;

  renderFindings(report.findings);
  renderMetricList(elements.levelsMetrics, [
    ["Peak", `${formatNumber(report.overview.peak, 4)} (${formatDb(report.overview.peakDbfs)})`],
    ["RMS", `${formatNumber(report.overview.rms, 4)} (${formatDb(report.overview.rmsDbfs)})`],
    ["Mean absolute", formatNumber(report.overview.meanAbsolute, 4)],
    ["Crest factor", report.overview.crestFactorDb === null ? "—" : `${report.overview.crestFactorDb.toFixed(2)} dB`],
    ["DC offset", formatNumber(report.overview.dcOffset, 5)],
  ]);

  const qualityRows = [
    ["Clipped samples", `${report.quality.clippedSampleCount.toLocaleString()} (${formatPercent(report.quality.clippedSamplePercent)})`],
    ["Near-silent samples", formatPercent(report.quality.nearSilentSamplePercent)],
    ["Channel RMS", report.quality.channelRms.map((value, index) => `Ch ${index + 1}: ${formatDb(amplitudeToDb(value))}`).join(" · ")],
  ];
  if (report.quality.stereoCorrelation !== null) qualityRows.push(["Stereo correlation", report.quality.stereoCorrelation.toFixed(3)]);
  if (report.quality.stereoBalanceDb !== null) qualityRows.push(["L/R RMS balance", `${report.quality.stereoBalanceDb.toFixed(2)} dB`]);
  renderMetricList(elements.qualityMetrics, qualityRows);

  renderSpectrum(report);
  renderPitch(report);
  renderRhythm(report);
  renderBackend(report.runtime.backend);
  renderProvenance(report.runtime.packages);

  elements.rawJson.textContent = JSON.stringify(report, null, 2);
  requestAnimationFrame(() => {
    drawWaveform(elements.waveform, buffer);
    drawSpectralTimeline(elements.spectralTimeline, report.spectrogram);
  });
}

function renderFindings(findings) {
  elements.findings.replaceChildren();
  if (!findings.length) {
    const empty = document.createElement("p");
    empty.textContent = "The available analyzers did not produce enough evidence for headline findings.";
    elements.findings.append(empty);
    return;
  }

  for (const finding of findings) {
    const article = document.createElement("article");
    article.className = `finding finding-${finding.tone}`;
    const title = document.createElement("strong");
    title.textContent = finding.title;
    const detail = document.createElement("p");
    detail.textContent = finding.detail;
    article.append(title, detail);
    elements.findings.append(article);
  }
}

function renderMetricList(target, rows) {
  target.replaceChildren();
  for (const [label, value] of rows) {
    const wrapper = document.createElement("div");
    wrapper.className = "metric-row";
    const term = document.createElement("dt");
    term.textContent = label;
    const definition = document.createElement("dd");
    definition.textContent = value;
    wrapper.append(term, definition);
    target.append(wrapper);
  }
}

function renderSpectrum(report) {
  const spectral = report.spectralFeatures;
  const spectrum = report.spectrum;
  const metrics = [
    ["Dominant", finiteNumber(spectrum?.dominantFrequencyHz) === null ? "—" : formatHz(spectrum.dominantFrequencyHz)],
    ["Centroid", finiteNumber(spectral?.centroidHz) === null ? "—" : formatHz(spectral.centroidHz)],
    ["Bandwidth", finiteNumber(spectral?.bandwidthHz) === null ? "—" : formatHz(spectral.bandwidthHz)],
    ["Rolloff", finiteNumber(spectral?.rolloffHz) === null ? "—" : formatHz(spectral.rolloffHz)],
  ];

  elements.spectrumMetrics.replaceChildren();
  for (const [label, value] of metrics) {
    const card = document.createElement("div");
    card.className = "mini-metric";
    const name = document.createElement("span");
    name.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    card.append(name, strong);
    elements.spectrumMetrics.append(card);
  }
}

function renderPitch(report) {
  const pitch = report.pitch;
  const key = report.musicalKey;
  elements.pitchContent.replaceChildren();

  const lead = document.createElement("p");
  lead.className = "result-lead";
  if (finiteNumber(pitch?.frequencyHz) !== null) {
    lead.textContent = `${pitch.noteName ?? "Pitch"} · ${formatHz(pitch.frequencyHz)}`;
  } else {
    lead.textContent = "No stable monophonic pitch estimate was returned.";
  }

  const details = document.createElement("div");
  details.className = "result-meta";
  addResultRow(details, "Pitch confidence", finiteNumber(pitch?.confidence) === null ? "—" : formatPercent(pitch.confidence * 100));
  addResultRow(details, "Estimated key", typeof key?.key === "string" ? key.key : "—");
  addResultRow(details, "Key confidence", finiteNumber(key?.confidence) === null ? "—" : formatPercent(key.confidence * 100));
  addResultRow(details, "Tuning offset", finiteNumber(key?.tuningCents) === null ? "—" : `${key.tuningCents.toFixed(1)} cents`);
  elements.pitchContent.append(lead, details);
}

function renderRhythm(report) {
  const rhythm = report.rhythm;
  elements.rhythmContent.replaceChildren();
  const bpm = finiteNumber(rhythm?.bpm);
  const lead = document.createElement("p");
  lead.className = "result-lead";
  lead.textContent = bpm === null ? "No stable tempo estimate was returned." : `${bpm.toFixed(1)} BPM`;

  const details = document.createElement("div");
  details.className = "result-meta";
  addResultRow(details, "Tempo confidence", finiteNumber(rhythm?.confidence) === null ? "—" : formatPercent(rhythm.confidence * 100));
  addResultRow(details, "Detected beats", Array.isArray(rhythm?.beats) ? rhythm.beats.length.toLocaleString() : "—");
  addResultRow(details, "Detected downbeats", Array.isArray(rhythm?.downbeats) ? rhythm.downbeats.length.toLocaleString() : "—");
  addResultRow(details, "Downbeat confidence", finiteNumber(rhythm?.downbeatConfidence) === null ? "—" : formatPercent(rhythm.downbeatConfidence * 100));
  elements.rhythmContent.append(lead, details);
}

function addResultRow(container, label, value) {
  const row = document.createElement("div");
  const name = document.createElement("span");
  const content = document.createElement("span");
  name.textContent = label;
  content.textContent = value;
  row.append(name, content);
  container.append(row);
}

function renderBackend(backend) {
  elements.backendContent.replaceChildren();
  const paragraph = document.createElement("p");
  const second = document.createElement("p");

  if (backend?.available) {
    elements.backendBadge.textContent = "local backend available";
    elements.backendBadge.className = "badge";
    paragraph.textContent = `A local backend was discovered at ${backend.baseUrl}. Browser/WASM remains the default analysis path.`;
    second.textContent = "The backend reports the transcription package surface as available for heavier model-backed enrichment. This inspector does not send the selected audio automatically; backend model execution remains an explicit opt-in boundary.";
  } else if (backend?.baseUrl) {
    elements.backendBadge.textContent = "backend unavailable";
    elements.backendBadge.className = "badge badge-muted";
    paragraph.textContent = `A backend was configured at ${backend.baseUrl}, but its transcription package surface could not be reached.`;
    second.textContent = "The report therefore used browser-local analysis only.";
  } else {
    elements.backendBadge.textContent = "browser only";
    elements.backendBadge.className = "badge badge-muted";
    paragraph.textContent = "This deployment is using the browser-local Rust/WASM path only. No audio was uploaded for analysis.";
    second.textContent = "When serving this site locally, a backend can be discovered at http://127.0.0.1:3000, or configured with ?backend=<base-url>. Heavy model capabilities stay separate from the public Pages boundary.";
  }

  elements.backendContent.append(paragraph, second);
}

function renderProvenance(packages) {
  elements.provenance.replaceChildren();
  for (const entry of packages) {
    const card = document.createElement("div");
    card.className = "provenance-card";
    const name = document.createElement("strong");
    name.textContent = entry.library;
    const detail = document.createElement("span");
    detail.textContent = entry.available ? `${entry.version} · ${entry.runtime}` : `unavailable · ${entry.error}`;
    card.append(name, detail);
    elements.provenance.append(card);
  }

  const inspector = document.createElement("div");
  inspector.className = "provenance-card";
  const name = document.createElement("strong");
  name.textContent = "Audio Inspector report schema";
  const detail = document.createElement("span");
  detail.textContent = "audio-analysis-inspector/v1";
  inspector.append(name, detail);
  elements.provenance.append(inspector);
}

function drawWaveform(canvas, buffer) {
  const context = prepareCanvas(canvas);
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111a1f";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#2dd4bf";
  context.lineWidth = Math.max(1, window.devicePixelRatio || 1);

  const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) => buffer.getChannelData(index));
  const framesPerPixel = Math.max(1, Math.floor(buffer.length / width));
  const mid = height / 2;
  const amplitude = height * 0.43;

  context.beginPath();
  for (let x = 0; x < width; x += 1) {
    const start = Math.min(buffer.length - 1, x * framesPerPixel);
    const end = Math.min(buffer.length, start + framesPerPixel);
    let min = 1;
    let max = -1;
    const sampleStride = Math.max(1, Math.floor((end - start) / 64));
    for (let frame = start; frame < end; frame += sampleStride) {
      let sample = 0;
      for (const channel of channels) sample += channel[frame] ?? 0;
      sample /= channels.length;
      min = Math.min(min, sample);
      max = Math.max(max, sample);
    }
    const y1 = mid - max * amplitude;
    const y2 = mid - min * amplitude;
    context.moveTo(x + 0.5, y1);
    context.lineTo(x + 0.5, y2);
  }
  context.stroke();

  context.strokeStyle = "rgba(255,255,255,0.18)";
  context.beginPath();
  context.moveTo(0, mid + 0.5);
  context.lineTo(width, mid + 0.5);
  context.stroke();
}

function drawSpectralTimeline(canvas, spectrogram) {
  const context = prepareCanvas(canvas);
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111a1f";
  context.fillRect(0, 0, width, height);

  const frames = Array.isArray(spectrogram?.frames) ? spectrogram.frames : [];
  if (!frames.length) {
    context.fillStyle = "#8fa0a8";
    context.font = `${14 * (window.devicePixelRatio || 1)}px system-ui`;
    context.fillText("Spectral frame summary unavailable", 18, 34);
    return;
  }

  const nyquist = Math.max(1, Number(spectrogram.sampleRate ?? 48_000) / 2);
  const dpr = window.devicePixelRatio || 1;
  const padding = 18 * dpr;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  context.strokeStyle = "rgba(255,255,255,0.12)";
  context.lineWidth = dpr;
  for (let i = 0; i <= 4; i += 1) {
    const y = padding + (plotHeight * i) / 4;
    context.beginPath();
    context.moveTo(padding, y);
    context.lineTo(width - padding, y);
    context.stroke();
  }

  drawFrameSeries(context, frames, "dominantFrequencyHz", nyquist, padding, plotWidth, plotHeight, "#2dd4bf", dpr);
  drawFrameSeries(context, frames, "centroidHz", nyquist, padding, plotWidth, plotHeight, "#fbbf24", dpr);

  context.fillStyle = "#2dd4bf";
  context.font = `${11 * dpr}px system-ui`;
  context.fillText("dominant", padding, 13 * dpr);
  context.fillStyle = "#fbbf24";
  context.fillText("centroid", padding + 68 * dpr, 13 * dpr);
}

function drawFrameSeries(context, frames, key, nyquist, padding, plotWidth, plotHeight, color, dpr) {
  context.strokeStyle = color;
  context.lineWidth = 1.6 * dpr;
  context.beginPath();
  let started = false;
  frames.forEach((frame, index) => {
    const value = finiteNumber(frame?.[key]);
    if (value === null) return;
    const x = padding + (frames.length === 1 ? plotWidth / 2 : (plotWidth * index) / (frames.length - 1));
    const normalized = Math.log10(1 + Math.max(0, value)) / Math.log10(1 + nyquist);
    const y = padding + plotHeight * (1 - normalized);
    if (!started) {
      context.moveTo(x, y);
      started = true;
    } else {
      context.lineTo(x, y);
    }
  });
  if (started) context.stroke();
}

function prepareCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width * dpr));
  const height = Math.max(140, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return canvas.getContext("2d");
}

async function initializeBackendBoundary() {
  const params = new URLSearchParams(window.location.search);
  const explicit = params.get("backend");
  const localHost = ["localhost", "127.0.0.1"].includes(window.location.hostname);
  const baseUrl = explicit || (localHost ? "http://127.0.0.1:3000" : null);

  if (!baseUrl || isGitHubPages()) {
    state.backend = {
      baseUrl: null,
      available: false,
      automaticUpload: false,
      reason: isGitHubPages() ? "github-pages-browser-only" : "not-configured",
    };
    elements.backendSummary.textContent = isGitHubPages()
      ? "Backend enrichment is off on GitHub Pages."
      : "Browser-local analysis; no backend configured.";
    return;
  }

  const normalized = baseUrl.replace(/\/$/, "");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1_800);
  try {
    const response = await fetch(`${normalized}/api/rust/packages/audio-analysis-transcription/api/package`, {
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const surface = await response.json();
    state.backend = {
      baseUrl: normalized,
      available: true,
      automaticUpload: false,
      transcription: {
        library: surface.library ?? "audio-analysis-transcription",
        version: surface.version ?? "unknown",
        operations: Array.isArray(surface.operations) ? surface.operations.map((operation) => operation.id).filter(Boolean) : [],
      },
    };
    elements.backendSummary.textContent = "Local backend discovered; heavier enrichment remains opt-in.";
  } catch (error) {
    state.backend = {
      baseUrl: normalized,
      available: false,
      automaticUpload: false,
      reason: errorMessage(error),
    };
    elements.backendSummary.textContent = "Browser-local analysis; configured backend is unavailable.";
  } finally {
    clearTimeout(timeout);
  }
}

function createExampleFile(kind, name) {
  const sampleRate = 8_000;
  let samples;
  if (kind === "tone") {
    samples = synthesizeTone(sampleRate, 3, 440);
  } else if (kind === "clicks") {
    samples = synthesizeClicks(sampleRate, 8, 120);
  } else {
    throw new Error(`Unknown example kind: ${kind}`);
  }
  return new File([encodeMonoPcm16Wav(samples, sampleRate)], `${slugify(name)}.wav`, { type: "audio/wav" });
}

function synthesizeTone(sampleRate, seconds, frequencyHz) {
  const count = Math.floor(sampleRate * seconds);
  const fadeSamples = Math.floor(sampleRate * 0.02);
  const samples = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const fadeIn = Math.min(1, index / Math.max(1, fadeSamples));
    const fadeOut = Math.min(1, (count - 1 - index) / Math.max(1, fadeSamples));
    samples[index] = 0.45 * Math.min(fadeIn, fadeOut) * Math.sin((2 * Math.PI * frequencyHz * index) / sampleRate);
  }
  return samples;
}

function synthesizeClicks(sampleRate, seconds, bpm) {
  const count = Math.floor(sampleRate * seconds);
  const samples = new Float32Array(count);
  const beatSamples = Math.floor((sampleRate * 60) / bpm);
  const clickSamples = Math.floor(sampleRate * 0.025);
  let beat = 0;
  for (let start = 0; start < count; start += beatSamples, beat += 1) {
    const amplitude = beat % 4 === 0 ? 0.8 : 0.5;
    const frequencyHz = beat % 4 === 0 ? 1_500 : 1_000;
    for (let offset = 0; offset < clickSamples && start + offset < count; offset += 1) {
      const envelope = Math.exp(-offset / (sampleRate * 0.006));
      samples[start + offset] += amplitude * envelope * Math.sin((2 * Math.PI * frequencyHz * offset) / sampleRate);
    }
  }
  return samples;
}

function encodeMonoPcm16Wav(samples, sampleRate) {
  const bytesPerSample = 2;
  const dataLength = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLength, true);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    const pcm = sample < 0 ? Math.round(sample * 32768) : Math.round(sample * 32767);
    view.setInt16(44 + index * 2, pcm, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
}

function replacePlayerSource(file) {
  if (state.currentObjectUrl) URL.revokeObjectURL(state.currentObjectUrl);
  state.currentObjectUrl = URL.createObjectURL(file);
  elements.audioPlayer.src = state.currentObjectUrl;
}

function exportReport() {
  if (!state.report) return;
  const blob = new Blob([`${JSON.stringify(state.report, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${fileStem(state.report.source.name)}.audio-analysis.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function resetInspector() {
  elements.report.hidden = true;
  elements.inputPanel.hidden = false;
  elements.fileInput.value = "";
  state.audioBuffer = null;
  state.report = null;
  elements.audioPlayer.pause();
  elements.audioPlayer.removeAttribute("src");
  elements.audioPlayer.load();
  if (state.currentObjectUrl) {
    URL.revokeObjectURL(state.currentObjectUrl);
    state.currentObjectUrl = null;
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function responseValue(response) {
  if (!response || response.error) return null;
  return response.value && typeof response.value === "object" ? response.value : null;
}

function setLoading(visible, title = "Analyzing audio…", detail = "") {
  elements.loadingPanel.hidden = !visible;
  elements.loadingTitle.textContent = title;
  elements.loadingDetail.textContent = detail;
}

function showError(message) {
  setLoading(false);
  elements.inputPanel.hidden = false;
  elements.inputError.hidden = false;
  elements.inputError.textContent = message;
}

function clearError() {
  elements.inputError.hidden = true;
  elements.inputError.textContent = "";
}

function isGitHubPages() {
  return window.location.hostname.endsWith("github.io");
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function amplitudeToDb(value) {
  return value > 0 ? 20 * Math.log10(value) : Number.NEGATIVE_INFINITY;
}

function formatDb(value) {
  if (value === Number.NEGATIVE_INFINITY) return "−∞ dBFS";
  return Number.isFinite(value) ? `${value.toFixed(2)} dBFS` : "—";
}

function formatHz(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number >= 1000 ? `${(number / 1000).toFixed(number >= 10_000 ? 1 : 2)} kHz` : `${number.toFixed(1)} Hz`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const total = Math.max(0, seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remaining = total % 60;
  if (hours) return `${hours}:${String(minutes).padStart(2, "0")}:${remaining.toFixed(1).padStart(4, "0")}`;
  return `${minutes}:${remaining.toFixed(1).padStart(4, "0")}`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : value >= 10 ? 1 : 2)} ${units[unit]}`;
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${value.toFixed(value < 1 ? 3 : 1)}%` : "—";
}

function formatNumber(value, digits) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function slugify(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function fileStem(name) {
  return name.replace(/\.[^.]+$/, "") || "audio";
}
