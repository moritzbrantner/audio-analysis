const MAX_TRACK_SECONDS = 15 * 60;
const RHYTHM_RATE = 16_000;
const RHYTHM_FFT_SIZE = 1024;
const RHYTHM_HOP_SIZE = 128;
const BEAT_PREVIEW_COUNT = 12;

const state = {
  analysis: null,
  generation: 0,
  analyzing: false,
  analyzerPromise: null,
};

const elements = {
  fileInput: document.querySelector("#song-file-input"),
  chooseFile: document.querySelector("#song-choose-file"),
  error: document.querySelector("#song-error"),
  status: document.querySelector("#song-status"),
  statusTitle: document.querySelector("#song-status-title"),
  statusDetail: document.querySelector("#song-status-detail"),
  result: document.querySelector("#song-result"),
  resultTitle: document.querySelector("#song-result-title"),
  rhythmSummary: document.querySelector("#song-rhythm-summary"),
  sections: document.querySelector("#song-sections"),
  jsonPreview: document.querySelector("#song-json-preview"),
  downloadJson: document.querySelector("#song-download-json"),
  analyzeAnother: document.querySelector("#song-analyze-another"),
};

wireUi();

function wireUi() {
  elements.chooseFile.addEventListener("click", () => {
    if (!state.analyzing) elements.fileInput.click();
  });

  elements.fileInput.addEventListener("change", () => {
    const file = elements.fileInput.files?.[0];
    if (file && !state.analyzing) void analyzeSongFile(file);
  });

  elements.downloadJson.addEventListener("click", downloadSongAnalysis);
  elements.analyzeAnother.addEventListener("click", resetSongAnalysis);
}

async function analyzeSongFile(file) {
  const generation = ++state.generation;
  setAnalyzing(true);
  clearError();
  elements.result.hidden = true;
  setStatus(true, "Decoding song…", "Your browser is decoding the selected file locally.");

  try {
    const audioBuffer = await decodeAudioFile(file);
    if (!isCurrent(generation)) return;

    if (!audioBuffer.length || !audioBuffer.numberOfChannels) {
      throw new Error("The decoded audio buffer is empty.");
    }
    if (audioBuffer.duration > MAX_TRACK_SECONDS + 0.001) {
      throw new Error(
        `Whole-song analysis currently supports songs up to ${MAX_TRACK_SECONDS / 60} minutes. This file is ${formatDuration(audioBuffer.duration)}.`,
      );
    }

    const analysisRate = Math.min(audioBuffer.sampleRate, RHYTHM_RATE);
    setStatus(
      true,
      "Preparing whole-track samples…",
      `Mixing channels and resampling the complete ${formatDuration(audioBuffer.duration)} track to ${Math.round(analysisRate / 1000)} kHz.`,
    );
    const samples = mixAndResample(audioBuffer, analysisRate);
    if (!isCurrent(generation)) return;

    setStatus(
      true,
      "Detecting tempo, beats and sections…",
      "Running the Rust/WASM whole-track rhythm analyzer locally in this browser.",
    );
    const analyzer = await loadRhythmAnalyzer();
    if (!isCurrent(generation)) return;

    const response = await analyzer.runOperation({
      operation: "audio.rhythm.analyze",
      input: {
        samples,
        sampleRate: analysisRate,
        minBpm: 45,
        maxBpm: 220,
        fftSize: RHYTHM_FFT_SIZE,
        hopSize: RHYTHM_HOP_SIZE,
        timeOffsetSeconds: 0,
      },
    });
    if (!isCurrent(generation)) return;

    const value = response?.value;
    if (!value || typeof value !== "object") {
      throw new Error("The rhythm analyzer returned no structured result.");
    }

    const { schemaVersion = "audio-analysis-song/v1", ...rhythm } = value;
    state.analysis = {
      schemaVersion,
      generatedAt: new Date().toISOString(),
      source: {
        name: file.name,
        mediaType: file.type || "unknown",
        byteLength: file.size,
        durationSeconds: audioBuffer.duration,
        sampleRate: audioBuffer.sampleRate,
        channels: audioBuffer.numberOfChannels,
        framesPerChannel: audioBuffer.length,
      },
      runtime: {
        mode: "client-wasm",
        privacy: "browser-local",
        analysisSampleRate: analysisRate,
      },
      ...rhythm,
    };

    renderSongAnalysis(state.analysis);
  } catch (error) {
    if (isCurrent(generation)) showError(errorMessage(error));
  } finally {
    if (isCurrent(generation)) {
      setAnalyzing(false);
      setStatus(false);
    }
  }
}

async function decodeAudioFile(file) {
  const bytes = await file.arrayBuffer();
  const context = new AudioContext();
  try {
    return await context.decodeAudioData(bytes.slice(0));
  } finally {
    await context.close();
  }
}

function mixAndResample(buffer, targetRate) {
  const sourceRate = buffer.sampleRate;
  const scale = sourceRate / targetRate;
  const outputLength = Math.max(1, Math.floor(buffer.length / scale));
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) => buffer.getChannelData(index));
  const output = new Array(outputLength);

  for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
    const start = outputIndex * scale;
    const end = Math.min(buffer.length, (outputIndex + 1) * scale);
    const first = Math.floor(start);
    const last = Math.min(buffer.length, Math.ceil(end));
    let weightedSum = 0;
    let weight = 0;

    for (let sourceIndex = first; sourceIndex < last; sourceIndex += 1) {
      const overlap = Math.min(end, sourceIndex + 1) - Math.max(start, sourceIndex);
      if (overlap <= 0) continue;
      let mono = 0;
      for (const channel of channels) mono += channel[sourceIndex] ?? 0;
      mono /= channels.length;
      weightedSum += mono * overlap;
      weight += overlap;
    }

    output[outputIndex] = weight > 0 ? weightedSum / weight : 0;
  }

  return output;
}

function loadRhythmAnalyzer() {
  state.analyzerPromise ??= import("./wasm/audio-analysis-rhythm/index.js").then(async (module) => {
    await module.init();
    return module;
  });
  return state.analyzerPromise;
}

function renderSongAnalysis(analysis) {
  elements.result.hidden = false;
  elements.resultTitle.textContent = analysis.source.name;
  renderRhythmSummary(analysis);
  renderSections(analysis.sections);

  const preview = {
    schemaVersion: analysis.schemaVersion,
    source: analysis.source,
    bpm: analysis.bpm,
    confidence: analysis.confidence,
    sectionsMethod: analysis.sectionsMethod,
    sections: analysis.sections,
    beats: Array.isArray(analysis.beats) ? analysis.beats.slice(0, BEAT_PREVIEW_COUNT) : [],
  };
  elements.jsonPreview.textContent = JSON.stringify(preview, null, 2);
  elements.resultTitle.focus?.({ preventScroll: true });
  elements.result.scrollIntoView?.({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
}

function renderRhythmSummary(analysis) {
  elements.rhythmSummary.replaceChildren();
  const bpm = finiteNumber(analysis.bpm);
  const confidence = finiteNumber(analysis.confidence);

  const lead = document.createElement("p");
  lead.className = "result-lead";
  lead.textContent = bpm === null ? "No stable tempo estimate was returned." : `${bpm.toFixed(1)} BPM`;

  const detail = document.createElement("p");
  detail.className = "result-note";
  detail.textContent =
    confidence === null
      ? "Beat and downbeat timestamps are included in the JSON when the analyzer finds a stable beat path."
      : `Tempo confidence: ${formatPercent(confidence * 100)}. Beat timestamps are exported with millisecond precision.`;

  const explanation = document.createElement("p");
  explanation.className = "result-note";
  explanation.textContent =
    "The JSON keeps alternative tempo candidates because half-time and double-time interpretations can both be musically plausible.";

  elements.rhythmSummary.append(lead, detail, explanation);
}

function renderSections(sections) {
  elements.sections.replaceChildren();
  if (!Array.isArray(sections) || sections.length === 0) {
    const empty = document.createElement("p");
    empty.textContent = "No stable rhythmic section boundaries were detected.";
    elements.sections.append(empty);
    return;
  }

  const list = document.createElement("ol");
  for (const section of sections) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = section.label ?? `section-${section.index ?? "?"}`;
    const range = document.createElement("span");
    range.textContent = ` ${section.start ?? formatDuration(section.startSeconds)} → ${section.end ?? formatDuration(section.endSeconds)}`;
    item.append(title, range);
    list.append(item);
  }
  elements.sections.append(list);
}

function downloadSongAnalysis() {
  if (!state.analysis) return;
  const blob = new Blob([`${JSON.stringify(state.analysis, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${fileStem(state.analysis.source.name)}.song-analysis.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function resetSongAnalysis() {
  state.generation += 1;
  state.analysis = null;
  setAnalyzing(false);
  setStatus(false);
  clearError();
  elements.fileInput.value = "";
  elements.result.hidden = true;
  elements.chooseFile.focus?.({ preventScroll: true });
  elements.chooseFile.scrollIntoView?.({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" });
}

function setAnalyzing(analyzing) {
  state.analyzing = analyzing;
  elements.fileInput.disabled = analyzing;
  elements.chooseFile.disabled = analyzing;
  elements.downloadJson.disabled = analyzing;
  elements.analyzeAnother.disabled = analyzing;
}

function setStatus(visible, title = "Analyzing song…", detail = "") {
  elements.status.hidden = !visible;
  elements.statusTitle.textContent = title;
  elements.statusDetail.textContent = detail;
}

function showError(message) {
  setStatus(false);
  elements.error.hidden = false;
  elements.error.textContent = message;
}

function clearError() {
  elements.error.hidden = true;
  elements.error.textContent = "";
}

function isCurrent(generation) {
  return generation === state.generation;
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${value.toFixed(value < 1 ? 3 : 1)}%` : "—";
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const total = Math.max(0, seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remaining = total % 60;
  if (hours) return `${hours}:${String(minutes).padStart(2, "0")}:${remaining.toFixed(3).padStart(6, "0")}`;
  return `${minutes}:${remaining.toFixed(3).padStart(6, "0")}`;
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function reducedMotion() {
  return Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches);
}

function fileStem(name) {
  return name.replace(/\.[^.]+$/, "") || "song";
}
