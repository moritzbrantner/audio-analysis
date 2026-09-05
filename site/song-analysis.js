const MAX_TRACK_SECONDS = 15 * 60;
const RHYTHM_RATE = 16_000;
const RHYTHM_FFT_SIZE = 1024;
const RHYTHM_HOP_SIZE = 128;
const BEAT_PREVIEW_COUNT = 12;

const state = {
  analysis: null,
  audioBuffer: null,
  currentObjectUrl: null,
  generation: 0,
  analyzing: false,
  analyzerPromise: null,
  timelineHoverTime: null,
  playbackAnimationFrame: null,
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
  audioPlayer: document.querySelector("#song-audio-player"),
  timeline: document.querySelector("#song-timeline"),
  timelineReadout: document.querySelector("#song-timeline-readout"),
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
  wireTimelineInteraction();

  window.addEventListener("resize", () => {
    if (state.audioBuffer && state.analysis) drawSongTimeline();
  });
}

function wireTimelineInteraction() {
  elements.timeline.addEventListener("pointermove", (event) => {
    if (!state.audioBuffer || !state.analysis) return;
    state.timelineHoverTime = timelineTimeFromEvent(event);
    updateTimelineReadout(state.timelineHoverTime);
    drawSongTimeline();
  });

  elements.timeline.addEventListener("pointerleave", () => {
    state.timelineHoverTime = null;
    updateTimelineReadout(elements.audioPlayer.currentTime || 0, true);
    drawSongTimeline();
  });

  elements.timeline.addEventListener("click", (event) => {
    if (!state.audioBuffer) return;
    seekSong(timelineTimeFromEvent(event));
  });

  elements.timeline.addEventListener("keydown", (event) => {
    if (!state.audioBuffer || !state.analysis) return;
    const current = Number(elements.audioPlayer.currentTime) || 0;
    let next = null;
    if (event.key === "ArrowLeft") next = adjacentAnalysisTime(current, -1, event.shiftKey);
    else if (event.key === "ArrowRight") next = adjacentAnalysisTime(current, 1, event.shiftKey);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = state.audioBuffer.duration;
    else return;
    event.preventDefault();
    seekSong(next);
  });

  elements.timeline.addEventListener("focus", () => {
    updateTimelineReadout(elements.audioPlayer.currentTime || 0);
  });

  elements.audioPlayer.addEventListener("timeupdate", () => {
    updateTimelineAria();
    if (state.timelineHoverTime === null) {
      updateTimelineReadout(elements.audioPlayer.currentTime || 0, true);
    }
    drawSongTimeline();
  });
  elements.audioPlayer.addEventListener("loadedmetadata", () => {
    updateTimelineAria();
    drawSongTimeline();
  });
  elements.audioPlayer.addEventListener("play", startPlaybackAnimation);
  elements.audioPlayer.addEventListener("pause", stopPlaybackAnimation);
  elements.audioPlayer.addEventListener("ended", stopPlaybackAnimation);
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

    state.audioBuffer = audioBuffer;
    state.timelineHoverTime = null;
    replacePlayerSource(file);

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

    const value = await analyzer.analyzeTrack(samples, analysisRate, {
      minBpm: 45,
      maxBpm: 220,
      fftSize: RHYTHM_FFT_SIZE,
      hopSize: RHYTHM_HOP_SIZE,
      timeOffsetSeconds: 0,
    });
    if (!isCurrent(generation)) return;

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
        pcmTransport: "float32array",
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
  const output = new Float32Array(outputLength);

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
  updateTimelineAria();
  updateTimelineReadout(0, true);
  elements.resultTitle.focus?.({ preventScroll: true });
  elements.result.scrollIntoView?.({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  requestAnimationFrame(drawSongTimeline);
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

function drawSongTimeline() {
  if (!state.audioBuffer || !state.analysis) return;
  const { context, width, height } = prepareTimelineCanvas(elements.timeline);
  const duration = state.audioBuffer.duration;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111a1f";
  context.fillRect(0, 0, width, height);

  drawSectionBands(context, width, height, duration, state.analysis.sections);
  drawTimelineWaveform(context, width, height, state.audioBuffer);
  drawBeatMarkers(context, width, height, duration, state.analysis.beats);
  drawSectionBoundaries(context, width, height, duration, state.analysis.sections);
  drawTimelineCursor(context, width, height, duration, Number(elements.audioPlayer.currentTime) || 0, "rgba(255,255,255,0.95)");
  if (state.timelineHoverTime !== null) {
    drawTimelineCursor(context, width, height, duration, state.timelineHoverTime, "rgba(251,191,36,0.95)");
  }
}

function drawSectionBands(context, width, height, duration, sections) {
  if (!Array.isArray(sections) || duration <= 0) return;
  sections.forEach((section, index) => {
    const start = finiteNumber(section?.startSeconds);
    const end = finiteNumber(section?.endSeconds);
    if (start === null || end === null || end <= start) return;
    const x = (Math.max(0, start) / duration) * width;
    const right = (Math.min(duration, end) / duration) * width;
    context.fillStyle = index % 2 === 0 ? "rgba(45,212,191,0.055)" : "rgba(251,191,36,0.04)";
    context.fillRect(x, 0, Math.max(0, right - x), height);
    if (right - x >= 72) {
      context.fillStyle = "rgba(255,255,255,0.55)";
      context.font = "11px system-ui";
      context.fillText(String(section.label ?? `section-${index + 1}`), x + 7, 17);
    }
  });
}

function drawTimelineWaveform(context, width, height, buffer) {
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) => buffer.getChannelData(index));
  const framesPerPixel = Math.max(1, Math.floor(buffer.length / width));
  const mid = height * 0.58;
  const amplitude = height * 0.25;
  context.strokeStyle = "rgba(143,160,168,0.72)";
  context.lineWidth = 1;
  context.beginPath();

  for (let x = 0; x < width; x += 1) {
    const start = Math.min(buffer.length - 1, x * framesPerPixel);
    const end = Math.min(buffer.length, start + framesPerPixel);
    const sampleStride = Math.max(1, Math.floor((end - start) / 48));
    let min = 1;
    let max = -1;
    for (let frame = start; frame < end; frame += sampleStride) {
      let sample = 0;
      for (const channel of channels) sample += channel[frame] ?? 0;
      sample /= channels.length;
      min = Math.min(min, sample);
      max = Math.max(max, sample);
    }
    context.moveTo(x + 0.5, mid - max * amplitude);
    context.lineTo(x + 0.5, mid - min * amplitude);
  }
  context.stroke();
}

function drawBeatMarkers(context, width, height, duration, beats) {
  if (!Array.isArray(beats) || duration <= 0) return;
  for (const beat of beats) {
    const timestamp = finiteNumber(beat?.timestampSeconds);
    if (timestamp === null || timestamp < 0 || timestamp > duration) continue;
    const x = (timestamp / duration) * width + 0.5;
    const downbeat = beat?.downbeat === true;
    context.strokeStyle = downbeat ? "rgba(251,191,36,0.88)" : "rgba(45,212,191,0.48)";
    context.lineWidth = downbeat ? 1.4 : 1;
    context.beginPath();
    context.moveTo(x, downbeat ? height * 0.22 : height * 0.48);
    context.lineTo(x, height * 0.94);
    context.stroke();
  }
}

function drawSectionBoundaries(context, width, height, duration, sections) {
  if (!Array.isArray(sections) || duration <= 0) return;
  context.save();
  context.setLineDash([4, 4]);
  context.strokeStyle = "rgba(255,255,255,0.68)";
  context.lineWidth = 1;
  for (const section of sections.slice(1)) {
    const start = finiteNumber(section?.startSeconds);
    if (start === null || start <= 0 || start >= duration) continue;
    const x = (start / duration) * width + 0.5;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  context.restore();
}

function drawTimelineCursor(context, width, height, duration, time, color) {
  if (!Number.isFinite(duration) || duration <= 0) return;
  const x = (clamp(time, 0, duration) / duration) * width + 0.5;
  context.strokeStyle = color;
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(x, 0);
  context.lineTo(x, height);
  context.stroke();
}

function prepareTimelineCanvas(canvas) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(160, Math.floor(rect.height));
  const backingWidth = Math.max(1, Math.round(width * dpr));
  const backingHeight = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(backingWidth / width, 0, 0, backingHeight / height, 0, 0);
  return { context, width, height };
}

function timelineTimeFromEvent(event) {
  const duration = state.audioBuffer?.duration ?? 0;
  const rect = elements.timeline.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const x = clamp((Number(event.clientX) || 0) - (rect.left || 0), 0, width);
  return duration > 0 ? (x / width) * duration : 0;
}

function adjacentAnalysisTime(current, direction, useSections) {
  const duration = state.audioBuffer?.duration ?? 0;
  const times = useSections
    ? sectionBoundaryTimes(state.analysis?.sections, duration)
    : beatTimes(state.analysis?.beats);
  if (!times.length) return clamp(current + direction, 0, duration);
  const epsilon = 0.001;
  if (direction < 0) {
    for (let index = times.length - 1; index >= 0; index -= 1) {
      if (times[index] < current - epsilon) return times[index];
    }
    return 0;
  }
  for (const time of times) {
    if (time > current + epsilon) return time;
  }
  return duration;
}

function beatTimes(beats) {
  if (!Array.isArray(beats)) return [];
  return beats
    .map((beat) => finiteNumber(beat?.timestampSeconds))
    .filter((time) => time !== null);
}

function sectionBoundaryTimes(sections, duration) {
  if (!Array.isArray(sections)) return [];
  const times = sections
    .map((section) => finiteNumber(section?.startSeconds))
    .filter((time) => time !== null && time >= 0 && time <= duration);
  if (duration > 0) times.push(duration);
  return Array.from(new Set(times)).sort((left, right) => left - right);
}

function nearestBeat(time) {
  const beats = Array.isArray(state.analysis?.beats) ? state.analysis.beats : [];
  if (!beats.length) return null;
  let low = 0;
  let high = beats.length - 1;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const timestamp = finiteNumber(beats[mid]?.timestampSeconds) ?? 0;
    if (timestamp < time) low = mid + 1;
    else high = mid - 1;
  }
  const candidates = [beats[Math.max(0, Math.min(beats.length - 1, low))], beats[Math.max(0, high)]].filter(Boolean);
  return candidates.reduce((best, candidate) => {
    if (!best) return candidate;
    const bestTime = finiteNumber(best.timestampSeconds) ?? 0;
    const candidateTime = finiteNumber(candidate.timestampSeconds) ?? 0;
    return Math.abs(candidateTime - time) < Math.abs(bestTime - time) ? candidate : best;
  }, null);
}

function sectionAtTime(time) {
  const sections = Array.isArray(state.analysis?.sections) ? state.analysis.sections : [];
  return (
    sections.find((section) => {
      const start = finiteNumber(section?.startSeconds);
      const end = finiteNumber(section?.endSeconds);
      return start !== null && end !== null && time >= start && time < end;
    }) ?? sections.at(-1) ?? null
  );
}

function seekSong(time) {
  if (!state.audioBuffer) return;
  const next = clamp(time, 0, state.audioBuffer.duration);
  elements.audioPlayer.currentTime = next;
  updateTimelineAria();
  updateTimelineReadout(next);
  drawSongTimeline();
}

function updateTimelineAria() {
  const duration = state.audioBuffer?.duration ?? 0;
  const current = clamp(Number(elements.audioPlayer.currentTime) || 0, 0, duration || 0);
  elements.timeline.setAttribute("aria-valuemax", String(duration));
  elements.timeline.setAttribute("aria-valuenow", String(current));
  elements.timeline.setAttribute("aria-valuetext", timelineContextText(current));
}

function updateTimelineReadout(time, playbackOnly = false) {
  if (!state.audioBuffer || !state.analysis) {
    elements.timelineReadout.textContent = "Click the timeline or use arrow keys to inspect detected beats.";
    return;
  }
  const prefix = playbackOnly ? "Playback" : state.timelineHoverTime !== null ? "Pointer" : "Position";
  elements.timelineReadout.textContent = `${prefix}: ${timelineContextText(time)}`;
}

function timelineContextText(time) {
  const pieces = [formatDuration(time)];
  const section = sectionAtTime(time);
  if (section) pieces.push(String(section.label ?? `section-${section.index ?? "?"}`));
  const beat = nearestBeat(time);
  if (beat) {
    const beatIndex = Number.isInteger(beat.index) ? `beat ${beat.index}` : "nearest beat";
    const timestamp = typeof beat.timestamp === "string" ? beat.timestamp : formatDuration(beat.timestampSeconds);
    pieces.push(`${beatIndex}${beat.downbeat === true ? " · downbeat" : ""} at ${timestamp}`);
  }
  return pieces.join(" · ");
}

function startPlaybackAnimation() {
  stopPlaybackAnimation();
  const tick = () => {
    if (!state.audioBuffer || elements.audioPlayer.paused) {
      state.playbackAnimationFrame = null;
      return;
    }
    updateTimelineAria();
    if (state.timelineHoverTime === null) updateTimelineReadout(elements.audioPlayer.currentTime || 0, true);
    drawSongTimeline();
    state.playbackAnimationFrame = requestAnimationFrame(tick);
  };
  state.playbackAnimationFrame = requestAnimationFrame(tick);
}

function stopPlaybackAnimation() {
  if (state.playbackAnimationFrame !== null && typeof cancelAnimationFrame === "function") {
    cancelAnimationFrame(state.playbackAnimationFrame);
  }
  state.playbackAnimationFrame = null;
  drawSongTimeline();
}

function replacePlayerSource(file) {
  if (state.currentObjectUrl) URL.revokeObjectURL(state.currentObjectUrl);
  state.currentObjectUrl = URL.createObjectURL(file);
  elements.audioPlayer.src = state.currentObjectUrl;
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
  state.audioBuffer = null;
  state.timelineHoverTime = null;
  setAnalyzing(false);
  setStatus(false);
  clearError();
  stopPlaybackAnimation();
  elements.fileInput.value = "";
  elements.result.hidden = true;
  elements.audioPlayer.pause();
  elements.audioPlayer.removeAttribute("src");
  elements.audioPlayer.load();
  if (state.currentObjectUrl) {
    URL.revokeObjectURL(state.currentObjectUrl);
    state.currentObjectUrl = null;
  }
  elements.timelineReadout.textContent = "Click the timeline or use arrow keys to inspect detected beats.";
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

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
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
