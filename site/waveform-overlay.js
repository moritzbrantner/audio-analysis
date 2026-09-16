const SNAPSHOT_READY_ATTRIBUTE = "data-waveform-snapshot";
const BEAT_TOGGLE_ID = "beat-overlay-toggle";
const TIME_EPSILON_SECONDS = 1e-6;

export function beatOverlayEvents(report) {
  const duration = finite(report?.source?.durationSeconds);
  const beats = Array.isArray(report?.rhythm?.beats) ? report.rhythm.beats : [];
  if (duration === null || duration <= 0 || !beats.length) return [];

  const coverageStart = Math.max(0, finite(report?.coverage?.rhythm?.startSeconds) ?? 0);
  const analysisStart = finite(report?.rhythm?.analysisStartSeconds);
  const timestampsAlreadyAbsolute =
    analysisStart !== null && Math.abs(analysisStart - coverageStart) <= TIME_EPSILON_SECONDS;
  const timestampOffset = timestampsAlreadyAbsolute || (analysisStart !== null && analysisStart > 0) ? 0 : coverageStart;

  return beats.flatMap((beat, index) => {
    const timestamp = finite(beat?.timestampSeconds);
    if (timestamp === null) return [];
    const timeSeconds = timestamp + timestampOffset;
    if (timeSeconds < 0 || timeSeconds > duration) return [];
    return [
      {
        index: Number.isInteger(beat?.index) ? beat.index : index + 1,
        timeSeconds,
        downbeat: Boolean(beat?.downbeat),
        strength: finite(beat?.strength),
      },
    ];
  });
}

export function rhythmCoverage(report) {
  const duration = finite(report?.source?.durationSeconds);
  const start = finite(report?.coverage?.rhythm?.startSeconds);
  const length = finite(report?.coverage?.rhythm?.sourceDurationSeconds);
  if (duration === null || duration <= 0 || start === null || length === null || length <= 0) return null;
  const boundedStart = clamp(start, 0, duration);
  const boundedEnd = clamp(start + length, boundedStart, duration);
  return boundedEnd > boundedStart ? { startSeconds: boundedStart, endSeconds: boundedEnd } : null;
}

export function overlayStatusText(report, events = beatOverlayEvents(report)) {
  if (!events.length) return "Beat overlay unavailable for this analysis";
  const coverage = rhythmCoverage(report);
  const scope = coverage ? ` from the ${formatSeconds(coverage.endSeconds - coverage.startSeconds)} rhythm window` : "";
  return `${events.length.toLocaleString()} detected beat${events.length === 1 ? "" : "s"}${scope}`;
}

function setupWaveformOverlay() {
  const waveform = document.querySelector("#waveform");
  const player = document.querySelector("#audio-player");
  const rawJson = document.querySelector("#raw-json");
  const waveformSection = document.querySelector("#waveform-section");
  const heading = waveformSection?.querySelector(".section-heading");
  const coverageBadge = document.querySelector("#statistics-coverage");
  if (!waveform || !player || !rawJson || !waveformSection || !heading || !coverageBadge) return;

  const stage = document.createElement("div");
  stage.className = "waveform-stage";
  waveform.before(stage);
  stage.append(waveform);

  const presentation = document.createElement("canvas");
  presentation.id = "waveform-presentation";
  presentation.className = "waveform-presentation";
  presentation.setAttribute("aria-hidden", "true");
  stage.append(presentation);

  const headingActions = document.createElement("div");
  headingActions.className = "waveform-heading-actions";
  coverageBadge.replaceWith(headingActions);
  headingActions.append(coverageBadge);

  const toggleLabel = document.createElement("label");
  toggleLabel.className = "waveform-overlay-toggle";
  toggleLabel.htmlFor = BEAT_TOGGLE_ID;

  const toggle = document.createElement("input");
  toggle.id = BEAT_TOGGLE_ID;
  toggle.type = "checkbox";
  toggle.checked = true;

  const toggleText = document.createElement("span");
  toggleText.textContent = "Beat overlay";
  toggleLabel.append(toggle, toggleText);
  headingActions.append(toggleLabel);

  const stable = {
    report: null,
    beatEvents: [],
    baseCanvas: null,
    hoverTime: null,
    playbackFrame: null,
    snapshotGeneration: 0,
  };

  const refreshReport = () => {
    stable.report = parseReport(rawJson.textContent);
    stable.beatEvents = beatOverlayEvents(stable.report);
    const enabled = stable.beatEvents.length > 0;
    toggle.disabled = !enabled;
    if (!enabled) toggle.checked = false;
    else if (toggle.dataset.userChanged !== "true") toggle.checked = true;
    toggleLabel.title = overlayStatusText(stable.report, stable.beatEvents);
    toggle.setAttribute("aria-label", `${toggleLabel.title}. Toggle beat markers on the waveform.`);
    queueSnapshotCapture();
  };

  const queueSnapshotCapture = () => {
    const generation = ++stable.snapshotGeneration;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (generation !== stable.snapshotGeneration) return;
        stable.baseCanvas = captureCanvas(waveform);
        if (stable.baseCanvas) stage.setAttribute(SNAPSHOT_READY_ATTRIBUTE, "ready");
        drawPresentation();
      });
    });
  };

  const drawPresentation = () => {
    const rect = stage.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const backingWidth = Math.max(1, Math.round(width * dpr));
    const backingHeight = Math.max(1, Math.round(height * dpr));
    if (presentation.width !== backingWidth || presentation.height !== backingHeight) {
      presentation.width = backingWidth;
      presentation.height = backingHeight;
    }

    const context = presentation.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    if (!stable.baseCanvas) return;
    context.drawImage(stable.baseCanvas, 0, 0, stable.baseCanvas.width, stable.baseCanvas.height, 0, 0, width, height);

    const duration = finite(stable.report?.source?.durationSeconds) ?? finite(player.duration);
    if (duration === null || duration <= 0) return;

    if (toggle.checked && stable.beatEvents.length) {
      drawRhythmCoverage(context, width, height, duration, rhythmCoverage(stable.report));
      drawBeatMarkers(context, width, height, duration, stable.beatEvents);
    }

    drawCursor(context, width, height, duration, finite(player.currentTime) ?? 0, "rgba(255,255,255,0.92)");
    if (stable.hoverTime !== null) {
      drawCursor(context, width, height, duration, stable.hoverTime, "rgba(251,191,36,0.95)");
    }
  };

  const startPlaybackAnimation = () => {
    stopPlaybackAnimation();
    const tick = () => {
      if (player.paused) {
        stable.playbackFrame = null;
        drawPresentation();
        return;
      }
      drawPresentation();
      stable.playbackFrame = requestAnimationFrame(tick);
    };
    stable.playbackFrame = requestAnimationFrame(tick);
  };

  const stopPlaybackAnimation = () => {
    if (stable.playbackFrame !== null && typeof cancelAnimationFrame === "function") {
      cancelAnimationFrame(stable.playbackFrame);
    }
    stable.playbackFrame = null;
    drawPresentation();
  };

  toggle.addEventListener("change", () => {
    toggle.dataset.userChanged = "true";
    drawPresentation();
  });

  waveform.addEventListener("pointermove", (event) => {
    const rect = waveform.getBoundingClientRect();
    const duration = finite(stable.report?.source?.durationSeconds) ?? finite(player.duration);
    if (duration === null || duration <= 0 || rect.width <= 0) return;
    const x = clamp((Number(event.clientX) || 0) - rect.left, 0, rect.width);
    stable.hoverTime = (x / rect.width) * duration;
    drawPresentation();
  });

  waveform.addEventListener("pointerleave", () => {
    stable.hoverTime = null;
    drawPresentation();
  });

  player.addEventListener("timeupdate", drawPresentation);
  player.addEventListener("seeked", drawPresentation);
  player.addEventListener("play", startPlaybackAnimation);
  player.addEventListener("pause", stopPlaybackAnimation);
  player.addEventListener("ended", stopPlaybackAnimation);
  window.addEventListener("resize", drawPresentation);

  const reportObserver = new MutationObserver(refreshReport);
  reportObserver.observe(rawJson, { childList: true, characterData: true, subtree: true });
  if (rawJson.textContent.trim()) refreshReport();
}

function captureCanvas(source) {
  if (!source.width || !source.height) return null;
  const snapshot = document.createElement("canvas");
  snapshot.width = source.width;
  snapshot.height = source.height;
  const context = snapshot.getContext("2d");
  context.drawImage(source, 0, 0);
  return snapshot;
}

function drawRhythmCoverage(context, width, height, duration, coverage) {
  if (!coverage) return;
  const startX = (coverage.startSeconds / duration) * width;
  const endX = (coverage.endSeconds / duration) * width;
  context.fillStyle = "rgba(129,140,248,0.08)";
  context.fillRect(startX, 0, Math.max(1, endX - startX), height);
}

function drawBeatMarkers(context, width, height, duration, events) {
  for (const event of events) {
    const x = Math.round((event.timeSeconds / duration) * width) + 0.5;
    context.strokeStyle = event.downbeat ? "rgba(251,191,36,0.92)" : "rgba(129,140,248,0.72)";
    context.lineWidth = event.downbeat ? 2 : 1;
    context.beginPath();
    context.moveTo(x, event.downbeat ? 0 : height * 0.18);
    context.lineTo(x, height);
    context.stroke();
  }
}

function drawCursor(context, width, height, duration, time, color) {
  if (!Number.isFinite(time) || duration <= 0) return;
  const x = Math.round((clamp(time, 0, duration) / duration) * width) + 0.5;
  context.strokeStyle = color;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(x, 0);
  context.lineTo(x, height);
  context.stroke();
}

function parseReport(value) {
  if (!value?.trim()) return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function finite(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function formatSeconds(value) {
  return `${value.toFixed(value >= 10 ? 1 : 2)} s`;
}

if (typeof document !== "undefined" && typeof window !== "undefined") setupWaveformOverlay();
