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
  const rawJson = document.querySelector("#raw-json");
  const waveformSection = document.querySelector("#waveform-section");
  const heading = waveformSection?.querySelector(".section-heading");
  const coverageBadge = document.querySelector("#statistics-coverage");
  if (!waveform || !rawJson || !waveformSection || !heading || !coverageBadge) return;

  const stage = document.createElement("div");
  stage.className = "waveform-stage";
  waveform.before(stage);
  stage.append(waveform);

  const overlayLayer = document.createElement("div");
  overlayLayer.className = "waveform-overlay-layer";
  overlayLayer.setAttribute("aria-hidden", "true");
  stage.append(overlayLayer);

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

  const refreshReport = () => {
    const report = parseReport(rawJson.textContent);
    const events = beatOverlayEvents(report);
    const enabled = events.length > 0;

    renderOverlayLayer(overlayLayer, report, events);
    toggle.disabled = !enabled;
    if (!enabled) toggle.checked = false;
    else if (toggle.dataset.userChanged !== "true") toggle.checked = true;
    toggleLabel.title = overlayStatusText(report, events);
    toggle.setAttribute("aria-label", `${toggleLabel.title}. Toggle beat markers on the waveform.`);
    overlayLayer.hidden = !enabled || !toggle.checked;
  };

  toggle.addEventListener("change", () => {
    toggle.dataset.userChanged = "true";
    overlayLayer.hidden = toggle.disabled || !toggle.checked;
  });

  const reportObserver = new MutationObserver(refreshReport);
  reportObserver.observe(rawJson, { childList: true, characterData: true, subtree: true });
  if (rawJson.textContent.trim()) refreshReport();
}

function renderOverlayLayer(layer, report, events) {
  layer.replaceChildren();
  const duration = finite(report?.source?.durationSeconds);
  if (duration === null || duration <= 0 || !events.length) return;

  const coverage = rhythmCoverage(report);
  if (coverage) {
    const coverageElement = document.createElement("span");
    coverageElement.className = "waveform-rhythm-coverage";
    coverageElement.style.left = `${percentage(coverage.startSeconds, duration)}%`;
    coverageElement.style.width = `${percentage(coverage.endSeconds - coverage.startSeconds, duration)}%`;
    layer.append(coverageElement);
  }

  for (const event of events) {
    const marker = document.createElement("span");
    marker.className = event.downbeat ? "waveform-beat-marker waveform-beat-marker-downbeat" : "waveform-beat-marker";
    marker.style.left = `${percentage(event.timeSeconds, duration)}%`;
    marker.dataset.beatIndex = String(event.index);
    layer.append(marker);
  }
}

function percentage(value, duration) {
  return clamp((value / duration) * 100, 0, 100);
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
  if (value === null || value === undefined || value === "") return null;
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
