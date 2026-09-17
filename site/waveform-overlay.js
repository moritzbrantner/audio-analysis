const BEAT_TOGGLE_ID = "beat-overlay-toggle";
const SECTION_TOGGLE_ID = "section-overlay-toggle";
const TIME_EPSILON_SECONDS = 1e-6;
const SECTION_TONE_COUNT = 4;

export function beatOverlayEvents(report) {
  const duration = finite(report?.source?.durationSeconds);
  const beats = Array.isArray(report?.rhythm?.beats) ? report.rhythm.beats : [];
  if (duration === null || duration <= 0 || !beats.length) return [];

  const timestampOffset = rhythmTimestampOffset(report);
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

export function sectionOverlaySegments(report) {
  const duration = finite(report?.source?.durationSeconds);
  const sections = Array.isArray(report?.rhythm?.sections) ? report.rhythm.sections : [];
  if (duration === null || duration <= 0 || !sections.length) return [];

  const timestampOffset = rhythmTimestampOffset(report);
  return sections.flatMap((section, index) => {
    const start = finite(section?.startSeconds);
    const end = finite(section?.endSeconds);
    if (start === null || end === null || end <= start) return [];

    const startSeconds = clamp(start + timestampOffset, 0, duration);
    const endSeconds = clamp(end + timestampOffset, startSeconds, duration);
    if (endSeconds <= startSeconds) return [];

    const identity = nonemptyString(section?.identity);
    const label = nonemptyString(section?.label) ?? `section-${index + 1}`;
    return [
      {
        index: Number.isInteger(section?.index) ? section.index : index + 1,
        startSeconds,
        endSeconds,
        identity,
        label,
        boundaryConfidence: finite(section?.startBoundaryConfidence),
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

export function beatOverlayStatusText(report, events = beatOverlayEvents(report)) {
  if (!events.length) return "Beat overlay unavailable for this analysis";
  const coverage = rhythmCoverage(report);
  const scope = coverage ? ` from the ${formatSeconds(coverage.endSeconds - coverage.startSeconds)} rhythm window` : "";
  return `${events.length.toLocaleString()} detected beat${events.length === 1 ? "" : "s"}${scope}`;
}

export function sectionOverlayStatusText(report, sections = sectionOverlaySegments(report)) {
  if (!sections.length) return "Section overlay unavailable for this analysis";
  const coverage = rhythmCoverage(report);
  const scope = coverage ? ` in the ${formatSeconds(coverage.endSeconds - coverage.startSeconds)} rhythm window` : "";
  return `${sections.length.toLocaleString()} detected section${sections.length === 1 ? "" : "s"}${scope}`;
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

  const sectionLayer = document.createElement("div");
  sectionLayer.className = "waveform-section-layer";
  overlayLayer.append(sectionLayer);

  const beatLayer = document.createElement("div");
  beatLayer.className = "waveform-beat-layer";
  overlayLayer.append(beatLayer);

  const headingActions = document.createElement("div");
  headingActions.className = "waveform-heading-actions";
  coverageBadge.replaceWith(headingActions);
  headingActions.append(coverageBadge);

  const beatToggle = createOverlayToggle(BEAT_TOGGLE_ID, "Beat overlay");
  const sectionToggle = createOverlayToggle(SECTION_TOGGLE_ID, "Section overlay");
  headingActions.append(beatToggle.label, sectionToggle.label);

  const refreshReport = () => {
    const report = parseReport(rawJson.textContent);
    const beats = beatOverlayEvents(report);
    const sections = sectionOverlaySegments(report);

    renderBeatLayer(beatLayer, report, beats);
    renderSectionLayer(sectionLayer, report, sections);
    configureToggle(
      beatToggle,
      beats.length > 0,
      beatOverlayStatusText(report, beats),
      "Toggle beat and downbeat markers on the waveform.",
    );
    configureToggle(
      sectionToggle,
      sections.length > 0,
      sectionOverlayStatusText(report, sections),
      "Toggle detected structural sections on the waveform.",
    );
    syncVisibility();
  };

  const syncVisibility = () => {
    beatLayer.hidden = beatToggle.input.disabled || !beatToggle.input.checked;
    sectionLayer.hidden = sectionToggle.input.disabled || !sectionToggle.input.checked;
    overlayLayer.hidden = beatLayer.hidden && sectionLayer.hidden;
  };

  for (const control of [beatToggle, sectionToggle]) {
    control.input.addEventListener("change", () => {
      control.input.dataset.userChanged = "true";
      syncVisibility();
    });
  }

  const reportObserver = new MutationObserver(refreshReport);
  reportObserver.observe(rawJson, { childList: true, characterData: true, subtree: true });
  if (rawJson.textContent.trim()) refreshReport();
}

function createOverlayToggle(id, text) {
  const label = document.createElement("label");
  label.className = "waveform-overlay-toggle";
  label.htmlFor = id;

  const input = document.createElement("input");
  input.id = id;
  input.type = "checkbox";
  input.checked = true;

  const labelText = document.createElement("span");
  labelText.textContent = text;
  label.append(input, labelText);
  return { label, input };
}

function configureToggle(control, enabled, status, action) {
  control.input.disabled = !enabled;
  if (!enabled) control.input.checked = false;
  else if (control.input.dataset.userChanged !== "true") control.input.checked = true;
  control.label.title = status;
  control.input.setAttribute("aria-label", `${status}. ${action}`);
}

function renderBeatLayer(layer, report, events) {
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

function renderSectionLayer(layer, report, sections) {
  layer.replaceChildren();
  const duration = finite(report?.source?.durationSeconds);
  if (duration === null || duration <= 0 || !sections.length) return;

  for (const section of sections) {
    const segment = document.createElement("span");
    const tone = sectionTone(section.identity, section.index);
    segment.className = `waveform-section-segment waveform-section-tone-${tone}`;
    segment.style.left = `${percentage(section.startSeconds, duration)}%`;
    segment.style.width = `${percentage(section.endSeconds - section.startSeconds, duration)}%`;
    segment.dataset.sectionIndex = String(section.index);
    segment.dataset.sectionIdentity = section.identity ?? "";

    const label = document.createElement("span");
    label.className = "waveform-section-label";
    label.textContent = section.identity ?? section.label;
    segment.append(label);
    layer.append(segment);
  }
}

function rhythmTimestampOffset(report) {
  const coverageStart = Math.max(0, finite(report?.coverage?.rhythm?.startSeconds) ?? 0);
  const analysisStart = finite(report?.rhythm?.analysisStartSeconds);
  const timestampsAlreadyAbsolute =
    analysisStart !== null && Math.abs(analysisStart - coverageStart) <= TIME_EPSILON_SECONDS;
  return timestampsAlreadyAbsolute || (analysisStart !== null && analysisStart > 0) ? 0 : coverageStart;
}

function sectionTone(identity, index) {
  if (!identity) return Math.abs(index - 1) % SECTION_TONE_COUNT;
  let hash = 0;
  for (let offset = 0; offset < identity.length; offset += 1) {
    hash = (hash * 31 + identity.charCodeAt(offset)) >>> 0;
  }
  return hash % SECTION_TONE_COUNT;
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

function nonemptyString(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function formatSeconds(value) {
  return `${value.toFixed(value >= 10 ? 1 : 2)} s`;
}

if (typeof document !== "undefined" && typeof window !== "undefined") setupWaveformOverlay();
