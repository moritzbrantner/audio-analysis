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

    const event = {
      index: Number.isInteger(beat?.index) ? beat.index : index + 1,
      timeSeconds,
      downbeat: Boolean(beat?.downbeat),
      strength: finite(beat?.strength),
    };
    addOptionalNumber(event, "localBpm", beat?.localBpm);
    addOptionalInteger(event, "beatInBar", beat?.beatInBar);
    addOptionalInteger(event, "barIndex", beat?.barIndex);
    addOptionalInteger(event, "sectionIndex", beat?.sectionIndex);
    const sectionIdentity = nonemptyString(beat?.sectionIdentity);
    if (sectionIdentity !== null) event.sectionIdentity = sectionIdentity;
    return [event];
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

export function musicalContextAtTime(
  report,
  requestedTimeSeconds,
  beats = beatOverlayEvents(report),
  sections = sectionOverlaySegments(report),
) {
  const duration = finite(report?.source?.durationSeconds);
  const requested = finite(requestedTimeSeconds);
  if (duration === null || duration <= 0 || requested === null) return null;

  const timeSeconds = clamp(requested, 0, duration);
  const coverage = rhythmCoverage(report);
  const inCoverage =
    coverage === null ||
    (timeSeconds >= coverage.startSeconds - TIME_EPSILON_SECONDS &&
      timeSeconds <= coverage.endSeconds + TIME_EPSILON_SECONDS);
  const section = inCoverage ? sectionAtTime(sections, timeSeconds) : null;
  const beat = inCoverage ? beatAtOrBefore(beats, timeSeconds) : null;
  const beatsPerBar = positiveInteger(report?.rhythm?.beatsPerBar) ?? 4;
  const bpm = finite(beat?.localBpm) ?? finite(report?.rhythm?.bpm);

  return {
    timeSeconds,
    inCoverage,
    sectionIndex: section?.index ?? null,
    sectionIdentity: section?.identity ?? null,
    sectionLabel: section?.label ?? null,
    barIndex: positiveInteger(beat?.barIndex),
    beatInBar: positiveInteger(beat?.beatInBar),
    beatsPerBar,
    bpm,
  };
}

export function formatMusicalContext(context) {
  if (!context) return "Musical context unavailable";
  const time = formatTimelineTime(context.timeSeconds);
  if (!context.inCoverage) return `${time} · Outside analyzed rhythm window`;

  const parts = [];
  const section = context.sectionIdentity ?? context.sectionLabel;
  if (section) parts.push(`Section ${section}`);
  if (context.barIndex !== null) parts.push(`Bar ${context.barIndex}`);
  if (context.beatInBar !== null) parts.push(`Beat ${context.beatInBar}/${context.beatsPerBar}`);
  if (context.bpm !== null) parts.push(`${context.bpm.toFixed(1)} BPM`);
  return parts.length ? `${time} · ${parts.join(" · ")}` : `${time} · Rhythm analysis window`;
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

  const structureRail = document.createElement("div");
  structureRail.className = "waveform-structure-rail";
  structureRail.setAttribute("role", "group");
  structureRail.setAttribute("aria-label", "Detected musical structure. Select a section to seek to its start.");
  stage.append(structureRail);

  const contextHud = document.createElement("div");
  contextHud.id = "waveform-context-hud";
  contextHud.className = "waveform-context-hud";
  contextHud.setAttribute("aria-label", "Musical timeline context");
  const contextTime = document.createElement("strong");
  contextTime.className = "waveform-context-time";
  const contextDetail = document.createElement("span");
  contextDetail.className = "waveform-context-detail";
  contextHud.append(contextTime, contextDetail);
  stage.after(contextHud);

  const describedBy = new Set((waveform.getAttribute("aria-describedby") ?? "").split(/\s+/).filter(Boolean));
  describedBy.add(contextHud.id);
  waveform.setAttribute("aria-describedby", Array.from(describedBy).join(" "));

  const headingActions = document.createElement("div");
  headingActions.className = "waveform-heading-actions";
  coverageBadge.replaceWith(headingActions);
  headingActions.append(coverageBadge);

  const beatToggle = createOverlayToggle(BEAT_TOGGLE_ID, "Beat overlay");
  const sectionToggle = createOverlayToggle(SECTION_TOGGLE_ID, "Section overlay");
  headingActions.append(beatToggle.label, sectionToggle.label);

  const state = {
    report: null,
    beats: [],
    sections: [],
    activeSectionIndex: null,
    pointerInspecting: false,
    railInspecting: false,
  };

  const syncActiveSection = (sectionIndex) => {
    if (state.activeSectionIndex === sectionIndex) return;
    structureRail.querySelector(".waveform-section-segment.is-active")?.classList.remove("is-active");
    state.activeSectionIndex = sectionIndex;
    if (sectionIndex === null) return;
    structureRail
      .querySelector(`[data-section-index="${sectionIndex}"]`)
      ?.classList.add("is-active");
  };

  const syncContext = (timeSeconds) => {
    const context = musicalContextAtTime(state.report, timeSeconds, state.beats, state.sections);
    renderContextHud(contextTime, contextDetail, context);
    syncActiveSection(context?.sectionIndex ?? null);
  };

  const restorePlaybackContext = () => {
    if (!state.pointerInspecting && !state.railInspecting) syncContext(Number(player.currentTime) || 0);
  };

  const syncVisibility = () => {
    beatLayer.hidden = beatToggle.input.disabled || !beatToggle.input.checked;
    const sectionsHidden = sectionToggle.input.disabled || !sectionToggle.input.checked;
    sectionLayer.hidden = sectionsHidden;
    structureRail.hidden = sectionsHidden;
    overlayLayer.hidden = beatLayer.hidden && sectionLayer.hidden;
  };

  const refreshReport = () => {
    state.report = parseReport(rawJson.textContent);
    state.beats = beatOverlayEvents(state.report);
    state.sections = sectionOverlaySegments(state.report);

    renderBeatLayer(beatLayer, state.report, state.beats);
    renderSectionBoundaries(sectionLayer, state.report, state.sections);
    renderStructureRail(structureRail, state.report, state.sections, {
      inspect: (timeSeconds) => {
        state.railInspecting = true;
        syncContext(timeSeconds);
      },
      restore: () => {
        state.railInspecting = false;
        restorePlaybackContext();
      },
      seek: (timeSeconds) => {
        player.currentTime = timeSeconds;
        state.railInspecting = false;
        syncContext(timeSeconds);
      },
    });

    configureToggle(
      beatToggle,
      state.beats.length > 0,
      beatOverlayStatusText(state.report, state.beats),
      "Toggle beat and downbeat markers on the waveform.",
    );
    configureToggle(
      sectionToggle,
      state.sections.length > 0,
      sectionOverlayStatusText(state.report, state.sections),
      "Toggle the musical structure rail and section boundaries.",
    );
    syncVisibility();
    syncContext(Number(player.currentTime) || 0);
  };

  for (const control of [beatToggle, sectionToggle]) {
    control.input.addEventListener("change", () => {
      control.input.dataset.userChanged = "true";
      syncVisibility();
    });
  }

  waveform.addEventListener("pointermove", (event) => {
    const duration = finite(state.report?.source?.durationSeconds);
    const rect = waveform.getBoundingClientRect();
    if (duration === null || duration <= 0 || rect.width <= 0) return;
    const x = clamp((Number(event.clientX) || 0) - rect.left, 0, rect.width);
    state.pointerInspecting = true;
    syncContext((x / rect.width) * duration);
  });

  waveform.addEventListener("pointerleave", () => {
    state.pointerInspecting = false;
    restorePlaybackContext();
  });
  waveform.addEventListener("focus", restorePlaybackContext);
  player.addEventListener("timeupdate", restorePlaybackContext);
  player.addEventListener("seeked", restorePlaybackContext);
  player.addEventListener("loadedmetadata", restorePlaybackContext);

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

function renderSectionBoundaries(layer, report, sections) {
  layer.replaceChildren();
  const duration = finite(report?.source?.durationSeconds);
  if (duration === null || duration <= 0 || !sections.length) return;

  for (const section of sections) {
    const boundary = document.createElement("span");
    boundary.className = `waveform-section-boundary ${boundaryConfidenceClass(section.boundaryConfidence)}`;
    boundary.style.left = `${percentage(section.startSeconds, duration)}%`;
    boundary.dataset.sectionIndex = String(section.index);
    if (section.boundaryConfidence !== null) {
      boundary.dataset.boundaryConfidence = section.boundaryConfidence.toFixed(3);
    }
    layer.append(boundary);
  }
}

function renderStructureRail(rail, report, sections, handlers) {
  rail.replaceChildren();
  const duration = finite(report?.source?.durationSeconds);
  if (duration === null || duration <= 0 || !sections.length) return;

  for (const section of sections) {
    const segment = document.createElement("button");
    const tone = sectionTone(section.identity, section.index);
    segment.type = "button";
    segment.className = `waveform-section-segment waveform-section-tone-${tone}`;
    segment.style.left = `${percentage(section.startSeconds, duration)}%`;
    segment.style.width = `${percentage(section.endSeconds - section.startSeconds, duration)}%`;
    segment.dataset.sectionIndex = String(section.index);
    segment.dataset.sectionIdentity = section.identity ?? "";
    segment.dataset.sectionStartSeconds = String(section.startSeconds);
    segment.dataset.sectionEndSeconds = String(section.endSeconds);

    const labelText = section.identity ?? section.label;
    const confidenceText =
      section.boundaryConfidence === null ? "" : ` Boundary confidence ${Math.round(section.boundaryConfidence * 100)}%.`;
    const accessibleLabel = `${labelText}, ${formatTimelineTime(section.startSeconds)} to ${formatTimelineTime(section.endSeconds)}.${confidenceText} Seek to section start.`;
    segment.setAttribute("aria-label", accessibleLabel);
    segment.title = accessibleLabel;

    const label = document.createElement("span");
    label.className = "waveform-section-label";
    label.textContent = labelText;
    segment.append(label);

    const inspectTime = Math.min(section.endSeconds, section.startSeconds + TIME_EPSILON_SECONDS);
    segment.addEventListener("pointerenter", () => handlers.inspect(inspectTime));
    segment.addEventListener("pointerleave", handlers.restore);
    segment.addEventListener("focus", () => handlers.inspect(inspectTime));
    segment.addEventListener("blur", handlers.restore);
    segment.addEventListener("click", () => handlers.seek(section.startSeconds));
    rail.append(segment);
  }
}

function renderContextHud(timeNode, detailNode, context) {
  if (!context) {
    timeNode.textContent = "—";
    detailNode.textContent = "Musical context unavailable";
    detailNode.classList.remove("is-outside");
    return;
  }

  timeNode.textContent = formatTimelineTime(context.timeSeconds);
  if (!context.inCoverage) {
    detailNode.textContent = "Outside analyzed rhythm window";
    detailNode.classList.add("is-outside");
    return;
  }

  const parts = [];
  const section = context.sectionIdentity ?? context.sectionLabel;
  if (section) parts.push(`Section ${section}`);
  if (context.barIndex !== null) parts.push(`Bar ${context.barIndex}`);
  if (context.beatInBar !== null) parts.push(`Beat ${context.beatInBar}/${context.beatsPerBar}`);
  if (context.bpm !== null) parts.push(`${context.bpm.toFixed(1)} BPM`);
  detailNode.textContent = parts.join(" · ") || "Rhythm analysis window";
  detailNode.classList.remove("is-outside");
}

function rhythmTimestampOffset(report) {
  const coverageStart = Math.max(0, finite(report?.coverage?.rhythm?.startSeconds) ?? 0);
  const analysisStart = finite(report?.rhythm?.analysisStartSeconds);
  const timestampsAlreadyAbsolute =
    analysisStart !== null && Math.abs(analysisStart - coverageStart) <= TIME_EPSILON_SECONDS;
  return timestampsAlreadyAbsolute || (analysisStart !== null && analysisStart > 0) ? 0 : coverageStart;
}

function sectionAtTime(sections, timeSeconds) {
  return (
    sections.find((section, index) => {
      const finalSection = index === sections.length - 1;
      return (
        timeSeconds >= section.startSeconds - TIME_EPSILON_SECONDS &&
        (timeSeconds < section.endSeconds - TIME_EPSILON_SECONDS ||
          (finalSection && timeSeconds <= section.endSeconds + TIME_EPSILON_SECONDS))
      );
    }) ?? null
  );
}

function beatAtOrBefore(beats, timeSeconds) {
  let candidate = null;
  for (const beat of beats) {
    if (beat.timeSeconds > timeSeconds + TIME_EPSILON_SECONDS) break;
    candidate = beat;
  }
  return candidate;
}

function boundaryConfidenceClass(confidence) {
  if (confidence === null) return "waveform-section-boundary-unknown";
  if (confidence >= 0.75) return "waveform-section-boundary-strong";
  if (confidence >= 0.45) return "waveform-section-boundary-medium";
  return "waveform-section-boundary-soft";
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

function addOptionalNumber(target, key, value) {
  const number = finite(value);
  if (number !== null) target[key] = number;
}

function addOptionalInteger(target, key, value) {
  const number = positiveInteger(value);
  if (number !== null) target[key] = number;
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function positiveInteger(value) {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : null;
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

function formatTimelineTime(value) {
  const total = Math.max(0, finite(value) ?? 0);
  const minutes = Math.floor(total / 60);
  const seconds = total - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

if (typeof document !== "undefined" && typeof window !== "undefined") setupWaveformOverlay();
