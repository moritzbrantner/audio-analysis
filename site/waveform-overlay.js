const BEAT_TOGGLE_ID = "beat-overlay-toggle";
const SECTION_TOGGLE_ID = "section-overlay-toggle";
const TIME_EPSILON_SECONDS = 1e-6;
const SECTION_TONE_COUNT = 4;
// A section's local tempo is annotated only when it differs from a neighbour by at least this
// relative amount, so estimator jitter (about 1%) never reads as a tempo change.
const MATERIAL_TEMPO_RATIO = 0.04;

// The Rust whole-song contracts (`audio-analysis-song/v1` plus the key track) once the inspector's
// whole-track analysis completed; null while it is pending or when it is unavailable.
export function wholeTrackTimeline(report) {
  const wholeTrack = report?.wholeTrack;
  const song = wholeTrack?.song;
  if (wholeTrack?.status !== "complete" || !song || !Array.isArray(song.sections) || !song.sections.length) return null;
  return { song, key: wholeTrack.key && typeof wholeTrack.key === "object" ? wholeTrack.key : null };
}

function wholeTrackPending(report) {
  return report?.wholeTrack?.status === "pending";
}

function rhythmSource(report) {
  const timeline = wholeTrackTimeline(report);
  if (timeline) return { contract: timeline.song, timestampOffset: 0, wholeTrack: timeline };
  return { contract: report?.rhythm ?? null, timestampOffset: rhythmTimestampOffset(report), wholeTrack: null };
}

export function beatOverlayEvents(report) {
  const duration = finite(report?.source?.durationSeconds);
  const { contract, timestampOffset } = rhythmSource(report);
  const beats = Array.isArray(contract?.beats) ? contract.beats : [];
  if (duration === null || duration <= 0 || !beats.length) return [];

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
  // While the whole-track analysis runs, the bounded rhythm window is not presented as structure.
  if (!wholeTrackTimeline(report) && wholeTrackPending(report)) return [];
  const { contract, timestampOffset, wholeTrack } = rhythmSource(report);
  const sections = Array.isArray(contract?.sections) ? contract.sections : [];
  if (duration === null || duration <= 0 || !sections.length) return [];

  const segments = sections.flatMap((section, index) => {
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
  return wholeTrack ? annotateSections(segments, wholeTrack) : segments;
}

// Attaches the Rust local tempo and key evidence to each whole-track section and marks the values
// that materially differ from a neighbouring section. This is presentation only: tempo comes from
// the Rust tempo map (or beat-local tempo) and key from the Rust key segments.
export function annotateSections(segments, { song, key }) {
  const tempoPoints = (Array.isArray(song?.tempoMap) && song.tempoMap.length ? song.tempoMap : song?.beats ?? [])
    .map((point) => ({
      timeSeconds: finite(point?.timestampSeconds),
      bpm: finite(point?.bpm ?? point?.localBpm),
    }))
    .filter((point) => point.timeSeconds !== null && point.bpm !== null && point.bpm > 0);
  const keySegments = (Array.isArray(key?.segments) ? key.segments : []).filter(
    (segment) => finite(segment?.startSeconds) !== null && finite(segment?.endSeconds) !== null && segment?.key,
  );

  const annotated = segments.map((segment, index) => {
    const finalSegment = index === segments.length - 1;
    const bpms = tempoPoints
      .filter(
        (point) =>
          point.timeSeconds >= segment.startSeconds - TIME_EPSILON_SECONDS &&
          (point.timeSeconds < segment.endSeconds - TIME_EPSILON_SECONDS ||
            (finalSegment && point.timeSeconds <= segment.endSeconds + TIME_EPSILON_SECONDS)),
      )
      .map((point) => point.bpm);
    return { ...segment, localBpm: median(bpms), key: dominantKeySegment(keySegments, segment) };
  });

  return annotated.map((segment, index) => {
    const neighbours = [annotated[index - 1], annotated[index + 1]].filter(Boolean);
    const tempoChanged =
      segment.localBpm !== null &&
      neighbours.some((neighbour) => neighbour.localBpm !== null && materialTempoChange(segment.localBpm, neighbour.localBpm));
    const keyChanged =
      segment.key !== null &&
      neighbours.some((neighbour) => neighbour.key !== null && neighbour.key.label !== segment.key.label);
    return { ...segment, tempoChanged, keyChanged };
  });
}

function materialTempoChange(left, right) {
  return Math.abs(left - right) / Math.min(left, right) >= MATERIAL_TEMPO_RATIO;
}

function dominantKeySegment(keySegments, section) {
  let best = null;
  let bestOverlap = 0;
  for (const segment of keySegments) {
    const overlap =
      Math.min(section.endSeconds, Number(segment.endSeconds)) - Math.max(section.startSeconds, Number(segment.startSeconds));
    if (overlap > bestOverlap) {
      best = segment;
      bestOverlap = overlap;
    }
  }
  // A key needs to cover at least half of the section before it is shown as the section's key.
  if (!best || bestOverlap < (section.endSeconds - section.startSeconds) / 2) return null;
  const tonic = nonemptyString(best.key?.tonic);
  const scale = nonemptyString(best.key?.scale);
  const label = tonic && scale ? `${tonic} ${scale}` : nonemptyString(best.key?.label);
  if (!label) return null;
  return { label, confidence: finite(best.confidence ?? best.key?.confidence) };
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function formatSectionTempo(bpm) {
  return `${Math.round(bpm)} BPM`;
}

export function rhythmCoverage(report) {
  if (wholeTrackTimeline(report)) return null;
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
  const { contract } = rhythmSource(report);
  const beatsPerBar = positiveInteger(contract?.beatsPerBar) ?? 4;
  const bpm = finite(beat?.localBpm) ?? finite(section?.localBpm) ?? finite(contract?.bpm);

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
  if (wholeTrackTimeline(report)) {
    return `${events.length.toLocaleString()} detected beat${events.length === 1 ? "" : "s"} across the whole track`;
  }
  const coverage = rhythmCoverage(report);
  const scope = coverage ? ` from the ${formatSeconds(coverage.endSeconds - coverage.startSeconds)} rhythm window` : "";
  return `${events.length.toLocaleString()} detected beat${events.length === 1 ? "" : "s"}${scope}`;
}

export function sectionOverlayStatusText(report, sections = sectionOverlaySegments(report)) {
  if (!sections.length && wholeTrackPending(report)) return "Analyzing whole-track sections";
  if (!sections.length) return "Section overlay unavailable for this analysis";
  if (wholeTrackTimeline(report)) {
    return `${sections.length.toLocaleString()} detected section${sections.length === 1 ? "" : "s"} across the whole track`;
  }
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
    state.activeSectionIndex = null;
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

  const tones = sectionTones(sections);
  for (const section of sections) {
    const segment = document.createElement("button");
    const tone = tones.get(section) ?? 0;
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
    const annotations = [];
    if (section.tempoChanged) annotations.push(formatSectionTempo(section.localBpm));
    if (section.keyChanged) annotations.push(section.key.label);
    const localText = sectionLocalDescription(section);
    const accessibleLabel = `${labelText}, ${formatTimelineTime(section.startSeconds)} to ${formatTimelineTime(section.endSeconds)}.${confidenceText}${localText} Seek to section start.`;
    segment.setAttribute("aria-label", accessibleLabel);
    segment.title = accessibleLabel;
    if (section.localBpm !== undefined && section.localBpm !== null) {
      segment.dataset.sectionLocalBpm = section.localBpm.toFixed(2);
    }
    if (section.key) segment.dataset.sectionKey = section.key.label;

    const label = document.createElement("span");
    label.className = "waveform-section-label";
    label.textContent = labelText;
    segment.append(label);
    if (annotations.length) {
      const meta = document.createElement("span");
      meta.className = "waveform-section-meta";
      meta.textContent = annotations.join(" · ");
      segment.append(meta);
    }

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

function sectionLocalDescription(section) {
  const parts = [];
  if (section.localBpm !== undefined && section.localBpm !== null) {
    parts.push(`Local tempo ${section.localBpm.toFixed(1)} BPM`);
  }
  if (section.key) {
    const confidence = section.key.confidence === null ? "" : ` (key confidence ${Math.round(section.key.confidence * 100)}%)`;
    parts.push(`key ${section.key.label}${confidence}`);
  }
  return parts.length ? ` ${parts.join(", ")}.` : "";
}

// Repeated identities share a tone; distinct identities get distinct tones in order of first
// appearance (cycling only after SECTION_TONE_COUNT identities).
function sectionTones(sections) {
  const byIdentity = new Map();
  const tones = new Map();
  for (const section of sections) {
    const key = section.identity ?? `#${section.index}`;
    if (!byIdentity.has(key)) byIdentity.set(key, byIdentity.size % SECTION_TONE_COUNT);
    tones.set(section, byIdentity.get(key));
  }
  return tones;
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
