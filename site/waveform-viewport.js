// Time↔pixel mapping, zoom and pan for the Audio Inspector waveform.
//
// A viewport is a plain `{ startSeconds, endSeconds }` window over a file of `durationSeconds`.
// Every surface that draws on the detailed waveform (canvas, beat and section overlays, playhead,
// hover readout, click-to-seek) maps time through the same viewport, so they stay aligned at every
// zoom level. These functions are pure and keep the window inside the file.

// The deepest zoom shows this many seconds: a few beats (three at 120 BPM), enough to resolve
// individual beats without reducing the window to single samples.
export const MIN_VIEW_SECONDS = 1.5;
const MIN_VIEW_FLOOR_SECONDS = 0.001;

export function wholeFileViewport(durationSeconds) {
  return { startSeconds: 0, endSeconds: positive(durationSeconds) };
}

export function viewportSpan(viewport) {
  return viewport.endSeconds - viewport.startSeconds;
}

export function timeToX(viewport, timeSeconds, widthPx) {
  return ((timeSeconds - viewport.startSeconds) / viewportSpan(viewport)) * widthPx;
}

export function xToTime(viewport, xPx, widthPx) {
  return viewport.startSeconds + (xPx / widthPx) * viewportSpan(viewport);
}

// The narrowest window for a file: beat scale, or a quarter of a file shorter than that.
export function minimumViewSpan(durationSeconds) {
  const duration = positive(durationSeconds);
  return Math.min(duration, Math.max(MIN_VIEW_FLOOR_SECONDS, Math.min(MIN_VIEW_SECONDS, duration / 4)));
}

export function isWholeFile(viewport, durationSeconds) {
  return viewport.startSeconds <= 0 && viewport.endSeconds >= positive(durationSeconds);
}

export function canZoomIn(viewport, durationSeconds) {
  return viewportSpan(viewport) > minimumViewSpan(durationSeconds) * (1 + 1e-9);
}

// Brings any window (for example one kept across a duration change) back inside the file.
export function clampViewport(viewport, durationSeconds) {
  const duration = positive(durationSeconds);
  const span = clamp(viewportSpan(viewport), minimumViewSpan(duration), duration);
  if (!(span < duration)) return wholeFileViewport(duration);
  const startSeconds = clamp(viewport.startSeconds, 0, duration - span);
  return { startSeconds, endSeconds: startSeconds + span };
}

// factor > 1 zooms in. The anchor time keeps its pixel position unless the window has to be
// clamped at a file edge; the span is bounded by beat scale and by the complete file.
export function zoomViewport(viewport, factor, anchorSeconds, durationSeconds) {
  const duration = positive(durationSeconds);
  const span = viewportSpan(viewport);
  if (!(factor > 0) || !Number.isFinite(factor) || !(span > 0)) return clampViewport(viewport, duration);
  const nextSpan = clamp(span / factor, minimumViewSpan(duration), duration);
  if (!(nextSpan < duration)) return wholeFileViewport(duration);
  const anchor = Number.isFinite(anchorSeconds) ? anchorSeconds : viewport.startSeconds + span / 2;
  const fraction = clamp((anchor - viewport.startSeconds) / span, 0, 1);
  const startSeconds = clamp(anchor - fraction * nextSpan, 0, duration - nextSpan);
  return { startSeconds, endSeconds: startSeconds + nextSpan };
}

export function panViewport(viewport, deltaSeconds, durationSeconds) {
  const duration = positive(durationSeconds);
  const span = Math.min(viewportSpan(viewport), duration);
  if (!(span < duration)) return wholeFileViewport(duration);
  const delta = Number.isFinite(deltaSeconds) ? deltaSeconds : 0;
  const startSeconds = clamp(viewport.startSeconds + delta, 0, duration - span);
  return { startSeconds, endSeconds: startSeconds + span };
}

function positive(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : 0;
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}
