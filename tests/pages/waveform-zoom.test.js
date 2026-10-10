// Acceptance for audio-analysis#140 (pre-implementation contract): zoom/pan viewport math and the
// bounded multiresolution peak structure behind the zoomable waveform and its whole-file overview.
//
// The seams are two small browser modules the inspector draws through:
//
// - `site/waveform-viewport.js`: pure time↔pixel transforms and zoom/pan with bounds. A viewport
//   is a plain `{ startSeconds, endSeconds }` window over a file of `durationSeconds`.
//     wholeFileViewport(durationSeconds) -> viewport
//     timeToX(viewport, timeSeconds, widthPx) -> x
//     xToTime(viewport, xPx, widthPx) -> seconds
//     zoomViewport(viewport, factor, anchorSeconds, durationSeconds) -> viewport (factor > 1 zooms in)
//     panViewport(viewport, deltaSeconds, durationSeconds) -> viewport
// - `site/waveform-peaks.js`: a min/max peak hierarchy built once from decoded PCM.
//     buildPeakPyramid(channels: Float32Array[], sampleRate) -> pyramid
//     viewportPeaks(pyramid, startSeconds, endSeconds, columns) -> { min, max } (length = columns)
//
// The tests assert behaviour only (timing semantics, bounds, conservative envelopes, cost that
// does not grow with track length); level layout and bucket sizes are the implementation's choice.
import { describe, expect, test } from "bun:test";
import { panViewport, timeToX, wholeFileViewport, xToTime, zoomViewport } from "../../site/waveform-viewport.js";
import { buildPeakPyramid, viewportPeaks } from "../../site/waveform-peaks.js";

const DURATION = 72;
// "Beat-scale inspection (a few seconds)": the deepest zoom must show at most 2 s, i.e. four beats
// at 120 BPM, so individual beats are resolvable.
const BEAT_SCALE_SECONDS = 2;

const span = (viewport) => viewport.endSeconds - viewport.startSeconds;

function expectInBounds(viewport, duration) {
  expect(Number.isFinite(viewport.startSeconds)).toBe(true);
  expect(Number.isFinite(viewport.endSeconds)).toBe(true);
  expect(viewport.startSeconds).toBeGreaterThanOrEqual(-1e-9);
  expect(viewport.endSeconds).toBeLessThanOrEqual(duration + 1e-9);
  expect(span(viewport)).toBeGreaterThan(0);
}

function zoomToLimit(viewport, anchor, duration) {
  let current = viewport;
  for (let step = 0; step < 200; step += 1) current = zoomViewport(current, 2, anchor, duration);
  return current;
}

describe("waveform viewport transforms", () => {
  test("the whole-file viewport is exactly the complete file", () => {
    expect(wholeFileViewport(DURATION)).toMatchObject({ startSeconds: 0, endSeconds: DURATION });
  });

  test("time↔pixel transforms are inverse linear maps of the visible window", () => {
    const viewport = { startSeconds: 20, endSeconds: 36 };
    const width = 1200;
    expect(timeToX(viewport, 20, width)).toBeCloseTo(0, 9);
    expect(timeToX(viewport, 36, width)).toBeCloseTo(width, 9);
    expect(timeToX(viewport, 28, width)).toBeCloseTo(600, 9);
    expect(xToTime(viewport, 0, width)).toBeCloseTo(20, 9);
    expect(xToTime(viewport, 300, width)).toBeCloseTo(24, 9);
    for (const time of [20, 21.37, 27.5, 33.333, 36]) {
      expect(xToTime(viewport, timeToX(viewport, time, width), width)).toBeCloseTo(time, 9);
    }
    // Width-independent: the same time sits at the same fraction of any width.
    expect(timeToX(viewport, 24, 300) / 300).toBeCloseTo(timeToX(viewport, 24, 1200) / 1200, 12);
  });

  test("zooming keeps the anchor time at the same pixel when the window is not clamped", () => {
    const width = 1000;
    let viewport = wholeFileViewport(DURATION);
    const anchor = 30.25;
    const anchorX = timeToX(viewport, anchor, width);
    for (const factor of [2, 1.5, 3, 1.25, 0.5, 4]) {
      viewport = zoomViewport(viewport, factor, anchor, DURATION);
      expectInBounds(viewport, DURATION);
      expect(timeToX(viewport, anchor, width)).toBeCloseTo(anchorX, 6);
    }
  });

  test("zoom factor scales the visible span", () => {
    const viewport = zoomViewport(wholeFileViewport(DURATION), 4, 36, DURATION);
    expect(span(viewport)).toBeCloseTo(DURATION / 4, 6);
    expect(span(zoomViewport(viewport, 0.5, 36, DURATION))).toBeCloseTo(DURATION / 2, 6);
  });

  test("zoom in is bounded at beat scale and never collapses the window", () => {
    const limit = zoomToLimit(wholeFileViewport(DURATION), 30, DURATION);
    expectInBounds(limit, DURATION);
    expect(span(limit)).toBeLessThanOrEqual(BEAT_SCALE_SECONDS + 1e-9);
    expect(span(limit)).toBeGreaterThanOrEqual(0.001);
    expect(limit.startSeconds).toBeLessThanOrEqual(30);
    expect(limit.endSeconds).toBeGreaterThanOrEqual(30);
    // At the limit, further zoom-in is a no-op.
    const again = zoomViewport(limit, 2, 30, DURATION);
    expect(span(again)).toBeCloseTo(span(limit), 9);
  });

  test("zoom out is bounded by the complete file", () => {
    let viewport = zoomViewport(wholeFileViewport(DURATION), 8, 50, DURATION);
    for (let step = 0; step < 50; step += 1) viewport = zoomViewport(viewport, 0.5, 50, DURATION);
    expect(viewport.startSeconds).toBeCloseTo(0, 9);
    expect(viewport.endSeconds).toBeCloseTo(DURATION, 9);
  });

  test("zooming near an edge clamps the window inside the file without changing its span", () => {
    const nearStart = zoomViewport(wholeFileViewport(DURATION), 6, 0.5, DURATION);
    expectInBounds(nearStart, DURATION);
    expect(span(nearStart)).toBeCloseTo(DURATION / 6, 6);
    expect(nearStart.startSeconds).toBeLessThanOrEqual(0.5);

    const nearEnd = zoomViewport(wholeFileViewport(DURATION), 6, DURATION - 0.1, DURATION);
    expectInBounds(nearEnd, DURATION);
    expect(span(nearEnd)).toBeCloseTo(DURATION / 6, 6);
    expect(nearEnd.endSeconds).toBeGreaterThanOrEqual(DURATION - 0.1);
  });

  test("a file shorter than the beat-scale limit still zooms out to exactly the whole file", () => {
    const short = 1.5;
    const zoomedIn = zoomToLimit(wholeFileViewport(short), 0.7, short);
    expectInBounds(zoomedIn, short);
    let viewport = zoomedIn;
    for (let step = 0; step < 50; step += 1) viewport = zoomViewport(viewport, 0.5, 0.7, short);
    expect(viewport).toMatchObject({ startSeconds: 0, endSeconds: short });
  });

  test("panning moves the window by the requested time and clamps at both file ends", () => {
    const viewport = { startSeconds: 20, endSeconds: 30 };
    const panned = panViewport(viewport, 7.5, DURATION);
    expect(panned.startSeconds).toBeCloseTo(27.5, 9);
    expect(panned.endSeconds).toBeCloseTo(37.5, 9);

    const atStart = panViewport(viewport, -100, DURATION);
    expect(atStart).toMatchObject({ startSeconds: 0 });
    expect(span(atStart)).toBeCloseTo(10, 9);

    const atEnd = panViewport(viewport, 100, DURATION);
    expect(atEnd.endSeconds).toBeCloseTo(DURATION, 9);
    expect(span(atEnd)).toBeCloseTo(10, 9);

    expect(panViewport(wholeFileViewport(DURATION), 5, DURATION)).toMatchObject({ startSeconds: 0, endSeconds: DURATION });
  });

  test("repeated zoom/pan sequences keep seek math exact and the window in bounds", () => {
    const width = 1280;
    let viewport = wholeFileViewport(DURATION);
    let random = 140;
    const next = () => {
      random = (Math.imul(random, 1_664_525) + 1_013_904_223) >>> 0;
      return random / 4_294_967_296;
    };
    for (let step = 0; step < 500; step += 1) {
      if (next() < 0.5) {
        const anchor = viewport.startSeconds + next() * span(viewport);
        viewport = zoomViewport(viewport, 0.4 + next() * 2.4, anchor, DURATION);
      } else {
        viewport = panViewport(viewport, (next() - 0.5) * span(viewport), DURATION);
      }
      expectInBounds(viewport, DURATION);
      expect(span(viewport)).toBeLessThanOrEqual(DURATION + 1e-9);
      const x = next() * width;
      const time = xToTime(viewport, x, width);
      expect(time).toBeGreaterThanOrEqual(viewport.startSeconds - 1e-9);
      expect(time).toBeLessThanOrEqual(viewport.endSeconds + 1e-9);
      expect(timeToX(viewport, time, width)).toBeCloseTo(x, 6);
    }
  });
});

describe("waveform peak pyramid", () => {
  const SAMPLE_RATE = 8_000;

  function impulseTrack(seconds, impulses, floor = 0.01) {
    const samples = new Float32Array(seconds * SAMPLE_RATE);
    for (let index = 0; index < samples.length; index += 1) samples[index] = index % 2 ? floor : -floor;
    for (const [time, value] of impulses) samples[Math.round(time * SAMPLE_RATE)] = value;
    return samples;
  }

  function column(startSeconds, endSeconds, columns, time) {
    return Math.floor(((time - startSeconds) / (endSeconds - startSeconds)) * columns);
  }

  test("returns one min/max pair per requested column", () => {
    const pyramid = buildPeakPyramid([impulseTrack(10, [])], SAMPLE_RATE);
    for (const columns of [1, 37, 320, 1280]) {
      const peaks = viewportPeaks(pyramid, 0, 10, columns);
      expect(peaks.min.length).toBe(columns);
      expect(peaks.max.length).toBe(columns);
      for (let index = 0; index < columns; index += 1) expect(peaks.min[index]).toBeLessThanOrEqual(peaks.max[index]);
    }
  });

  test("single-sample peaks stay at their true time at every zoom level and are never dropped", () => {
    const positive = 37.25;
    const negative = 12.5;
    const pyramid = buildPeakPyramid([impulseTrack(60, [[positive, 0.9], [negative, -0.7]])], SAMPLE_RATE);
    const columns = 1000;
    const views = [
      [0, 60],
      [30, 45],
      [36.5, 38.5],
      [37.2, 37.3],
    ];
    for (const [start, end] of views) {
      const { min, max } = viewportPeaks(pyramid, start, end, columns);
      const expected = column(start, end, columns, positive);
      const hits = [];
      for (let index = 0; index < columns; index += 1) if (max[index] > 0.5) hits.push(index);
      expect(hits.length).toBeGreaterThan(0);
      expect(Math.max(...hits.map((index) => Math.abs(index - expected)))).toBeLessThanOrEqual(1);
      expect(Math.max(...hits.map((index) => max[index]))).toBeCloseTo(0.9, 5);

      const negativeHits = [];
      for (let index = 0; index < columns; index += 1) if (min[index] < -0.5) negativeHits.push(index);
      if (negative >= start && negative < end) {
        const expectedNegative = column(start, end, columns, negative);
        expect(negativeHits.length).toBeGreaterThan(0);
        expect(Math.max(...negativeHits.map((index) => Math.abs(index - expectedNegative)))).toBeLessThanOrEqual(1);
      } else {
        // A peak outside the visible window must not leak into it.
        expect(negativeHits).toEqual([]);
      }
    }
  });

  test("zooming in reveals progressively finer detail of the same signal", () => {
    // 10 ms blocks alternating between full scale and a tenth of it.
    const seconds = 60;
    const samples = new Float32Array(seconds * SAMPLE_RATE);
    const block = SAMPLE_RATE / 100;
    for (let index = 0; index < samples.length; index += 1) {
      const loud = Math.floor(index / block) % 2 === 0;
      samples[index] = (loud ? 1 : 0.1) * (index % 2 ? 1 : -1);
    }
    const pyramid = buildPeakPyramid([samples], SAMPLE_RATE);
    const columns = 1000;
    const quietFraction = (start, end) => {
      const { max } = viewportPeaks(pyramid, start, end, columns);
      let quiet = 0;
      for (let index = 0; index < columns; index += 1) {
        expect(max[index]).toBeLessThanOrEqual(1 + 1e-6);
        expect(max[index]).toBeGreaterThanOrEqual(0.1 - 1e-6);
        if (max[index] < 0.5) quiet += 1;
      }
      return quiet / columns;
    };
    // Whole file: 60 ms per column, every column contains a loud block.
    expect(quietFraction(0, seconds)).toBe(0);
    // Beat scale: 1 ms per column, the quiet blocks are resolved (about half of the columns).
    const fine = quietFraction(30, 31);
    expect(fine).toBeGreaterThan(0.4);
    expect(fine).toBeLessThan(0.6);
  });

  test("query cost depends on the viewport width, not on the track's sample count", () => {
    const sampleRate = 44_100;
    const shortTrack = new Float32Array(20 * sampleRate);
    const longTrack = new Float32Array(10 * 60 * sampleRate);
    for (const track of [shortTrack, longTrack]) {
      for (let index = 0; index < track.length; index += 1) track[index] = Math.sin(index * 0.05) * 0.5;
    }
    const shortPyramid = buildPeakPyramid([shortTrack], sampleRate);
    const longPyramid = buildPeakPyramid([longTrack], sampleRate);
    const columns = 1280;

    const medianMs = (query) => {
      for (let warm = 0; warm < 5; warm += 1) query();
      const timings = [];
      for (let run = 0; run < 21; run += 1) {
        const started = performance.now();
        query();
        timings.push(performance.now() - started);
      }
      timings.sort((left, right) => left - right);
      return timings[Math.floor(timings.length / 2)];
    };

    const shortWhole = medianMs(() => viewportPeaks(shortPyramid, 0, 20, columns));
    const longWhole = medianMs(() => viewportPeaks(longPyramid, 0, 600, columns));
    const longPhrase = medianMs(() => viewportPeaks(longPyramid, 300, 330, columns));
    // A per-frame scan of all 26.5 M samples is tens of milliseconds and ~30× the short track's
    // cost; a bounded hierarchy keeps the 10-minute whole-file view within a small factor of it.
    expect(longWhole).toBeLessThanOrEqual(shortWhole * 4 + 1.5);
    expect(longPhrase).toBeLessThanOrEqual(shortWhole * 4 + 1.5);

    // Conservative envelope at the whole-file view of the long track.
    const { min, max } = viewportPeaks(longPyramid, 0, 600, columns);
    for (let index = 0; index < columns; index += 1) {
      expect(max[index]).toBeGreaterThan(0.49);
      expect(min[index]).toBeLessThan(-0.49);
    }
  });
});
