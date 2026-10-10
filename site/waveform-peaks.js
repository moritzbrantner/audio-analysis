// Bounded multiresolution min/max peaks for the Audio Inspector waveform.
//
// The decoded file is summarized once into a pyramid of min/max buckets over the mono mix: the
// finest level holds one pair per BASE_BUCKET_FRAMES frames and every coarser level halves the
// bucket count. A query for `columns` pixel columns reads the coarsest level whose buckets are no
// wider than a column, so it touches at most a few buckets per column whatever the file length.
// Views narrower than the finest level read the PCM directly, which is then at most
// BASE_BUCKET_FRAMES frames per column. Envelopes are conservative: a column covers every bucket
// that overlaps it, so single-sample peaks are never dropped and stay within one column of their
// true time. The overview and the detailed view share one pyramid.

export const BASE_BUCKET_FRAMES = 32;

// Incremental O(n) builder: PCM can be appended in consecutive chunks of any size (the inspector
// streams bounded chunks to `waveform-peaks-worker.js`), so no complete copy of the file is needed.
// `finish()` returns a summary that holds only typed arrays and can be transferred from a worker.
export function createPeakBuilder(channelCount, length, sampleRate) {
  const totalFrames = Math.max(0, Math.floor(Number(length) || 0));
  const channelTotal = Math.max(1, Math.floor(Number(channelCount) || 1));
  const count = Math.ceil(totalFrames / BASE_BUCKET_FRAMES);
  const min = new Float32Array(count).fill(Infinity);
  const max = new Float32Array(count).fill(-Infinity);
  const scale = 1 / channelTotal;
  let position = 0;

  return {
    append(channels) {
      const sources = Array.from(channels ?? []);
      if (sources.length !== channelTotal) throw new Error(`expected ${channelTotal} channels, got ${sources.length}`);
      const frames = Math.min(totalFrames - position, ...sources.map((channel) => channel.length));
      let offset = 0;
      while (offset < frames) {
        const bucket = Math.floor(position / BASE_BUCKET_FRAMES);
        const stop = Math.min(frames, offset + (bucket + 1) * BASE_BUCKET_FRAMES - position);
        let low = min[bucket];
        let high = max[bucket];
        if (channelTotal === 1) {
          const channel = sources[0];
          for (let frame = offset; frame < stop; frame += 1) {
            const value = channel[frame];
            if (value < low) low = value;
            if (value > high) high = value;
          }
        } else {
          for (let frame = offset; frame < stop; frame += 1) {
            let value = 0;
            for (const channel of sources) value += channel[frame];
            value *= scale;
            if (value < low) low = value;
            if (value > high) high = value;
          }
        }
        min[bucket] = low;
        max[bucket] = high;
        position += stop - offset;
        offset = stop;
      }
    },
    get appendedFrames() {
      return position;
    },
    finish() {
      if (position < totalFrames) throw new Error(`peak builder received ${position} of ${totalFrames} frames`);
      return summarize(min, max, totalFrames, sampleRate);
    },
  };
}

// Builds the summary of complete channels in one pass (tests, and callers that already hold the PCM).
export function buildPeakLevels(channels, sampleRate) {
  const sources = Array.from(channels ?? []).filter((channel) => channel && typeof channel.length === "number");
  const length = sources.length ? Math.min(...sources.map((channel) => channel.length)) : 0;
  if (!length) return { sampleRate: Number(sampleRate) || 0, length: 0, levels: [] };
  const builder = createPeakBuilder(sources.length, length, sampleRate);
  builder.append(sources);
  return builder.finish();
}

function summarize(min, max, length, sampleRate) {
  const levels = [];
  if (!length) return { sampleRate: Number(sampleRate) || 0, length: 0, levels };
  levels.push({ bucketFrames: BASE_BUCKET_FRAMES, min, max });
  while (levels.at(-1).min.length > 1) {
    const previous = levels.at(-1);
    const nextCount = Math.ceil(previous.min.length / 2);
    const nextMin = new Float32Array(nextCount);
    const nextMax = new Float32Array(nextCount);
    for (let bucket = 0; bucket < nextCount; bucket += 1) {
      const left = bucket * 2;
      const right = Math.min(previous.min.length - 1, left + 1);
      nextMin[bucket] = Math.min(previous.min[left], previous.min[right]);
      nextMax[bucket] = Math.max(previous.max[left], previous.max[right]);
    }
    levels.push({ bucketFrames: previous.bucketFrames * 2, min: nextMin, max: nextMax });
  }
  return { sampleRate: Number(sampleRate) || 0, length, levels };
}

// Attaches the PCM for sub-bucket views to a summary (for example one received from the worker).
export function peakPyramidFromLevels(summary, channels) {
  return { ...summary, channels: Array.from(channels ?? []) };
}

export function buildPeakPyramid(channels, sampleRate) {
  return peakPyramidFromLevels(buildPeakLevels(channels, sampleRate), channels);
}

export function peakSummaryTransferables(summary) {
  return summary.levels.flatMap((level) => [level.min.buffer, level.max.buffer]);
}

export function viewportPeaks(pyramid, startSeconds, endSeconds, columns) {
  const count = Math.max(0, Math.floor(Number(columns) || 0));
  const min = new Float32Array(count);
  const max = new Float32Array(count);
  const { sampleRate, length } = pyramid;
  if (!count || !length || !(sampleRate > 0) || !(endSeconds > startSeconds)) return { min, max };

  const startFrame = startSeconds * sampleRate;
  const framesPerColumn = ((endSeconds - startSeconds) * sampleRate) / count;
  const level = levelFor(pyramid.levels, framesPerColumn);
  if (level) fillFromLevel(level, startFrame, framesPerColumn, min, max);
  else fillFromSamples(pyramid, startFrame, framesPerColumn, min, max);
  return { min, max };
}

function levelFor(levels, framesPerColumn) {
  let chosen = null;
  for (const level of levels) {
    if (level.bucketFrames > framesPerColumn) break;
    chosen = level;
  }
  return chosen;
}

function fillFromLevel(level, startFrame, framesPerColumn, min, max) {
  const buckets = level.min.length;
  const size = level.bucketFrames;
  for (let column = 0; column < min.length; column += 1) {
    const from = Math.max(0, Math.floor((startFrame + column * framesPerColumn) / size));
    const to = Math.min(buckets, Math.ceil((startFrame + (column + 1) * framesPerColumn) / size));
    if (to <= from) continue;
    let low = level.min[from];
    let high = level.max[from];
    for (let bucket = from + 1; bucket < to; bucket += 1) {
      if (level.min[bucket] < low) low = level.min[bucket];
      if (level.max[bucket] > high) high = level.max[bucket];
    }
    min[column] = low;
    max[column] = high;
  }
}

function fillFromSamples(pyramid, startFrame, framesPerColumn, min, max) {
  const channels = pyramid.channels ?? [];
  if (!channels.length) return;
  const scale = 1 / channels.length;
  for (let column = 0; column < min.length; column += 1) {
    const columnStart = startFrame + column * framesPerColumn;
    const from = Math.max(0, Math.floor(columnStart));
    const to = Math.min(pyramid.length, Math.max(Math.floor(columnStart) + 1, Math.ceil(columnStart + framesPerColumn)));
    if (to <= from) continue;
    let low = Infinity;
    let high = -Infinity;
    for (let frame = from; frame < to; frame += 1) {
      let value = 0;
      for (const channel of channels) value += channel[frame];
      value *= scale;
      if (value < low) low = value;
      if (value > high) high = value;
    }
    min[column] = low;
    max[column] = high;
  }
}
