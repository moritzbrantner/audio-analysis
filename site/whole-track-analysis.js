// Whole-track musical analysis for the Audio Inspector waveform.
//
// This module only orchestrates the existing Rust/WASM whole-song entry points (`analyzeTrack`
// for the `audio-analysis-song/v1` contract and `analyzeTrackKey` for the
// `audio-analysis-key-track/*` contract). It adds no browser-side semantics: tempo, beats,
// sections and key segments are returned exactly as Rust produced them. It is DOM-free so it can
// run inside `whole-track-worker.js` and keep the full-file analysis off the UI thread.

export const WHOLE_TRACK_RATE = 16_000;
export const WHOLE_TRACK_MAX_SECONDS = 15 * 60;

const RHYTHM_OPTIONS = Object.freeze({
  minBpm: 45,
  maxBpm: 220,
  fftSize: 1024,
  hopSize: 128,
  timeOffsetSeconds: 0,
});

const KEY_OPTIONS = Object.freeze({
  fftSize: 4096,
  hopSize: 2048,
  profile: "ensemble",
  timelineWindowSeconds: 24,
  timelineHopSeconds: 8,
  timelineMinConfidence: 0.1,
});

let rhythmModulePromise = null;
let pitchModulePromise = null;

// Mixes decoded channel PCM to mono and resamples it for the whole-track analyzers. Runs inside the
// worker so full-file preparation never blocks the UI thread.
export function prepareWholeTrackSamples(channels, sourceRate, targetRate = Math.min(sourceRate, WHOLE_TRACK_RATE)) {
  const length = channels.reduce((shortest, channel) => Math.min(shortest, channel.length), Number.POSITIVE_INFINITY);
  if (!channels.length || !Number.isFinite(length) || length <= 0) {
    throw new Error("The decoded audio buffer is empty.");
  }
  const scale = sourceRate / targetRate;
  const outputLength = Math.max(1, Math.floor(length / scale));
  const output = new Float32Array(outputLength);

  for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
    const start = outputIndex * scale;
    const end = Math.min(length, (outputIndex + 1) * scale);
    const first = Math.floor(start);
    const last = Math.min(length, Math.ceil(end));
    let weightedSum = 0;
    let weight = 0;
    for (let sourceIndex = first; sourceIndex < last; sourceIndex += 1) {
      const overlap = Math.min(end, sourceIndex + 1) - Math.max(start, sourceIndex);
      if (overlap <= 0) continue;
      let mono = 0;
      for (const channel of channels) mono += channel[sourceIndex];
      weightedSum += (mono / channels.length) * overlap;
      weight += overlap;
    }
    output[outputIndex] = weight > 0 ? weightedSum / weight : 0;
  }
  return { samples: output, sampleRate: targetRate };
}

// Analyzes mono PCM. Every duration handed to Rust is derived from the analyzed samples, never from
// the decoded buffer, because resampling floors the sample count.
export async function analyzeWholeTrack(samples, sampleRate) {
  const analyzedDurationSeconds = samples.length / sampleRate;
  const [rhythm, pitch] = await Promise.all([loadRhythm(), loadPitch()]);
  const song = await rhythm.analyzeTrack(samples, sampleRate, RHYTHM_OPTIONS);
  if (!song || typeof song !== "object") {
    throw new Error("The rhythm analyzer returned no whole-track result.");
  }
  const key = await pitch.analyzeTrackKey(samples, sampleRate, {
    ...KEY_OPTIONS,
    barBoundariesSeconds: keyBarBoundaries(song.downbeats, analyzedDurationSeconds),
  });
  if (!key || typeof key !== "object") {
    throw new Error("The musical-key analyzer returned no whole-track result.");
  }
  return { song, key, analyzedDurationSeconds };
}

// Bar boundaries for the key timeline: strictly increasing, inside [0, duration], where duration
// is the analyzed sample duration that Rust validates against.
export function keyBarBoundaries(downbeats, duration) {
  if (!Array.isArray(downbeats) || downbeats.length < 2 || !Number.isFinite(duration) || duration <= 0) {
    return [];
  }
  const boundaries = downbeats
    .map((time) => Number(time))
    .filter((time) => Number.isFinite(time) && time >= 0)
    .map((time) => Math.min(time, duration))
    .sort((left, right) => left - right);
  const unique = Array.from(new Set(boundaries));
  if (unique.length < 2) return [];
  if (unique[0] > 0) unique.unshift(0);
  if (unique.at(-1) < duration) unique.push(duration);
  return unique;
}

function loadRhythm() {
  rhythmModulePromise ??= import("./wasm/audio-analysis-rhythm/index.js").then(async (module) => {
    await module.init();
    return module;
  });
  return rhythmModulePromise;
}

function loadPitch() {
  pitchModulePromise ??= import("./wasm/audio-analysis-pitch/index.js").then(async (module) => {
    await module.init();
    return module;
  });
  return pitchModulePromise;
}
