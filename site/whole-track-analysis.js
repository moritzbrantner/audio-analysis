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

export async function analyzeWholeTrack(samples, sampleRate, durationSeconds) {
  const [rhythm, pitch] = await Promise.all([loadRhythm(), loadPitch()]);
  const song = await rhythm.analyzeTrack(samples, sampleRate, RHYTHM_OPTIONS);
  if (!song || typeof song !== "object") {
    throw new Error("The rhythm analyzer returned no whole-track result.");
  }
  const key = await pitch.analyzeTrackKey(samples, sampleRate, {
    ...KEY_OPTIONS,
    barBoundariesSeconds: keyBarBoundaries(song.downbeats, durationSeconds),
  });
  if (!key || typeof key !== "object") {
    throw new Error("The musical-key analyzer returned no whole-track result.");
  }
  return { song, key };
}

export function keyBarBoundaries(downbeats, duration) {
  if (!Array.isArray(downbeats) || downbeats.length < 2 || !Number.isFinite(duration) || duration <= 0) {
    return [];
  }
  const boundaries = downbeats
    .map((time) => Number(time))
    .filter((time) => Number.isFinite(time) && time >= 0 && time <= duration)
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
