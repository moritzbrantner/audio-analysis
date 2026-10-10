// Deterministic synthetic songs for audio-analysis#138 acceptance.
//
// Each section is 24 s, a whole number of bars at both 120 BPM (12 bars) and 150 BPM (15 bars), so
// the Rust structure detector sees a section change exactly on a downbeat.
//
// - `contrastingSong`: A (120 BPM, C major, bass) → B (150 BPM, F# major, bright shimmer and hats)
//   → A again. A sustained tempo and key change with a returning section identity.
// - `steadySong`: the same A → B → A timbre/arrangement contrast, but every section stays at 120 BPM
//   in C major (B drifts by a negligible 0.5 BPM). The structure changes; tempo and key do not.
export const SAMPLE_RATE = 16_000;
export const SECTION_SECONDS = 24;

const HZ = {
  C2: 65.41,
  C3: 130.81,
  C4: 261.63,
  E4: 329.63,
  G4: 392.0,
  Fs4: 369.99,
  As4: 466.16,
  Cs5: 554.37,
};
const SHIMMER_HZ = [4_100, 5_300, 6_200, 7_300];

const A = { bpm: 120, chord: ["C2", "C3", "C4", "E4", "G4"], bass: 0.3, bright: false };
const B_CONTRAST = { bpm: 150, chord: ["Fs4", "As4", "Cs5"], bass: 0, bright: true };
const B_STEADY = { bpm: 120.5, chord: ["C4", "E4", "G4"], bass: 0, bright: true };

export const contrastingSong = {
  name: "whole-track-sections-tempo-key-change.wav",
  sections: [A, B_CONTRAST, A],
};

export const steadySong = {
  name: "whole-track-sections-steady-tempo-key.wav",
  sections: [A, B_STEADY, A],
};

export function songDurationSeconds(song) {
  return song.sections.length * SECTION_SECONDS;
}

function lcg(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 4_294_967_296;
  };
}

export function songSamples(song) {
  const random = lcg(138);
  const total = songDurationSeconds(song) * SAMPLE_RATE;
  const samples = new Float32Array(total);
  for (let index = 0; index < total; index += 1) {
    const time = index / SAMPLE_RATE;
    const sectionIndex = Math.min(song.sections.length - 1, Math.floor(time / SECTION_SECONDS));
    const section = song.sections[sectionIndex];
    const local = time - sectionIndex * SECTION_SECONDS;
    const beatSeconds = 60 / section.bpm;
    const beat = Math.floor(local / beatSeconds);
    const sinceBeat = local - beat * beatSeconds;

    let value = 0;
    for (const note of section.chord) value += 0.07 * Math.sin(2 * Math.PI * HZ[note] * time);
    if (section.bass) value += section.bass * Math.sin(2 * Math.PI * HZ.C2 * time);
    if (section.bright) {
      for (const frequency of SHIMMER_HZ) value += 0.06 * Math.sin(2 * Math.PI * frequency * time);
      const sinceEighth = local % (beatSeconds / 2);
      value += 0.15 * Math.exp(-sinceEighth * 80) * (random() * 2 - 1);
    }
    const downbeat = beat % 4 === 0;
    if (sinceBeat < 0.025) {
      value +=
        (downbeat ? 0.8 : 0.5) *
        Math.exp(-sinceBeat / 0.006) *
        Math.sin(2 * Math.PI * (downbeat ? 1_500 : 1_000) * sinceBeat);
    }
    samples[index] = Math.max(-0.95, Math.min(0.95, value));
  }
  return samples;
}

export function songWav(song) {
  const samples = songSamples(song);
  const dataBytes = samples.length * 2;
  const buffer = Buffer.alloc(44 + dataBytes);
  buffer.write("RIFF", 0, "ascii");
  buffer.writeUInt32LE(36 + dataBytes, 4);
  buffer.write("WAVE", 8, "ascii");
  buffer.write("fmt ", 12, "ascii");
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(SAMPLE_RATE, 24);
  buffer.writeUInt32LE(SAMPLE_RATE * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36, "ascii");
  buffer.writeUInt32LE(dataBytes, 40);
  for (let index = 0; index < samples.length; index += 1) {
    buffer.writeInt16LE(Math.round(samples[index] * 32_767), 44 + index * 2);
  }
  return buffer;
}
