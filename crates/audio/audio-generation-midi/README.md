# audio-generation-midi

MIDI-like note sequencing, Standard MIDI export, deterministic audio rendering, and neutral vocal/karaoke chart generation helpers for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use audio_generation_midi::{MidiNote, MidiNoteEvent, MidiSong, MidiTrack, NoteName};

let mut lead = MidiTrack::new("lead");
lead.push(MidiNoteEvent::new(
    MidiNote::from_name(NoteName::A, 4)?,
    0.0,
    1.0,
)?)?;

let song = MidiSong::new(120.0)?.with_track(lead)?;
let midi_bytes = song.to_midi_bytes()?;
let audio = song.render(Default::default())?;

let _ = (midi_bytes, audio);
# Ok::<(), audio_contracts::DetectError>(())
```

## Karaoke chart generation

The `karaoke` module combines canonical `media_core::TimedTextWordContract` values with a timestamped vocal pitch track. It deliberately does not define a second transcript/alignment DTO. It reuses this crate's pitch-track-to-note consolidation so neutral karaoke construction and MIDI generation agree on musical note boundaries.

The neutral `KaraokeChart` keeps seconds, MIDI pitch, phrase structure, confidence evidence, and whether a note starts or continues an already-aligned lyric fragment. This crate does not own UltraStar, SingStar, or other karaoke-file serialization. Format-specific exporters belong in downstream adapters so their quantization, metadata, and grammar rules cannot leak back into the neutral model.

Callers should provide syllable-level timed-text words when available. Word-level alignments remain word-level: the builder never guesses a syllable boundary. The default `KaraokeMelismaMode::SingleBestNote` preserves one generated note per aligned lyric fragment. Callers that already trust the lyric interval can opt into `SplitAcrossPitchNotes`; every consolidated pitch note sufficiently covered by that interval is retained, the first carries `KaraokeLyricRole::Primary`, and later notes carry `Continuation`. `min_pitch_note_overlap_ratio` rejects incidental edge overlaps while `min_overlap_ratio` still requires enough of the complete lyric interval to be explained by accepted notes.

```rust,ignore
use audio_generation_midi::karaoke::{
    build_karaoke_chart, KaraokeChartBuildOptions, KaraokeMelismaMode,
};
use audio_generation_midi::{MidiNote, PitchTrackFrame};
use media_core::TimedTextWordContract;

let lyrics = vec![TimedTextWordContract::new("Hello")
    .with_time_range(Some(0.5), Some(1.0))?
    .with_confidence(Some(0.95))?];
let pitch = vec![PitchTrackFrame {
    start_seconds: 0.5,
    end_seconds: 1.0,
    frequency_hz: MidiNote::new(60)?.frequency_hz(),
    confidence: 0.9,
}];
let built = build_karaoke_chart(
    &lyrics,
    &pitch,
    KaraokeChartBuildOptions {
        tempo_bpm: 120.0,
        beat_zero_seconds: 0.5,
        melisma_mode: KaraokeMelismaMode::SplitAcrossPitchNotes,
        ..KaraokeChartBuildOptions::default()
    },
)?;

let _ = built.chart;
# Ok::<(), audio_contracts::DetectError>(())
```

Downstream integer-grid exporters must validate their encoded note sequence as a whole. If minimum-duration rounding would make adjacent notes overlap, the adapter must reject the representation (or use a documented joint quantization policy) rather than silently changing the single-lead timing.

Rap/golden-note classification, duet tracks, and format-specific integer grids remain explicit downstream or later neutral capabilities rather than hidden heuristics in pitch/lyric fusion.

## Package surface

Primary workflow: `audio.midi.render`.

Workflow operations:

- `audio.midi.encode`: Encodes a deterministic single-track MIDI byte stream and returns a byte summary.
- `audio.midi.render`: Renders a MIDI-like note sequence into deterministic in-memory audio samples.
- `audio.midi.fromPitchTrack`: Converts pitch-track frames into merged MIDI-like note events and a byte summary.

Karaoke chart construction is currently library-first and intentionally does not add another transport operation in this slice.

Debug operations:

- `describe`: inspect package metadata and runtime support.
- `audio.midi.note`: Inspects frequency metadata for a MIDI note number or note name.

Runtime support: library, CLI, server, and WASM wrappers expose the MIDI workflow operations above.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-audio-generation-midi-cli -- run \
  --operation audio.midi.render \
  --json '{"notes":[{"durationBeats":1.0,"note":69,"startBeats":0.0}],"sampleRate":48000,"tempoBpm":120.0}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `audio-analysis-pitch`
- `audio-analysis-rhythm`
- `audio-analysis-separation`
- `audio-analysis-transcription`
- `audio-analysis-synthesis`
- `video-analysis-core`
