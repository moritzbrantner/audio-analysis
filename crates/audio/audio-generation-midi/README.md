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

The `karaoke` module combines already-aligned lyric fragments with a timestamped vocal pitch track. It reuses this crate's pitch-track-to-note consolidation so karaoke export and MIDI generation agree on musical note boundaries.

The neutral `KaraokeChart` keeps seconds, MIDI pitch, phrase structure, and confidence evidence. UltraStar v1 timing and pitch conventions are applied only by `export_ultrastar_v1`, so another singing-game or karaoke format can reuse the same chart later.

Callers should provide syllable-level alignments when available. Word-level alignments are accepted without guessing syllable boundaries. The first slice supports one lead voice, regular pitched notes, and at most one generated note per aligned lyric fragment; melisma splitting, rap/golden notes, and duet tracks remain explicit future capabilities.

```rust,ignore
use audio_generation_midi::karaoke::{
    build_karaoke_chart, export_ultrastar_v1, AlignedLyricFragment,
    KaraokeChartBuildOptions, UltraStarV1Metadata,
};
use audio_generation_midi::{MidiNote, PitchTrackFrame};

let lyrics = vec![AlignedLyricFragment {
    text: "Hello".to_string(),
    start_seconds: 0.5,
    end_seconds: 1.0,
    confidence: Some(0.95),
}];
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
        ..KaraokeChartBuildOptions::default()
    },
)?;
let ultrastar = export_ultrastar_v1(
    &built.chart,
    &UltraStarV1Metadata {
        title: "Example".to_string(),
        artist: "Singer".to_string(),
        audio_file: "song.ogg".to_string(),
    },
)?;

let _ = ultrastar;
# Ok::<(), audio_contracts::DetectError>(())
```

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
