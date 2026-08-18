# audio-generation-midi

MIDI-like note sequencing, Standard MIDI export, and deterministic audio rendering helpers for `moritzbrantner-video-analysis`.

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

## Package surface

Primary workflow: `audio.midi.render`.

Workflow operations:

- `audio.midi.encode`: Encodes a deterministic single-track MIDI byte stream and returns a byte summary.
- `audio.midi.render`: Renders a MIDI-like note sequence into deterministic in-memory audio samples.
- `audio.midi.fromPitchTrack`: Converts pitch-track frames into merged MIDI-like note events and a byte summary.

Debug operations:

- `describe`: inspect package metadata and runtime support.
- `audio.midi.note`: Inspects frequency metadata for a MIDI note number or note name.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

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

- `audio-analysis-synthesis`
- `video-analysis-core`
