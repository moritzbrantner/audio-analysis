# audio-analysis-synthesis

Deterministic audio synthesis from analysis events for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use audio_analysis_synthesis::{synthesize_tone_timeline, ToneEvent};

let events = vec![ToneEvent::new(440.0, 0.0, 0.5)?];
let frames = synthesize_tone_timeline(&events, 48_000)?;

let _ = frames;
```

## Package surface

Primary workflow: `audio.synthesis.tone`.

Workflow operations:

- `audio.synthesis.tone`: Generates an in-memory analytic tone frame.
- `audio.synthesis.timeline`: Generates an in-memory tone timeline from segment specs.
- `audio.synthesis.fromEvents`: Converts pitch/onset event labels into tone segments and synthesizes them.
- `audio.synthesis.clickTrack`: Generates a deterministic in-memory click track from BPM or explicit beat positions.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-audio-analysis-synthesis-cli -- run \
  --operation audio.synthesis.tone \
  --json '{"channels":1,"durationSeconds":0.1,"frequencyHz":440.0,"sampleRate":48000}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `audio-analysis-core`
- `data-inversion-core`
- `video-analysis-core`
