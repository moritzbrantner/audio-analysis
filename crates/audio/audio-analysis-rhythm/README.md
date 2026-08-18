# audio-analysis-rhythm

Onset and tempo analysis for `moritzbrantner-video-analysis` audio pipelines.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use audio_analysis_rhythm::{RhythmAnalyzer, RhythmAnalyzerOptions};

let analyzer = RhythmAnalyzer::new(RhythmAnalyzerOptions::default())?;
let _ = analyzer;
```

## Package surface

Primary workflow: `audio.rhythm.onsets`.

Workflow operations:

- `audio.rhythm.onsets`: Computes an onset envelope and deterministic onset list.
- `audio.rhythm.tempo`: Estimates BPM from detected onset intervals.
- `audio.rhythm.beatGrid`: Creates a beat grid from start time, BPM, and beat count.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-audio-analysis-rhythm-cli -- run \
  --operation audio.rhythm.onsets \
  --json '{"frameSize":2,"hopSize":1,"sampleRate":1000,"samples":[1.0,0.0,0.0,1.0]}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `audio-analysis-core`
- `audio-analysis-fourier`
- `audio-analysis-pitch`
