# audio-analysis-pitch

Autocorrelation pitch detection for `moritzbrantner-video-analysis` audio pipelines.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use audio_analysis_pitch::{PitchAnalyzer, PitchAnalyzerOptions};

let analyzer = PitchAnalyzer::new(PitchAnalyzerOptions::default())?;
let _ = analyzer;
```

## Package surface

Primary workflow: `audio.pitch.estimate`.

Workflow operations:

- `audio.pitch.estimate`: Estimates one fundamental frequency from normalized samples.
- `audio.pitch.track`: Estimates pitch over fixed frames and groups contiguous note segments.
- `audio.pitch.chroma`: Summarizes normalized samples into a 12-bin pitch-class chroma vector.

Debug operations:

- `describe`: inspect package metadata and runtime support.
- `audio.pitch.noteName`: Inspects the MIDI note and scientific note name for a frequency in hertz.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-audio-analysis-pitch-cli -- run \
  --operation audio.pitch.estimate \
  --json '{"sampleRate":48000,"samples":[0.0,1.0,0.0,-1.0]}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `audio-analysis-core`
- `audio-analysis-rhythm`
- `video-analysis-core`
