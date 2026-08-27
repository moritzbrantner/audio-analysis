# audio-analysis-rhythm

Onset, tempo, beat, and downbeat analysis for music and media audio pipelines.

## Feature flags

- No optional feature flags today.

## Whole-track music analysis

`track::analyze_rhythm_track` is the production path for music and DJ preparation. It keeps the older deterministic onset helpers available for compatibility, but replaces their simple interval estimator with a fuller pipeline:

1. Hann-window STFT analysis through `audio-analysis-fourier`.
2. Log-compressed, frequency-weighted spectral flux with local adaptive whitening.
3. Ranked tempo candidates from onset-envelope autocorrelation. Alternative candidates remain visible so half/double-tempo ambiguity is not hidden.
4. Dynamic-programming beat tracking around the selected beat period.
5. Four-beat bar-phase inference from transient and low-frequency beat accents.

The downbeat result is intentionally a confidence-bearing 4/4 heuristic rather than a claim of meter recognition. Tracks with unusual meter, weak bar accents, changing tempo, or intentionally ambiguous half-time feel should use the returned candidates and confidence values rather than treating one number as ground truth.

```rust,ignore
use audio_analysis_rhythm::track::{analyze_rhythm_track, TrackRhythmConfig};

let analysis = analyze_rhythm_track(&samples, 48_000, TrackRhythmConfig::default())?;
println!("BPM: {:?}", analysis.bpm);
println!("beats: {}", analysis.beats.len());
println!("downbeats: {}", analysis.downbeats.len());
```

For a file-backed smoke path using the repository's normal FFmpeg decoder plus musical-key analysis:

```text
cargo run -p moenarch-audio-analysis-rhythm --example dj_analyze -- song.mp3
```

## Package surface

Primary music workflow: `audio.rhythm.analyze`.

Workflow operations:

- `audio.rhythm.analyze`: Estimates ranked BPM candidates, beats, and 4/4 downbeats from a spectral-flux novelty curve.
- `audio.rhythm.onsets`: Computes the legacy deterministic onset envelope and onset list.
- `audio.rhythm.tempo`: Estimates BPM from detected onset intervals for compatibility and small deterministic inputs.
- `audio.rhythm.beatGrid`: Creates an exact beat grid from an already-known start time, BPM, and beat count.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations. The JSON runtime surface remains deliberately sample-count bounded; full songs should use the library or file-backed example rather than embedding millions of PCM samples in JSON.

Run the music workflow through the CLI with an in-memory preview:

```bash
cargo run -p moritzbrantner-audio-analysis-rhythm-cli -- run \
  --operation audio.rhythm.analyze \
  --json '{"sampleRate":48000,"samples":[1.0,0.0,0.0,1.0]}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Golden evaluation

`scripts/evaluate-dj-goldens.py` downloads checksum-pinned reusable music fixtures into `target/`, runs this crate's real file-analysis path, and compares tempo against librosa and, when available, Essentia. Half/double tempo is treated as an explicit equivalence class. The same harness checks the public-domain Brahms *Hungarian Dance No. 5* fixture against its known F-sharp-minor key through `audio-analysis-pitch`.

The evaluator is opt-in because the third-party Python packages and audio downloads are intentionally outside ordinary deterministic repository CI.

## Related crates

- `audio-analysis-core`
- `audio-analysis-fourier`
- `audio-analysis-pitch`
