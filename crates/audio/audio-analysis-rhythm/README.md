# audio-analysis-rhythm

Onset, tempo, beat, downbeat, and rhythmic-section analysis for music and media audio pipelines.

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

- `audio.rhythm.analyze`: Estimates ranked BPM candidates, beats, and 4/4 downbeats from a spectral-flux novelty curve. The runtime result uses the versioned `audio-analysis-song/v1` contract, keeps every tracked beat, adds integer-millisecond and `HH:MM:SS.mmm` timestamps, and derives conservative rhythmic structural sections from downbeat-aligned intensity changes.
- `audio.rhythm.onsets`: Computes the legacy deterministic onset envelope and onset list.
- `audio.rhythm.tempo`: Estimates BPM from detected onset intervals for compatibility and small deterministic inputs.
- `audio.rhythm.beatGrid`: Creates an exact beat grid from an already-known start time, BPM, and beat count.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations. Preview-oriented operations keep the smaller sample limit. `audio.rhythm.analyze` accepts up to 15 minutes of PCM at the supplied sample rate so the browser whole-song workflow can explicitly opt into heavier local analysis without making the fast Audio Inspector pay that cost by default.

Structural `sections` are intentionally rhythmic change points named `section-1`, `section-2`, and so on. They are not claims that a segment is a verse, chorus, bridge, or another semantic song form; that would require a separate classifier and evaluation contract.

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

`scripts/evaluate-dj-goldens.py` downloads checksum-pinned reusable music fixtures into `target/`, runs this crate's real file-analysis path, and compares tempo against librosa and, when available, Essentia. Half/double tempo is treated as an explicit equivalence class. Key references are pinned from Essentia 2.1b6.dev1389 on the exact fixture bytes (Choice: G major; the Brahms fixture: G minor) rather than inferred from filenames or titles.

The evaluator is opt-in because the third-party Python packages and audio downloads are intentionally outside ordinary deterministic repository CI.

## Related crates

- `audio-analysis-core`
- `audio-analysis-fourier`
- `audio-analysis-pitch`