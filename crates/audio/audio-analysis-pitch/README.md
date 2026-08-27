# audio-analysis-pitch

Monophonic pitch tracking plus polyphonic chroma and musical-key analysis for audio pipelines.

## Feature flags

- No optional feature flags today.

## Monophonic pitch

The existing autocorrelation path remains the right tool for a voice or single pitched source. `audio.pitch.estimate`, `audio.pitch.track`, and the compatibility `audio.pitch.chroma` operation keep their existing semantics.

```rust,ignore
use audio_analysis_pitch::AutocorrelationPitchDetector;

let detector = AutocorrelationPitchDetector::default();
let estimate = detector.estimate_samples(&samples, 48_000)?;
```

## Polyphonic musical key

`key::estimate_musical_key` is the music/DJ path. It does not reuse the monophonic fundamental detector. Instead it:

1. computes Hann-window STFT frames through `audio-analysis-fourier`;
2. extracts significant local spectral peaks between the configured frequency bounds;
3. estimates the track's global tuning offset relative to A440 and corrects it before pitch-class folding;
4. accumulates a normalized 12-bin HPCP-like chroma vector across frames; and
5. scores all 24 major/minor candidates using Krumhansl-Kessler, Temperley, or an ensemble of both profile correlations.

The result contains the best key, runner-up, profile strength, ambiguity confidence, tuning offset, and the underlying chroma vector. A low-confidence estimate should remain low-confidence rather than being promoted to authoritative metadata.

```rust,ignore
use audio_analysis_pitch::key::{estimate_musical_key, HarmonicKeyConfig};

if let Some(key) = estimate_musical_key(&samples, 48_000, HarmonicKeyConfig::default())? {
    println!("{} ({:.2})", key.label(), key.confidence);
}
```

## Package surface

Primary music workflow: `audio.pitch.key`.

Workflow operations:

- `audio.pitch.key`: Estimates major/minor musical key from tuning-corrected polyphonic spectral chroma.
- `audio.pitch.estimate`: Estimates one fundamental frequency from normalized samples.
- `audio.pitch.track`: Estimates monophonic pitch over fixed frames and groups contiguous note segments.
- `audio.pitch.chroma`: Preserves the existing monophonic 12-bin pitch-class projection for compatibility.

Debug operations:

- `describe`: inspect package metadata and runtime support.
- `audio.pitch.noteName`: Inspects the MIDI note and scientific note name for a frequency in hertz.

Runtime support: library, CLI, server, and WASM wrappers expose these operations. The JSON surface is intended for bounded sample payloads; full-track analysis should use the Rust library or the `audio-analysis-rhythm` `dj_analyze` file example.

Run an in-memory key-analysis preview through the CLI:

```bash
cargo run -p moritzbrantner-audio-analysis-pitch-cli -- run \
  --operation audio.pitch.key \
  --json '{"sampleRate":48000,"samples":[0.0,1.0,0.0,-1.0],"profile":"ensemble"}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Golden evaluation

`scripts/evaluate-dj-goldens.py` cross-checks the full-file Rust path against librosa and, when installed, Essentia. The reusable Brahms fixture is also checked against its known F-sharp-minor key, so the real-music evaluation is not based only on synthesized tones.

## Related crates

- `audio-analysis-core`
- `audio-analysis-fourier`
- `audio-analysis-rhythm`
