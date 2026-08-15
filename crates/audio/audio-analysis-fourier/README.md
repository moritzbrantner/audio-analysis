# audio-analysis-fourier

FFT, STFT, and spectral audio analysis for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use audio_analysis_fourier::{DominantFrequencyAnalyzer, SpectrogramOptions};

let analyzer = DominantFrequencyAnalyzer::default();
let options = SpectrogramOptions {
    fft_size: 2_048,
    hop_size: 512,
    ..SpectrogramOptions::default()
};

let _ = analyzer;
let _ = options;
```

## Package surface

Primary workflow: `audio.fourier.spectrum`.

Workflow operations:

- `audio.fourier.spectrum`: Computes an FFT spectrum and returns dominant-frequency metadata.
- `audio.fourier.spectrogram`: Computes deterministic STFT frame summaries.
- `audio.fourier.features`: Returns spectral centroid, bandwidth, rolloff, flatness, zero-crossing rate, and optional mel-style band features.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-audio-analysis-fourier-cli -- run \
  --operation audio.fourier.spectrum \
  --json '{"fftSize":4,"sampleRate":48000,"samples":[0.0,1.0,0.0,-1.0]}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `audio-analysis-core`
- `audio-analysis-pitch`
- `audio-analysis-rhythm`
