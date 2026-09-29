# audio-analysis-recognition

Deterministic audio embeddings, similarity search, and recognition contracts for
`moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use audio_analysis_recognition::{
    compare_audio_samples, SpectralAudioEmbedder, SpectralEmbeddingConfig,
};

let extractor = SpectralAudioEmbedder::new(SpectralEmbeddingConfig::default())?;
let response = compare_audio_samples(
    &[0.0, 1.0, 0.0, -1.0],
    &[0.0, 0.9, 0.0, -0.9],
    48_000,
    &extractor,
)?;

assert!(response.score.is_finite());
```

`SpeechRecognitionRequest` and `transcribe_audio` remain available as
compatibility shims for existing callers. New transcription integrations should
use `audio-analysis-transcription` for real ASR or `text-transcripts` for
imported transcript normalization.

Default builds do not run native ASR, download models, call network services, or
spawn external transcription commands. Native transcription orchestration lives
in `audio-analysis-transcription`; imported transcript contracts live in
`text-transcripts`.

## Package surface

Primary workflow: `audio.recognition.embed`.

Workflow operations:

- `audio.recognition.embed`: Computes a deterministic spectral embedding for normalized samples.
- `audio.recognition.compare`: Compares two in-memory sample arrays by cosine similarity.
- `audio.recognition.search`: Builds a transient sample-backed reference library and searches it.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-audio-analysis-recognition-cli -- run \
  --operation audio.recognition.embed \
  --json '{"bands":8,"sampleRate":48000,"samples":[0.0,1.0,0.0,-1.0]}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `audio-analysis-core`
- `audio-analysis-fourier`
- `text-transcripts`
