# audio-analysis-speakers

Speaker-domain APIs for `moritzbrantner-video-analysis`.

This crate keeps `moritzbrantner-audio-analysis-recognition` focused on generic embeddings and reference search while adding speaker-specific concepts:

- speaker IDs and labels
- model-versioned speaker embeddings
- speaker profiles and library snapshots
- enrollment and thresholded identification
- the shared `VoiceActivityDetector` trait, `SpeechSpan`, `VadOptions` and the
  deterministic `EnergyVoiceActivityDetector`, also used by transcription chunking
- diarization traits and a simple VAD/window/cluster diarizer
- shared speaker diarization options, response DTOs, and Transcript Speaker
  Assignment policies

`SpectralSpeakerEmbedder` and the native diarization baseline are deterministic
heuristics intended for tests and prototypes. They are not production-grade
speaker verification or diarization. Production systems should use a
model-backed embedder such as ECAPA-TDNN, x-vector, pyannote-style, or
SpeechBrain-compatible speaker verification models.

`SpeakerDiarizationOptions`, `SpeakerDiarizationResponse`,
`SpeakerSegmentPrediction`, and `SpeakerTranscriptAssignmentPolicy` are the
speaker-owned contracts used by transcription orchestration and package
surfaces. Transcript Speaker Assignment applies diarization spans to transcript
words and segments while preserving the existing JSON field names used by
package consumers.

`WindowedSpeakerDiarizer` supports optional unknown-speaker cluster bounds via
`speaker_bounds(min_speakers, max_speakers)`. The diarizer first preserves
accepted known/enrolled speaker matches, then clusters only unknown windows. A
maximum speaker bound assigns later unmatched unknown windows to the nearest
existing cluster once the maximum is reached. A minimum speaker bound splits
the most internally dispersed unknown clusters until the requested minimum is
reached or there are too few unknown speech windows. Unknown labels are
assigned deterministically by first occurrence as `speaker_0`, `speaker_1`, and
so on.

Uncertainty policy: speech spans shorter than `min_speaker_evidence_seconds`
(default `DEFAULT_MIN_SPEAKER_EVIDENCE_SECONDS` = 0.5 s) carry too little
embedding context to attribute. They are not embedded, identified or clustered;
they are returned as `DiarizedSpeaker::Unresolved` with their original start,
end and VAD score, so a short fragment never creates an extra speaker. Flat
`SpeakerSegmentPrediction` outputs omit unresolved spans, which leaves
transcript words there unassigned. The model-backed ONNX path uses the
default; the heuristic spectral baselines set it to `0.0` and attribute every
span. Set `min_speaker_evidence_seconds(0.0)` to opt out.

The first reviewed production direction is an opt-in ONNX speaker embedding
provider, documented in
`docs/adr/0003-native-speaker-diarization-provider.md`. Heuristic diarization
remains the default and no default test requires model files, network access,
Python, pyannote auth, Hugging Face tokens, or CUDA.

Model-backed embedders can implement `SpeakerEmbeddingProvider`, which returns
`SpeakerEmbeddingResponse` with model id, runtime, normalized
`SpeakerEmbedding`, and diagnostics. Existing `SpeakerEmbeddingExtractor`
callers remain supported.

## Feature flags

- `external-tests`: enables ignored local smoke tests that require caller-owned
  WAV fixtures.
- `onnx`: enables `runtime-onnx` execution for caller-owned local ONNX speaker
  embedding models.
- `model-bundles`: enables model-runtime manifest lookup for local ONNX bundles.
- `pyannote-diarization`: enables the native, opt-in community-1 runtime for an
  already-local approved bundle. The runtime validates the pinned source
  revision, checksum-addressed artifact set, ONNX names/dtypes/ranks, typed
  PLDA transforms, and VBx configuration before inference. It never downloads
  or hydrates model files.

The `PyannoteCommunityDiarizationConfig` /
`PyannoteCommunityDiarizer` interface consumes the converted community bundle:
`pyannote_diarization_manifest.json`, `segmentation.onnx`, `embedding.onnx`,
`plda_transform.json`, `plda_model.json`, `clustering.json`, provenance, and
license documentation. Segmentation executes with f32 `[1, 1, N]`; embedding
executes with f32 waveform `[1, 1, N]` and masks `[1, 589]`. Diagnostics report
the approved artifact digest and applied runtime stages, never caller paths.

The ignored retained two-speaker check is explicit and fails when its
caller-owned resources are absent:

```bash
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
PYANNOTE_COMMUNITY_BUNDLE=/path/to/sha256-0a121898...c577 \
PYANNOTE_TWO_SPEAKER_WAV=/path/to/retained-two-speaker.wav \
cargo test -p moenarch-audio-analysis-speakers \
  --features pyannote-diarization \
  --test pyannote_community_external -- --ignored
```

It compares native turns against the pinned upstream pipeline
permutation-invariantly with exact two-speaker bounds and a retained
frame-disagreement tolerance. No model or media artifact is committed.

Ignored ONNX smoke:

```bash
RUN_NATIVE_SPEAKER_MODEL_TESTS=1 \
ORT_DYLIB_PATH="$PWD/.audio-tools/whisperx-venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so.1.26.0" \
SPEAKER_EMBEDDING_MODEL_BUNDLE=/path/to/onnx-speaker-model \
SPEAKER_EMBEDDING_MODEL_FILE=model.onnx \
DIARIZATION_AUDIO_PATH=/path/to/meeting.wav \
cargo test -p moritzbrantner-audio-analysis-speakers \
  --features onnx,model-bundles \
  onnx_speaker_embedding_smoke_when_requested -- --ignored --nocapture
```

Set `SPEAKER_EMBEDDING_DIMENSION` when the local model output dimension is not
192. Set `SPEAKER_EMBEDDING_INPUT_NAME` and
`SPEAKER_EMBEDDING_OUTPUT_NAME` when ONNX IO names need to be selected
explicitly. The smoke expects an already-local 16 kHz WAV and never downloads
models.

2026-06-10 validation update: the local bundle
`wespeaker-voxceleb-resnet34-LM/main` was provisioned under the ignored smoke
model root using the current upstream `speaker-embedding.onnx` filename. The
smoke now prints static ONNX metadata and graph diagnostics before session
construction. The current model is feature-input f32 `feats` `[B,T,80]` to f32
`embs` `[B,256]`; the graph uses default-domain ONNX ops only, opset 14,
110 nodes, and 75 initializers. The native adapter auto-detects this shape and
builds deterministic CPU fbank/log-mel features from 16 kHz mono audio.

With `ORT_DYLIB_PATH` unset, both file and memory diagnostic load modes timed
out via external `timeout 120s` exit `124` after `onnxSessionBuilder=begin`
and before `onnxSessionBuilder=ok`. Python ONNX Runtime 1.26.0 loaded the same
artifact successfully. With `ORT_DYLIB_PATH` pointed at the local
`.audio-tools/whisperx-venv` `libonnxruntime.so.1.26.0`, the speaker smoke
passed for both file and memory load modes, printed `onnxSessionCommit=ok`,
`onnxSessionLoad=ok`, `speakerEmbeddingInputKind=feature`, and
`speakerFeatureBins=80`. Classification: implicit ORT dynamic-library
selection/setup blocker on this host, not a model graph or feature-shape
blocker.

## Package surface

Primary workflow: `audio.speakers.vad`.

Workflow operations:

- `audio.speakers.embed`: Computes a deterministic spectral speaker embedding from normalized samples.
- `audio.speakers.identify`: Builds a transient enrolled-speaker library and identifies a query embedding.
- `audio.speakers.assignTranscript`: Applies diarization segments to an existing transcription contract through Transcript Speaker Assignment.
- `audio.speakers.vad`: Detects speech-like regions with a deterministic RMS voice activity detector.
- `audio.speakers.diarize`: Runs deterministic VAD/window/spectral speaker diarization or normalizes imported diarization segments.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-audio-analysis-speakers-cli -- run \
  --operation audio.speakers.vad \
  --json '{"channels":1,"frameSize":2,"hopSize":1,"minSilenceSeconds":0.0,"minSpeechSeconds":0.0,"sampleRate":4,"samples":[0.0,0.2,0.2,0.0],"threshold":0.01}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Canonical library transfer

`SpeakerLibrary::export_new` writes a complete, validated canonical `library.json`
copy to a new destination. It preserves stable IDs, labels, metadata and embeddings,
never overwrites an existing path, and creates private files on Unix. Destination
folders must already exist. An empty portable library has `profiles: []` and
`embedding_model: null`; it does not invent a model. Existing populated JSON and
the populated `SpeakerLibrarySnapshot` Rust API remain compatible. The existing
`save` operation still removes an empty library; explicit export can materialize
one for a user's New/portable-library workflow.

`prepare_import` produces a non-mutating plan and a vector-free summary. Merge
retains identical identities, adds compatible new ones, and rejects conflicting
IDs or exact labels. Replace is an explicit whole-library decision. Both modes
reject incompatible model family/name/version/dimensions. Neither operation
changes draft/confirmed metadata or compares incompatible embedding vectors.

After the caller presents the summary and obtains its product-specific consent,
`plan.commit(active_path, new_backup_path)` verifies the active identities still
match the reviewed baseline. It preserves an existing document in a new backup
before atomically promoting the validated replacement. A stale preview or an
occupied backup path fails without replacing active content. An identical import
is a no-op: it creates no backup and does not rewrite the library.

All canonical writers coordinate through a persistent `.<filename>.lock` sidecar;
busy writers fail rather than wait indefinitely. The file is intentionally kept
across atomic renames. Speaker Directory selection, consent, backup naming,
cancellation before commit, and `speaker-trace.json` remain consumer-owned.
Transfer methods never upload or log embeddings, and plan debugging prints only
the safe count summary.

```rust,no_run
use audio_analysis_speakers::{SpeakerLibrary, SpeakerLibraryImportMode};
# fn example() -> Result<(), Box<dyn std::error::Error>> {
let incoming = SpeakerLibrary::load("portable-library.json")?;
let active = SpeakerLibrary::load("speakers/library.json")?;
let plan = active.prepare_import(&incoming, SpeakerLibraryImportMode::Merge)?;
let summary = plan.summary(); // counts only; the product presents its consent UI
# let _ = summary;
// Only after explicit confirmation, with a fresh, non-existing backup path:
let receipt = plan.commit("speakers/library.json", "speakers/library.before-import")?;
# let _ = receipt;
# Ok(())
# }
```
