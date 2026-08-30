# ADR 0013: Consolidate Fourier and recognition behind stable audio interfaces

## Status

Accepted as a compatibility plan. This decision does not move implementation,
change a package version, deprecate a published artifact, or authorize a
release or package-surface removal.

## Context

`moenarch-audio-analysis-fourier` and
`moenarch-audio-analysis-recognition` are independently versioned even though
their source consumers do not require those package seams. Fourier is shared
in-process implementation for pitch, rhythm, and recognition. Recognition is a
mixture of cross-capability runtime selection, a speaker spectral baseline,
generic recognition helpers, deprecated transcription compatibility, and
transport operations.

The stable independently addressable audio libraries remain:

- `moenarch-audio-analysis-core` for shared audio primitives;
- `moenarch-audio-analysis-io` for source and external audio I/O;
- `moenarch-audio-analysis-speakers` for speaker-domain behavior; and
- `moenarch-audio-analysis-transcription` for transcription execution.

Pitch, rhythm, separation, and synthesis also remain existing focused
libraries. Consolidation must deepen those existing modules; it must not add a
facade crate or another CLI, server, WASM, or app package.

## Public symbol inventory and destination

This inventory covers every public module, root item, re-export, and associated
item at the issue baseline `785d858617bbb64d1fe683f4934d7da110ecc032`.
Public fields, enum variants, standard trait implementations, and a type's
public associated methods travel with the named type.

### Fourier

| Current public symbols | Public associated items | Smallest stable owner |
| --- | --- | --- |
| `SpectrumBin`, `Spectrum`, `SpectralFeatures` | `Spectrum::{dominant_frequency_hz, features}` | `audio-analysis-core::spectral` |
| `SpectralFeatureOptions`, `SpectralFeatureFrame` | `SpectralFeatureOptions::{new, mel_band_count, validate}` | `audio-analysis-core::spectral` |
| `FourierTransform` | `new`, `with_window`, `analyze_samples`, `analyze_frame`, `Default` | `audio-analysis-core::spectral` |
| `StftConfig`, `SpectrogramFrame` | `StftConfig::{new, window, pad_final_frame}` | `audio-analysis-core::spectral` |
| `spectrogram`, `zero_crossing_rate`, `spectral_flux`, `spectral_features`, `spectral_feature_frames` | n/a | `audio-analysis-core::spectral` |
| `SpectralAnalyzer` | `new`, `min_magnitude`, `Default`, and its `AudioAnalyzer` implementation | `audio-analysis-core::spectral` |
| `surface` module, `surface::package_surface`, `surface::run_surface_operation` | n/a | Compatibility-only. Keep in the Fourier package and its existing adapters until adapter migration evidence supports retirement; do not make package metadata part of core's interface. |

All computational Fourier symbols have one owner because pitch, rhythm, and
speaker recognition use the same pure in-process transform and feature
implementation. Keeping separate copies in pitch and rhythm would spread the
same implementation across callers. The package-specific runtime surface does
not pass that deletion test: its only purpose is compatibility for the
existing Fourier adapters.

### Recognition

| Current public symbols | Public associated items | Smallest stable owner |
| --- | --- | --- |
| `AudioRuntime`, `FallbackPolicy`, `AudioRuntimeSelection`, deprecated `AudioModelSelection` | enum variants, public fields, and existing serde/default behavior | `audio-analysis-core::runtime`; speakers, separation, and synthesis re-export the selection types used in their public requests so callers need not learn the internal owner |
| `SpectralEmbeddingConfig` | `new`, `window`, `Default` | `audio-analysis-speakers`; it configures the existing `SpectralSpeakerEmbedder` fallback |
| `SpectralAudioEmbedder` | `new`, `streaming_config`, `Default`, and its `AudioEmbeddingExtractor` implementation | Speaker-private implementation behind `SpectralSpeakerEmbedder`; `SpectralSpeakerEmbedder::inner` is compatibility-only and receives a deprecation before removal |
| `AudioEmbedding`, `AudioEmbeddingExtractor` | `AudioEmbedding::{new, values, dimensions, cosine_similarity}` and `AudioEmbeddingExtractor::{embed_samples, embed_frame}` | No new public owner. Speaker callers already have the deeper `SpeakerEmbedding` and `SpeakerEmbeddingExtractor` interfaces; retain and deprecate these generic recognition paths in place |
| `AudioReference`, `AudioReferenceLibrary`, `AudioReferenceLibrarySnapshot`, `AudioMatchOptions`, `AudioRecognitionMatch`, `AudioSimilarity`, `AudioRecognitionAnalyzer`, `compare_audio_samples` | all constructors, builders, accessors, snapshot/JSON/search methods, analyzer accessors/reset/spectral constructor, and `AudioAnalyzer` implementation | No new public owner. No independent source consumer uses this generic reference-search interface; retain and deprecate it in recognition rather than widen core speculatively |
| `AudioFeatureSummary`, `AudioFeatureFrame`, `ImportedAudioPrediction`, `AudioClassPrediction`, `AudioClassificationRequest`, `AudioClassificationResponse`, `AudioEventDetectionRequest`, `AudioEventPrediction`, `AudioEventDetectionResponse`, `AudioEmbeddingRequest`, `AudioEmbeddingResponse` | public fields and existing serde/default behavior | No new public owner. These are recognition package-surface DTOs with no independent source consumer |
| `classify_audio`, `detect_audio_events`, `embed_audio` | n/a | No new public owner. Keep with the DTO compatibility surface until deprecation and adapter evidence permit removal |
| `FingerprintRecord`, `FingerprintMatchedSegment`, `FingerprintCandidate`, `run_fpcalc`, `run_default_fpcalc`, `parse_fpcalc_json`, `parse_fpcalc_record`, `duration_prefilter`, `shifted_similarity`, `fingerprint_similarity_with_offset`, `rank_fingerprint_candidates` | public fields | No new public owner. There is no source consumer to justify adding this command-backed interface to audio I/O or its matching implementation to core |
| deprecated `SpeechRecognitionRequest`, deprecated `SpeechRecognitionResponse`, deprecated `transcribe_audio`, `speech_recognition_response_from_transcription` | `SpeechRecognitionResponse::{text, segments}` | `audio-analysis-transcription`; preserve old recognition paths only as deprecated compatibility shims |
| re-exported `TranscriptSegmentContract`, `TranscriptionContract` | their media-core interfaces | `moenarch-media-core`; transcription consumes those contracts without taking ownership of timed text |
| `transcription::{TranscriptionInput, TranscriptionRuntimeSelection, TranscriptionRequest, TranscriptionResponse, TranscriptionProviderKind, TranscriptionBackendPlan, WhisperCppTranscriptionPlan, AudioTranscriptionProvider, ImportedTranscriptionProvider}` and the same root re-exports | `WhisperCppTranscriptionPlan::new`, provider trait methods, and conversion implementations | `audio-analysis-transcription`; the already-deprecated recognition paths remain until consumers use transcription directly |
| `transcription::{transcribe, transcription_plan, audio_runtime_to_model_backend}` and root re-exports of the first two | n/a | `audio-analysis-transcription` |
| `surface` module, `surface::package_surface`, `surface::run_surface_operation` | n/a | Compatibility-only. Keep in recognition and its existing adapters until the adapter/removal gate is met |

The recognition crate declares 31 root types or traits, 14 root functions, two
public modules, two media-contract re-exports, eleven transcription root
re-exports, one additional public item under `transcription`, and two public
surface functions. The grouped rows above account for all of them without
promoting unused compatibility interfaces into core.

## Consumer evidence

The dependency and organization searches were repeated on 2026-08-30 against
the issue baseline. Searches covered both Cargo package names and Rust import
names for Fourier and recognition.

- Inside this repository, Fourier has only pitch, rhythm, recognition, and its
  existing CLI/server/WASM adapters as direct manifest or source consumers.
- Inside this repository, recognition has only speakers, separation,
  synthesis, and its existing CLI/server/WASM adapters as direct manifest or
  source consumers.
- The `moritzbrantner/rust-packages` hits are compatibility copies retained by
  the repository extraction, not an independent product implementation.
- Native WhisperX contains Fourier and recognition only transitively in
  `Cargo.lock`. Its manifest directly addresses audio I/O, speakers, and
  transcription, and its source-development declaration patches those same
  three audio packages.
- `moritzbrantner/media-similarity` directly addresses audio core, rhythm,
  speakers, and transcription. Its source-development declaration patches
  those stable libraries, not Fourier or recognition.
- The remaining organization hits are workspace manifests, locks, extraction
  metadata, focused compatibility wrappers, and implementation imports within
  this repository. No independent source manifest or product source imports
  either package.

Consequently no known downstream consumer loses a required independently
versioned Fourier or recognition seam. A later implementation must repeat the
same package-name and Rust-import searches; this evidence is not a permanent
claim about future consumers.

## Compatibility sequence

Consolidation proceeds additively and source-first in separately reviewed
changes:

1. Add the `audio_analysis_core::spectral` and
   `audio_analysis_core::runtime` interfaces with behavior-equivalence tests.
   Add the speaker-owned spectral fallback interface. Do not change existing
   Fourier or recognition paths.
2. Change pitch and rhythm to the core spectral interface. Change speakers,
   separation, and synthesis to the core runtime interface. Move the speaker
   spectral implementation behind `SpectralSpeakerEmbedder`. Change
   transcription compatibility to use the transcription library and
   media-core contracts directly. This ordering removes the
   speakers-to-recognition dependency before any recognition-to-speakers
   compatibility re-export could be introduced, so no dependency cycle is
   created.
3. Make the Fourier crate re-export the core spectral symbols under their exact
   existing names. Make recognition re-export the core runtime and
   speaker-owned configuration symbols under their exact existing names.
   Preserve existing serde shapes, error behavior, trait implementations, and
   operation IDs. Add compile tests for the old imports and behavior tests that
   compare old and new paths.
4. Only after the additive paths are source-proven, prepare a dedicated
   compatibility release that marks the old Fourier and recognition exports as
   deprecated and points each one to the destination above or explicitly says
   that it has no replacement. Existing published versions and artifacts are
   immutable; this plan does not authorize a version bump, publication, yank,
   or tag.
5. Migrate downstream source declarations and imports, then collect the source
   evidence below. Deprecation must precede removal; a warning-free source
   migration must precede any removal proposal.

The existing Fourier and recognition CLI, server, WASM, and app packages remain
compatibility adapters throughout these steps. No new adapter package is
created. Their operation IDs are not silently reassigned to core or speakers.

## Required source-mode evidence

Before any release or removal task, a candidate consolidation revision must
record all of the following with exact Git SHAs:

1. In this repository, activate the declared exact foundation checkout with
   `bash scripts/source-deps activate`, run `scripts/check-fast.sh`, and
   deactivate source mode. Generated `.cargo/config.toml` must not be
   committed.
2. In Native WhisperX, temporarily point its declared audio source packages to
   the exact candidate audio revision, activate its declared source mode, and
   run its focused library checks. The resolved graph must contain direct audio
   I/O, speakers, and transcription edges and no direct Fourier or recognition
   edge.
3. In media-similarity, temporarily point its declared audio packages to the
   same exact candidate revision, activate its full declared source workspace,
   and run the focused audio/media checks. The resolved graph must keep core,
   rhythm, speakers, and transcription direct while keeping Fourier and
   recognition transitive or absent.
4. Save command output and resolved source revisions as review evidence. A
   registry-only check is later release evidence and is not a substitute for
   these source-mode checks.

## Removal gate

Removing either library package or any focused adapter requires a new,
explicitly authorized compatibility/release change. That review must include:

- green source-mode evidence for this repository, Native WhisperX, and
  media-similarity at the exact candidate revisions;
- a repeated organization usage search showing no remaining direct source
  consumer;
- warning-free downstream migration evidence after a published deprecation
  path exists;
- explicit adapter usage and deployment evidence for every CLI, server, WASM,
  or app surface proposed for removal;
- an immutable release plan for any changed package versions; and
- a rollback that restores source compatibility without deleting or mutating a
  published artifact.

Until every gate is satisfied, Fourier and recognition remain workspace
members and publishable compatibility surfaces. This ADR authorizes no
publication, integration, source removal, or adapter removal.
