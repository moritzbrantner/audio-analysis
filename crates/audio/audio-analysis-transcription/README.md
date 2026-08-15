# audio-analysis-transcription

Rust-native and compatibility transcription orchestration for `video-analysis`.

This crate is the higher-layer audio adapter for the transcript contracts owned
by `text-transcripts`. It owns audio decoding, VAD and ASR execution, external
WhisperX commands, whisper.cpp compatibility configuration, alignment and
speaker-model orchestration, and model resource setup; every provider returns a
`text_transcripts::TranscriptionContract` without creating a reverse NLP-to-audio
dependency.

Native Candle Whisper ASR is available behind the `candle` feature. CUDA device
selection is available behind `cuda`, and local model bundle validation is
available behind `model-bundles`. Native path decoding defaults to direct
samples or readable WAV files. With the explicit `audio-io` feature, non-WAV
paths use FFmpeg-backed `audio-analysis-io` decode and then normalize/resample
through the same native 16 kHz mono boundary. This opt-in media decode is not
WhisperX parity and is not part of default tests.

The compatibility whisper.cpp build, model store, and file adapter live behind
the `native` feature. That adapter preserves the existing model/download
behavior while returning the transcript document contract owned by
`text-transcripts`.

The `candle` feature also exposes `CandleQ8WhisperDecoder`, a low-level CPU Q8_0
decoder building block for caller-owned GGUF bundles. Its incremental operation
accepts token and encoder-feature tensors, an absolute token-position offset,
and a cache-reset signal. It returns decoder activations with diagnostics for
self-attention cache growth and cross-attention encoder projection reuse. Reset
the cache for each new audio window, prefill the prompt once, and then supply
only newly generated tokens at contiguous absolute offsets. This decoder
also provides a greedy active-row batch operation: completed rows are removed
from encoder features and decoder caches together while surviving rows keep
their cached positions. Ordered token IDs and aggregate active-batch,
compaction, cache-reuse, and generated-token diagnostics are returned. This
building block does not select bundles or change the existing full-precision
native transcription provider.

Reusable native VAD providers are opt-in and independent from diarization.
The `silero-vad` feature exposes `SileroVadOptions` and
`SileroVadTranscriptionProvider`; the `pyannote-vad` feature exposes
`PyannoteVadOptions` and `PyannoteVadTranscriptionProvider`. Both implement
`TranscriptionVadProvider`, require 16 kHz mono audio, and execute caller-
supplied local ONNX resources through `runtime-onnx`. Pyannote VAD validates
compatible local model metadata or a colocated manifest and never aliases
energy or Silero behavior. Provider IDs and diagnostics remain distinct so
pipeline consumers can observe `silero-vad`, `pyannote-vad`, window/frame
counts, and native completion.

Native Whisper tries model timestamp tokens automatically when the tokenizer
defines Whisper timestamp metadata. If timestamp decoding does not produce
bounded text segments, it falls back to chunk/window segment timing. Native
Whisper segment times are emitted as global transcript times. Timestamp-token
segments also include approximate projected word timings based on
character-weighted distribution inside each segment. wav2vec2/CTC alignment
can run through the same native transcription pipeline: `alignment.enabled=true`
with no bundle uses deterministic transcript timing alignment, while
`alignment.enabled=true` with a supported local wav2vec2 bundle uses Candle
wav2vec2 CTC alignment.

Native Candle Whisper also supports Whisper's built-in translate-to-English
task with `provider.task="translate"`. This is GPU-backed when built with
`cuda` and requested with `provider.device="cuda"`. Translation uses the
Whisper decoder task token; it is not OPUS-MT/Marian post-ASR machine
translation. Because translated text is not source-language text,
`alignment.enabled=true` is rejected for native translation requests.
Diarization remains allowed because speaker assignment can use segment timing.

Native runtime controls are request-scoped through the additive
`CandleWhisperRuntimeControls` API. Pass the controls to
`CandleWhisperTranscriber::transcribe_with_runtime_controls` or
`ReusableCandleWhisperTranscriber::transcribe_with_runtime_controls`.
`cudaDeviceIndex` selects exactly one non-negative CUDA device index and
defaults to `0`; explicit CPU requests reject nonzero indices instead of
silently ignoring them. `decoderThreads` accepts an optional positive CPU
decoder thread count. CPU decoding runs inside a request-local thread pool, so
different concurrent requests do not mutate process environment variables or
share thread limits. CUDA execution reports an explicitly requested decoder
thread count as ignored because the pool is CPU-specific. Existing
`AudioTranscriptionProvider::transcribe` calls use default controls, preserving
automatic device selection, CUDA index `0`, and the existing greedy decoder
behavior. Keeping these controls separate also preserves exhaustive downstream
`CandleWhisperOptions` struct literals.

The canonical native request surface is
`CandleWhisperTranscriptionRequestConfig`, accepted by
`transcribe_with_request_config` and
`transcribe_with_request_config_and_observer` on both the single-use and
reusable Candle transcribers. It aggregates runtime, decode, and window controls
without adding fields to the existing public option types. Legacy entrypoints
delegate through this interface with defaults.

`CandleWhisperWindowControls` selects automatic timestamp-token timing,
no-timestamp expanded-window timing, or required timestamp-token timing and
configures leading/trailing VAD context. Defaults preserve the previous
automatic timing, 250 ms leading context, and 40 ms trailing context. Context
values must be finite and non-negative.

### German no-align parity resources

The opt-in `german_no_align_fp32_matches_pinned_transformers_greedy_trace`
contract uses `openai/whisper-small` revision
`973afd24965f72e36ca33b3055d56a652f456b4d`, fp32 CPU, German,
no-timestamps timing, zero leading/trailing context, and greedy beam size one.
The input window is the exact `0.322–5.000s` Silero VAD span retained from
WhisperX 3.8.6. The contract retains the complete 80×3000 mel tensor and
complete filtered first-step vocabulary logits generated by Transformers 4.57.6
with PyTorch 2.8.0. The oracles are produced offline with `WhisperFeatureExtractor` and
`WhisperForConditionalGeneration`; the logit oracle applies the pinned
`suppress_tokens`, `begin_suppress_tokens`, no-timestamps token, and timestamp
range before serialization as little-endian f32.

The test fails closed unless these pinned resource hashes match:

- German five-second WAV:
  `80df13c0cf5733684be7ccc9a243ce45637debfc5186be575a1d0608ad929ca6`
- WhisperX 3.8.6 JSON:
  `1b38bea5200eea4c037eea0420ff71c73283c238f9820c978a60ca0c7612e095`
- Transformers mel oracle:
  `2e5702b857570249e43ab92804c28cabf8194f8f03c1fd48f97c3bfd9c091fbf`
- Transformers filtered-logit oracle:
  `751015116dcded3c45314938e471679439abd6ddc08ee5509bc16a66cd540412`
- Whisper-small safetensors:
  `1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b`

Run the cache-only contract with caller-owned local paths:

```bash
CANDLE_WHISPER_SMALL_BUNDLE=/path/to/973afd24965f72e36ca33b3055d56a652f456b4d \
CANDLE_WHISPER_GERMAN_WAV=/path/to/german-cache-probe.wav \
CANDLE_WHISPER_GERMAN_WHISPERX_ORACLE=/path/to/whisperx-3.8.6.json \
CANDLE_WHISPER_GERMAN_MEL_ORACLE=/path/to/transformers-mel-f32le.raw \
CANDLE_WHISPER_GERMAN_LOGITS_ORACLE=/path/to/transformers-filtered-logits-f32le.raw \
cargo test -p moenarch-audio-analysis-transcription --features candle \
  german_no_align_fp32_matches_pinned_transformers_greedy_trace \
  -- --ignored --nocapture
```

The corresponding external reference is WhisperX 3.8.6 with `--no_align`,
`--device cpu`, `--compute_type float32`, `--language de`, `--temperature 0`,
`--best_of 1`, and `--beam_size 1`. Model resolution remains cache-only and
tokens are supplied only through standard Hugging Face authentication.

Token selection is independently request-scoped through the additive
`CandleWhisperDecodeConfig` API and the `transcribe_with_decode_config` or
`transcribe_with_runtime_controls_and_decode_config` provider methods. The
default temperature schedule `[0.0]` uses the unchanged deterministic,
KV-cached greedy path. Positive temperatures use the caller's `seed` and
`bestOf` candidates, ranked by average log probability. An all-zero schedule
with `beamSize > 1` enables independent beam hypotheses; `patience` controls
how many completed hypotheses are collected and `lengthPenalty` applies
length-normalized ranking. Sampling and beam search are mutually exclusive,
and invalid widths, temperatures, or search-only combinations return
`DetectError::InvalidArgument`. Non-default search recomputes each independent
hypothesis instead of sharing branched decoder caches. Diagnostics report the
selected strategy and controls.

Complete request-scoped decoding is available through the additive
`CandleWhisperDecodeRequestConfig` API and the
`transcribe_with_decode_request_config` or
`transcribe_with_runtime_controls_and_decode_request_config` methods. It wraps
the search config while adding initial prompt token IDs, explicit suppressed
token IDs, tokenizer-aware numeral suppression, and optional previous-window
text conditioning. Previous text is kept only in the current request and is
bounded by the model prompt budget; it is never stored in a reusable session.
Previous-text conditioning uses the sequential autoregressive runtime because
later windows depend on earlier output.

Threshold fallback evaluates `temperatureSchedule` strictly in declaration
order. A high no-speech probability rejects the window when average log
probability is also below its configured minimum (or when no minimum is set),
so confident text is retained. A low average log probability or high zlib
compression ratio advances to
the next temperature, and the first passing attempt is returned (or the final
attempt when none pass). Candidates from different temperatures are never
ranked against one another. Response diagnostics expose
`averageLogProbability`, `noSpeechProbability`, `compressionRatio`, the ordered
`temperatureFallbackAttempts`, and whether no-speech rejection occurred. The
default request config still dispatches the exact pre-existing KV-cached greedy
path.

The external WhisperX command provider remains compatibility and parity tooling.
It keeps Python-based execution explicit for callers that still need WhisperX
decoding, batched ASR, alignment, or pyannote-backed diarization outside the
default Rust path. It is also the current path for video/container inputs.

Speaker diarization contracts are owned by `audio-analysis-speakers`.
Transcription owns pipeline orchestration and adapts the speaker-owned
`SpeakerDiarizationOptions` into its flattened `diarization` request JSON.
Native deterministic diarization execution is still available only behind
`diarization` as a heuristic spectral baseline, not a pyannote replacement or
production speaker recognition model. Transcript Speaker Assignment runs after
alignment when both options are enabled, so the native diarization seam can use
aligned word timings, or segment timings when word timings are absent, as
speech-span hints. When no transcript timing is available it falls back to the
energy-VAD baseline. `min_speakers` and `max_speakers` are validated by the
speaker-owned contract, reported in diagnostics, and applied as native
unknown-speaker clustering constraints. Known/enrolled speaker IDs are
preserved; bounds only affect generated unknown speaker labels.
An ONNX speaker embedding provider is available only when explicitly configured
with `diarization.speakerEmbeddingModelBundle` and the crate is built with
`diarization,onnx`. It feeds the existing `WindowedSpeakerDiarizer`; heuristic
diarization remains the default. ONNX diagnostics include
`diarizationRuntime=onnx`, `speakerEmbeddingProvider=onnx`,
`speakerEmbeddingDimension=N`, and `diarizationBaseline=false`. When speaker
bounds are requested, diagnostics include
`diarizationSpeakerBoundsApplied=true`; if the requested minimum cannot be
reached because too few usable speech spans exist, diagnostics include
`diarizationSpeakerBoundsSaturated=true`.

CTC alignment validates local wav2vec2 bundle files, config, tokenizer
vocabulary, and preprocessor metadata. Tokenizer discovery accepts
`tokenizer.json` and `vocab.json`; CTC blank resolution can use `pad_token_id`
when tokenizer metadata does not name a pad token. Supported local
`Wav2Vec2ForCTC` `model.safetensors` bundles execute through a private Candle
implementation and feed native CTC trellis/backtracking. Unsupported
architectures, stable-layer-norm configs, inconsistent positional-convolution
weight-norm tensors, or other unsupported safetensors layouts return typed
errors instead of falling back to deterministic timing. The Debug-only
`audio.transcription.alignmentBundlePlan` operation inspects local wav2vec2
bundle layout metadata without model inference and reports architecture,
stable-layer-norm status, positional-convolution layout, feature-extractor
norm, encoder layer count, missing tensor keys, and unsupported reasons.

Candle Whisper batch options are deterministic rather than concurrent in this
phase. `max_batch_size=0` is rejected, chunk order and global timing are
preserved, and diagnostics report `chunkCount`, `batchChunks`, `maxBatchSize`,
`batchCount`, and `batchExecution=candle-whisper-sequential`. This is semantic
batch grouping, not throughput parity or tensor-batched model execution.

## Choosing a transcription entry point

Use `transcribe(request)` for one-off selected-provider execution. It is the
compatibility entry point for callers that have a complete
`TranscriptionPipelineRequest` and do not need to keep native ASR provider state
alive after the call returns. It preserves the selected provider behavior,
including external WhisperX command execution when explicitly requested.

Use `run_transcription_pipeline` or
`run_transcription_pipeline_with_observer(...)` when advanced callers need to
provide their own `TranscriptionVadProvider`, `AudioTranscriptionProvider`,
`ForcedAlignmentProvider`, or `TranscriptDiarizationProvider` adapters directly.
This is the provider-agnostic primitive seam for tests, experiments, and package
surfaces that already own custom provider construction. The observer variant
emits phase-level `TranscriptionPipelineObserver` events for validation, decode,
VAD, ASR, alignment, diarization, model resolution/download, and model
load/reuse activity. Observer methods for resolution, download, and cooperative
cancellation have default no-op behavior so existing observers remain source
compatible. A cancellation request stops execution at the next safe pipeline
or model-resolution boundary.

Use `NativeTranscriptionRunner` with `NativeTranscriptionRunnerOptions` for
repeated native finite transcription requests. The runner owns the native
provider stack across compatible requests and uses
`ReusableCandleWhisperTranscriber` as the public Candle Whisper provider-reuse
primitive. This keeps compatible native model session state behind the existing
provider traits instead of requiring callers to reach into private session
internals. Reuse remains observable through public surfaces: compatible repeated
requests emit `TranscriptionPipelineEvent::ModelReuse` through the observer path
and response diagnostics include `asrModelSession=loaded` or
`asrModelSession=reused`.

This crate owns reusable primitive transcription execution, provider traits,
pipeline request/response contracts, native runner reuse, and phase-level
observer events. `native-whisperx` owns workflow composition above these
primitives: output writing, WhisperX/Rust-native parity decisions, automatic
workflow selection, Speaker Directory effects, and Transcription Progress Stream
formatting. Runner progress is therefore the raw
`TranscriptionPipelineObserver` phase event stream; it is not the formatted
native-whisperx progress stream that workflow consumers see.

Transcript contracts, normalization, caption formatting, and WhisperX JSON
import remain owned by `text-transcripts`.

## Package Operations

- `audio.transcription.transcribe`: run real transcription through the selected
  provider. Native Candle Whisper uses local WAV input or direct samples and can
  run `task="translate"` for Whisper translate-to-English; the external
  WhisperX provider remains available for compatibility.
- `audio.transcription.importWhisperX`: import existing WhisperX JSON without
  running external tools.
- `audio.transcription.providers`: inspect available provider families.
- `audio.transcription.plan`: describe runtime setup without execution.
- `audio.transcription.modelPlan`: inspect ASR model requirements.
- `audio.transcription.vadPlan`: inspect deterministic VAD defaults.
- `audio.transcription.alignmentPlan`: inspect CTC alignment requirements.
- `audio.transcription.alignmentBundlePlan`: inspect local wav2vec2 bundle
  readiness without model inference.
- `audio.transcription.decodePlan`: explain source decode routing without
  opening files or running FFmpeg.
- `audio.transcription.diarizationPlan`: explain heuristic diarization status,
  speaker-owned assignment policies, and future model-backed provider
  directions.
- `describe`: inspect package metadata.

## Setup

For native Candle Whisper execution, provide a local model bundle containing:

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `preprocessor_config.json`
- `model.safetensors`

`computeType="int8"` is CPU-only and requires an explicit local Q8_0 bundle
with the same four JSON companion files plus `model.q8_0.gguf` instead of
`model.safetensors`. Int8 never downloads or falls back to safetensors. Bundle
resolution validates GGUF architecture/file-type metadata, Q8_0 tensor types,
and config/tokenizer dimensions before ASR execution.

For native wav2vec2 CTC alignment, provide a local `Wav2Vec2ForCTC` bundle
containing:

- `config.json`
- `tokenizer.json` or `vocab.json`
- `preprocessor_config.json`
- `model.safetensors`

The real wav2vec2 smoke prints the alignment bundle layout report before
execution. Stable-layer-norm bundles remain `unsupported_runtime` until that
architecture path is implemented.

2026-06-10 validation update: `scripts/sync_model_bundles.sh` provisioned
`facebook/wav2vec2-base-960h` under the ignored smoke model root. The ignored
real-bundle alignment smoke passed with `vocab.json` tokenizer discovery and
reported `architecture="Wav2Vec2ForCTC"`,
`do_stable_layer_norm=false`, `positional_conv_layout="weight-norm"`,
`feature_extractor_norm="group"`, `encoder_layer_count=12`, no missing keys,
and no unsupported layout reasons. Native positional convolution reconstruction
now supports the observed per-kernel weight-norm `weight_g` layout used by this
bundle.

Use `candle,model-bundles` for CPU local smoke tests and add `cuda` for
CUDA-backed Whisper smoke tests. No runtime downloads are performed by this
crate.

Use `audio-io` only when native non-WAV media/container decode is explicitly
needed and local FFmpeg support is available:

```rust,ignore
use audio_analysis_transcription::{transcribe_selected_media, TranscriptionPipelineRequest};

let request: TranscriptionPipelineRequest = build_request_with_path_source();
let response = transcribe_selected_media(request, Some(1))?;
# let _ = response;
```

`transcribe_selected_media` keeps the existing `TranscriptionSource::Path`
request shape and accepts an optional zero-based audio-stream ordinal. It
validates and predecodes through `audio-analysis-io` before constructing native
providers or emitting pipeline progress. `None` preserves the first/default
stream. Invalid selections return `SelectedMediaTranscriptionError::Decode`
with the typed FFmpeg stream inventory. `NativeTranscriptionRunner::run_selected_media`
provides the same ordering for reusable runners.

```bash
RUN_NATIVE_MEDIA_DECODE_TESTS=1 \
TRANSCRIPTION_MEDIA_PATH=/path/to/video-or-audio-container \
cargo test -p moritzbrantner-audio-analysis-transcription \
  --features audio-io \
  native_media_decode_when_requested -- --ignored --nocapture
```

Use `diarization,onnx` only when caller-owned local ONNX speaker embeddings are
explicitly configured. Add `model-bundles` when the path is a model-runtime
bundle manifest:

```json
{
  "diarization": {
    "enabled": true,
    "speakerEmbeddingModelBundle": "/path/to/onnx-speaker-model",
    "speakerEmbeddingDimension": 192,
    "speakerEmbeddingSampleRate": 16000
  }
}
```

The ONNX path accepts a direct `.onnx` file, a directory containing
`model.onnx`, or a model-runtime manifest when `model-bundles` is enabled.
It does not resample in this tranche; request audio must already match the
configured speaker embedding sample rate. Speaker model input shape is detected
from ONNX metadata: waveform inputs use `[B,S]` or `[B,1,S]`, while feature
inputs such as `[B,T,80]` use the shared speaker fbank/log-mel CPU
preprocessor.

Ignored local transcription ONNX diarization smoke:

```bash
RUN_NATIVE_SPEAKER_MODEL_TESTS=1 \
ORT_DYLIB_PATH="$PWD/.audio-tools/whisperx-venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so.1.26.0" \
SPEAKER_EMBEDDING_MODEL_BUNDLE=/path/to/onnx-speaker-model \
SPEAKER_EMBEDDING_MODEL_FILE=model.onnx \
DIARIZATION_AUDIO_PATH=/path/to/meeting-16khz.wav \
SPEAKER_EMBEDDING_DIMENSION=192 \
cargo test -p moritzbrantner-audio-analysis-transcription \
  --features diarization,onnx,model-bundles \
  native_onnx_diarization_smoke_when_requested -- --ignored --nocapture
```

The smoke uses direct samples and mock ASR timing, so it validates only the
native transcription pipeline's explicit ONNX diarization path. It does not use
FFmpeg, Python, CUDA, pyannote, Hugging Face auth, network, or downloaded model
files.

2026-06-10 validation: the checked-in WebM fixture decoded successfully with
`audio-io` after the ignored smoke harness resolved workspace-root-relative
fixture paths. After local ONNX bundle provisioning, the ONNX diarization smoke
used direct-sample setup and mock ASR timing. Static ONNX metadata inspection
succeeded for the selected current `wespeaker-voxceleb-resnet34-LM` artifact:
f32 input `feats` `[B,T,80]` and f32 output `embs` `[B,256]`. Static graph
diagnostics reported default-domain ONNX ops only, opset 14, 110 nodes, and
75 initializers. The speaker adapter supports this feature-input category.

With `ORT_DYLIB_PATH` unset, file and memory diagnostic load modes timed out
via external `timeout 120s` exit `124` after `onnxSessionBuilder=begin` and
before `onnxSessionBuilder=ok`. Python ONNX Runtime 1.26.0 loaded the same
artifact. With `ORT_DYLIB_PATH` pointed at the local `.audio-tools/whisperx-venv`
`libonnxruntime.so.1.26.0`, the transcription ONNX diarization smoke passed:
direct samples and mock ASR timing reached `diarizationRuntime=onnx`,
`speakerEmbeddingProvider=onnx`, `speakerEmbeddingDimension=256`,
`diarizationBaseline=false`, and non-empty speaker assignments. ONNX
diarization remains window-by-window embedding, not throughput parity.
Classification: implicit ORT dynamic-library selection/setup blocker on this
host, fixed for local smokes by an explicit compatible dylib.

On the current RTX 3060 Ti smoke host, `/usr/local/cuda` points at CUDA 13.3
while the passing smoke uses a local CUDA 12.3 library shim at
`$SMOKE_ROOT/cuda12-libs`. The local smoke
bundle and fixture used there are:

- `$SMOKE_ROOT/whisper-tiny`
- `$SMOKE_ROOT/audio/native-transcription-smoke.wav`

Install and configure external compatibility tools outside the default test
flow only when using the WhisperX provider:

```bash
whisperx --help
ffmpeg -version
python -c 'import whisperx'
```

WhisperX diarization requires a Hugging Face token accepted by pyannote:

```bash
export HF_TOKEN=...
```

No default build or test downloads models, requires CUDA, or requires network
access. Default tests also do not require Python, WhisperX, Hugging Face tokens,
external model files, or FFmpeg.

The opt-in native tiny-model decode smoke exercises greedy, seeded sampling,
and beam search against one local bundle:

```bash
RUN_CANDLE_WHISPER_DECODE_TESTS=1 \
CANDLE_WHISPER_TINY_BUNDLE="$SMOKE_ROOT/whisper-tiny" \
cargo test -p moenarch-audio-analysis-transcription \
  --features candle \
  real_tiny_whisper_bundle_runs_greedy_sampling_and_beam_paths_when_requested \
  -- --nocapture
```

The ignored Q8 provider smoke uses caller-owned local resources:

```bash
CANDLE_WHISPER_Q8_BUNDLE=/path/to/whisper-tiny-q8 \
CANDLE_WHISPER_Q8_WAV=/path/to/short-16khz-fixture.wav \
cargo test -p moenarch-audio-analysis-transcription \
  --features candle \
  real_q8_whisper_bundle_transcribes_short_fixture_with_valid_contract \
  -- --ignored --nocapture
```

Optional external WhisperX parity can be run manually when local tools and media
are configured:

```bash
RUN_WHISPERX_PARITY_TESTS=1 \
WHISPERX_COMMAND=whisperx \
WHISPERX_AUDIO_PATH=/path/to/audio.wav \
cargo test -p moritzbrantner-audio-analysis-transcription external_whisperx_parity_when_requested -- --ignored --nocapture
```

Set `WHISPERX_EXPECTED_JSON=/path/to/expected.json` to compare command output
against a known WhisperX JSON fixture. Optional parity-only overrides are
`WHISPERX_MODEL`, `WHISPERX_LANGUAGE`, `WHISPERX_DEVICE`, and
`WHISPERX_COMPUTE_TYPE`. Set `WHISPERX_DIARIZE=1` only when `HF_TOKEN` is
available.

2026-06-10 validation update: the broken global console entry point was bypassed
with an ignored local venv at `.audio-tools/whisperx-venv`. Its
`bin/whisperx --help` command worked, and non-diarization parity passed with
`WHISPERX_MODEL=tiny.en`, `WHISPERX_LANGUAGE=en`, `WHISPERX_DEVICE=cpu`, and
`WHISPERX_COMPUTE_TYPE=int8`. Token-gated diarization parity was not run because
`HF_TOKEN` was absent, so pyannote-backed diarization parity remains incomplete.

Token-gated continuation command shape:

```bash
test -n "$HF_TOKEN"

RUN_WHISPERX_PARITY_TESTS=1 \
WHISPERX_COMMAND="$PWD/.audio-tools/whisperx-venv/bin/whisperx" \
WHISPERX_MODEL="tiny.en" \
WHISPERX_LANGUAGE="en" \
WHISPERX_DEVICE="cpu" \
WHISPERX_COMPUTE_TYPE="int8" \
WHISPERX_DIARIZE=1 \
WHISPERX_AUDIO_PATH="$SMOKE_ROOT/audio/native-transcription-smoke.wav" \
HF_TOKEN="$HF_TOKEN" \
cargo test --test audio_transcription_native_contracts \
  external_whisperx_parity_when_requested -- --ignored --nocapture
```

2026-06-10 token-gated result: setup/auth failure. WhisperX reached pyannote
diarization, but Hugging Face rejected access to
`pyannote/speaker-diarization-community-1`. Initial non-secret error prefix:
`huggingface_hub.errors.GatedRepoError: 403 Client Error`; after refreshing the
token, the rerun failed with
`huggingface_hub.errors.HfHubHTTPError: 403 Forbidden` until the fine-grained
token allowed public gated repositories and access to the pyannote model.

2026-06-11 token-gated result: pass. WhisperX diarization parity completed with
`WHISPERX_DIARIZE=1`, `.audio-tools/whisperx-venv/bin/whisperx`, `tiny.en`,
CPU, int8, and `HF_TOKEN="$HF_TOKEN"`. Pyannote/Hugging Face access was valid
for this run. No Rust parser or contract bug was exposed, and no token value
was documented.
