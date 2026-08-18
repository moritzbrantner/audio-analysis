# audio-generation-tts

`audio-generation-tts` owns the first stable package-consumer contract for
generic text-to-speech and speaker-conditioned TTS.

The primary workflow operation is `audio.tts.synthesize`. Default builds keep
the operation side-effect free and return explicit setup/unsupported-runtime
diagnostics. When the crate is built with `candle` and the request explicitly
selects native F5 plus local F5 and Vocos bundles, the same primary operation
runs F5 mel generation through Vocos vocoding and returns in-memory PCM audio.
The F5 mel and Vocos vocoder diagnostics remain opt-in debug operations for
setup inspection.

Package-surface operations:

- `audio.tts.synthesize` validates a synthesis request and returns either a
  side-effect-free setup response or explicit native F5 + Vocos PCM output.
- `audio.tts.plan` previews provider, runtime, and output requirements.
- `audio.tts.models` reports the current model inventory state.
- `audio.tts.referencePromptPlan` inspects Reference Voice Prompt readiness.
- `audio.tts.debug.f5Mel` validates a local F5 bundle and returns a mel-level
  diagnostic without running a vocoder.
- `audio.tts.debug.vocosVocoder` validates a local Vocos bundle and converts a
  constrained generated or inline mel input into PCM audio.
- `describe` returns package and operation metadata.

Native planning is side-effect free. `audio.tts.plan` can explain an explicit
`provider.modelId`, optional `provider.modelBundle.bundlePath`,
`provider.modelBundle.autoDownload`, and `provider.modelBundle.cacheOnly`
without resolving files, downloading models, probing hardware, or running
inference. Cache-only mode forbids downloads even when `autoDownload` is set.
Model download planning is only allowed when the crate is built with the
explicit `model-bundles` feature; default builds report the requirement instead
of materializing files.

Reference Voice Prompts accept either inline PCM samples or a caller-managed
path source. A prompt transcript must be non-empty when provided.
Speaker-conditioned synthesis requires either a transcript or an explicit
`referenceVoicePrompt.asrFallback` setup; default builds report missing ASR
fallback support as setup-required diagnostics. Building with `asr` enables
fallback planning through `audio-analysis-transcription` provider metadata, but
planning remains side-effect free and does not run ASR.

Device planning uses `provider.device`:

- `auto` is the default and is CUDA-preferred when the crate is built with
  `cuda` and a CUDA device is available; otherwise native providers should use
  CPU.
- `cpu` requests CPU execution.
- `cuda` requests CUDA execution and requires the `cuda` feature plus an
  available CUDA device in later native providers.

Cargo features are explicit and default to off: `candle`, `cuda`,
`model-bundles`, `audio-io`, `asr`, and `external-tests`.

Native synthesis uses `provider.providerId = "f5"`, `provider.native = true`,
`provider.modelBundle.bundlePath` for the F5 bundle, and
`provider.vocoder.modelBundle.bundlePath` for the Vocos bundle. Responses
include `nativeDiagnostics` with provider, model id, vocoder, runtime, device,
bundle-source, resolved inference-control fields, and target text chunking
metadata. The native path accepts `options.seed`, `options.steps`,
`options.cfgStrength`, `options.speed`, `options.maxDurationSeconds`, and
`options.removeSilence`; debug diagnostics report the accepted controls and the
constrained diagnostic audio path applies them to generated mel/audio shape and
silence trimming.

Long target text is planned with deterministic sentence/whitespace chunking.
`audio.tts.plan` reports the chunk count and character spans under
`textChunking`, and native diagnostics report the same metadata. Chunked native
synthesis still returns the core in-memory output contract as one
`PcmAudio` value with concatenated PCM samples; it does not expose chunked audio
buffers or require callers to stitch audio manually.

Supported model-family metadata:

- F5: `f5-tts-v1-base` and `f5-tts-base` are listed as explicit opt-in
  speaker-conditioned TTS presets. Native synthesis is wired for F5 plus Vocos
  when the crate is built with `candle` and callers provide local bundles.
- E2: `e2-tts-base` is listed as an explicit opt-in preset for package planning
  and parity documentation. Native E2 inference is not wired in this crate yet.
- Vocos: `vocos-mel-24khz` is the native vocoder preset used by the F5 path.

Reference parity fixture metadata lives at
`tests/fixtures/f5-e2-tts-parity-fixtures.json`. The fixture file documents the
upstream Python reference families, a short F5 reference-transcript scenario, a
long-text F5 chunking scenario, and the current E2 metadata-only support gap.
It intentionally avoids checked-in model weights and audio artifacts so default
tests remain model-free, network-free, and side-effect free.

Quality and parity limitations:

- Default tests use synthetic local bundle fixtures and constrained diagnostic
  tensors. They verify contracts, setup diagnostics, chunking behavior, and PCM
  shape, not acoustic quality.
- Chunk boundaries prefer sentence and whitespace boundaries, but they are not a
  byte-for-byte copy of upstream Python preprocessing.
- Prosody continuity, reference-speaker similarity, tokenizer parity, duration
  prediction, and full F5/E2 sampling parity remain native parity risks.
- E2 is documented through model metadata and fixture coverage only; requests
  for native E2 synthesis still report unsupported runtime behavior.

`audio.tts.debug.f5Mel` requires an explicit local `bundlePath`. With `candle`
enabled it validates compatible F5 config, vocab, and safetensors files, then
allocates a constrained mel diagnostic tensor. It never downloads models,
emits PCM audio, or invokes Vocos.

`audio.tts.debug.vocosVocoder` requires an explicit local `bundlePath`. It
validates `config.yaml` and `pytorch_model.bin` from the Vocos bundle before
audio generation. With `candle` enabled it converts a constrained mel diagnostic
input into a mono `OwnedAudioFrame` and returns a JSON PCM summary; it does not
download models or run full F5/E2 synthesis.

CUDA smoke coverage is opt-in and ignored by default. Run it only on a
CUDA-capable host with local `F5_TTS_BUNDLE` and `VOCOS_BUNDLE` paths:
`cargo test -p moritzbrantner-audio-generation-tts --features candle,cuda,external-tests native_f5_vocos_cuda_preferred_synthesis_smoke_when_requested -- --ignored`.
