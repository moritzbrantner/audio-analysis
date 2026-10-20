# Explicit browser transcription language

The existing Transformers.js model boundary now receives a per-request optional
language code for both finite PCM and bounded streaming transcription. The same
resident model can handle French, Japanese, automatic detection and Spanish
streaming in sequence without retaining a previous request's language or loading
the model again. Explicit language is retained in result/segment metadata;
automatic results remain null because detected-language metadata is unavailable.

The public catalog contains the 99 codes shared by the Tiny/Base/Small multilingual
models. Their actual generation configs were compared, rather than treating all
Whisper model families as interchangeable:

- [Tiny generation config](https://huggingface.co/onnx-community/whisper-tiny/blob/main/generation_config.json)
- [Base generation config](https://huggingface.co/onnx-community/whisper-base/blob/main/generation_config.json)
- [Small generation config](https://huggingface.co/onnx-community/whisper-small/blob/main/generation_config.json)

The public API integration test first observed missing language arguments for both
explicit requests. It passes after the change, including automatic omission,
streaming propagation, unsupported-code rejection before inference, unchanged
model-load count and preserved explicit result language. This is a faithful
provider-boundary fixture, not a physical WebGPU/model-accuracy claim.

All 20 browser package tests, declaration checking, extraction structure and
repository source readiness/handoff pass. Shared convention source:
`6cc05789dab39cee6758b288f83fb48b4ee6dbdb`.

Two old resampler assertions also failed against unchanged baseline
`b8d9d639c282853c4e0fbed02adcb30fd544f038`: they expected unfiltered ramp samples
and instant DC response despite the existing causal FIR antialiasing filter.
The tests now compare chunked results to a contiguous run (including stereo
mixing and noninteger phase continuity), and check DC after warmup. The existing
out-of-band rejection check remains. No resampler production code changed.
