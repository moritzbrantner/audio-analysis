# @moritzbrantner/audio-analysis-speakers-wasm

Browser/WASM package for `audio-analysis-speakers`.

The package keeps the generic WASM surface and also exposes a browser-local
deterministic diarization baseline for static consumers that cannot ship the
native pyannote/ONNX runtime. The browser adapter decodes to 16 kHz mono locally,
clusters speech windows using spectral features, emits anonymous
`speaker_N` segments, and can assign those speakers to timed transcript
segments. It never routes audio to a server or Python runtime.

The deterministic browser baseline is intended for preview/prototyping and is
not a claim of pyannote parity.

```bash
bun run --cwd packages/audio-analysis-speakers-wasm build
bun test packages/audio-analysis-speakers-wasm/tests/package.test.ts
```
