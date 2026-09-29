# @moritzbrantner/audio-analysis-io-wasm

Browser/WASM package for `audio-analysis-io`.

The browser adapter owns finite local audio decoding for composed audio-analysis
workflows. `decodeBrowserAudioBlob()` decodes a caller-provided Blob locally,
downmixes/resamples it through the browser Web Audio implementation, validates
the result, and returns one explicit mono `Float32Array` PCM snapshot plus
source/output audio facts.

Composed products should decode once here and pass the same trusted PCM snapshot
to reusable consumers such as browser ASR and diarization instead of asking
each model adapter to decode the selected file again. Standalone capability
adapters may retain their Blob convenience entry points for compatibility.

No selected media is uploaded and there is no server or Python fallback.

```bash
bun run --cwd packages/audio-analysis-io-wasm build
bun test packages/audio-analysis-io-wasm/tests/package.test.ts
```
