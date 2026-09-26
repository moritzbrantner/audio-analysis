# @moritzbrantner/audio-analysis-speakers-wasm

WASM package for `audio-analysis-speakers`.

```bash
bun run --cwd packages/audio-analysis-speakers-wasm build
```

## Browser speaker diarization

The package also exposes a browser-local diarization adapter that does not
require the generated Rust/WASM package. It consumes normalized 16 kHz mono PCM
and keeps reusable speaker mechanics in `audio-analysis`:

- `browserDiarizationCapabilities()` reports the exact runtime and model
  contract.
- `supportsBrowserDiarization()` checks the browser runtime before model work.
- `diarizeBrowserAudioSamples()` segments audio in bounded 10-second windows,
  extracts speaker embeddings, clusters identities across the full recording,
  and returns timed speaker segments.
- `assignBrowserDiarizationToTranscript()` applies those segments to an
  existing timed transcription contract by majority overlap.

The browser adapter uses the public
`onnx-community/pyannote-segmentation-3.0` segmentation model and
`Xenova/wavlm-base-plus-sv` speaker embeddings through the exact
Transformers.js 3.8.1 browser runtime. Both inference models run locally through
WASM with q8 weights and are cached by the browser. Model assets may be fetched
on first use, but inference never falls back to a server or Python runtime.

The segmentation model is deliberately executed in its documented 10-second
window shape. Speaker embeddings provide the cross-window identity signal, so
local segmentation labels are not treated as globally stable speaker IDs.
