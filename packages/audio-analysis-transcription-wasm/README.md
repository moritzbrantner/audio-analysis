# @moritzbrantner/audio-analysis-transcription-wasm

Browser/WASM adapter for `audio-analysis-transcription`.

The Rust/WASM surface remains the deterministic package contract. The package also owns the first
browser-local Whisper provider used by static consumers: audio is decoded and resampled to 16 kHz
mono in the browser, then transcribed with Whisper tiny on WebGPU. Model assets are cached by the
browser and no server, Python, or CPU fallback is used.

```js
import {
  browserTranscriptionCapabilities,
  supportsBrowserTranscription,
  transcribeAudioBlob,
} from "@moritzbrantner/audio-analysis-transcription-wasm";

if (await supportsBrowserTranscription()) {
  const result = await transcribeAudioBlob(audioBlob, {
    source: "captured-tab-audio",
    onProgress: ({ message }) => console.log(message),
  });
  console.log(result.text, result.segments);
}
```

The WebGPU browser provider intentionally stops at transcription plus timed segments. Alignment,
diarization, and translation remain native/server capabilities. The in-progress Burn provider keeps
its own caller-provisioned model contract until it passes a real browser inference gate; this
browser provider does not falsely mark that Rust path complete.

```bash
bun run --cwd packages/audio-analysis-transcription-wasm build
bun run --cwd packages/audio-analysis-transcription-wasm test
```
