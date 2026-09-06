# @moritzbrantner/audio-analysis-transcription-wasm

Browser/WASM adapter for `audio-analysis-transcription`.

The Rust/WASM surface remains the deterministic package contract. The package also owns the first
browser-local Whisper provider used by static consumers: audio is decoded and resampled to 16 kHz
mono in the browser, then transcribed with Whisper tiny on WebGPU. Model assets are cached by the
browser and no server, Python, or CPU fallback is used.

For short files, use the Blob helper:

```js
import {
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

Long-running capture should use the bounded PCM session instead of retaining an entire recording.
The session accepts 16 kHz mono `Float32Array` chunks, runs deterministic 29-second windows with a
5-second overlap, commits only the non-overlap timeline, and keeps at most 58 seconds of queued PCM.
Callers must await `push()` as the backpressure boundary. If acquisition outruns inference far enough
to exceed the bound, the session fails closed with `BrowserTranscriptionBackpressureError` rather
than growing memory without limit.

```js
import {
  createBrowserTranscriptionSession,
} from "@moritzbrantner/audio-analysis-transcription-wasm";

const session = createBrowserTranscriptionSession({ source: "shared-tab-audio" });

for await (const pcmChunk of capture16kMonoPcm()) {
  await session.push(pcmChunk);
}

const result = await session.flush();
console.log(result.text, result.segments);
```

Media acquisition remains consumer-owned. `audio-analysis` owns the 16 kHz mono input contract,
window/stride semantics, overlap stitching, model/runtime choice, backpressure limit, and normalized
timed result. The WebGPU browser provider intentionally stops at transcription plus timed segments;
alignment, diarization, and translation remain native/server capabilities rather than being
approximated in this browser adapter.

```bash
bun run --cwd packages/audio-analysis-transcription-wasm build
bun run --cwd packages/audio-analysis-transcription-wasm test
```
