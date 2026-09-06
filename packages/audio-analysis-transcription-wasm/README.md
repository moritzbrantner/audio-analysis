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

Browser consumers that already own a `MediaStream` can keep the acquisition boundary even thinner.
`createBrowserMediaStreamTranscriptionSession()` does **not** request capture permission and does not
stop the caller's track. It adapts the already-approved audio track through an exact 16 kHz
`AudioContext` and an `AudioWorklet`, mixes channels to mono, sends one-second PCM chunks into the
bounded transcription session, and keeps the output graph silent. The consumer decides when capture
is finished and then calls `finish()`.

```js
import {
  createBrowserMediaStreamTranscriptionSession,
} from "@moritzbrantner/audio-analysis-transcription-wasm";

const displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
const transcription = await createBrowserMediaStreamTranscriptionSession(displayStream, {
  source: "shared-tab-audio",
  onSegments: (segments) => renderCommittedSegments(segments),
});

await waitUntilDisplayTrackEnds(displayStream);
const result = await transcription.finish();
```

If the browser cannot create an exact 16 kHz `AudioContext`, cannot run `AudioWorklet`, or suspends
the capture graph outside a valid user interaction, the adapter fails explicitly rather than
silently changing the sample-rate contract. The caller still owns `getDisplayMedia`, track lifetime,
and UI permission/error handling.

Media acquisition remains consumer-owned. `audio-analysis` owns the 16 kHz mono input contract,
window/stride semantics, overlap stitching, model/runtime choice, backpressure limit, browser PCM
adaptation, and normalized timed result. The WebGPU browser provider intentionally stops at
transcription plus timed segments; alignment, diarization, and translation remain native/server
capabilities rather than being approximated in this browser adapter.

```bash
bun run --cwd packages/audio-analysis-transcription-wasm build
bun run --cwd packages/audio-analysis-transcription-wasm test
```
