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

## Pyannote community-1

Use `./resources` to load approved resources with a request-scoped Hugging Face token, and `./community` for actual CPU WASM community-1 diarization. Accept the upstream model conditions first. The existing root API remains a separate baseline capability. Community diarization never substitutes that baseline on failure. See [resource, privacy and real-browser evidence](../../docs/evidence/browser-community-diarization.md).

```ts
import { loadCommunityDiarizationResources } from "@moritzbrantner/audio-analysis-speakers-wasm/resources";
import { diarizeCommunityAudioSamples } from "@moritzbrantner/audio-analysis-speakers-wasm/community";
const bundle = await loadCommunityDiarizationResources({ token, signal });
const result = await diarizeCommunityAudioSamples(samples16k, bundle, { signal });
```

Keep tokens out of persisted application state. Returned counts are tentative and labels anonymous. Resources remain in caller-owned memory; use `archiveCommunityDiarizationResources` only when explicitly provisioning a native cache.
