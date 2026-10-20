# Browser community diarization

Issue #172 exposes the existing Rust community-1 mask selection, PLDA/VBx clustering and reconstruction through WASM. Native ONNX execution uses the same extracted functions. The browser executes the approved segmentation and embedding graphs using ONNX Runtime Web 1.30 CPU WASM, with no Python or server inference.

## Resources and credentials

`loadCommunityDiarizationResources({token, signal, onProgress})` requests the two original gated PLDA NPZ files at community-1 revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` with an Authorization header. Callers must accept the upstream model conditions. Credentials are omitted elsewhere and never returned in the bundle or archive. Errors do not include response bodies or credentials. The caller controls input clearing and must not persist the token.

Two pinned public ONNX donors supply the byte-identical learned tensors. Checked-in gzip templates contain graph structure and deterministic frontend buffers; all trained tensor payloads are zero. The development generator audits every retained float buffer and Constant attribute, rejects unfamiliar structures, and verifies exact reconstruction against the approved native model hashes. Ordinary tests independently inspect every zero span. No trained weights are committed.

The loader verifies original NPZ hashes, array types/shapes/order and finite values, reproduces the approved PLDA JSON bytes, restores the original graphs, and verifies the complete approved artifact set. `archiveCommunityDiarizationResources` creates a verified flat stored ZIP for a native consumer's authorized model cache. It contains neither the token nor original NPZ files.

## Real browser evidence

Chromium 1248 executed actual ONNX graphs and the Rust WASM module on licensed VoxConverse conversations, using mono 16 kHz PCM. Only resource HTTP acquisition was replaced with already licensed local bytes; browser cryptography, NPZ parsing, model reconstruction, ONNX inference and Rust clustering were real. No HF_TOKEN was available, so live gated authorization is not claimed. Offline tests verify authorization routing and denied-access behavior.

| Case | Duration | Expected speakers | Browser speakers | Turns | CPU WASM wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| ltgmz | 209.664 s | 2 | 2 | 48 | 455.40 s |
| wcxfk | 155.648 s | 3 | 3 | 62 | 393.82 s |

These counts and turn totals agree with the native consumer evidence in native-whisperx #362. Results are recorded in `browser-community-diarization-results.json`; no audio, transcript or identity embeddings are included. This small selected check is not a general accuracy estimate. Counts remain tentative automatic observations; count uncertainty is not reported. Speaker labels are anonymous, not verified identities.

A macrotask yield after each diarization window and alignment segment permits rendering and cancellation during CPU WASM execution. Cancellation is checked between inference calls; an executing individual ONNX call cannot be interrupted. Failed or cancelled runs publish no partial result. One adapter admits one active run and caches its two model sessions; dispose releases them.

## Checks

Portable-core tests: 62 passed, 1 ignored; native optional-core tests: 71 passed, 2 ignored; three public resource/mask tests; speaker SDK: 13 tests; alignment SDK: 21 tests. Strict TypeScript, targeted strict Clippy, WASM build, source readiness and repository handoff passed. Real resource conversion produced eight files byte-identical to the approved bundle and a 34,143,397-byte stored ZIP. Shared convention sourceRevision: `6cc05789dab39cee6758b288f83fb48b4ee6dbdb`.
