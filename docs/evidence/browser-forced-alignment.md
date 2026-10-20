# Browser forced alignment

Consumer: native-whisperx#392. Upstream: #170.

The browser adapter runs real English/German wav2vec2 ONNX emissions on CPU
WebAssembly and calls the same portable Rust CTC trellis used by native alignment.
There is no Python or server inference fallback. Models and revisions are exposed
through the capability API and documented in the package README.

Public integration tests cover exact word times, segment identity, malformed and
impossible emissions, bounded trellis allocation, immutable source input and
unsupported language rejection before model acquisition. The browser package's
normal test command also checks the typed alignment entry.

A Chromium 1248 probe loaded the actual bundled adapter, ONNX Runtime Web 1.30.0,
the generated Rust WebAssembly module and both actual pinned ONNX models. Model
and runtime HTTPS requests were redirected to a local HTTP cache to avoid repeated
large downloads; model bytes and execution were real. The owned synthetic speech
fixtures were generated with espeak-ng 1.52.0 and resampled to mono 16 kHz.

| Language | Input text | Words | Aligned range | Audio duration |
| --- | --- | --- | --- | --- |
| en | Hello there. This is a browser alignment test. | 8 | 0.040–2.574 s | 2.875 s |
| de | Guten Tag. Dies ist ein Test der Zeitstempel. | 8 | 0.000–2.792 s | 3.033 s |

Both preserved original text, segment ID 7 and unchanged source objects; every
word stayed within the audio bounds. This establishes real browser execution,
not conversational accuracy or physical WebGPU performance. The alignment backend
uses CPU WASM. Long recordings require bounded ASR segments of at most 30 seconds.

Native optional alignment/model-bundle library tests: 263 passed, 8 ignored.
Native alignment module: 11 passed, 1 ignored. Browser package: 21 passed.
Shared convention sourceRevision: 6cc05789dab39cee6758b288f83fb48b4ee6dbdb.
