# WASM packages during source development

The WASM npm wrappers used only inside this repository are development and compatibility surfaces, not release prerequisites.

For the Native WhisperX-driven audio closure, the core, Fourier, recognition, I/O, speakers, and transcription npm wrappers are private. Their Rust/WASM builds and internal workspace consumers remain valid, but ordinary feature work must not publish them or wait on npm publication.

A wrapper should become publishable only when an independent consumer requires an npm distribution boundary and that release surface is explicitly authorized. Until then, source development and repository-local workspace consumption are the canonical paths.
