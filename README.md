# audio-analysis

Rust-first audio analysis, recognition, transcription, synthesis, and generation
packages extracted from `moritzbrantner/rust-packages`.

This repository currently preserves the 53 Cargo and 26 Bun package surfaces
reviewed for the audio capability. The focused `packages/audio-app-ui` adapter
is private destination support and is not part of that source inventory.

## Extraction status

This change is restructuring-first. Behavioral, unit, integration, consumer,
WASM, Clippy, documentation, build, and package suites are deliberately not
part of the extraction gate. The retained gate checks ownership, provenance,
dependency boundaries, byte identity for unadapted copies, and Cargo metadata
when all exact registry dependencies are available.

`moenarch-audio-contracts = 0.1.0` remains an external foundation dependency.
The checked-in lockfile resolves it and the other required foundation/NLP
contracts from crates.io; it does not encode sibling-path or Git dependencies.
