# Agent Instructions

## Purpose

This repository owns the audio-analysis capability family. Core crates in `crates/audio/` own audio semantics and implementations; focused CLI/server/WASM adapters and matching package surfaces preserve adapter parity. Foundation and narrow transcript/NLP contracts are consumed as released dependencies, never through committed cross-repository paths or moving Git branches.

## Working agreement

1. Check `git status --short --branch` before and after edits.
2. Keep crate behavior composable and preserve focused adapters until an explicit semver/migration decision approves removal.
3. Do not publish, tag, or remove the source copy in `rust-packages` without an exact release issue and validated release manifest.
4. Before merge, run the ordered commands in `.agent-loop.toml` against the exact PR head in an isolated checkout and attach the resulting receipt.

## Ownership

The canonical copied set is `docs/ownership/audio-package-ownership.json`. Audio-specific FFmpeg, model, native-runtime, and external-tool composition stays here. Generic media/time, cancellation, diagnostics, model lifecycle, and narrow transcript contracts remain foundation/NLP responsibilities.

## Generated files

Do not commit `target/`, `node_modules/`, `dist/`, WASM `pkg/` output, or `.agent-loop/verification/`.
