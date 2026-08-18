# audio-analysis

`audio-analysis` is the capability-owned home for the Rust audio library family: IO and frame conversion, Fourier/signal analysis, pitch and rhythm analysis, recognition, separation, speaker diarization, transcription, deterministic synthesis, MIDI generation, speaker-conditioned TTS planning, and focused CLI, server, WASM, npm, and app adapters.

This is a clean-copy bootstrap from [`moritzbrantner/rust-packages`](https://github.com/moritzbrantner/rust-packages) at source commit `b8b29cf8db0b86ed1b133a18155adf24992f9483`. It is additive: the source family deliberately remains in `rust-packages` until release and consumer gates have passed.

## Status

The repository records the exact package selection, provenance, and a registry-shaped dependency boundary, but is not a release authorization. Do not publish from this repository without a later exact release issue and reviewed manifest.

Focused adapters remain intentionally. Removing a CLI, server, WASM, npm, or app adapter requires separate usage, deployment, semver, and migration evidence.

## Guides

- [Agent guidance](AGENTS.md)
- [Domain context](CONTEXT.md)
- [Package ownership](docs/ownership/audio-package-ownership.json)
- [Clean-copy provenance](docs/provenance.md)
- [Draft release plan](docs/repository-split/release-plan.draft.json)
- [Draft verification harness](.harness/README.md)

The intended checks are documented in [docs/checks.md](docs/checks.md) and configured in `.agent-loop.toml`; they were intentionally not run for this bootstrap.
