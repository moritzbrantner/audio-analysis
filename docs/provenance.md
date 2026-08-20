# Clean-copy provenance

| Field | Value |
| --- | --- |
| Source repository | `moritzbrantner/rust-packages` |
| Source commit | `b8b29cf8db0b86ed1b133a18155adf24992f9483` |
| Extraction issue | [rust-packages#115](https://github.com/moritzbrantner/rust-packages/issues/115) |
| Parent PRD | [rust-packages#106](https://github.com/moritzbrantner/rust-packages/issues/106) |
| Destination | `moritzbrantner/audio-analysis` |
| License | MIT OR Apache-2.0; copied `LICENSE-MIT` and `LICENSE-APACHE` |

The exact copied Cargo, npm, app, CLI, server, and WASM package selection is frozen in `docs/ownership/audio-package-ownership.json`, filtered directly from the source repository's `docs/repository-split/package-ownership.json` where `target_repository == "audio-analysis"`.

Copied code keeps its focused source layout under `crates/audio/`, `crates/bindings/`, and `packages/`. No source history was rewritten and no code was removed from the source repository. Generated `dist/` and WASM `pkg/` artifacts are ignored.

## Focused app UI boundary

The extraction supersedes the bootstrap's external UI prerequisite: the 13
focused app adapters now depend on the private destination-owned
`@moritzbrantner/audio-app-ui` package. This removes the visual-analysis UI
edge while preserving the adapters' package-surface contract.

The private adapter is not publishable under this extraction. Any app-adapter
release requires a separate decision covering whether to publish or bundle it,
plus an isolated consumer check; this does not authorize UI publication from
this repository.
