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

## External UI prerequisite

The 13 focused app adapters retain their dependency on
`@moritzbrantner/video-analysis-ui` at `^0.1.0`; the UI package is deliberately
neither copied nor excluded from this extraction. The source ownership manifest
records it as a `rust-packages`-owned facade, with
`moritzbrantner/rust-packages` as its intended semantic/publication owner,
`current_published_version: null`, and a separate npm/WASM release decision.

Consequently, app adapters are not release-ready merely because their package
manifests name that range. Before any app-adapter release can be ready, the
release manifest must record a verified registry availability that satisfies the
range and an isolated registry-only consumer gate must install and exercise the
adapter with the published UI package. This prerequisite does not authorize UI
publication from this repository.
