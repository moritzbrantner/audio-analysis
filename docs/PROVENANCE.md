# Clean-copy provenance

This repository was created by clean-copy extraction. Git history was not
rewritten or filtered.

- Source repository: `moritzbrantner/rust-packages`
- Reviewed Phase-A ownership baseline: `d032ad2890c1df3c6a5b9eff024562f00d017fce`
- Exact extraction commit: `b8b29cf8db0b86ed1b133a18155adf24992f9483`
- Extraction issue: `moritzbrantner/rust-packages#115`
- Parent PRD: `moritzbrantner/rust-packages#106`
- Destination: `moritzbrantner/audio-analysis`
- Reviewed 79-record digest: `b3e231c734b8615c524b012971458ea1370997c20bc2c57ced41934e6af317fc`

The 53 Cargo directories and 26 Bun package directories named in the reviewed
ownership inventory were copied from the exact commit. `audio-contracts` was
not copied because it is foundation-owned.

Adapted trees are recorded by `docs/repository-split/copy-adaptations.json`.
All other inventory trees must remain byte-identical to the source. Root
manifests, locks, documentation, CI, Harness metadata, validators, and
`packages/audio-app-ui` are destination-authored support.

The source `Cargo.lock` was deliberately not retained: lock generation stops at
the absent exact registry package `moenarch-audio-contracts 0.1.0`, while the
source lock identifies it as a monolith-local path package. A destination lock
must be generated after that exact foundation release becomes visible.

`packages/audio-app-ui` derives from the extraction commit's package-surface
workbench and shared primitive seam. The 13 app packages were retargeted from
the unpublished broad UI workspace package to this private focused adapter.

No package publication, tag, release, consumer migration, source deletion, or
source relocation is authorized by this extraction.
