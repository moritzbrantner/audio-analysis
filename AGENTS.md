# Agent instructions

This is the capability repository for audio analysis and generation packages.

- Keep reusable Rust code under `crates/` and focused package surfaces under `packages/`.
- Do not reintroduce sibling paths, moving Git dependencies, or visual-analysis dependencies into committed package manifests.
- `packages/audio-app-ui` is private destination support, not an automatically publishable package.
- Ordinary feature work is source-first. Use `bash scripts/source-deps activate` when it needs unreleased `moenarch-foundation` or `nlp-stack` changes; the committed declaration pins exact reviewed revisions.
- Do not publish crates, bump package versions, create tags, or start a release train merely to unblock a consumer or an upstream dependency.
- Keep package versions stable during source-development work when possible; a dedicated release change owns version bumps and registry publication.
- Native WhisperX and other consumers may validate exact audio source revisions through their source-development configuration before registry releases exist.
- Generated `.cargo/config.toml` is development state and must not be committed. Deactivate source mode before registry-only release verification.
- Do not create a new independently versioned crate unless there is a second independent consumer, a hard dependency/isolation boundary, or another concrete independent-versioning reason.
- The capability library crates are the default Cargo workspace members. Existing per-capability CLI, server, WASM, and app packages are compatibility shells, not the default development surface.
- Do not create another per-capability CLI/server/WASM/app shell. If a genuinely new transport surface is required, prefer one repository-level adapter and reuse the library-owned operation contracts.
- Do not mechanically update compatibility shells when only implementation internals change. Touch them when their public operation/transport contract actually changes.
- Use `scripts/check-fast.sh` for the ordinary local library loop. Run `bash scripts/check-adapters.sh` when adapter shells change or when distribution compatibility is being checked. CPU CI covers the full workspace with default features plus the important non-CUDA optional feature combinations.
- CUDA is resource-backed. Do not enable workspace `--all-features` on ordinary CPU CI and then treat missing `nvcc` as an application failure. Run `bash scripts/check-cuda.sh` on a CUDA-equipped machine and keep CPU and CUDA evidence distinct.
- Treat repeated co-change across independently versioned library crates as evidence for future consolidation. Consolidate only when the ownership and public API boundary are clear; do not create another abstraction layer to hide the problem.
- If a consumer task expands beyond the consumer plus two upstream repositories, treat that as an architecture boundary problem unless broader migration scope was explicitly assigned.
- Publication and source removal require separate exact authorization.
- Registry-only consumer verification is release evidence; it is not required before source-mode implementation evidence is useful.
- For the restructuring-first extraction, use only the structural commands in `.agent-loop.toml`; do not imply that skipped behavioral checks passed.
