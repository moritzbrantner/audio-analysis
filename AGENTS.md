# Agent instructions

This is the capability repository for audio analysis and generation packages.

- Keep reusable Rust code under `crates/` and focused package surfaces under `packages/`.
- Do not reintroduce sibling paths, moving Git dependencies, visual-analysis dependencies, or NLP implementation dependencies into committed package manifests.
- `packages/audio-app-ui` is private destination support, not an automatically publishable package.
- On a fresh machine or after the declared toolchain/environment contract changes, run `bash scripts/codex-environment.sh setup`. Use `maintenance` for an existing environment when dependency state changes.
- Before starting implementation, run `bash scripts/check-agent-readiness.sh`. It verifies the semantic environment fingerprint, locked Cargo metadata, and free build-disk capacity before model time is spent. Use `--with-source` when the task requires the exact Foundation source graph.
- The default free-space floor is 8 GiB. `AGENT_MIN_FREE_GIB` may be raised for larger builds; lower it only for an intentionally constrained environment, never to hide an exhausted target filesystem.
- Ordinary feature work is source-first. Use `bash scripts/source-deps activate` when it needs unreleased `moenarch-foundation` changes; the committed declaration pins one exact reviewed foundation revision.
- Timed-text interchange belongs to `moenarch-media-core`; transcript parsing, formatting, and NLP enrichment remain optional downstream behavior rather than audio dependencies.
- Source mode is local-workspace owned: every declared sibling checkout must exist at the exact pinned revision. Do not add private-repository tokens or authenticated Git fallback so hosted CI can reproduce the multi-repository workspace.
- Source verification uses a temporary Cargo.lock resolution because exact local Foundation patches can differ from the registry graph. `scripts/source-lock` materializes that graph once, validates it against `.coding-tooling.source-lock.json`, keeps all verification commands `--locked`, and restores the registry Cargo.lock afterward.
- Never commit the temporary source-mode Cargo.lock. If a deliberate source dependency or registry dependency change alters its resolution, review the graph and deliberately update `.coding-tooling.source-lock.json`; an unexpected hash change is dependency drift, not a reason to remove `--locked`.
- If an interrupted run leaves `/.cargo/source-lock-state/`, inspect with `bash scripts/source-lock status` and recover the saved registry lock with `bash scripts/source-lock restore` before starting another verification scope.
- Do not publish crates, bump package versions, create tags, or start a release train merely to unblock a consumer or an upstream dependency.
- Keep package versions stable during source-development work when possible; a dedicated release change owns version bumps and registry publication.
- Native WhisperX and other consumers may validate exact audio source revisions through their source-development configuration before registry releases exist.
- Generated `.cargo/config.toml` is development state and must not be committed. Deactivate source mode before registry-only release verification.
- Do not create a new independently versioned crate unless there is a second independent consumer, a hard dependency/isolation boundary, or another concrete independent-versioning reason.
- The capability library crates are the default Cargo workspace members. Existing per-capability CLI, server, WASM, and app packages are compatibility shells, not the default development surface.
- Do not create another per-capability CLI/server/WASM/app shell. If a genuinely new transport surface is required, prefer one repository-level adapter and reuse the library-owned operation contracts.
- Do not mechanically update compatibility shells when only implementation internals change. Touch them when their public operation/transport contract actually changes.
- During implementation, use `bash scripts/check-fast.sh <package>` for the narrow package you are changing. Use `bash scripts/check-fast.sh` without a package only when the change genuinely spans the workspace.
- Before handing work off, run `bash scripts/check-handoff.sh` once. It is the authoritative Agent Loop gate: diff hygiene, repository structure, locked metadata, workspace Clippy, and workspace tests.
- `bash scripts/check-preflight.sh` is the exhaustive CPU/distribution gate: it layers important non-CUDA feature combinations, compatibility adapters, documentation, and packaging on top of handoff. Hosted CI runs it; do not repeatedly replay it after every edit.
- CUDA is resource-backed. Do not enable workspace `--all-features` on ordinary CPU CI and then treat missing `nvcc` as an application failure. Run `bash scripts/check-cuda.sh` on a CUDA-equipped machine and keep CPU and CUDA evidence distinct.
- Treat repeated co-change across independently versioned library crates as evidence for future consolidation. Consolidate only when the ownership and public API boundary are clear; do not create another abstraction layer to hide the problem.
- If a consumer task expands beyond the consumer plus two upstream repositories, treat that as an architecture boundary problem unless broader migration scope was explicitly assigned.
- Publication and source removal require separate exact authorization.
- Registry-only consumer verification is release evidence; it is not required before source-mode implementation evidence is useful.
