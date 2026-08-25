# Agent instructions

This is the capability repository for audio analysis and generation packages.

- Keep reusable Rust code under `crates/` and focused package surfaces under `packages/`.
- Do not reintroduce sibling paths, moving Git dependencies, or visual-analysis dependencies into committed package manifests.
- `packages/audio-app-ui` is private destination support, not an automatically publishable package.
- Ordinary feature work is source-first. Do not publish crates, bump package versions, create tags, or start a release train merely to unblock a consumer.
- Keep package versions stable during source-development work when possible; a dedicated release change owns version bumps and registry publication.
- Native WhisperX and other consumers may validate exact audio source revisions through their source-development configuration before registry releases exist.
- Do not create a new independently versioned crate unless there is a second independent consumer, a hard dependency/isolation boundary, or another concrete independent-versioning reason.
- If a consumer task expands beyond the consumer plus two upstream repositories, treat that as an architecture boundary problem unless broader migration scope was explicitly assigned.
- Publication and source removal require separate exact authorization.
- Registry-only consumer verification is release evidence; it is not required before source-mode implementation evidence is useful.
- For the restructuring-first extraction, use only the structural commands in `.agent-loop.toml`; do not imply that skipped behavioral checks passed.
