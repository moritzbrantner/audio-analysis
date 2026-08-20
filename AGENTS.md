# Agent instructions

This is the capability repository for audio analysis and generation packages.

- Keep reusable Rust code under `crates/` and focused package surfaces under
  `packages/`.
- Do not reintroduce sibling paths, moving Git dependencies, or visual-analysis
  dependencies.
- `packages/audio-app-ui` is private destination support, not an automatically
  publishable package.
- Publication and source removal require separate exact authorization.
- For the restructuring-first extraction, use only the structural commands in
  `.agent-loop.toml`; do not imply that skipped behavioral checks passed.
