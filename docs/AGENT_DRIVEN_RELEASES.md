# Agent-driven Cargo release

The active release is the exact six-package native-whisperx audio contract
closure in `releases/native-whisperx-audio-contract-closure.toml`.

Preparation and publication are separate operations. A release preparation PR
must leave crates.io, tags, GitHub Releases, labels, and downstream branches
unchanged. Publication is permitted only from the exact clean control head after
issue #6 records that head and the manifest SHA-256, every declared check passes,
and the live `release:approved` label is present.

The publisher is fail-closed and resumes only from a valid dependency-ordered
registry prefix. Never republish, overwrite, delete, or automatically yank an
artifact.
