#!/usr/bin/env bash
set -euo pipefail

cargo metadata --format-version 1 --no-deps
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
