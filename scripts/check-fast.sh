#!/usr/bin/env bash
set -euo pipefail

cargo metadata --locked --format-version 1 --no-deps
cargo clippy --locked --workspace --all-targets
cargo test --locked --workspace
