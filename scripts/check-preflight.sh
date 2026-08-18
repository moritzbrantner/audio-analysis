#!/usr/bin/env bash
set -euo pipefail

scripts/check-fast.sh
cargo doc --workspace --no-deps
cargo package --workspace --locked --no-verify
