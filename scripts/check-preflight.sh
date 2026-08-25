#!/usr/bin/env bash
set -euo pipefail

scripts/check-fast.sh
bash scripts/check-adapters.sh
cargo doc --workspace --no-deps
cargo package --workspace --locked --no-verify
