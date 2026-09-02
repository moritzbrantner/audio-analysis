#!/usr/bin/env bash
set -euo pipefail

bash scripts/check-handoff.sh
bash scripts/check-adapters.sh
cargo doc --locked --workspace --no-deps
cargo package --workspace --locked --no-verify
