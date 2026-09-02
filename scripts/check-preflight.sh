#!/usr/bin/env bash
set -euo pipefail

if [[ "${AUDIO_ANALYSIS_SOURCE_LOCK_SCOPE:-0}" != "1" ]]; then
  exec bash scripts/source-lock run -- bash "$0" "$@"
fi

bash scripts/check-handoff.sh
bash scripts/check-adapters.sh
cargo doc --locked --workspace --no-deps
cargo package --workspace --locked --no-verify
