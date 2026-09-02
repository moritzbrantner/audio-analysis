#!/usr/bin/env bash
set -euo pipefail

if [[ "${AUDIO_ANALYSIS_SOURCE_LOCK_SCOPE:-0}" != "1" ]]; then
  exec bash scripts/source-lock run -- bash "$0" "$@"
fi

if (( $# > 1 )); then
  printf 'usage: %s [package]\n' "$0" >&2
  exit 2
fi

package="${1:-}"

cargo metadata --locked --format-version 1 --no-deps >/dev/null

if [[ -n "$package" ]]; then
  cargo clippy --locked -p "$package" --all-targets
  cargo test --locked -p "$package"
else
  cargo clippy --locked --workspace --all-targets
  cargo test --locked --workspace
fi
