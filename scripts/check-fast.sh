#!/usr/bin/env bash
set -euo pipefail

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
