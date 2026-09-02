#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  printf 'usage: %s [package]\n' "$0" >&2
  exit 2
fi

package="${1:-}"

# Temporary discovery guard for PR #46: source patches can require a distinct
# lock resolution. Materialize it once so CI can reveal the exact lock identity;
# the final contract will validate that identity instead of resolving silently.
if [[ -f .cargo/config.toml ]]; then
  cargo metadata --format-version 1 >/dev/null
  printf 'source lock sha256: '
  sha256sum Cargo.lock | awk '{print $1}'
fi

cargo metadata --locked --format-version 1 --no-deps >/dev/null

if [[ -n "$package" ]]; then
  cargo clippy --locked -p "$package" --all-targets
  cargo test --locked -p "$package"
else
  cargo clippy --locked --workspace --all-targets
  cargo test --locked --workspace
fi
