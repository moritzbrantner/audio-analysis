#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

git diff --check

if [[ -f .cargo/config.toml ]]; then
  bash scripts/check-agent-readiness.sh --with-source
else
  bash scripts/check-agent-readiness.sh
fi

bash scripts/check-fast.sh
