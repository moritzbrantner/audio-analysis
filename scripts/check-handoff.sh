#!/usr/bin/env bash
set -euo pipefail

if [[ "${AUDIO_ANALYSIS_SOURCE_LOCK_SCOPE:-0}" != "1" ]]; then
  exec bash scripts/source-lock run -- bash "$0" "$@"
fi

git diff --check
python3 scripts/check_extraction.py
bash scripts/check-fast.sh
