#!/usr/bin/env bash
set -euo pipefail

git diff --check
python3 scripts/check_extraction.py
bash scripts/check-fast.sh
