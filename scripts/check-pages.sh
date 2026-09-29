#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/_site}"

cd "$ROOT_DIR"

bun test tests/pages
bash scripts/build-pages.sh "$OUTPUT_DIR"
node scripts/check-pages-artifact.mjs "$OUTPUT_DIR"
