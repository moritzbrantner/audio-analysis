#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/_site}"

if ! command -v wasm-pack >/dev/null 2>&1; then
  printf '%s\n' "wasm-pack is required to build the Audio Inspector Pages artifact." >&2
  exit 2
fi

if ! command -v node >/dev/null 2>&1; then
  printf '%s\n' "node is required to validate the Audio Inspector JavaScript." >&2
  exit 2
fi

bash "$ROOT_DIR/packages/audio-analysis-core-wasm/scripts/build-wasm.sh"
bash "$ROOT_DIR/packages/audio-analysis-fourier-wasm/scripts/build-wasm.sh"
bash "$ROOT_DIR/packages/audio-analysis-pitch-wasm/scripts/build-wasm.sh"
bash "$ROOT_DIR/packages/audio-analysis-rhythm-wasm/scripts/build-wasm.sh"

node --check "$ROOT_DIR/site/app.js"
python3 -m json.tool "$ROOT_DIR/site/analysis-capabilities.json" >/dev/null

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/wasm"
cp -R "$ROOT_DIR/site/." "$OUTPUT_DIR/"

for package in audio-analysis-core audio-analysis-fourier audio-analysis-pitch audio-analysis-rhythm; do
  mkdir -p "$OUTPUT_DIR/wasm/$package"
  cp "$ROOT_DIR/packages/${package}-wasm/index.js" "$OUTPUT_DIR/wasm/$package/index.js"
  cp -R "$ROOT_DIR/packages/${package}-wasm/pkg" "$OUTPUT_DIR/wasm/$package/pkg"
done

touch "$OUTPUT_DIR/.nojekyll"
printf 'Audio Inspector Pages artifact: %s\n' "$OUTPUT_DIR"
