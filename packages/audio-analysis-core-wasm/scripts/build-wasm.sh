#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
wasm-pack build "$ROOT_DIR/crates/bindings/audio-analysis-core-wasm" --target web --out-dir "$ROOT_DIR/packages/audio-analysis-core-wasm/pkg"
