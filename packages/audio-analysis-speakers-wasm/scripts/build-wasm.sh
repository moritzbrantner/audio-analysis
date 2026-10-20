#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
wasm-pack build "$ROOT_DIR/crates/bindings/audio-analysis-speakers-wasm" --target web --out-name audio_analysis_speakers_wasm --out-dir "$ROOT_DIR/packages/audio-analysis-speakers-wasm/pkg"
