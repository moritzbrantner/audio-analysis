#!/usr/bin/env bash
set -euo pipefail

target="wasm32-unknown-unknown"

if ! rustup target list --installed | grep -Fxq "$target"; then
  rustup target add "$target"
fi

cargo check --locked \
  -p moenarch-audio-analysis-transcription-wasm \
  --target "$target"
