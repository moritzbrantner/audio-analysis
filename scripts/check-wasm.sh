#!/usr/bin/env bash
set -euo pipefail

cargo test --locked \
  -p moenarch-audio-analysis-transcription-wasm \
  --no-default-features \
  --features burn-webgpu

cargo check --locked \
  --target wasm32-unknown-unknown \
  -p moenarch-audio-analysis-transcription-wasm \
  --no-default-features \
  --features burn-webgpu
