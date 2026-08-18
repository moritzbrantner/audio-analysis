#!/usr/bin/env bash
set -euo pipefail

wasm-pack build ../../crates/bindings/audio-generation-tts-wasm \
  --target web \
  --out-dir ../../../packages/audio-generation-tts-wasm/pkg
