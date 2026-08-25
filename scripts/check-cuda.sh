#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA verification requires nvcc on PATH; run this on a CUDA-equipped machine." >&2
  exit 2
fi

cargo test -p moenarch-audio-analysis-transcription --features cuda
cargo test -p moenarch-audio-generation-tts --features cuda
cargo test -p moenarch-audio-generation-tts-cli --features cuda
cargo test -p moenarch-audio-generation-tts-server --features cuda
