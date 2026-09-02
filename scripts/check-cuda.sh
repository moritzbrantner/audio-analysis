#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA verification requires nvcc on PATH; run this on a CUDA-equipped machine." >&2
  exit 2
fi

if [[ "${AUDIO_ANALYSIS_SOURCE_LOCK_SCOPE:-0}" != "1" ]]; then
  exec bash scripts/source-lock run -- bash "$0" "$@"
fi

cargo test --locked -p moenarch-audio-analysis-transcription --features cuda
cargo test --locked -p moenarch-audio-generation-tts --features cuda
cargo test --locked -p moenarch-audio-generation-tts-cli --features cuda
cargo test --locked -p moenarch-audio-generation-tts-server --features cuda
