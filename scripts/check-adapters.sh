#!/usr/bin/env bash
set -euo pipefail

# The handoff gate already covers the complete workspace with default features.
# This script adds the important non-CUDA optional feature combinations and
# compatibility adapter surfaces. CUDA remains resource-backed and is checked
# separately by scripts/check-cuda.sh on a machine with nvcc.
cargo test --locked -p moenarch-audio-analysis-speakers \
  --features onnx,pyannote-diarization,model-bundles

cargo test --locked -p moenarch-audio-analysis-transcription \
  --features candle,alignment,diarization,onnx,pyannote-diarization,silero-vad,pyannote-vad,model-bundles,audio-io,native

cargo test --locked -p moenarch-audio-generation-tts \
  --features candle,model-bundles,audio-io,asr
cargo test --locked -p moenarch-audio-generation-tts-cli \
  --features candle,model-bundles,audio-io,asr
cargo test --locked -p moenarch-audio-generation-tts-server \
  --features candle,model-bundles,audio-io,asr
