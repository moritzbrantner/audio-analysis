#!/usr/bin/env bash
set -euo pipefail

# Adapter shells remain compatibility surfaces, but they are not part of the
# ordinary capability-development loop. CPU CI covers the complete workspace
# plus the non-CUDA optional feature surface. CUDA is resource-backed and is
# checked separately by scripts/check-cuda.sh on a machine with nvcc.
cargo test --workspace

cargo test -p moenarch-audio-analysis-speakers \
  --features onnx,pyannote-diarization,model-bundles

cargo test -p moenarch-audio-analysis-transcription \
  --features candle,alignment,diarization,onnx,pyannote-diarization,silero-vad,pyannote-vad,model-bundles,audio-io,native

cargo test -p moenarch-audio-generation-tts \
  --features candle,model-bundles,audio-io,asr
cargo test -p moenarch-audio-generation-tts-cli \
  --features candle,model-bundles,audio-io,asr
cargo test -p moenarch-audio-generation-tts-server \
  --features candle,model-bundles,audio-io,asr
