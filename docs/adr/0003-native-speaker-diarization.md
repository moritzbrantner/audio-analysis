# ADR 0003: Native speaker diarization provider

Default diarization remains a deterministic hermetic heuristic. The first model-backed target is an opt-in ONNX speaker embedding provider in `moenarch-audio-analysis-speakers`, using shared runtime infrastructure rather than a direct ONNX Runtime dependency. No default downloads, tokens, network, CUDA, Python, or model files are introduced.
