# Native pyannote community diarization bundle provenance

- Source: `pyannote/speaker-diarization-community-1`
- Pinned revision: `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`
- License: `CC-BY-4.0`
- Distribution: caller-owned checksum-addressed local cache snapshot
- Conversion command: `python scripts/convert_pyannote_community.py --source-dir <standard-hugging-face-cache-snapshot> --two-speaker-fixture <local-two-speaker-wav>`
- Python: `3.11.15`
- ONNX opset: `17`
- Segmentation contract: `waveform` [1, 1, 160000] -> `segmentations` [1, 589, 3]
- Embedding contract: `waveform` [1, 1, 160000] + `masks` [1, 589] -> `embeddings` [1, 256]
- Segmentation max absolute difference: 0.0
- Embedding max absolute difference: 1.862645149230957e-06
- Numerical tolerance: 0.0001
- PLDA: 256-to-128 x-vector transform plus 128-dimensional model
- Clustering: pinned VBx threshold/Fa/Fb configuration

Source and generated SHA-256 values are retained in `pyannote_diarization_manifest.json`. No credentials, source weights, audio, or absolute source paths are stored in this bundle.

Automatic remote hydration is intentionally out of scope. Verify and install this immutable local snapshot before cache-only execution.
