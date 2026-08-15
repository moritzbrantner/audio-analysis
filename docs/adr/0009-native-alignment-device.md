# ADR 0009: Native alignment inherits workflow device

Direct alignment calls retain CPU as the compatibility default. A native transcription workflow that explicitly owns a device may pass CUDA or Auto so the alignment provider resolves the same device and emits alignment-device diagnostics. CTC emissions still return to CPU vectors for trellis construction and backtracking.
