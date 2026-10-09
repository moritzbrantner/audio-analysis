# @moritzbrantner/audio-analysis-core-wasm

WASM package for `audio-analysis-core`.

```bash
bun run --cwd packages/audio-analysis-core-wasm build
```

## Capture metrics

`captureMetrics(samples, sampleRate, channels?, options?)` measures generic capture quality over
interleaved PCM normalized to [-1, 1]: duration, clipped-sample ratio at a declared clip level, seconds
of no-input frames and the longest no-input run (frame RMS below a floor), and seconds of activity
frames (frame RMS above a floor). Defaults: 20 ms frames, clip level 0.999, no-input floor 1e-4
(about -80 dBFS), activity floor 0.01 (about -40 dBFS). The result echoes the thresholds used.

It runs the Rust `audio_analysis_core::capture_metrics`, so outputs match the Rust library; both are
checked against `crates/audio/audio-analysis-core/tests/fixtures/capture-metrics.json`. Whether a
capture is usable for a given purpose is the consumer's decision.
