# audio-analysis-core

Shared audio frame conversion, whole-buffer clip editing primitives, windowing,
and streaming helpers for `moritzbrantner-video-analysis`.

## Feature flags

- No optional feature flags today.

## Example

```rust,ignore
use audio_analysis_core::{AudioClip, ConcatPolicy, FrameSpec, StreamingFrameBuffer};
use audio_contracts::{AudioBuffer, OwnedAudioFrame, Timebase, Timestamp};

let frame = OwnedAudioFrame::new(
    Timestamp::new(0, Timebase::new(1, 48_000)),
    48_000,
    1,
    AudioBuffer::F32(vec![0.0; 4_096]),
)?;

let spec = FrameSpec::new(2_048, 512)?;
let mut windows = StreamingFrameBuffer::new(spec);
let frames = windows.push_frame(&frame.as_frame()?)?;

let clip = AudioClip::from_frames(&[frame])?;
let parts = clip.split_at_seconds(&[0.025, 0.05])?;
let joined = AudioClip::concat(&parts, ConcatPolicy::RequireSameFormat)?;

assert!(!frames.is_empty());
assert_eq!(joined.channels, 1);
```

## Spectral ownership

`audio_analysis_core::spectral` owns reusable FFT, STFT, spectrum, spectral-feature,
and phase-aware novelty primitives. The `audio-analysis-fourier` package remains
the compatibility/runtime surface and re-exports these computational symbols.

STFT execution reuses one planned FFT and one complex scratch buffer across a frame
sequence instead of rebuilding planner and window buffers for every frame.

## Whole-Buffer Editing

`AudioClip` stores validated interleaved `f32` audio with sample rate and channel
metadata. It supports sample/second slicing, timeline splitting, concat, and
mixing. Format-changing concat is explicit through `ConcatPolicy::ResampleToFirst`;
mixing still requires matching sample rate and channels.

## Capture metrics

`capture_metrics(interleaved, sample_rate, channels, &CaptureMetricsConfig)` measures generic
capture quality: duration, channel count, clipped-sample ratio at a declared clip level, seconds of
no-input frames (frame RMS below a floor) and the longest continuous no-input run, and seconds of
activity frames (frame RMS above a floor). Frame RMS spans all channels of the frame. Defaults are
20 ms frames, clip level 0.999, no-input floor 1e-4 and activity floor 0.01; the result echoes the
configuration used. Interpretation (what is usable for a purpose) belongs to the consumer.

`tests/fixtures/capture-metrics.json` holds shared fixtures that the Rust library and the
`@moritzbrantner/audio-analysis-core-wasm` `captureMetrics` export must both reproduce.

## Package surface

Primary workflow: `audio.levels`.

Workflow operations:

- `audio.levels`: Returns deterministic level metrics for normalized audio samples.
- `audio.frames`: Summarizes fixed-size analysis frames over normalized samples.
- `audio.captureMetrics`: Measures clipping, no-input and activity over interleaved samples with declared thresholds.
- `audio.timestamps`: Converts between seconds, samples, and timestamp ticks for a sample rate.

Debug operations:

- `describe`: inspect package metadata and runtime support.

Runtime support: library, CLI, server, and WASM wrappers expose these operations.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-audio-analysis-core-cli -- run \
  --operation audio.levels \
  --json '{"channels":1,"sampleRate":48000,"samples":[0.0,0.5,-0.5]}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `video-analysis-core`
- `audio-analysis-fourier`
- `audio-analysis-processing`
