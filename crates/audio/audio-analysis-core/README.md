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

## Whole-Buffer Editing

`AudioClip` stores validated interleaved `f32` audio with sample rate and channel
metadata. It supports sample/second slicing, timeline splitting, concat, and
mixing. Format-changing concat is explicit through `ConcatPolicy::ResampleToFirst`;
mixing still requires matching sample rate and channels.

## Package surface

Primary workflow: `audio.levels`.

Workflow operations:

- `audio.levels`: Returns deterministic level metrics for normalized audio samples.
- `audio.frames`: Summarizes fixed-size analysis frames over normalized samples.
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
