# audio-analysis-separation

HTDemucs/Demucs-based audio stem separation for `moritzbrantner-video-analysis`.

This crate is a Rust-first wrapper around the external `demucs` command. It does
not run native inference itself; instead it builds typed separation commands,
executes them, and discovers the expected stem outputs on disk.

## Feature flags

- `external-tests`: enables ignored real-tool smoke tests that invoke `demucs`.

## Requirements

- `demucs` must be installed and available on `PATH`, or supplied explicitly via
  [`HtdemucsOptions::command`].

## Example: split a full song into stems

```rust,ignore
use audio_analysis_separation::{DemucsModel, HtdemucsOptions, HtdemucsSeparator};

let separator = HtdemucsSeparator::new(
    HtdemucsOptions::new("separated").model(DemucsModel::Htdemucs),
)?;

let result = separator.separate("song.wav")?;
assert!(result.all_outputs_present);
```

## Example: extract vocals and accompaniment only

```rust,ignore
use audio_analysis_separation::{
    HtdemucsOptions, HtdemucsSeparator, SeparationOutputFormat, Stem,
};

let separator = HtdemucsSeparator::new(
    HtdemucsOptions::new("separated")
        .two_stems(Stem::Vocals)
        .output_format(SeparationOutputFormat::Flac),
)?;

let preview = separator.dry_run("song.wav")?;
assert_eq!(preview.result.stems.len(), 2);
```

## Output layout

- Default four-stem models write `vocals`, `drums`, `bass`, and `other`.
- `htdemucs_6s` additionally writes `guitar` and `piano`.
- `two_stems(vocals)` predicts `vocals` and `no_vocals`.
- Outputs are discovered under `output_dir/model_name/...`.

## Package surface

Primary workflow: `audio.separation.expectedStems`.

Workflow operations:

- `audio.separation.expectedStems`: Returns expected separated stem filenames for a model/layout.
- `audio.separation.runDemucs`: Opt-in native/server Demucs execution with setup checks before spawning.

Debug operations:

- `describe`: inspect package metadata and runtime support.
- `audio.separation.models`: Inspects known Demucs model layouts without executing Demucs.
- `audio.separation.plan`: Previews a Demucs command without running an external process.

Runtime support: library, CLI, and server wrappers expose these operations. WASM excludes `audio.separation.runDemucs`.

Run the primary workflow through the CLI:

```bash
cargo run -p moritzbrantner-audio-analysis-separation-cli -- run \
  --operation audio.separation.expectedStems \
  --json '{"format":"wav","model":"htdemucs"}'
```

Successful responses use the shared package-surface shape with `operation`,
`title`, `message`, `summary`, and `result`. Default surface calls are
deterministic, local-first, and do not download models, write persistent files,
or execute external tools unless an operation explicitly documents native or
external-tool execution.

## Related crates

- `audio-analysis-io`
- `audio-analysis-processing`
- `video-analysis-use-cases`
