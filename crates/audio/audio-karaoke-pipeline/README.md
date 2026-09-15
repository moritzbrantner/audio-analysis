# audio-karaoke-pipeline

Song-level orchestration for SingStar/UltraStar-style chart creation using the
existing `audio-analysis` capabilities.

This crate is intentionally an orchestrator rather than another analysis
implementation. It composes:

- `audio-analysis-rhythm` for whole-track tempo/beat evidence from the full mix;
- `audio-analysis-pitch` for monophonic vocal pitch estimates;
- `audio-generation-midi::karaoke` for neutral lyric/pitch fusion and melismas;
- `audio-karaoke-formats` for downstream UltraStar serialization.

The core path is pure, in-memory, and deterministic. `KaraokeAudio` keeps each
decoded mono stream together with its own sample rate, so the full mix used for
rhythm and the vocal stem used for pitch do not need to be resampled to a common
rate. Canonical aligned `media_core::TimedTextWordContract` lyrics remain
caller-owned; the pipeline does not transcribe, realign, or syllabify text.

Tempo can be explicit or estimated from the full mix. Explicit tempo preserves
caller authority and does not require a full-mix PCM input. Estimated tempo
uses the whole-track rhythm analyzer and returns its confidence/evidence; lack
of a selected tempo fails closed instead of silently falling back to 120 BPM.
`beat_zero_seconds` remains caller-owned through the neutral chart options
rather than being guessed from the first onset.

Vocal pitch uses overlapping analysis windows, including zero-padded trailing
windows, but emits non-overlapping hop-sized evidence intervals clipped to the
real samples. This preserves detector context, retains the vocal tail and short
clips, and still satisfies the single-lead chart timing contract.

```rust,ignore
use audio_karaoke_formats::UltraStarV1Metadata;
use audio_karaoke_pipeline::{
    build_ultrastar_from_audio, KaraokeAudio, KaraokePipelineOptions, KaraokeTempoSource,
};

let result = build_ultrastar_from_audio(
    Some(KaraokeAudio::new(&full_mix_samples, 44_100)),
    KaraokeAudio::new(&vocal_samples, 48_000),
    &aligned_lyrics,
    KaraokePipelineOptions {
        tempo_source: KaraokeTempoSource::AnalyzeFullMix,
        ..KaraokePipelineOptions::default()
    },
    &UltraStarV1Metadata::new("Song", "Artist", "song.ogg"),
)?;

let _ = (result.chart, result.ultrastar_text, result.evidence);
# Ok::<(), audio_contracts::DetectError>(())
```

`build_ultrastar_from_samples` remains as a compatibility helper for callers
whose full mix and vocal PCM already share one sample rate.

## Media input adapter

Enable the `audio-io` feature to decode finite files/containers through the
repository-owned `audio-analysis-io` boundary. `media::build_ultrastar_from_media`
accepts `SelectedMediaSource` values, so callers can explicitly select an audio
stream in a container. Decoding uses recorded-input defaults and equal channel
averaging.

The vocal source is always decoded. The full mix is decoded only when
`KaraokeTempoSource::AnalyzeFullMix` is selected; explicit tempo can omit it.
Decoded streams retain independent sample rates and then enter the same pure
`build_ultrastar_from_audio` path. `KaraokeMediaPipelineError` keeps media I/O
failures distinct from analysis/export failures instead of flattening errors to
strings.

Pure pipeline options, output metadata, and required-source presence are
validated before the first decode so invalid requests cannot trigger expensive
FFmpeg work or have their validation errors masked by media failures.

```rust,ignore
use audio_analysis_io::SelectedMediaSource;
use audio_karaoke_formats::UltraStarV1Metadata;
use audio_karaoke_pipeline::{media::build_ultrastar_from_media, KaraokePipelineOptions};

let result = build_ultrastar_from_media(
    Some(SelectedMediaSource::new("song.flac")),
    SelectedMediaSource::new("separated/vocals.wav"),
    &aligned_lyrics,
    KaraokePipelineOptions::default(),
    &UltraStarV1Metadata::new("Song", "Artist", "song.flac"),
)?;
# Ok::<(), audio_karaoke_pipeline::media::KaraokeMediaPipelineError>(())
```

## Vocal separation adapter

Enable the `separation` feature for the explicit Demucs stage. It includes the
`audio-io` feature and reuses `audio-analysis-separation::HtdemucsSeparator`;
this crate does not build another command wrapper or duplicate Demucs output
layout rules. Because separation is a file/media concern, the API lives under
`media::separation` rather than the pure analysis root.

`media::separation::karaoke_vocal_separation_options(output_dir)` creates the
efficient karaoke default: Demucs two-stem output with `vocals` and
`no_vocals`. Callers may supply any `HtdemucsOptions`, but the adapter rejects
layouts that cannot produce `Stem::Vocals` before external execution.
`plan_vocal_separation` returns the exact command and expected output layout
without invoking Demucs.

`build_ultrastar_with_demucs` validates karaoke options and UltraStar metadata,
runs the caller-configured separator, retains the complete typed
`SeparationResult`, finds the non-empty vocal stem, and passes that stem plus the
original song through the same media adapter above. The caller owns the
separation output directory and its lifecycle; no hidden temporary directory is
created or deleted.

```rust,ignore
use audio_karaoke_formats::UltraStarV1Metadata;
use audio_karaoke_pipeline::{
    media::separation::{build_ultrastar_with_demucs, karaoke_vocal_separation_options},
    KaraokePipelineOptions,
};

let result = build_ultrastar_with_demucs(
    "song.flac",
    &aligned_lyrics,
    KaraokePipelineOptions::default(),
    &UltraStarV1Metadata::new("Song", "Artist", "song.flac"),
    karaoke_vocal_separation_options("separated"),
)?;

let _ = (result.separation, result.vocal_path, result.karaoke);
# Ok::<(), audio_karaoke_pipeline::media::separation::KaraokeSeparationPipelineError>(())
```

Transcription/alignment remains a later, separate stage. Separation is explicit
external-tool execution, while lyric timing remains canonical caller input; the
pipeline still never invents lyrics or syllable boundaries.
