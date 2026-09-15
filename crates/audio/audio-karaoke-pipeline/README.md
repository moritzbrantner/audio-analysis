# audio-karaoke-pipeline

Song-level orchestration for SingStar/UltraStar-style chart creation using the
existing `audio-analysis` capabilities.

This crate is intentionally an orchestrator rather than another analysis
implementation. It composes:

- `audio-analysis-rhythm` for whole-track tempo/beat evidence from the full mix;
- `audio-analysis-pitch` for monophonic vocal pitch estimates;
- `audio-generation-midi::karaoke` for neutral lyric/pitch fusion and melismas;
- `audio-karaoke-formats` for downstream UltraStar serialization.

The first slice is pure, in-memory, and deterministic. Callers provide decoded
mono full-mix samples, decoded mono vocal samples, and canonical aligned
`media_core::TimedTextWordContract` lyrics. File decoding, Demucs execution,
and transcription/alignment remain explicit adapter concerns for later slices.
This keeps external tools and filesystem policy out of the reusable analysis
core.

Tempo can be explicit or estimated from the full mix. Explicit tempo preserves
caller authority. Estimated tempo uses the whole-track rhythm analyzer and
returns its confidence/evidence; lack of a selected tempo fails closed instead
of silently falling back to 120 BPM. `beat_zero_seconds` remains caller-owned
through the neutral chart options rather than being guessed from the first
onset.

Vocal pitch uses overlapping analysis windows but emits non-overlapping
hop-sized evidence intervals into the neutral chart builder. This preserves the
pitch detector's analysis context while satisfying the single-lead chart timing
contract.

```rust,ignore
use audio_karaoke_formats::UltraStarV1Metadata;
use audio_karaoke_pipeline::{
    build_ultrastar_from_samples, KaraokePipelineOptions, KaraokeTempoSource,
};

let result = build_ultrastar_from_samples(
    &full_mix_samples,
    &vocal_samples,
    48_000,
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
