//! File/media adapter for the pure karaoke analysis pipeline.
//!
//! This module is intentionally feature-gated because it owns FFmpeg-backed
//! decoding policy. The neutral pipeline remains pure and in-memory.

use std::fmt::{Display, Formatter};

use audio_analysis_io::{
    decode_selected_media_to_mono_f32, AudioInputOptions, AudioIoError, ChannelMix,
    SelectedMediaSource,
};
use audio_contracts::DetectError;
use audio_karaoke_formats::UltraStarV1Metadata;
use media_core::TimedTextWordContract;

use crate::{
    build_ultrastar_from_audio, KaraokeAudio, KaraokePipelineOptions, KaraokePipelineResult,
    KaraokeTempoSource,
};

/// Errors produced while decoding media and running the karaoke pipeline.
#[derive(Debug)]
pub enum KaraokeMediaPipelineError {
    /// FFmpeg/source decoding failed before analysis.
    AudioIo(AudioIoError),
    /// Decoded audio reached the pure pipeline but analysis/export failed.
    Pipeline(DetectError),
}

impl Display for KaraokeMediaPipelineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AudioIo(error) => write!(formatter, "karaoke media decode failed: {error}"),
            Self::Pipeline(error) => write!(formatter, "karaoke analysis failed: {error}"),
        }
    }
}

impl std::error::Error for KaraokeMediaPipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AudioIo(error) => Some(error),
            Self::Pipeline(error) => Some(error),
        }
    }
}

impl From<AudioIoError> for KaraokeMediaPipelineError {
    fn from(error: AudioIoError) -> Self {
        Self::AudioIo(error)
    }
}

impl From<DetectError> for KaraokeMediaPipelineError {
    fn from(error: DetectError) -> Self {
        Self::Pipeline(error)
    }
}

/// Result returned by the FFmpeg-backed media adapter.
pub type KaraokeMediaPipelineResult<T> = std::result::Result<T, KaraokeMediaPipelineError>;

/// Decodes selected media streams and runs the pure karaoke pipeline.
///
/// `vocal_source` is always decoded because pitch analysis is vocal-only. The
/// full mix is decoded only when `AnalyzeFullMix` tempo is selected; callers
/// using explicit tempo may omit it entirely. Each decoded stream keeps its own
/// sample rate, so a separator-produced vocal stem does not need to match the
/// source song's sample rate.
///
/// Decoding uses the repository's recorded-input defaults and equal channel
/// averaging. Audio stream selection stays caller-owned through
/// [`SelectedMediaSource::audio_stream_index`].
pub fn build_ultrastar_from_media(
    full_mix_source: Option<SelectedMediaSource>,
    vocal_source: SelectedMediaSource,
    lyrics: &[TimedTextWordContract],
    options: KaraokePipelineOptions,
    metadata: &UltraStarV1Metadata,
) -> KaraokeMediaPipelineResult<KaraokePipelineResult> {
    let (vocal_metadata, vocal_samples) = decode_selected_media_to_mono_f32(
        vocal_source,
        AudioInputOptions::recorded(),
        ChannelMix::Average,
    )?;
    let vocals = KaraokeAudio::new(&vocal_samples, vocal_metadata.sample_rate);

    let decoded_full_mix = match options.tempo_source {
        KaraokeTempoSource::Explicit => None,
        KaraokeTempoSource::AnalyzeFullMix => {
            let source = full_mix_source.ok_or_else(|| {
                KaraokeMediaPipelineError::Pipeline(DetectError::InvalidArgument(
                    "full_mix_source is required when tempo_source is AnalyzeFullMix".to_string(),
                ))
            })?;
            Some(decode_selected_media_to_mono_f32(
                source,
                AudioInputOptions::recorded(),
                ChannelMix::Average,
            )?)
        }
    };

    let full_mix = decoded_full_mix
        .as_ref()
        .map(|(metadata, samples)| KaraokeAudio::new(samples, metadata.sample_rate));

    build_ultrastar_from_audio(full_mix, vocals, lyrics, options, metadata).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_error_preserves_pipeline_category() {
        let error = KaraokeMediaPipelineError::Pipeline(DetectError::InvalidArgument(
            "missing full mix".to_string(),
        ));
        assert!(error.to_string().contains("karaoke analysis failed"));
        assert!(error.source().is_some());
    }
}
