//! File/media adapter for the pure karaoke analysis pipeline.
//!
//! This module is intentionally feature-gated because it owns FFmpeg-backed
//! decoding policy. The neutral pipeline remains pure and in-memory.

#[cfg(feature = "separation")]
pub mod separation;

use std::fmt::{Display, Formatter};

use audio_analysis_io::{
    decode_selected_media_to_mono_f32, AudioInputOptions, AudioIoError, ChannelMix,
    SelectedMediaSource,
};
use audio_contracts::DetectError;
use audio_karaoke_formats::UltraStarV1Metadata;
use media_core::{TimedTextWordContract, TranscriptionContract};

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

/// Extracts canonical aligned lyric words from a transcription contract.
///
/// `audio-analysis-transcription` owns ASR and forced-alignment execution. This
/// adapter consumes only its shared `media-core` contract, preserving that
/// authority boundary and avoiding a second transcription configuration surface.
/// Segment-level timing is deliberately insufficient for karaoke generation:
/// every retained word must have a complete, positive-duration time range.
pub fn aligned_lyrics_from_transcription(
    transcription: &TranscriptionContract,
) -> KaraokeMediaPipelineResult<Vec<TimedTextWordContract>> {
    transcription.validate_strict().map_err(|error| {
        invalid_transcription(format!("invalid canonical transcription contract: {error}"))
    })?;

    let mut lyrics = Vec::new();
    for segment in &transcription.segments {
        let words = segment.words();
        if words.is_empty() {
            if !segment.text.trim().is_empty() {
                return Err(invalid_transcription(format!(
                    "transcription segment {} has text but no word-level alignment",
                    segment.index
                )));
            }
            continue;
        }

        for (word_index, word) in words.iter().enumerate() {
            if word.text.trim().is_empty() {
                return Err(invalid_transcription(format!(
                    "transcription segment {} word {} has empty text",
                    segment.index, word_index
                )));
            }
            let start_seconds = word.start_seconds().ok_or_else(|| {
                invalid_transcription(format!(
                    "transcription segment {} word {} is missing a start time",
                    segment.index, word_index
                ))
            })?;
            let end_seconds = word.end_seconds().ok_or_else(|| {
                invalid_transcription(format!(
                    "transcription segment {} word {} is missing an end time",
                    segment.index, word_index
                ))
            })?;
            if end_seconds <= start_seconds {
                return Err(invalid_transcription(format!(
                    "transcription segment {} word {} must have positive duration",
                    segment.index, word_index
                )));
            }
            lyrics.push(word.clone());
        }
    }

    if lyrics.is_empty() {
        return Err(invalid_transcription(
            "transcription contains no word-level aligned lyrics",
        ));
    }

    Ok(lyrics)
}

/// Decodes selected media streams and runs the pure karaoke pipeline.
///
/// `vocal_source` is always decoded because pitch analysis is vocal-only. The
/// full mix is decoded only when `AnalyzeFullMix` tempo is selected; callers
/// using explicit tempo may omit it entirely. Each decoded stream keeps its own
/// sample rate, so a separator-produced vocal stem does not need to match the
/// source song's sample rate.
///
/// Pure request validation runs before the first media decode so invalid
/// analysis options, invalid output metadata, or a missing required full-mix
/// source cannot trigger expensive FFmpeg work or be masked by a decode error.
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
    validate_media_request(&full_mix_source, options, metadata)?;

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

/// Validates canonical transcription timing, decodes selected media, and builds UltraStar output.
///
/// The transcription contract is checked before the first FFmpeg decode. This
/// function never derives word timing from segment timing and never syllabifies
/// transcript text; callers must run the transcription/alignment authority first.
pub fn build_ultrastar_from_media_transcription(
    full_mix_source: Option<SelectedMediaSource>,
    vocal_source: SelectedMediaSource,
    transcription: &TranscriptionContract,
    options: KaraokePipelineOptions,
    metadata: &UltraStarV1Metadata,
) -> KaraokeMediaPipelineResult<KaraokePipelineResult> {
    validate_media_request(&full_mix_source, options, metadata)?;
    let lyrics = aligned_lyrics_from_transcription(transcription)?;
    build_ultrastar_from_media(full_mix_source, vocal_source, &lyrics, options, metadata)
}

fn validate_media_request(
    full_mix_source: &Option<SelectedMediaSource>,
    options: KaraokePipelineOptions,
    metadata: &UltraStarV1Metadata,
) -> KaraokeMediaPipelineResult<()> {
    options.validate()?;
    metadata.validate()?;
    if options.tempo_source == KaraokeTempoSource::AnalyzeFullMix && full_mix_source.is_none() {
        return Err(KaraokeMediaPipelineError::Pipeline(
            DetectError::InvalidArgument(
                "full_mix_source is required when tempo_source is AnalyzeFullMix".to_string(),
            ),
        ));
    }
    Ok(())
}

fn invalid_transcription(message: impl Into<String>) -> KaraokeMediaPipelineError {
    KaraokeMediaPipelineError::Pipeline(DetectError::InvalidArgument(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_core::TimedTextSegmentContract;

    fn metadata() -> UltraStarV1Metadata {
        UltraStarV1Metadata::new("Song", "Artist", "song.ogg")
    }

    fn aligned_transcription() -> TranscriptionContract {
        let word = TimedTextWordContract::new("hello")
            .with_time_range(Some(0.25), Some(1.5))
            .expect("word timing must be valid");
        let mut segment = TimedTextSegmentContract::new(0, "hello")
            .with_time_range(Some(0.25), Some(1.5))
            .expect("segment timing must be valid");
        segment
            .push_word(word)
            .expect("word must fit inside segment timing");
        TranscriptionContract {
            text: Some("hello".to_string()),
            segments: vec![segment],
            ..TranscriptionContract::default()
        }
    }

    #[test]
    fn media_error_preserves_pipeline_category() {
        let error = KaraokeMediaPipelineError::Pipeline(DetectError::InvalidArgument(
            "missing full mix".to_string(),
        ));
        assert!(error.to_string().contains("karaoke analysis failed"));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn transcription_adapter_preserves_canonical_aligned_words() {
        let transcription = aligned_transcription();
        let lyrics = aligned_lyrics_from_transcription(&transcription)
            .expect("word-aligned transcription should be accepted");

        assert_eq!(lyrics.len(), 1);
        assert_eq!(lyrics[0], transcription.segments[0].words()[0]);
    }

    #[test]
    fn transcription_adapter_rejects_segment_only_timing() {
        let segment = TimedTextSegmentContract::new(0, "hello")
            .with_time_range(Some(0.25), Some(1.5))
            .expect("segment timing must be valid");
        let transcription = TranscriptionContract {
            text: Some("hello".to_string()),
            segments: vec![segment],
            ..TranscriptionContract::default()
        };

        let error = aligned_lyrics_from_transcription(&transcription)
            .expect_err("segment timing must not be promoted to word timing");
        assert!(error.to_string().contains("word-level alignment"));
    }

    #[test]
    fn transcription_validation_fails_before_attempting_media_decode() {
        let segment = TimedTextSegmentContract::new(0, "hello")
            .with_time_range(Some(0.25), Some(1.5))
            .expect("segment timing must be valid");
        let transcription = TranscriptionContract {
            text: Some("hello".to_string()),
            segments: vec![segment],
            ..TranscriptionContract::default()
        };
        let mut options = KaraokePipelineOptions::default();
        options.tempo_source = KaraokeTempoSource::Explicit;

        let result = build_ultrastar_from_media_transcription(
            None,
            SelectedMediaSource::new("definitely-missing-vocals.wav"),
            &transcription,
            options,
            &metadata(),
        );
        let Err(KaraokeMediaPipelineError::Pipeline(error)) = result else {
            panic!("invalid aligned lyrics must fail before media decoding");
        };
        assert!(error.to_string().contains("word-level alignment"));
    }

    #[test]
    fn invalid_options_fail_before_attempting_vocal_decode() {
        let mut options = KaraokePipelineOptions::default();
        options.tempo_source = KaraokeTempoSource::Explicit;
        options.pitch_hop_size = 0;

        let result = build_ultrastar_from_media(
            None,
            SelectedMediaSource::new("definitely-missing-vocals.wav"),
            &[],
            options,
            &metadata(),
        );
        let Err(KaraokeMediaPipelineError::Pipeline(error)) = result else {
            panic!("invalid options must fail as a pipeline error before media decoding");
        };
        assert!(error.to_string().contains("hop"));
    }

    #[test]
    fn missing_required_full_mix_fails_before_attempting_vocal_decode() {
        let result = build_ultrastar_from_media(
            None,
            SelectedMediaSource::new("definitely-missing-vocals.wav"),
            &[],
            KaraokePipelineOptions::default(),
            &metadata(),
        );
        let Err(KaraokeMediaPipelineError::Pipeline(error)) = result else {
            panic!("missing full mix must fail before media decoding");
        };
        assert!(error.to_string().contains("full_mix_source"));
    }
}
