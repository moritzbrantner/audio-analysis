//! Opt-in Demucs vocal-separation adapter for the karaoke pipeline.
//!
//! External separation stays explicit and caller-owned. This module reuses the
//! repository's typed Demucs wrapper and then feeds the discovered vocal stem
//! into the `audio-io` media adapter.

use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

use audio_analysis_io::SelectedMediaSource;
use audio_analysis_separation::{
    HtdemucsOptions, HtdemucsSeparator, SeparationExecution, SeparationResult, Stem,
};
use audio_contracts::DetectError as PipelineDetectError;
use audio_karaoke_formats::UltraStarV1Metadata;
use media_core::{DetectError as SeparationDetectError, TimedTextWordContract};

use crate::media::{build_ultrastar_from_media, KaraokeMediaPipelineError};
use crate::{KaraokePipelineOptions, KaraokePipelineResult};

/// Error categories retained across validation, Demucs execution, and media analysis.
#[derive(Debug)]
pub enum KaraokeSeparationPipelineError {
    /// Pure karaoke/export request validation failed before external execution.
    Validation(PipelineDetectError),
    /// Demucs planning or execution failed.
    Separation(SeparationDetectError),
    /// Separated/output media decoding or karaoke analysis failed.
    Media(KaraokeMediaPipelineError),
}

impl Display for KaraokeSeparationPipelineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(error) => write!(formatter, "karaoke request validation failed: {error}"),
            Self::Separation(error) => write!(formatter, "karaoke vocal separation failed: {error}"),
            Self::Media(error) => write!(formatter, "karaoke separated-media pipeline failed: {error}"),
        }
    }
}

impl std::error::Error for KaraokeSeparationPipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(error) => Some(error),
            Self::Separation(error) => Some(error),
            Self::Media(error) => Some(error),
        }
    }
}

/// Result returned by Demucs-backed karaoke helpers.
pub type KaraokeSeparationPipelineResult<T> =
    std::result::Result<T, KaraokeSeparationPipelineError>;

/// Completed separation plus the chart/export result produced from its vocal stem.
#[derive(Debug, Clone, PartialEq)]
pub struct KaraokeSeparatedSongResult {
    /// Typed Demucs result retained as execution evidence.
    pub separation: SeparationResult,
    /// Vocal stem path consumed by the media adapter.
    pub vocal_path: PathBuf,
    /// Neutral chart, UltraStar text, diagnostics, and analysis evidence.
    pub karaoke: KaraokePipelineResult,
}

/// Convenience configuration for karaoke: Demucs two-stem vocals + accompaniment.
pub fn karaoke_vocal_separation_options(output_dir: impl Into<PathBuf>) -> HtdemucsOptions {
    HtdemucsOptions::new(output_dir).two_stems(Stem::Vocals)
}

/// Builds the exact Demucs command/result layout without running the external tool.
pub fn plan_vocal_separation(
    input: impl AsRef<Path>,
    options: HtdemucsOptions,
) -> KaraokeSeparationPipelineResult<SeparationExecution> {
    let separator = validated_vocal_separator(options)?;
    separator
        .dry_run(input)
        .map_err(KaraokeSeparationPipelineError::Separation)
}

/// Runs Demucs and returns its typed result after requiring a real vocal stem.
pub fn separate_vocals(
    input: impl AsRef<Path>,
    options: HtdemucsOptions,
) -> KaraokeSeparationPipelineResult<SeparationResult> {
    let separator = validated_vocal_separator(options)?;
    separator
        .separate(input)
        .map_err(KaraokeSeparationPipelineError::Separation)
}

/// Separates vocals from a song path, decodes the song/stem, and builds UltraStar output.
///
/// Pipeline options and UltraStar metadata are validated before Demucs is spawned. The
/// caller owns the separation output directory through `HtdemucsOptions`; this adapter
/// does not create hidden temporary directories or delete separation evidence.
pub fn build_ultrastar_with_demucs(
    input: impl AsRef<Path>,
    lyrics: &[TimedTextWordContract],
    pipeline_options: KaraokePipelineOptions,
    metadata: &UltraStarV1Metadata,
    separation_options: HtdemucsOptions,
) -> KaraokeSeparationPipelineResult<KaraokeSeparatedSongResult> {
    pipeline_options
        .validate()
        .map_err(KaraokeSeparationPipelineError::Validation)?;
    metadata
        .validate()
        .map_err(KaraokeSeparationPipelineError::Validation)?;

    let input = input.as_ref();
    let separation = separate_vocals(input, separation_options)?;
    let vocal_path = required_vocal_path(&separation)?;
    let karaoke = build_ultrastar_from_media(
        Some(SelectedMediaSource::new(input.to_path_buf())),
        SelectedMediaSource::new(vocal_path.clone()),
        lyrics,
        pipeline_options,
        metadata,
    )
    .map_err(KaraokeSeparationPipelineError::Media)?;

    Ok(KaraokeSeparatedSongResult {
        separation,
        vocal_path,
        karaoke,
    })
}

fn validated_vocal_separator(
    options: HtdemucsOptions,
) -> KaraokeSeparationPipelineResult<HtdemucsSeparator> {
    let separator = HtdemucsSeparator::new(options)
        .map_err(KaraokeSeparationPipelineError::Separation)?;
    if !separator.expected_stems().contains(&Stem::Vocals) {
        return Err(KaraokeSeparationPipelineError::Separation(
            SeparationDetectError::InvalidArgument(
                "karaoke separation layout must produce a vocals stem".to_string(),
            ),
        ));
    }
    Ok(separator)
}

fn required_vocal_path(
    separation: &SeparationResult,
) -> KaraokeSeparationPipelineResult<PathBuf> {
    separation
        .stems
        .iter()
        .find(|stem| stem.stem == Stem::Vocals && stem.exists)
        .map(|stem| stem.path.clone())
        .ok_or_else(|| {
            KaraokeSeparationPipelineError::Separation(
                SeparationDetectError::Source(
                    "Demucs result did not contain a non-empty vocals stem".to_string(),
                ),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::KaraokeMediaPipelineResult;
    use crate::KaraokeTempoSource;

    #[test]
    fn default_karaoke_plan_requests_two_stem_vocals_without_execution() {
        let execution = plan_vocal_separation(
            "song.wav",
            karaoke_vocal_separation_options("separated"),
        );
        let Ok(execution) = execution else {
            panic!("valid dry-run separation plan should succeed");
        };

        assert_eq!(
            execution.result.layout.stems(),
            vec![Stem::Vocals, Stem::NoVocals]
        );
        let args = execution
            .command
            .args
            .iter()
            .map(|arg| arg.to_string_lossy())
            .collect::<Vec<_>>();
        assert!(args.windows(2).any(|pair| {
            pair[0].as_ref() == "--two-stems" && pair[1].as_ref() == "vocals"
        }));
    }

    #[test]
    fn rejects_separation_layout_without_vocals_before_execution() {
        let result = plan_vocal_separation(
            "song.wav",
            HtdemucsOptions::new("separated").two_stems(Stem::Drums),
        );
        let Err(KaraokeSeparationPipelineError::Separation(error)) = result else {
            panic!("non-vocal separation layout must fail before execution");
        };
        assert!(error.to_string().contains("vocals stem"));
    }

    #[test]
    fn invalid_pipeline_request_fails_before_demucs_input_is_touched() {
        let mut pipeline_options = KaraokePipelineOptions::default();
        pipeline_options.tempo_source = KaraokeTempoSource::Explicit;
        pipeline_options.pitch_hop_size = 0;

        let result = build_ultrastar_with_demucs(
            "definitely-missing-song.wav",
            &[],
            pipeline_options,
            &UltraStarV1Metadata::new("Song", "Artist", "song.ogg"),
            karaoke_vocal_separation_options("separated"),
        );
        let Err(KaraokeSeparationPipelineError::Validation(error)) = result else {
            panic!("invalid pipeline request must fail before Demucs execution");
        };
        assert!(error.to_string().contains("hop"));
    }

    #[test]
    fn media_result_type_remains_distinct_from_separation_errors() {
        fn accepts_media_result(
            value: KaraokeMediaPipelineResult<KaraokePipelineResult>,
        ) -> KaraokeMediaPipelineResult<KaraokePipelineResult> {
            value
        }

        let error = KaraokeMediaPipelineError::Pipeline(PipelineDetectError::InvalidArgument(
            "analysis".to_string(),
        ));
        let result = accepts_media_result(Err(error));
        assert!(result.is_err());
    }
}
