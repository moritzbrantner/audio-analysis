//! Transcript speaker assignment and speaker prediction normalization.

use std::collections::BTreeMap;

use audio_analysis_recognition::{TranscriptSegmentContract, TranscriptionContract};
use audio_contracts::{DetectError, Result};

use crate::{SpeakerDiarizationResponse, SpeakerSegmentPrediction};

/// Speaker assignment policy for transcript segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SpeakerTranscriptAssignmentPolicy {
    /// Assign the speaker with the largest positive time overlap.
    #[default]
    Majority,
    /// Assign the speaker whose segment start is nearest the transcript start.
    NearestStart,
    /// Assign only when the diarization segment fully contains the transcript segment.
    StrictContained,
}

/// Backward-compatible alias for the speaker transcript assignment policy.
pub type TranscriptSpeakerOverlapPolicy = SpeakerTranscriptAssignmentPolicy;

/// Assigns speaker diarization output onto transcript segments by time overlap.
pub fn assign_speakers_to_transcript(
    transcript: &TranscriptionContract,
    diarization: &SpeakerDiarizationResponse,
) -> Result<TranscriptionContract> {
    assign_speakers_to_transcript_with_policy(
        transcript,
        diarization,
        SpeakerTranscriptAssignmentPolicy::Majority,
    )
}

/// Assigns speaker diarization output onto transcript words and segments using an explicit policy.
pub fn assign_speakers_to_transcript_with_policy(
    transcript: &TranscriptionContract,
    diarization: &SpeakerDiarizationResponse,
    policy: SpeakerTranscriptAssignmentPolicy,
) -> Result<TranscriptionContract> {
    let diarization_segments = diarization
        .segments
        .iter()
        .cloned()
        .map(normalize_speaker_segment)
        .collect::<Result<Vec<_>>>()?;
    let mut assigned_segments = Vec::with_capacity(transcript.segments.len());
    for source_segment in &transcript.segments {
        let mut segment =
            TranscriptSegmentContract::new(source_segment.index, source_segment.text.trim())
                .with_time_range(source_segment.start_seconds(), source_segment.end_seconds())
                .map_err(|error| DetectError::InvalidArgument(error.to_string()))?
                .with_confidence(source_segment.confidence())
                .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
        segment.language = normalize_optional_string(source_segment.language.clone());
        segment.speaker = normalize_optional_string(source_segment.speaker.clone());
        segment.is_final = source_segment.is_final;
        segment.attributes = source_segment.attributes.clone();

        for source_word in source_segment.words() {
            let text = source_word.text.trim();
            if text.is_empty() {
                continue;
            }
            let mut word = source_word.clone();
            word.text = text.to_string();
            word.speaker = match (source_word.start_seconds(), source_word.end_seconds()) {
                (Some(start), Some(end)) => {
                    speaker_for_range(start, end, &diarization_segments, policy)
                        .map(|speaker_segment| speaker_segment.speaker.clone())
                }
                _ => normalize_optional_string(source_word.speaker.clone()),
            };
            word.attributes = source_word.attributes.clone();
            segment
                .push_word(word)
                .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
        }
        for character in source_segment.chars() {
            if character.character.is_empty() {
                continue;
            }
            segment
                .push_char(character.clone())
                .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
        }
        if segment.speaker.is_some() {
            assigned_segments.push(segment);
            continue;
        }
        if !segment.words().is_empty() {
            segment.speaker = majority_word_speaker(&segment);
            assigned_segments.push(segment);
            continue;
        }
        let Some(best) = best_speaker_match(&segment, &diarization_segments, policy) else {
            if policy == SpeakerTranscriptAssignmentPolicy::StrictContained
                && segment.speaker.is_none()
            {
                segment.speaker = Some("unknown".to_string());
            }
            assigned_segments.push(segment);
            continue;
        };
        segment.speaker = Some(best.speaker.clone());
        if let Some(score) = best.score {
            segment
                .attributes
                .insert("speakerScore".to_string(), score.to_string());
        }
        assigned_segments.push(segment);
    }

    let mut transcript = TranscriptionContract {
        text: normalize_optional_string(transcript.text.clone()),
        language: normalize_optional_string(transcript.language.clone()),
        segments: assigned_segments,
        source: normalize_optional_string(transcript.source.clone()),
        attributes: transcript.attributes.clone(),
    };
    if transcript.text.is_none() {
        let joined = transcript
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !joined.is_empty() {
            transcript.text = Some(joined);
        }
    }
    transcript
        .validate_strict()
        .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
    Ok(transcript)
}

fn majority_word_speaker(segment: &TranscriptSegmentContract) -> Option<String> {
    let mut counts = BTreeMap::<String, usize>::new();
    for speaker in segment
        .words()
        .iter()
        .filter_map(|word| word.speaker.as_ref())
    {
        *counts.entry(speaker.clone()).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(speaker, _)| speaker)
}

pub(crate) fn normalize_speaker_segment(
    mut segment: SpeakerSegmentPrediction,
) -> Result<SpeakerSegmentPrediction> {
    segment.speaker = segment.speaker.trim().to_string();
    if segment.speaker.is_empty() {
        return Err(DetectError::InvalidArgument(
            "speaker label must not be empty".to_string(),
        ));
    }
    if !segment.start_seconds.is_finite() || !segment.end_seconds.is_finite() {
        return Err(DetectError::InvalidArgument(
            "speaker segment timestamps must be finite".to_string(),
        ));
    }
    if segment.end_seconds < segment.start_seconds {
        return Err(DetectError::InvalidArgument(
            "speaker segment end_seconds must be greater than or equal to start_seconds"
                .to_string(),
        ));
    }
    segment.score = segment
        .score
        .and_then(|score| score.is_finite().then(|| score.clamp(0.0, 1.0)));
    Ok(segment)
}

fn best_speaker_match<'a>(
    transcript_segment: &TranscriptSegmentContract,
    diarization_segments: &'a [SpeakerSegmentPrediction],
    policy: SpeakerTranscriptAssignmentPolicy,
) -> Option<&'a SpeakerSegmentPrediction> {
    let (Some(start), Some(end)) = (
        transcript_segment.start_seconds(),
        transcript_segment.end_seconds(),
    ) else {
        return None;
    };
    speaker_for_range(start, end, diarization_segments, policy)
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn speaker_for_range(
    start: f64,
    end: f64,
    diarization_segments: &[SpeakerSegmentPrediction],
    policy: SpeakerTranscriptAssignmentPolicy,
) -> Option<&SpeakerSegmentPrediction> {
    match policy {
        SpeakerTranscriptAssignmentPolicy::Majority => {
            best_majority_speaker_match(start, end, (start + end) / 2.0, diarization_segments)
        }
        SpeakerTranscriptAssignmentPolicy::NearestStart => {
            nearest_start_speaker_match(start, diarization_segments)
        }
        SpeakerTranscriptAssignmentPolicy::StrictContained => {
            strict_contained_speaker_match(start, end, diarization_segments)
        }
    }
}

fn best_majority_speaker_match(
    transcript_start: f64,
    transcript_end: f64,
    transcript_midpoint: f64,
    diarization_segments: &[SpeakerSegmentPrediction],
) -> Option<&SpeakerSegmentPrediction> {
    let positive_overlap = diarization_segments
        .iter()
        .filter_map(|candidate| {
            let overlap = speaker_overlap_seconds(transcript_start, transcript_end, candidate);
            (overlap > 0.0).then_some((candidate, overlap))
        })
        .max_by(compare_speaker_candidates);
    if let Some((candidate, _)) = positive_overlap {
        return Some(candidate);
    }

    diarization_segments
        .iter()
        .filter(|candidate| {
            let start = candidate.start_seconds as f64;
            let end = candidate.end_seconds as f64;
            start <= transcript_midpoint && transcript_midpoint <= end
        })
        .map(|candidate| (candidate, 0.0))
        .max_by(compare_speaker_candidates)
        .map(|(candidate, _)| candidate)
}

fn nearest_start_speaker_match(
    transcript_start: f64,
    diarization_segments: &[SpeakerSegmentPrediction],
) -> Option<&SpeakerSegmentPrediction> {
    diarization_segments
        .iter()
        .map(|candidate| {
            (
                candidate,
                (candidate.start_seconds as f64 - transcript_start).abs(),
            )
        })
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| speaker_score(right.0).total_cmp(&speaker_score(left.0)))
                .then_with(|| left.0.start_seconds.total_cmp(&right.0.start_seconds))
                .then_with(|| left.0.speaker.cmp(&right.0.speaker))
        })
        .map(|(candidate, _)| candidate)
}

fn strict_contained_speaker_match(
    transcript_start: f64,
    transcript_end: f64,
    diarization_segments: &[SpeakerSegmentPrediction],
) -> Option<&SpeakerSegmentPrediction> {
    diarization_segments
        .iter()
        .filter(|candidate| {
            candidate.start_seconds as f64 <= transcript_start
                && candidate.end_seconds as f64 >= transcript_end
        })
        .map(|candidate| (candidate, (transcript_end - transcript_start).max(0.0)))
        .max_by(compare_speaker_candidates)
        .map(|(candidate, _)| candidate)
}

fn speaker_overlap_seconds(
    transcript_start: f64,
    transcript_end: f64,
    speaker_segment: &SpeakerSegmentPrediction,
) -> f64 {
    let start = transcript_start.max(speaker_segment.start_seconds as f64);
    let end = transcript_end.min(speaker_segment.end_seconds as f64);
    (end - start).max(0.0)
}

fn compare_speaker_candidates(
    left: &(&SpeakerSegmentPrediction, f64),
    right: &(&SpeakerSegmentPrediction, f64),
) -> std::cmp::Ordering {
    left.1
        .total_cmp(&right.1)
        .then_with(|| speaker_score(left.0).total_cmp(&speaker_score(right.0)))
        .then_with(|| right.0.start_seconds.total_cmp(&left.0.start_seconds))
        .then_with(|| right.0.speaker.cmp(&left.0.speaker))
}

fn speaker_score(segment: &SpeakerSegmentPrediction) -> f32 {
    segment.score.unwrap_or(f32::NEG_INFINITY)
}
