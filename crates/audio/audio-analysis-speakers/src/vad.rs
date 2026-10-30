//! Voice activity detection shared by transcription chunking and diarization.

use audio_analysis_core::rms;
use audio_contracts::{DetectError, Result};
use serde::{Deserialize, Serialize};

use crate::SpeakerAudio;

/// Trait for voice activity detector implementations.
pub trait VoiceActivityDetector {
    /// Stable provider identifier reported in pipeline events.
    fn provider_id(&self) -> &str;

    /// Detects speech spans in `audio`.
    ///
    /// Detectors that own their configuration (for example model-backed ones)
    /// may ignore option fields that do not apply to them.
    fn detect_speech(
        &mut self,
        audio: &SpeakerAudio<'_>,
        options: &VadOptions,
    ) -> Result<VadResponse>;
}

/// Speech spans detected by a [`VoiceActivityDetector`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VadResponse {
    /// Detected speech spans in chronological order.
    pub segments: Vec<SpeechSpan>,
    /// Provider diagnostics.
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

/// Speech activity span.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSpan {
    /// Start in seconds.
    pub start_seconds: f64,
    /// End in seconds.
    pub end_seconds: f64,
    /// Speech confidence or energy score.
    pub score: f32,
}

impl SpeechSpan {
    /// Creates a validated span.
    pub fn new(start_seconds: f64, end_seconds: f64, score: f32) -> Result<Self> {
        let span = Self {
            start_seconds,
            end_seconds,
            score,
        };
        span.validate()?;
        Ok(span)
    }

    /// Returns duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.end_seconds - self.start_seconds
    }

    /// Validates finite, non-negative, positive-duration values.
    pub fn validate(&self) -> Result<()> {
        if !self.start_seconds.is_finite()
            || !self.end_seconds.is_finite()
            || !self.score.is_finite()
        {
            return Err(DetectError::InvalidArgument(
                "speech span values must be finite".to_string(),
            ));
        }
        if self.start_seconds < 0.0 || self.end_seconds <= self.start_seconds {
            return Err(DetectError::InvalidArgument(
                "speech span must have non-negative start and positive duration".to_string(),
            ));
        }
        if self.score < 0.0 {
            return Err(DetectError::InvalidArgument(
                "speech span score must be non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

/// Voice activity detection options.
///
/// The defaults are the transcription chunking defaults. Diarization uses
/// [`VadOptions::unpadded`], which keeps spans at their detected bounds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VadOptions {
    /// RMS threshold at or above which a frame is speech.
    #[serde(default = "default_rms_threshold")]
    pub rms_threshold: f32,
    /// Frame duration in seconds.
    #[serde(default = "default_frame_seconds")]
    pub frame_seconds: f64,
    /// Hop duration in seconds.
    #[serde(default = "default_hop_seconds")]
    pub hop_seconds: f64,
    /// Minimum accepted speech span duration.
    #[serde(default = "default_min_speech_seconds")]
    pub min_speech_seconds: f64,
    /// Padding added to both sides of each merged span.
    #[serde(default = "default_padding_seconds")]
    pub padding_seconds: f64,
    /// Merge speech spans separated by no more than this gap.
    #[serde(default = "default_merge_gap_seconds")]
    pub merge_gap_seconds: f64,
    /// ASR chunk size: split spans longer than this; `None` keeps spans whole.
    /// The transcription pipeline also merges adjacent spans up to this size.
    #[serde(default = "default_max_chunk_seconds")]
    pub max_chunk_seconds: Option<f64>,
}

fn default_rms_threshold() -> f32 {
    0.01
}
fn default_frame_seconds() -> f64 {
    0.03
}
fn default_hop_seconds() -> f64 {
    0.01
}
fn default_min_speech_seconds() -> f64 {
    0.08
}
fn default_padding_seconds() -> f64 {
    0.02
}
fn default_merge_gap_seconds() -> f64 {
    0.05
}
fn default_max_chunk_seconds() -> Option<f64> {
    Some(30.0)
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            rms_threshold: default_rms_threshold(),
            frame_seconds: default_frame_seconds(),
            hop_seconds: default_hop_seconds(),
            min_speech_seconds: default_min_speech_seconds(),
            padding_seconds: default_padding_seconds(),
            merge_gap_seconds: default_merge_gap_seconds(),
            max_chunk_seconds: default_max_chunk_seconds(),
        }
    }
}

impl VadOptions {
    /// Default options without padding or max-chunk splitting, used by diarization.
    pub fn unpadded() -> Self {
        Self {
            padding_seconds: 0.0,
            max_chunk_seconds: None,
            ..Self::default()
        }
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if !self.rms_threshold.is_finite() || self.rms_threshold < 0.0 {
            return Err(DetectError::InvalidArgument(
                "VAD RMS threshold must be finite and non-negative".to_string(),
            ));
        }
        for (name, value, positive) in [
            ("frameSeconds", self.frame_seconds, true),
            ("hopSeconds", self.hop_seconds, true),
            ("minSpeechSeconds", self.min_speech_seconds, false),
            ("paddingSeconds", self.padding_seconds, false),
            ("mergeGapSeconds", self.merge_gap_seconds, false),
            (
                "maxChunkSeconds",
                self.max_chunk_seconds.unwrap_or(f64::MAX),
                true,
            ),
        ] {
            if !value.is_finite() || (positive && value <= 0.0) || (!positive && value < 0.0) {
                return Err(DetectError::InvalidArgument(format!(
                    "VAD option `{name}` has an invalid value"
                )));
            }
        }
        Ok(())
    }
}

/// Deterministic energy/RMS voice activity detector.
///
/// Frames the mono mix, keeps frames whose RMS reaches the threshold, merges
/// gaps, pads, drops short spans and splits spans longer than the max chunk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnergyVoiceActivityDetector;

impl VoiceActivityDetector for EnergyVoiceActivityDetector {
    fn provider_id(&self) -> &str {
        "energy-vad"
    }

    fn detect_speech(
        &mut self,
        audio: &SpeakerAudio<'_>,
        options: &VadOptions,
    ) -> Result<VadResponse> {
        Ok(VadResponse {
            segments: energy_speech_spans(audio, options)?,
            diagnostics: vec!["deterministic energy VAD completed".to_string()],
        })
    }
}

fn energy_speech_spans(audio: &SpeakerAudio<'_>, options: &VadOptions) -> Result<Vec<SpeechSpan>> {
    options.validate()?;
    let samples = audio.to_mono()?;
    let sample_rate = audio.sample_rate() as f64;
    let frame_size = seconds_to_samples(options.frame_seconds, audio.sample_rate());
    let hop_size = seconds_to_samples(options.hop_seconds, audio.sample_rate());
    let mut active = Vec::new();
    let mut start = 0;
    while start < samples.len() {
        let end = (start + frame_size).min(samples.len());
        let score = rms(&samples[start..end]);
        if score >= options.rms_threshold {
            active.push(SpeechSpan::new(
                start as f64 / sample_rate,
                end as f64 / sample_rate,
                score,
            )?);
        }
        if start + hop_size >= samples.len() {
            break;
        }
        start += hop_size;
    }
    let duration = audio.duration_seconds();
    let mut merged = merge_spans(active, options.merge_gap_seconds)?;
    if options.padding_seconds > 0.0 {
        for span in &mut merged {
            span.start_seconds = (span.start_seconds - options.padding_seconds).max(0.0);
            span.end_seconds = (span.end_seconds + options.padding_seconds).min(duration);
        }
        merged = merge_spans(merged, options.merge_gap_seconds)?;
    }
    Ok(merged
        .into_iter()
        .filter(|span| span.duration_seconds() >= options.min_speech_seconds)
        .flat_map(|span| split_max_chunk(span, options.max_chunk_seconds))
        .collect())
}

fn seconds_to_samples(seconds: f64, sample_rate: u32) -> usize {
    (seconds * sample_rate as f64).round().max(1.0) as usize
}

fn merge_spans(mut spans: Vec<SpeechSpan>, merge_gap_seconds: f64) -> Result<Vec<SpeechSpan>> {
    spans.sort_by(|left, right| left.start_seconds.total_cmp(&right.start_seconds));
    let mut merged: Vec<SpeechSpan> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut() {
            if span.start_seconds - last.end_seconds <= merge_gap_seconds {
                last.end_seconds = last.end_seconds.max(span.end_seconds);
                last.score = last.score.max(span.score);
                continue;
            }
        }
        merged.push(span);
    }
    for span in &merged {
        span.validate()?;
    }
    Ok(merged)
}

fn split_max_chunk(span: SpeechSpan, max_chunk_seconds: Option<f64>) -> Vec<SpeechSpan> {
    let Some(max_chunk_seconds) = max_chunk_seconds else {
        return vec![span];
    };
    if span.duration_seconds() <= max_chunk_seconds {
        return vec![span];
    }
    let mut chunks = Vec::new();
    let mut start = span.start_seconds;
    while start < span.end_seconds {
        let end = (start + max_chunk_seconds).min(span.end_seconds);
        if end > start {
            chunks.push(SpeechSpan {
                start_seconds: start,
                end_seconds: end,
                score: span.score,
            });
        }
        start = end;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect(samples: &[f32], sample_rate: u32, options: &VadOptions) -> Vec<SpeechSpan> {
        let audio = SpeakerAudio::mono(samples, sample_rate).unwrap();
        EnergyVoiceActivityDetector
            .detect_speech(&audio, options)
            .unwrap()
            .segments
    }

    #[test]
    fn energy_vad_detects_and_merges_speech_spans() {
        let mut samples = vec![0.0; 100];
        samples.extend(vec![0.2; 120]);
        samples.extend(vec![0.0; 20]);
        samples.extend(vec![0.2; 120]);
        samples.extend(vec![0.0; 100]);
        let spans = detect(
            &samples,
            1_000,
            &VadOptions {
                rms_threshold: 0.05,
                frame_seconds: 0.04,
                hop_seconds: 0.02,
                min_speech_seconds: 0.05,
                merge_gap_seconds: 0.05,
                ..VadOptions::unpadded()
            },
        );

        assert_eq!(spans.len(), 1);
        assert!(spans[0].duration_seconds() >= 0.25);
    }

    #[test]
    fn energy_vad_rejects_invalid_options_and_filters_short_or_quiet_audio() {
        let audio = SpeakerAudio::mono(&[0.2; 20], 1_000).unwrap();
        for options in [
            VadOptions {
                rms_threshold: -1.0,
                ..VadOptions::default()
            },
            VadOptions {
                frame_seconds: 0.0,
                ..VadOptions::default()
            },
            VadOptions {
                max_chunk_seconds: Some(0.0),
                ..VadOptions::default()
            },
        ] {
            assert!(EnergyVoiceActivityDetector
                .detect_speech(&audio, &options)
                .is_err());
        }

        assert!(detect(&[0.0; 200], 1_000, &VadOptions::unpadded()).is_empty());
        assert!(detect(
            &[0.2; 20],
            1_000,
            &VadOptions {
                rms_threshold: 0.05,
                frame_seconds: 0.01,
                hop_seconds: 0.01,
                min_speech_seconds: 0.05,
                merge_gap_seconds: 0.0,
                ..VadOptions::unpadded()
            },
        )
        .is_empty());
    }

    #[test]
    fn energy_vad_pads_and_splits_long_spans() {
        let mut samples = vec![0.0; 100];
        samples.extend(vec![0.2; 500]);
        samples.extend(vec![0.0; 100]);
        let spans = detect(
            &samples,
            1_000,
            &VadOptions {
                frame_seconds: 0.01,
                hop_seconds: 0.01,
                padding_seconds: 0.02,
                max_chunk_seconds: Some(0.2),
                ..VadOptions::default()
            },
        );

        assert_eq!(spans.len(), 3);
        assert!((spans[0].start_seconds - 0.08).abs() < 1e-9);
        assert!((spans[2].end_seconds - 0.62).abs() < 1e-9);
        assert!(spans
            .iter()
            .all(|span| span.duration_seconds() <= 0.2 + 1e-9));
    }
}
