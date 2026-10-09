//! Generic capture-quality measurements over interleaved PCM.
//!
//! These are observations only: clipping, frames without input, and frames with activity, with the
//! thresholds that produced them. Whether a capture is usable for a downstream purpose is decided by
//! the consumer, not here.

use audio_contracts::{DetectError, Result};
use serde::{Deserialize, Serialize};

/// Default analysis frame length in seconds (20 ms).
pub const DEFAULT_CAPTURE_FRAME_SECONDS: f64 = 0.02;
/// Default absolute sample level at or above which a sample counts as clipped.
pub const DEFAULT_CAPTURE_CLIP_LEVEL: f64 = 0.999;
/// Default frame RMS below which a frame counts as having no input (about -80 dBFS).
pub const DEFAULT_CAPTURE_NO_INPUT_RMS: f64 = 1.0e-4;
/// Default frame RMS above which a frame counts as active (about -40 dBFS).
pub const DEFAULT_CAPTURE_ACTIVITY_RMS: f64 = 0.01;

/// Parameters of [`capture_metrics`]. Every field has a documented default.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct CaptureMetricsConfig {
    /// Frame length in seconds; rounded to whole samples per channel, at least one.
    pub frame_seconds: f64,
    /// A sample whose absolute value is at or above this level counts as clipped.
    pub clip_level: f64,
    /// A frame whose RMS is strictly below this floor counts as no input.
    pub no_input_rms: f64,
    /// A frame whose RMS is strictly above this floor counts as activity.
    pub activity_rms: f64,
}

impl Default for CaptureMetricsConfig {
    fn default() -> Self {
        Self {
            frame_seconds: DEFAULT_CAPTURE_FRAME_SECONDS,
            clip_level: DEFAULT_CAPTURE_CLIP_LEVEL,
            no_input_rms: DEFAULT_CAPTURE_NO_INPUT_RMS,
            activity_rms: DEFAULT_CAPTURE_ACTIVITY_RMS,
        }
    }
}

impl CaptureMetricsConfig {
    /// Validates this configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.frame_seconds.is_finite() || self.frame_seconds <= 0.0 {
            return Err(invalid("frameSeconds must be finite and positive"));
        }
        if !self.clip_level.is_finite() || self.clip_level <= 0.0 {
            return Err(invalid("clipLevel must be finite and positive"));
        }
        if !self.no_input_rms.is_finite() || self.no_input_rms < 0.0 {
            return Err(invalid("noInputRms must be finite and non-negative"));
        }
        if !self.activity_rms.is_finite() || self.activity_rms < 0.0 {
            return Err(invalid("activityRms must be finite and non-negative"));
        }
        Ok(())
    }

    /// Samples per channel in one analysis frame at `sample_rate` (saturating at `usize::MAX`).
    pub fn frame_samples(&self, sample_rate: u32) -> usize {
        ((self.frame_seconds * f64::from(sample_rate)).round() as usize).max(1)
    }
}

/// Capture-quality observations of one interleaved PCM buffer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureMetrics {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples_per_channel: usize,
    pub duration_seconds: f64,
    /// Interleaved samples at or above the clip level, over all channels.
    pub clipped_sample_count: usize,
    /// `clipped_sample_count` over all interleaved samples; 0 for an empty buffer.
    pub clipped_sample_ratio: f64,
    /// Samples per channel in one analysis frame; the last frame may be shorter.
    pub frame_samples: usize,
    pub frame_count: usize,
    /// Total duration of frames whose RMS is below the no-input floor.
    pub no_input_seconds: f64,
    /// Longest continuous run of no-input frames.
    pub longest_no_input_seconds: f64,
    /// Total duration of frames whose RMS is above the activity floor.
    pub activity_seconds: f64,
    /// The configuration that produced these observations.
    pub config: CaptureMetricsConfig,
}

/// Measures clipping, no-input and activity over interleaved samples normalized to [-1, 1].
///
/// A frame's RMS is taken over every interleaved sample of all channels in that frame, so channels
/// that cancel each other out do not hide input. Durations are whole frames (the last one partial),
/// accumulated in samples and converted to seconds once, so equal inputs give equal outputs on every
/// runtime that builds this crate.
pub fn capture_metrics(
    interleaved: &[f32],
    sample_rate: u32,
    channels: u16,
    config: &CaptureMetricsConfig,
) -> Result<CaptureMetrics> {
    if sample_rate == 0 || channels == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate,
            channels,
        });
    }
    config.validate()?;
    let channel_count = usize::from(channels);
    if !interleaved.len().is_multiple_of(channel_count) {
        return Err(invalid("sample count must be divisible by channels"));
    }
    if interleaved.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid("samples must be finite"));
    }

    let samples_per_channel = interleaved.len() / channel_count;
    let frame_samples = config.frame_samples(sample_rate);
    // On 32-bit targets (WASM) a long frame of many channels can exceed usize.
    let interleaved_frame_len = frame_samples
        .checked_mul(channel_count)
        .ok_or_else(|| invalid("frameSeconds × sampleRate × channels is too large"))?;
    let clipped_sample_count = interleaved
        .iter()
        .filter(|sample| f64::from(sample.abs()) >= config.clip_level)
        .count();

    let mut frame_count = 0;
    let mut no_input_samples = 0usize;
    let mut longest_no_input_samples = 0usize;
    let mut current_no_input_samples = 0usize;
    let mut activity_samples = 0usize;
    for frame in interleaved.chunks(interleaved_frame_len) {
        frame_count += 1;
        let frame_len = frame.len() / channel_count;
        let sum_squares = frame
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>();
        let frame_rms = (sum_squares / frame.len() as f64).sqrt();
        if frame_rms < config.no_input_rms {
            no_input_samples += frame_len;
            current_no_input_samples += frame_len;
            longest_no_input_samples = longest_no_input_samples.max(current_no_input_samples);
        } else {
            current_no_input_samples = 0;
        }
        if frame_rms > config.activity_rms {
            activity_samples += frame_len;
        }
    }

    let seconds = |samples: usize| samples as f64 / f64::from(sample_rate);
    Ok(CaptureMetrics {
        sample_rate,
        channels,
        samples_per_channel,
        duration_seconds: seconds(samples_per_channel),
        clipped_sample_count,
        clipped_sample_ratio: if interleaved.is_empty() {
            0.0
        } else {
            clipped_sample_count as f64 / interleaved.len() as f64
        },
        frame_samples,
        frame_count,
        no_input_seconds: seconds(no_input_samples),
        longest_no_input_seconds: seconds(longest_no_input_samples),
        activity_seconds: seconds(activity_samples),
        config: *config,
    })
}

fn invalid(message: &str) -> DetectError {
    DetectError::InvalidArgument(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(frame_seconds: f64) -> CaptureMetricsConfig {
        CaptureMetricsConfig {
            frame_seconds,
            ..CaptureMetricsConfig::default()
        }
    }

    #[test]
    fn measures_clipping_no_input_and_activity() {
        // 10 Hz, 1-sample frames: silence, silence, loud, clipped, silence, quiet-but-present.
        let samples = [0.0, 0.0, 0.5, -1.0, 0.0, 0.001];
        let metrics = capture_metrics(&samples, 10, 1, &config(0.1)).unwrap();
        assert_eq!(metrics.samples_per_channel, 6);
        assert_eq!(metrics.frame_samples, 1);
        assert_eq!(metrics.frame_count, 6);
        assert_eq!(metrics.clipped_sample_count, 1);
        assert!((metrics.clipped_sample_ratio - 1.0 / 6.0).abs() < 1e-12);
        assert!((metrics.no_input_seconds - 0.3).abs() < 1e-12);
        assert!((metrics.longest_no_input_seconds - 0.2).abs() < 1e-12);
        assert!((metrics.activity_seconds - 0.2).abs() < 1e-12);
    }

    #[test]
    fn counts_the_partial_last_frame_by_its_length() {
        let samples = [0.0; 5];
        let metrics = capture_metrics(&samples, 10, 1, &config(0.2)).unwrap();
        assert_eq!(metrics.frame_count, 3);
        assert!((metrics.no_input_seconds - 0.5).abs() < 1e-12);
        assert!((metrics.longest_no_input_seconds - 0.5).abs() < 1e-12);
    }

    #[test]
    fn frames_span_all_channels() {
        // Opposite-phase stereo has a zero mono mix but real input on both channels.
        let samples = [0.5, -0.5, 0.5, -0.5];
        let metrics = capture_metrics(&samples, 2, 2, &config(1.0)).unwrap();
        assert_eq!(metrics.samples_per_channel, 2);
        assert_eq!(metrics.frame_count, 1);
        assert_eq!(metrics.no_input_seconds, 0.0);
        assert_eq!(metrics.activity_seconds, 1.0);
    }

    #[test]
    fn empty_input_has_no_frames() {
        let metrics = capture_metrics(&[], 48_000, 1, &CaptureMetricsConfig::default()).unwrap();
        assert_eq!(metrics.frame_count, 0);
        assert_eq!(metrics.duration_seconds, 0.0);
        assert_eq!(metrics.clipped_sample_ratio, 0.0);
    }

    #[test]
    fn rejects_invalid_input_and_config() {
        let default = CaptureMetricsConfig::default();
        assert!(capture_metrics(&[0.0], 0, 1, &default).is_err());
        assert!(capture_metrics(&[0.0], 10, 0, &default).is_err());
        assert!(capture_metrics(&[0.0, 0.0, 0.0], 10, 2, &default).is_err());
        assert!(capture_metrics(&[f32::NAN], 10, 1, &default).is_err());
        assert!(capture_metrics(&[0.0], 10, 1, &config(0.0)).is_err());
        let negative_floor = CaptureMetricsConfig {
            no_input_rms: -1.0,
            ..default
        };
        assert!(capture_metrics(&[0.0], 10, 1, &negative_floor).is_err());
        let huge_frame = CaptureMetricsConfig {
            frame_seconds: f64::MAX,
            ..default
        };
        assert!(capture_metrics(&[], 48_000, 2, &huge_frame).is_err());
    }
}
