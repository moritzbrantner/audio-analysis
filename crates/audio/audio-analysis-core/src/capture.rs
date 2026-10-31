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
///
/// This is [`CaptureMetricsMeter`] fed the whole buffer at once, so a meter fed the same samples in
/// any chunking returns exactly these metrics.
pub fn capture_metrics(
    interleaved: &[f32],
    sample_rate: u32,
    channels: u16,
    config: &CaptureMetricsConfig,
) -> Result<CaptureMetrics> {
    let mut meter = CaptureMetricsMeter::new(sample_rate, channels, config)?;
    meter.push(interleaved)?;
    Ok(meter.finish())
}

/// Incremental [`capture_metrics`] for a live capture.
///
/// [`push`](Self::push) accepts interleaved chunks of any length (a whole number of sample frames
/// each). The meter keeps one partial analysis frame and a few counters, never earlier samples, and
/// each push only reads its own chunk. [`snapshot`](Self::snapshot) reports the metrics of the
/// samples so far with the partial frame counted as the last frame, which is exactly
/// [`capture_metrics`] of everything pushed so far; [`finish`](Self::finish) is the final
/// snapshot.
#[derive(Debug, Clone)]
pub struct CaptureMetricsMeter {
    sample_rate: u32,
    channels: u16,
    config: CaptureMetricsConfig,
    frame_samples: usize,
    interleaved_frame_len: usize,
    interleaved_samples: usize,
    clipped_sample_count: usize,
    /// Counters over the completed frames.
    frames: FrameCounters,
    /// Interleaved samples and their sum of squares in the current, incomplete frame.
    partial_len: usize,
    partial_sum_squares: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct FrameCounters {
    frame_count: usize,
    no_input_samples: usize,
    longest_no_input_samples: usize,
    current_no_input_samples: usize,
    activity_samples: usize,
}

impl FrameCounters {
    fn close_frame(
        &mut self,
        config: &CaptureMetricsConfig,
        channels: u16,
        interleaved_len: usize,
        sum_squares: f64,
    ) {
        let frame_len = interleaved_len / usize::from(channels);
        let frame_rms = (sum_squares / interleaved_len as f64).sqrt();
        self.frame_count += 1;
        if frame_rms < config.no_input_rms {
            self.no_input_samples += frame_len;
            self.current_no_input_samples += frame_len;
            self.longest_no_input_samples = self
                .longest_no_input_samples
                .max(self.current_no_input_samples);
        } else {
            self.current_no_input_samples = 0;
        }
        if frame_rms > config.activity_rms {
            self.activity_samples += frame_len;
        }
    }
}

impl CaptureMetricsMeter {
    /// Creates a meter for interleaved PCM at `sample_rate` with `channels` channels.
    pub fn new(sample_rate: u32, channels: u16, config: &CaptureMetricsConfig) -> Result<Self> {
        if sample_rate == 0 || channels == 0 {
            return Err(DetectError::InvalidAudioFormat {
                sample_rate,
                channels,
            });
        }
        config.validate()?;
        let frame_samples = config.frame_samples(sample_rate);
        // On 32-bit targets (WASM) a long frame of many channels can exceed usize.
        let interleaved_frame_len = frame_samples
            .checked_mul(usize::from(channels))
            .ok_or_else(|| invalid("frameSeconds × sampleRate × channels is too large"))?;
        Ok(Self {
            sample_rate,
            channels,
            config: *config,
            frame_samples,
            interleaved_frame_len,
            interleaved_samples: 0,
            clipped_sample_count: 0,
            frames: FrameCounters::default(),
            partial_len: 0,
            partial_sum_squares: 0.0,
        })
    }

    /// Adds the next interleaved chunk. A rejected chunk leaves the meter unchanged.
    pub fn push(&mut self, interleaved: &[f32]) -> Result<()> {
        if !interleaved.len().is_multiple_of(usize::from(self.channels)) {
            return Err(invalid("sample count must be divisible by channels"));
        }
        if interleaved.iter().any(|sample| !sample.is_finite()) {
            return Err(invalid("samples must be finite"));
        }
        for sample in interleaved {
            if f64::from(sample.abs()) >= self.config.clip_level {
                self.clipped_sample_count += 1;
            }
            self.partial_sum_squares += f64::from(*sample) * f64::from(*sample);
            self.partial_len += 1;
            if self.partial_len == self.interleaved_frame_len {
                self.frames.close_frame(
                    &self.config,
                    self.channels,
                    self.partial_len,
                    self.partial_sum_squares,
                );
                self.partial_len = 0;
                self.partial_sum_squares = 0.0;
            }
        }
        self.interleaved_samples += interleaved.len();
        Ok(())
    }

    /// Metrics of every sample pushed so far, the incomplete frame counted as a partial last frame.
    pub fn snapshot(&self) -> CaptureMetrics {
        let mut counters = self.frames;
        if self.partial_len > 0 {
            counters.close_frame(
                &self.config,
                self.channels,
                self.partial_len,
                self.partial_sum_squares,
            );
        }
        let samples_per_channel = self.interleaved_samples / usize::from(self.channels);
        let seconds = |samples: usize| samples as f64 / f64::from(self.sample_rate);
        CaptureMetrics {
            sample_rate: self.sample_rate,
            channels: self.channels,
            samples_per_channel,
            duration_seconds: seconds(samples_per_channel),
            clipped_sample_count: self.clipped_sample_count,
            clipped_sample_ratio: if self.interleaved_samples == 0 {
                0.0
            } else {
                self.clipped_sample_count as f64 / self.interleaved_samples as f64
            },
            frame_samples: self.frame_samples,
            frame_count: counters.frame_count,
            no_input_seconds: seconds(counters.no_input_samples),
            longest_no_input_seconds: seconds(counters.longest_no_input_samples),
            activity_seconds: seconds(counters.activity_samples),
            config: self.config,
        }
    }

    /// The final metrics; equal to [`capture_metrics`] over the concatenated chunks.
    pub fn finish(self) -> CaptureMetrics {
        self.snapshot()
    }
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
    fn meter_matches_the_whole_buffer_in_any_chunking() {
        // Stereo at 1 kHz with 7-sample frames: silence, a burst with clipping, low-level hiss.
        let mut samples = Vec::new();
        for index in 0..400usize {
            let value = match index {
                0..=99 => 0.0,
                100..=219 => ((index as f32) * 0.37).sin() * 1.2,
                _ => ((index as f32) * 1.3).sin() * 0.003,
            };
            samples.extend([value.clamp(-1.0, 1.0), -value.clamp(-1.0, 1.0) * 0.5]);
        }
        let config = config(0.007);
        let whole = capture_metrics(&samples, 1_000, 2, &config).unwrap();
        for chunk_frames in [1usize, 3, 7, 50, 401] {
            let mut meter = CaptureMetricsMeter::new(1_000, 2, &config).unwrap();
            for chunk in samples.chunks(chunk_frames * 2) {
                meter.push(chunk).unwrap();
                let pushed = meter.snapshot().samples_per_channel * 2;
                assert_eq!(
                    meter.snapshot(),
                    capture_metrics(&samples[..pushed], 1_000, 2, &config).unwrap()
                );
            }
            assert_eq!(meter.finish(), whole, "chunk of {chunk_frames} frames");
        }
    }

    #[test]
    fn meter_rejects_a_bad_chunk_without_changing_state() {
        let mut meter = CaptureMetricsMeter::new(10, 2, &CaptureMetricsConfig::default()).unwrap();
        meter.push(&[0.5, 0.5]).unwrap();
        let before = meter.snapshot();
        assert!(meter.push(&[0.1]).is_err());
        assert!(meter.push(&[0.1, f32::INFINITY]).is_err());
        assert_eq!(meter.snapshot(), before);
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
