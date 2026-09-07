//! Confidence-bearing whole-track musical-key timeline analysis.
//!
//! This module keeps key-change semantics in the reusable pitch library. Transport
//! adapters may choose bounded PCM and configuration, but dominant-key decisions,
//! local-window decisions, and uncertainty handling remain Rust-owned here.

use audio_contracts::{DetectError, Result};

use super::{estimate_musical_key, HarmonicKeyConfig, MusicalKeyEstimate};

/// Configuration for change-aware whole-track key analysis.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyTimelineConfig {
    /// Duration of each overlapping local key window in seconds.
    pub window_seconds: f64,
    /// Distance between successive local key windows in seconds.
    pub hop_seconds: f64,
    /// Minimum key confidence required before a local window is treated as known.
    pub min_confidence: f32,
    /// Defensive upper bound on emitted windows.
    pub max_windows: usize,
}

impl Default for KeyTimelineConfig {
    fn default() -> Self {
        Self {
            window_seconds: 24.0,
            hop_seconds: 8.0,
            min_confidence: 0.10,
            max_windows: 256,
        }
    }
}

impl KeyTimelineConfig {
    /// Validates the timeline configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.window_seconds.is_finite() || self.window_seconds <= 0.0 {
            return Err(DetectError::InvalidArgument(
                "key timeline window_seconds must be finite and positive".to_string(),
            ));
        }
        if !self.hop_seconds.is_finite() || self.hop_seconds <= 0.0 {
            return Err(DetectError::InvalidArgument(
                "key timeline hop_seconds must be finite and positive".to_string(),
            ));
        }
        if !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(DetectError::InvalidArgument(
                "key timeline min_confidence must be between zero and one".to_string(),
            ));
        }
        if self.max_windows == 0 {
            return Err(DetectError::InvalidArgument(
                "key timeline max_windows must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// One confidence-bearing local key window.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyTimelineWindow {
    /// Window start in seconds from the beginning of the analyzed PCM.
    pub start_seconds: f64,
    /// Window end in seconds from the beginning of the analyzed PCM.
    pub end_seconds: f64,
    /// Window center in seconds.
    pub center_seconds: f64,
    /// Key estimate when confidence clears the configured threshold; otherwise unknown.
    pub key: Option<MusicalKeyEstimate>,
}

/// Reusable whole-track key result, separate from browser or WASM transport.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackKeyAnalysis {
    /// Versioned machine-readable contract identifier.
    pub schema_version: String,
    /// PCM sample rate used by the analysis.
    pub sample_rate: u32,
    /// Number of mono PCM samples analyzed.
    pub sample_count: usize,
    /// Analyzed duration in seconds.
    pub duration_seconds: f64,
    /// Dominant full-track key, if the complete track contains sufficient tonal evidence.
    pub dominant: Option<MusicalKeyEstimate>,
    /// Timeline window duration in seconds.
    pub timeline_window_seconds: f64,
    /// Timeline hop duration in seconds.
    pub timeline_hop_seconds: f64,
    /// Confidence threshold below which local windows remain explicitly unknown.
    pub timeline_min_confidence: f32,
    /// Whether the timeline reached the end of the input without hitting `max_windows`.
    pub timeline_complete: bool,
    /// Overlapping local key windows. Uncertain windows are retained with `key: null`.
    pub timeline: Vec<KeyTimelineWindow>,
}

/// Analyzes a complete mono track for dominant key and confidence-bearing local key changes.
pub fn analyze_key_track(
    samples: &[f32],
    sample_rate: u32,
    harmonic_config: HarmonicKeyConfig,
    timeline_config: KeyTimelineConfig,
) -> Result<TrackKeyAnalysis> {
    harmonic_config.validate()?;
    timeline_config.validate()?;
    if sample_rate == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate,
            channels: 1,
        });
    }
    if samples.is_empty() {
        return Err(DetectError::InvalidArgument(
            "track key samples must not be empty".to_string(),
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "track key samples must contain only finite values".to_string(),
        ));
    }

    let dominant = estimate_musical_key(samples, sample_rate, harmonic_config)?;
    let requested_window_samples =
        (timeline_config.window_seconds * sample_rate as f64).round() as usize;
    let window_samples = requested_window_samples.max(1).min(samples.len());
    let hop_samples = ((timeline_config.hop_seconds * sample_rate as f64).round() as usize).max(1);

    let mut timeline = Vec::new();
    let mut start = 0_usize;
    let mut timeline_complete = true;
    while start < samples.len() {
        if timeline.len() >= timeline_config.max_windows {
            timeline_complete = false;
            break;
        }
        let end = start.saturating_add(window_samples).min(samples.len());
        let estimate = estimate_musical_key(
            &samples[start..end],
            sample_rate,
            harmonic_config,
        )?
        .filter(|estimate| estimate.confidence >= timeline_config.min_confidence);
        let start_seconds = start as f64 / sample_rate as f64;
        let end_seconds = end as f64 / sample_rate as f64;
        timeline.push(KeyTimelineWindow {
            start_seconds,
            end_seconds,
            center_seconds: (start_seconds + end_seconds) * 0.5,
            key: estimate,
        });
        if end == samples.len() {
            break;
        }
        start = start.saturating_add(hop_samples);
    }

    Ok(TrackKeyAnalysis {
        schema_version: "audio-analysis-key-track/v1".to_string(),
        sample_rate,
        sample_count: samples.len(),
        duration_seconds: samples.len() as f64 / sample_rate as f64,
        dominant,
        timeline_window_seconds: timeline_config.window_seconds,
        timeline_hop_seconds: timeline_config.hop_seconds,
        timeline_min_confidence: timeline_config.min_confidence,
        timeline_complete,
        timeline,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_config_rejects_invalid_values() {
        assert!(KeyTimelineConfig {
            window_seconds: 0.0,
            ..KeyTimelineConfig::default()
        }
        .validate()
        .is_err());
        assert!(KeyTimelineConfig {
            hop_seconds: f64::NAN,
            ..KeyTimelineConfig::default()
        }
        .validate()
        .is_err());
        assert!(KeyTimelineConfig {
            min_confidence: 1.1,
            ..KeyTimelineConfig::default()
        }
        .validate()
        .is_err());
        assert!(KeyTimelineConfig {
            max_windows: 0,
            ..KeyTimelineConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn silence_keeps_uncertain_windows_explicit() {
        let analysis = analyze_key_track(
            &vec![0.0; 16_000 * 3],
            16_000,
            HarmonicKeyConfig {
                fft_size: 1024,
                hop_size: 512,
                ..HarmonicKeyConfig::default()
            },
            KeyTimelineConfig {
                window_seconds: 1.0,
                hop_seconds: 0.5,
                min_confidence: 0.1,
                ..KeyTimelineConfig::default()
            },
        )
        .expect("track key timeline");

        assert!(analysis.dominant.is_none());
        assert!(!analysis.timeline.is_empty());
        assert!(analysis.timeline.iter().all(|window| window.key.is_none()));
        assert!(analysis.timeline_complete);
    }

    #[test]
    fn max_windows_reports_incomplete_timeline_instead_of_hiding_truncation() {
        let analysis = analyze_key_track(
            &vec![0.0; 16_000 * 5],
            16_000,
            HarmonicKeyConfig {
                fft_size: 1024,
                hop_size: 512,
                ..HarmonicKeyConfig::default()
            },
            KeyTimelineConfig {
                window_seconds: 1.0,
                hop_seconds: 0.5,
                min_confidence: 0.1,
                max_windows: 2,
            },
        )
        .expect("bounded timeline");

        assert_eq!(analysis.timeline.len(), 2);
        assert!(!analysis.timeline_complete);
    }
}
