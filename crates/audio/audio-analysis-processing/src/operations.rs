/// Pure playback and DJ-oriented planning helpers shared by audio consumers.
///
/// The APIs here are independently implemented from general signal-processing
/// and DJ workflow behavior. They do not copy or translate Mixxx source code.
pub mod playback {
    use std::f64::consts::FRAC_PI_2;

    /// Gain pair for an equal-power crossfade.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct CrossfadeGains {
        /// Gain for the left/A source.
        pub left: f64,
        /// Gain for the right/B source.
        pub right: f64,
    }

    /// Inclusive tempo-control range expressed as percentages around normal speed.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TempoRange {
        /// Minimum tempo percentage.
        pub min_percent: f64,
        /// Maximum tempo percentage.
        pub max_percent: f64,
    }

    impl TempoRange {
        /// Creates a tempo range. Public operations reject invalid ranges.
        pub const fn new(min_percent: f64, max_percent: f64) -> Self {
            Self {
                min_percent,
                max_percent,
            }
        }

        /// Returns whether the range can map to positive playback rates.
        pub fn is_valid(self) -> bool {
            self.min_percent.is_finite()
                && self.max_percent.is_finite()
                && self.min_percent <= self.max_percent
                && self.min_percent > -100.0
        }

        /// Returns the inclusive playback-rate bounds represented by this range.
        pub fn playback_rate_bounds(self) -> Option<(f64, f64)> {
            if !self.is_valid() {
                return None;
            }
            let min_rate = 1.0 + self.min_percent / 100.0;
            let max_rate = 1.0 + self.max_percent / 100.0;
            (min_rate.is_finite() && max_rate.is_finite()).then_some((min_rate, max_rate))
        }
    }

    /// Result of planning tempo-only BPM synchronization.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct BpmSyncPlan {
        /// Playback rate to apply to the source.
        pub playback_rate: f64,
        /// Effective BPM of the target after its existing playback rate.
        pub target_effective_bpm: f64,
        /// Effective BPM reached by the source after clamping.
        pub source_effective_bpm: f64,
        /// Whether the requested match exceeded the configured tempo range.
        pub limited: bool,
    }

    /// Beat-grid-aligned loop boundaries.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct BeatLoop {
        /// Loop start time.
        pub start_seconds: f64,
        /// Loop end time.
        pub end_seconds: f64,
        /// Number of beat intervals covered by the loop.
        pub beat_count: usize,
    }

    /// Minimum and maximum sample values for one waveform summary bucket.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct WaveformExtrema {
        /// Minimum normalized sample value.
        pub min: f32,
        /// Maximum normalized sample value.
        pub max: f32,
    }

    /// Computes equal-power crossfader gains for a position in `[-1, 1]`.
    pub fn equal_power_crossfade_gains(position: f64) -> CrossfadeGains {
        if !position.is_finite() {
            return CrossfadeGains {
                left: f64::NAN,
                right: f64::NAN,
            };
        }

        let normalized = (position.clamp(-1.0, 1.0) + 1.0) * 0.5;
        let angle = normalized * FRAC_PI_2;
        CrossfadeGains {
            left: angle.cos(),
            right: angle.sin(),
        }
    }

    /// Maps a tempo percentage to a playback rate within the supplied range.
    pub fn playback_rate_for_tempo(tempo_percent: f64, range: TempoRange) -> Option<f64> {
        if !tempo_percent.is_finite() || !range.is_valid() {
            return None;
        }

        Some(1.0 + tempo_percent.clamp(range.min_percent, range.max_percent) / 100.0)
    }

    /// Maps a positive playback rate back to a tempo percentage within the range.
    pub fn tempo_percent_for_rate(playback_rate: f64, range: TempoRange) -> Option<f64> {
        let (min_rate, max_rate) = range.playback_rate_bounds()?;
        if !playback_rate.is_finite() || playback_rate <= 0.0 {
            return None;
        }

        Some(
            ((playback_rate.clamp(min_rate, max_rate) - 1.0) * 100.0)
                .clamp(range.min_percent, range.max_percent),
        )
    }

    /// Returns the effective BPM produced by a positive playback rate.
    pub fn effective_bpm(base_bpm: f64, playback_rate: f64) -> Option<f64> {
        if !base_bpm.is_finite()
            || base_bpm <= 0.0
            || !playback_rate.is_finite()
            || playback_rate <= 0.0
        {
            return None;
        }

        let bpm = base_bpm * playback_rate;
        bpm.is_finite().then_some(bpm)
    }

    /// Plans tempo-only BPM synchronization without changing transport phase.
    pub fn plan_bpm_sync(
        source_bpm: f64,
        target_bpm: f64,
        target_playback_rate: f64,
        source_tempo_range: TempoRange,
    ) -> Option<BpmSyncPlan> {
        let target_effective_bpm = effective_bpm(target_bpm, target_playback_rate)?;
        if !source_bpm.is_finite() || source_bpm <= 0.0 {
            return None;
        }
        let (min_rate, max_rate) = source_tempo_range.playback_rate_bounds()?;

        let requested_rate = target_effective_bpm / source_bpm;
        let playback_rate = requested_rate.clamp(min_rate, max_rate);
        let source_effective_bpm = source_bpm * playback_rate;

        Some(BpmSyncPlan {
            playback_rate,
            target_effective_bpm,
            source_effective_bpm,
            limited: (playback_rate - requested_rate).abs() > f64::EPSILON,
        })
    }

    /// Selects the nearest valid beat-grid start for a loop of `beat_count` beats.
    ///
    /// The caller owns policy such as which beat counts are exposed in the UI.
    pub fn plan_beat_loop(
        beats: &[f64],
        current_seconds: f64,
        beat_count: usize,
        duration_seconds: f64,
    ) -> Option<BeatLoop> {
        if !current_seconds.is_finite()
            || current_seconds < 0.0
            || !duration_seconds.is_finite()
            || duration_seconds <= 0.0
            || beat_count == 0
            || beats.len() <= beat_count
            || !valid_beat_grid(beats)
            || current_seconds > *beats.last()?
        {
            return None;
        }

        let last_start_index = beats.len() - beat_count - 1;
        let mut selected_index = None;
        let mut selected_distance = f64::INFINITY;

        for index in 0..=last_start_index {
            let start = beats[index];
            let end = beats[index + beat_count];
            if start < 0.0 || end <= start || end > duration_seconds {
                continue;
            }

            let distance = (start - current_seconds).abs();
            if distance < selected_distance {
                selected_index = Some(index);
                selected_distance = distance;
            }
        }

        let index = selected_index?;
        Some(BeatLoop {
            start_seconds: beats[index],
            end_seconds: beats[index + beat_count],
            beat_count,
        })
    }

    /// Summarizes normalized PCM into evenly distributed min/max buckets.
    pub fn waveform_extrema(samples: &[f32], point_count: usize) -> Vec<WaveformExtrema> {
        if samples.is_empty() || point_count == 0 {
            return Vec::new();
        }

        let point_count = point_count.min(samples.len());
        let mut extrema = Vec::with_capacity(point_count);

        for index in 0..point_count {
            let start = index * samples.len() / point_count;
            let mut end = (index + 1) * samples.len() / point_count;
            if end <= start {
                end = start + 1;
            }

            let mut minimum = 0.0_f32;
            let mut maximum = 0.0_f32;
            for sample in &samples[start..end.min(samples.len())] {
                if !sample.is_finite() {
                    continue;
                }
                minimum = minimum.min(*sample);
                maximum = maximum.max(*sample);
            }
            extrema.push(WaveformExtrema {
                min: minimum.clamp(-1.0, 1.0),
                max: maximum.clamp(-1.0, 1.0),
            });
        }

        extrema
    }

    fn valid_beat_grid(beats: &[f64]) -> bool {
        !beats.is_empty()
            && beats.iter().all(|beat| beat.is_finite())
            && beats.windows(2).all(|pair| pair[1] > pair[0])
    }
}

pub fn effects_catalog_value() -> serde_json::Value {
    serde_json::json!({
        "streamingEffects": [
            {"type": "gain", "fields": ["linear"]},
            {"type": "distortion", "fields": ["mode", "driveDb", "mix", "outputGainDb"]},
            {"type": "delay", "fields": ["delaySeconds", "feedback", "wet", "dry"]},
            {"type": "echo", "fields": ["delaySeconds", "feedback", "wet", "dry"]},
            {"type": "reverb", "fields": ["roomSize", "damping", "wet", "dry", "width"]},
            {"type": "compressor", "fields": ["thresholdDb", "ratio", "attackMs", "releaseMs", "makeupGainDb", "kneeDb"]},
            {"type": "limiter", "fields": ["ceilingDb", "releaseMs"]},
            {"type": "eq", "fields": ["bands"]},
            {"type": "chorus", "fields": ["baseDelayMs", "depthMs", "rateHz", "feedback", "wet", "dry"]},
            {"type": "flanger", "fields": ["baseDelayMs", "depthMs", "rateHz", "feedback", "wet", "dry"]},
            {"type": "tremolo", "fields": ["rateHz", "depth"]},
            {"type": "pan", "fields": ["position"]},
            {"type": "stereoWidth", "fields": ["width"]}
        ],
        "offlineEdits": ["trim", "reverse", "fade", "normalize", "insertSilence", "delete", "resample", "speed", "pitchShift"],
        "presets": ["VocalClean", "PodcastVoice", "LoFi", "WideChorus", "SmallRoomReverb", "HardLimiter"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use playback::*;

    const EPSILON: f64 = 1e-12;
    const DJ_RANGE: TempoRange = TempoRange::new(-16.0, 16.0);

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn effects_catalog_stays_local_and_deterministic() {
        let catalog = effects_catalog_value();
        assert!(catalog["streamingEffects"].as_array().unwrap().len() >= 10);
        assert!(catalog["offlineEdits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|edit| edit == "reverse"));
    }

    #[test]
    fn equal_power_crossfade_preserves_edges_and_center_power() {
        let left = equal_power_crossfade_gains(-1.0);
        assert_close(left.left, 1.0);
        assert_close(left.right, 0.0);

        let center = equal_power_crossfade_gains(0.0);
        let expected = 1.0 / 2.0_f64.sqrt();
        assert_close(center.left, expected);
        assert_close(center.right, expected);

        let right = equal_power_crossfade_gains(1.0);
        assert_close(right.left, 0.0);
        assert_close(right.right, 1.0);
    }

    #[test]
    fn tempo_mapping_is_clamped_and_reversible_inside_range() {
        assert_close(playback_rate_for_tempo(-50.0, DJ_RANGE).unwrap(), 0.84);
        assert_close(playback_rate_for_tempo(50.0, DJ_RANGE).unwrap(), 1.16);
        assert_close(playback_rate_for_tempo(7.5, DJ_RANGE).unwrap(), 1.075);
        assert_close(tempo_percent_for_rate(1.075, DJ_RANGE).unwrap(), 7.5);
    }

    #[test]
    fn bpm_sync_matches_reachable_target_and_reports_limits() {
        let plan = plan_bpm_sync(120.0, 128.0, 1.0, DJ_RANGE).unwrap();
        assert!(!plan.limited);
        assert_close(plan.playback_rate, 128.0 / 120.0);
        assert_close(plan.source_effective_bpm, 128.0);

        let limited = plan_bpm_sync(100.0, 140.0, 1.0, DJ_RANGE).unwrap();
        assert!(limited.limited);
        assert_close(limited.playback_rate, 1.16);
        assert_close(limited.source_effective_bpm, 116.0);
    }

    #[test]
    fn beat_loop_uses_nearest_valid_grid_start() {
        let beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let plan = plan_beat_loop(&beats, 1.12, 4, 3.0).unwrap();
        assert_close(plan.start_seconds, 1.0);
        assert_close(plan.end_seconds, 3.0);
        assert_eq!(plan.beat_count, 4);
    }

    #[test]
    fn beat_loop_rejects_invalid_or_stale_grids() {
        assert!(plan_beat_loop(&[0.0, 0.5, 0.5, 1.0], 0.4, 1, 1.0).is_none());
        assert!(plan_beat_loop(&[0.0, 0.5, 1.0], 1.5, 1, 2.0).is_none());
    }

    #[test]
    fn waveform_extrema_preserves_bucket_extremes_and_bounds() {
        let samples = [0.25, 0.75, -0.5, -2.0, f32::NAN, 2.0];
        let extrema = waveform_extrema(&samples, 3);
        assert_eq!(extrema.len(), 3);
        assert_eq!(
            extrema[0],
            WaveformExtrema {
                min: 0.0,
                max: 0.75
            }
        );
        assert_eq!(
            extrema[1],
            WaveformExtrema {
                min: -1.0,
                max: 0.0
            }
        );
        assert_eq!(
            extrema[2],
            WaveformExtrema {
                min: 0.0,
                max: 1.0
            }
        );
    }
}
