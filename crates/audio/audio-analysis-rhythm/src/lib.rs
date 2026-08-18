#![doc = include_str!("../README.md")]

pub mod surface;
use audio_analysis_core::{mono_samples, rms, FrameSpec};
use audio_contracts::{AnalysisEvent, AudioAnalyzer, AudioFrame, DetectError, Result, Timestamp};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for onset strength.
pub struct OnsetStrength {
    /// The frame index value.
    pub frame_index: usize,
    /// The start sample value.
    pub start_sample: usize,
    /// The timestamp seconds value.
    pub timestamp_seconds: f64,
    /// The energy value.
    pub energy: f32,
    /// The strength value.
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Variants describing onset envelope strategy.
pub enum OnsetEnvelopeStrategy {
    #[default]
    /// The energy rise variant.
    EnergyRise,
    /// The spectral flux like variant.
    SpectralFluxLike,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for onset.
pub struct Onset {
    /// The timestamp seconds value.
    pub timestamp_seconds: f64,
    /// The strength value.
    pub strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for onset detector config.
pub struct OnsetDetectorConfig {
    /// The strength threshold value.
    pub strength_threshold: f32,
    /// The min interval seconds value.
    pub min_interval_seconds: f64,
}

impl OnsetDetectorConfig {
    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if !self.strength_threshold.is_finite() || self.strength_threshold < 0.0 {
            return Err(DetectError::InvalidArgument(
                "onset strength_threshold must be finite and non-negative".to_string(),
            ));
        }
        if !self.min_interval_seconds.is_finite() || self.min_interval_seconds < 0.0 {
            return Err(DetectError::InvalidArgument(
                "onset min_interval_seconds must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for OnsetDetectorConfig {
    fn default() -> Self {
        Self {
            strength_threshold: 0.05,
            min_interval_seconds: 0.08,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for onset detector.
pub struct OnsetDetector {
    name: String,
    /// The config value.
    pub config: OnsetDetectorConfig,
    previous_energy: Option<f32>,
    last_onset_seconds: Option<f64>,
}

impl OnsetDetector {
    /// Creates a new value.
    pub fn new(config: OnsetDetectorConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            name: "audio_onset".to_string(),
            config,
            previous_energy: None,
            last_onset_seconds: None,
        })
    }

    /// Returns reset.
    pub fn reset(&mut self) {
        self.previous_energy = None;
        self.last_onset_seconds = None;
    }

    /// Returns process energy.
    pub fn process_energy(&mut self, timestamp: Timestamp, energy: f32) -> Option<Onset> {
        let strength = (energy - self.previous_energy.unwrap_or(energy)).max(0.0);
        self.previous_energy = Some(energy);
        if strength < self.config.strength_threshold {
            return None;
        }
        let seconds = timestamp.seconds();
        if self
            .last_onset_seconds
            .is_some_and(|last| seconds - last < self.config.min_interval_seconds)
        {
            return None;
        }
        self.last_onset_seconds = Some(seconds);
        Some(Onset {
            timestamp_seconds: seconds,
            strength,
        })
    }
}

impl Default for OnsetDetector {
    fn default() -> Self {
        Self::new(OnsetDetectorConfig::default()).expect("default onset config is valid")
    }
}

impl AudioAnalyzer for OnsetDetector {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Vec<AnalysisEvent>> {
        let mono = mono_samples(frame)?;
        let energy = rms(&mono.samples);
        let Some(onset) = self.process_energy(frame.timestamp, energy) else {
            return Ok(Vec::new());
        };
        Ok(vec![AnalysisEvent::new(self.name(), "audio:onset")
            .at_timestamp(frame.timestamp)
            .score(onset.strength)])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for tempo estimator config.
pub struct TempoEstimatorConfig {
    /// The min BPM value.
    pub min_bpm: f32,
    /// The max BPM value.
    pub max_bpm: f32,
}

impl TempoEstimatorConfig {
    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if !self.min_bpm.is_finite()
            || !self.max_bpm.is_finite()
            || self.min_bpm <= 0.0
            || self.max_bpm <= self.min_bpm
        {
            return Err(DetectError::InvalidArgument(
                "tempo BPM range must be finite and increasing".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for TempoEstimatorConfig {
    fn default() -> Self {
        Self {
            min_bpm: 60.0,
            max_bpm: 240.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for tempo estimate.
pub struct TempoEstimate {
    /// The BPM value.
    pub bpm: Option<f32>,
    /// Confidence score for this value.
    pub confidence: f32,
    /// The onset count value.
    pub onset_count: usize,
}

/// Returns onset envelope.
pub fn onset_envelope(
    samples: &[f32],
    sample_rate: u32,
    frame_spec: FrameSpec,
) -> Result<Vec<OnsetStrength>> {
    onset_envelope_with_strategy(
        samples,
        sample_rate,
        frame_spec,
        OnsetEnvelopeStrategy::EnergyRise,
    )
}

/// Returns onset envelope with strategy.
pub fn onset_envelope_with_strategy(
    samples: &[f32],
    sample_rate: u32,
    frame_spec: FrameSpec,
    strategy: OnsetEnvelopeStrategy,
) -> Result<Vec<OnsetStrength>> {
    if sample_rate == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate,
            channels: 1,
        });
    }
    let mut previous_energy = None;
    let mut previous_abs_mean = None;
    Ok(frame_spec
        .frames(samples)
        .enumerate()
        .map(|(frame_index, (start_sample, frame))| {
            let energy = rms(frame);
            let strength = match strategy {
                OnsetEnvelopeStrategy::EnergyRise => {
                    (energy - previous_energy.unwrap_or(energy)).max(0.0)
                }
                OnsetEnvelopeStrategy::SpectralFluxLike => {
                    let abs_mean =
                        frame.iter().map(|sample| sample.abs()).sum::<f32>() / frame.len() as f32;
                    let strength = (abs_mean - previous_abs_mean.unwrap_or(abs_mean)).max(0.0);
                    previous_abs_mean = Some(abs_mean);
                    strength
                }
            };
            previous_energy = Some(energy);
            OnsetStrength {
                frame_index,
                start_sample,
                timestamp_seconds: start_sample as f64 / sample_rate as f64,
                energy,
                strength,
            }
        })
        .collect())
}

/// Returns inter onset intervals.
pub fn inter_onset_intervals(onsets: &[Onset]) -> Vec<f64> {
    onsets
        .windows(2)
        .map(|pair| pair[1].timestamp_seconds - pair[0].timestamp_seconds)
        .filter(|interval| *interval > 0.0)
        .collect()
}

/// Returns beat grid.
pub fn beat_grid(start_seconds: f64, bpm: f32, beats: usize) -> Result<Vec<f64>> {
    if !start_seconds.is_finite() || start_seconds < 0.0 {
        return Err(DetectError::InvalidArgument(
            "beat grid start_seconds must be finite and non-negative".to_string(),
        ));
    }
    if !bpm.is_finite() || bpm <= 0.0 {
        return Err(DetectError::InvalidArgument(
            "beat grid bpm must be finite and positive".to_string(),
        ));
    }
    let interval = 60.0 / bpm as f64;
    Ok((0..beats)
        .map(|index| start_seconds + index as f64 * interval)
        .collect())
}

/// Returns detect onsets.
pub fn detect_onsets(
    envelope: &[OnsetStrength],
    config: OnsetDetectorConfig,
) -> Result<Vec<Onset>> {
    config.validate()?;
    let mut onsets = Vec::new();
    for window in envelope.windows(3) {
        let candidate = window[1];
        if candidate.strength < config.strength_threshold
            || candidate.strength < window[0].strength
            || candidate.strength < window[2].strength
        {
            continue;
        }
        if onsets.last().is_some_and(|last: &Onset| {
            candidate.timestamp_seconds - last.timestamp_seconds < config.min_interval_seconds
        }) {
            continue;
        }
        onsets.push(Onset {
            timestamp_seconds: candidate.timestamp_seconds,
            strength: candidate.strength,
        });
    }
    Ok(onsets)
}

/// Returns estimate tempo.
pub fn estimate_tempo(onsets: &[Onset], config: TempoEstimatorConfig) -> Result<TempoEstimate> {
    config.validate()?;
    if onsets.len() < 2 {
        return Ok(TempoEstimate {
            bpm: None,
            confidence: 0.0,
            onset_count: onsets.len(),
        });
    }

    let mut bpms = onsets
        .windows(2)
        .filter_map(|pair| {
            let interval = pair[1].timestamp_seconds - pair[0].timestamp_seconds;
            if interval <= 0.0 {
                return None;
            }
            let mut bpm = 60.0 / interval as f32;
            while bpm < config.min_bpm {
                bpm *= 2.0;
            }
            while bpm > config.max_bpm {
                bpm *= 0.5;
            }
            (config.min_bpm..=config.max_bpm)
                .contains(&bpm)
                .then_some(bpm)
        })
        .collect::<Vec<_>>();

    if bpms.is_empty() {
        return Ok(TempoEstimate {
            bpm: None,
            confidence: 0.0,
            onset_count: onsets.len(),
        });
    }

    bpms.sort_by(f32::total_cmp);
    let bpm = bpms[bpms.len() / 2];
    let close = bpms
        .iter()
        .filter(|candidate| (*candidate - bpm).abs() <= bpm * 0.05)
        .count();
    Ok(TempoEstimate {
        bpm: Some(bpm),
        confidence: close as f32 / bpms.len() as f32,
        onset_count: onsets.len(),
    })
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for rhythm analyzer.
pub struct RhythmAnalyzer {
    name: String,
    onset_detector: OnsetDetector,
    tempo_config: TempoEstimatorConfig,
    onsets: Vec<Onset>,
}

impl RhythmAnalyzer {
    /// Creates a new value.
    pub fn new(
        onset_config: OnsetDetectorConfig,
        tempo_config: TempoEstimatorConfig,
    ) -> Result<Self> {
        tempo_config.validate()?;
        Ok(Self {
            name: "audio_rhythm".to_string(),
            onset_detector: OnsetDetector::new(onset_config)?,
            tempo_config,
            onsets: Vec::new(),
        })
    }

    /// Returns onsets.
    pub fn onsets(&self) -> &[Onset] {
        &self.onsets
    }
}

impl Default for RhythmAnalyzer {
    fn default() -> Self {
        Self::new(
            OnsetDetectorConfig::default(),
            TempoEstimatorConfig::default(),
        )
        .expect("default rhythm config is valid")
    }
}

impl AudioAnalyzer for RhythmAnalyzer {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Vec<AnalysisEvent>> {
        let mut events = self.onset_detector.process_frame(frame)?;
        if let Some(event) = events.last() {
            self.onsets.push(Onset {
                timestamp_seconds: frame.timestamp.seconds(),
                strength: event.score.unwrap_or(0.0),
            });
        }
        for event in &mut events {
            event.analyzer = self.name().to_string();
        }
        Ok(events)
    }

    fn finish(&mut self, last_timestamp: Option<Timestamp>) -> Result<Vec<AnalysisEvent>> {
        let tempo = estimate_tempo(&self.onsets, self.tempo_config)?;
        let Some(bpm) = tempo.bpm else {
            return Ok(Vec::new());
        };
        let mut event = AnalysisEvent::new(self.name(), format!("audio:tempo:{bpm:.2}bpm"))
            .score(tempo.confidence);
        if let Some(timestamp) = last_timestamp {
            event = event.at_timestamp(timestamp);
        }
        Ok(vec![event])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_contracts::{AudioBuffer, AudioFrame, Timebase};

    fn click_track(sample_rate: u32, bpm: f32, seconds: f32) -> Vec<f32> {
        let len = (sample_rate as f32 * seconds) as usize;
        let interval = (sample_rate as f32 * 60.0 / bpm) as usize;
        let mut samples = vec![0.0; len];
        for start in (0..len).step_by(interval) {
            for sample in samples.iter_mut().skip(start).take(8) {
                *sample = 1.0;
            }
        }
        samples
    }

    #[test]
    fn configs_reject_invalid_values() {
        assert!(OnsetDetectorConfig {
            strength_threshold: f32::NAN,
            min_interval_seconds: 0.1,
        }
        .validate()
        .is_err());
        assert!(OnsetDetectorConfig {
            strength_threshold: -0.1,
            min_interval_seconds: 0.1,
        }
        .validate()
        .is_err());
        assert!(OnsetDetectorConfig {
            strength_threshold: 0.1,
            min_interval_seconds: -0.1,
        }
        .validate()
        .is_err());
        assert!(TempoEstimatorConfig {
            min_bpm: 0.0,
            max_bpm: 240.0,
        }
        .validate()
        .is_err());
        assert!(TempoEstimatorConfig {
            min_bpm: 120.0,
            max_bpm: 60.0,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn onset_detector_respects_min_interval() {
        let mut detector = OnsetDetector::new(OnsetDetectorConfig {
            strength_threshold: 0.1,
            min_interval_seconds: 0.5,
        })
        .unwrap();
        assert!(detector
            .process_energy(Timestamp::new(0, Timebase::new(1, 1000)), 0.0)
            .is_none());
        assert!(detector
            .process_energy(Timestamp::new(100, Timebase::new(1, 1000)), 1.0)
            .is_some());
        assert!(detector
            .process_energy(Timestamp::new(200, Timebase::new(1, 1000)), 2.0)
            .is_none());
        assert!(detector
            .process_energy(Timestamp::new(700, Timebase::new(1, 1000)), 3.0)
            .is_some());
    }

    #[test]
    fn envelope_detects_click_onsets() {
        let samples = click_track(1_000, 120.0, 4.0);
        let envelope = onset_envelope(&samples, 1_000, FrameSpec::new(50, 25).unwrap()).unwrap();
        let onsets = detect_onsets(
            &envelope,
            OnsetDetectorConfig {
                strength_threshold: 0.1,
                min_interval_seconds: 0.2,
            },
        )
        .unwrap();
        assert!(onsets.len() >= 6);
    }

    #[test]
    fn estimates_tempo_from_even_onsets() {
        let onsets = [0.0, 0.5, 1.0, 1.5, 2.0]
            .into_iter()
            .map(|timestamp_seconds| Onset {
                timestamp_seconds,
                strength: 1.0,
            })
            .collect::<Vec<_>>();
        let tempo = estimate_tempo(&onsets, TempoEstimatorConfig::default()).unwrap();
        assert_eq!(tempo.bpm, Some(120.0));
        assert_eq!(tempo.confidence, 1.0);
    }

    #[test]
    fn estimates_tempo_for_generated_click_tracks() {
        for bpm in [60.0, 90.0, 120.0, 180.0] {
            let samples = click_track(2_000, bpm, 5.0);
            let envelope =
                onset_envelope(&samples, 2_000, FrameSpec::new(80, 20).unwrap()).unwrap();
            let onsets = detect_onsets(
                &envelope,
                OnsetDetectorConfig {
                    strength_threshold: 0.05,
                    min_interval_seconds: 0.1,
                },
            )
            .unwrap();
            let tempo = estimate_tempo(&onsets, TempoEstimatorConfig::default()).unwrap();
            let actual = tempo.bpm.unwrap();
            assert!((actual - bpm).abs() <= 2.0, "expected {bpm}, got {actual}");
        }
    }

    #[test]
    fn silence_produces_no_onsets_or_tempo() {
        let envelope =
            onset_envelope(&vec![0.0; 2_000], 1_000, FrameSpec::new(100, 50).unwrap()).unwrap();
        let onsets = detect_onsets(&envelope, OnsetDetectorConfig::default()).unwrap();
        assert!(onsets.is_empty());
        let tempo = estimate_tempo(&onsets, TempoEstimatorConfig::default()).unwrap();
        assert_eq!(tempo.bpm, None);
        assert_eq!(tempo.confidence, 0.0);
    }

    #[test]
    fn irregular_onsets_have_lower_confidence() {
        let regular = [0.0, 0.5, 1.0, 1.5, 2.0]
            .into_iter()
            .map(|timestamp_seconds| Onset {
                timestamp_seconds,
                strength: 1.0,
            })
            .collect::<Vec<_>>();
        let irregular = [0.0, 0.5, 1.3, 1.9, 3.0]
            .into_iter()
            .map(|timestamp_seconds| Onset {
                timestamp_seconds,
                strength: 1.0,
            })
            .collect::<Vec<_>>();
        let regular = estimate_tempo(&regular, TempoEstimatorConfig::default()).unwrap();
        let irregular = estimate_tempo(&irregular, TempoEstimatorConfig::default()).unwrap();
        assert!(irregular.confidence < regular.confidence);
    }

    #[test]
    fn alternate_envelope_strategy_and_interval_helpers_work() {
        let samples = click_track(1_000, 120.0, 2.0);
        let envelope = onset_envelope_with_strategy(
            &samples,
            1_000,
            FrameSpec::new(50, 25).unwrap(),
            OnsetEnvelopeStrategy::SpectralFluxLike,
        )
        .unwrap();
        assert!(!envelope.is_empty());
        let onsets = detect_onsets(&envelope, OnsetDetectorConfig::default()).unwrap();
        let intervals = inter_onset_intervals(&onsets);
        assert!(intervals.iter().all(|interval| *interval > 0.0));
    }

    #[test]
    fn beat_grid_returns_evenly_spaced_beats() {
        let grid = beat_grid(0.5, 120.0, 4).unwrap();
        assert_eq!(grid, vec![0.5, 1.0, 1.5, 2.0]);
        assert!(beat_grid(-0.1, 120.0, 4).is_err());
        assert!(beat_grid(0.0, 0.0, 4).is_err());
    }

    #[test]
    fn rhythm_analyzer_finish_emits_tempo_only_with_enough_onsets() {
        let mut analyzer = RhythmAnalyzer::default();
        assert!(analyzer.finish(None).unwrap().is_empty());

        for sample in [0, 500, 1000] {
            let samples = AudioBuffer::F32(vec![1.0; 128]);
            let frame = AudioFrame::new(
                Timestamp::new(sample, Timebase::new(1, 1000)),
                1000,
                1,
                &samples,
            )
            .unwrap();
            analyzer.process_frame(&frame).unwrap();
            let silence = AudioBuffer::F32(vec![0.0; 128]);
            let frame = AudioFrame::new(
                Timestamp::new(sample + 250, Timebase::new(1, 1000)),
                1000,
                1,
                &silence,
            )
            .unwrap();
            analyzer.process_frame(&frame).unwrap();
        }
        let events = analyzer
            .finish(Some(Timestamp::new(1500, Timebase::new(1, 1000))))
            .unwrap();
        assert_eq!(events.len(), 1);
        assert!(events[0].label.starts_with("audio:tempo:"));
        assert_eq!(
            events[0].timestamp,
            Some(Timestamp::new(1500, Timebase::new(1, 1000)))
        );
    }
}
