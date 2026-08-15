#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::VecDeque;

use audio_analysis_core::mono_samples;
use audio_contracts::{AnalysisEvent, AudioAnalyzer, AudioFrame, DetectError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for pitch detector config.
pub struct PitchDetectorConfig {
    /// The min frequency hz value.
    pub min_frequency_hz: f32,
    /// The max frequency hz value.
    pub max_frequency_hz: f32,
    /// The confidence threshold value.
    pub confidence_threshold: f32,
}

impl PitchDetectorConfig {
    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if !self.min_frequency_hz.is_finite()
            || !self.max_frequency_hz.is_finite()
            || self.min_frequency_hz <= 0.0
            || self.max_frequency_hz <= self.min_frequency_hz
        {
            return Err(DetectError::InvalidArgument(
                "pitch frequency range must be finite and increasing".to_string(),
            ));
        }
        if !self.confidence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.confidence_threshold)
        {
            return Err(DetectError::InvalidArgument(
                "pitch confidence_threshold must be between 0 and 1".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for PitchDetectorConfig {
    fn default() -> Self {
        Self {
            min_frequency_hz: 50.0,
            max_frequency_hz: 2_000.0,
            confidence_threshold: 0.6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for pitch estimate.
pub struct PitchEstimate {
    /// The frequency hz value.
    pub frequency_hz: Option<f32>,
    /// Confidence score for this value.
    pub confidence: f32,
}

impl PitchEstimate {
    /// Returns unpitched.
    pub fn unpitched(confidence: f32) -> Self {
        Self {
            frequency_hz: None,
            confidence,
        }
    }

    /// Returns midi note.
    pub fn midi_note(&self) -> Option<f32> {
        self.frequency_hz.and_then(frequency_to_midi_note)
    }

    /// Returns note name.
    pub fn note_name(&self) -> Option<String> {
        self.frequency_hz.and_then(frequency_to_note_name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for pitch frame estimate.
pub struct PitchFrameEstimate {
    /// The start seconds value.
    pub start_seconds: f64,
    /// The end seconds value.
    pub end_seconds: f64,
    /// The frequency hz value.
    pub frequency_hz: Option<f32>,
    /// Confidence score for this value.
    pub confidence: f32,
}

impl PitchFrameEstimate {
    /// Returns note name.
    pub fn note_name(&self) -> Option<String> {
        self.frequency_hz.and_then(frequency_to_note_name)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for sung note segment.
pub struct SungNoteSegment {
    /// The start seconds value.
    pub start_seconds: f64,
    /// The end seconds value.
    pub end_seconds: f64,
    /// The frequency hz value.
    pub frequency_hz: f32,
    /// The midi note value.
    pub midi_note: f32,
    /// The note name value.
    pub note_name: String,
    /// Confidence score for this value.
    pub confidence: f32,
    /// The frames value.
    pub frames: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Pitch-class note names.
pub enum NoteName {
    /// C pitch class.
    C,
    /// C sharp pitch class.
    #[serde(rename = "C#")]
    CSharp,
    /// D pitch class.
    D,
    /// D sharp pitch class.
    #[serde(rename = "D#")]
    DSharp,
    /// E pitch class.
    E,
    /// F pitch class.
    F,
    /// F sharp pitch class.
    #[serde(rename = "F#")]
    FSharp,
    /// G pitch class.
    G,
    /// G sharp pitch class.
    #[serde(rename = "G#")]
    GSharp,
    /// A pitch class.
    A,
    /// A sharp pitch class.
    #[serde(rename = "A#")]
    ASharp,
    /// B pitch class.
    B,
}

impl NoteName {
    /// Returns stable display text.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::C => "C",
            Self::CSharp => "C#",
            Self::D => "D",
            Self::DSharp => "D#",
            Self::E => "E",
            Self::F => "F",
            Self::FSharp => "F#",
            Self::G => "G",
            Self::GSharp => "G#",
            Self::A => "A",
            Self::ASharp => "A#",
            Self::B => "B",
        }
    }

    /// Returns from pitch-class index.
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::C),
            1 => Some(Self::CSharp),
            2 => Some(Self::D),
            3 => Some(Self::DSharp),
            4 => Some(Self::E),
            5 => Some(Self::F),
            6 => Some(Self::FSharp),
            7 => Some(Self::G),
            8 => Some(Self::GSharp),
            9 => Some(Self::A),
            10 => Some(Self::ASharp),
            11 => Some(Self::B),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Normalized 12-bin chroma vector.
pub struct ChromaVector {
    /// Pitch-class bins ordered C through B.
    pub bins: [f32; 12],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Pitch-class summary.
pub struct PitchClassSummary {
    /// Normalized chroma vector.
    pub chroma: ChromaVector,
    /// Strongest pitch class when one is detected.
    pub strongest_pitch_class: Option<NoteName>,
}

/// Returns a deterministic pitch-class summary for normalized samples.
pub fn pitch_class_summary(samples: &[f32], sample_rate: u32) -> Result<PitchClassSummary> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "pitch-class samples must contain only finite values".to_string(),
        ));
    }
    if samples.is_empty() {
        return Ok(PitchClassSummary {
            chroma: ChromaVector { bins: [0.0; 12] },
            strongest_pitch_class: None,
        });
    }
    let detector = AutocorrelationPitchDetector::default();
    let estimate = detector.estimate_samples(samples, sample_rate)?;
    let Some(frequency_hz) = estimate.frequency_hz else {
        return Ok(PitchClassSummary {
            chroma: ChromaVector { bins: [0.0; 12] },
            strongest_pitch_class: None,
        });
    };
    let Some(midi) = frequency_to_midi_note(frequency_hz) else {
        return Ok(PitchClassSummary {
            chroma: ChromaVector { bins: [0.0; 12] },
            strongest_pitch_class: None,
        });
    };
    let index = (midi.round() as i32).rem_euclid(12) as usize;
    let mut bins = [0.0_f32; 12];
    bins[index] = estimate.confidence.max(0.0);
    let max = bins.iter().copied().fold(0.0_f32, f32::max);
    if max > f32::EPSILON {
        for bin in &mut bins {
            *bin /= max;
        }
    }
    Ok(PitchClassSummary {
        chroma: ChromaVector { bins },
        strongest_pitch_class: NoteName::from_index(index),
    })
}

/// Returns segment pitch track.
pub fn segment_pitch_track(
    frames: &[PitchFrameEstimate],
    max_gap_seconds: f64,
    min_duration_seconds: f64,
) -> Vec<SungNoteSegment> {
    let mut segments: Vec<SungNoteSegment> = Vec::new();
    for frame in frames {
        let Some(frequency_hz) = frame.frequency_hz else {
            continue;
        };
        let Some(note_name) = frame.note_name() else {
            continue;
        };
        let midi_note = frequency_to_midi_note(frequency_hz).unwrap_or(0.0);
        if let Some(last) = segments.last_mut() {
            let same_note = last.note_name == note_name;
            let small_gap = frame.start_seconds - last.end_seconds <= max_gap_seconds.max(0.0);
            if same_note && small_gap {
                let total_frames = last.frames + 1;
                last.end_seconds = last.end_seconds.max(frame.end_seconds);
                last.frequency_hz =
                    ((last.frequency_hz * last.frames as f32) + frequency_hz) / total_frames as f32;
                last.midi_note =
                    ((last.midi_note * last.frames as f32) + midi_note) / total_frames as f32;
                last.confidence = ((last.confidence * last.frames as f32) + frame.confidence)
                    / total_frames as f32;
                last.frames = total_frames;
                continue;
            }
        }

        segments.push(SungNoteSegment {
            start_seconds: frame.start_seconds,
            end_seconds: frame.end_seconds,
            frequency_hz,
            midi_note,
            note_name,
            confidence: frame.confidence,
            frames: 1,
        });
    }

    segments
        .into_iter()
        .filter(|segment| segment.end_seconds > segment.start_seconds)
        .filter(|segment| {
            segment.end_seconds - segment.start_seconds >= min_duration_seconds.max(0.0)
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for pitch smoother.
pub struct PitchSmoother {
    window_size: usize,
    history: VecDeque<PitchEstimate>,
}

impl PitchSmoother {
    /// Creates a new value.
    pub fn new(window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(DetectError::InvalidArgument(
                "pitch smoother window_size must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            window_size,
            history: VecDeque::new(),
        })
    }

    /// Returns smooth.
    pub fn smooth(&mut self, estimate: PitchEstimate) -> PitchEstimate {
        self.history.push_back(estimate);
        while self.history.len() > self.window_size {
            self.history.pop_front();
        }

        let mut frequencies = self
            .history
            .iter()
            .filter_map(|estimate| estimate.frequency_hz)
            .collect::<Vec<_>>();
        if frequencies.is_empty() {
            let confidence = self
                .history
                .iter()
                .map(|estimate| estimate.confidence)
                .fold(0.0_f32, f32::max);
            return PitchEstimate::unpitched(confidence);
        }

        frequencies.sort_by(f32::total_cmp);
        let confidence = self
            .history
            .iter()
            .map(|estimate| estimate.confidence)
            .sum::<f32>()
            / self.history.len() as f32;
        PitchEstimate {
            frequency_hz: Some(frequencies[frequencies.len() / 2]),
            confidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for autocorrelation pitch detector.
pub struct AutocorrelationPitchDetector {
    name: String,
    /// The config value.
    pub config: PitchDetectorConfig,
}

impl AutocorrelationPitchDetector {
    /// Creates a new value.
    pub fn new(config: PitchDetectorConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            name: "audio_pitch".to_string(),
            config,
        })
    }

    /// Returns estimate frame.
    pub fn estimate_frame(&self, frame: &AudioFrame<'_>) -> Result<PitchEstimate> {
        let mono = mono_samples(frame)?;
        self.estimate_samples(&mono.samples, mono.sample_rate)
    }

    /// Returns estimate samples.
    pub fn estimate_samples(&self, samples: &[f32], sample_rate: u32) -> Result<PitchEstimate> {
        if sample_rate == 0 {
            return Err(DetectError::InvalidAudioFormat {
                sample_rate,
                channels: 1,
            });
        }
        if samples.len() < 3 {
            return Ok(PitchEstimate::unpitched(0.0));
        }

        let min_lag = (sample_rate as f32 / self.config.max_frequency_hz).floor() as usize;
        let max_lag = (sample_rate as f32 / self.config.min_frequency_hz).ceil() as usize;
        let max_lag = max_lag.min(samples.len().saturating_sub(2));
        if min_lag == 0 || min_lag >= max_lag {
            return Ok(PitchEstimate::unpitched(0.0));
        }

        let centered = remove_dc(samples);
        let mut best_lag = min_lag;
        let mut best_score = 0.0_f32;
        let mut scores = Vec::with_capacity(max_lag - min_lag + 1);

        for lag in min_lag..=max_lag {
            let score = normalized_autocorrelation(&centered, lag);
            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
            scores.push(score);
        }

        let (selected_lag, selected_score) =
            first_confident_peak(min_lag, &scores, self.config.confidence_threshold)
                .unwrap_or((best_lag, best_score));

        if selected_score < self.config.confidence_threshold {
            return Ok(PitchEstimate::unpitched(best_score.max(0.0)));
        }

        let refined_lag = parabolic_lag(selected_lag, min_lag, &scores);
        Ok(PitchEstimate {
            frequency_hz: Some(sample_rate as f32 / refined_lag),
            confidence: selected_score.clamp(0.0, 1.0),
        })
    }
}

impl Default for AutocorrelationPitchDetector {
    fn default() -> Self {
        Self::new(PitchDetectorConfig::default()).expect("default pitch config is valid")
    }
}

impl AudioAnalyzer for AutocorrelationPitchDetector {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Vec<AnalysisEvent>> {
        let estimate = self.estimate_frame(frame)?;
        let Some(frequency_hz) = estimate.frequency_hz else {
            return Ok(Vec::new());
        };
        Ok(vec![AnalysisEvent::new(
            self.name(),
            format!("audio:pitch:{frequency_hz:.2}hz"),
        )
        .at_timestamp(frame.timestamp)
        .score(estimate.confidence)])
    }
}

fn remove_dc(samples: &[f32]) -> Vec<f32> {
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    samples.iter().map(|sample| sample - mean).collect()
}

fn normalized_autocorrelation(samples: &[f32], lag: usize) -> f32 {
    let len = samples.len().saturating_sub(lag);
    if len == 0 {
        return 0.0;
    }
    let mut numerator = 0.0;
    let mut energy_a = 0.0;
    let mut energy_b = 0.0;
    for index in 0..len {
        let a = samples[index];
        let b = samples[index + lag];
        numerator += a * b;
        energy_a += a * a;
        energy_b += b * b;
    }
    let denom = (energy_a * energy_b).sqrt();
    if denom <= f32::EPSILON {
        0.0
    } else {
        numerator / denom
    }
}

fn first_confident_peak(min_lag: usize, scores: &[f32], threshold: f32) -> Option<(usize, f32)> {
    scores
        .windows(3)
        .enumerate()
        .find(|(_, scores)| {
            scores[1] >= threshold && scores[1] >= scores[0] && scores[1] >= scores[2]
        })
        .map(|(index, scores)| (min_lag + index + 1, scores[1]))
}

fn parabolic_lag(best_lag: usize, min_lag: usize, scores: &[f32]) -> f32 {
    let index = best_lag - min_lag;
    if index == 0 || index + 1 >= scores.len() {
        return best_lag as f32;
    }
    let left = scores[index - 1];
    let center = scores[index];
    let right = scores[index + 1];
    let denom = left - 2.0 * center + right;
    if denom.abs() <= f32::EPSILON {
        best_lag as f32
    } else {
        best_lag as f32 + 0.5 * (left - right) / denom
    }
}

/// Returns frequency to midi note.
pub fn frequency_to_midi_note(frequency_hz: f32) -> Option<f32> {
    (frequency_hz.is_finite() && frequency_hz > 0.0)
        .then_some(69.0 + 12.0 * (frequency_hz / 440.0).log2())
}

/// Returns frequency to note name.
pub fn frequency_to_note_name(frequency_hz: f32) -> Option<String> {
    let midi = frequency_to_midi_note(frequency_hz)?;
    let rounded = midi.round() as i32;
    let octave = rounded.div_euclid(12) - 1;
    let note = match rounded.rem_euclid(12) {
        0 => "C",
        1 => "C#",
        2 => "D",
        3 => "D#",
        4 => "E",
        5 => "F",
        6 => "F#",
        7 => "G",
        8 => "G#",
        9 => "A",
        10 => "A#",
        11 => "B",
        _ => return None,
    };
    Some(format!("{note}{octave}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_contracts::{AudioBuffer, AudioFrame, Timebase, Timestamp};

    fn white_noise(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let value = ((state >> 32) as u32) as f32 / u32::MAX as f32;
                value * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_approx_eq(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    fn sine(freq_hz: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let samples = (sample_rate as f32 * seconds) as usize;
        (0..samples)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    #[test]
    fn config_validation_rejects_invalid_values() {
        assert!(PitchDetectorConfig {
            min_frequency_hz: 0.0,
            max_frequency_hz: 2_000.0,
            confidence_threshold: 0.6,
        }
        .validate()
        .is_err());
        assert!(PitchDetectorConfig {
            min_frequency_hz: 200.0,
            max_frequency_hz: 100.0,
            confidence_threshold: 0.6,
        }
        .validate()
        .is_err());
        assert!(PitchDetectorConfig {
            min_frequency_hz: 50.0,
            max_frequency_hz: f32::INFINITY,
            confidence_threshold: 0.6,
        }
        .validate()
        .is_err());
        assert!(PitchDetectorConfig {
            min_frequency_hz: 50.0,
            max_frequency_hz: 2_000.0,
            confidence_threshold: f32::NAN,
        }
        .validate()
        .is_err());
        assert!(PitchDetectorConfig {
            min_frequency_hz: 50.0,
            max_frequency_hz: 2_000.0,
            confidence_threshold: 1.1,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn estimates_pitch_for_sine_wave() {
        let detector = AutocorrelationPitchDetector::default();
        let estimate = detector
            .estimate_samples(&sine(440.0, 44_100, 0.1), 44_100)
            .unwrap();
        let frequency = estimate.frequency_hz.unwrap();
        assert!((frequency - 440.0).abs() < 3.0);
        assert!(estimate.confidence > 0.8);
    }

    #[test]
    fn estimates_pitch_across_reference_frequencies() {
        let detector = AutocorrelationPitchDetector::default();
        for frequency in [80.0, 220.0, 440.0, 1_000.0] {
            let estimate = detector
                .estimate_samples(&sine(frequency, 48_000, 0.12), 48_000)
                .unwrap();
            let actual = estimate.frequency_hz.unwrap();
            assert!(
                (actual - frequency).abs() <= frequency * 0.02,
                "expected {frequency}, got {actual}"
            );
        }
    }

    #[test]
    fn returns_unpitched_for_silence() {
        let detector = AutocorrelationPitchDetector::default();
        let estimate = detector.estimate_samples(&vec![0.0; 2048], 48_000).unwrap();
        assert_eq!(estimate.frequency_hz, None);
        assert_eq!(estimate.confidence, 0.0);
        assert_eq!(
            detector
                .estimate_samples(&[0.0, 1.0], 48_000)
                .unwrap()
                .frequency_hz,
            None
        );
        assert!(detector.estimate_samples(&[0.0; 2048], 0).is_err());
    }

    #[test]
    fn estimates_pitch_with_dc_offset_and_seeded_noise() {
        let detector = AutocorrelationPitchDetector::default();
        let with_dc = sine(220.0, 48_000, 0.12)
            .into_iter()
            .map(|sample| sample * 0.5 + 0.25)
            .collect::<Vec<_>>();
        let dc_estimate = detector.estimate_samples(&with_dc, 48_000).unwrap();
        assert_approx_eq(dc_estimate.frequency_hz.unwrap(), 220.0, 220.0 * 0.02);

        let mut noisy = sine(440.0, 48_000, 0.12);
        for (sample, noise) in noisy.iter_mut().zip(white_noise(7, 48_000)) {
            *sample = *sample * 0.85 + noise * 0.05;
        }
        let noisy_estimate = detector.estimate_samples(&noisy, 48_000).unwrap();
        assert_approx_eq(noisy_estimate.frequency_hz.unwrap(), 440.0, 440.0 * 0.05);
    }

    #[test]
    fn emits_audio_pipeline_event() {
        let mut detector = AutocorrelationPitchDetector::default();
        let samples = AudioBuffer::F32(sine(220.0, 44_100, 0.1));
        let frame = AudioFrame::new(
            Timestamp::new(0, Timebase::new(1, 44_100)),
            44_100,
            1,
            &samples,
        )
        .unwrap();
        let events = detector.process_frame(&frame).unwrap();
        assert_eq!(events.len(), 1);
        assert!(events[0].label.starts_with("audio:pitch:"));
        assert_eq!(events[0].timestamp, Some(frame.timestamp));
        assert!(events[0].score.unwrap() > 0.6);
    }

    #[test]
    fn pitch_helpers_convert_to_midi_and_note_names() {
        let estimate = PitchEstimate {
            frequency_hz: Some(440.0),
            confidence: 0.9,
        };
        assert!((estimate.midi_note().unwrap() - 69.0).abs() < 0.01);
        assert_eq!(estimate.note_name().as_deref(), Some("A4"));
        assert_eq!(frequency_to_note_name(261.63).as_deref(), Some("C4"));
        assert!(frequency_to_midi_note(0.0).is_none());
    }

    #[test]
    fn pitch_class_summary_handles_a4_and_silence() {
        let a4 = pitch_class_summary(&sine(440.0, 8_192, 0.125), 8_192).unwrap();
        assert_eq!(a4.strongest_pitch_class, Some(NoteName::A));
        assert_eq!(a4.chroma.bins[9], 1.0);

        let silence = pitch_class_summary(&vec![0.0; 1024], 8_192).unwrap();
        assert_eq!(silence.strongest_pitch_class, None);
        assert!(silence.chroma.bins.iter().all(|bin| *bin == 0.0));

        let empty = pitch_class_summary(&[], 8_192).unwrap();
        assert_eq!(empty.strongest_pitch_class, None);
        assert!(pitch_class_summary(&[f32::NAN], 8_192).is_err());
    }

    #[test]
    fn smoother_median_stabilizes_pitch_history() {
        let mut smoother = PitchSmoother::new(3).unwrap();
        let a = smoother.smooth(PitchEstimate {
            frequency_hz: Some(440.0),
            confidence: 0.8,
        });
        let b = smoother.smooth(PitchEstimate {
            frequency_hz: Some(460.0),
            confidence: 0.8,
        });
        let c = smoother.smooth(PitchEstimate {
            frequency_hz: Some(442.0),
            confidence: 0.8,
        });
        assert_eq!(a.frequency_hz, Some(440.0));
        assert_eq!(b.frequency_hz, Some(460.0));
        assert_eq!(c.frequency_hz, Some(442.0));
    }

    #[test]
    fn segment_pitch_track_merges_adjacent_notes() {
        let segments = segment_pitch_track(
            &[
                PitchFrameEstimate {
                    start_seconds: 0.0,
                    end_seconds: 0.08,
                    frequency_hz: Some(440.0),
                    confidence: 0.8,
                },
                PitchFrameEstimate {
                    start_seconds: 0.08,
                    end_seconds: 0.16,
                    frequency_hz: Some(441.0),
                    confidence: 0.85,
                },
                PitchFrameEstimate {
                    start_seconds: 0.30,
                    end_seconds: 0.44,
                    frequency_hz: Some(493.88),
                    confidence: 0.9,
                },
            ],
            0.08,
            0.1,
        );

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].note_name, "A4");
        assert!(segments[0].end_seconds >= 0.16);
        assert_eq!(segments[1].note_name, "B4");
    }
}
