//! Whole-track rhythm analysis aimed at music and DJ preparation.
//!
//! The legacy onset/tempo helpers in the crate remain intentionally small and
//! deterministic. This module is the higher-quality path: it derives a true
//! spectral-flux novelty curve, estimates multiple tempo hypotheses from its
//! autocorrelation, tracks a globally consistent beat sequence with dynamic
//! programming, and infers a 4/4 downbeat phase from beat accents.

use audio_analysis_fourier::{spectrogram, StftConfig};
use audio_contracts::{DetectError, Result};

/// Configuration for whole-track rhythm analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackRhythmConfig {
    /// Minimum tempo considered by the estimator.
    pub min_bpm: f32,
    /// Maximum tempo considered by the estimator.
    pub max_bpm: f32,
    /// FFT size used for spectral-flux analysis.
    pub fft_size: usize,
    /// Hop size used for spectral-flux analysis.
    pub hop_size: usize,
    /// Number of tempo hypotheses retained in the result.
    pub tempo_candidate_count: usize,
    /// Number of beats per bar used by the downbeat phase estimator.
    pub beats_per_bar: usize,
    /// Strength of the beat-period transition penalty in dynamic programming.
    pub beat_tightness: f32,
}

impl Default for TrackRhythmConfig {
    fn default() -> Self {
        Self {
            min_bpm: 55.0,
            max_bpm: 220.0,
            fft_size: 2048,
            hop_size: 512,
            tempo_candidate_count: 5,
            beats_per_bar: 4,
            beat_tightness: 1.25,
        }
    }
}

impl TrackRhythmConfig {
    /// Validates the configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.min_bpm.is_finite()
            || !self.max_bpm.is_finite()
            || self.min_bpm <= 0.0
            || self.max_bpm <= self.min_bpm
        {
            return Err(DetectError::InvalidArgument(
                "track rhythm BPM range must be finite, positive, and increasing".to_string(),
            ));
        }
        StftConfig::new(self.fft_size, self.hop_size)?;
        if self.tempo_candidate_count == 0 {
            return Err(DetectError::InvalidArgument(
                "tempo_candidate_count must be greater than zero".to_string(),
            ));
        }
        if self.beats_per_bar == 0 {
            return Err(DetectError::InvalidArgument(
                "beats_per_bar must be greater than zero".to_string(),
            ));
        }
        if !self.beat_tightness.is_finite() || self.beat_tightness < 0.0 {
            return Err(DetectError::InvalidArgument(
                "beat_tightness must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

/// One tempo hypothesis returned by the whole-track estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoCandidate {
    /// Tempo in beats per minute.
    pub bpm: f32,
    /// Normalized autocorrelation support in the range 0..=1.
    pub score: f32,
}

/// One tracked beat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackedBeat {
    /// Beat time in seconds.
    pub timestamp_seconds: f64,
    /// Novelty strength at the tracked beat.
    pub strength: f32,
    /// One-based position inside the inferred bar.
    pub beat_in_bar: usize,
    /// Whether this beat is the inferred downbeat.
    pub downbeat: bool,
}

/// Whole-track rhythm result.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackRhythmAnalysis {
    /// Selected tempo, if rhythmic evidence was sufficient.
    pub bpm: Option<f32>,
    /// Confidence in the selected tempo and beat path.
    pub confidence: f32,
    /// Ranked alternative tempo hypotheses.
    pub tempo_candidates: Vec<TempoCandidate>,
    /// Globally tracked beats.
    pub beats: Vec<TrackedBeat>,
    /// Convenience list of inferred downbeat times.
    pub downbeats: Vec<f64>,
    /// Confidence that one bar phase is more accented than the alternatives.
    pub downbeat_confidence: f32,
    /// Analysis hop duration in seconds.
    pub hop_seconds: f64,
}

#[derive(Debug, Clone)]
struct OnsetFeatures {
    novelty: Vec<f32>,
    low_energy: Vec<f32>,
    timestamps: Vec<f64>,
}

/// Analyzes a mono, normalized music track for tempo, beats, and downbeats.
///
/// The returned tempo candidates deliberately remain visible because half/double
/// tempo ambiguity is intrinsic to musical audio and is useful information for
/// DJ applications.
pub fn analyze_rhythm_track(
    samples: &[f32],
    sample_rate: u32,
    config: TrackRhythmConfig,
) -> Result<TrackRhythmAnalysis> {
    config.validate()?;
    if sample_rate == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate,
            channels: 1,
        });
    }
    if samples.is_empty() {
        return Err(DetectError::InvalidArgument(
            "track rhythm samples must not be empty".to_string(),
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "track rhythm samples must contain only finite values".to_string(),
        ));
    }

    let features = spectral_onset_features(samples, sample_rate, config)?;
    let hop_seconds = config.hop_size as f64 / sample_rate as f64;
    if features.novelty.is_empty()
        || features
            .novelty
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
            <= f32::EPSILON
    {
        return Ok(TrackRhythmAnalysis {
            bpm: None,
            confidence: 0.0,
            tempo_candidates: Vec::new(),
            beats: Vec::new(),
            downbeats: Vec::new(),
            downbeat_confidence: 0.0,
            hop_seconds,
        });
    }

    let frame_rate = sample_rate as f32 / config.hop_size as f32;
    let candidates = estimate_tempo_candidates(
        &features.novelty,
        frame_rate,
        config.min_bpm,
        config.max_bpm,
        config.tempo_candidate_count,
    );
    let Some(selected) = candidates.first().copied() else {
        return Ok(TrackRhythmAnalysis {
            bpm: None,
            confidence: 0.0,
            tempo_candidates: Vec::new(),
            beats: Vec::new(),
            downbeats: Vec::new(),
            downbeat_confidence: 0.0,
            hop_seconds,
        });
    };

    let beat_frames = track_beat_frames(
        &features.novelty,
        frame_rate,
        selected.bpm,
        config.beat_tightness,
    );
    let (phase, downbeat_confidence) = infer_downbeat_phase(
        &beat_frames,
        &features.novelty,
        &features.low_energy,
        config.beats_per_bar,
    );

    let beats = beat_frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let beat_in_bar = ((index + config.beats_per_bar - phase) % config.beats_per_bar) + 1;
            TrackedBeat {
                timestamp_seconds: features.timestamps[*frame],
                strength: features.novelty[*frame],
                beat_in_bar,
                downbeat: beat_in_bar == 1,
            }
        })
        .collect::<Vec<_>>();
    let downbeats = beats
        .iter()
        .filter(|beat| beat.downbeat)
        .map(|beat| beat.timestamp_seconds)
        .collect::<Vec<_>>();

    let runner_up = candidates.get(1).map_or(0.0, |candidate| candidate.score);
    let tempo_margin = if selected.score > f32::EPSILON {
        ((selected.score - runner_up).max(0.0) / selected.score).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let beat_support = if beat_frames.is_empty() {
        0.0
    } else {
        beat_frames
            .iter()
            .map(|frame| features.novelty[*frame])
            .sum::<f32>()
            / beat_frames.len() as f32
    };
    let confidence = (0.55 * selected.score + 0.25 * tempo_margin + 0.20 * beat_support)
        .clamp(0.0, 1.0);

    Ok(TrackRhythmAnalysis {
        bpm: Some(selected.bpm),
        confidence,
        tempo_candidates: candidates,
        beats,
        downbeats,
        downbeat_confidence,
        hop_seconds,
    })
}

fn spectral_onset_features(
    samples: &[f32],
    sample_rate: u32,
    config: TrackRhythmConfig,
) -> Result<OnsetFeatures> {
    let stft = StftConfig::new(config.fft_size, config.hop_size)?.pad_final_frame(true);
    let frames = spectrogram(samples, sample_rate, &stft)?;
    if frames.is_empty() {
        return Ok(OnsetFeatures {
            novelty: Vec::new(),
            low_energy: Vec::new(),
            timestamps: Vec::new(),
        });
    }

    let mut novelty = Vec::with_capacity(frames.len());
    let mut low_energy = Vec::with_capacity(frames.len());
    let mut timestamps = Vec::with_capacity(frames.len());
    novelty.push(0.0);

    for frame in &frames {
        let low = frame
            .spectrum
            .bins
            .iter()
            .filter(|bin| (30.0..=220.0).contains(&bin.frequency_hz))
            .map(|bin| bin.power)
            .sum::<f32>();
        low_energy.push(low.sqrt());
        timestamps.push(
            frame.start_seconds + config.fft_size as f64 / (2.0 * sample_rate as f64),
        );
    }

    for pair in frames.windows(2) {
        let previous = &pair[0].spectrum;
        let current = &pair[1].spectrum;
        let flux = previous
            .bins
            .iter()
            .zip(current.bins.iter())
            .skip(1)
            .map(|(left, right)| {
                let previous_log = (1.0 + 64.0 * left.magnitude).ln();
                let current_log = (1.0 + 64.0 * right.magnitude).ln();
                let frequency_weight = if right.frequency_hz <= 220.0 {
                    1.5
                } else if right.frequency_hz <= 2_000.0 {
                    1.0
                } else {
                    0.55
                };
                (current_log - previous_log).max(0.0) * frequency_weight
            })
            .sum::<f32>();
        novelty.push(flux);
    }

    adaptive_whiten(&mut novelty, 16);
    normalize_nonnegative(&mut novelty);
    normalize_nonnegative(&mut low_energy);

    Ok(OnsetFeatures {
        novelty,
        low_energy,
        timestamps,
    })
}

fn adaptive_whiten(values: &mut [f32], radius: usize) {
    if values.is_empty() {
        return;
    }
    let original = values.to_vec();
    for (index, value) in values.iter_mut().enumerate() {
        let start = index.saturating_sub(radius);
        let end = (index + radius + 1).min(original.len());
        let mean = original[start..end].iter().sum::<f32>() / (end - start) as f32;
        *value = (original[index] - mean).max(0.0);
    }
}

fn normalize_nonnegative(values: &mut [f32]) {
    let max = values.iter().copied().fold(0.0_f32, f32::max);
    if max > f32::EPSILON {
        for value in values {
            *value = (*value / max).clamp(0.0, 1.0);
        }
    }
}

fn estimate_tempo_candidates(
    novelty: &[f32],
    frame_rate: f32,
    min_bpm: f32,
    max_bpm: f32,
    candidate_count: usize,
) -> Vec<TempoCandidate> {
    if novelty.len() < 3 || frame_rate <= 0.0 {
        return Vec::new();
    }
    let min_lag = ((60.0 * frame_rate / max_bpm).floor() as usize).max(1);
    let max_lag = ((60.0 * frame_rate / min_bpm).ceil() as usize)
        .min(novelty.len().saturating_sub(1));
    if min_lag > max_lag {
        return Vec::new();
    }
    let total_energy = novelty.iter().map(|value| value * value).sum::<f32>();
    if total_energy <= f32::EPSILON {
        return Vec::new();
    }

    let mut all = (min_lag..=max_lag)
        .map(|lag| {
            let support = novelty
                .iter()
                .skip(lag)
                .zip(novelty.iter())
                .map(|(right, left)| right * left)
                .sum::<f32>();
            TempoCandidate {
                bpm: 60.0 * frame_rate / lag as f32,
                score: (support / total_energy).clamp(0.0, 1.0),
            }
        })
        .collect::<Vec<_>>();

    // Prefer local autocorrelation peaks over adjacent lag bins.
    all.retain(|candidate| candidate.score > 0.0);
    all.sort_by(|left, right| right.score.total_cmp(&left.score));

    let mut selected: Vec<TempoCandidate> = Vec::new();
    for candidate in all {
        if selected.iter().any(|existing| {
            ((existing.bpm - candidate.bpm).abs() / existing.bpm.max(candidate.bpm)) < 0.025
        }) {
            continue;
        }
        selected.push(candidate);
        if selected.len() == candidate_count {
            break;
        }
    }
    selected
}

fn track_beat_frames(
    novelty: &[f32],
    frame_rate: f32,
    bpm: f32,
    tightness: f32,
) -> Vec<usize> {
    if novelty.is_empty() || frame_rate <= 0.0 || bpm <= 0.0 {
        return Vec::new();
    }
    let period = frame_rate * 60.0 / bpm;
    if !period.is_finite() || period < 1.0 {
        return Vec::new();
    }
    let min_gap = (period * 0.75).floor().max(1.0) as usize;
    let max_gap = (period * 1.33).ceil().max(min_gap as f32) as usize;
    let mut cumulative = vec![0.0_f32; novelty.len()];
    let mut back = vec![None; novelty.len()];

    for index in 0..novelty.len() {
        let mut best_score = 0.0_f32;
        let mut best_previous = None;
        for gap in min_gap..=max_gap {
            if gap > index {
                break;
            }
            let previous = index - gap;
            let ratio = gap as f32 / period;
            let transition = -tightness * ratio.ln().powi(2);
            let score = cumulative[previous] + transition;
            if best_previous.is_none() || score > best_score {
                best_score = score;
                best_previous = Some(previous);
            }
        }
        if best_score > 0.0 {
            cumulative[index] = novelty[index] + best_score;
            back[index] = best_previous;
        } else {
            cumulative[index] = novelty[index];
        }
    }

    let Some(mut endpoint) = cumulative
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
    else {
        return Vec::new();
    };
    if cumulative[endpoint] <= f32::EPSILON {
        return Vec::new();
    }

    let mut path = vec![endpoint];
    while let Some(previous) = back[endpoint] {
        if previous >= endpoint {
            break;
        }
        endpoint = previous;
        path.push(endpoint);
    }
    path.reverse();

    // Very short paths are not useful as beat grids.
    if path.len() < 2 {
        return Vec::new();
    }
    path
}

fn infer_downbeat_phase(
    beat_frames: &[usize],
    novelty: &[f32],
    low_energy: &[f32],
    beats_per_bar: usize,
) -> (usize, f32) {
    if beats_per_bar == 0 || beat_frames.is_empty() {
        return (0, 0.0);
    }
    let mut sums = vec![0.0_f32; beats_per_bar];
    let mut counts = vec![0_usize; beats_per_bar];
    for (beat_index, frame) in beat_frames.iter().enumerate() {
        let novelty_strength = novelty.get(*frame).copied().unwrap_or(0.0);
        let low = low_energy.get(*frame).copied().unwrap_or(0.0);
        let accent = 0.65 * novelty_strength + 0.35 * low;
        let phase = beat_index % beats_per_bar;
        sums[phase] += accent;
        counts[phase] += 1;
    }
    let mut scores = sums
        .iter()
        .zip(counts.iter())
        .enumerate()
        .map(|(phase, (sum, count))| {
            let score = if *count == 0 {
                0.0
            } else {
                *sum / *count as f32
            };
            (phase, score)
        })
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| right.1.total_cmp(&left.1));
    let best = scores.first().copied().unwrap_or((0, 0.0));
    let second = scores.get(1).copied().unwrap_or((0, 0.0));
    let confidence = if best.1 <= f32::EPSILON || beat_frames.len() < beats_per_bar * 2 {
        0.0
    } else {
        ((best.1 - second.1).max(0.0) / best.1).clamp(0.0, 1.0)
    };
    (best.0, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pulse_envelope(period: usize, beat_count: usize) -> Vec<f32> {
        let mut values = vec![0.0; period * beat_count + 1];
        for beat in 0..beat_count {
            values[beat * period] = if beat % 4 == 0 { 1.0 } else { 0.7 };
        }
        values
    }

    fn dance_click_track(sample_rate: u32, bpm: f32, beats: usize) -> Vec<f32> {
        let period = (sample_rate as f32 * 60.0 / bpm).round() as usize;
        let mut samples = vec![0.0; period * beats + sample_rate as usize];
        for beat in 0..beats {
            let start = beat * period;
            let amplitude = if beat % 4 == 0 { 1.0 } else { 0.72 };
            for offset in 0..80.min(samples.len().saturating_sub(start)) {
                let decay = 1.0 - offset as f32 / 80.0;
                samples[start + offset] += amplitude * decay;
            }
        }
        samples
    }

    #[test]
    fn tempo_candidates_recover_regular_pulse_rate() {
        let novelty = pulse_envelope(50, 20);
        let candidates = estimate_tempo_candidates(&novelty, 100.0, 55.0, 220.0, 5);
        assert!(candidates.iter().any(|candidate| (candidate.bpm - 120.0).abs() < 1.0));
        assert!((candidates[0].bpm - 120.0).abs() < 1.0);
    }

    #[test]
    fn dynamic_programming_tracks_regular_beats() {
        let novelty = pulse_envelope(40, 16);
        let beats = track_beat_frames(&novelty, 80.0, 120.0, 1.25);
        assert!(beats.len() >= 14);
        assert!(beats.windows(2).all(|pair| {
            let gap = pair[1] - pair[0];
            (30..=54).contains(&gap)
        }));
    }

    #[test]
    fn downbeat_phase_finds_four_beat_accent() {
        let novelty = pulse_envelope(20, 16);
        let beats = (0..16).map(|beat| beat * 20).collect::<Vec<_>>();
        let mut low = vec![0.1; novelty.len()];
        for beat in (0..16).step_by(4) {
            low[beat * 20] = 1.0;
        }
        let (phase, confidence) = infer_downbeat_phase(&beats, &novelty, &low, 4);
        assert_eq!(phase, 0);
        assert!(confidence > 0.0);
    }

    #[test]
    fn whole_track_path_returns_music_ready_result() {
        let sample_rate = 8_000;
        let samples = dance_click_track(sample_rate, 120.0, 24);
        let analysis = analyze_rhythm_track(
            &samples,
            sample_rate,
            TrackRhythmConfig {
                fft_size: 512,
                hop_size: 128,
                ..TrackRhythmConfig::default()
            },
        )
        .expect("track rhythm analysis");
        assert!(analysis
            .tempo_candidates
            .iter()
            .any(|candidate| (candidate.bpm - 120.0).abs() < 4.0));
        assert!(!analysis.beats.is_empty());
    }

    #[test]
    fn silence_returns_an_empty_analysis() {
        let analysis = analyze_rhythm_track(
            &vec![0.0; 16_000],
            8_000,
            TrackRhythmConfig {
                fft_size: 512,
                hop_size: 128,
                ..TrackRhythmConfig::default()
            },
        )
        .expect("silence");
        assert_eq!(analysis.bpm, None);
        assert!(analysis.beats.is_empty());
    }
}