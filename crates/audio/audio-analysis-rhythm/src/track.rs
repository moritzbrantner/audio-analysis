//! Whole-track rhythm analysis aimed at music and DJ preparation.
//!
//! This path keeps multiple tempo hypotheses visible, but no longer treats the
//! strongest autocorrelation lag as sufficient evidence by itself. Each tempo
//! candidate is rescored from the beat path it can actually support. The beat
//! tracker is intentionally elastic so recorded/live material can produce a
//! variable beat map instead of being snapped to one synthetic constant grid.

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
    /// Final score after autocorrelation and beat-path support are combined.
    pub score: f32,
    /// Normalized onset-envelope autocorrelation support.
    pub autocorrelation_score: f32,
    /// Support from the candidate's actual dynamic-programming beat path.
    pub beat_support: f32,
}

/// One point in the variable-tempo beat map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoPoint {
    /// Beat timestamp in seconds.
    pub timestamp_seconds: f64,
    /// Locally smoothed tempo at this beat.
    pub bpm: f32,
    /// Confidence from local onset strength and interval smoothness.
    pub confidence: f32,
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
    /// Locally smoothed tempo at this beat when enough neighbours exist.
    pub local_bpm: Option<f32>,
}

/// Per-frame musical descriptors retained for structural analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralDescriptor {
    /// Frame-center timestamp in seconds.
    pub timestamp_seconds: f64,
    /// Normalized onset novelty.
    pub onset_novelty: f32,
    /// Normalized low-band energy.
    pub low_energy: f32,
    /// Normalized mid-band energy.
    pub mid_energy: f32,
    /// Normalized high-band energy.
    pub high_energy: f32,
    /// Lightweight normalized spectral pitch-class energy, C through B.
    pub chroma: [f32; 12],
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
    /// Globally tracked, individually timestamped beats.
    pub beats: Vec<TrackedBeat>,
    /// Convenience list of inferred downbeat times.
    pub downbeats: Vec<f64>,
    /// Confidence that one bar phase is more accented than the alternatives.
    pub downbeat_confidence: f32,
    /// Analysis hop duration in seconds.
    pub hop_seconds: f64,
    /// Local tempo attached to the real beat positions rather than a synthetic grid.
    pub tempo_map: Vec<TempoPoint>,
    /// Multi-band, onset, and harmonic descriptors for structural analysis.
    pub structural_descriptors: Vec<StructuralDescriptor>,
}

#[derive(Debug, Clone)]
struct OnsetFeatures {
    novelty: Vec<f32>,
    low_energy: Vec<f32>,
    timestamps: Vec<f64>,
    structural_descriptors: Vec<StructuralDescriptor>,
}

/// Analyzes a mono, normalized music track for tempo, beats, and downbeats.
///
/// The returned candidates deliberately remain visible because half/double
/// tempo ambiguity is intrinsic to musical audio. The selected candidate is
/// the one with the strongest combination of autocorrelation and real beat-path
/// evidence, rather than simply the strongest autocorrelation lag.
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
        || features.novelty.iter().copied().fold(0.0_f32, f32::max) <= f32::EPSILON
    {
        return Ok(empty_analysis(hop_seconds, features.structural_descriptors));
    }

    let frame_rate = sample_rate as f32 / config.hop_size as f32;
    let raw_count = config.tempo_candidate_count.saturating_mul(3).min(24);
    let candidates = estimate_tempo_candidates(
        &features.novelty,
        frame_rate,
        config.min_bpm,
        config.max_bpm,
        raw_count,
    );
    let candidates = rescore_tempo_candidates(
        candidates,
        &features.novelty,
        frame_rate,
        config.beat_tightness,
        config.tempo_candidate_count,
    );
    let Some(selected) = candidates.first().copied() else {
        return Ok(empty_analysis(hop_seconds, features.structural_descriptors));
    };

    let beat_frames = track_beat_frames(
        &features.novelty,
        frame_rate,
        selected.bpm,
        config.beat_tightness,
    );
    let tempo_map = build_tempo_map(&beat_frames, &features.timestamps, &features.novelty);
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
                local_bpm: tempo_map.get(index).map(|point| point.bpm),
            }
        })
        .collect::<Vec<_>>();
    let downbeats = beats
        .iter()
        .filter(|beat| beat.downbeat)
        .map(|beat| beat.timestamp_seconds)
        .collect::<Vec<_>>();

    let runner_up = candidates.get(1).map_or(0.0, |candidate| candidate.score);
    let candidate_margin = if selected.score > f32::EPSILON {
        ((selected.score - runner_up).max(0.0) / selected.score).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let confidence =
        (0.60 * selected.score + 0.25 * candidate_margin + 0.15 * downbeat_confidence)
            .clamp(0.0, 1.0);

    Ok(TrackRhythmAnalysis {
        bpm: Some(selected.bpm),
        confidence,
        tempo_candidates: candidates,
        beats,
        downbeats,
        downbeat_confidence,
        hop_seconds,
        tempo_map,
        structural_descriptors: features.structural_descriptors,
    })
}

fn empty_analysis(
    hop_seconds: f64,
    structural_descriptors: Vec<StructuralDescriptor>,
) -> TrackRhythmAnalysis {
    TrackRhythmAnalysis {
        bpm: None,
        confidence: 0.0,
        tempo_candidates: Vec::new(),
        beats: Vec::new(),
        downbeats: Vec::new(),
        downbeat_confidence: 0.0,
        hop_seconds,
        tempo_map: Vec::new(),
        structural_descriptors,
    }
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
            structural_descriptors: Vec::new(),
        });
    }

    let mut novelty = Vec::with_capacity(frames.len());
    let mut low_energy = Vec::with_capacity(frames.len());
    let mut mid_energy = Vec::with_capacity(frames.len());
    let mut high_energy = Vec::with_capacity(frames.len());
    let mut chroma = Vec::with_capacity(frames.len());
    let mut timestamps = Vec::with_capacity(frames.len());
    novelty.push(0.0);

    for frame in &frames {
        let mut low = 0.0_f32;
        let mut mid = 0.0_f32;
        let mut high = 0.0_f32;
        let mut frame_chroma = [0.0_f32; 12];
        for bin in &frame.spectrum.bins {
            if (30.0..=220.0).contains(&bin.frequency_hz) {
                low += bin.power;
            } else if (220.0..=2_500.0).contains(&bin.frequency_hz) {
                mid += bin.power;
            } else if (2_500.0..=8_000.0).contains(&bin.frequency_hz) {
                high += bin.power;
            }
            if (55.0..=3_500.0).contains(&bin.frequency_hz) && bin.magnitude > 0.0 {
                let midi = 69.0 + 12.0 * (bin.frequency_hz / 440.0).log2();
                let class = (midi.round() as i32).rem_euclid(12) as usize;
                frame_chroma[class] += bin.magnitude.sqrt();
            }
        }
        normalize_chroma(&mut frame_chroma);
        low_energy.push(low.sqrt());
        mid_energy.push(mid.sqrt());
        high_energy.push(high.sqrt());
        chroma.push(frame_chroma);
        timestamps.push(frame.start_seconds + config.fft_size as f64 / (2.0 * sample_rate as f64));
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
    normalize_nonnegative(&mut mid_energy);
    normalize_nonnegative(&mut high_energy);

    let structural_descriptors = timestamps
        .iter()
        .enumerate()
        .map(|(index, timestamp)| StructuralDescriptor {
            timestamp_seconds: *timestamp,
            onset_novelty: novelty[index],
            low_energy: low_energy[index],
            mid_energy: mid_energy[index],
            high_energy: high_energy[index],
            chroma: chroma[index],
        })
        .collect();

    Ok(OnsetFeatures {
        novelty,
        low_energy,
        timestamps,
        structural_descriptors,
    })
}

fn normalize_chroma(chroma: &mut [f32; 12]) {
    let sum = chroma.iter().sum::<f32>();
    if sum > f32::EPSILON {
        for value in chroma {
            *value = (*value / sum).max(0.0);
        }
    }
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
    let max_lag =
        ((60.0 * frame_rate / min_bpm).ceil() as usize).min(novelty.len().saturating_sub(1));
    if min_lag > max_lag {
        return Vec::new();
    }
    let total_energy = novelty.iter().map(|value| value * value).sum::<f32>();
    if total_energy <= f32::EPSILON {
        return Vec::new();
    }

    let scored = (min_lag..=max_lag)
        .map(|lag| {
            let support = novelty
                .iter()
                .skip(lag)
                .zip(novelty.iter())
                .map(|(right, left)| right * left)
                .sum::<f32>();
            (lag, (support / total_energy).clamp(0.0, 1.0))
        })
        .collect::<Vec<_>>();

    let mut peaks = Vec::new();
    for index in 0..scored.len() {
        let (lag, score) = scored[index];
        if score <= 0.0 {
            continue;
        }
        let left = index.checked_sub(1).map_or(0.0, |item| scored[item].1);
        let right = scored.get(index + 1).map_or(0.0, |item| item.1);
        if score >= left && score >= right {
            let bpm = 60.0 * frame_rate / lag as f32;
            peaks.push(TempoCandidate {
                bpm,
                score,
                autocorrelation_score: score,
                beat_support: 0.0,
            });
        }
    }
    peaks.sort_by(|left, right| right.score.total_cmp(&left.score));

    let mut selected = Vec::new();
    for candidate in peaks {
        if selected.iter().any(|existing: &TempoCandidate| {
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

fn rescore_tempo_candidates(
    mut candidates: Vec<TempoCandidate>,
    novelty: &[f32],
    frame_rate: f32,
    tightness: f32,
    candidate_count: usize,
) -> Vec<TempoCandidate> {
    for candidate in &mut candidates {
        let path = track_beat_frames(novelty, frame_rate, candidate.bpm, tightness);
        candidate.beat_support = beat_path_support(&path, novelty, frame_rate, candidate.bpm);
        candidate.score =
            (0.45 * candidate.autocorrelation_score + 0.55 * candidate.beat_support).clamp(0.0, 1.0);
    }
    candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
    candidates.truncate(candidate_count);
    candidates
}

fn beat_path_support(path: &[usize], novelty: &[f32], frame_rate: f32, bpm: f32) -> f32 {
    if path.len() < 2 || novelty.len() < 2 || frame_rate <= 0.0 || bpm <= 0.0 {
        return 0.0;
    }
    let mean_novelty = path
        .iter()
        .filter_map(|frame| novelty.get(*frame))
        .sum::<f32>()
        / path.len() as f32;
    let coverage = (path[path.len() - 1] - path[0]) as f32 / (novelty.len() - 1) as f32;
    let gaps = path
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as f32)
        .collect::<Vec<_>>();
    let period = frame_rate * 60.0 / bpm;
    let expected = ((path[path.len() - 1] - path[0]) as f32 / period).max(1.0);
    let count_support = ((path.len() - 1) as f32 / expected).min(expected / (path.len() - 1) as f32);
    let smoothness = if gaps.len() < 2 {
        1.0
    } else {
        let change = gaps
            .windows(2)
            .map(|pair| (pair[1] / pair[0]).ln().abs())
            .sum::<f32>()
            / (gaps.len() - 1) as f32;
        (1.0 - change / 0.35).clamp(0.0, 1.0)
    };
    (0.40 * mean_novelty
        + 0.25 * coverage.clamp(0.0, 1.0)
        + 0.20 * smoothness
        + 0.15 * count_support.clamp(0.0, 1.0))
        .clamp(0.0, 1.0)
}

fn track_beat_frames(novelty: &[f32], frame_rate: f32, bpm: f32, tightness: f32) -> Vec<usize> {
    if novelty.is_empty() || frame_rate <= 0.0 || bpm <= 0.0 {
        return Vec::new();
    }
    let period = frame_rate * 60.0 / bpm;
    if !period.is_finite() || period < 1.0 {
        return Vec::new();
    }

    // The wider transition band is deliberate: the selected BPM is a global
    // hypothesis, while individual beat timestamps may accelerate or decelerate.
    let min_gap = (period * 0.58).floor().max(1.0) as usize;
    let max_gap = (period * 1.72).ceil().max(min_gap as f32) as usize;
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
    if path.len() < 2 {
        Vec::new()
    } else {
        path
    }
}

fn build_tempo_map(
    beat_frames: &[usize],
    timestamps: &[f64],
    novelty: &[f32],
) -> Vec<TempoPoint> {
    if beat_frames.len() < 2 {
        return Vec::new();
    }
    let intervals = beat_frames
        .windows(2)
        .filter_map(|pair| {
            let left = *timestamps.get(pair[0])?;
            let right = *timestamps.get(pair[1])?;
            let interval = right - left;
            (interval.is_finite() && interval > 0.0).then_some(interval)
        })
        .collect::<Vec<_>>();
    if intervals.is_empty() {
        return Vec::new();
    }

    beat_frames
        .iter()
        .enumerate()
        .filter_map(|(index, frame)| {
            let start = index.saturating_sub(2).min(intervals.len() - 1);
            let end = (index + 2).min(intervals.len() - 1);
            let mut local = intervals[start..=end].to_vec();
            local.sort_by(f64::total_cmp);
            let interval = local[local.len() / 2];
            let bpm = (60.0 / interval) as f32;
            if !bpm.is_finite() || bpm <= 0.0 {
                return None;
            }
            let dispersion = local
                .iter()
                .map(|value| ((value - interval) / interval).abs() as f32)
                .sum::<f32>()
                / local.len() as f32;
            let onset = novelty.get(*frame).copied().unwrap_or(0.0);
            Some(TempoPoint {
                timestamp_seconds: *timestamps.get(*frame)?,
                bpm,
                confidence: (0.65 * onset + 0.35 * (1.0 - dispersion * 4.0).clamp(0.0, 1.0))
                    .clamp(0.0, 1.0),
            })
        })
        .collect()
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
            let score = if *count == 0 { 0.0 } else { *sum / *count as f32 };
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
        let candidates = estimate_tempo_candidates(&novelty, 100.0, 55.0, 220.0, 8);
        assert!(candidates
            .iter()
            .any(|candidate| (candidate.bpm - 120.0).abs() < 1.0));
    }

    #[test]
    fn beat_path_rescoring_keeps_real_path_support_visible() {
        let novelty = pulse_envelope(50, 24);
        let raw = estimate_tempo_candidates(&novelty, 100.0, 55.0, 220.0, 12);
        let candidates = rescore_tempo_candidates(raw, &novelty, 100.0, 1.25, 5);
        assert!(!candidates.is_empty());
        assert!(candidates.iter().all(|candidate| candidate.beat_support > 0.0));
        assert!(candidates
            .iter()
            .any(|candidate| (candidate.bpm - 120.0).abs() < 1.0));
    }

    #[test]
    fn dynamic_programming_tracks_regular_beats() {
        let novelty = pulse_envelope(40, 16);
        let beats = track_beat_frames(&novelty, 80.0, 120.0, 1.25);
        assert!(beats.len() >= 14);
        assert!(beats.windows(2).all(|pair| {
            let gap = pair[1] - pair[0];
            (23..=69).contains(&gap)
        }));
    }

    #[test]
    fn tempo_map_preserves_acceleration() {
        let mut timestamps = vec![0.0];
        let mut current = 0.0;
        for index in 0..20 {
            let bpm = 100.0 + index as f64 * 2.0;
            current += 60.0 / bpm;
            timestamps.push(current);
        }
        let frames = (0..timestamps.len()).collect::<Vec<_>>();
        let novelty = vec![1.0; timestamps.len()];
        let map = build_tempo_map(&frames, &timestamps, &novelty);
        assert_eq!(map.len(), frames.len());
        assert!(map.last().unwrap().bpm > map.first().unwrap().bpm + 20.0);
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
        assert_eq!(analysis.tempo_map.len(), analysis.beats.len());
        assert!(!analysis.structural_descriptors.is_empty());
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
        assert!(analysis.tempo_map.is_empty());
    }
}
