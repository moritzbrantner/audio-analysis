//! Polyphonic chroma and musical-key analysis for full music tracks.
//!
//! The crate's existing autocorrelation APIs remain the monophonic pitch path.
//! This module instead builds an HPCP-like pitch-class representation from
//! spectral peaks, estimates global tuning drift, and correlates the aggregate
//! chroma against established major/minor key profiles.

use audio_analysis_fourier::{spectrogram, StftConfig};
use audio_contracts::{DetectError, Result};

use crate::{ChromaVector, NoteName};

const KRUMHANSL_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const KRUMHANSL_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];
const TEMPERLEY_MAJOR: [f32; 12] = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0];
const TEMPERLEY_MINOR: [f32; 12] = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0];
const MIN_TONAL_KEY_STRENGTH: f32 = 0.70;

/// Key profile family used to score the chroma vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum KeyProfile {
    /// Krumhansl-Kessler probe-tone profiles.
    Krumhansl,
    /// Temperley's computational key profiles.
    Temperley,
    /// Mean of Krumhansl-Kessler and Temperley correlation scores.
    #[default]
    Ensemble,
}

/// Major/minor musical mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MusicalScale {
    /// Major mode.
    Major,
    /// Minor mode.
    Minor,
}

/// Configuration for polyphonic harmonic analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicKeyConfig {
    /// FFT size used for spectral peak extraction.
    pub fft_size: usize,
    /// Hop size used for spectral peak extraction.
    pub hop_size: usize,
    /// Lowest spectral peak included in chroma accumulation.
    pub min_frequency_hz: f32,
    /// Highest spectral peak included in chroma accumulation.
    pub max_frequency_hz: f32,
    /// Relative per-frame peak floor against the strongest spectral bin.
    pub peak_threshold: f32,
    /// Key-profile family used for major/minor scoring.
    pub profile: KeyProfile,
}

impl Default for HarmonicKeyConfig {
    fn default() -> Self {
        Self {
            fft_size: 4096,
            hop_size: 2048,
            min_frequency_hz: 55.0,
            max_frequency_hz: 3_500.0,
            peak_threshold: 0.08,
            profile: KeyProfile::Ensemble,
        }
    }
}

impl HarmonicKeyConfig {
    /// Validates this configuration.
    pub fn validate(&self) -> Result<()> {
        StftConfig::new(self.fft_size, self.hop_size)?;
        if !self.min_frequency_hz.is_finite()
            || !self.max_frequency_hz.is_finite()
            || self.min_frequency_hz <= 0.0
            || self.max_frequency_hz <= self.min_frequency_hz
        {
            return Err(DetectError::InvalidArgument(
                "harmonic key frequency range must be finite, positive, and increasing".to_string(),
            ));
        }
        if !self.peak_threshold.is_finite() || !(0.0..=1.0).contains(&self.peak_threshold) {
            return Err(DetectError::InvalidArgument(
                "harmonic key peak_threshold must be between zero and one".to_string(),
            ));
        }
        Ok(())
    }
}

/// Polyphonic pitch-class analysis before key-profile scoring.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HarmonicChromaAnalysis {
    /// Normalized pitch-class energy, C through B.
    pub chroma: ChromaVector,
    /// Estimated global tuning offset from A440 equal temperament, in cents.
    pub tuning_cents: f32,
    /// Number of STFT frames contributing to the estimate.
    pub frame_count: usize,
    /// Number of accepted spectral peaks contributing to the estimate.
    pub peak_count: usize,
}

/// One key candidate.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KeyCandidate {
    /// Tonic pitch class.
    pub tonic: NoteName,
    /// Major/minor mode.
    pub scale: MusicalScale,
    /// Profile correlation in the range -1..=1.
    pub correlation: f32,
}

/// Musical-key estimate for a polyphonic track.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MusicalKeyEstimate {
    /// Selected tonic pitch class.
    pub tonic: NoteName,
    /// Selected major/minor mode.
    pub scale: MusicalScale,
    /// Best profile correlation mapped to 0..=1.
    pub strength: f32,
    /// Separation of the best candidate from the runner-up, normalized to 0..=1.
    pub confidence: f32,
    /// Second-best key candidate for ambiguity inspection.
    pub runner_up: KeyCandidate,
    /// Global tuning offset used before pitch-class folding.
    pub tuning_cents: f32,
    /// Aggregate chroma used for key scoring.
    pub chroma: ChromaVector,
    /// Number of contributing spectral peaks.
    pub peak_count: usize,
}

impl MusicalKeyEstimate {
    /// Stable human-readable key label such as `F# minor`.
    pub fn label(&self) -> String {
        format!(
            "{} {}",
            self.tonic.as_str(),
            match self.scale {
                MusicalScale::Major => "major",
                MusicalScale::Minor => "minor",
            }
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct SpectralPeak {
    frequency_hz: f32,
    magnitude: f32,
}

/// Extracts a tuning-corrected polyphonic chroma vector from normalized samples.
pub fn harmonic_chroma(
    samples: &[f32],
    sample_rate: u32,
    config: HarmonicKeyConfig,
) -> Result<HarmonicChromaAnalysis> {
    config.validate()?;
    if sample_rate == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate,
            channels: 1,
        });
    }
    if samples.is_empty() {
        return Err(DetectError::InvalidArgument(
            "harmonic key samples must not be empty".to_string(),
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "harmonic key samples must contain only finite values".to_string(),
        ));
    }

    let stft = StftConfig::new(config.fft_size, config.hop_size)?.pad_final_frame(true);
    let frames = spectrogram(samples, sample_rate, &stft)?;
    let peaks = frames
        .iter()
        .map(|frame| frame_peaks(&frame.spectrum.bins, config))
        .collect::<Vec<_>>();
    let tuning_cents = estimate_tuning_cents(&peaks);

    let mut aggregate = [0.0_f32; 12];
    let mut contributing_frames = 0_usize;
    let mut peak_count = 0_usize;
    for frame_peaks in &peaks {
        let mut frame_chroma = [0.0_f32; 12];
        for peak in frame_peaks {
            add_peak_to_chroma(&mut frame_chroma, *peak, tuning_cents);
            peak_count += 1;
        }
        let frame_sum = frame_chroma.iter().sum::<f32>();
        if frame_sum > f32::EPSILON {
            for value in &mut frame_chroma {
                *value /= frame_sum;
            }
            for (target, value) in aggregate.iter_mut().zip(frame_chroma) {
                *target += value;
            }
            contributing_frames += 1;
        }
    }

    if contributing_frames > 0 {
        for value in &mut aggregate {
            *value /= contributing_frames as f32;
        }
        normalize_chroma(&mut aggregate);
    }

    Ok(HarmonicChromaAnalysis {
        chroma: ChromaVector { bins: aggregate },
        tuning_cents,
        frame_count: contributing_frames,
        peak_count,
    })
}

/// Estimates major/minor musical key from polyphonic normalized samples.
///
/// `Ok(None)` means the clip did not contain enough tonal spectral evidence for
/// a meaningful key estimate.
pub fn estimate_musical_key(
    samples: &[f32],
    sample_rate: u32,
    config: HarmonicKeyConfig,
) -> Result<Option<MusicalKeyEstimate>> {
    let analysis = harmonic_chroma(samples, sample_rate, config)?;
    if analysis.peak_count == 0 || analysis.chroma.bins.iter().sum::<f32>() <= f32::EPSILON {
        return Ok(None);
    }

    let mut candidates = Vec::with_capacity(24);
    for tonic in 0..12 {
        let tonic_name = NoteName::from_index(tonic).expect("pitch-class index is bounded");
        for scale in [MusicalScale::Major, MusicalScale::Minor] {
            candidates.push(KeyCandidate {
                tonic: tonic_name,
                scale,
                correlation: profile_correlation(
                    &analysis.chroma.bins,
                    tonic,
                    scale,
                    config.profile,
                ),
            });
        }
    }
    candidates.sort_by(|left, right| right.correlation.total_cmp(&left.correlation));
    let best = candidates[0];
    let runner_up = candidates[1];
    let strength = ((best.correlation + 1.0) * 0.5).clamp(0.0, 1.0);
    if strength < MIN_TONAL_KEY_STRENGTH {
        return Ok(None);
    }
    let confidence = ((best.correlation - runner_up.correlation).max(0.0) / 0.35).clamp(0.0, 1.0);

    Ok(Some(MusicalKeyEstimate {
        tonic: best.tonic,
        scale: best.scale,
        strength,
        confidence,
        runner_up,
        tuning_cents: analysis.tuning_cents,
        chroma: analysis.chroma,
        peak_count: analysis.peak_count,
    }))
}

fn frame_peaks(
    bins: &[audio_analysis_fourier::SpectrumBin],
    config: HarmonicKeyConfig,
) -> Vec<SpectralPeak> {
    if bins.len() < 3 {
        return Vec::new();
    }
    let strongest = bins
        .iter()
        .filter(|bin| {
            (config.min_frequency_hz..=config.max_frequency_hz).contains(&bin.frequency_hz)
        })
        .map(|bin| bin.magnitude)
        .fold(0.0_f32, f32::max);
    if strongest <= f32::EPSILON {
        return Vec::new();
    }
    let floor = strongest * config.peak_threshold;
    bins.windows(3)
        .filter_map(|window| {
            let left = &window[0];
            let center = &window[1];
            let right = &window[2];
            let in_range =
                (config.min_frequency_hz..=config.max_frequency_hz).contains(&center.frequency_hz);
            (in_range
                && center.magnitude >= floor
                && center.magnitude >= left.magnitude
                && center.magnitude > right.magnitude)
                .then_some(SpectralPeak {
                    frequency_hz: center.frequency_hz,
                    magnitude: center.magnitude,
                })
        })
        .collect()
}

fn estimate_tuning_cents(frames: &[Vec<SpectralPeak>]) -> f32 {
    let mut sin_sum = 0.0_f32;
    let mut cos_sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for peak in frames.iter().flatten() {
        if peak.frequency_hz <= 0.0 || peak.magnitude <= 0.0 {
            continue;
        }
        let midi = 69.0 + 12.0 * (peak.frequency_hz / 440.0).log2();
        let fractional = midi - midi.round();
        let angle = std::f32::consts::TAU * fractional;
        let weight = peak.magnitude.sqrt();
        sin_sum += weight * angle.sin();
        cos_sum += weight * angle.cos();
        weight_sum += weight;
    }
    if weight_sum <= f32::EPSILON || (sin_sum.abs() + cos_sum.abs()) <= f32::EPSILON {
        return 0.0;
    }
    let fractional = sin_sum.atan2(cos_sum) / std::f32::consts::TAU;
    (fractional * 100.0).clamp(-50.0, 50.0)
}

fn add_peak_to_chroma(chroma: &mut [f32; 12], peak: SpectralPeak, tuning_cents: f32) {
    if peak.frequency_hz <= 0.0 || peak.magnitude <= 0.0 {
        return;
    }
    let midi = 69.0 + 12.0 * (peak.frequency_hz / 440.0).log2() - tuning_cents / 100.0;
    let pitch = midi.rem_euclid(12.0);
    let lower = pitch.floor() as usize % 12;
    let fraction = pitch - pitch.floor();
    let upper = (lower + 1) % 12;
    let weight = peak.magnitude.sqrt();
    chroma[lower] += weight * (1.0 - fraction);
    chroma[upper] += weight * fraction;
}

fn normalize_chroma(chroma: &mut [f32; 12]) {
    let sum = chroma.iter().sum::<f32>();
    if sum > f32::EPSILON {
        for value in chroma {
            *value = (*value / sum).max(0.0);
        }
    }
}

fn profile_correlation(
    chroma: &[f32; 12],
    tonic: usize,
    scale: MusicalScale,
    profile: KeyProfile,
) -> f32 {
    match profile {
        KeyProfile::Krumhansl => pearson_for_profile(
            chroma,
            tonic,
            match scale {
                MusicalScale::Major => &KRUMHANSL_MAJOR,
                MusicalScale::Minor => &KRUMHANSL_MINOR,
            },
        ),
        KeyProfile::Temperley => pearson_for_profile(
            chroma,
            tonic,
            match scale {
                MusicalScale::Major => &TEMPERLEY_MAJOR,
                MusicalScale::Minor => &TEMPERLEY_MINOR,
            },
        ),
        KeyProfile::Ensemble => {
            let krumhansl = profile_correlation(chroma, tonic, scale, KeyProfile::Krumhansl);
            let temperley = profile_correlation(chroma, tonic, scale, KeyProfile::Temperley);
            0.5 * (krumhansl + temperley)
        }
    }
}

fn pearson_for_profile(chroma: &[f32; 12], tonic: usize, profile: &[f32; 12]) -> f32 {
    let chroma_mean = chroma.iter().sum::<f32>() / 12.0;
    let profile_mean = profile.iter().sum::<f32>() / 12.0;
    let mut numerator = 0.0_f32;
    let mut chroma_energy = 0.0_f32;
    let mut profile_energy = 0.0_f32;
    for (pitch_class, chroma_value) in chroma.iter().enumerate() {
        let x = *chroma_value - chroma_mean;
        let relative = (pitch_class + 12 - tonic) % 12;
        let y = profile[relative] - profile_mean;
        numerator += x * y;
        chroma_energy += x * x;
        profile_energy += y * y;
    }
    let denominator = (chroma_energy * profile_energy).sqrt();
    if denominator <= f32::EPSILON {
        0.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_progression(sample_rate: u32, chords: &[&[f32]], seconds_per_chord: f32) -> Vec<f32> {
        let chord_samples = (sample_rate as f32 * seconds_per_chord) as usize;
        let mut output = Vec::with_capacity(chord_samples * chords.len());
        for chord in chords {
            for index in 0..chord_samples {
                let time = index as f32 / sample_rate as f32;
                let envelope = if index < 256 {
                    index as f32 / 256.0
                } else if chord_samples.saturating_sub(index) < 256 {
                    chord_samples.saturating_sub(index) as f32 / 256.0
                } else {
                    1.0
                };
                let sample = chord
                    .iter()
                    .map(|frequency| (std::f32::consts::TAU * frequency * time).sin())
                    .sum::<f32>()
                    / chord.len() as f32;
                output.push(sample * envelope);
            }
        }
        output
    }

    #[test]
    fn profile_scoring_recovers_rotated_f_sharp_minor_profile() {
        let mut chroma = [0.0_f32; 12];
        for pitch_class in 0..12 {
            chroma[pitch_class] = KRUMHANSL_MINOR[(pitch_class + 12 - 6) % 12];
        }
        normalize_chroma(&mut chroma);
        let mut candidates = Vec::new();
        for tonic in 0..12 {
            for scale in [MusicalScale::Major, MusicalScale::Minor] {
                candidates.push((
                    tonic,
                    scale,
                    profile_correlation(&chroma, tonic, scale, KeyProfile::Krumhansl),
                ));
            }
        }
        candidates.sort_by(|left, right| right.2.total_cmp(&left.2));
        assert_eq!(candidates[0].0, 6);
        assert_eq!(candidates[0].1, MusicalScale::Minor);
    }

    #[test]
    fn tuning_estimator_detects_small_positive_offset() {
        let cents = 25.0_f32;
        let frequency = 440.0 * 2.0_f32.powf(cents / 1200.0);
        let frames = vec![vec![SpectralPeak {
            frequency_hz: frequency,
            magnitude: 1.0,
        }]];
        let estimated = estimate_tuning_cents(&frames);
        assert!((estimated - cents).abs() < 1.0);
    }

    #[test]
    fn polyphonic_chroma_keeps_multiple_pitch_classes() {
        let sample_rate = 8_000;
        let c_major = [261.6256, 329.6276, 391.9954];
        let samples = synth_progression(sample_rate, &[&c_major], 2.0);
        let analysis = harmonic_chroma(
            &samples,
            sample_rate,
            HarmonicKeyConfig {
                fft_size: 2048,
                hop_size: 512,
                max_frequency_hz: 3_000.0,
                ..HarmonicKeyConfig::default()
            },
        )
        .expect("chroma");
        assert!(analysis.chroma.bins[0] > 0.0);
        assert!(analysis.chroma.bins[4] > 0.0);
        assert!(analysis.chroma.bins[7] > 0.0);
    }

    #[test]
    fn progression_produces_a_major_minor_key_estimate() {
        let sample_rate = 8_000;
        let c = [261.6256, 329.6276, 391.9954];
        let f = [174.6141, 220.0, 261.6256];
        let g = [195.9977, 246.9417, 293.6648];
        let samples = synth_progression(sample_rate, &[&c, &f, &g, &c], 1.0);
        let estimate = estimate_musical_key(
            &samples,
            sample_rate,
            HarmonicKeyConfig {
                fft_size: 2048,
                hop_size: 512,
                max_frequency_hz: 3_000.0,
                ..HarmonicKeyConfig::default()
            },
        )
        .expect("key")
        .expect("tonal estimate");
        assert_eq!(estimate.tonic, NoteName::C);
        assert_eq!(estimate.scale, MusicalScale::Major);
    }

    #[test]
    fn silence_has_no_key() {
        let estimate = estimate_musical_key(
            &vec![0.0; 16_000],
            8_000,
            HarmonicKeyConfig {
                fft_size: 2048,
                hop_size: 512,
                max_frequency_hz: 3_000.0,
                ..HarmonicKeyConfig::default()
            },
        )
        .expect("silence");
        assert!(estimate.is_none());
    }

    #[test]
    fn broadband_noise_has_no_key() {
        let mut keyed_seeds = Vec::new();
        for seed in 1_u32..=8 {
            let mut state = seed;
            let samples = (0..32_000)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect::<Vec<_>>();
            let estimate = estimate_musical_key(
                &samples,
                8_000,
                HarmonicKeyConfig {
                    fft_size: 1024,
                    hop_size: 256,
                    max_frequency_hz: 3_000.0,
                    ..HarmonicKeyConfig::default()
                },
            )
            .expect("broadband noise");
            if estimate.is_some() {
                keyed_seeds.push(seed);
            }
        }
        assert!(
            keyed_seeds.is_empty(),
            "noise received keys for seeds {keyed_seeds:?}"
        );
    }
}
