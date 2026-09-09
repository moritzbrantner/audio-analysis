//! Polyphonic chroma and musical-key analysis for full music tracks.
//!
//! The crate's existing autocorrelation APIs remain the monophonic pitch path.
//! This module builds a multi-resolution, 36-bin-per-octave HPCP representation
//! from tonal spectral peaks, estimates global tuning drift, collapses that high-
//! resolution representation to stable pitch classes, and correlates the result
//! against established major/minor key profiles.

use audio_analysis_fourier::{spectrogram, SpectrumBin, StftConfig};
use audio_contracts::{DetectError, Result};

use crate::{ChromaVector, NoteName};

const HPCP_BINS_PER_OCTAVE: usize = 36;
const HPCP_BINS_PER_SEMITONE: usize = HPCP_BINS_PER_OCTAVE / 12;
const LOW_BAND_MAX_HZ: f32 = 220.0;
const MID_BAND_MAX_HZ: f32 = 880.0;
const KRUMHANSL_MAJOR: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const KRUMHANSL_MINOR: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];
const TEMPERLEY_MAJOR: [f32; 12] = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0];
const TEMPERLEY_MINOR: [f32; 12] = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0];
const MIN_TONAL_KEY_STRENGTH: f32 = 0.70;
// Pearson correlation is scale-invariant, so nearly uniform broadband chroma can
// correlate strongly by chance. Require a minimum absolute pitch-class contrast
// before a profile match is allowed to become key metadata.
const MIN_CHROMA_CONTRAST: f32 = 0.05;

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
    /// Base FFT size. Lower octaves automatically use larger FFTs.
    pub fft_size: usize,
    /// Base hop size. Lower-octave analysis scales the hop with the FFT size.
    pub hop_size: usize,
    /// Lowest spectral peak included in HPCP accumulation.
    pub min_frequency_hz: f32,
    /// Highest spectral peak included in HPCP accumulation.
    pub max_frequency_hz: f32,
    /// Relative per-frame peak floor against the strongest bin in each resolution band.
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
        for factor in [1_usize, 2, 4] {
            let fft_size = self.fft_size.checked_mul(factor).ok_or_else(|| {
                DetectError::InvalidArgument("harmonic key FFT pyramid is too large".to_string())
            })?;
            let hop_size = self.hop_size.checked_mul(factor).ok_or_else(|| {
                DetectError::InvalidArgument("harmonic key hop pyramid is too large".to_string())
            })?;
            StftConfig::new(fft_size, hop_size)?;
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
    /// Number of multi-resolution STFT frames contributing to the estimate.
    pub frame_count: usize,
    /// Number of accepted tonal spectral peaks contributing to the estimate.
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
    /// Number of contributing tonal spectral peaks.
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
struct ResolutionBand {
    min_frequency_hz: f32,
    max_frequency_hz: f32,
    fft_factor: usize,
}

#[derive(Debug, Clone, Copy)]
struct SpectralPeak {
    frequency_hz: f32,
    magnitude: f32,
}

/// Extracts a tuning-corrected polyphonic chroma vector from normalized samples.
///
/// The frontend uses a three-resolution FFT pyramid: four-times resolution for
/// bass, two-times resolution for low/mid harmonics, and the configured base
/// resolution for upper harmonics. Peaks are projected into a 36-bin-per-octave
/// HPCP before being collapsed to 12 pitch classes. A local peak-prominence mask
/// attenuates broadband/percussive energy without requiring source separation.
pub fn harmonic_chroma(
    samples: &[f32],
    sample_rate: u32,
    config: HarmonicKeyConfig,
) -> Result<HarmonicChromaAnalysis> {
    config.validate()?;
    validate_samples(samples, sample_rate)?;

    let bands = resolution_bands(config);
    let mut peak_bands = Vec::with_capacity(bands.len());
    let mut all_frames = Vec::new();
    for band in bands {
        let frames = collect_peak_frames(samples, sample_rate, config, band)?;
        all_frames.extend(frames.iter().cloned());
        peak_bands.push(frames);
    }
    let tuning_cents = estimate_tuning_cents(&all_frames);

    let mut aggregate_hpcp = [0.0_f32; HPCP_BINS_PER_OCTAVE];
    let mut contributing_bands = 0_usize;
    let mut contributing_frames = 0_usize;
    let mut peak_count = 0_usize;

    for band_frames in &peak_bands {
        let mut band_hpcp = [0.0_f32; HPCP_BINS_PER_OCTAVE];
        let mut band_frame_count = 0_usize;
        for frame_peaks in band_frames {
            let mut frame_hpcp = [0.0_f32; HPCP_BINS_PER_OCTAVE];
            for peak in frame_peaks {
                add_peak_to_hpcp(&mut frame_hpcp, *peak, tuning_cents);
                peak_count += 1;
            }
            if normalize_nonnegative(&mut frame_hpcp) {
                for (target, value) in band_hpcp.iter_mut().zip(frame_hpcp) {
                    *target += value;
                }
                band_frame_count += 1;
                contributing_frames += 1;
            }
        }
        if band_frame_count > 0 {
            for value in &mut band_hpcp {
                *value /= band_frame_count as f32;
            }
            normalize_nonnegative(&mut band_hpcp);
            for (target, value) in aggregate_hpcp.iter_mut().zip(band_hpcp) {
                *target += value;
            }
            contributing_bands += 1;
        }
    }

    if contributing_bands > 0 {
        for value in &mut aggregate_hpcp {
            *value /= contributing_bands as f32;
        }
        normalize_nonnegative(&mut aggregate_hpcp);
    }

    let mut chroma = collapse_hpcp(&aggregate_hpcp);
    normalize_chroma(&mut chroma);
    Ok(HarmonicChromaAnalysis {
        chroma: ChromaVector { bins: chroma },
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
    Ok(estimate_from_chroma(&analysis, config.profile, None))
}

pub(crate) fn score_key_candidates(chroma: &[f32; 12], profile: KeyProfile) -> Vec<KeyCandidate> {
    let mut candidates = Vec::with_capacity(24);
    for tonic in 0..12 {
        let tonic_name = NoteName::from_index(tonic).expect("pitch-class index is bounded");
        for scale in [MusicalScale::Major, MusicalScale::Minor] {
            candidates.push(KeyCandidate {
                tonic: tonic_name,
                scale,
                correlation: profile_correlation(chroma, tonic, scale, profile),
            });
        }
    }
    candidates.sort_by(|left, right| right.correlation.total_cmp(&left.correlation));
    candidates
}

pub(crate) fn estimate_from_chroma(
    analysis: &HarmonicChromaAnalysis,
    profile: KeyProfile,
    selected: Option<(NoteName, MusicalScale)>,
) -> Option<MusicalKeyEstimate> {
    if analysis.peak_count == 0 || analysis.chroma.bins.iter().sum::<f32>() <= f32::EPSILON {
        return None;
    }
    if chroma_contrast(&analysis.chroma.bins) < MIN_CHROMA_CONTRAST {
        return None;
    }
    let candidates = score_key_candidates(&analysis.chroma.bins, profile);
    let selected_index = selected
        .and_then(|(tonic, scale)| {
            candidates
                .iter()
                .position(|candidate| candidate.tonic == tonic && candidate.scale == scale)
        })
        .unwrap_or(0);
    let best = candidates[selected_index];
    let runner_up = candidates
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| *index != selected_index)
        .map(|(_, candidate)| candidate)
        .max_by(|left, right| left.correlation.total_cmp(&right.correlation))?;
    let strength = ((best.correlation + 1.0) * 0.5).clamp(0.0, 1.0);
    if strength < MIN_TONAL_KEY_STRENGTH {
        return None;
    }
    let confidence = ((best.correlation - runner_up.correlation).max(0.0) / 0.35).clamp(0.0, 1.0);
    Some(MusicalKeyEstimate {
        tonic: best.tonic,
        scale: best.scale,
        strength,
        confidence,
        runner_up,
        tuning_cents: analysis.tuning_cents,
        chroma: analysis.chroma.clone(),
        peak_count: analysis.peak_count,
    })
}

fn validate_samples(samples: &[f32], sample_rate: u32) -> Result<()> {
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
    Ok(())
}

fn resolution_bands(config: HarmonicKeyConfig) -> Vec<ResolutionBand> {
    let edges = [
        (config.min_frequency_hz, config.max_frequency_hz.min(LOW_BAND_MAX_HZ), 4),
        (
            config.min_frequency_hz.max(LOW_BAND_MAX_HZ),
            config.max_frequency_hz.min(MID_BAND_MAX_HZ),
            2,
        ),
        (
            config.min_frequency_hz.max(MID_BAND_MAX_HZ),
            config.max_frequency_hz,
            1,
        ),
    ];
    edges
        .into_iter()
        .filter(|(min_frequency_hz, max_frequency_hz, _)| max_frequency_hz > min_frequency_hz)
        .map(|(min_frequency_hz, max_frequency_hz, fft_factor)| ResolutionBand {
            min_frequency_hz,
            max_frequency_hz,
            fft_factor,
        })
        .collect()
}

fn collect_peak_frames(
    samples: &[f32],
    sample_rate: u32,
    config: HarmonicKeyConfig,
    band: ResolutionBand,
) -> Result<Vec<Vec<SpectralPeak>>> {
    let fft_size = config
        .fft_size
        .checked_mul(band.fft_factor)
        .ok_or_else(|| DetectError::InvalidArgument("harmonic key FFT pyramid is too large".to_string()))?;
    let hop_size = config
        .hop_size
        .checked_mul(band.fft_factor)
        .ok_or_else(|| DetectError::InvalidArgument("harmonic key hop pyramid is too large".to_string()))?;
    let stft = StftConfig::new(fft_size, hop_size)?.pad_final_frame(true);
    let frames = spectrogram(samples, sample_rate, &stft)?;
    Ok(frames
        .iter()
        .map(|frame| frame_peaks(&frame.spectrum.bins, config.peak_threshold, band))
        .collect())
}

fn frame_peaks(
    bins: &[SpectrumBin],
    peak_threshold: f32,
    band: ResolutionBand,
) -> Vec<SpectralPeak> {
    if bins.len() < 5 {
        return Vec::new();
    }
    let strongest = bins
        .iter()
        .filter(|bin| (band.min_frequency_hz..=band.max_frequency_hz).contains(&bin.frequency_hz))
        .map(|bin| bin.magnitude)
        .fold(0.0_f32, f32::max);
    if strongest <= f32::EPSILON {
        return Vec::new();
    }
    let floor = strongest * peak_threshold;
    bins.windows(5)
        .filter_map(|window| {
            let center = &window[2];
            let in_range =
                (band.min_frequency_hz..=band.max_frequency_hz).contains(&center.frequency_hz);
            if !in_range
                || center.magnitude < floor
                || center.magnitude < window[1].magnitude
                || center.magnitude <= window[3].magnitude
            {
                return None;
            }
            let neighborhood =
                (window[0].magnitude + window[1].magnitude + window[3].magnitude + window[4].magnitude)
                    * 0.25;
            let tonal_mask = center.magnitude / (center.magnitude + 2.0 * neighborhood + f32::EPSILON);
            let weighted_magnitude = center.magnitude * tonal_mask * tonal_mask;
            (weighted_magnitude > f32::EPSILON).then_some(SpectralPeak {
                frequency_hz: center.frequency_hz,
                magnitude: weighted_magnitude,
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

fn add_peak_to_hpcp(
    hpcp: &mut [f32; HPCP_BINS_PER_OCTAVE],
    peak: SpectralPeak,
    tuning_cents: f32,
) {
    if peak.frequency_hz <= 0.0 || peak.magnitude <= 0.0 {
        return;
    }
    let midi = 69.0 + 12.0 * (peak.frequency_hz / 440.0).log2() - tuning_cents / 100.0;
    let position = midi.rem_euclid(12.0) * HPCP_BINS_PER_SEMITONE as f32;
    let lower_float = position.floor();
    let lower = lower_float as usize % HPCP_BINS_PER_OCTAVE;
    let upper = (lower + 1) % HPCP_BINS_PER_OCTAVE;
    let fraction = position - lower_float;
    let octave_distance = (midi - 69.0).abs() / 12.0;
    let octave_weight = 1.0 / (1.0 + 0.08 * octave_distance);
    let weight = peak.magnitude.sqrt() * octave_weight;
    hpcp[lower] += weight * (1.0 - fraction);
    hpcp[upper] += weight * fraction;
}

fn collapse_hpcp(hpcp: &[f32; HPCP_BINS_PER_OCTAVE]) -> [f32; 12] {
    let mut chroma = [0.0_f32; 12];
    for (pitch_class, value) in chroma.iter_mut().enumerate() {
        let center = pitch_class * HPCP_BINS_PER_SEMITONE;
        let left = (center + HPCP_BINS_PER_OCTAVE - 1) % HPCP_BINS_PER_OCTAVE;
        let right = (center + 1) % HPCP_BINS_PER_OCTAVE;
        *value = hpcp[center] + 0.5 * (hpcp[left] + hpcp[right]);
    }
    chroma
}

fn normalize_nonnegative<const N: usize>(values: &mut [f32; N]) -> bool {
    let sum = values.iter().sum::<f32>();
    if sum <= f32::EPSILON {
        return false;
    }
    for value in values {
        *value = (*value / sum).max(0.0);
    }
    true
}

fn normalize_chroma(chroma: &mut [f32; 12]) {
    let _ = normalize_nonnegative(chroma);
}

fn chroma_contrast(chroma: &[f32; 12]) -> f32 {
    let mean = chroma.iter().sum::<f32>() / chroma.len() as f32;
    chroma
        .iter()
        .map(|value| (*value - mean).powi(2))
        .sum::<f32>()
        .sqrt()
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
        let candidates = score_key_candidates(&chroma, KeyProfile::Krumhansl);
        assert_eq!(candidates[0].tonic, NoteName::FSharp);
        assert_eq!(candidates[0].scale, MusicalScale::Minor);
    }

    #[test]
    fn near_uniform_chroma_cannot_become_a_key_from_correlation_alone() {
        let profile_mean = KRUMHANSL_MAJOR.iter().sum::<f32>() / 12.0;
        let mut chroma = [1.0 / 12.0; 12];
        for (value, profile_value) in chroma.iter_mut().zip(KRUMHANSL_MAJOR) {
            *value += (profile_value - profile_mean) * 0.0001;
        }
        normalize_chroma(&mut chroma);
        let analysis = HarmonicChromaAnalysis {
            chroma: ChromaVector { bins: chroma },
            tuning_cents: 0.0,
            frame_count: 100,
            peak_count: 100,
        };
        let candidates = score_key_candidates(&analysis.chroma.bins, KeyProfile::Krumhansl);
        assert!(candidates[0].correlation > 0.99);
        assert!(chroma_contrast(&analysis.chroma.bins) < MIN_CHROMA_CONTRAST);
        assert!(estimate_from_chroma(&analysis, KeyProfile::Krumhansl, None).is_none());
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
    fn hpcp_collapse_preserves_semitone_centers() {
        let mut hpcp = [0.0_f32; HPCP_BINS_PER_OCTAVE];
        hpcp[0] = 1.0;
        hpcp[4 * HPCP_BINS_PER_SEMITONE] = 0.8;
        hpcp[7 * HPCP_BINS_PER_SEMITONE] = 0.9;
        let chroma = collapse_hpcp(&hpcp);
        assert!(chroma[0] > chroma[1]);
        assert!(chroma[4] > chroma[3]);
        assert!(chroma[7] > chroma[6]);
    }

    #[test]
    fn resolution_pyramid_gives_bass_more_frequency_resolution() {
        let config = HarmonicKeyConfig {
            fft_size: 1024,
            hop_size: 256,
            ..HarmonicKeyConfig::default()
        };
        let bands = resolution_bands(config);
        assert_eq!(bands[0].fft_factor, 4);
        assert_eq!(bands[1].fft_factor, 2);
        assert_eq!(bands.last().unwrap().fft_factor, 1);
    }

    #[test]
    fn tonal_mask_reduces_broadband_peak_weight() {
        let bins = (0..9)
            .map(|index| SpectrumBin {
                index,
                frequency_hz: 100.0 + index as f32 * 10.0,
                magnitude: if index == 4 { 1.05 } else { 1.0 },
                power: 1.0,
            })
            .collect::<Vec<_>>();
        let peaks = frame_peaks(
            &bins,
            0.0,
            ResolutionBand {
                min_frequency_hz: 100.0,
                max_frequency_hz: 200.0,
                fft_factor: 1,
            },
        );
        assert!(peaks.iter().all(|peak| peak.magnitude < 0.2));
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
                fft_size: 1024,
                hop_size: 256,
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
                fft_size: 1024,
                hop_size: 256,
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
                fft_size: 1024,
                hop_size: 256,
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
                    fft_size: 512,
                    hop_size: 128,
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
