#![doc = include_str!("../README.md")]

pub mod surface;
use audio_analysis_core::{mono_samples, zero_pad_to, FrameSpec, WindowFunction};
use audio_contracts::{AnalysisEvent, AudioAnalyzer, AudioFrame, DetectError, Result};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

#[derive(Debug, Clone, PartialEq)]
/// Data type for spectrum bin.
pub struct SpectrumBin {
    /// The index value.
    pub index: usize,
    /// The frequency hz value.
    pub frequency_hz: f32,
    /// The magnitude value.
    pub magnitude: f32,
    /// The power value.
    pub power: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for spectrum.
pub struct Spectrum {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// The FFT size value.
    pub fft_size: usize,
    /// The bins value.
    pub bins: Vec<SpectrumBin>,
}

impl Spectrum {
    /// Returns dominant frequency hz.
    pub fn dominant_frequency_hz(&self) -> Option<f32> {
        self.bins
            .iter()
            .skip(1)
            .filter(|bin| bin.magnitude > f32::EPSILON)
            .max_by(|a, b| a.magnitude.total_cmp(&b.magnitude))
            .map(|bin| bin.frequency_hz)
    }

    /// Returns features.
    pub fn features(&self) -> SpectralFeatures {
        spectral_features(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for spectral features.
pub struct SpectralFeatures {
    /// The centroid hz value.
    pub centroid_hz: f32,
    /// The bandwidth hz value.
    pub bandwidth_hz: f32,
    /// The rolloff hz value.
    pub rolloff_hz: f32,
    /// The flatness value.
    pub flatness: f32,
    /// The dominant frequency hz value.
    pub dominant_frequency_hz: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// Options for windowed spectral feature extraction.
pub struct SpectralFeatureOptions {
    /// FFT size in samples.
    pub fft_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Optional number of mel-style bands to summarize.
    pub mel_band_count: Option<usize>,
}

impl SpectralFeatureOptions {
    /// Creates a validated spectral feature options value.
    pub fn new(fft_size: usize, hop_size: usize, sample_rate: u32) -> Result<Self> {
        let options = Self {
            fft_size,
            hop_size,
            sample_rate,
            mel_band_count: None,
        };
        options.validate()?;
        Ok(options)
    }

    /// Sets mel-style band count.
    pub fn mel_band_count(mut self, band_count: Option<usize>) -> Result<Self> {
        self.mel_band_count = band_count;
        self.validate()?;
        Ok(self)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        FourierTransform::new(self.fft_size)?;
        FrameSpec::new(self.fft_size, self.hop_size)?;
        if self.sample_rate == 0 {
            return Err(DetectError::InvalidAudioFormat {
                sample_rate: self.sample_rate,
                channels: 1,
            });
        }
        if self.mel_band_count == Some(0) {
            return Err(DetectError::InvalidArgument(
                "mel_band_count must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// One window of spectral features.
pub struct SpectralFeatureFrame {
    /// Frame start time in seconds.
    pub start_seconds: f32,
    /// Spectral centroid in hertz.
    pub centroid_hz: f32,
    /// Spectral bandwidth in hertz.
    pub bandwidth_hz: f32,
    /// Spectral rolloff in hertz.
    pub rolloff_hz: f32,
    /// Spectral flatness ratio.
    pub flatness: f32,
    /// Sum of bin power values.
    pub energy: f32,
    /// Optional mel-style band energies.
    pub mel_bands: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for fourier transform.
pub struct FourierTransform {
    /// The FFT size value.
    pub fft_size: usize,
    /// The window value.
    pub window: WindowFunction,
}

impl FourierTransform {
    /// Creates a new value.
    pub fn new(fft_size: usize) -> Result<Self> {
        Self::with_window(fft_size, WindowFunction::Hann)
    }

    /// Returns this value with window.
    pub fn with_window(fft_size: usize, window: WindowFunction) -> Result<Self> {
        if fft_size == 0 || !fft_size.is_power_of_two() {
            return Err(DetectError::InvalidArgument(
                "fft_size must be a non-zero power of two".to_string(),
            ));
        }
        Ok(Self { fft_size, window })
    }

    /// Returns analyze samples.
    pub fn analyze_samples(&self, samples: &[f32], sample_rate: u32) -> Result<Spectrum> {
        if sample_rate == 0 {
            return Err(DetectError::InvalidAudioFormat {
                sample_rate,
                channels: 1,
            });
        }
        let mut windowed = zero_pad_to(
            samples.iter().copied().take(self.fft_size).collect(),
            self.fft_size,
        );
        windowed = self.window.apply(&windowed);

        let mut fft_buffer = windowed
            .into_iter()
            .map(|sample| Complex::new(sample, 0.0))
            .collect::<Vec<_>>();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.fft_size);
        fft.process(&mut fft_buffer);

        let scale = 1.0 / self.fft_size as f32;
        let bin_hz = sample_rate as f32 / self.fft_size as f32;
        let bins = fft_buffer
            .iter()
            .take(self.fft_size / 2 + 1)
            .enumerate()
            .map(|(index, value)| {
                let magnitude = value.norm() * scale;
                SpectrumBin {
                    index,
                    frequency_hz: index as f32 * bin_hz,
                    magnitude,
                    power: magnitude * magnitude,
                }
            })
            .collect();

        Ok(Spectrum {
            sample_rate,
            fft_size: self.fft_size,
            bins,
        })
    }

    /// Returns analyze frame.
    pub fn analyze_frame(&self, frame: &AudioFrame<'_>) -> Result<Spectrum> {
        let mono = mono_samples(frame)?;
        self.analyze_samples(&mono.samples, mono.sample_rate)
    }
}

impl Default for FourierTransform {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            window: WindowFunction::Hann,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for STFT config.
pub struct StftConfig {
    /// The FFT size value.
    pub fft_size: usize,
    /// The hop size value.
    pub hop_size: usize,
    /// The window value.
    pub window: WindowFunction,
    /// The pad final frame value.
    pub pad_final_frame: bool,
}

impl StftConfig {
    /// Creates a new value.
    pub fn new(fft_size: usize, hop_size: usize) -> Result<Self> {
        FourierTransform::new(fft_size)?;
        FrameSpec::new(fft_size, hop_size)?;
        Ok(Self {
            fft_size,
            hop_size,
            window: WindowFunction::Hann,
            pad_final_frame: false,
        })
    }

    /// Returns window.
    pub fn window(mut self, window: WindowFunction) -> Self {
        self.window = window;
        self
    }

    /// Returns pad final frame.
    pub fn pad_final_frame(mut self, pad_final_frame: bool) -> Self {
        self.pad_final_frame = pad_final_frame;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for spectrogram frame.
pub struct SpectrogramFrame {
    /// The start sample value.
    pub start_sample: usize,
    /// The start seconds value.
    pub start_seconds: f64,
    /// The spectrum value.
    pub spectrum: Spectrum,
}

/// Returns spectrogram.
pub fn spectrogram(
    samples: &[f32],
    sample_rate: u32,
    config: &StftConfig,
) -> Result<Vec<SpectrogramFrame>> {
    let transform = FourierTransform::with_window(config.fft_size, config.window)?;
    let spec = FrameSpec::new(config.fft_size, config.hop_size)?;
    let mut frames = spec
        .frames(samples)
        .map(|(start_sample, frame)| {
            Ok(SpectrogramFrame {
                start_sample,
                start_seconds: start_sample as f64 / sample_rate as f64,
                spectrum: transform.analyze_samples(frame, sample_rate)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let needs_partial = config.pad_final_frame
        && !samples.is_empty()
        && (frames.is_empty()
            || (frames.last().unwrap().start_sample + config.fft_size < samples.len()));
    if needs_partial {
        let start_sample = if samples.len() <= config.fft_size {
            0
        } else {
            ((samples.len() - 1) / config.hop_size) * config.hop_size
        };
        let padded = zero_pad_to(samples[start_sample..].to_vec(), config.fft_size);
        frames.push(SpectrogramFrame {
            start_sample,
            start_seconds: start_sample as f64 / sample_rate as f64,
            spectrum: transform.analyze_samples(&padded, sample_rate)?,
        });
    }
    Ok(frames)
}

/// Returns zero crossing rate.
pub fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|pair| {
            let left = pair[0];
            let right = pair[1];
            (left < 0.0 && right >= 0.0) || (left >= 0.0 && right < 0.0)
        })
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

/// Returns spectral flux.
pub fn spectral_flux(previous: &Spectrum, current: &Spectrum) -> f32 {
    previous
        .bins
        .iter()
        .zip(current.bins.iter())
        .map(|(left, right)| (right.magnitude - left.magnitude).max(0.0))
        .sum()
}

/// Returns spectral features.
pub fn spectral_features(spectrum: &Spectrum) -> SpectralFeatures {
    let total_power = spectrum.bins.iter().map(|bin| bin.power).sum::<f32>();
    if total_power <= f32::EPSILON {
        return SpectralFeatures {
            centroid_hz: 0.0,
            bandwidth_hz: 0.0,
            rolloff_hz: 0.0,
            flatness: 0.0,
            dominant_frequency_hz: None,
        };
    }

    let centroid_hz = spectrum
        .bins
        .iter()
        .map(|bin| bin.frequency_hz * bin.power)
        .sum::<f32>()
        / total_power;
    let bandwidth_hz = (spectrum
        .bins
        .iter()
        .map(|bin| (bin.frequency_hz - centroid_hz).powi(2) * bin.power)
        .sum::<f32>()
        / total_power)
        .sqrt();
    let rolloff_hz = rolloff_hz(spectrum, total_power, 0.85);
    let flatness = spectral_flatness(spectrum);

    SpectralFeatures {
        centroid_hz,
        bandwidth_hz,
        rolloff_hz,
        flatness,
        dominant_frequency_hz: spectrum.dominant_frequency_hz(),
    }
}

/// Extracts windowed spectral feature frames.
pub fn spectral_feature_frames(
    samples: &[f32],
    options: SpectralFeatureOptions,
) -> Result<Vec<SpectralFeatureFrame>> {
    options.validate()?;
    let config = StftConfig::new(options.fft_size, options.hop_size)?;
    let frames = spectrogram(samples, options.sample_rate, &config)?;
    Ok(frames
        .iter()
        .map(|frame| {
            let features = frame.spectrum.features();
            SpectralFeatureFrame {
                start_seconds: frame.start_seconds as f32,
                centroid_hz: features.centroid_hz,
                bandwidth_hz: features.bandwidth_hz,
                rolloff_hz: features.rolloff_hz,
                flatness: features.flatness,
                energy: frame.spectrum.bins.iter().map(|bin| bin.power).sum(),
                mel_bands: options
                    .mel_band_count
                    .map(|count| mel_style_bands(&frame.spectrum, count))
                    .unwrap_or_default(),
            }
        })
        .collect())
}

fn mel_style_bands(spectrum: &Spectrum, band_count: usize) -> Vec<f32> {
    if band_count == 0 {
        return Vec::new();
    }
    let mut bands = vec![0.0_f32; band_count];
    let max_hz = spectrum.sample_rate as f32 / 2.0;
    if max_hz <= 0.0 {
        return bands;
    }
    for bin in spectrum.bins.iter().skip(1) {
        let normalized = (hz_to_mel(bin.frequency_hz) / hz_to_mel(max_hz)).clamp(0.0, 1.0);
        let index = ((normalized * band_count as f32).floor() as usize).min(band_count - 1);
        bands[index] += bin.power;
    }
    bands
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz.max(0.0) / 700.0).log10()
}

fn rolloff_hz(spectrum: &Spectrum, total_power: f32, threshold: f32) -> f32 {
    let target = total_power * threshold;
    let mut accumulated = 0.0;
    for bin in &spectrum.bins {
        accumulated += bin.power;
        if accumulated >= target {
            return bin.frequency_hz;
        }
    }
    spectrum.bins.last().map_or(0.0, |bin| bin.frequency_hz)
}

fn spectral_flatness(spectrum: &Spectrum) -> f32 {
    let powers = spectrum
        .bins
        .iter()
        .skip(1)
        .map(|bin| bin.power.max(1.0e-12))
        .collect::<Vec<_>>();
    if powers.is_empty() {
        return 0.0;
    }
    let geometric =
        (powers.iter().map(|power| power.ln()).sum::<f32>() / powers.len() as f32).exp();
    let arithmetic = powers.iter().sum::<f32>() / powers.len() as f32;
    if arithmetic <= f32::EPSILON {
        0.0
    } else {
        geometric / arithmetic
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for spectral analyzer.
pub struct SpectralAnalyzer {
    name: String,
    transform: FourierTransform,
    min_magnitude: f32,
}

impl SpectralAnalyzer {
    /// Creates a new value.
    pub fn new(transform: FourierTransform) -> Self {
        Self {
            name: "audio_spectral".to_string(),
            transform,
            min_magnitude: 0.001,
        }
    }

    /// Returns min magnitude.
    pub fn min_magnitude(mut self, value: f32) -> Self {
        self.min_magnitude = value.max(0.0);
        self
    }
}

impl Default for SpectralAnalyzer {
    fn default() -> Self {
        Self::new(FourierTransform::default())
    }
}

impl AudioAnalyzer for SpectralAnalyzer {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Vec<AnalysisEvent>> {
        let spectrum = self.transform.analyze_frame(frame)?;
        let Some((dominant, magnitude)) = spectrum
            .bins
            .iter()
            .skip(1)
            .map(|bin| (bin.frequency_hz, bin.magnitude))
            .max_by(|a, b| a.1.total_cmp(&b.1))
        else {
            return Ok(Vec::new());
        };
        if magnitude < self.min_magnitude {
            return Ok(Vec::new());
        }
        Ok(vec![AnalysisEvent::new(
            self.name(),
            format!("audio:dominant_frequency:{dominant:.2}hz"),
        )
        .at_timestamp(frame.timestamp)
        .score(magnitude)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_analysis_core::WindowFunction;
    use audio_contracts::{AudioBuffer, AudioFrame, Timebase, Timestamp};

    fn sine_len(freq_hz: f32, sample_rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

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

    fn sine(freq_hz: f32, sample_rate: u32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    fn frame(samples: Vec<f32>, sample_rate: u32) -> AudioBuffer {
        let _ = sample_rate;
        AudioBuffer::F32(samples)
    }

    #[test]
    fn rejects_invalid_fft_sizes_and_sample_rates() {
        assert!(FourierTransform::new(0).is_err());
        assert!(FourierTransform::new(1000).is_err());
        let transform = FourierTransform::new(512).unwrap();
        assert!(transform.analyze_samples(&[0.0; 512], 0).is_err());
        assert!(StftConfig::new(512, 0).is_err());
    }

    #[test]
    fn fft_detects_dominant_sine_frequency() {
        let transform = FourierTransform::with_window(1024, WindowFunction::Rectangular).unwrap();
        let spectrum = transform
            .analyze_samples(&sine(440.0, 8192, 1024), 8192)
            .unwrap();
        let dominant = spectrum.dominant_frequency_hz().unwrap();
        assert!((dominant - 440.0).abs() <= 8.0);
    }

    #[test]
    fn fft_detects_multiple_reference_tones() {
        let transform = FourierTransform::with_window(4096, WindowFunction::Rectangular).unwrap();
        for frequency in [220.0, 440.0, 1_000.0, 1_010.0] {
            let spectrum = transform
                .analyze_samples(&sine_len(frequency, 48_000, 4096), 48_000)
                .unwrap();
            let dominant = spectrum.dominant_frequency_hz().unwrap();
            assert!(
                (dominant - frequency).abs() <= 48_000.0 / 4096.0,
                "expected {frequency}, got {dominant}"
            );
        }
    }

    #[test]
    fn zero_pads_short_input_to_fft_size() {
        let transform = FourierTransform::with_window(512, WindowFunction::Rectangular).unwrap();
        let spectrum = transform.analyze_samples(&[1.0], 48_000).unwrap();
        assert_eq!(spectrum.fft_size, 512);
        assert_eq!(spectrum.bins.len(), 257);
        assert!(spectrum.bins.iter().all(|bin| bin.magnitude.is_finite()));
    }

    #[test]
    fn stft_returns_expected_frame_count() {
        let config = StftConfig::new(256, 128).unwrap();
        let frames = spectrogram(&vec![0.0; 1024], 48_000, &config).unwrap();
        assert_eq!(frames.len(), 7);
        assert_eq!(frames[1].start_sample, 128);
        assert_approx_eq(frames[1].start_seconds as f32, 128.0 / 48_000.0, 1.0e-7);
    }

    #[test]
    fn stft_can_include_a_padded_final_frame() {
        let config = StftConfig::new(256, 128).unwrap().pad_final_frame(true);
        let frames = spectrogram(&vec![0.0; 300], 48_000, &config).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].start_sample, 256);
    }

    #[test]
    fn silent_spectrum_has_empty_features() {
        let transform = FourierTransform::with_window(512, WindowFunction::Rectangular).unwrap();
        let features = transform
            .analyze_samples(&vec![0.0; 512], 48_000)
            .unwrap()
            .features();
        assert_eq!(features.dominant_frequency_hz, None);
        assert_eq!(features.centroid_hz, 0.0);
        assert_eq!(features.bandwidth_hz, 0.0);
        assert_eq!(features.rolloff_hz, 0.0);
        assert_eq!(features.flatness, 0.0);
    }

    #[test]
    fn spectral_features_characterize_tone_and_noise() {
        let transform = FourierTransform::with_window(1024, WindowFunction::Rectangular).unwrap();
        let tone = transform
            .analyze_samples(&sine_len(1024.0, 8192, 1024), 8192)
            .unwrap()
            .features();
        assert!((tone.centroid_hz - 1024.0).abs() <= 8.0);
        let noise = transform
            .analyze_samples(&white_noise(42, 1024), 8192)
            .unwrap()
            .features();
        assert!(noise.flatness > tone.flatness);
    }

    #[test]
    fn spectral_feature_frames_include_mel_bands() {
        let options = SpectralFeatureOptions::new(512, 256, 8_192)
            .unwrap()
            .mel_band_count(Some(6))
            .unwrap();
        let frames = spectral_feature_frames(&sine_len(440.0, 8_192, 1024), options).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].mel_bands.len(), 6);
        assert!(frames.iter().all(|frame| frame.energy.is_finite()));
        assert!(SpectralFeatureOptions::new(512, 0, 8_192).is_err());
        assert!(SpectralFeatureOptions::new(512, 256, 8_192)
            .unwrap()
            .mel_band_count(Some(0))
            .is_err());
    }

    #[test]
    fn zero_crossing_rate_and_spectral_flux_are_exposed() {
        let alternating = [1.0, -1.0, 1.0, -1.0];
        assert!(zero_crossing_rate(&alternating) > 0.9);

        let transform = FourierTransform::with_window(512, WindowFunction::Rectangular).unwrap();
        let quiet = transform.analyze_samples(&vec![0.0; 512], 8_000).unwrap();
        let tone = transform
            .analyze_samples(&sine_len(440.0, 8_000, 512), 8_000)
            .unwrap();
        assert!(spectral_flux(&quiet, &tone) > 0.0);
    }

    #[test]
    fn spectral_analyzer_respects_min_magnitude() {
        let buffer = frame(sine_len(440.0, 8192, 1024), 8192);
        let frame =
            AudioFrame::new(Timestamp::new(10, Timebase::new(1, 8192)), 8192, 1, &buffer).unwrap();
        let transform = FourierTransform::with_window(1024, WindowFunction::Rectangular).unwrap();
        let mut quiet = SpectralAnalyzer::new(transform.clone()).min_magnitude(100.0);
        assert!(quiet.process_frame(&frame).unwrap().is_empty());

        let mut audible = SpectralAnalyzer::new(transform).min_magnitude(0.001);
        let events = audible.process_frame(&frame).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].timestamp, Some(frame.timestamp));
        assert!(events[0].label.starts_with("audio:dominant_frequency:"));
    }
}
