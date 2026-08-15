#![doc = include_str!("../README.md")]

pub mod contracts;
mod effects;
mod offline;
pub mod operations;
pub mod surface;
pub use effects::*;
pub use offline::*;
use std::collections::VecDeque;

use audio_analysis_core::{
    interleaved_to_mono, normalized_samples, peak, rms, windowed_level_series, AudioFeatureSeries,
    ChannelMix, FrameSpec,
};
use audio_contracts::{
    AnalysisEvent, AudioAnalyzer, AudioBuffer, AudioFrame, DetectError, OwnedAudioFrame, Result,
};
use math_signal_core::{BiquadCoefficients as DesignedBiquadCoefficients, SampleRate};
use video_analysis_ingest::{AudioFrameSource, MediaSourceInfo};

/// Trait for audio transform implementations.
pub trait AudioTransform {
    /// Returns name.
    fn name(&self) -> &str;
    /// Returns reset.
    fn reset(&mut self);
    /// Returns process frame.
    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>>;

    /// Returns finish.
    fn finish(&mut self) -> Result<Vec<OwnedAudioFrame>> {
        Ok(Vec::new())
    }
}

#[derive(Default)]
/// Data type for audio processor.
pub struct AudioProcessor {
    transforms: Vec<Box<dyn AudioTransform>>,
    output_channels: Option<u16>,
}

impl AudioProcessor {
    /// Creates a new value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns transform.
    pub fn transform<T: AudioTransform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Returns gain.
    pub fn gain(self, linear: f32) -> Self {
        self.transform(Gain { linear })
    }

    /// Returns gain db.
    pub fn gain_db(self, db: f32) -> Self {
        self.gain(10.0_f32.powf(db / 20.0))
    }

    /// Returns hard clip.
    pub fn hard_clip(self, min: f32, max: f32) -> Self {
        self.transform(HardClip { min, max })
    }

    /// Returns mono.
    pub fn mono(mut self, mix: ChannelMix) -> Self {
        self.output_channels = Some(1);
        self.transform(Mono { mix })
    }

    /// Returns dc block.
    pub fn dc_block(self, coefficient: f32) -> Self {
        self.transform(DcBlock::new(coefficient))
    }

    /// Returns biquad.
    pub fn biquad(self, spec: BiquadSpec) -> Self {
        self.transform(BiquadFilter::new(spec))
    }

    /// Returns noise gate.
    pub fn noise_gate(self, spec: NoiseGateSpec) -> Self {
        self.transform(NoiseGate { spec })
    }

    /// Returns process frame.
    pub fn process_frame(&mut self, frame: OwnedAudioFrame) -> Result<Option<OwnedAudioFrame>> {
        let mut current = Some(frame);
        for transform in &mut self.transforms {
            let Some(frame) = current else {
                return Ok(None);
            };
            let frame_ref = frame.as_frame()?;
            current = transform.process_frame(&frame_ref)?;
        }
        Ok(current)
    }

    /// Returns finish.
    pub fn finish(&mut self) -> Result<Vec<OwnedAudioFrame>> {
        let mut output = Vec::new();
        for transform in &mut self.transforms {
            output.extend(transform.finish()?);
        }
        Ok(output)
    }

    /// Returns reset.
    pub fn reset(&mut self) {
        for transform in &mut self.transforms {
            transform.reset();
        }
    }

    /// Returns output channels.
    pub fn output_channels(&self) -> Option<u16> {
        self.output_channels
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// Options for deterministic loudness-oriented metrics.
pub struct LoudnessOptions {
    /// Analysis frame size in samples.
    pub frame_size: usize,
    /// Analysis hop size in samples.
    pub hop_size: usize,
    /// Optional gate threshold in dBFS for approximate loudness aggregation.
    pub gate_threshold_db: Option<f32>,
}

impl LoudnessOptions {
    /// Creates a validated loudness options value.
    pub fn new(frame_size: usize, hop_size: usize) -> Result<Self> {
        let options = Self {
            frame_size,
            hop_size,
            gate_threshold_db: None,
        };
        options.validate()?;
        Ok(options)
    }

    /// Sets the optional gate threshold.
    pub fn gate_threshold_db(mut self, threshold_db: Option<f32>) -> Result<Self> {
        self.gate_threshold_db = threshold_db;
        self.validate()?;
        Ok(self)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        FrameSpec::new(self.frame_size, self.hop_size)?;
        if let Some(threshold) = self.gate_threshold_db {
            if !threshold.is_finite() {
                return Err(DetectError::InvalidArgument(
                    "gate_threshold_db must be finite".to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl Default for LoudnessOptions {
    fn default() -> Self {
        Self {
            frame_size: 1024,
            hop_size: 512,
            gate_threshold_db: Some(-70.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Deterministic loudness report.
pub struct LoudnessReport {
    /// Peak level in dBFS, floored for silence.
    pub peak_dbfs: f32,
    /// RMS level in dBFS, floored for silence.
    pub rms_dbfs: f32,
    /// Peak-to-RMS difference in decibels.
    pub crest_factor_db: f32,
    /// Approximate LUFS-style integrated value.
    pub approximate_lufs: f32,
    /// Windowed level series used to derive the report.
    pub frame_series: AudioFeatureSeries,
}

/// Computes deterministic loudness-oriented metrics for mono normalized samples.
pub fn analyze_loudness(
    samples: &[f32],
    sample_rate: u32,
    options: LoudnessOptions,
) -> Result<LoudnessReport> {
    options.validate()?;
    for sample in samples {
        if !sample.is_finite() {
            return Err(DetectError::InvalidArgument(
                "loudness samples must contain only finite values".to_string(),
            ));
        }
    }
    let frame_series = windowed_level_series(
        samples,
        sample_rate,
        FrameSpec::new(options.frame_size, options.hop_size)?,
    )?;
    let peak_dbfs = amplitude_to_dbfs(peak(samples));
    let rms_dbfs = amplitude_to_dbfs(rms(samples));
    let crest_factor_db = (peak_dbfs - rms_dbfs).max(0.0);
    let approximate_lufs = approximate_lufs(samples, &frame_series, options.gate_threshold_db)?;
    Ok(LoudnessReport {
        peak_dbfs,
        rms_dbfs,
        crest_factor_db,
        approximate_lufs,
        frame_series,
    })
}

fn approximate_lufs(
    samples: &[f32],
    frame_series: &AudioFeatureSeries,
    gate_threshold_db: Option<f32>,
) -> Result<f32> {
    let threshold = gate_threshold_db.unwrap_or(f32::NEG_INFINITY);
    let mut gated_square_sum = 0.0_f32;
    let mut gated_count = 0_usize;
    for point in &frame_series.points {
        let frame_rms_db = point
            .values
            .get("rms")
            .copied()
            .map(amplitude_to_dbfs)
            .unwrap_or(DBFS_FLOOR);
        if frame_rms_db >= threshold {
            let start = (point.start_seconds * frame_series.sample_rate as f32).round() as usize;
            let end = (point.end_seconds * frame_series.sample_rate as f32).round() as usize;
            for sample in &samples[start.min(samples.len())..end.min(samples.len())] {
                gated_square_sum += sample * sample;
                gated_count += 1;
            }
        }
    }
    let gated_rms = if gated_count == 0 {
        0.0
    } else {
        (gated_square_sum / gated_count as f32).sqrt()
    };
    let dbfs = amplitude_to_dbfs(gated_rms);
    Ok((dbfs - 0.691).max(DBFS_FLOOR))
}

const DBFS_FLOOR: f32 = -120.0;

fn amplitude_to_dbfs(amplitude: f32) -> f32 {
    if !amplitude.is_finite() || amplitude <= 0.0 {
        return DBFS_FLOOR;
    }
    (20.0 * amplitude.log10()).max(DBFS_FLOOR)
}

/// Data type for processed audio source.
pub struct ProcessedAudioSource<S> {
    source: S,
    processor: AudioProcessor,
    source_info: MediaSourceInfo,
    pending: VecDeque<OwnedAudioFrame>,
    finished: bool,
}

impl<S: AudioFrameSource> ProcessedAudioSource<S> {
    /// Creates a new value.
    pub fn new(source: S, processor: AudioProcessor) -> Self {
        let mut source_info = source.source_info().clone();
        if let Some(channels) = processor.output_channels() {
            for audio in &mut source_info.audio {
                audio.channels = channels;
            }
        }
        Self {
            source,
            processor,
            source_info,
            pending: VecDeque::new(),
            finished: false,
        }
    }

    /// Returns inner.
    pub fn inner(&self) -> &S {
        &self.source
    }

    /// Returns inner mut.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.source
    }

    /// Returns processor.
    pub fn processor(&self) -> &AudioProcessor {
        &self.processor
    }

    /// Returns processor mut.
    pub fn processor_mut(&mut self) -> &mut AudioProcessor {
        &mut self.processor
    }
}

impl<S: AudioFrameSource> AudioFrameSource for ProcessedAudioSource<S> {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.source_info
    }

    fn next_audio_frame(&mut self) -> Result<Option<OwnedAudioFrame>> {
        loop {
            if let Some(frame) = self.pending.pop_front() {
                return Ok(Some(frame));
            }
            if self.finished {
                return Ok(None);
            }
            match self.source.next_audio_frame()? {
                Some(frame) => {
                    if let Some(processed) = self.processor.process_frame(frame)? {
                        return Ok(Some(processed));
                    }
                }
                None => {
                    self.pending.extend(self.processor.finish()?);
                    self.finished = true;
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for gain.
pub struct Gain {
    /// The linear value.
    pub linear: f32,
}

impl AudioTransform for Gain {
    fn name(&self) -> &str {
        "audio_gain"
    }

    fn reset(&mut self) {}

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if !self.linear.is_finite() {
            return Err(DetectError::InvalidArgument(
                "gain must be finite".to_string(),
            ));
        }
        map_samples(frame, |sample| sample * self.linear).map(Some)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for hard clip.
pub struct HardClip {
    /// The min value.
    pub min: f32,
    /// The max value.
    pub max: f32,
}

impl AudioTransform for HardClip {
    fn name(&self) -> &str {
        "audio_hard_clip"
    }

    fn reset(&mut self) {}

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if !self.min.is_finite() || !self.max.is_finite() || self.min > self.max {
            return Err(DetectError::InvalidArgument(
                "clip bounds must be finite and min must be less than or equal to max".to_string(),
            ));
        }
        map_samples(frame, |sample| sample.clamp(self.min, self.max)).map(Some)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for mono.
pub struct Mono {
    /// The mix value.
    pub mix: ChannelMix,
}

impl AudioTransform for Mono {
    fn name(&self) -> &str {
        "audio_mono"
    }

    fn reset(&mut self) {}

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        let samples = interleaved_to_mono(frame.data, frame.channels, self.mix)?;
        OwnedAudioFrame::new(
            frame.timestamp,
            frame.sample_rate,
            1,
            AudioBuffer::F32(samples),
        )
        .map(Some)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for dc block.
pub struct DcBlock {
    /// The coefficient value.
    pub coefficient: f32,
    states: Vec<DcBlockState>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct DcBlockState {
    previous_input: f32,
    previous_output: f32,
}

impl DcBlock {
    /// Creates a new value.
    pub fn new(coefficient: f32) -> Self {
        Self {
            coefficient,
            states: Vec::new(),
        }
    }
}

impl AudioTransform for DcBlock {
    fn name(&self) -> &str {
        "audio_dc_block"
    }

    fn reset(&mut self) {
        self.states.clear();
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if !self.coefficient.is_finite() || !(0.0..1.0).contains(&self.coefficient) {
            return Err(DetectError::InvalidArgument(
                "dc block coefficient must be finite and in the range [0, 1)".to_string(),
            ));
        }
        let samples = normalized_interleaved(frame)?;
        let channels = frame.channels as usize;
        self.states.resize(channels, DcBlockState::default());
        let mut output = Vec::with_capacity(samples.len());
        for frame_samples in samples.chunks_exact(channels) {
            for (channel, sample) in frame_samples.iter().enumerate() {
                let state = &mut self.states[channel];
                let filtered =
                    *sample - state.previous_input + self.coefficient * state.previous_output;
                state.previous_input = *sample;
                state.previous_output = filtered;
                output.push(filtered);
            }
        }
        owned_like(frame, output).map(Some)
    }
}

/// Re-exports the biquad kind API.
pub use math_signal_core::BiquadDesign as BiquadKind;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for biquad spec.
pub struct BiquadSpec {
    /// The kind value.
    pub kind: BiquadKind,
    /// The cutoff hz value.
    pub cutoff_hz: f32,
    /// The q value.
    pub q: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for biquad filter.
pub struct BiquadFilter {
    /// The spec value.
    pub spec: BiquadSpec,
    states: Vec<BiquadState>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct BiquadState {
    z1: f32,
    z2: f32,
}

impl BiquadFilter {
    /// Creates a new value.
    pub fn new(spec: BiquadSpec) -> Self {
        Self {
            spec,
            states: Vec::new(),
        }
    }
}

impl AudioTransform for BiquadFilter {
    fn name(&self) -> &str {
        "audio_biquad"
    }

    fn reset(&mut self) {
        self.states.clear();
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        let coefficients = coefficients(self.spec, frame.sample_rate)?;
        let samples = normalized_interleaved(frame)?;
        let channels = frame.channels as usize;
        self.states.resize(channels, BiquadState::default());
        let mut output = Vec::with_capacity(samples.len());
        for frame_samples in samples.chunks_exact(channels) {
            for (channel, sample) in frame_samples.iter().enumerate() {
                let state = &mut self.states[channel];
                let filtered = coefficients.b0 * *sample + state.z1;
                state.z1 = coefficients.b1 * *sample - coefficients.a1 * filtered + state.z2;
                state.z2 = coefficients.b2 * *sample - coefficients.a2 * filtered;
                output.push(filtered);
            }
        }
        owned_like(frame, output).map(Some)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for noise gate spec.
pub struct NoiseGateSpec {
    /// The threshold value.
    pub threshold: f32,
    /// The attenuation value.
    pub attenuation: f32,
}

impl Default for NoiseGateSpec {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            attenuation: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for noise gate.
pub struct NoiseGate {
    /// The spec value.
    pub spec: NoiseGateSpec,
}

impl AudioTransform for NoiseGate {
    fn name(&self) -> &str {
        "audio_noise_gate"
    }

    fn reset(&mut self) {}

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if !self.spec.threshold.is_finite()
            || self.spec.threshold < 0.0
            || !self.spec.attenuation.is_finite()
            || !(0.0..=1.0).contains(&self.spec.attenuation)
        {
            return Err(DetectError::InvalidArgument(
                "noise gate threshold must be finite and non-negative, and attenuation must be in [0, 1]"
                    .to_string(),
            ));
        }
        map_samples(frame, |sample| {
            if sample.abs() < self.spec.threshold {
                sample * self.spec.attenuation
            } else {
                sample
            }
        })
        .map(Some)
    }
}

/// Returns collect audio frames.
pub fn collect_audio_frames<S: AudioFrameSource>(source: &mut S) -> Result<Vec<OwnedAudioFrame>> {
    let mut frames = Vec::new();
    while let Some(frame) = source.next_audio_frame()? {
        frames.push(frame);
    }
    Ok(frames)
}

/// Returns process audio source.
pub fn process_audio_source<S: AudioFrameSource>(
    source: &mut S,
    processor: &mut AudioProcessor,
) -> Result<Vec<OwnedAudioFrame>> {
    let mut frames = Vec::new();
    while let Some(frame) = source.next_audio_frame()? {
        if let Some(processed) = processor.process_frame(frame)? {
            frames.push(processed);
        }
    }
    frames.extend(processor.finish()?);
    Ok(frames)
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for audio energy analyzer.
pub struct AudioEnergyAnalyzer {
    silence_threshold: f32,
    loud_threshold: f32,
    current_label: Option<String>,
}

impl AudioEnergyAnalyzer {
    /// Creates a new value.
    pub fn new(silence_threshold: f32, loud_threshold: f32) -> Result<Self> {
        if !silence_threshold.is_finite()
            || !loud_threshold.is_finite()
            || silence_threshold < 0.0
            || loud_threshold <= silence_threshold
        {
            return Err(DetectError::InvalidArgument(
                "audio energy thresholds must be finite and ordered".to_string(),
            ));
        }
        Ok(Self {
            silence_threshold,
            loud_threshold,
            current_label: None,
        })
    }

    /// Returns rms.
    pub fn rms(buffer: &AudioBuffer) -> f32 {
        audio_buffer_rms(buffer)
    }
}

impl Default for AudioEnergyAnalyzer {
    fn default() -> Self {
        Self {
            silence_threshold: 0.01,
            loud_threshold: 0.2,
            current_label: None,
        }
    }
}

impl AudioAnalyzer for AudioEnergyAnalyzer {
    fn name(&self) -> &str {
        "audio_energy"
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Vec<AnalysisEvent>> {
        let rms = audio_buffer_rms(frame.data);
        let label = if rms < self.silence_threshold {
            "audio:silence"
        } else if rms > self.loud_threshold {
            "audio:loud"
        } else {
            "audio:active"
        };
        if self.current_label.as_deref() == Some(label) {
            return Ok(Vec::new());
        }
        self.current_label = Some(label.to_string());
        Ok(vec![AnalysisEvent::new(self.name(), label)
            .at_timestamp(frame.timestamp)
            .score(rms)])
    }
}

/// Returns audio buffer rms.
pub fn audio_buffer_rms(buffer: &AudioBuffer) -> f32 {
    let (sum, count) = match buffer {
        AudioBuffer::U8(values) => values.iter().fold((0.0, 0_usize), |(sum, count), value| {
            let sample = (*value as f32 - 128.0) / 128.0;
            (sum + sample * sample, count + 1)
        }),
        AudioBuffer::I16(values) => values.iter().fold((0.0, 0_usize), |(sum, count), value| {
            let sample = *value as f32 / i16::MAX as f32;
            (sum + sample * sample, count + 1)
        }),
        AudioBuffer::I32(values) => values.iter().fold((0.0, 0_usize), |(sum, count), value| {
            let sample = *value as f32 / i32::MAX as f32;
            (sum + sample * sample, count + 1)
        }),
        AudioBuffer::F32(values) => values.iter().fold((0.0, 0_usize), |(sum, count), value| {
            (sum + value * value, count + 1)
        }),
    };
    if count == 0 {
        0.0
    } else {
        (sum / count as f32).sqrt()
    }
}

fn map_samples(frame: &AudioFrame<'_>, mut map: impl FnMut(f32) -> f32) -> Result<OwnedAudioFrame> {
    let samples = normalized_interleaved(frame)?
        .into_iter()
        .map(&mut map)
        .collect();
    owned_like(frame, samples)
}

fn normalized_interleaved(frame: &AudioFrame<'_>) -> Result<Vec<f32>> {
    if frame.channels == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate: frame.sample_rate,
            channels: frame.channels,
        });
    }
    if !frame.data.len().is_multiple_of(frame.channels as usize) {
        return Err(DetectError::InvalidArgument(format!(
            "audio buffer length {} is not divisible by channel count {}",
            frame.data.len(),
            frame.channels
        )));
    }
    Ok(normalized_samples(frame.data))
}

fn owned_like(frame: &AudioFrame<'_>, samples: Vec<f32>) -> Result<OwnedAudioFrame> {
    OwnedAudioFrame::new(
        frame.timestamp,
        frame.sample_rate,
        frame.channels,
        AudioBuffer::F32(samples),
    )
}

fn coefficients(spec: BiquadSpec, sample_rate: u32) -> Result<DesignedBiquadCoefficients> {
    spec.kind
        .design(SampleRate::new(sample_rate)?, spec.cutoff_hz, spec.q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_contracts::{AudioPipeline, Timebase, Timestamp};
    use proptest::prelude::*;
    use video_analysis_ingest::{analyze_audio_source, AudioStreamInfo, SourceMode};

    fn assert_approx_eq(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    fn frame(samples: Vec<f32>, channels: u16) -> OwnedAudioFrame {
        OwnedAudioFrame::new(
            Timestamp::new(0, Timebase::new(1, 48_000)),
            48_000,
            channels,
            AudioBuffer::F32(samples),
        )
        .unwrap()
    }

    fn samples(frame: &OwnedAudioFrame) -> &[f32] {
        match &frame.data {
            AudioBuffer::F32(samples) => samples,
            _ => panic!("expected f32 samples"),
        }
    }

    fn sine(freq_hz: f32, sample_rate: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    fn mean_abs(samples: &[f32]) -> f32 {
        samples.iter().map(|sample| sample.abs()).sum::<f32>() / samples.len() as f32
    }

    #[test]
    fn gain_changes_samples_and_preserves_timing() {
        let input = frame(vec![0.25, -0.5], 1);
        let timestamp = input.timestamp;
        let mut processor = AudioProcessor::new().gain(2.0);

        let output = processor.process_frame(input).unwrap().unwrap();

        assert_eq!(output.timestamp, timestamp);
        assert_eq!(samples(&output), &[0.5, -1.0]);
    }

    #[test]
    fn gain_db_matches_linear_gain() {
        let input = frame(vec![0.25, -0.5], 1);
        let mut processor = AudioProcessor::new().gain_db(6.020_600_3);
        let output = processor.process_frame(input).unwrap().unwrap();
        assert_approx_eq(samples(&output)[0], 0.5, 1.0e-5);
        assert_approx_eq(samples(&output)[1], -1.0, 1.0e-5);
    }

    #[test]
    fn clipping_clamps_samples() {
        let mut processor = AudioProcessor::new().hard_clip(-0.5, 0.5);
        let output = processor
            .process_frame(frame(vec![-1.0, -0.25, 0.75], 1))
            .unwrap()
            .unwrap();

        assert_eq!(samples(&output), &[-0.5, -0.25, 0.5]);
    }

    #[test]
    fn invalid_clip_and_noise_gate_configs_error() {
        let mut clip = AudioProcessor::new().hard_clip(1.0, -1.0);
        assert!(clip.process_frame(frame(vec![0.0], 1)).is_err());

        let mut gate = AudioProcessor::new().noise_gate(NoiseGateSpec {
            threshold: -0.1,
            attenuation: 0.0,
        });
        assert!(gate.process_frame(frame(vec![0.0], 1)).is_err());

        let mut gate = AudioProcessor::new().noise_gate(NoiseGateSpec {
            threshold: 0.1,
            attenuation: 1.1,
        });
        assert!(gate.process_frame(frame(vec![0.0], 1)).is_err());
    }

    #[test]
    fn mono_conversion_changes_channel_count() {
        let mut processor = AudioProcessor::new().mono(ChannelMix::Average);
        let timestamp = Timestamp::new(24, Timebase::new(1, 48_000));
        let input = OwnedAudioFrame::new(
            timestamp,
            48_000,
            2,
            AudioBuffer::F32(vec![1.0, -1.0, 0.5, 0.25]),
        )
        .unwrap();
        let output = processor.process_frame(input).unwrap().unwrap();

        assert_eq!(output.channels, 1);
        assert_eq!(output.sample_rate, 48_000);
        assert_eq!(output.timestamp, timestamp);
        assert_eq!(samples(&output), &[0.0, 0.375]);
    }

    #[test]
    fn dc_block_reduces_constant_offset_across_frames() {
        let mut processor = AudioProcessor::new().dc_block(0.95);
        let first = processor
            .process_frame(frame(vec![1.0; 64], 1))
            .unwrap()
            .unwrap();
        let second = processor
            .process_frame(frame(vec![1.0; 64], 1))
            .unwrap()
            .unwrap();

        assert!(mean_abs(samples(&second)) < mean_abs(samples(&first)));
        processor.reset();
        let reset = processor
            .process_frame(frame(vec![1.0; 64], 1))
            .unwrap()
            .unwrap();
        assert_approx_eq(samples(&reset)[0], samples(&first)[0], 1.0e-6);
    }

    #[test]
    fn biquad_config_validation_rejects_invalid_values() {
        for spec in [
            BiquadSpec {
                kind: BiquadKind::LowPass,
                cutoff_hz: 0.0,
                q: 0.707,
            },
            BiquadSpec {
                kind: BiquadKind::LowPass,
                cutoff_hz: 24_000.0,
                q: 0.707,
            },
            BiquadSpec {
                kind: BiquadKind::LowPass,
                cutoff_hz: 1_000.0,
                q: 0.0,
            },
        ] {
            let mut processor = AudioProcessor::new().biquad(spec);
            assert!(processor.process_frame(frame(vec![0.0; 16], 1)).is_err());
        }
    }

    #[test]
    fn low_pass_attenuates_high_frequency_more_than_low_frequency() {
        let spec = BiquadSpec {
            kind: BiquadKind::LowPass,
            cutoff_hz: 1_000.0,
            q: 0.707,
        };
        let mut low_filter = AudioProcessor::new().biquad(spec);
        let mut high_filter = AudioProcessor::new().biquad(spec);
        let low = low_filter
            .process_frame(frame(sine(200.0, 48_000, 4096), 1))
            .unwrap()
            .unwrap();
        let high = high_filter
            .process_frame(frame(sine(8_000.0, 48_000, 4096), 1))
            .unwrap()
            .unwrap();

        assert!(mean_abs(samples(&low)) > mean_abs(samples(&high)) * 4.0);
    }

    #[test]
    fn high_pass_attenuates_low_frequency_content() {
        let spec = BiquadSpec {
            kind: BiquadKind::HighPass,
            cutoff_hz: 1_000.0,
            q: 0.707,
        };
        let mut low_filter = AudioProcessor::new().biquad(spec);
        let mut high_filter = AudioProcessor::new().biquad(spec);
        let low = low_filter
            .process_frame(frame(sine(80.0, 48_000, 4096), 1))
            .unwrap()
            .unwrap();
        let high = high_filter
            .process_frame(frame(sine(8_000.0, 48_000, 4096), 1))
            .unwrap()
            .unwrap();

        assert!(mean_abs(samples(&high)) > mean_abs(samples(&low)) * 4.0);
    }

    #[test]
    fn band_pass_and_notch_filters_shape_center_frequency() {
        let band_pass = BiquadSpec {
            kind: BiquadKind::BandPass,
            cutoff_hz: 1_000.0,
            q: 2.0,
        };
        let mut centered_filter = AudioProcessor::new().biquad(band_pass);
        let mut distant_filter = AudioProcessor::new().biquad(band_pass);
        let centered = centered_filter
            .process_frame(frame(sine(1_000.0, 48_000, 4096), 1))
            .unwrap()
            .unwrap();
        let distant = distant_filter
            .process_frame(frame(sine(8_000.0, 48_000, 4096), 1))
            .unwrap()
            .unwrap();
        assert!(mean_abs(samples(&centered)) > mean_abs(samples(&distant)) * 2.0);

        let notch = BiquadSpec {
            kind: BiquadKind::Notch,
            cutoff_hz: 1_000.0,
            q: 4.0,
        };
        let mut notched_filter = AudioProcessor::new().biquad(notch);
        let mut distant_filter = AudioProcessor::new().biquad(notch);
        let notched = notched_filter
            .process_frame(frame(sine(1_000.0, 48_000, 4096), 1))
            .unwrap()
            .unwrap();
        let distant = distant_filter
            .process_frame(frame(sine(8_000.0, 48_000, 4096), 1))
            .unwrap()
            .unwrap();
        assert!(mean_abs(samples(&distant)) > mean_abs(samples(&notched)) * 2.0);
    }

    #[test]
    fn processor_chain_order_is_deterministic() {
        let mut processor = AudioProcessor::new().gain(2.0).hard_clip(-0.75, 0.75);
        let output = processor
            .process_frame(frame(vec![0.5, -0.5], 1))
            .unwrap()
            .unwrap();
        assert_eq!(samples(&output), &[0.75, -0.75]);
    }

    #[derive(Default)]
    struct CountingAnalyzer;

    impl AudioAnalyzer for CountingAnalyzer {
        fn name(&self) -> &str {
            "counting"
        }

        fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Vec<AnalysisEvent>> {
            Ok(vec![
                AnalysisEvent::new(self.name(), "frame").at_timestamp(frame.timestamp)
            ])
        }
    }

    struct VecAudioSource {
        source_info: MediaSourceInfo,
        frames: VecDeque<OwnedAudioFrame>,
    }

    impl VecAudioSource {
        fn new(frames: Vec<OwnedAudioFrame>) -> Self {
            Self {
                source_info: MediaSourceInfo::recorded("memory").with_audio(AudioStreamInfo {
                    sample_rate: 48_000,
                    channels: 1,
                    sample_format: audio_contracts::AudioSampleFormat::F32,
                }),
                frames: frames.into(),
            }
        }
    }

    impl AudioFrameSource for VecAudioSource {
        fn source_info(&self) -> &MediaSourceInfo {
            &self.source_info
        }

        fn next_audio_frame(&mut self) -> Result<Option<OwnedAudioFrame>> {
            Ok(self.frames.pop_front())
        }
    }

    #[test]
    fn processed_audio_source_can_feed_audio_pipeline() {
        let source = VecAudioSource::new(vec![frame(vec![0.25], 1), frame(vec![0.5], 1)]);
        let processor = AudioProcessor::new().gain(2.0);
        let mut source = ProcessedAudioSource::new(source, processor);
        let mut pipeline = AudioPipeline::builder()
            .analyzer(CountingAnalyzer)
            .build()
            .unwrap();

        let result = analyze_audio_source(&mut source, &mut pipeline, |_| Ok(())).unwrap();

        assert_eq!(result.frames_processed, 2);
        assert_eq!(result.events.len(), 2);
        assert_eq!(source.source_info().mode, SourceMode::Recorded);
        assert_eq!(source.source_info().audio[0].channels, 1);
    }

    #[test]
    fn process_audio_source_preserves_frame_count_and_data_shape() {
        let mut source = VecAudioSource::new(vec![frame(vec![0.25], 1), frame(vec![0.5], 1)]);
        let mut processor = AudioProcessor::new().gain(2.0);
        let frames = process_audio_source(&mut source, &mut processor).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(samples(&frames[0]), &[0.5]);
        assert_eq!(samples(&frames[1]), &[1.0]);
    }

    #[test]
    fn audio_energy_analyzer_emits_only_label_changes() {
        let mut analyzer = AudioEnergyAnalyzer::default();
        let silence = frame(vec![0.0; 16], 1);
        let active = frame(vec![0.05; 16], 1);
        let loud = frame(vec![0.5; 16], 1);

        let first = analyzer
            .process_frame(&silence.as_frame().unwrap())
            .unwrap();
        let repeated = analyzer
            .process_frame(&silence.as_frame().unwrap())
            .unwrap();
        let active = analyzer.process_frame(&active.as_frame().unwrap()).unwrap();
        let loud = analyzer.process_frame(&loud.as_frame().unwrap()).unwrap();

        assert_eq!(first[0].label, "audio:silence");
        assert!(repeated.is_empty());
        assert_eq!(active[0].label, "audio:active");
        assert_eq!(loud[0].label, "audio:loud");
    }

    #[test]
    fn loudness_report_handles_silence_and_sine() {
        let silence = analyze_loudness(
            &[0.0; 1024],
            48_000,
            LoudnessOptions::new(256, 128).unwrap(),
        )
        .unwrap();
        assert_eq!(silence.peak_dbfs, DBFS_FLOOR);
        assert_eq!(silence.rms_dbfs, DBFS_FLOOR);
        assert_eq!(silence.approximate_lufs, DBFS_FLOOR);
        assert!(silence.frame_series.points.len() > 1);

        let sine = sine(1_000.0, 48_000, 4096);
        let report = analyze_loudness(
            &sine,
            48_000,
            LoudnessOptions::new(1024, 512)
                .unwrap()
                .gate_threshold_db(Some(-80.0))
                .unwrap(),
        )
        .unwrap();
        assert!(report.peak_dbfs <= 0.0);
        assert!(report.rms_dbfs < report.peak_dbfs);
        assert!(report.crest_factor_db > 0.0);
        assert!(report.approximate_lufs.is_finite());
    }

    #[test]
    fn loudness_options_reject_invalid_values() {
        assert!(LoudnessOptions::new(0, 1).is_err());
        assert!(LoudnessOptions::new(256, 0).is_err());
        assert!(LoudnessOptions::new(256, 128)
            .unwrap()
            .gate_threshold_db(Some(f32::NAN))
            .is_err());
        assert!(analyze_loudness(&[f32::INFINITY], 48_000, LoudnessOptions::default()).is_err());
    }

    #[test]
    fn stateful_filters_preserve_state_across_chunks() {
        let mut processor = AudioProcessor::new().dc_block(0.95);
        let first = processor
            .process_frame(frame(vec![1.0; 4], 1))
            .unwrap()
            .unwrap();
        let second = processor
            .process_frame(frame(vec![1.0; 4], 1))
            .unwrap()
            .unwrap();

        assert!(samples(&second)[0].abs() < samples(&first)[0].abs());
    }

    proptest! {
        #[test]
        fn gain_then_clip_stays_within_bounds(input_samples in proptest::collection::vec(-2.0_f32..2.0, 1..128)) {
            let mut processor = AudioProcessor::new().gain(1.5).hard_clip(-0.75, 0.75);
            let output = processor.process_frame(frame(input_samples, 1)).unwrap().unwrap();
            prop_assert!(samples(&output).iter().all(|sample| (-0.75..=0.75).contains(sample)));
        }

        #[test]
        fn processor_output_frame_shape_remains_valid(
            channels in 1_u16..=4,
            frames in 1_usize..32,
            values in proptest::collection::vec(-1.0_f32..1.0, 1..256),
        ) {
            let len = frames * channels as usize;
            let mut input = values;
            input.resize(len, 0.0);
            let mut processor = AudioProcessor::new().gain(0.5);
            let output = processor.process_frame(frame(input, channels)).unwrap().unwrap();
            prop_assert_eq!(output.channels, channels);
            prop_assert_eq!(output.data.len() % channels as usize, 0);
            prop_assert_eq!(output.sample_rate, 48_000);
        }
    }
}
