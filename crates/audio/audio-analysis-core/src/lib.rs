#![doc = include_str!("../README.md")]

mod clip;
pub mod surface;
pub use clip::{AudioClip, ConcatPolicy, FadeCurve, MixPolicy};
/// Re-exports the math signal core API.
pub use math_signal_core::{
    BiquadCoefficients, BiquadDesign, FirKernel1d, FrameStride, InterpolationMode, ResampleRatio,
    ResampleSpec, SampleRate, WindowFunction, WindowSpec,
};
use std::collections::{BTreeMap, BTreeSet};

use audio_contracts::{AudioBuffer, AudioFrame, DetectError, Result, Timebase, Timestamp};
use tensor_data::{F32Tensor, F32TensorView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for audio format spec.
pub struct AudioFormatSpec {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
    /// The frame samples value.
    pub frame_samples: Option<usize>,
}

impl AudioFormatSpec {
    /// Creates a new value.
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let spec = Self {
            sample_rate,
            channels,
            frame_samples: None,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Returns frame samples.
    pub fn frame_samples(mut self, frame_samples: usize) -> Result<Self> {
        self.frame_samples = Some(frame_samples);
        self.validate()?;
        Ok(self)
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 || self.channels == 0 {
            return Err(DetectError::InvalidAudioFormat {
                sample_rate: self.sample_rate,
                channels: self.channels,
            });
        }
        if self.frame_samples == Some(0) {
            return Err(DetectError::InvalidArgument(
                "frame_samples must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns duration seconds.
    pub fn duration_seconds(&self, samples_per_channel: usize) -> Result<f64> {
        self.validate()?;
        Ok(samples_per_channel as f64 / self.sample_rate as f64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing channel mix.
pub enum ChannelMix {
    /// The average variant.
    Average,
    /// The first variant.
    First,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for mono samples.
pub struct MonoSamples {
    /// Timestamp associated with this value.
    pub timestamp: Timestamp,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// The samples value.
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// One window of generic audio feature values.
pub struct AudioFeaturePoint {
    /// Window start time in seconds.
    pub start_seconds: f32,
    /// Window end time in seconds.
    pub end_seconds: f32,
    /// Named finite feature values for this window.
    pub values: BTreeMap<String, f32>,
}

impl AudioFeaturePoint {
    /// Creates a validated feature point.
    pub fn new(
        start_seconds: f32,
        end_seconds: f32,
        values: BTreeMap<String, f32>,
    ) -> Result<Self> {
        let point = Self {
            start_seconds,
            end_seconds,
            values,
        };
        point.validate()?;
        Ok(point)
    }

    /// Validates this feature point.
    pub fn validate(&self) -> Result<()> {
        validate_time_range(self.start_seconds, self.end_seconds, "audio feature point")?;
        validate_feature_values(&self.values)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A windowed series of generic audio features.
pub struct AudioFeatureSeries {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of source channels represented by the series.
    pub channels: u16,
    /// Analysis frame size in samples per channel.
    pub frame_size: usize,
    /// Analysis hop size in samples per channel.
    pub hop_size: usize,
    /// Feature points ordered by time.
    pub points: Vec<AudioFeaturePoint>,
}

impl AudioFeatureSeries {
    /// Creates a validated feature series.
    pub fn new(
        sample_rate: u32,
        channels: u16,
        frame_size: usize,
        hop_size: usize,
        points: Vec<AudioFeaturePoint>,
    ) -> Result<Self> {
        let series = Self {
            sample_rate,
            channels,
            frame_size,
            hop_size,
            points,
        };
        series.validate()?;
        Ok(series)
    }

    /// Validates this feature series.
    pub fn validate(&self) -> Result<()> {
        AudioFormatSpec::new(self.sample_rate, self.channels)?.frame_samples(self.frame_size)?;
        FrameSpec::new(self.frame_size, self.hop_size)?;
        let mut previous_start = 0.0_f32;
        for point in &self.points {
            point.validate()?;
            if point.start_seconds < previous_start
                && !nearly_equal(point.start_seconds, previous_start)
            {
                return Err(DetectError::InvalidArgument(
                    "audio feature points must be ordered by start time".to_string(),
                ));
            }
            previous_start = point.start_seconds;
        }
        Ok(())
    }

    /// Returns the covered duration in seconds.
    pub fn duration_seconds(&self) -> f32 {
        self.points
            .last()
            .map(|point| point.end_seconds)
            .unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Summary metrics for an audio feature series.
pub struct AudioFeatureSummary {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Duration covered by the series.
    pub duration_seconds: f32,
    /// Number of feature frames.
    pub frame_count: usize,
    /// Named finite summary metrics.
    pub metrics: BTreeMap<String, f32>,
}

impl AudioFeatureSummary {
    /// Creates a validated feature summary.
    pub fn new(
        sample_rate: u32,
        duration_seconds: f32,
        frame_count: usize,
        metrics: BTreeMap<String, f32>,
    ) -> Result<Self> {
        let summary = Self {
            sample_rate,
            duration_seconds,
            frame_count,
            metrics,
        };
        summary.validate()?;
        Ok(summary)
    }

    /// Validates this summary.
    pub fn validate(&self) -> Result<()> {
        AudioFormatSpec::new(self.sample_rate, 1)?;
        if !self.duration_seconds.is_finite() || self.duration_seconds < 0.0 {
            return Err(DetectError::InvalidArgument(
                "audio feature summary duration_seconds must be finite and non-negative"
                    .to_string(),
            ));
        }
        validate_feature_values(&self.metrics)
    }
}

impl MonoSamples {
    /// Returns duration seconds.
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for audio waveform batch view.
pub struct AudioWaveformBatchView<'a> {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    tensor: F32TensorView<'a>,
}

impl<'a> AudioWaveformBatchView<'a> {
    /// Creates a new value.
    pub fn new(sample_rate: u32, tensor: F32TensorView<'a>) -> Result<Self> {
        let batch = Self {
            sample_rate,
            tensor,
        };
        batch.validate()?;
        Ok(batch)
    }

    /// Builds this value from dims.
    pub fn from_dims(
        sample_rate: u32,
        dims: impl Into<Vec<usize>>,
        values: &'a [f32],
    ) -> Result<Self> {
        Self::new(sample_rate, F32TensorView::from_dims(dims, values)?)
    }

    /// Returns tensor.
    pub fn tensor(&self) -> &F32TensorView<'a> {
        &self.tensor
    }

    /// Returns batch size.
    pub fn batch_size(&self) -> usize {
        self.tensor.shape().dimensions()[0]
    }

    /// Returns channel count.
    pub fn channel_count(&self) -> usize {
        self.tensor.shape().dimensions()[1]
    }

    /// Returns time steps.
    pub fn time_steps(&self) -> usize {
        self.tensor.shape().dimensions()[2]
    }

    /// Returns duration seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.time_steps() as f64 / self.sample_rate as f64
    }

    /// Returns waveform.
    pub fn waveform(&self, batch_index: usize, channel_index: usize) -> Result<&'a [f32]> {
        if batch_index >= self.batch_size() || channel_index >= self.channel_count() {
            return Err(DetectError::InvalidArgument(format!(
                "waveform index [{batch_index}, {channel_index}] is out of bounds for [{}, {}]",
                self.batch_size(),
                self.channel_count()
            )));
        }
        let time_steps = self.time_steps();
        let start = batch_index * self.channel_count() * time_steps + channel_index * time_steps;
        Ok(&self.tensor.values()[start..start + time_steps])
    }

    fn validate(&self) -> Result<()> {
        AudioFormatSpec::new(self.sample_rate, 1)?;
        self.tensor.validate()?;
        if self.tensor.shape().rank() != 3 {
            return Err(DetectError::InvalidArgument(
                "audio waveform batches must use rank 3 [B,C,T] tensors".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for owned audio waveform batch.
pub struct OwnedAudioWaveformBatch {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    tensor: F32Tensor,
}

impl OwnedAudioWaveformBatch {
    /// Creates a new value.
    pub fn new(sample_rate: u32, tensor: F32Tensor) -> Result<Self> {
        let batch = Self {
            sample_rate,
            tensor,
        };
        batch.as_view()?;
        Ok(batch)
    }

    /// Builds this value from audio frames.
    pub fn from_audio_frames(frames: &[audio_contracts::OwnedAudioFrame]) -> Result<Self> {
        if frames.is_empty() {
            return Err(DetectError::InvalidArgument(
                "audio waveform batches must contain at least one frame".to_string(),
            ));
        }
        let first = &frames[0];
        let sample_rate = first.sample_rate;
        let channels = first.channels as usize;
        let time_steps = first.samples_per_channel();
        let mut values = Vec::with_capacity(frames.len() * channels * time_steps);

        for frame in frames {
            if frame.sample_rate != sample_rate
                || frame.channels as usize != channels
                || frame.samples_per_channel() != time_steps
            {
                return Err(DetectError::InvalidArgument(
                    "all audio frames in a batch must share sample rate, channel count, and samples per channel"
                        .to_string(),
                ));
            }
            let normalized = normalized_samples(&frame.data);
            for channel in 0..channels {
                for time_index in 0..time_steps {
                    values.push(normalized[time_index * channels + channel]);
                }
            }
        }

        Self::new(
            sample_rate,
            F32Tensor::from_dims([frames.len(), channels, time_steps], values)?,
        )
    }

    /// Returns tensor.
    pub fn tensor(&self) -> &F32Tensor {
        &self.tensor
    }

    /// Borrows this value as a view.
    pub fn as_view(&self) -> Result<AudioWaveformBatchView<'_>> {
        AudioWaveformBatchView::new(self.sample_rate, self.tensor.as_view())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for frame spec.
pub struct FrameSpec {
    /// The frame size value.
    pub frame_size: usize,
    /// The hop size value.
    pub hop_size: usize,
}

impl FrameSpec {
    /// Creates a new value.
    pub fn new(frame_size: usize, hop_size: usize) -> Result<Self> {
        FrameStride::new(frame_size, hop_size)?;
        Ok(Self {
            frame_size,
            hop_size,
        })
    }

    /// Returns frames.
    pub fn frames<'a>(&self, samples: &'a [f32]) -> AudioFrames<'a> {
        AudioFrames {
            samples,
            spec: *self,
            offset: 0,
        }
    }

    /// Returns frame count.
    pub fn frame_count(&self, samples_len: usize) -> usize {
        FrameStride::from(*self).frame_count(samples_len)
    }
}

impl From<FrameSpec> for FrameStride {
    fn from(value: FrameSpec) -> Self {
        Self {
            frame_size: value.frame_size,
            hop_size: value.hop_size,
        }
    }
}

impl TryFrom<FrameStride> for FrameSpec {
    type Error = DetectError;

    fn try_from(value: FrameStride) -> Result<Self> {
        Self::new(value.frame_size, value.hop_size)
    }
}

#[derive(Debug, Clone)]
/// Data type for audio frames.
pub struct AudioFrames<'a> {
    samples: &'a [f32],
    spec: FrameSpec,
    offset: usize,
}

impl<'a> Iterator for AudioFrames<'a> {
    type Item = (usize, &'a [f32]);

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.offset.checked_add(self.spec.frame_size)?;
        if end > self.samples.len() {
            return None;
        }
        let offset = self.offset;
        self.offset += self.spec.hop_size;
        Some((offset, &self.samples[offset..end]))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Data type for streaming frame config.
pub struct StreamingFrameConfig {
    /// The frame size value.
    pub frame_size: usize,
    /// The hop size value.
    pub hop_size: usize,
    /// The channel mix value.
    pub channel_mix: ChannelMix,
    /// The max buffered samples value.
    pub max_buffered_samples: usize,
}

impl StreamingFrameConfig {
    /// Creates a new value.
    pub fn new(frame_size: usize, hop_size: usize) -> Result<Self> {
        FrameSpec::new(frame_size, hop_size)?;
        Ok(Self {
            frame_size,
            hop_size,
            channel_mix: ChannelMix::Average,
            max_buffered_samples: frame_size.saturating_add(hop_size).max(frame_size),
        })
    }

    /// Returns channel mix.
    pub fn channel_mix(mut self, mix: ChannelMix) -> Self {
        self.channel_mix = mix;
        self
    }

    /// Returns max buffered samples.
    pub fn max_buffered_samples(mut self, samples: usize) -> Self {
        self.max_buffered_samples = samples.max(self.frame_size);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for audio window.
pub struct AudioWindow {
    /// Timestamp associated with this value.
    pub timestamp: Timestamp,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// The start sample value.
    pub start_sample: u64,
    /// The samples value.
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for streaming frame buffer.
pub struct StreamingFrameBuffer {
    config: StreamingFrameConfig,
    sample_rate: Option<u32>,
    channels: Option<u16>,
    buffer: Vec<f32>,
    buffered_start_sample: u64,
    next_window_start_sample: Option<u64>,
}

impl StreamingFrameBuffer {
    /// Creates a new value.
    pub fn new(config: StreamingFrameConfig) -> Result<Self> {
        FrameSpec::new(config.frame_size, config.hop_size)?;
        if config.max_buffered_samples < config.frame_size {
            return Err(DetectError::InvalidArgument(
                "max_buffered_samples must be at least frame_size".to_string(),
            ));
        }
        Ok(Self {
            config,
            sample_rate: None,
            channels: None,
            buffer: Vec::new(),
            buffered_start_sample: 0,
            next_window_start_sample: None,
        })
    }

    /// Adds push frame to this value.
    pub fn push_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Vec<AudioWindow>> {
        self.validate_stream_format(frame)?;
        let frame_start_sample = timestamp_to_sample(frame.timestamp, frame.sample_rate)?;
        if self.next_window_start_sample.is_none() {
            self.buffered_start_sample = frame_start_sample;
            self.next_window_start_sample = Some(frame_start_sample);
        }

        let buffered_end_sample = self.buffered_start_sample + self.buffer.len() as u64;
        if frame_start_sample > buffered_end_sample {
            self.buffer.clear();
            self.buffered_start_sample = frame_start_sample;
            self.next_window_start_sample = Some(frame_start_sample);
        } else if frame_start_sample < buffered_end_sample {
            return Err(DetectError::InvalidArgument(
                "streaming audio frames must not overlap".to_string(),
            ));
        }

        self.buffer.extend(interleaved_to_mono(
            frame.data,
            frame.channels,
            self.config.channel_mix,
        )?);

        let mut windows = Vec::new();
        let mut next_start = self
            .next_window_start_sample
            .expect("next window start is initialized above");
        let buffered_end_sample = self.buffered_start_sample + self.buffer.len() as u64;
        while next_start + self.config.frame_size as u64 <= buffered_end_sample {
            let offset = (next_start - self.buffered_start_sample) as usize;
            let end = offset + self.config.frame_size;
            windows.push(AudioWindow {
                timestamp: sample_to_timestamp(next_start, frame.sample_rate),
                sample_rate: frame.sample_rate,
                start_sample: next_start,
                samples: self.buffer[offset..end].to_vec(),
            });
            next_start += self.config.hop_size as u64;
        }
        self.next_window_start_sample = Some(next_start);
        self.trim_consumed();
        self.enforce_buffer_bound()?;
        Ok(windows)
    }

    /// Returns reset.
    pub fn reset(&mut self) {
        self.sample_rate = None;
        self.channels = None;
        self.buffer.clear();
        self.buffered_start_sample = 0;
        self.next_window_start_sample = None;
    }

    /// Returns buffered samples.
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    fn validate_stream_format(&mut self, frame: &AudioFrame<'_>) -> Result<()> {
        match (self.sample_rate, self.channels) {
            (None, None) => {
                self.sample_rate = Some(frame.sample_rate);
                self.channels = Some(frame.channels);
                Ok(())
            }
            (Some(sample_rate), Some(channels))
                if sample_rate == frame.sample_rate && channels == frame.channels =>
            {
                Ok(())
            }
            _ => Err(DetectError::InvalidArgument(
                "streaming audio sample_rate and channels must remain stable".to_string(),
            )),
        }
    }

    fn trim_consumed(&mut self) {
        let Some(next_start) = self.next_window_start_sample else {
            return;
        };
        if next_start <= self.buffered_start_sample {
            return;
        }
        let drop = (next_start - self.buffered_start_sample).min(self.buffer.len() as u64) as usize;
        if drop > 0 {
            self.buffer.drain(0..drop);
            self.buffered_start_sample += drop as u64;
        }
    }

    fn enforce_buffer_bound(&mut self) -> Result<()> {
        if self.buffer.len() <= self.config.max_buffered_samples {
            return Ok(());
        }
        Err(DetectError::InvalidArgument(format!(
            "streaming audio buffer exceeded max_buffered_samples ({})",
            self.config.max_buffered_samples
        )))
    }
}

/// Returns mono samples.
pub fn mono_samples(frame: &AudioFrame<'_>) -> Result<MonoSamples> {
    mono_samples_with_mix(frame, ChannelMix::Average)
}

/// Returns mono samples with mix.
pub fn mono_samples_with_mix(frame: &AudioFrame<'_>, mix: ChannelMix) -> Result<MonoSamples> {
    let samples = interleaved_to_mono(frame.data, frame.channels, mix)?;
    Ok(MonoSamples {
        timestamp: frame.timestamp,
        sample_rate: frame.sample_rate,
        samples,
    })
}

/// Returns interleaved to mono.
pub fn interleaved_to_mono(
    buffer: &AudioBuffer,
    channels: u16,
    mix: ChannelMix,
) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate: 1,
            channels,
        });
    }
    let channels = channels as usize;
    if !buffer.len().is_multiple_of(channels) {
        return Err(DetectError::InvalidArgument(format!(
            "audio buffer length {} is not divisible by channel count {channels}",
            buffer.len()
        )));
    }
    let normalized = normalized_samples(buffer);
    Ok(match mix {
        ChannelMix::First => normalized
            .chunks_exact(channels)
            .map(|frame| frame[0])
            .collect(),
        ChannelMix::Average => normalized
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect(),
    })
}

/// Returns normalized samples.
pub fn normalized_samples(buffer: &AudioBuffer) -> Vec<f32> {
    match buffer {
        AudioBuffer::U8(values) => values
            .iter()
            .map(|value| (*value as f32 - 128.0) / 128.0)
            .collect(),
        AudioBuffer::I16(values) => values
            .iter()
            .map(|value| *value as f32 / i16::MAX as f32)
            .collect(),
        AudioBuffer::I32(values) => values
            .iter()
            .map(|value| *value as f32 / i32::MAX as f32)
            .collect(),
        AudioBuffer::F32(values) => values.clone(),
    }
}

/// Returns rms.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Returns peak.
pub fn peak(samples: &[f32]) -> f32 {
    samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max)
}

/// Returns mean absolute.
pub fn mean_absolute(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|sample| sample.abs()).sum::<f32>() / samples.len() as f32
}

/// Returns zero crossing rate for adjacent sample pairs.
pub fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|pair| pair[0].is_sign_positive() != pair[1].is_sign_positive())
        .filter(|pair| pair[0] != 0.0 && pair[1] != 0.0)
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

/// Converts mono samples into a windowed level feature series.
pub fn windowed_level_series(
    samples: &[f32],
    sample_rate: u32,
    frame_spec: FrameSpec,
) -> Result<AudioFeatureSeries> {
    AudioFormatSpec::new(sample_rate, 1)?;
    FrameSpec::new(frame_spec.frame_size, frame_spec.hop_size)?;
    validate_samples(samples)?;
    let mut points = Vec::with_capacity(frame_spec.frame_count(samples.len()));
    for (start_sample, frame) in frame_spec.frames(samples) {
        let end_sample = start_sample + frame.len();
        let mut values = BTreeMap::new();
        values.insert("rms".to_string(), rms(frame));
        values.insert("peak".to_string(), peak(frame));
        values.insert("meanAbsolute".to_string(), mean_absolute(frame));
        values.insert("zeroCrossingRate".to_string(), zero_crossing_rate(frame));
        points.push(AudioFeaturePoint::new(
            start_sample as f32 / sample_rate as f32,
            end_sample as f32 / sample_rate as f32,
            values,
        )?);
    }
    AudioFeatureSeries::new(
        sample_rate,
        1,
        frame_spec.frame_size,
        frame_spec.hop_size,
        points,
    )
}

/// Summarizes a windowed audio feature series.
pub fn summarize_feature_series(series: &AudioFeatureSeries) -> Result<AudioFeatureSummary> {
    series.validate()?;
    let mut names = BTreeSet::new();
    for point in &series.points {
        names.extend(point.values.keys().cloned());
    }

    let mut metrics = BTreeMap::new();
    for name in names {
        let values = series
            .points
            .iter()
            .filter_map(|point| point.values.get(&name).copied())
            .collect::<Vec<_>>();
        if values.is_empty() {
            continue;
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        metrics.insert(format!("{name}.mean"), mean);
        metrics.insert(format!("{name}.max"), max);
    }

    AudioFeatureSummary::new(
        series.sample_rate,
        series.duration_seconds(),
        series.points.len(),
        metrics,
    )
}

/// Returns zero pad to.
pub fn zero_pad_to(mut samples: Vec<f32>, target_len: usize) -> Vec<f32> {
    samples.resize(target_len, 0.0);
    samples
}

/// Returns seconds to samples.
pub fn seconds_to_samples(seconds: f64, sample_rate: u32) -> Result<u64> {
    AudioFormatSpec::new(sample_rate, 1)?;
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(DetectError::InvalidArgument(
            "audio duration must be a finite non-negative value".to_string(),
        ));
    }
    let samples = seconds * sample_rate as f64;
    if !samples.is_finite() || samples < 0.0 {
        return Err(DetectError::InvalidArgument(
            "audio timestamp must resolve to a finite non-negative sample index".to_string(),
        ));
    }
    Ok(samples.round() as u64)
}

/// Returns samples to seconds.
pub fn samples_to_seconds(samples: u64, sample_rate: u32) -> Result<f64> {
    AudioFormatSpec::new(sample_rate, 1)?;
    Ok(samples as f64 / sample_rate as f64)
}

/// Returns timestamp to sample.
pub fn timestamp_to_sample(timestamp: Timestamp, sample_rate: u32) -> Result<u64> {
    if timestamp.timebase.den == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate,
            channels: 1,
        });
    }
    seconds_to_samples(timestamp.seconds(), sample_rate)
}

/// Returns sample to timestamp.
pub fn sample_to_timestamp(sample: u64, sample_rate: u32) -> Timestamp {
    Timestamp::new(sample as i64, Timebase::new(1, sample_rate as i32))
}

fn validate_time_range(start_seconds: f32, end_seconds: f32, label: &str) -> Result<()> {
    if !start_seconds.is_finite() || start_seconds < 0.0 {
        return Err(DetectError::InvalidArgument(format!(
            "{label} start_seconds must be finite and non-negative"
        )));
    }
    if !end_seconds.is_finite() || end_seconds < 0.0 {
        return Err(DetectError::InvalidArgument(format!(
            "{label} end_seconds must be finite and non-negative"
        )));
    }
    if end_seconds < start_seconds {
        return Err(DetectError::InvalidArgument(format!(
            "{label} end_seconds must be greater than or equal to start_seconds"
        )));
    }
    Ok(())
}

fn validate_feature_values(values: &BTreeMap<String, f32>) -> Result<()> {
    for (name, value) in values {
        if name.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "audio feature names must not be empty".to_string(),
            ));
        }
        if !value.is_finite() {
            return Err(DetectError::InvalidArgument(format!(
                "audio feature `{name}` must be finite"
            )));
        }
    }
    Ok(())
}

fn validate_samples(samples: &[f32]) -> Result<()> {
    for sample in samples {
        if !sample.is_finite() {
            return Err(DetectError::InvalidArgument(
                "audio samples must contain only finite values".to_string(),
            ));
        }
    }
    Ok(())
}

fn nearly_equal(left: f32, right: f32) -> bool {
    (left - right).abs() <= f32::EPSILON * 16.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_contracts::{AudioBuffer, AudioFrame, Timebase, Timestamp};
    use proptest::prelude::*;

    fn assert_approx_eq(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    fn assert_approx_slice(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "slice lengths differ");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "index {index}: expected {actual} to be within {tolerance} of {expected}"
            );
        }
    }

    fn ts() -> Timestamp {
        Timestamp::new(0, Timebase::new(1, 48_000))
    }

    fn frame_at(sample: u64, samples: Vec<f32>) -> AudioBuffer {
        let _ = sample;
        AudioBuffer::F32(samples)
    }

    #[test]
    fn mixes_interleaved_stereo_to_mono() {
        let buffer = AudioBuffer::F32(vec![1.0, -1.0, 0.5, 0.25]);
        let mono = interleaved_to_mono(&buffer, 2, ChannelMix::Average).unwrap();
        assert_eq!(mono, vec![0.0, 0.375]);
    }

    #[test]
    fn normalizes_all_supported_sample_formats() {
        assert_approx_slice(
            &normalized_samples(&AudioBuffer::U8(vec![0, 128, 255])),
            &[-1.0, 0.0, 127.0 / 128.0],
            1.0e-6,
        );
        assert_approx_slice(
            &normalized_samples(&AudioBuffer::I16(vec![i16::MIN, 0, i16::MAX])),
            &[i16::MIN as f32 / i16::MAX as f32, 0.0, 1.0],
            1.0e-6,
        );
        assert_approx_slice(
            &normalized_samples(&AudioBuffer::I32(vec![i32::MIN, 0, i32::MAX])),
            &[i32::MIN as f32 / i32::MAX as f32, 0.0, 1.0],
            1.0e-6,
        );
        assert_eq!(
            normalized_samples(&AudioBuffer::F32(vec![-0.25, 0.0, 0.5])),
            vec![-0.25, 0.0, 0.5]
        );
    }

    #[test]
    fn first_channel_mix_uses_first_interleaved_sample() {
        let buffer = AudioBuffer::F32(vec![1.0, -1.0, 0.5, 0.25]);
        let mono = interleaved_to_mono(&buffer, 2, ChannelMix::First).unwrap();
        assert_eq!(mono, vec![1.0, 0.5]);
    }

    #[test]
    fn batches_existing_audio_frames_into_channel_major_waveforms() {
        let first = audio_contracts::OwnedAudioFrame::new(
            ts(),
            48_000,
            1,
            AudioBuffer::F32(vec![0.1, 0.2]),
        )
        .unwrap();
        let second = audio_contracts::OwnedAudioFrame::new(
            ts(),
            48_000,
            1,
            AudioBuffer::F32(vec![0.3, 0.4]),
        )
        .unwrap();

        let batch = OwnedAudioWaveformBatch::from_audio_frames(&[first, second]).unwrap();
        let view = batch.as_view().unwrap();
        assert_eq!(view.batch_size(), 2);
        assert_eq!(view.waveform(1, 0).unwrap(), &[0.3, 0.4]);
    }

    #[test]
    fn mono_mix_rejects_invalid_channel_layouts() {
        assert!(interleaved_to_mono(&AudioBuffer::F32(vec![1.0]), 0, ChannelMix::Average).is_err());
        assert!(interleaved_to_mono(
            &AudioBuffer::F32(vec![1.0, 2.0, 3.0]),
            2,
            ChannelMix::Average
        )
        .is_err());
    }

    #[test]
    fn frame_spec_validates_sizes_and_counts_frames() {
        assert!(FrameSpec::new(0, 1).is_err());
        assert!(FrameSpec::new(4, 0).is_err());
        let spec = FrameSpec::new(4, 2).unwrap();
        assert_eq!(spec.frame_count(3), 0);
        assert_eq!(spec.frame_count(4), 1);
        assert_eq!(spec.frame_count(6), 2);
        assert_eq!(spec.frame_count(7), 2);
    }

    #[test]
    fn frame_spec_iterates_over_hops() {
        let spec = FrameSpec::new(4, 2).unwrap();
        let samples = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let frames = spec.frames(&samples).collect::<Vec<_>>();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], (0, &[0.0, 1.0, 2.0, 3.0][..]));
        assert_eq!(frames[1], (2, &[2.0, 3.0, 4.0, 5.0][..]));
    }

    #[test]
    fn feature_contracts_validate_ranges_and_values() {
        assert!(AudioFeaturePoint::new(1.0, 0.5, BTreeMap::new()).is_err());
        assert!(AudioFeaturePoint::new(f32::NAN, 1.0, BTreeMap::new()).is_err());

        let mut values = BTreeMap::new();
        values.insert("rms".to_string(), f32::INFINITY);
        assert!(AudioFeaturePoint::new(0.0, 1.0, values).is_err());

        let point =
            AudioFeaturePoint::new(0.0, 0.5, BTreeMap::from([("rms".to_string(), 0.25)])).unwrap();
        assert!(AudioFeatureSeries::new(0, 1, 128, 64, vec![point.clone()]).is_err());
        assert!(AudioFeatureSeries::new(48_000, 1, 0, 64, vec![point.clone()]).is_err());
        assert!(AudioFeatureSummary::new(
            48_000,
            f32::NAN,
            1,
            BTreeMap::from([("rms.mean".to_string(), 0.25)])
        )
        .is_err());
    }

    #[test]
    fn windowed_level_series_summarizes_deterministic_metrics() {
        let series =
            windowed_level_series(&[0.0, 1.0, -1.0, 0.0], 4, FrameSpec::new(2, 1).unwrap())
                .unwrap();
        assert_eq!(series.points.len(), 3);
        assert_eq!(series.points[0].start_seconds, 0.0);
        assert_eq!(series.points[0].end_seconds, 0.5);
        assert_approx_eq(series.points[0].values["rms"], 0.5_f32.sqrt(), 1.0e-6);
        assert_approx_eq(series.points[1].values["zeroCrossingRate"], 1.0, 1.0e-6);

        let summary = summarize_feature_series(&series).unwrap();
        assert_eq!(summary.sample_rate, 4);
        assert_eq!(summary.frame_count, 3);
        assert_approx_eq(summary.duration_seconds, 1.0, 1.0e-6);
        assert!(summary.metrics["rms.mean"] > 0.0);
        assert_eq!(zero_crossing_rate(&[0.0, 1.0, 0.0]), 0.0);
    }

    #[test]
    fn audio_frame_to_mono_preserves_timing() {
        let buffer = AudioBuffer::I16(vec![0, i16::MAX]);
        let frame = AudioFrame::new(ts(), 48_000, 1, &buffer).unwrap();
        let mono = mono_samples(&frame).unwrap();
        assert_eq!(mono.timestamp, ts());
        assert_eq!(mono.sample_rate, 48_000);
        assert_eq!(mono.samples, vec![0.0, 1.0]);
    }

    #[test]
    fn hann_window_tapers_edges() {
        let windowed = WindowFunction::Hann.apply(&[1.0, 1.0, 1.0, 1.0]);
        assert!(windowed[0].abs() < 0.000_001);
        assert!(windowed[1] > 0.7);
        assert!(windowed[2] > 0.7);
        assert!(windowed[3].abs() < 0.000_001);
    }

    #[test]
    fn streaming_buffer_emits_windows_inside_one_chunk() {
        let config = StreamingFrameConfig::new(4, 2).unwrap();
        let mut buffer = StreamingFrameBuffer::new(config).unwrap();
        let samples = AudioBuffer::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let frame = AudioFrame::new(ts(), 48_000, 1, &samples).unwrap();

        let windows = buffer.push_frame(&frame).unwrap();

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].start_sample, 0);
        assert_eq!(windows[0].samples, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(windows[1].start_sample, 2);
        assert_eq!(windows[1].samples, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn streaming_buffer_emits_windows_across_chunks() {
        let config = StreamingFrameConfig::new(4, 2).unwrap();
        let mut buffer = StreamingFrameBuffer::new(config).unwrap();
        let first = AudioBuffer::F32(vec![0.0, 1.0, 2.0]);
        let second = AudioBuffer::F32(vec![3.0, 4.0, 5.0]);
        let first_frame = AudioFrame::new(ts(), 48_000, 1, &first).unwrap();
        let second_frame = AudioFrame::new(
            Timestamp::new(3, Timebase::new(1, 48_000)),
            48_000,
            1,
            &second,
        )
        .unwrap();

        assert!(buffer.push_frame(&first_frame).unwrap().is_empty());
        let windows = buffer.push_frame(&second_frame).unwrap();

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].samples, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(windows[1].samples, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn streaming_buffer_preserves_window_timestamps() {
        let config = StreamingFrameConfig::new(4, 2).unwrap();
        let mut buffer = StreamingFrameBuffer::new(config).unwrap();
        let samples = AudioBuffer::F32(vec![0.0; 6]);
        let frame = AudioFrame::new(
            Timestamp::new(10, Timebase::new(1, 48_000)),
            48_000,
            1,
            &samples,
        )
        .unwrap();

        let windows = buffer.push_frame(&frame).unwrap();

        assert_eq!(
            windows
                .iter()
                .map(|window| window.timestamp)
                .collect::<Vec<_>>(),
            vec![
                Timestamp::new(10, Timebase::new(1, 48_000)),
                Timestamp::new(12, Timebase::new(1, 48_000)),
            ]
        );
    }

    #[test]
    fn streaming_buffer_requires_stable_format() {
        let config = StreamingFrameConfig::new(4, 2).unwrap();
        let mut buffer = StreamingFrameBuffer::new(config).unwrap();
        let first = AudioBuffer::F32(vec![0.0; 4]);
        let second = AudioBuffer::F32(vec![0.0; 4]);
        let first_frame = AudioFrame::new(ts(), 48_000, 1, &first).unwrap();
        let second_frame = AudioFrame::new(
            Timestamp::new(4, Timebase::new(1, 44_100)),
            44_100,
            1,
            &second,
        )
        .unwrap();

        buffer.push_frame(&first_frame).unwrap();

        assert!(buffer.push_frame(&second_frame).is_err());
    }

    #[test]
    fn streaming_buffer_keeps_retained_samples_bounded() {
        let config = StreamingFrameConfig::new(8, 8)
            .unwrap()
            .max_buffered_samples(8);
        let mut buffer = StreamingFrameBuffer::new(config).unwrap();
        let samples = AudioBuffer::F32(vec![0.0; 32]);
        let frame = AudioFrame::new(ts(), 48_000, 1, &samples).unwrap();

        assert!(buffer.push_frame(&frame).is_ok());
        assert!(buffer.buffered_samples() <= 8);
    }

    #[test]
    fn streaming_buffer_reset_allows_new_format() {
        let config = StreamingFrameConfig::new(4, 2).unwrap();
        let mut buffer = StreamingFrameBuffer::new(config).unwrap();
        let first = AudioBuffer::F32(vec![0.0; 4]);
        let second = AudioBuffer::F32(vec![0.0; 4]);
        buffer
            .push_frame(&AudioFrame::new(ts(), 48_000, 1, &first).unwrap())
            .unwrap();
        buffer.reset();
        assert!(buffer
            .push_frame(
                &AudioFrame::new(
                    Timestamp::new(0, Timebase::new(1, 44_100)),
                    44_100,
                    1,
                    &second
                )
                .unwrap()
            )
            .is_ok());
    }

    proptest! {
        #[test]
        fn generated_interleaved_mono_length_matches_samples_per_channel(
            channels in 1_u16..=8,
            frames in 0_usize..64,
            samples in proptest::collection::vec(-1.0_f32..1.0, 0..512),
        ) {
            let channels = channels as usize;
            let len = frames * channels;
            let mut values = samples;
            values.resize(len, 0.0);
            let mono = interleaved_to_mono(&AudioBuffer::F32(values), channels as u16, ChannelMix::Average).unwrap();
            prop_assert_eq!(mono.len(), frames);
        }

        #[test]
        fn streaming_windows_do_not_depend_on_chunk_partition(
            len in 16_usize..96,
            chunk_size in 1_usize..24,
        ) {
            let samples = (0..len).map(|value| value as f32).collect::<Vec<_>>();
            let config = StreamingFrameConfig::new(8, 4).unwrap();

            let all_buffer = AudioBuffer::F32(samples.clone());
            let all_frame = AudioFrame::new(ts(), 48_000, 1, &all_buffer).unwrap();
            let mut all = StreamingFrameBuffer::new(config).unwrap();
            let expected = all.push_frame(&all_frame).unwrap();

            let mut chunked = StreamingFrameBuffer::new(config).unwrap();
            let mut actual = Vec::new();
            let mut start = 0;
            while start < samples.len() {
                let end = (start + chunk_size).min(samples.len());
                let buffer = frame_at(start as u64, samples[start..end].to_vec());
                let frame = AudioFrame::new(
                    Timestamp::new(start as i64, Timebase::new(1, 48_000)),
                    48_000,
                    1,
                    &buffer,
                )
                .unwrap();
                actual.extend(chunked.push_frame(&frame).unwrap());
                start = end;
            }

            prop_assert_eq!(actual, expected);
        }
    }

    #[test]
    fn scalar_level_helpers_are_empty_safe() {
        assert_approx_eq(rms(&[1.0, -1.0]), 1.0, 1.0e-6);
        assert_eq!(peak(&[]), 0.0);
        assert_eq!(mean_absolute(&[]), 0.0);
    }

    #[test]
    fn audio_format_spec_validates_and_reports_duration() {
        let spec = AudioFormatSpec::new(48_000, 2)
            .unwrap()
            .frame_samples(2_048)
            .unwrap();
        assert_eq!(spec.duration_seconds(4_800).unwrap(), 0.1);
        assert!(AudioFormatSpec::new(0, 2).is_err());
        assert!(AudioFormatSpec::new(48_000, 0).is_err());
        assert!(AudioFormatSpec::new(48_000, 2)
            .unwrap()
            .frame_samples(0)
            .is_err());
    }

    #[test]
    fn sample_and_timestamp_helpers_round_trip() {
        let timestamp = Timestamp::new(2_205, Timebase::new(1, 44_100));
        let sample = timestamp_to_sample(timestamp, 44_100).unwrap();
        assert_eq!(sample, 2_205);
        assert_eq!(sample_to_timestamp(sample, 44_100), timestamp);
        assert_eq!(seconds_to_samples(0.5, 16_000).unwrap(), 8_000);
        assert_eq!(samples_to_seconds(8_000, 16_000).unwrap(), 0.5);
        assert!(seconds_to_samples(-1.0, 16_000).is_err());
    }

    #[test]
    fn streaming_buffer_detects_overlapping_chunks() {
        let config = StreamingFrameConfig::new(4, 2).unwrap();
        let mut buffer = StreamingFrameBuffer::new(config).unwrap();
        let first = AudioBuffer::F32(vec![0.0, 1.0, 2.0, 3.0]);
        let second = AudioBuffer::F32(vec![2.0, 3.0, 4.0, 5.0]);
        let first_frame = AudioFrame::new(ts(), 48_000, 1, &first).unwrap();
        let overlapping = AudioFrame::new(
            Timestamp::new(2, Timebase::new(1, 48_000)),
            48_000,
            1,
            &second,
        )
        .unwrap();
        buffer.push_frame(&first_frame).unwrap();
        assert!(buffer.push_frame(&overlapping).is_err());
    }
}
