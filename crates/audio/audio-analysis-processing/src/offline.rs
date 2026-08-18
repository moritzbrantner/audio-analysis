use audio_analysis_core::{
    peak, rms, seconds_to_samples, AudioClip, AudioFormatSpec, FadeCurve, InterpolationMode,
    SampleRate,
};
use audio_contracts::{DetectError, Result};
use math_signal_core::resample_interleaved;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Fade settings for offline processing.
pub struct FadeSpec {
    /// Fade-in duration.
    pub fade_in_seconds: f64,
    /// Fade-out duration.
    pub fade_out_seconds: f64,
    /// Fade curve.
    pub curve: FadeCurve,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Normalization target.
pub struct NormalizeSpec {
    /// Target peak linear amplitude.
    pub target_peak: Option<f32>,
    /// Target RMS linear amplitude.
    pub target_rms: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
/// Placement for offline audio mixdown.
pub struct AudioClipPlacement {
    /// Clip to place on the output timeline.
    pub clip: AudioClip,
    /// Timeline start in seconds.
    pub start_seconds: f64,
    /// Linear gain applied to this placement.
    pub gain: f32,
}

impl AudioClipPlacement {
    /// Creates a validated placement.
    pub fn new(clip: AudioClip, start_seconds: f64, gain: f32) -> Result<Self> {
        let placement = Self {
            clip,
            start_seconds,
            gain,
        };
        placement.validate()?;
        Ok(placement)
    }

    fn validate(&self) -> Result<()> {
        if !self.start_seconds.is_finite() || self.start_seconds < 0.0 || !self.gain.is_finite() {
            return Err(DetectError::InvalidArgument(
                "audio placement start_seconds must be finite and non-negative, and gain must be finite"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Speed processing mode.
pub enum SpeedMode {
    /// Resampling changes pitch and duration.
    ResampleChangesPitch,
    /// Deterministic overlap-add stretch attempts to preserve pitch.
    PreservePitchOverlapAdd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Pitch shift processing mode.
pub enum PitchShiftMode {
    /// Resampling changes duration.
    ResampleChangesDuration,
    /// Deterministic overlap-add stretch attempts to preserve duration.
    PreserveDurationOverlapAdd,
}

/// Trait for whole-clip transforms.
pub trait OfflineAudioTransform {
    /// Returns transform name.
    fn name(&self) -> &str;
    /// Processes a whole clip.
    fn process_clip(&mut self, clip: AudioClip) -> Result<AudioClip>;
}

#[derive(Default)]
/// Whole-buffer offline audio processor.
pub struct OfflineAudioProcessor {
    transforms: Vec<Box<dyn OfflineAudioTransform>>,
}

impl OfflineAudioProcessor {
    /// Creates an empty processor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a transform.
    pub fn transform<T: OfflineAudioTransform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Adds trim.
    pub fn trim(self, start_seconds: f64, end_seconds: f64) -> Self {
        self.transform(Trim {
            start_seconds,
            end_seconds,
        })
    }

    /// Adds reverse.
    pub fn reverse(self) -> Self {
        self.transform(Reverse)
    }

    /// Adds fade.
    pub fn fade(self, spec: FadeSpec) -> Self {
        self.transform(Fade { spec })
    }

    /// Adds normalize.
    pub fn normalize(self, spec: NormalizeSpec) -> Self {
        self.transform(Normalize { spec })
    }

    /// Adds insert silence.
    pub fn insert_silence(self, at_seconds: f64, duration_seconds: f64) -> Self {
        self.transform(InsertSilence {
            at_seconds,
            duration_seconds,
        })
    }

    /// Adds delete seconds.
    pub fn delete_seconds(self, start_seconds: f64, end_seconds: f64) -> Self {
        self.transform(DeleteSeconds {
            start_seconds,
            end_seconds,
        })
    }

    /// Adds sample-rate conversion.
    pub fn resample(self, output_sample_rate: u32) -> Self {
        self.transform(Resample { output_sample_rate })
    }

    /// Adds speed change.
    pub fn speed(self, factor: f32, mode: SpeedMode) -> Self {
        self.transform(Speed { factor, mode })
    }

    /// Adds pitch shift.
    pub fn pitch_shift(self, semitones: f32, mode: PitchShiftMode) -> Self {
        self.transform(PitchShift { semitones, mode })
    }

    /// Processes the supplied clip through all transforms.
    pub fn process_clip(&mut self, mut clip: AudioClip) -> Result<AudioClip> {
        for transform in &mut self.transforms {
            clip = transform.process_clip(clip)?;
        }
        Ok(clip)
    }
}

/// Inserts silence into a clip at the requested time.
pub fn insert_silence(
    clip: AudioClip,
    at_seconds: f64,
    duration_seconds: f64,
) -> Result<AudioClip> {
    if !duration_seconds.is_finite() || duration_seconds < 0.0 {
        return Err(DetectError::InvalidArgument(
            "silence duration must be finite and non-negative".to_string(),
        ));
    }
    let at_sample = seconds_to_samples(at_seconds, clip.sample_rate)?;
    let total = clip.samples_per_channel() as u64;
    if at_sample > total {
        return Err(DetectError::InvalidArgument(
            "insert position must be inside the clip duration".to_string(),
        ));
    }
    let silence_samples = seconds_to_samples(duration_seconds, clip.sample_rate)? as usize;
    let channels = clip.channels as usize;
    let split = at_sample as usize * channels;
    let mut samples =
        Vec::with_capacity(clip.samples.len() + silence_samples.saturating_mul(channels));
    samples.extend_from_slice(&clip.samples[..split]);
    samples.extend(std::iter::repeat_n(0.0, silence_samples * channels));
    samples.extend_from_slice(&clip.samples[split..]);
    AudioClip::new(clip.sample_rate, clip.channels, samples)
}

/// Deletes a time range from a clip.
pub fn delete_seconds(clip: AudioClip, start_seconds: f64, end_seconds: f64) -> Result<AudioClip> {
    if start_seconds > end_seconds {
        return Err(DetectError::InvalidArgument(
            "delete start_seconds must be less than or equal to end_seconds".to_string(),
        ));
    }
    let start = seconds_to_samples(start_seconds, clip.sample_rate)?;
    let end = seconds_to_samples(end_seconds, clip.sample_rate)?;
    let total = clip.samples_per_channel() as u64;
    if end > total {
        return Err(DetectError::InvalidArgument(
            "delete range must be inside the clip duration".to_string(),
        ));
    }
    let channels = clip.channels as usize;
    let mut samples = Vec::with_capacity(clip.samples.len() - (end - start) as usize * channels);
    samples.extend_from_slice(&clip.samples[..start as usize * channels]);
    samples.extend_from_slice(&clip.samples[end as usize * channels..]);
    AudioClip::new(clip.sample_rate, clip.channels, samples)
}

/// Concatenates clips with a deterministic crossfade between adjacent clips.
pub fn crossfade_concat(
    clips: &[AudioClip],
    duration_seconds: f64,
    curve: FadeCurve,
) -> Result<AudioClip> {
    let first = clips.first().ok_or_else(|| {
        DetectError::InvalidArgument("crossfade concat requires at least one clip".to_string())
    })?;
    if !duration_seconds.is_finite() || duration_seconds < 0.0 {
        return Err(DetectError::InvalidArgument(
            "crossfade duration must be finite and non-negative".to_string(),
        ));
    }
    for clip in clips {
        if clip.sample_rate != first.sample_rate || clip.channels != first.channels {
            return Err(DetectError::InvalidArgument(
                "crossfade concat requires matching sample rates and channel counts".to_string(),
            ));
        }
    }
    let fade_frames = seconds_to_samples(duration_seconds, first.sample_rate)? as usize;
    if fade_frames == 0 {
        return AudioClip::concat(clips, audio_analysis_core::ConcatPolicy::RequireSameFormat);
    }
    let channels = first.channels as usize;
    let mut output = first.samples.clone();
    for clip in &clips[1..] {
        let output_frames = output.len() / channels;
        let clip_frames = clip.samples_per_channel();
        if fade_frames > output_frames || fade_frames > clip_frames {
            return Err(DetectError::InvalidArgument(
                "crossfade duration must fit inside adjacent clips".to_string(),
            ));
        }
        let keep_len = (output_frames - fade_frames) * channels;
        let mut next = Vec::with_capacity(keep_len + clip.samples.len());
        next.extend_from_slice(&output[..keep_len]);
        for frame in 0..fade_frames {
            let position = if fade_frames == 1 {
                1.0
            } else {
                frame as f32 / (fade_frames - 1) as f32
            };
            let fade_in = fade_gain(position, curve);
            let fade_out = fade_gain(1.0 - position, curve);
            for channel in 0..channels {
                let left = output[keep_len + frame * channels + channel];
                let right = clip.samples[frame * channels + channel];
                next.push(left * fade_out + right * fade_in);
            }
        }
        next.extend_from_slice(&clip.samples[fade_frames * channels..]);
        output = next;
    }
    AudioClip::new(first.sample_rate, first.channels, output)
}

/// Mixes placed clips onto a single output timeline.
pub fn mixdown_placements(
    placements: &[AudioClipPlacement],
    sample_rate: u32,
    channels: u16,
) -> Result<AudioClip> {
    AudioFormatSpec::new(sample_rate, channels)?;
    if placements.is_empty() {
        return AudioClip::new(sample_rate, channels, Vec::new());
    }
    let mut output_frames = 0usize;
    let mut starts = Vec::with_capacity(placements.len());
    for placement in placements {
        placement.validate()?;
        if placement.clip.sample_rate != sample_rate || placement.clip.channels != channels {
            return Err(DetectError::InvalidArgument(
                "mixdown placements must match the requested sample rate and channel count"
                    .to_string(),
            ));
        }
        let start = seconds_to_samples(placement.start_seconds, sample_rate)? as usize;
        output_frames = output_frames.max(start + placement.clip.samples_per_channel());
        starts.push(start);
    }
    let channels_usize = channels as usize;
    let mut output = vec![0.0; output_frames * channels_usize];
    for (placement, start_frame) in placements.iter().zip(starts) {
        let offset = start_frame * channels_usize;
        for (index, sample) in placement.clip.samples.iter().enumerate() {
            output[offset + index] += sample * placement.gain;
        }
    }
    AudioClip::new(sample_rate, channels, output)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Trim {
    start_seconds: f64,
    end_seconds: f64,
}

impl OfflineAudioTransform for Trim {
    fn name(&self) -> &str {
        "audio_trim"
    }

    fn process_clip(&mut self, clip: AudioClip) -> Result<AudioClip> {
        clip.slice_seconds(self.start_seconds, self.end_seconds)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Reverse;

impl OfflineAudioTransform for Reverse {
    fn name(&self) -> &str {
        "audio_reverse"
    }

    fn process_clip(&mut self, clip: AudioClip) -> Result<AudioClip> {
        let channels = clip.channels as usize;
        let mut samples = Vec::with_capacity(clip.samples.len());
        for frame in clip.samples.chunks_exact(channels).rev() {
            samples.extend_from_slice(frame);
        }
        AudioClip::new(clip.sample_rate, clip.channels, samples)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Fade {
    spec: FadeSpec,
}

impl OfflineAudioTransform for Fade {
    fn name(&self) -> &str {
        "audio_fade"
    }

    fn process_clip(&mut self, mut clip: AudioClip) -> Result<AudioClip> {
        if !self.spec.fade_in_seconds.is_finite()
            || self.spec.fade_in_seconds < 0.0
            || !self.spec.fade_out_seconds.is_finite()
            || self.spec.fade_out_seconds < 0.0
        {
            return Err(DetectError::InvalidArgument(
                "fade durations must be finite and non-negative".to_string(),
            ));
        }
        let channels = clip.channels as usize;
        let frames = clip.samples_per_channel();
        let fade_in = (self.spec.fade_in_seconds * clip.sample_rate as f64).round() as usize;
        let fade_out = (self.spec.fade_out_seconds * clip.sample_rate as f64).round() as usize;
        for frame_index in 0..frames {
            let mut gain = 1.0;
            if fade_in > 0 && frame_index < fade_in {
                gain *= fade_gain(frame_index as f32 / fade_in as f32, self.spec.curve);
            }
            if fade_out > 0 && frame_index >= frames.saturating_sub(fade_out) {
                let remaining = frames.saturating_sub(frame_index + 1);
                gain *= fade_gain(remaining as f32 / fade_out as f32, self.spec.curve);
            }
            for channel in 0..channels {
                clip.samples[frame_index * channels + channel] *= gain;
            }
        }
        AudioClip::new(clip.sample_rate, clip.channels, clip.samples)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Normalize {
    spec: NormalizeSpec,
}

impl OfflineAudioTransform for Normalize {
    fn name(&self) -> &str {
        "audio_normalize"
    }

    fn process_clip(&mut self, mut clip: AudioClip) -> Result<AudioClip> {
        let gain = if let Some(target_peak) = self.spec.target_peak {
            if !target_peak.is_finite() || target_peak < 0.0 {
                return Err(DetectError::InvalidArgument(
                    "target_peak must be finite and non-negative".to_string(),
                ));
            }
            let current = peak(&clip.samples);
            if current == 0.0 {
                1.0
            } else {
                target_peak / current
            }
        } else if let Some(target_rms) = self.spec.target_rms {
            if !target_rms.is_finite() || target_rms < 0.0 {
                return Err(DetectError::InvalidArgument(
                    "target_rms must be finite and non-negative".to_string(),
                ));
            }
            let current = rms(&clip.samples);
            if current == 0.0 {
                1.0
            } else {
                target_rms / current
            }
        } else {
            1.0
        };
        for sample in &mut clip.samples {
            *sample *= gain;
        }
        AudioClip::new(clip.sample_rate, clip.channels, clip.samples)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct InsertSilence {
    at_seconds: f64,
    duration_seconds: f64,
}

impl OfflineAudioTransform for InsertSilence {
    fn name(&self) -> &str {
        "audio_insert_silence"
    }

    fn process_clip(&mut self, clip: AudioClip) -> Result<AudioClip> {
        insert_silence(clip, self.at_seconds, self.duration_seconds)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DeleteSeconds {
    start_seconds: f64,
    end_seconds: f64,
}

impl OfflineAudioTransform for DeleteSeconds {
    fn name(&self) -> &str {
        "audio_delete_seconds"
    }

    fn process_clip(&mut self, clip: AudioClip) -> Result<AudioClip> {
        delete_seconds(clip, self.start_seconds, self.end_seconds)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Resample {
    output_sample_rate: u32,
}

impl OfflineAudioTransform for Resample {
    fn name(&self) -> &str {
        "audio_resample"
    }

    fn process_clip(&mut self, clip: AudioClip) -> Result<AudioClip> {
        let samples = resample_interleaved(
            &clip.samples,
            clip.channels,
            SampleRate::new(clip.sample_rate)?,
            SampleRate::new(self.output_sample_rate)?,
            InterpolationMode::Linear,
        )?;
        AudioClip::new(self.output_sample_rate, clip.channels, samples)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Speed {
    factor: f32,
    mode: SpeedMode,
}

impl OfflineAudioTransform for Speed {
    fn name(&self) -> &str {
        "audio_speed"
    }

    fn process_clip(&mut self, clip: AudioClip) -> Result<AudioClip> {
        validate_factor(self.factor, "speed factor")?;
        let samples = match self.mode {
            SpeedMode::ResampleChangesPitch => resample_duration(&clip, self.factor)?,
            SpeedMode::PreservePitchOverlapAdd => overlap_add_time_scale(&clip, 1.0 / self.factor)?,
        };
        AudioClip::new(clip.sample_rate, clip.channels, samples)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PitchShift {
    semitones: f32,
    mode: PitchShiftMode,
}

impl OfflineAudioTransform for PitchShift {
    fn name(&self) -> &str {
        "audio_pitch_shift"
    }

    fn process_clip(&mut self, clip: AudioClip) -> Result<AudioClip> {
        if !self.semitones.is_finite() {
            return Err(DetectError::InvalidArgument(
                "pitch semitones must be finite".to_string(),
            ));
        }
        let factor = 2.0_f32.powf(self.semitones / 12.0);
        let shifted = resample_duration(&clip, factor)?;
        match self.mode {
            PitchShiftMode::ResampleChangesDuration => {
                AudioClip::new(clip.sample_rate, clip.channels, shifted)
            }
            PitchShiftMode::PreserveDurationOverlapAdd => {
                let shifted_clip = AudioClip::new(clip.sample_rate, clip.channels, shifted)?;
                let restored = overlap_add_to_frames(&shifted_clip, clip.samples_per_channel())?;
                AudioClip::new(clip.sample_rate, clip.channels, restored)
            }
        }
    }
}

fn validate_factor(factor: f32, name: &str) -> Result<()> {
    if factor.is_finite() && factor > 0.0 {
        Ok(())
    } else {
        Err(DetectError::InvalidArgument(format!(
            "{name} must be finite and greater than zero"
        )))
    }
}

fn resample_duration(clip: &AudioClip, factor: f32) -> Result<Vec<f32>> {
    validate_factor(factor, "resample duration factor")?;
    let channels = clip.channels as usize;
    let input_frames = clip.samples_per_channel();
    if input_frames == 0 {
        return Ok(Vec::new());
    }
    let output_frames = ((input_frames as f32) / factor).round().max(1.0) as usize;
    let mut output = Vec::with_capacity(output_frames * channels);
    for output_frame in 0..output_frames {
        let source = output_frame as f32 * factor;
        for channel in 0..channels {
            output.push(interpolate_channel(
                &clip.samples,
                channels,
                channel,
                source,
            ));
        }
    }
    Ok(output)
}

fn overlap_add_to_frames(clip: &AudioClip, target_frames: usize) -> Result<Vec<f32>> {
    if clip.samples_per_channel() == target_frames {
        return Ok(clip.samples.clone());
    }
    let scale = target_frames as f32 / clip.samples_per_channel().max(1) as f32;
    let stretched = overlap_add_time_scale(clip, scale)?;
    let channels = clip.channels as usize;
    let target_len = target_frames * channels;
    let mut output = stretched;
    output.resize(target_len, 0.0);
    output.truncate(target_len);
    Ok(output)
}

fn overlap_add_time_scale(clip: &AudioClip, scale: f32) -> Result<Vec<f32>> {
    validate_factor(scale, "time scale factor")?;
    let channels = clip.channels as usize;
    let input_frames = clip.samples_per_channel();
    if input_frames == 0 {
        return Ok(Vec::new());
    }
    let output_frames = ((input_frames as f32) * scale).round().max(1.0) as usize;
    let frame_size = (clip.sample_rate as usize / 20).clamp(64, 2048);
    let analysis_hop = (frame_size / 4).max(1);
    let synthesis_hop = ((analysis_hop as f32) * scale).round().max(1.0) as usize;
    if input_frames < frame_size {
        let mut output = Vec::with_capacity(output_frames * channels);
        for frame in 0..output_frames {
            let source = frame as f32 / scale;
            for channel in 0..channels {
                output.push(interpolate_channel(
                    &clip.samples,
                    channels,
                    channel,
                    source,
                ));
            }
        }
        return Ok(output);
    }

    let mut output = vec![0.0; (output_frames + frame_size) * channels];
    let mut weights = vec![0.0; output_frames + frame_size];
    let window = hann_window(frame_size);
    let mut analysis = 0usize;
    let mut synthesis = 0usize;
    while analysis + frame_size <= input_frames {
        for (index, weight) in window.iter().copied().enumerate() {
            let out_frame = synthesis + index;
            weights[out_frame] += weight;
            for channel in 0..channels {
                output[out_frame * channels + channel] +=
                    clip.samples[(analysis + index) * channels + channel] * weight;
            }
        }
        analysis += analysis_hop;
        synthesis += synthesis_hop;
    }
    for frame in 0..output_frames {
        let weight = weights[frame];
        if weight > 1.0e-6 {
            for channel in 0..channels {
                output[frame * channels + channel] /= weight;
            }
        }
    }
    output.truncate(output_frames * channels);
    Ok(output)
}

fn interpolate_channel(samples: &[f32], channels: usize, channel: usize, position: f32) -> f32 {
    let frames = samples.len() / channels;
    if frames == 0 {
        return 0.0;
    }
    let clamped = position.clamp(0.0, (frames - 1) as f32);
    let left = clamped.floor() as usize;
    let right = clamped.ceil() as usize;
    if left == right {
        samples[left * channels + channel]
    } else {
        let fraction = clamped - left as f32;
        let left_sample = samples[left * channels + channel];
        let right_sample = samples[right * channels + channel];
        left_sample * (1.0 - fraction) + right_sample * fraction
    }
}

fn fade_gain(position: f32, curve: FadeCurve) -> f32 {
    let position = position.clamp(0.0, 1.0);
    match curve {
        FadeCurve::Linear => position,
        FadeCurve::EqualPower => (position * std::f32::consts::FRAC_PI_2).sin(),
        FadeCurve::Exponential => position * position,
    }
}

fn hann_window(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    (0..len)
        .map(|index| {
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / (len - 1) as f32).cos()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(samples: Vec<f32>) -> AudioClip {
        AudioClip::new(8, 1, samples).unwrap()
    }

    #[test]
    fn reverse_fade_and_normalize_work() {
        let mut processor = OfflineAudioProcessor::new()
            .reverse()
            .fade(FadeSpec {
                fade_in_seconds: 0.125,
                fade_out_seconds: 0.125,
                curve: FadeCurve::Linear,
            })
            .normalize(NormalizeSpec {
                target_peak: Some(1.0),
                target_rms: None,
            });
        let output = processor
            .process_clip(clip(vec![0.25, 0.5, 0.75, 1.0]))
            .unwrap();
        assert_eq!(output.samples_per_channel(), 4);
        assert!((peak(&output.samples) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn speed_changes_duration() {
        let input = clip(vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]);
        let mut faster = OfflineAudioProcessor::new().speed(2.0, SpeedMode::ResampleChangesPitch);
        let mut slower = OfflineAudioProcessor::new().speed(0.5, SpeedMode::ResampleChangesPitch);
        assert!(faster.process_clip(input.clone()).unwrap().samples.len() < input.samples.len());
        assert!(slower.process_clip(input).unwrap().samples.len() > 8);
    }

    #[test]
    fn pitch_shift_preserve_duration_keeps_length() {
        let input = clip((0..128).map(|index| (index as f32 * 0.1).sin()).collect());
        let mut processor = OfflineAudioProcessor::new()
            .pitch_shift(12.0, PitchShiftMode::PreserveDurationOverlapAdd);
        let output = processor.process_clip(input.clone()).unwrap();
        assert_eq!(output.samples.len(), input.samples.len());
        assert!(output.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn insert_delete_crossfade_and_mixdown_work() {
        let input = clip(vec![1.0, 2.0, 3.0, 4.0]);
        let inserted = insert_silence(input.clone(), 0.25, 0.25).unwrap();
        assert_eq!(inserted.samples, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);

        let deleted = delete_seconds(inserted, 0.25, 0.5).unwrap();
        assert_eq!(deleted.samples, input.samples);

        let a = clip(vec![1.0, 1.0, 1.0, 1.0]);
        let b = clip(vec![0.0, 0.0, 0.0, 0.0]);
        let crossfaded =
            crossfade_concat(&[a.clone(), b.clone()], 0.25, FadeCurve::Linear).unwrap();
        assert_eq!(crossfaded.samples.len(), 6);

        let mixed = mixdown_placements(
            &[
                AudioClipPlacement::new(a, 0.0, 1.0).unwrap(),
                AudioClipPlacement::new(b, 0.25, 0.5).unwrap(),
            ],
            8,
            1,
        )
        .unwrap();
        assert_eq!(mixed.samples_per_channel(), 6);
    }
}
