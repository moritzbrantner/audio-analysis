use crate::{
    normalized_samples, seconds_to_samples, AudioFormatSpec, InterpolationMode, SampleRate,
};
use audio_contracts::{AudioBuffer, DetectError, OwnedAudioFrame, Result, Timestamp};
use math_signal_core::resample_interleaved;

#[derive(Debug, Clone, PartialEq)]
/// Owned interleaved f32 clip for whole-buffer editing.
pub struct AudioClip {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u16,
    /// Interleaved f32 samples.
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Policy for concatenating clips with different formats.
pub enum ConcatPolicy {
    /// Require sample rate and channel count to match exactly.
    RequireSameFormat,
    /// Resample all clips to the first clip's sample rate. Channel counts must match.
    ResampleToFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Policy for mixing clips with different lengths.
pub enum MixPolicy {
    /// Require identical format and length.
    RequireSameFormat,
    /// Pad shorter clips with silence.
    PadToLongest,
    /// Truncate longer clips to the shortest input.
    TruncateToShortest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Fade curve shape.
pub enum FadeCurve {
    /// Linear gain ramp.
    Linear,
    /// Equal-power sine/cosine ramp.
    EqualPower,
    /// Smooth exponential-style ramp.
    Exponential,
}

impl AudioClip {
    /// Creates a validated clip.
    pub fn new(sample_rate: u32, channels: u16, samples: Vec<f32>) -> Result<Self> {
        AudioFormatSpec::new(sample_rate, channels)?;
        if !samples.len().is_multiple_of(channels as usize) {
            return Err(DetectError::InvalidArgument(format!(
                "audio clip sample length {} is not divisible by channel count {channels}",
                samples.len()
            )));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(DetectError::InvalidArgument(
                "audio clip samples must be finite".to_string(),
            ));
        }
        Ok(Self {
            sample_rate,
            channels,
            samples,
        })
    }

    /// Builds a single clip from ordered frames.
    pub fn from_frames(frames: &[OwnedAudioFrame]) -> Result<Self> {
        if frames.is_empty() {
            return Err(DetectError::InvalidArgument(
                "audio clip requires at least one frame".to_string(),
            ));
        }
        let sample_rate = frames[0].sample_rate;
        let channels = frames[0].channels;
        let mut samples = Vec::new();
        for frame in frames {
            if frame.sample_rate != sample_rate || frame.channels != channels {
                return Err(DetectError::InvalidArgument(
                    "all frames must share sample rate and channel count".to_string(),
                ));
            }
            let normalized = normalized_samples(&frame.data);
            if !normalized.len().is_multiple_of(channels as usize) {
                return Err(DetectError::InvalidArgument(format!(
                    "audio frame sample length {} is not divisible by channel count {channels}",
                    normalized.len()
                )));
            }
            samples.extend(normalized);
        }
        Self::new(sample_rate, channels, samples)
    }

    /// Converts the clip to one owned frame at the supplied timestamp.
    pub fn to_frame(&self, timestamp: Timestamp) -> Result<OwnedAudioFrame> {
        OwnedAudioFrame::new(
            timestamp,
            self.sample_rate,
            self.channels,
            AudioBuffer::F32(self.samples.clone()),
        )
    }

    /// Returns the number of samples per channel.
    pub fn samples_per_channel(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Returns clip duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples_per_channel() as f64 / self.sample_rate as f64
    }

    /// Slices by per-channel sample indices.
    pub fn slice_samples(&self, start_sample: u64, end_sample: u64) -> Result<Self> {
        if start_sample > end_sample {
            return Err(DetectError::InvalidArgument(
                "slice start_sample must be less than or equal to end_sample".to_string(),
            ));
        }
        let total = self.samples_per_channel() as u64;
        if end_sample > total {
            return Err(DetectError::InvalidArgument(format!(
                "slice end_sample {end_sample} exceeds clip length {total}"
            )));
        }
        let channels = self.channels as usize;
        let start = start_sample as usize * channels;
        let end = end_sample as usize * channels;
        Self::new(
            self.sample_rate,
            self.channels,
            self.samples[start..end].to_vec(),
        )
    }

    /// Slices by seconds.
    pub fn slice_seconds(&self, start_seconds: f64, end_seconds: f64) -> Result<Self> {
        if start_seconds > end_seconds {
            return Err(DetectError::InvalidArgument(
                "slice start_seconds must be less than or equal to end_seconds".to_string(),
            ));
        }
        let start = seconds_to_samples(start_seconds, self.sample_rate)?;
        let end = seconds_to_samples(end_seconds, self.sample_rate)?;
        self.slice_samples(start, end)
    }

    /// Splits a clip at ordered boundary times.
    pub fn split_at_seconds(&self, boundaries: &[f64]) -> Result<Vec<Self>> {
        let mut sample_boundaries = Vec::with_capacity(boundaries.len() + 2);
        sample_boundaries.push(0);
        let total = self.samples_per_channel() as u64;
        let mut previous = 0;
        for boundary in boundaries {
            let sample = seconds_to_samples(*boundary, self.sample_rate)?;
            if sample < previous || sample > total {
                return Err(DetectError::InvalidArgument(
                    "split boundaries must be ordered and inside the clip duration".to_string(),
                ));
            }
            sample_boundaries.push(sample);
            previous = sample;
        }
        sample_boundaries.push(total);
        sample_boundaries
            .windows(2)
            .map(|range| self.slice_samples(range[0], range[1]))
            .collect()
    }

    /// Concatenates clips.
    pub fn concat(clips: &[Self], policy: ConcatPolicy) -> Result<Self> {
        let first = clips.first().ok_or_else(|| {
            DetectError::InvalidArgument("concat requires at least one clip".to_string())
        })?;
        let mut samples = Vec::new();
        for clip in clips {
            if clip.channels != first.channels {
                return Err(DetectError::InvalidArgument(
                    "concat requires matching channel counts".to_string(),
                ));
            }
            match policy {
                ConcatPolicy::RequireSameFormat if clip.sample_rate != first.sample_rate => {
                    return Err(DetectError::InvalidArgument(
                        "concat requires matching sample rates".to_string(),
                    ));
                }
                ConcatPolicy::RequireSameFormat => samples.extend_from_slice(&clip.samples),
                ConcatPolicy::ResampleToFirst => {
                    let converted = if clip.sample_rate == first.sample_rate {
                        clip.samples.clone()
                    } else {
                        resample_interleaved(
                            &clip.samples,
                            clip.channels,
                            SampleRate::new(clip.sample_rate)?,
                            SampleRate::new(first.sample_rate)?,
                            InterpolationMode::Linear,
                        )?
                    };
                    samples.extend(converted);
                }
            }
        }
        Self::new(first.sample_rate, first.channels, samples)
    }

    /// Mixes clips by summing matching interleaved samples.
    pub fn mix(clips: &[Self], policy: MixPolicy) -> Result<Self> {
        let first = clips.first().ok_or_else(|| {
            DetectError::InvalidArgument("mix requires at least one clip".to_string())
        })?;
        for clip in clips {
            if clip.sample_rate != first.sample_rate || clip.channels != first.channels {
                return Err(DetectError::InvalidArgument(
                    "mix requires matching sample rates and channel counts".to_string(),
                ));
            }
        }
        let target_len = match policy {
            MixPolicy::RequireSameFormat => {
                let len = first.samples.len();
                if clips.iter().any(|clip| clip.samples.len() != len) {
                    return Err(DetectError::InvalidArgument(
                        "mix RequireSameFormat requires identical sample lengths".to_string(),
                    ));
                }
                len
            }
            MixPolicy::PadToLongest => clips
                .iter()
                .map(|clip| clip.samples.len())
                .max()
                .unwrap_or(0),
            MixPolicy::TruncateToShortest => clips
                .iter()
                .map(|clip| clip.samples.len())
                .min()
                .unwrap_or(0),
        };
        let mut mixed = vec![0.0; target_len];
        for clip in clips {
            for (index, sample) in clip.samples.iter().take(target_len).enumerate() {
                mixed[index] += *sample;
            }
        }
        Self::new(first.sample_rate, first.channels, mixed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samples_to_seconds;
    use audio_contracts::{AudioBuffer, Timebase, Timestamp};

    fn clip(samples: Vec<f32>) -> AudioClip {
        AudioClip::new(4, 2, samples).unwrap()
    }

    #[test]
    fn validates_audio_clip_format() {
        assert!(AudioClip::new(0, 1, vec![0.0]).is_err());
        assert!(AudioClip::new(48_000, 0, vec![0.0]).is_err());
        assert!(AudioClip::new(48_000, 2, vec![0.0]).is_err());
        assert!(AudioClip::new(48_000, 1, vec![f32::NAN]).is_err());
    }

    #[test]
    fn converts_frames_and_slices() {
        let frame = OwnedAudioFrame::new(
            Timestamp::new(0, Timebase::new(1, 4)),
            4,
            2,
            AudioBuffer::F32(vec![0.0, 0.1, 0.2, 0.3]),
        )
        .unwrap();
        let clip = AudioClip::from_frames(&[frame]).unwrap();
        assert_eq!(clip.samples_per_channel(), 2);
        assert_eq!(samples_to_seconds(2, 4).unwrap(), clip.duration_seconds());
        assert_eq!(clip.slice_samples(1, 2).unwrap().samples, vec![0.2, 0.3]);
        assert!(clip.slice_seconds(0.75, 0.25).is_err());
    }

    #[test]
    fn split_and_concat_round_trip() {
        let input = clip(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
        let parts = input.split_at_seconds(&[0.5]).unwrap();
        assert_eq!(parts.len(), 2);
        let output = AudioClip::concat(&parts, ConcatPolicy::RequireSameFormat).unwrap();
        assert_eq!(output.samples, input.samples);
    }

    #[test]
    fn concat_and_mix_validate_policies() {
        let a = AudioClip::new(4, 1, vec![1.0, 2.0]).unwrap();
        let b = AudioClip::new(8, 1, vec![3.0, 4.0]).unwrap();
        assert!(
            AudioClip::concat(&[a.clone(), b.clone()], ConcatPolicy::RequireSameFormat).is_err()
        );
        assert!(AudioClip::concat(&[a.clone(), b], ConcatPolicy::ResampleToFirst).is_ok());

        let c = AudioClip::new(4, 1, vec![1.0]).unwrap();
        assert!(AudioClip::mix(&[a.clone(), c.clone()], MixPolicy::RequireSameFormat).is_err());
        assert_eq!(
            AudioClip::mix(&[a.clone(), c.clone()], MixPolicy::PadToLongest)
                .unwrap()
                .samples,
            vec![2.0, 2.0]
        );
        assert_eq!(
            AudioClip::mix(&[a, c], MixPolicy::TruncateToShortest)
                .unwrap()
                .samples,
            vec![2.0]
        );
    }
}
