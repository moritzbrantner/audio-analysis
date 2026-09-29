#![doc = include_str!("../README.md")]

pub mod surface;
use audio_analysis_recognition::{AudioRuntime, AudioRuntimeSelection};
use audio_contracts::{
    AnalysisEvent, AudioBuffer, DetectError, OwnedAudioFrame, Result, Timebase, Timestamp,
};
use data_inversion_core::{Generated, InformationFidelity, InversionMethod, InversionTrace};

/// Request for prompt-based external audio generation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioGenerationRequest {
    /// Text prompt.
    pub prompt: String,
    /// Requested duration in seconds.
    #[serde(default = "default_generation_duration")]
    pub duration_seconds: f32,
    /// Runtime selection.
    #[serde(default)]
    pub model: AudioRuntimeSelection,
}

/// Response for prompt-based external audio generation planning.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioGenerationResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Selected model id.
    pub model_id: String,
    /// Runtime used.
    pub runtime: AudioRuntime,
    /// Prompt accepted by the model layer.
    pub prompt: String,
    /// Planned duration in seconds.
    pub duration_seconds: f32,
    /// Human-readable generation plan.
    pub plan: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing waveform.
pub enum Waveform {
    /// The sine variant.
    Sine,
    /// The square variant.
    Square,
    /// The saw variant.
    Saw,
    /// The triangle variant.
    Triangle,
    /// The pulse variant.
    Pulse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Variants describing clipping policy.
pub enum ClippingPolicy {
    /// The clamp variant.
    Clamp,
    /// The normalize variant.
    Normalize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for amplitude envelope.
pub struct AmplitudeEnvelope {
    /// The gain value.
    pub gain: f32,
    /// The attack seconds value.
    pub attack_seconds: f32,
    /// The release seconds value.
    pub release_seconds: f32,
}

impl Default for AmplitudeEnvelope {
    fn default() -> Self {
        Self {
            gain: 0.8,
            attack_seconds: 0.005,
            release_seconds: 0.02,
        }
    }
}

impl AmplitudeEnvelope {
    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if !self.gain.is_finite() || self.gain < 0.0 {
            return Err(invalid_argument(
                "envelope gain must be finite and non-negative",
            ));
        }
        if !self.attack_seconds.is_finite()
            || !self.release_seconds.is_finite()
            || self.attack_seconds < 0.0
            || self.release_seconds < 0.0
        {
            return Err(invalid_argument(
                "envelope attack and release must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for tone spec.
pub struct ToneSpec {
    /// The frequency hz value.
    pub frequency_hz: f32,
    /// Duration in seconds.
    pub duration_seconds: f32,
    /// The amplitude value.
    pub amplitude: f32,
    /// The waveform value.
    pub waveform: Waveform,
}

impl ToneSpec {
    /// Generates a sine wave.
    pub fn sine(frequency_hz: f32, duration_seconds: f32) -> Self {
        Self {
            frequency_hz,
            duration_seconds,
            amplitude: 0.5,
            waveform: Waveform::Sine,
        }
    }

    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(invalid_argument(
                "tone frequency_hz must be finite and greater than zero",
            ));
        }
        if !self.duration_seconds.is_finite() || self.duration_seconds <= 0.0 {
            return Err(invalid_argument(
                "tone duration_seconds must be finite and greater than zero",
            ));
        }
        if !self.amplitude.is_finite() || self.amplitude < 0.0 {
            return Err(invalid_argument(
                "tone amplitude must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for tone segment.
pub struct ToneSegment {
    /// The start seconds value.
    pub start_seconds: f32,
    /// The tone value.
    pub tone: ToneSpec,
}

impl ToneSegment {
    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if !self.start_seconds.is_finite() || self.start_seconds < 0.0 {
            return Err(invalid_argument(
                "tone segment start_seconds must be finite and non-negative",
            ));
        }
        self.tone.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for audio synthesis config.
pub struct AudioSynthesisConfig {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
    /// The start timestamp value.
    pub start_timestamp: Timestamp,
    /// The envelope value.
    pub envelope: AmplitudeEnvelope,
    /// The clipping policy value.
    pub clipping_policy: ClippingPolicy,
}

impl AudioSynthesisConfig {
    /// Creates a new value.
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        if sample_rate == 0 || channels == 0 {
            return Err(DetectError::InvalidAudioFormat {
                sample_rate,
                channels,
            });
        }
        if sample_rate > i32::MAX as u32 {
            return Err(invalid_argument("sample_rate is too large for a timebase"));
        }
        Ok(Self {
            sample_rate,
            channels,
            start_timestamp: Timestamp::new(0, Timebase::new(1, sample_rate as i32)),
            envelope: AmplitudeEnvelope::default(),
            clipping_policy: ClippingPolicy::Clamp,
        })
    }

    /// Returns envelope.
    pub fn envelope(mut self, envelope: AmplitudeEnvelope) -> Result<Self> {
        envelope.validate()?;
        self.envelope = envelope;
        Ok(self)
    }

    /// Returns clipping policy.
    pub fn clipping_policy(mut self, clipping_policy: ClippingPolicy) -> Self {
        self.clipping_policy = clipping_policy;
        self
    }
}

impl Default for AudioSynthesisConfig {
    fn default() -> Self {
        Self::new(48_000, 1).expect("default audio synthesis config is valid")
    }
}

/// Returns synthesize tone.
pub fn synthesize_tone(
    tone: ToneSpec,
    config: AudioSynthesisConfig,
) -> Result<Generated<OwnedAudioFrame>> {
    synthesize_timeline(
        &[ToneSegment {
            start_seconds: 0.0,
            tone,
        }],
        config,
    )
}

/// Returns synthesize timeline.
pub fn synthesize_timeline(
    segments: &[ToneSegment],
    config: AudioSynthesisConfig,
) -> Result<Generated<OwnedAudioFrame>> {
    if segments.is_empty() {
        return Err(invalid_argument("tone timeline must not be empty"));
    }
    config.envelope.validate()?;
    let channels = config.channels as usize;
    let mut end_sample = 0_usize;
    for segment in segments {
        segment.validate()?;
        let start = seconds_to_samples(segment.start_seconds, config.sample_rate)?;
        let len = seconds_to_samples(segment.tone.duration_seconds, config.sample_rate)?.max(1);
        end_sample = end_sample.max(start + len);
    }

    let mut mono = vec![0.0_f32; end_sample];
    for segment in segments {
        let start = seconds_to_samples(segment.start_seconds, config.sample_rate)?;
        let len = seconds_to_samples(segment.tone.duration_seconds, config.sample_rate)?.max(1);
        for offset in 0..len {
            let t = offset as f32 / config.sample_rate as f32;
            let env = envelope_gain(offset, len, config.sample_rate, config.envelope);
            let sample = wave_sample(segment.tone.waveform, segment.tone.frequency_hz, t)
                * segment.tone.amplitude
                * env;
            mono[start + offset] += sample;
        }
    }

    apply_clipping_policy(&mut mono, config.clipping_policy);

    let mut interleaved = Vec::with_capacity(mono.len() * channels);
    for sample in mono {
        for _ in 0..channels {
            interleaved.push(sample);
        }
    }

    let frame = OwnedAudioFrame::new(
        config.start_timestamp,
        config.sample_rate,
        config.channels,
        AudioBuffer::F32(interleaved),
    )?;
    let trace = InversionTrace::new(
        "tone_timeline",
        "owned_audio_frame",
        InformationFidelity::Interpolated,
    )
    .assumption("phase starts at zero for every synthesized segment")
    .note(
        "waveform",
        InversionMethod::Template,
        "samples are generated from analytic waveform templates",
    )
    .note(
        "samples",
        InversionMethod::Interpolated,
        "continuous tones are sampled at the configured sample rate",
    );
    Ok(Generated::new(frame, trace))
}

/// Returns synthesize click.
pub fn synthesize_click(
    start_seconds: f32,
    duration_seconds: f32,
    amplitude: f32,
    config: AudioSynthesisConfig,
) -> Result<Generated<OwnedAudioFrame>> {
    synthesize_timeline(
        &[ToneSegment {
            start_seconds,
            tone: ToneSpec {
                frequency_hz: 1_800.0,
                duration_seconds,
                amplitude,
                waveform: Waveform::Pulse,
            },
        }],
        config,
    )
}

/// Synthesizes a fixed-length click track from beat positions in seconds.
pub fn synthesize_click_track(
    beat_seconds: &[f32],
    duration_seconds: f32,
    click_frequency_hz: f32,
    click_duration_seconds: f32,
    amplitude: f32,
    config: AudioSynthesisConfig,
) -> Result<Generated<OwnedAudioFrame>> {
    if beat_seconds.is_empty() {
        return Err(invalid_argument("click track requires at least one beat"));
    }
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err(invalid_argument(
            "click track duration_seconds must be finite and greater than zero",
        ));
    }
    let click = ToneSpec {
        frequency_hz: click_frequency_hz,
        duration_seconds: click_duration_seconds,
        amplitude,
        waveform: Waveform::Pulse,
    };
    click.validate()?;
    let total_samples = seconds_to_samples(duration_seconds, config.sample_rate)?.max(1);
    let click_samples = seconds_to_samples(click_duration_seconds, config.sample_rate)?.max(1);
    let mut mono = vec![0.0_f32; total_samples];
    let mut previous = None;
    for beat in beat_seconds {
        if !beat.is_finite() || *beat < 0.0 {
            return Err(invalid_argument(
                "click track beats must be finite and non-negative",
            ));
        }
        if let Some(previous) = previous {
            if *beat < previous {
                return Err(invalid_argument("click track beats must be sorted by time"));
            }
        }
        previous = Some(*beat);
        let start = seconds_to_samples(*beat, config.sample_rate)?;
        if start >= total_samples {
            continue;
        }
        let end = (start + click_samples).min(total_samples);
        for (offset, sample) in mono[start..end].iter_mut().enumerate() {
            let t = offset as f32 / config.sample_rate as f32;
            *sample += wave_sample(Waveform::Pulse, click_frequency_hz, t) * amplitude;
        }
    }
    apply_clipping_policy(&mut mono, config.clipping_policy);
    let mut interleaved = Vec::with_capacity(mono.len() * usize::from(config.channels));
    for sample in mono {
        for _ in 0..config.channels {
            interleaved.push(sample);
        }
    }
    let frame = OwnedAudioFrame::new(
        config.start_timestamp,
        config.sample_rate,
        config.channels,
        AudioBuffer::F32(interleaved),
    )?;
    Ok(Generated::new(
        frame,
        InversionTrace::new(
            "click_track",
            "owned_audio_frame",
            InformationFidelity::Interpolated,
        )
        .note(
            "beats",
            InversionMethod::Template,
            "beat times are rendered as analytic pulse clicks",
        ),
    ))
}

/// Returns synthesize noise burst.
pub fn synthesize_noise_burst(
    duration_seconds: f32,
    amplitude: f32,
    seed: u64,
    config: AudioSynthesisConfig,
) -> Result<Generated<OwnedAudioFrame>> {
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err(invalid_argument(
            "noise duration_seconds must be finite and greater than zero",
        ));
    }
    if !amplitude.is_finite() || amplitude < 0.0 {
        return Err(invalid_argument(
            "noise amplitude must be finite and non-negative",
        ));
    }

    let len = seconds_to_samples(duration_seconds, config.sample_rate)?.max(1);
    let channels = config.channels as usize;
    let mut state = seed;
    let mut mono = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (((state >> 32) as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        mono.push(noise * amplitude);
    }
    apply_clipping_policy(&mut mono, config.clipping_policy);

    let mut interleaved = Vec::with_capacity(mono.len() * channels);
    for sample in mono {
        for _ in 0..channels {
            interleaved.push(sample);
        }
    }
    let frame = OwnedAudioFrame::new(
        config.start_timestamp,
        config.sample_rate,
        config.channels,
        AudioBuffer::F32(interleaved),
    )?;
    Ok(Generated::new(
        frame,
        InversionTrace::new(
            "noise_burst",
            "owned_audio_frame",
            InformationFidelity::Interpolated,
        ),
    ))
}

/// Returns event to tone segment.
pub fn event_to_tone_segment(
    event: &AnalysisEvent,
    default_duration_seconds: f32,
) -> Option<ToneSegment> {
    if !default_duration_seconds.is_finite() || default_duration_seconds <= 0.0 {
        return None;
    }
    let start_seconds = event
        .timestamp
        .map_or(0.0, |timestamp| timestamp.seconds() as f32);
    if let Some(frequency_hz) = parse_pitch_label(&event.label) {
        return Some(ToneSegment {
            start_seconds,
            tone: ToneSpec {
                frequency_hz,
                duration_seconds: default_duration_seconds,
                amplitude: event.score.unwrap_or(0.5).clamp(0.05, 1.0),
                waveform: Waveform::Sine,
            },
        });
    }
    if event.label == "audio:onset" {
        return Some(ToneSegment {
            start_seconds,
            tone: ToneSpec {
                frequency_hz: 1_200.0,
                duration_seconds: default_duration_seconds.min(0.04),
                amplitude: event.score.unwrap_or(0.8).clamp(0.1, 1.0),
                waveform: Waveform::Pulse,
            },
        });
    }
    None
}

/// Returns events to tone segments.
pub fn events_to_tone_segments(
    events: &[AnalysisEvent],
    default_duration_seconds: f32,
) -> Vec<ToneSegment> {
    events
        .iter()
        .filter_map(|event| event_to_tone_segment(event, default_duration_seconds))
        .collect()
}

/// Returns synthesize events.
pub fn synthesize_events(
    events: &[AnalysisEvent],
    default_duration_seconds: f32,
    config: AudioSynthesisConfig,
) -> Result<Generated<OwnedAudioFrame>> {
    let segments = events_to_tone_segments(events, default_duration_seconds);
    if segments.is_empty() {
        return Err(invalid_argument(
            "events did not contain pitch or onset labels that can be synthesized",
        ));
    }
    let mut generated = synthesize_timeline(&segments, config)?;
    generated.trace.source_type = "analysis_events".to_string();
    generated.trace = generated.trace.note(
        "events",
        InversionMethod::Inferred,
        "pitch and onset labels are interpreted as tones",
    );
    Ok(generated)
}

fn parse_pitch_label(label: &str) -> Option<f32> {
    label
        .strip_prefix("audio:pitch:")
        .and_then(|value| value.strip_suffix("hz"))
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn seconds_to_samples(seconds: f32, sample_rate: u32) -> Result<usize> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(invalid_argument("seconds must be finite and non-negative"));
    }
    Ok((seconds * sample_rate as f32).round() as usize)
}

fn envelope_gain(offset: usize, len: usize, sample_rate: u32, envelope: AmplitudeEnvelope) -> f32 {
    let attack_samples = seconds_to_samples(envelope.attack_seconds, sample_rate).unwrap_or(0);
    let release_samples = seconds_to_samples(envelope.release_seconds, sample_rate).unwrap_or(0);
    let attack = if attack_samples > 0 && offset < attack_samples {
        offset as f32 / attack_samples as f32
    } else {
        1.0
    };
    let release_start = len.saturating_sub(release_samples);
    let release = if release_samples > 0 && offset >= release_start {
        (len.saturating_sub(offset) as f32 / release_samples as f32).clamp(0.0, 1.0)
    } else {
        1.0
    };
    envelope.gain * attack.min(release)
}

fn apply_clipping_policy(samples: &mut [f32], clipping_policy: ClippingPolicy) {
    match clipping_policy {
        ClippingPolicy::Clamp => {
            for sample in samples {
                *sample = sample.clamp(-1.0, 1.0);
            }
        }
        ClippingPolicy::Normalize => {
            let peak = samples
                .iter()
                .map(|sample| sample.abs())
                .fold(0.0_f32, f32::max);
            if peak > 1.0 {
                for sample in samples {
                    *sample /= peak;
                }
            }
        }
    }
}

fn wave_sample(waveform: Waveform, frequency_hz: f32, t: f32) -> f32 {
    let phase = (frequency_hz * t).fract();
    match waveform {
        Waveform::Sine => (2.0 * std::f32::consts::PI * frequency_hz * t).sin(),
        Waveform::Square => {
            if phase < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        Waveform::Saw => (2.0 * phase) - 1.0,
        Waveform::Triangle => 1.0 - (4.0 * (phase - 0.5).abs()),
        Waveform::Pulse => {
            if phase < 0.08 {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

/// Validates an audio generation request and returns a generation plan.
pub fn generate_audio(request: AudioGenerationRequest) -> Result<AudioGenerationResponse> {
    if request.prompt.trim().is_empty() {
        return Err(invalid_argument(
            "request body must include a non-empty `prompt` string",
        ));
    }
    let model_id = request
        .model
        .model_id
        .clone()
        .or_else(|| request.model.model.as_ref().map(|model| model.name.clone()))
        .unwrap_or_else(|| "musicgen-small".to_string());
    let duration_seconds = request.duration_seconds.max(0.1);
    Ok(AudioGenerationResponse {
        accepted: true,
        operation: "generate".to_string(),
        model_id,
        runtime: AudioRuntime::External,
        prompt: request.prompt.clone(),
        duration_seconds,
        plan: format!(
            "Queue `{}` for external audio generation; requested duration {:.1}s.",
            request.prompt, duration_seconds
        ),
    })
}

fn default_generation_duration() -> f32 {
    8.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_generation_validates_non_empty_prompt_and_clamps_duration() {
        assert!(generate_audio(AudioGenerationRequest {
            prompt: " ".to_string(),
            duration_seconds: 1.0,
            model: AudioRuntimeSelection::default(),
        })
        .is_err());

        let response = generate_audio(AudioGenerationRequest {
            prompt: "soft piano".to_string(),
            duration_seconds: 0.0,
            model: AudioRuntimeSelection::default(),
        })
        .unwrap();
        assert_eq!(response.runtime, AudioRuntime::External);
        assert_eq!(response.duration_seconds, 0.1);
    }

    #[test]
    fn synthesizes_tone_as_audio_frame() {
        let generated = synthesize_tone(
            ToneSpec::sine(440.0, 0.01),
            AudioSynthesisConfig::new(1_000, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(generated.value.channels, 2);
        assert_eq!(generated.value.samples_per_channel(), 10);
        assert_eq!(generated.trace.fidelity, InformationFidelity::Interpolated);
    }

    #[test]
    fn maps_pitch_event_to_tone() {
        let event = AnalysisEvent::new("audio_pitch", "audio:pitch:220.00hz").score(0.7);
        let segment = event_to_tone_segment(&event, 0.2).unwrap();
        assert_eq!(segment.tone.frequency_hz, 220.0);
        assert_eq!(segment.tone.amplitude, 0.7);
    }

    #[test]
    fn synthesizes_click_track_at_expected_samples() {
        let generated = synthesize_click_track(
            &[0.0, 0.5],
            1.0,
            1_800.0,
            0.02,
            0.8,
            AudioSynthesisConfig::new(1_000, 1).unwrap(),
        )
        .unwrap();
        let AudioBuffer::F32(samples) = generated.value.data else {
            panic!("expected f32 samples");
        };

        assert_eq!(samples.len(), 1_000);
        assert!(samples[0] > 0.0);
        assert!(samples[500] > 0.0);
    }

    #[test]
    fn clipping_policy_can_normalize_overlap() {
        let generated = synthesize_timeline(
            &[
                ToneSegment {
                    start_seconds: 0.0,
                    tone: ToneSpec {
                        frequency_hz: 440.0,
                        duration_seconds: 0.05,
                        amplitude: 1.0,
                        waveform: Waveform::Sine,
                    },
                },
                ToneSegment {
                    start_seconds: 0.0,
                    tone: ToneSpec {
                        frequency_hz: 440.0,
                        duration_seconds: 0.05,
                        amplitude: 1.0,
                        waveform: Waveform::Sine,
                    },
                },
            ],
            AudioSynthesisConfig::new(8_000, 1)
                .unwrap()
                .clipping_policy(ClippingPolicy::Normalize),
        )
        .unwrap();
        let AudioBuffer::F32(samples) = generated.value.data else {
            panic!("expected synthesized f32 data");
        };
        assert!(samples.iter().all(|sample| sample.abs() <= 1.0));
    }

    #[test]
    fn synthesize_click_and_noise_burst_generate_audio() {
        let click =
            synthesize_click(0.0, 0.01, 0.8, AudioSynthesisConfig::new(8_000, 1).unwrap()).unwrap();
        let noise =
            synthesize_noise_burst(0.01, 0.5, 7, AudioSynthesisConfig::new(8_000, 2).unwrap())
                .unwrap();
        assert_eq!(click.value.channels, 1);
        assert_eq!(noise.value.channels, 2);
        assert!(noise.value.samples_per_channel() > 0);
    }
}
