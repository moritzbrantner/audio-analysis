use audio_analysis_core::normalized_samples;
use audio_contracts::{AudioBuffer, AudioFrame, DetectError, OwnedAudioFrame, Result};
use math_signal_core::{
    db_to_linear, design_parametric_biquad, BiquadCoefficients, ParametricBiquadDesign, SampleRate,
};

use crate::{AudioProcessor, AudioTransform};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Distortion transfer function.
pub enum DistortionMode {
    /// Hard clipping.
    HardClip,
    /// Rational soft clipping.
    SoftClip,
    /// Hyperbolic tangent saturation.
    Tanh,
    /// Foldback distortion.
    Foldback,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Distortion settings.
pub struct DistortionSpec {
    /// Distortion mode.
    pub mode: DistortionMode,
    /// Input drive in dB.
    pub drive_db: f32,
    /// Wet mix in [0, 1].
    pub mix: f32,
    /// Output gain in dB.
    pub output_gain_db: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Feedback delay settings.
pub struct DelaySpec {
    /// Delay length in seconds.
    pub delay_seconds: f64,
    /// Feedback amount in [0, 1).
    pub feedback: f32,
    /// Wet gain.
    pub wet: f32,
    /// Dry gain.
    pub dry: f32,
}

/// Echo uses the same settings as delay.
pub type EchoSpec = DelaySpec;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Lightweight algorithmic reverb settings.
pub struct ReverbSpec {
    /// Room size in [0, 1].
    pub room_size: f32,
    /// Damping in [0, 1].
    pub damping: f32,
    /// Wet gain.
    pub wet: f32,
    /// Dry gain.
    pub dry: f32,
    /// Stereo width in [0, 1].
    pub width: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Compressor settings.
pub struct DynamicsSpec {
    /// Threshold in dBFS.
    pub threshold_db: f32,
    /// Compression ratio.
    pub ratio: f32,
    /// Attack time in milliseconds.
    pub attack_ms: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
    /// Makeup gain in dB.
    pub makeup_gain_db: f32,
    /// Soft knee width in dB.
    pub knee_db: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Limiter settings.
pub struct LimiterSpec {
    /// Output ceiling in dBFS.
    pub ceiling_db: f32,
    /// Release time in milliseconds.
    pub release_ms: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// EQ band kind.
pub enum EqBandKind {
    /// Low pass.
    LowPass,
    /// High pass.
    HighPass,
    /// Band pass.
    BandPass,
    /// Notch.
    Notch,
    /// Peaking EQ.
    Peaking,
    /// Low shelf.
    LowShelf,
    /// High shelf.
    HighShelf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// EQ band settings.
pub struct EqBandSpec {
    /// Band kind.
    pub kind: EqBandKind,
    /// Band frequency.
    pub frequency_hz: f32,
    /// Band Q.
    pub q: f32,
    /// Gain in dB for peaking and shelf bands.
    pub gain_db: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Chorus/flanger modulated delay settings.
pub struct ModulationDelaySpec {
    /// Base delay in milliseconds.
    pub base_delay_ms: f32,
    /// Modulation depth in milliseconds.
    pub depth_ms: f32,
    /// LFO rate in hertz.
    pub rate_hz: f32,
    /// Feedback amount.
    pub feedback: f32,
    /// Wet gain.
    pub wet: f32,
    /// Dry gain.
    pub dry: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Tremolo settings.
pub struct TremoloSpec {
    /// LFO rate in hertz.
    pub rate_hz: f32,
    /// Modulation depth in [0, 1].
    pub depth: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Constant pan position.
pub struct PanSpec {
    /// -1.0 left, 0.0 center, 1.0 right.
    pub position: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Named effect presets.
pub enum AudioEffectPreset {
    /// Clean vocal cleanup.
    VocalClean,
    /// Podcast voice chain.
    PodcastVoice,
    /// Lo-fi saturation.
    LoFi,
    /// Wide chorus.
    WideChorus,
    /// Small room reverb.
    SmallRoomReverb,
    /// Hard limiter.
    HardLimiter,
}

impl AudioProcessor {
    /// Adds distortion.
    pub fn distortion(self, spec: DistortionSpec) -> Self {
        self.transform(Distortion { spec })
    }

    /// Adds feedback delay.
    pub fn delay(self, spec: DelaySpec) -> Self {
        self.transform(Delay::new(spec))
    }

    /// Adds echo.
    pub fn echo(self, spec: EchoSpec) -> Self {
        self.delay(spec)
    }

    /// Adds algorithmic reverb.
    pub fn reverb(self, spec: ReverbSpec) -> Self {
        self.transform(Reverb::new(spec))
    }

    /// Adds compressor.
    pub fn compressor(self, spec: DynamicsSpec) -> Self {
        self.transform(Compressor::new(spec))
    }

    /// Adds limiter.
    pub fn limiter(self, spec: LimiterSpec) -> Self {
        self.transform(Limiter::new(spec))
    }

    /// Adds a multi-band EQ.
    pub fn eq(self, bands: Vec<EqBandSpec>) -> Self {
        self.transform(Eq::new(bands))
    }

    /// Adds chorus.
    pub fn chorus(self, spec: ModulationDelaySpec) -> Self {
        self.transform(ModulationDelay::new("audio_chorus", spec, 0.0))
    }

    /// Adds flanger.
    pub fn flanger(self, spec: ModulationDelaySpec) -> Self {
        self.transform(ModulationDelay::new("audio_flanger", spec, 0.25))
    }

    /// Adds tremolo.
    pub fn tremolo(self, spec: TremoloSpec) -> Self {
        self.transform(Tremolo { spec, phase: 0.0 })
    }

    /// Adds constant pan.
    pub fn pan(self, spec: PanSpec) -> Self {
        self.transform(Pan { spec })
    }

    /// Adds stereo width adjustment.
    pub fn stereo_width(self, width: f32) -> Self {
        self.transform(StereoWidth { width })
    }
}

/// Builds a processor for a named preset.
pub fn preset_chain(preset: AudioEffectPreset) -> AudioProcessor {
    match preset {
        AudioEffectPreset::VocalClean => AudioProcessor::new()
            .eq(vec![
                EqBandSpec {
                    kind: EqBandKind::HighPass,
                    frequency_hz: 80.0,
                    q: 0.707,
                    gain_db: 0.0,
                },
                EqBandSpec {
                    kind: EqBandKind::Peaking,
                    frequency_hz: 3_000.0,
                    q: 1.0,
                    gain_db: 2.0,
                },
            ])
            .compressor(DynamicsSpec {
                threshold_db: -18.0,
                ratio: 2.5,
                attack_ms: 8.0,
                release_ms: 90.0,
                makeup_gain_db: 2.0,
                knee_db: 4.0,
            }),
        AudioEffectPreset::PodcastVoice => AudioProcessor::new()
            .eq(vec![EqBandSpec {
                kind: EqBandKind::HighPass,
                frequency_hz: 70.0,
                q: 0.707,
                gain_db: 0.0,
            }])
            .compressor(DynamicsSpec {
                threshold_db: -20.0,
                ratio: 3.0,
                attack_ms: 5.0,
                release_ms: 120.0,
                makeup_gain_db: 3.0,
                knee_db: 6.0,
            })
            .limiter(LimiterSpec {
                ceiling_db: -1.0,
                release_ms: 50.0,
            }),
        AudioEffectPreset::LoFi => AudioProcessor::new()
            .distortion(DistortionSpec {
                mode: DistortionMode::Tanh,
                drive_db: 12.0,
                mix: 0.6,
                output_gain_db: -4.0,
            })
            .eq(vec![EqBandSpec {
                kind: EqBandKind::LowPass,
                frequency_hz: 6_000.0,
                q: 0.707,
                gain_db: 0.0,
            }]),
        AudioEffectPreset::WideChorus => AudioProcessor::new()
            .chorus(ModulationDelaySpec {
                base_delay_ms: 18.0,
                depth_ms: 8.0,
                rate_hz: 0.35,
                feedback: 0.1,
                wet: 0.45,
                dry: 1.0,
            })
            .stereo_width(1.3),
        AudioEffectPreset::SmallRoomReverb => AudioProcessor::new().reverb(ReverbSpec {
            room_size: 0.25,
            damping: 0.45,
            wet: 0.25,
            dry: 1.0,
            width: 0.8,
        }),
        AudioEffectPreset::HardLimiter => AudioProcessor::new().limiter(LimiterSpec {
            ceiling_db: -0.5,
            release_ms: 35.0,
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Distortion {
    spec: DistortionSpec,
}

impl AudioTransform for Distortion {
    fn name(&self) -> &str {
        "audio_distortion"
    }

    fn reset(&mut self) {}

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        validate_mix(self.spec.mix)?;
        let drive = db_to_linear(self.spec.drive_db)?;
        let output_gain = db_to_linear(self.spec.output_gain_db)?;
        let samples = normalized_interleaved(frame)?
            .into_iter()
            .map(|sample| {
                let driven = sample * drive;
                let wet = match self.spec.mode {
                    DistortionMode::HardClip => driven.clamp(-1.0, 1.0),
                    DistortionMode::SoftClip => driven / (1.0 + driven.abs()),
                    DistortionMode::Tanh => driven.tanh(),
                    DistortionMode::Foldback => foldback(driven),
                };
                (sample * (1.0 - self.spec.mix) + wet * self.spec.mix) * output_gain
            })
            .collect();
        owned_like(frame, samples).map(Some)
    }
}

fn foldback(sample: f32) -> f32 {
    if sample.abs() <= 1.0 {
        sample
    } else {
        let folded = ((sample - 1.0).abs() % 4.0) - 2.0;
        folded.abs() - 1.0
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Delay {
    spec: DelaySpec,
    buffers: Vec<Vec<f32>>,
    positions: Vec<usize>,
    configured_sample_rate: Option<u32>,
}

impl Delay {
    fn new(spec: DelaySpec) -> Self {
        Self {
            spec,
            buffers: Vec::new(),
            positions: Vec::new(),
            configured_sample_rate: None,
        }
    }
}

impl AudioTransform for Delay {
    fn name(&self) -> &str {
        "audio_delay"
    }

    fn reset(&mut self) {
        self.buffers.clear();
        self.positions.clear();
        self.configured_sample_rate = None;
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        validate_delay(self.spec)?;
        let samples = normalized_interleaved(frame)?;
        let channels = frame.channels as usize;
        let delay_samples = (self.spec.delay_seconds * frame.sample_rate as f64)
            .round()
            .max(1.0) as usize;
        if self.configured_sample_rate != Some(frame.sample_rate)
            || self.buffers.len() != channels
            || self.buffers.first().map(Vec::len) != Some(delay_samples)
        {
            self.buffers = vec![vec![0.0; delay_samples]; channels];
            self.positions = vec![0; channels];
            self.configured_sample_rate = Some(frame.sample_rate);
        }
        let mut output = Vec::with_capacity(samples.len());
        for frame_samples in samples.chunks_exact(channels) {
            for (channel, sample) in frame_samples.iter().enumerate() {
                let pos = self.positions[channel];
                let delayed = self.buffers[channel][pos];
                let value = *sample * self.spec.dry + delayed * self.spec.wet;
                self.buffers[channel][pos] = *sample + delayed * self.spec.feedback;
                self.positions[channel] = (pos + 1) % delay_samples;
                output.push(value);
            }
        }
        owned_like(frame, output).map(Some)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Reverb {
    spec: ReverbSpec,
    combs: Vec<DelayLine>,
    allpasses: Vec<DelayLine>,
    configured: Option<(u32, u16)>,
}

impl Reverb {
    fn new(spec: ReverbSpec) -> Self {
        Self {
            spec,
            combs: Vec::new(),
            allpasses: Vec::new(),
            configured: None,
        }
    }

    fn configure(&mut self, sample_rate: u32, channels: u16) {
        let base = [0.0297, 0.0371, 0.0411, 0.0437];
        let allpass = [0.005, 0.0017];
        let channels_usize = channels as usize;
        self.combs = base
            .iter()
            .flat_map(|seconds| {
                (0..channels_usize).map(move |channel| {
                    let spread = channel as f64 * 0.000_37;
                    DelayLine::new(((*seconds + spread) * sample_rate as f64).round() as usize)
                })
            })
            .collect();
        self.allpasses = allpass
            .iter()
            .flat_map(|seconds| {
                (0..channels_usize).map(move |channel| {
                    let spread = channel as f64 * 0.000_23;
                    DelayLine::new(((*seconds + spread) * sample_rate as f64).round() as usize)
                })
            })
            .collect();
        self.configured = Some((sample_rate, channels));
    }
}

impl AudioTransform for Reverb {
    fn name(&self) -> &str {
        "audio_reverb"
    }

    fn reset(&mut self) {
        self.combs.clear();
        self.allpasses.clear();
        self.configured = None;
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if !self.spec.room_size.is_finite()
            || !self.spec.damping.is_finite()
            || !self.spec.wet.is_finite()
            || !self.spec.dry.is_finite()
            || !self.spec.width.is_finite()
        {
            return Err(DetectError::InvalidArgument(
                "reverb parameters must be finite".to_string(),
            ));
        }
        if self.configured != Some((frame.sample_rate, frame.channels)) {
            self.configure(frame.sample_rate, frame.channels);
        }
        let samples = normalized_interleaved(frame)?;
        let channels = frame.channels as usize;
        let comb_count = self.combs.len() / channels;
        let allpass_count = self.allpasses.len() / channels;
        let feedback = 0.55 + self.spec.room_size.clamp(0.0, 1.0) * 0.35;
        let damping = self.spec.damping.clamp(0.0, 0.99);
        let mut output = Vec::with_capacity(samples.len());
        for frame_samples in samples.chunks_exact(channels) {
            for (channel, sample) in frame_samples.iter().enumerate() {
                let mut wet = 0.0;
                for comb in 0..comb_count {
                    let line = &mut self.combs[comb * channels + channel];
                    let delayed = line.read();
                    line.write(*sample + delayed * feedback * (1.0 - damping));
                    wet += delayed;
                }
                wet /= comb_count.max(1) as f32;
                for allpass in 0..allpass_count {
                    let line = &mut self.allpasses[allpass * channels + channel];
                    let delayed = line.read();
                    line.write(wet + delayed * 0.5);
                    wet = -wet + delayed;
                }
                output.push(*sample * self.spec.dry + wet * self.spec.wet);
            }
        }
        if channels == 2 {
            apply_width(&mut output, self.spec.width.clamp(0.0, 1.0), 2);
        }
        owned_like(frame, output).map(Some)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Compressor {
    spec: DynamicsSpec,
    envelope: Vec<f32>,
}

impl Compressor {
    fn new(spec: DynamicsSpec) -> Self {
        Self {
            spec,
            envelope: Vec::new(),
        }
    }
}

impl AudioTransform for Compressor {
    fn name(&self) -> &str {
        "audio_compressor"
    }

    fn reset(&mut self) {
        self.envelope.clear();
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        validate_dynamics(self.spec)?;
        let samples = normalized_interleaved(frame)?;
        let channels = frame.channels as usize;
        self.envelope.resize(channels, 0.0);
        let attack = smoothing(self.spec.attack_ms, frame.sample_rate);
        let release = smoothing(self.spec.release_ms, frame.sample_rate);
        let makeup = db_to_linear(self.spec.makeup_gain_db)?;
        let mut output = Vec::with_capacity(samples.len());
        for frame_samples in samples.chunks_exact(channels) {
            for (channel, sample) in frame_samples.iter().enumerate() {
                let level = sample.abs();
                let coeff = if level > self.envelope[channel] {
                    attack
                } else {
                    release
                };
                self.envelope[channel] = coeff * self.envelope[channel] + (1.0 - coeff) * level;
                let env_db = amplitude_to_db(self.envelope[channel]);
                let gain_db = compression_gain_db(
                    env_db,
                    self.spec.threshold_db,
                    self.spec.ratio,
                    self.spec.knee_db,
                );
                output.push(*sample * db_to_linear(gain_db)? * makeup);
            }
        }
        owned_like(frame, output).map(Some)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Limiter {
    spec: LimiterSpec,
    compressor: Compressor,
}

impl Limiter {
    fn new(spec: LimiterSpec) -> Self {
        Self {
            spec,
            compressor: Compressor::new(DynamicsSpec {
                threshold_db: spec.ceiling_db,
                ratio: 20.0,
                attack_ms: 0.1,
                release_ms: spec.release_ms,
                makeup_gain_db: 0.0,
                knee_db: 0.0,
            }),
        }
    }
}

impl AudioTransform for Limiter {
    fn name(&self) -> &str {
        "audio_limiter"
    }

    fn reset(&mut self) {
        self.compressor.reset();
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if !self.spec.ceiling_db.is_finite() || !self.spec.release_ms.is_finite() {
            return Err(DetectError::InvalidArgument(
                "limiter parameters must be finite".to_string(),
            ));
        }
        let processed = self
            .compressor
            .process_frame(frame)?
            .ok_or_else(|| DetectError::InvalidArgument("limiter produced no frame".to_string()))?;
        let ceiling = db_to_linear(self.spec.ceiling_db)?.abs();
        let samples = normalized_samples(&processed.data)
            .into_iter()
            .map(|sample| sample.clamp(-ceiling, ceiling))
            .collect();
        OwnedAudioFrame::new(
            processed.timestamp,
            processed.sample_rate,
            processed.channels,
            AudioBuffer::F32(samples),
        )
        .map(Some)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Eq {
    bands: Vec<EqBandSpec>,
    filters: Vec<ParametricBiquadFilter>,
    configured_sample_rate: Option<u32>,
}

impl Eq {
    fn new(bands: Vec<EqBandSpec>) -> Self {
        Self {
            filters: Vec::new(),
            bands,
            configured_sample_rate: None,
        }
    }

    fn configure(&mut self, sample_rate: u32) -> Result<()> {
        self.filters = self
            .bands
            .iter()
            .copied()
            .map(|band| ParametricBiquadFilter::new(band, sample_rate))
            .collect::<Result<_>>()?;
        self.configured_sample_rate = Some(sample_rate);
        Ok(())
    }
}

impl AudioTransform for Eq {
    fn name(&self) -> &str {
        "audio_eq"
    }

    fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.states.clear();
        }
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if self.configured_sample_rate != Some(frame.sample_rate) {
            self.configure(frame.sample_rate)?;
        }
        let mut current = normalized_interleaved(frame)?;
        let channels = frame.channels as usize;
        for filter in &mut self.filters {
            current = filter.process(&current, channels);
        }
        owned_like(frame, current).map(Some)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct BiquadState {
    z1: f32,
    z2: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct ParametricBiquadFilter {
    coefficients: BiquadCoefficients,
    states: Vec<BiquadState>,
}

impl ParametricBiquadFilter {
    fn new(spec: EqBandSpec, sample_rate: u32) -> Result<Self> {
        let design = match spec.kind {
            EqBandKind::LowPass => ParametricBiquadDesign::LowPass,
            EqBandKind::HighPass => ParametricBiquadDesign::HighPass,
            EqBandKind::BandPass => ParametricBiquadDesign::BandPass,
            EqBandKind::Notch => ParametricBiquadDesign::Notch,
            EqBandKind::Peaking => ParametricBiquadDesign::PeakingEq {
                gain_db: spec.gain_db,
            },
            EqBandKind::LowShelf => ParametricBiquadDesign::LowShelf {
                gain_db: spec.gain_db,
            },
            EqBandKind::HighShelf => ParametricBiquadDesign::HighShelf {
                gain_db: spec.gain_db,
            },
        };
        Ok(Self {
            coefficients: design_parametric_biquad(
                design,
                SampleRate::new(sample_rate)?,
                spec.frequency_hz,
                spec.q,
            )?,
            states: Vec::new(),
        })
    }

    fn process(&mut self, samples: &[f32], channels: usize) -> Vec<f32> {
        self.states.resize(channels, BiquadState::default());
        let mut output = Vec::with_capacity(samples.len());
        for frame_samples in samples.chunks_exact(channels) {
            for (channel, sample) in frame_samples.iter().enumerate() {
                let state = &mut self.states[channel];
                let filtered = self.coefficients.b0 * *sample + state.z1;
                state.z1 =
                    self.coefficients.b1 * *sample - self.coefficients.a1 * filtered + state.z2;
                state.z2 = self.coefficients.b2 * *sample - self.coefficients.a2 * filtered;
                output.push(filtered);
            }
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ModulationDelay {
    name: &'static str,
    spec: ModulationDelaySpec,
    phase_offset: f32,
    buffers: Vec<Vec<f32>>,
    positions: Vec<usize>,
    phase: f32,
    configured: Option<(u32, u16, usize)>,
}

impl ModulationDelay {
    fn new(name: &'static str, spec: ModulationDelaySpec, phase_offset: f32) -> Self {
        Self {
            name,
            spec,
            phase_offset,
            buffers: Vec::new(),
            positions: Vec::new(),
            phase: 0.0,
            configured: None,
        }
    }
}

impl AudioTransform for ModulationDelay {
    fn name(&self) -> &str {
        self.name
    }

    fn reset(&mut self) {
        self.buffers.clear();
        self.positions.clear();
        self.phase = 0.0;
        self.configured = None;
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        validate_modulation(self.spec)?;
        let samples = normalized_interleaved(frame)?;
        let channels = frame.channels as usize;
        let max_delay = ((self.spec.base_delay_ms + self.spec.depth_ms).max(0.1)
            * frame.sample_rate as f32
            / 1_000.0)
            .ceil() as usize
            + 2;
        if self.configured != Some((frame.sample_rate, frame.channels, max_delay)) {
            self.buffers = vec![vec![0.0; max_delay]; channels];
            self.positions = vec![0; channels];
            self.configured = Some((frame.sample_rate, frame.channels, max_delay));
        }
        let phase_step = self.spec.rate_hz / frame.sample_rate as f32;
        let mut output = Vec::with_capacity(samples.len());
        for frame_samples in samples.chunks_exact(channels) {
            let lfo = (2.0 * std::f32::consts::PI * (self.phase + self.phase_offset)).sin();
            let delay = (self.spec.base_delay_ms + self.spec.depth_ms * (0.5 + 0.5 * lfo))
                * frame.sample_rate as f32
                / 1_000.0;
            for (channel, sample) in frame_samples.iter().enumerate() {
                let pos = self.positions[channel];
                let delayed = read_delay(&self.buffers[channel], pos, delay);
                let value = *sample * self.spec.dry + delayed * self.spec.wet;
                self.buffers[channel][pos] = *sample + delayed * self.spec.feedback;
                self.positions[channel] = (pos + 1) % max_delay;
                output.push(value);
            }
            self.phase = (self.phase + phase_step) % 1.0;
        }
        owned_like(frame, output).map(Some)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Tremolo {
    spec: TremoloSpec,
    phase: f32,
}

impl AudioTransform for Tremolo {
    fn name(&self) -> &str {
        "audio_tremolo"
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if !self.spec.rate_hz.is_finite()
            || self.spec.rate_hz < 0.0
            || !self.spec.depth.is_finite()
            || !(0.0..=1.0).contains(&self.spec.depth)
        {
            return Err(DetectError::InvalidArgument(
                "tremolo rate must be non-negative and depth must be in [0, 1]".to_string(),
            ));
        }
        let samples = normalized_interleaved(frame)?;
        let channels = frame.channels as usize;
        let phase_step = self.spec.rate_hz / frame.sample_rate as f32;
        let mut output = Vec::with_capacity(samples.len());
        for frame_samples in samples.chunks_exact(channels) {
            let lfo = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * self.phase).sin();
            let gain = 1.0 - self.spec.depth + self.spec.depth * lfo;
            output.extend(frame_samples.iter().map(|sample| *sample * gain));
            self.phase = (self.phase + phase_step) % 1.0;
        }
        owned_like(frame, output).map(Some)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Pan {
    spec: PanSpec,
}

impl AudioTransform for Pan {
    fn name(&self) -> &str {
        "audio_pan"
    }

    fn reset(&mut self) {}

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if !self.spec.position.is_finite() || !(-1.0..=1.0).contains(&self.spec.position) {
            return Err(DetectError::InvalidArgument(
                "pan position must be finite and in [-1, 1]".to_string(),
            ));
        }
        let samples = normalized_interleaved(frame)?;
        let position = self.spec.position;
        match frame.channels {
            1 => {
                let angle = (position + 1.0) * std::f32::consts::FRAC_PI_4;
                let left_gain = angle.cos();
                let right_gain = angle.sin();
                let mut output = Vec::with_capacity(samples.len() * 2);
                for sample in samples {
                    output.push(sample * left_gain);
                    output.push(sample * right_gain);
                }
                OwnedAudioFrame::new(
                    frame.timestamp,
                    frame.sample_rate,
                    2,
                    AudioBuffer::F32(output),
                )
                .map(Some)
            }
            2 => {
                let angle = (position + 1.0) * std::f32::consts::FRAC_PI_4;
                let left_gain = angle.cos() * std::f32::consts::SQRT_2;
                let right_gain = angle.sin() * std::f32::consts::SQRT_2;
                let output = samples
                    .chunks_exact(2)
                    .flat_map(|frame| [frame[0] * left_gain, frame[1] * right_gain])
                    .collect();
                owned_like(frame, output).map(Some)
            }
            channels => Err(DetectError::InvalidArgument(format!(
                "pan supports mono or stereo frames, got {channels} channels"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct StereoWidth {
    width: f32,
}

impl AudioTransform for StereoWidth {
    fn name(&self) -> &str {
        "audio_stereo_width"
    }

    fn reset(&mut self) {}

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Option<OwnedAudioFrame>> {
        if !self.width.is_finite() {
            return Err(DetectError::InvalidArgument(
                "stereo width must be finite".to_string(),
            ));
        }
        let mut samples = normalized_interleaved(frame)?;
        match frame.channels {
            1 => owned_like(frame, samples).map(Some),
            2 => {
                apply_width(&mut samples, self.width, 2);
                owned_like(frame, samples).map(Some)
            }
            channels => Err(DetectError::InvalidArgument(format!(
                "stereo width supports mono or stereo frames, got {channels} channels"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DelayLine {
    buffer: Vec<f32>,
    position: usize,
}

impl DelayLine {
    fn new(len: usize) -> Self {
        Self {
            buffer: vec![0.0; len.max(1)],
            position: 0,
        }
    }

    fn read(&self) -> f32 {
        self.buffer[self.position]
    }

    fn write(&mut self, sample: f32) {
        self.buffer[self.position] = sample;
        self.position = (self.position + 1) % self.buffer.len();
    }
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

fn validate_mix(mix: f32) -> Result<()> {
    if mix.is_finite() && (0.0..=1.0).contains(&mix) {
        Ok(())
    } else {
        Err(DetectError::InvalidArgument(
            "mix must be finite and in [0, 1]".to_string(),
        ))
    }
}

fn validate_delay(spec: DelaySpec) -> Result<()> {
    if !spec.delay_seconds.is_finite()
        || spec.delay_seconds <= 0.0
        || !spec.feedback.is_finite()
        || !(0.0..1.0).contains(&spec.feedback)
        || !spec.wet.is_finite()
        || !spec.dry.is_finite()
    {
        return Err(DetectError::InvalidArgument(
            "delay requires positive delay_seconds, feedback in [0, 1), and finite gains"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_dynamics(spec: DynamicsSpec) -> Result<()> {
    if !spec.threshold_db.is_finite()
        || !spec.ratio.is_finite()
        || spec.ratio < 1.0
        || !spec.attack_ms.is_finite()
        || spec.attack_ms < 0.0
        || !spec.release_ms.is_finite()
        || spec.release_ms < 0.0
        || !spec.makeup_gain_db.is_finite()
        || !spec.knee_db.is_finite()
        || spec.knee_db < 0.0
    {
        return Err(DetectError::InvalidArgument(
            "dynamics parameters are out of range".to_string(),
        ));
    }
    Ok(())
}

fn validate_modulation(spec: ModulationDelaySpec) -> Result<()> {
    if !spec.base_delay_ms.is_finite()
        || spec.base_delay_ms <= 0.0
        || !spec.depth_ms.is_finite()
        || spec.depth_ms < 0.0
        || !spec.rate_hz.is_finite()
        || spec.rate_hz < 0.0
        || !spec.feedback.is_finite()
        || !(-0.99..0.99).contains(&spec.feedback)
        || !spec.wet.is_finite()
        || !spec.dry.is_finite()
    {
        return Err(DetectError::InvalidArgument(
            "modulation delay parameters are out of range".to_string(),
        ));
    }
    Ok(())
}

fn smoothing(ms: f32, sample_rate: u32) -> f32 {
    if ms <= 0.0 {
        0.0
    } else {
        (-1.0 / (sample_rate as f32 * ms / 1_000.0)).exp()
    }
}

fn amplitude_to_db(amplitude: f32) -> f32 {
    20.0 * amplitude.max(1.0e-9).log10()
}

fn compression_gain_db(level_db: f32, threshold_db: f32, ratio: f32, knee_db: f32) -> f32 {
    let over = level_db - threshold_db;
    if knee_db > 0.0 && over.abs() <= knee_db / 2.0 {
        let x = over + knee_db / 2.0;
        let compressed = x * x / (2.0 * knee_db) * (1.0 / ratio - 1.0);
        compressed.min(0.0)
    } else if over > 0.0 {
        over * (1.0 / ratio - 1.0)
    } else {
        0.0
    }
}

fn read_delay(buffer: &[f32], write_pos: usize, delay: f32) -> f32 {
    let len = buffer.len();
    let read = write_pos as f32 - delay;
    let wrapped = if read < 0.0 { read + len as f32 } else { read };
    let left = wrapped.floor() as usize % len;
    let right = (left + 1) % len;
    let fraction = wrapped - wrapped.floor();
    buffer[left] * (1.0 - fraction) + buffer[right] * fraction
}

fn apply_width(samples: &mut [f32], width: f32, channels: usize) {
    if channels != 2 {
        return;
    }
    for frame in samples.chunks_exact_mut(2) {
        let mid = (frame[0] + frame[1]) * 0.5;
        let side = (frame[0] - frame[1]) * 0.5 * width;
        frame[0] = mid + side;
        frame[1] = mid - side;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_contracts::{AudioBuffer, Timebase, Timestamp};

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
            _ => panic!("expected f32"),
        }
    }

    #[test]
    fn distortion_and_limiter_bound_output() {
        let mut processor = AudioProcessor::new()
            .distortion(DistortionSpec {
                mode: DistortionMode::HardClip,
                drive_db: 12.0,
                mix: 1.0,
                output_gain_db: 0.0,
            })
            .limiter(LimiterSpec {
                ceiling_db: -6.0,
                release_ms: 20.0,
            });
        let output = processor
            .process_frame(frame(vec![2.0, -2.0, 0.25], 1))
            .unwrap()
            .unwrap();
        let ceiling = 10.0_f32.powf(-6.0 / 20.0) + 1.0e-6;
        assert!(samples(&output)
            .iter()
            .all(|sample| sample.abs() <= ceiling));
    }

    #[test]
    fn delay_and_reverb_emit_tail_across_chunks() {
        let mut delay = AudioProcessor::new().delay(DelaySpec {
            delay_seconds: 1.0 / 48_000.0,
            feedback: 0.5,
            wet: 1.0,
            dry: 0.0,
        });
        let first = delay
            .process_frame(frame(vec![1.0, 0.0], 1))
            .unwrap()
            .unwrap();
        let second = delay.process_frame(frame(vec![0.0], 1)).unwrap().unwrap();
        assert_eq!(samples(&first)[0], 0.0);
        assert!(samples(&first)[1] > 0.9 || samples(&second)[0] > 0.4);

        let mut reverb = AudioProcessor::new().reverb(ReverbSpec {
            room_size: 0.4,
            damping: 0.2,
            wet: 1.0,
            dry: 0.0,
            width: 0.5,
        });
        let _ = reverb.process_frame(frame(vec![1.0], 1)).unwrap().unwrap();
        let tail = reverb
            .process_frame(frame(vec![0.0; 3_000], 1))
            .unwrap()
            .unwrap();
        assert!(samples(&tail).iter().any(|sample| sample.abs() > 0.0));
    }

    #[test]
    fn pan_and_width_shape_channels() {
        let mut pan = AudioProcessor::new().pan(PanSpec { position: -1.0 });
        let output = pan.process_frame(frame(vec![1.0], 1)).unwrap().unwrap();
        assert_eq!(output.channels, 2);
        assert!(samples(&output)[0] > samples(&output)[1]);

        let mut width = AudioProcessor::new().stereo_width(0.0);
        let output = width
            .process_frame(frame(vec![1.0, -1.0], 2))
            .unwrap()
            .unwrap();
        assert_eq!(samples(&output), &[0.0, 0.0]);
    }
}
