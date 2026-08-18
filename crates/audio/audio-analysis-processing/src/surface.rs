//! Library-owned runtime surface for `audio-analysis-processing`.

use audio_analysis_core::{
    interleaved_to_mono, mean_absolute, peak, rms, summarize_feature_series, AudioClip, ChannelMix,
    FadeCurve,
};
use audio_contracts::{AudioBuffer, OwnedAudioFrame, Timebase, Timestamp};
use runtime_core::{
    describe_surface_response, structured_surface_response, surface_operation, PackageSurface,
    RuntimeCapabilities, SurfaceRequest, SurfaceResponse,
};

use crate::operations::effects_catalog_value;
use crate::{
    mixdown_placements, preset_chain, AudioClipPlacement, AudioEffectPreset, AudioEnergyAnalyzer,
    AudioProcessor, DelaySpec, DistortionMode, DistortionSpec, DynamicsSpec, EqBandKind,
    EqBandSpec, FadeSpec, LimiterSpec, LoudnessOptions, ModulationDelaySpec, NoiseGateSpec,
    NormalizeSpec, OfflineAudioProcessor, PanSpec, PitchShiftMode, ReverbSpec, SpeedMode,
    TremoloSpec,
};

const MAX_SAMPLES: usize = 192_000;
const DEFAULT_PREVIEW_SAMPLES: usize = 1024;

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            surface_operation(
                "describe",
                "Describe package",
                "Realtime-safe audio transforms and processed sources for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            surface_operation(
                "audio.processing.apply",
                "Apply processing",
                "Applies an in-memory streaming chain to normalized samples.",
                serde_json::json!({"samples": [0.0, 0.5, -0.5], "sampleRate": 48000, "channels": 1, "chain": [{"type": "gain", "linear": 0.8}, {"type": "distortion", "mode": "tanh", "driveDb": 12.0, "mix": 0.5}]}),
            ),
            surface_operation(
                "audio.processing.effectsCatalog",
                "Inspect effects catalog",
                "Inspects supported streaming and offline audio processing operations without applying them.",
                serde_json::json!({}),
            ),
            surface_operation(
                "audio.processing.offlineEdit",
                "Offline edit",
                "Applies deterministic whole-clip edits such as trim, reverse, fade, normalize, speed, and pitch shift.",
                serde_json::json!({"samples": [0.0, 0.5, -0.5], "sampleRate": 48000, "channels": 1, "chain": [{"type": "reverse"}]}),
            ),
            surface_operation(
                "audio.processing.mixdown",
                "Mixdown",
                "Mixes multiple whole-clip placements onto one deterministic output timeline.",
                serde_json::json!({"sampleRate": 48000, "channels": 1, "placements": [{"samples": [0.0, 0.5], "startSeconds": 0.0, "gain": 1.0}, {"samples": [0.25], "startSeconds": 0.5, "gain": 0.8}]}),
            ),
            surface_operation(
                "audio.processing.preset",
                "Preset",
                "Applies or describes a named audio effect preset.",
                serde_json::json!({"preset": "PodcastVoice", "samples": [0.0, 0.5, -0.5], "sampleRate": 48000, "channels": 1}),
            ),
            surface_operation(
                "audio.processing.energy",
                "Energy",
                "Returns RMS, peak, mean absolute value, and silence/loud labels.",
                serde_json::json!({"samples": [0.0, 0.5, -0.5], "sampleRate": 48000, "channels": 1}),
            ),
            surface_operation(
                "audio.processing.loudness",
                "Loudness",
                "Computes deterministic loudness-oriented peak, RMS, crest factor, and approximate LUFS metrics.",
                serde_json::json!({"samples": [0.0, 0.5, -0.5, 0.25], "sampleRate": 48000, "channels": 1, "frameSize": 2, "hopSize": 1}),
            ),
            surface_operation(
                "audio.processing.chainSummary",
                "Inspect chain summary",
                "Inspects the deterministic transform chain that would be applied.",
                serde_json::json!({"gain": 0.8, "mono": true, "noiseGateThreshold": 0.01}),
            ),
        ],
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&package_surface(), request)),
        "audio.processing.apply" => apply_value(request.input)?,
        "audio.processing.effectsCatalog" => effects_catalog_value(),
        "audio.processing.offlineEdit" => offline_edit_value(request.input)?,
        "audio.processing.mixdown" => mixdown_value(request.input)?,
        "audio.processing.preset" => preset_value(request.input)?,
        "audio.processing.energy" => energy_value(request.input)?,
        "audio.processing.loudness" => loudness_value(request.input)?,
        "audio.processing.chainSummary" => chain_summary_value(request.input)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ));
        }
    };
    Ok(response(operation, value))
}

fn response(operation: runtime_core::OperationId, value: serde_json::Value) -> SurfaceResponse {
    let (title, message, summary) = match operation.as_str() {
        "audio.processing.apply" => (
            "Audio processing result",
            "Applied an in-memory streaming processing chain to normalized samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "channels": value.get("channels").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "processorCount": value.get("processorCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.processing.offlineEdit" => (
            "Offline audio edit result",
            "Applied deterministic whole-clip edits to the supplied audio samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "channels": value.get("channels").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.processing.mixdown" => (
            "Audio mixdown result",
            "Mixed deterministic whole-clip placements onto one output timeline.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "channels": value.get("channels").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.processing.preset" => (
            "Audio preset result",
            "Applied or described the requested audio processing preset.",
            serde_json::json!({
                "preset": value.get("preset").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.processing.energy" => (
            "Audio energy result",
            "Computed RMS, peak, mean absolute value, and level labels for normalized samples.",
            serde_json::json!({
                "rms": value.get("rms").cloned().unwrap_or(serde_json::Value::Null),
                "peak": value.get("peak").cloned().unwrap_or(serde_json::Value::Null),
                "label": value.get("label").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.processing.loudness" => (
            "Audio loudness result",
            "Computed deterministic peak, RMS, crest factor, and approximate LUFS-style loudness metrics. The LUFS value is a lightweight approximation, not an EBU R128 compliance result.",
            serde_json::json!({
                "peakDbfs": value.get("peakDbfs").cloned().unwrap_or(serde_json::Value::Null),
                "rmsDbfs": value.get("rmsDbfs").cloned().unwrap_or(serde_json::Value::Null),
                "crestFactorDb": value.get("crestFactorDb").cloned().unwrap_or(serde_json::Value::Null),
                "approximateLufs": value.get("approximateLufs").cloned().unwrap_or(serde_json::Value::Null),
                "frameCount": value.get("frameSummary").and_then(|summary| summary.get("frame_count")).cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.processing.effectsCatalog" => (
            "Audio effects catalog",
            "Inspected supported streaming and offline processing operations without processing audio.",
            serde_json::json!({
                "streamingEffectCount": value.get("streamingEffects").and_then(serde_json::Value::as_array).map_or(0, Vec::len),
                "offlineEditCount": value.get("offlineEdits").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.processing.chainSummary" => (
            "Audio processing chain summary",
            "Inspected the processing chain that would be built from the supplied request.",
            serde_json::json!({
                "operationCount": value.get("operations").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        _ => (
            "Audio processing operation result",
            "Completed the audio processing package surface operation.",
            serde_json::json!({}),
        ),
    };
    structured_surface_response(operation, title, message, summary, value)
}

fn apply_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let processor = if let Some(chain) = input.get("chain") {
        processor_from_chain(chain)?
    } else {
        legacy_processor(&input)?
    };
    apply_processor(input, samples, sample_rate, channels, processor)
}

fn apply_processor(
    input: serde_json::Value,
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    mut processor: AudioProcessor,
) -> Result<serde_json::Value, String> {
    let frame = OwnedAudioFrame::new(
        Timestamp::new(0, Timebase::new(1, sample_rate as i32)),
        sample_rate,
        channels,
        AudioBuffer::F32(samples),
    )
    .map_err(|error| error.to_string())?;
    let processed = processor
        .process_frame(frame)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "processing chain produced no frame".to_string())?;
    let processed_samples = match &processed.data {
        AudioBuffer::F32(samples) => samples.as_slice(),
        _ => return Err("processing output was not f32".to_string()),
    };
    Ok(serde_json::json!({
        "sampleRate": processed.sample_rate,
        "channels": processed.channels,
        "sampleCount": processed_samples.len(),
        "samplesPerChannel": processed.samples_per_channel(),
        "rms": rms(processed_samples),
        "peak": peak(processed_samples),
        "samplePreview": preview(processed_samples, preview_limit(&input)?)
    }))
}

fn legacy_processor(input: &serde_json::Value) -> Result<AudioProcessor, String> {
    let mut processor = AudioProcessor::new();
    if let Some(gain) = finite_f32(input, "gain")? {
        processor = processor.gain(gain);
    }
    if input
        .get("mono")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        processor = processor.mono(ChannelMix::Average);
    }
    if input.get("clipMin").is_some() || input.get("clipMax").is_some() {
        processor = processor.hard_clip(
            finite_f32(input, "clipMin")?.unwrap_or(-1.0),
            finite_f32(input, "clipMax")?.unwrap_or(1.0),
        );
    }
    if let Some(threshold) = finite_f32(input, "noiseGateThreshold")? {
        processor = processor.noise_gate(NoiseGateSpec {
            threshold,
            attenuation: finite_f32(input, "noiseGateAttenuation")?.unwrap_or(0.0),
        });
    }
    Ok(processor)
}

fn processor_from_chain(chain: &serde_json::Value) -> Result<AudioProcessor, String> {
    let chain = chain
        .as_array()
        .ok_or_else(|| "chain must be an array".to_string())?;
    let mut processor = AudioProcessor::new();
    for item in chain {
        let effect_type = string_field(item, "type", "")?;
        processor = match effect_type.as_str() {
            "gain" => processor.gain(
                finite_f32(item, "linear")?
                    .or_else(|| finite_f32(item, "gain").ok().flatten())
                    .unwrap_or(1.0),
            ),
            "gainDb" | "gain_db" => processor.gain_db(finite_f32(item, "db")?.unwrap_or(0.0)),
            "hardClip" | "hard_clip" => processor.hard_clip(
                finite_f32(item, "min")?.unwrap_or(-1.0),
                finite_f32(item, "max")?.unwrap_or(1.0),
            ),
            "mono" => processor.mono(ChannelMix::Average),
            "noiseGate" | "noise_gate" => processor.noise_gate(NoiseGateSpec {
                threshold: finite_f32(item, "threshold")?.unwrap_or(0.01),
                attenuation: finite_f32(item, "attenuation")?.unwrap_or(0.0),
            }),
            "dcBlock" | "dc_block" => {
                processor.dc_block(finite_f32(item, "coefficient")?.unwrap_or(0.995))
            }
            "distortion" => processor.distortion(DistortionSpec {
                mode: distortion_mode(&string_field(item, "mode", "tanh")?)?,
                drive_db: finite_f32(item, "driveDb")?.unwrap_or(0.0),
                mix: finite_f32(item, "mix")?.unwrap_or(1.0),
                output_gain_db: finite_f32(item, "outputGainDb")?.unwrap_or(0.0),
            }),
            "delay" | "echo" => processor.delay(DelaySpec {
                delay_seconds: finite_f64(item, "delaySeconds")?.unwrap_or(0.25),
                feedback: finite_f32(item, "feedback")?.unwrap_or(0.0),
                wet: finite_f32(item, "wet")?.unwrap_or(0.5),
                dry: finite_f32(item, "dry")?.unwrap_or(1.0),
            }),
            "reverb" => processor.reverb(ReverbSpec {
                room_size: finite_f32(item, "roomSize")?.unwrap_or(0.5),
                damping: finite_f32(item, "damping")?.unwrap_or(0.5),
                wet: finite_f32(item, "wet")?.unwrap_or(0.3),
                dry: finite_f32(item, "dry")?.unwrap_or(1.0),
                width: finite_f32(item, "width")?.unwrap_or(0.8),
            }),
            "compressor" => processor.compressor(DynamicsSpec {
                threshold_db: finite_f32(item, "thresholdDb")?.unwrap_or(-18.0),
                ratio: finite_f32(item, "ratio")?.unwrap_or(3.0),
                attack_ms: finite_f32(item, "attackMs")?.unwrap_or(10.0),
                release_ms: finite_f32(item, "releaseMs")?.unwrap_or(100.0),
                makeup_gain_db: finite_f32(item, "makeupGainDb")?.unwrap_or(0.0),
                knee_db: finite_f32(item, "kneeDb")?.unwrap_or(0.0),
            }),
            "limiter" => processor.limiter(LimiterSpec {
                ceiling_db: finite_f32(item, "ceilingDb")?.unwrap_or(-1.0),
                release_ms: finite_f32(item, "releaseMs")?.unwrap_or(50.0),
            }),
            "eq" => processor.eq(eq_bands(item)?),
            "chorus" => processor.chorus(modulation_spec(item)?),
            "flanger" => processor.flanger(modulation_spec(item)?),
            "tremolo" => processor.tremolo(TremoloSpec {
                rate_hz: finite_f32(item, "rateHz")?.unwrap_or(5.0),
                depth: finite_f32(item, "depth")?.unwrap_or(0.5),
            }),
            "pan" => processor.pan(PanSpec {
                position: finite_f32(item, "position")?.unwrap_or(0.0),
            }),
            "stereoWidth" | "stereo_width" => {
                processor.stereo_width(finite_f32(item, "width")?.unwrap_or(1.0))
            }
            other => return Err(format!("unsupported chain effect type `{other}`")),
        };
    }
    Ok(processor)
}

fn energy_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let silence_threshold = finite_f32(&input, "silenceThreshold")?.unwrap_or(0.01);
    let loud_threshold = finite_f32(&input, "loudThreshold")?.unwrap_or(0.5);
    AudioEnergyAnalyzer::new(silence_threshold, loud_threshold)
        .map_err(|error| error.to_string())?;
    let level = rms(&samples);
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "channels": channels,
        "sampleCount": samples.len(),
        "rms": level,
        "peak": peak(&samples),
        "meanAbsolute": mean_absolute(&samples),
        "isSilent": level < silence_threshold,
        "isLoud": level >= loud_threshold
    }))
}

fn loudness_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let mono = if channels == 1 {
        samples.clone()
    } else {
        interleaved_to_mono(
            &AudioBuffer::F32(samples.clone()),
            channels,
            ChannelMix::Average,
        )
        .map_err(|error| error.to_string())?
    };
    let frame_size = positive_usize(&input, "frameSize", mono.len().clamp(1, 1024))?;
    let hop_size = positive_usize(&input, "hopSize", frame_size)?;
    let options = LoudnessOptions::new(frame_size, hop_size)
        .map_err(|error| error.to_string())?
        .gate_threshold_db(finite_f32(&input, "gateThresholdDb")?)
        .map_err(|error| error.to_string())?;
    let report =
        crate::analyze_loudness(&mono, sample_rate, options).map_err(|error| error.to_string())?;
    let frame_summary =
        summarize_feature_series(&report.frame_series).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "channels": channels,
        "analysisChannels": 1,
        "sampleCount": samples.len(),
        "analysisSampleCount": mono.len(),
        "peakDbfs": report.peak_dbfs,
        "rmsDbfs": report.rms_dbfs,
        "crestFactorDb": report.crest_factor_db,
        "approximateLufs": report.approximate_lufs,
        "approximation": "Lightweight RMS-gated LUFS-style estimate; not EBU R128 compliant.",
        "frameSeries": report.frame_series,
        "frameSummary": frame_summary
    }))
}

fn chain_summary_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let mut transforms = Vec::new();
    if input.get("gain").is_some() {
        transforms.push(serde_json::json!({"name": "gain", "linear": finite_f32(&input, "gain")?}));
    }
    if input
        .get("mono")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        transforms.push(serde_json::json!({"name": "mono", "mix": "average"}));
    }
    if input.get("clipMin").is_some() || input.get("clipMax").is_some() {
        transforms.push(serde_json::json!({
            "name": "hard_clip",
            "min": finite_f32(&input, "clipMin")?.unwrap_or(-1.0),
            "max": finite_f32(&input, "clipMax")?.unwrap_or(1.0)
        }));
    }
    if input.get("noiseGateThreshold").is_some() {
        transforms.push(serde_json::json!({
            "name": "noise_gate",
            "threshold": finite_f32(&input, "noiseGateThreshold")?,
            "attenuation": finite_f32(&input, "noiseGateAttenuation")?.unwrap_or(0.0)
        }));
    }
    Ok(serde_json::json!({
        "transformCount": transforms.len(),
        "transforms": transforms,
        "outputSampleFormat": "f32"
    }))
}

fn offline_edit_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let chain = input
        .get("chain")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "chain must be an array".to_string())?;
    let mut processor = OfflineAudioProcessor::new();
    for item in chain {
        let edit_type = string_field(item, "type", "")?;
        processor = match edit_type.as_str() {
            "trim" => processor.trim(
                finite_f64(item, "startSeconds")?.unwrap_or(0.0),
                finite_f64(item, "endSeconds")?
                    .ok_or_else(|| "trim requires endSeconds in offlineEdit chain".to_string())?,
            ),
            "reverse" => processor.reverse(),
            "fade" => processor.fade(FadeSpec {
                fade_in_seconds: finite_f64(item, "fadeInSeconds")?.unwrap_or(0.0),
                fade_out_seconds: finite_f64(item, "fadeOutSeconds")?.unwrap_or(0.0),
                curve: fade_curve(&string_field(item, "curve", "linear")?)?,
            }),
            "normalize" => processor.normalize(NormalizeSpec {
                target_peak: finite_f32(item, "targetPeak")?,
                target_rms: finite_f32(item, "targetRms")?,
            }),
            "insertSilence" | "insert_silence" => processor.insert_silence(
                finite_f64(item, "atSeconds")?.unwrap_or(0.0),
                finite_f64(item, "durationSeconds")?.unwrap_or(0.0),
            ),
            "delete" | "deleteSeconds" | "delete_seconds" => processor.delete_seconds(
                finite_f64(item, "startSeconds")?.unwrap_or(0.0),
                finite_f64(item, "endSeconds")?
                    .ok_or_else(|| "delete requires endSeconds in offlineEdit chain".to_string())?,
            ),
            "resample" => processor.resample(
                item.get("outputSampleRate")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| "resample requires outputSampleRate".to_string())?
                    .try_into()
                    .map_err(|_| "outputSampleRate must fit u32".to_string())?,
            ),
            "speed" => processor.speed(
                finite_f32(item, "factor")?.unwrap_or(1.0),
                speed_mode(&string_field(item, "mode", "resampleChangesPitch")?)?,
            ),
            "pitchShift" | "pitch_shift" => processor.pitch_shift(
                finite_f32(item, "semitones")?.unwrap_or(0.0),
                pitch_mode(&string_field(item, "mode", "preserveDurationOverlapAdd")?)?,
            ),
            other => return Err(format!("unsupported offline edit type `{other}`")),
        };
    }
    let clip = AudioClip::new(sample_rate, channels, samples).map_err(|error| error.to_string())?;
    let output = processor
        .process_clip(clip)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": output.sample_rate,
        "channels": output.channels,
        "sampleCount": output.samples.len(),
        "samplesPerChannel": output.samples_per_channel(),
        "durationSeconds": output.duration_seconds(),
        "rms": rms(&output.samples),
        "peak": peak(&output.samples),
        "samplePreview": preview(&output.samples, preview_limit(&input)?)
    }))
}

fn mixdown_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let placements = input
        .get("placements")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "placements must be an array".to_string())?;
    let mut parsed = Vec::with_capacity(placements.len());
    for placement in placements {
        let placement_sample_rate = placement
            .get("sampleRate")
            .and_then(serde_json::Value::as_u64)
            .map(|value| {
                u32::try_from(value)
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| "placement sampleRate must be a positive u32".to_string())
            })
            .transpose()?
            .unwrap_or(sample_rate);
        let placement_channels = placement
            .get("channels")
            .and_then(serde_json::Value::as_u64)
            .map(|value| {
                u16::try_from(value)
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| "placement channels must be a positive u16".to_string())
            })
            .transpose()?
            .unwrap_or(channels);
        let clip = AudioClip::new(
            placement_sample_rate,
            placement_channels,
            sample_array(placement, "samples")?,
        )
        .map_err(|error| error.to_string())?;
        parsed.push(
            AudioClipPlacement::new(
                clip,
                finite_f64(placement, "startSeconds")?.unwrap_or(0.0),
                finite_f32(placement, "gain")?.unwrap_or(1.0),
            )
            .map_err(|error| error.to_string())?,
        );
    }
    let output =
        mixdown_placements(&parsed, sample_rate, channels).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": output.sample_rate,
        "channels": output.channels,
        "sampleCount": output.samples.len(),
        "samplesPerChannel": output.samples_per_channel(),
        "durationSeconds": output.duration_seconds(),
        "rms": rms(&output.samples),
        "peak": peak(&output.samples),
        "samplePreview": preview(&output.samples, preview_limit(&input)?)
    }))
}

fn preset_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let preset_name = string_field(&input, "preset", "PodcastVoice")?;
    let preset = audio_preset(&preset_name)?;
    if input.get("samples").is_some() {
        let samples = sample_array(&input, "samples")?;
        let sample_rate = sample_rate(&input)?;
        let channels = channels(&input)?;
        apply_processor(input, samples, sample_rate, channels, preset_chain(preset))
    } else {
        Ok(serde_json::json!({
            "preset": preset_name,
            "availablePresets": ["VocalClean", "PodcastVoice", "LoFi", "WideChorus", "SmallRoomReverb", "HardLimiter"]
        }))
    }
}

fn eq_bands(input: &serde_json::Value) -> Result<Vec<EqBandSpec>, String> {
    let Some(bands) = input.get("bands") else {
        return Ok(vec![EqBandSpec {
            kind: EqBandKind::Peaking,
            frequency_hz: finite_f32(input, "frequencyHz")?.unwrap_or(1_000.0),
            q: finite_f32(input, "q")?.unwrap_or(1.0),
            gain_db: finite_f32(input, "gainDb")?.unwrap_or(0.0),
        }]);
    };
    bands
        .as_array()
        .ok_or_else(|| "eq bands must be an array".to_string())?
        .iter()
        .map(|band| {
            Ok(EqBandSpec {
                kind: eq_kind(&string_field(band, "kind", "peaking")?)?,
                frequency_hz: finite_f32(band, "frequencyHz")?.unwrap_or(1_000.0),
                q: finite_f32(band, "q")?.unwrap_or(1.0),
                gain_db: finite_f32(band, "gainDb")?.unwrap_or(0.0),
            })
        })
        .collect()
}

fn modulation_spec(input: &serde_json::Value) -> Result<ModulationDelaySpec, String> {
    Ok(ModulationDelaySpec {
        base_delay_ms: finite_f32(input, "baseDelayMs")?.unwrap_or(15.0),
        depth_ms: finite_f32(input, "depthMs")?.unwrap_or(5.0),
        rate_hz: finite_f32(input, "rateHz")?.unwrap_or(0.4),
        feedback: finite_f32(input, "feedback")?.unwrap_or(0.0),
        wet: finite_f32(input, "wet")?.unwrap_or(0.5),
        dry: finite_f32(input, "dry")?.unwrap_or(1.0),
    })
}

fn distortion_mode(value: &str) -> Result<DistortionMode, String> {
    match value {
        "hardClip" | "hard_clip" => Ok(DistortionMode::HardClip),
        "softClip" | "soft_clip" => Ok(DistortionMode::SoftClip),
        "tanh" => Ok(DistortionMode::Tanh),
        "foldback" => Ok(DistortionMode::Foldback),
        other => Err(format!("unsupported distortion mode `{other}`")),
    }
}

fn eq_kind(value: &str) -> Result<EqBandKind, String> {
    match value {
        "lowPass" | "low_pass" => Ok(EqBandKind::LowPass),
        "highPass" | "high_pass" => Ok(EqBandKind::HighPass),
        "bandPass" | "band_pass" => Ok(EqBandKind::BandPass),
        "notch" => Ok(EqBandKind::Notch),
        "peaking" => Ok(EqBandKind::Peaking),
        "lowShelf" | "low_shelf" => Ok(EqBandKind::LowShelf),
        "highShelf" | "high_shelf" => Ok(EqBandKind::HighShelf),
        other => Err(format!("unsupported eq kind `{other}`")),
    }
}

fn fade_curve(value: &str) -> Result<FadeCurve, String> {
    match value {
        "linear" => Ok(FadeCurve::Linear),
        "equalPower" | "equal_power" => Ok(FadeCurve::EqualPower),
        "exponential" => Ok(FadeCurve::Exponential),
        other => Err(format!("unsupported fade curve `{other}`")),
    }
}

fn speed_mode(value: &str) -> Result<SpeedMode, String> {
    match value {
        "resampleChangesPitch" | "resample_changes_pitch" => Ok(SpeedMode::ResampleChangesPitch),
        "preservePitchOverlapAdd" | "preserve_pitch_overlap_add" => {
            Ok(SpeedMode::PreservePitchOverlapAdd)
        }
        other => Err(format!("unsupported speed mode `{other}`")),
    }
}

fn pitch_mode(value: &str) -> Result<PitchShiftMode, String> {
    match value {
        "resampleChangesDuration" | "resample_changes_duration" => {
            Ok(PitchShiftMode::ResampleChangesDuration)
        }
        "preserveDurationOverlapAdd" | "preserve_duration_overlap_add" => {
            Ok(PitchShiftMode::PreserveDurationOverlapAdd)
        }
        other => Err(format!("unsupported pitch mode `{other}`")),
    }
}

fn audio_preset(value: &str) -> Result<AudioEffectPreset, String> {
    match value {
        "VocalClean" | "vocalClean" | "vocal_clean" => Ok(AudioEffectPreset::VocalClean),
        "PodcastVoice" | "podcastVoice" | "podcast_voice" => Ok(AudioEffectPreset::PodcastVoice),
        "LoFi" | "loFi" | "lo_fi" => Ok(AudioEffectPreset::LoFi),
        "WideChorus" | "wideChorus" | "wide_chorus" => Ok(AudioEffectPreset::WideChorus),
        "SmallRoomReverb" | "smallRoomReverb" | "small_room_reverb" => {
            Ok(AudioEffectPreset::SmallRoomReverb)
        }
        "HardLimiter" | "hardLimiter" | "hard_limiter" => Ok(AudioEffectPreset::HardLimiter),
        other => Err(format!("unsupported audio preset `{other}`")),
    }
}

fn sample_array(input: &serde_json::Value, field: &str) -> Result<Vec<f32>, String> {
    let values = input
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?;
    if values.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    if values.len() > MAX_SAMPLES {
        return Err(format!(
            "{field} must not contain more than {MAX_SAMPLES} samples"
        ));
    }
    values
        .iter()
        .map(|value| {
            let sample = value
                .as_f64()
                .ok_or_else(|| format!("{field} must contain only numbers"))?
                as f32;
            if sample.is_finite() {
                Ok(sample)
            } else {
                Err(format!("{field} must contain only finite numbers"))
            }
        })
        .collect()
}

fn sample_rate(input: &serde_json::Value) -> Result<u32, String> {
    let value = input
        .get("sampleRate")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(48_000);
    u32::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| "sampleRate must be a positive u32".to_string())
}

fn channels(input: &serde_json::Value) -> Result<u16, String> {
    let value = input
        .get("channels")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);
    u16::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| "channels must be a positive u16".to_string())
}

fn positive_usize(
    input: &serde_json::Value,
    field: &str,
    default_value: usize,
) -> Result<usize, String> {
    let value = input
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(default_value as u64);
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{field} must be positive"))
}

fn finite_f32(input: &serde_json::Value, field: &str) -> Result<Option<f32>, String> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    let value = value
        .as_f64()
        .ok_or_else(|| format!("{field} must be a number"))? as f32;
    if value.is_finite() {
        Ok(Some(value))
    } else {
        Err(format!("{field} must be finite"))
    }
}

fn finite_f64(input: &serde_json::Value, field: &str) -> Result<Option<f64>, String> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    let value = value
        .as_f64()
        .ok_or_else(|| format!("{field} must be a number"))?;
    if value.is_finite() {
        Ok(Some(value))
    } else {
        Err(format!("{field} must be finite"))
    }
}

fn string_field(
    input: &serde_json::Value,
    field: &str,
    default_value: &str,
) -> Result<String, String> {
    Ok(input
        .get(field)
        .and_then(serde_json::Value::as_str)
        .unwrap_or(default_value)
        .to_string())
}

fn preview_limit(input: &serde_json::Value) -> Result<usize, String> {
    let value = input
        .get("previewSamples")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(DEFAULT_PREVIEW_SAMPLES as u64);
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .map(|value| value.min(DEFAULT_PREVIEW_SAMPLES))
        .ok_or_else(|| "previewSamples must be positive".to_string())
}

fn preview(samples: &[f32], limit: usize) -> Vec<f32> {
    samples.iter().copied().take(limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_core::OperationId;

    #[test]
    fn package_surface_lists_processing_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.processing.apply"));
        assert!(ids.contains(&"audio.processing.energy"));
        assert!(ids.contains(&"audio.processing.effectsCatalog"));
        assert!(ids.contains(&"audio.processing.offlineEdit"));
        assert!(ids.contains(&"audio.processing.mixdown"));
        assert!(ids.contains(&"audio.processing.preset"));
        assert!(ids.contains(&"audio.processing.loudness"));
    }

    #[test]
    fn energy_operation_returns_summary() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.processing.energy"),
            input: serde_json::json!({"samples": [0.0, 1.0, -1.0], "sampleRate": 3, "channels": 1}),
        })
        .expect("energy");
        assert_eq!(response.value["operation"], "audio.processing.energy");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert!(response.value["rms"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn loudness_operation_returns_finite_report() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.processing.loudness"),
            input: serde_json::json!({
                "samples": [0.0, 1.0, -1.0, 0.0],
                "sampleRate": 4,
                "channels": 1,
                "frameSize": 2,
                "hopSize": 1
            }),
        })
        .expect("loudness");
        assert_eq!(response.value["operation"], "audio.processing.loudness");
        assert!(response.value["peakDbfs"].as_f64().unwrap().is_finite());
        assert!(response.value["rmsDbfs"].as_f64().unwrap().is_finite());
        assert!(response.value["approximateLufs"]
            .as_f64()
            .unwrap()
            .is_finite());
        assert_eq!(response.value["frameSummary"]["frame_count"], 3);
    }

    #[test]
    fn example_requests_run_with_structured_outputs() {
        for operation in package_surface().operations {
            let response = run_surface_operation(SurfaceRequest {
                operation: operation.id.clone(),
                input: operation.example_request.clone(),
            })
            .unwrap_or_else(|error| panic!("{} example failed: {error}", operation.id.as_str()));
            assert_eq!(response.value["operation"], operation.id.as_str());
            assert!(response.value["title"].is_string());
            assert!(response.value["summary"].is_object());
            assert!(response.value["result"].is_object());
        }
    }

    #[test]
    fn invalid_samples_return_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.processing.energy"),
            input: serde_json::json!({"samples": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("samples"));
    }

    #[test]
    fn apply_accepts_chain_effects() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.processing.apply"),
            input: serde_json::json!({
                "samples": [0.0, 0.5, -0.5, 0.25],
                "sampleRate": 48000,
                "channels": 1,
                "chain": [
                    {"type": "gain", "linear": 0.8},
                    {"type": "distortion", "mode": "tanh", "driveDb": 6.0, "mix": 0.5},
                    {"type": "delay", "delaySeconds": 0.001, "feedback": 0.2, "wet": 0.3, "dry": 1.0},
                    {"type": "compressor", "thresholdDb": -18.0, "ratio": 3.0}
                ]
            }),
        })
        .expect("processing chain");
        assert_eq!(response.value["sampleRate"], 48_000);
        assert_eq!(response.value["channels"], 1);
    }

    #[test]
    fn offline_edit_handles_core_edits() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.processing.offlineEdit"),
            input: serde_json::json!({
                "samples": [0.0, 0.25, 0.5, 1.0, 0.5, 0.25],
                "sampleRate": 6,
                "channels": 1,
                "chain": [
                    {"type": "reverse"},
                    {"type": "insertSilence", "atSeconds": 0.5, "durationSeconds": 0.1666667},
                    {"type": "delete", "startSeconds": 0.5, "endSeconds": 0.6666667},
                    {"type": "fade", "fadeInSeconds": 0.1666667, "fadeOutSeconds": 0.1666667},
                    {"type": "normalize", "targetPeak": 1.0},
                    {"type": "speed", "factor": 2.0, "mode": "resampleChangesPitch"}
                ]
            }),
        })
        .expect("offline edit");
        assert_eq!(response.value["channels"], 1);
        assert!(response.value["sampleCount"].as_u64().unwrap() < 6);
    }

    #[test]
    fn mixdown_operation_places_clips_on_timeline() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.processing.mixdown"),
            input: serde_json::json!({
                "sampleRate": 4,
                "channels": 1,
                "placements": [
                    {"samples": [1.0, 1.0], "startSeconds": 0.0, "gain": 1.0},
                    {"samples": [0.5, 0.5], "startSeconds": 0.25, "gain": 1.0}
                ],
                "previewSamples": 4
            }),
        })
        .expect("mixdown");
        assert_eq!(
            response.value["samplePreview"],
            serde_json::json!([1.0, 1.5, 0.5])
        );
    }

    #[test]
    fn catalog_and_preset_operations_work() {
        let catalog = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.processing.effectsCatalog"),
            input: serde_json::json!({}),
        })
        .expect("catalog");
        assert!(catalog.value["streamingEffects"].as_array().unwrap().len() >= 10);

        let preset = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.processing.preset"),
            input: serde_json::json!({"preset": "HardLimiter", "samples": [2.0], "sampleRate": 48000, "channels": 1}),
        })
        .expect("preset");
        assert_eq!(preset.value["sampleCount"], 1);
    }
}
