//! Library-owned runtime surface for `audio-analysis-io`.

use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{build_ffmpeg_audio_filter_chain, FfmpegAudioEditSpec, FfmpegAudioEffect};

const MAX_SAMPLES: usize = 192_000;

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Describe package",
                "Audio input planning, waveform batch contracts, and file-oriented audio helpers.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.io.inputPlan",
                "Preview input plan",
                "Previews the audio input that would be opened without scanning files or touching the filesystem.",
                serde_json::json!({"source": "clip.wav", "mode": "recorded", "samplesPerChunk": 4096}),
            ),
            operation(
                "audio.io.waveformBatchSummary",
                "Waveform batch summary",
                "Summarizes an in-memory batch/channel/time waveform tensor.",
                serde_json::json!({"sampleRate": 48000, "waveforms": [[[0.0, 0.5, -0.5]]]}),
            ),
            operation(
                "audio.io.wavSummary",
                "WAV summary",
                "Summarizes inline WAV-like samples or previews a native/server WAV path read without invoking FFmpeg.",
                serde_json::json!({"sampleRate": 48000, "channels": 1, "samples": [0.0, 0.5, -0.5]}),
            ),
            operation(
                "audio.io.probePlan",
                "Preview probe plan",
                "Explains whether a probe would need FFprobe without executing it.",
                serde_json::json!({"source": "clip.wav", "format": "wav"}),
            ),
            operation(
                "audio.io.decodePlan",
                "Preview decode plan",
                "Previews deterministic decode settings and backend requirements without decoding audio.",
                serde_json::json!({"source": "clip.wav", "target": "mono-f32", "sampleRate": 48000}),
            ),
            operation(
                "audio.io.editPlan",
                "Preview edit plan",
                "Previews a deterministic file edit plan without editing media or executing FFmpeg.",
                serde_json::json!({"input": "clip.wav", "output": "out.wav", "edit": {"speedFactor": 1.25, "effects": [{"type": "normalize"}]}}),
            ),
            operation(
                "audio.io.splitPlan",
                "Preview split plan",
                "Previews deterministic split output paths without splitting media or touching the filesystem.",
                serde_json::json!({"input": "clip.wav", "outputDir": "segments", "segments": [{"startSeconds": 0.0, "endSeconds": 1.0}], "outputFormat": "wav"}),
            ),
            operation(
                "audio.io.joinPlan",
                "Preview join plan",
                "Previews deterministic join settings without joining media or touching the filesystem.",
                serde_json::json!({"inputs": ["a.wav", "b.wav"], "output": "joined.wav", "crossfadeSeconds": 0.05}),
            ),
            operation(
                "audio.io.ffmpegFilterPlan",
                "Preview FFmpeg filter plan",
                "Previews the FFmpeg audio filter chain for an edit spec without executing FFmpeg.",
                serde_json::json!({"speedFactor": 1.25, "pitchShiftSemitones": 2.0, "effects": [{"type": "compressor", "thresholdDb": -18.0, "ratio": 3.0}]}),
            ),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true, "xOperationCategory": runtime_core::operation_category(id)}),
        output_schema: serde_json::json!({"type": "object", "xOperationCategory": runtime_core::operation_category(id)}),
        example_request,
        wasm_supported: true,
        server_supported: true,
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "audio.io.inputPlan" => input_plan_value(request.input)?,
        "audio.io.waveformBatchSummary" => waveform_batch_summary_value(request.input)?,
        "audio.io.wavSummary" => wav_summary_value(request.input)?,
        "audio.io.probePlan" => probe_plan_value(request.input)?,
        "audio.io.decodePlan" => decode_plan_value(request.input)?,
        "audio.io.editPlan" => edit_plan_value(request.input)?,
        "audio.io.splitPlan" => split_plan_value(request.input)?,
        "audio.io.joinPlan" => join_plan_value(request.input)?,
        "audio.io.ffmpegFilterPlan" => ffmpeg_filter_plan_value(request.input)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ));
        }
    };
    Ok(response(operation, value))
}

fn response(operation: OperationId, value: serde_json::Value) -> SurfaceResponse {
    let (title, message, summary) = match operation.as_str() {
        "describe" => (
            "Audio IO package metadata",
            "Inspected the audio IO planning and waveform summary operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.io.waveformBatchSummary" => (
            "Waveform batch summary",
            "Summarized an in-memory batch/channel/time waveform tensor.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "batchSize": value.get("batchSize").cloned().unwrap_or(serde_json::Value::Null),
                "totalSamples": value.get("totalSamples").cloned().unwrap_or(serde_json::Value::Null),
                "durationSeconds": value.get("durationSeconds").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.io.inputPlan" => (
            "Audio input plan preview",
            "Previewed input settings only; this operation does not scan files, open media, or run FFmpeg.",
            serde_json::json!({
                "backend": value.get("backend").cloned().unwrap_or(serde_json::Value::Null),
                "executed": false
            }),
        ),
        "audio.io.wavSummary" => (
            "WAV summary",
            "Summarized inline WAV-like samples or previewed a native/server WAV path read.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "channels": value.get("channels").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "executed": value.get("executed").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.io.probePlan" => (
            "Audio probe plan preview",
            "Planned audio probing only; this operation does not execute FFprobe.",
            serde_json::json!({
                "backend": value.get("backend").cloned().unwrap_or(serde_json::Value::Null),
                "requiresExternalTool": value.get("requiresExternalTool").cloned().unwrap_or(serde_json::Value::Null),
                "executed": value.get("executed").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.io.decodePlan" => (
            "Audio decode plan preview",
            "Previewed decode settings only; this operation does not decode audio or run FFmpeg.",
            serde_json::json!({
                "backend": value.get("backend").cloned().unwrap_or(serde_json::Value::Null),
                "requiresExternalTool": value.get("requiresExternalTool").cloned().unwrap_or(serde_json::Value::Null),
                "executed": value.get("executed").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.io.editPlan" => (
            "Audio edit plan preview",
            "Previewed edit settings only; this operation does not edit media or run FFmpeg.",
            serde_json::json!({
                "backend": value.get("backend").cloned().unwrap_or(serde_json::Value::Null),
                "executed": value.get("executed").cloned().unwrap_or(serde_json::Value::Null),
                "hasFilterChain": value.get("filterChain").is_some()
            }),
        ),
        "audio.io.splitPlan" => (
            "Audio split plan preview",
            "Previewed split output paths only; this operation does not split files or touch the filesystem.",
            serde_json::json!({
                "backend": value.get("backend").cloned().unwrap_or(serde_json::Value::Null),
                "executed": value.get("executed").cloned().unwrap_or(serde_json::Value::Null),
                "segmentCount": value.get("segments").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.io.joinPlan" => (
            "Audio join plan preview",
            "Previewed join settings only; this operation does not join files or run FFmpeg.",
            serde_json::json!({
                "backend": value.get("backend").cloned().unwrap_or(serde_json::Value::Null),
                "executed": value.get("executed").cloned().unwrap_or(serde_json::Value::Null),
                "inputCount": value.get("inputs").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.io.ffmpegFilterPlan" => (
            "FFmpeg filter plan preview",
            "Built a filter-chain preview only; this operation does not execute FFmpeg.",
            serde_json::json!({
                "backend": value.get("backend").cloned().unwrap_or(serde_json::Value::Null),
                "executed": value.get("executed").cloned().unwrap_or(serde_json::Value::Null),
                "hasFilterChain": value.get("filterChain").is_some()
            }),
        ),
        _ => (
            "Audio IO operation result",
            "Completed the audio IO package surface operation.",
            serde_json::json!({}),
        ),
    };
    structured_surface_response(operation, title, message, summary, value)
}

fn describe_value(input: serde_json::Value) -> serde_json::Value {
    let surface = package_surface();
    serde_json::json!({
        "library": surface.library,
        "version": surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface.operations.iter().map(|operation| operation.id.as_str()).collect::<Vec<_>>(),
        "input": input
    })
}

fn input_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let source = string_field(&input, "source", "stdin")?;
    let mode = string_field(&input, "mode", "recorded")?;
    let samples_per_chunk = positive_u64(&input, "samplesPerChunk", 4096)?;
    Ok(serde_json::json!({
        "source": source,
        "mode": mode,
        "samplesPerChunk": samples_per_chunk,
        "realtime": mode == "live",
        "backend": "ffmpeg",
        "opensExternalProcess": false,
        "executed": false,
        "doesNot": ["scan files", "open media", "run FFmpeg"],
        "notes": ["plan only", "no filesystem or FFmpeg access during surface execution"]
    }))
}

fn waveform_batch_summary_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let sample_rate = positive_u64(&input, "sampleRate", 48_000)?;
    let batches = input
        .get("waveforms")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "waveforms must be a batch array".to_string())?;
    let mut total_samples = 0usize;
    let mut batch_shapes = Vec::new();
    for batch in batches {
        let channels = batch
            .as_array()
            .ok_or_else(|| "each waveform batch item must be an array of channels".to_string())?;
        let mut channel_lengths = Vec::new();
        for channel in channels {
            let samples = sample_array(channel)?;
            total_samples += samples.len();
            if total_samples > MAX_SAMPLES {
                return Err(format!(
                    "waveforms must not contain more than {MAX_SAMPLES} samples"
                ));
            }
            channel_lengths.push(samples.len());
        }
        batch_shapes.push(serde_json::json!({
            "channels": channels.len(),
            "timeSteps": channel_lengths.into_iter().max().unwrap_or(0)
        }));
    }
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "batchSize": batches.len(),
        "totalSamples": total_samples,
        "durationSeconds": if sample_rate == 0 { 0.0 } else { total_samples as f64 / sample_rate as f64 },
        "batches": batch_shapes
    }))
}

fn wav_summary_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    if input.get("paths").is_some() {
        return Err(
            "wavSummary accepts a single inline sample set or one path, not multi-file batches"
                .to_string(),
        );
    }
    if let Some(path) = input.get("path").and_then(serde_json::Value::as_str) {
        return Ok(serde_json::json!({
            "source": path,
            "sourceKind": "path",
            "format": "wav",
            "backend": "hound",
            "requiresExternalTool": false,
            "requiresFilesystem": true,
            "serverSupported": true,
            "wasmSupportedForPath": false,
            "executed": false,
            "doesNot": ["read files", "run FFmpeg", "run FFprobe"],
            "diagnostics": ["Path WAV reads are available through native/server library APIs such as read_wav_as_clip."]
        }));
    }
    let sample_rate = positive_u64(&input, "sampleRate", 48_000)?;
    let channels = positive_u64(&input, "channels", 1)?;
    let samples = input
        .get("samples")
        .map(sample_array)
        .transpose()?
        .ok_or_else(|| "samples are required for inline wavSummary".to_string())?;
    if !samples.len().is_multiple_of(channels as usize) {
        return Err("samples length must be divisible by channels".to_string());
    }
    if samples.len() > MAX_SAMPLES {
        return Err(format!(
            "samples must not contain more than {MAX_SAMPLES} values"
        ));
    }
    let frames = samples.len() / channels as usize;
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    Ok(serde_json::json!({
        "sourceKind": "inline",
        "format": "wav",
        "backend": "hound-compatible",
        "requiresExternalTool": false,
        "requiresFilesystem": false,
        "executed": true,
        "sampleRate": sample_rate,
        "channels": channels,
        "sampleCount": samples.len(),
        "frames": frames,
        "durationSeconds": frames as f64 / sample_rate as f64,
        "peak": peak
    }))
}

fn probe_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let source = string_field(&input, "source", "clip.wav")?;
    let format = string_field(&input, "format", "")?;
    let wav_like = source.ends_with(".wav") || format == "wav";
    Ok(serde_json::json!({
        "source": source,
        "formatHint": if format.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(format) },
        "backend": if wav_like { "hound metadata or ffprobe" } else { "ffprobe" },
        "requiresExternalTool": !wav_like,
        "executed": false,
        "doesNot": ["open media", "read files", "run FFprobe"],
        "supportedNativeApis": ["probe_audio_input_metadata", "read_wav_as_clip"],
        "diagnostics": if wav_like {
            vec!["WAV metadata can be read through the pure hound path in native/server code."]
        } else {
            vec!["Non-WAV probing requires FFprobe through the optional FFmpeg-backed helpers."]
        }
    }))
}

fn decode_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let source = string_field(&input, "source", "input")?;
    let target = string_field(&input, "target", "waveform-batch")?;
    Ok(serde_json::json!({
        "source": source,
        "target": target,
        "requestedSampleRate": input.get("sampleRate").and_then(serde_json::Value::as_u64),
        "backend": "ffmpeg",
        "requiresExternalTool": true,
        "executed": false,
        "doesNot": ["decode audio", "open media", "run FFmpeg"],
        "supportedTargets": ["f32", "mono-f32", "waveform-batch", "wav"]
    }))
}

fn edit_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let source = string_field(&input, "input", "input.wav")?;
    let output = string_field(&input, "output", "output.wav")?;
    let edit_input = input.get("edit").unwrap_or(&input);
    let spec = ffmpeg_edit_spec(edit_input)?;
    let filter_chain = build_ffmpeg_audio_filter_chain(&spec).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "input": source,
        "output": output,
        "backend": "ffmpeg",
        "requiresExternalTool": true,
        "executed": false,
        "doesNot": ["edit media", "write files", "run FFmpeg"],
        "filterChain": filter_chain,
        "outputSampleRate": spec.output_sample_rate,
        "outputChannels": spec.output_channels
    }))
}

fn split_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let source = string_field(&input, "input", "input.wav")?;
    let output_dir = string_field(&input, "outputDir", "segments")?;
    let output_format = string_field(&input, "outputFormat", "wav")?;
    let extension = audio_extension(&output_format)?;
    let segments = input
        .get("segments")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "segments must be an array".to_string())?;
    let mut outputs = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let start = finite_f64(segment, "startSeconds")?.unwrap_or(0.0);
        let end = finite_f64(segment, "endSeconds")?
            .ok_or_else(|| "segment endSeconds is required".to_string())?;
        if start < 0.0 || end <= start {
            return Err("segment start/end must be non-negative and ordered".to_string());
        }
        outputs.push(serde_json::json!({
            "index": index,
            "input": source,
            "startSeconds": start,
            "endSeconds": end,
            "output": format!("{output_dir}/segment_{index:03}.{extension}")
        }));
    }
    Ok(serde_json::json!({
        "backend": "ffmpeg",
        "requiresExternalTool": true,
        "executed": false,
        "doesNot": ["split media", "write files", "run FFmpeg"],
        "outputFormat": output_format,
        "segments": outputs
    }))
}

fn join_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let inputs = input
        .get("inputs")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "inputs must be an array".to_string())?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| "join inputs must be strings".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if inputs.is_empty() {
        return Err("join requires at least one input".to_string());
    }
    let output = string_field(&input, "output", "joined.wav")?;
    let crossfade = finite_f64(&input, "crossfadeSeconds")?;
    Ok(serde_json::json!({
        "inputs": inputs,
        "output": output,
        "crossfadeSeconds": crossfade,
        "backend": "ffmpeg",
        "requiresExternalTool": true,
        "executed": false,
        "doesNot": ["join media", "write files", "run FFmpeg"],
        "filter": if crossfade.unwrap_or(0.0) > 0.0 { "acrossfade" } else { "concat" }
    }))
}

fn ffmpeg_filter_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let spec = ffmpeg_edit_spec(&input)?;
    let filter_chain = build_ffmpeg_audio_filter_chain(&spec).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "filterChain": filter_chain,
        "backend": "ffmpeg",
        "requiresExternalTool": true,
        "executed": false,
        "doesNot": ["open media", "write files", "run FFmpeg"]
    }))
}

fn ffmpeg_edit_spec(input: &serde_json::Value) -> Result<FfmpegAudioEditSpec, String> {
    let speed_factor = finite_f32(input, "speedFactor")?;
    let pitch_shift_semitones = finite_f32(input, "pitchShiftSemitones")?;
    let output_sample_rate = input
        .get("outputSampleRate")
        .and_then(serde_json::Value::as_u64)
        .map(|value| u32::try_from(value).map_err(|_| "outputSampleRate must fit u32".to_string()))
        .transpose()?;
    let output_channels = input
        .get("outputChannels")
        .and_then(serde_json::Value::as_u64)
        .map(|value| u16::try_from(value).map_err(|_| "outputChannels must fit u16".to_string()))
        .transpose()?;
    let effects = input
        .get("effects")
        .and_then(serde_json::Value::as_array)
        .map(|effects| {
            effects
                .iter()
                .map(ffmpeg_effect)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    Ok(FfmpegAudioEditSpec {
        speed_factor,
        pitch_shift_semitones,
        effects,
        output_sample_rate,
        output_channels,
    })
}

fn ffmpeg_effect(input: &serde_json::Value) -> Result<FfmpegAudioEffect, String> {
    let effect_type = string_field(input, "type", "")?;
    Ok(match effect_type.as_str() {
        "reverse" => FfmpegAudioEffect::Reverse,
        "trim" => FfmpegAudioEffect::Trim {
            start_seconds: finite_f64(input, "startSeconds")?.unwrap_or(0.0),
            end_seconds: finite_f64(input, "endSeconds")?
                .ok_or_else(|| "trim effect requires endSeconds".to_string())?,
        },
        "fade" => FfmpegAudioEffect::Fade {
            fade_in_seconds: finite_f64(input, "fadeInSeconds")?.unwrap_or(0.0),
            fade_out_seconds: finite_f64(input, "fadeOutSeconds")?.unwrap_or(0.0),
            duration_seconds: finite_f64(input, "durationSeconds")?,
        },
        "delay" | "echo" => FfmpegAudioEffect::Echo {
            in_gain: finite_f32(input, "inGain")?.unwrap_or(0.8),
            out_gain: finite_f32(input, "outGain")?.unwrap_or(0.9),
            delay_seconds: finite_f64(input, "delaySeconds")?.unwrap_or(0.25),
            decay: finite_f32(input, "decay")?
                .or_else(|| finite_f32(input, "feedback").ok().flatten())
                .unwrap_or(0.35),
        },
        "reverb" => FfmpegAudioEffect::Reverb {
            room_size: finite_f32(input, "roomSize")?.unwrap_or(0.5),
            wet: finite_f32(input, "wet")?.unwrap_or(0.3),
        },
        "compressor" => FfmpegAudioEffect::Compressor {
            threshold_db: finite_f32(input, "thresholdDb")?.unwrap_or(-18.0),
            ratio: finite_f32(input, "ratio")?.unwrap_or(3.0),
            attack_ms: finite_f32(input, "attackMs")?.unwrap_or(20.0),
            release_ms: finite_f32(input, "releaseMs")?.unwrap_or(250.0),
        },
        "limiter" => FfmpegAudioEffect::Limiter {
            ceiling_db: finite_f32(input, "ceilingDb")?.unwrap_or(-1.0),
        },
        "eq" => FfmpegAudioEffect::Eq {
            frequency_hz: finite_f32(input, "frequencyHz")?.unwrap_or(1_000.0),
            width_q: finite_f32(input, "q")?.unwrap_or(1.0),
            gain_db: finite_f32(input, "gainDb")?.unwrap_or(0.0),
        },
        "lowpass" | "lowPass" => FfmpegAudioEffect::LowPass {
            frequency_hz: finite_f32(input, "frequencyHz")?.unwrap_or(8_000.0),
        },
        "highpass" | "highPass" => FfmpegAudioEffect::HighPass {
            frequency_hz: finite_f32(input, "frequencyHz")?.unwrap_or(80.0),
        },
        "chorus" => FfmpegAudioEffect::Chorus,
        "flanger" => FfmpegAudioEffect::Flanger,
        "tremolo" => FfmpegAudioEffect::Tremolo {
            frequency_hz: finite_f32(input, "frequencyHz")?
                .or_else(|| finite_f32(input, "rateHz").ok().flatten())
                .unwrap_or(5.0),
            depth: finite_f32(input, "depth")?.unwrap_or(0.5),
        },
        "normalize" => FfmpegAudioEffect::Normalize,
        other => return Err(format!("unsupported FFmpeg audio effect `{other}`")),
    })
}

fn sample_array(value: &serde_json::Value) -> Result<Vec<f32>, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "sample channel must be an array".to_string())?;
    if values.is_empty() {
        return Err("sample channel must not be empty".to_string());
    }
    values
        .iter()
        .map(|sample| {
            let sample = sample
                .as_f64()
                .ok_or_else(|| "samples must be numbers".to_string())?
                as f32;
            if sample.is_finite() {
                Ok(sample)
            } else {
                Err("samples must be finite".to_string())
            }
        })
        .collect()
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

fn audio_extension(format: &str) -> Result<&'static str, String> {
    match format {
        "wav" => Ok("wav"),
        "mp3" => Ok("mp3"),
        "flac" => Ok("flac"),
        "m4a" | "aac" => Ok("m4a"),
        "ogg" => Ok("ogg"),
        other => Err(format!("unsupported audio output format `{other}`")),
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

fn positive_u64(input: &serde_json::Value, field: &str, default_value: u64) -> Result<u64, String> {
    input
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(default_value)
        .checked_add(0)
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{field} must be positive"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_io_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.io.inputPlan"));
        assert!(ids.contains(&"audio.io.waveformBatchSummary"));
        assert!(ids.contains(&"audio.io.wavSummary"));
        assert!(ids.contains(&"audio.io.probePlan"));
        assert!(ids.contains(&"audio.io.ffmpegFilterPlan"));
        assert!(ids.contains(&"audio.io.splitPlan"));
        assert!(ids.contains(&"audio.io.joinPlan"));
    }

    #[test]
    fn input_plan_operation_returns_backend() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.io.inputPlan"),
            input: serde_json::json!({"source": "clip.wav"}),
        })
        .expect("input plan");
        assert_eq!(response.value["operation"], "audio.io.inputPlan");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert_eq!(response.value["backend"], "ffmpeg");
        assert_eq!(response.value["opensExternalProcess"], false);
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
    fn invalid_waveforms_return_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.io.waveformBatchSummary"),
            input: serde_json::json!({"waveforms": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("waveforms"));
    }

    #[test]
    fn wav_summary_handles_inline_samples_and_path_plans() {
        let inline = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.io.wavSummary"),
            input: serde_json::json!({"sampleRate": 4, "channels": 1, "samples": [0.0, 0.5, -0.5, 0.0]}),
        })
        .expect("inline wav summary");
        assert_eq!(inline.value["operation"], "audio.io.wavSummary");
        assert_eq!(inline.value["executed"], true);
        assert_eq!(inline.value["durationSeconds"], 1.0);
        assert_eq!(inline.value["peak"], 0.5);

        let path_plan = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.io.wavSummary"),
            input: serde_json::json!({"path": "clip.wav"}),
        })
        .expect("path wav summary plan");
        assert_eq!(path_plan.value["executed"], false);
        assert_eq!(path_plan.value["wasmSupportedForPath"], false);
    }

    #[test]
    fn probe_plan_reports_ffprobe_requirement() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.io.probePlan"),
            input: serde_json::json!({"source": "clip.mp3"}),
        })
        .expect("probe plan");
        assert_eq!(response.value["operation"], "audio.io.probePlan");
        assert_eq!(response.value["requiresExternalTool"], true);
        assert_eq!(response.value["executed"], false);
    }

    #[test]
    fn ffmpeg_filter_plan_lists_expected_filters() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.io.ffmpegFilterPlan"),
            input: serde_json::json!({
                "speedFactor": 2.5,
                "pitchShiftSemitones": 2.0,
                "effects": [
                    {"type": "eq", "frequencyHz": 1000.0, "q": 1.0, "gainDb": 3.0},
                    {"type": "compressor", "thresholdDb": -18.0, "ratio": 3.0},
                    {"type": "limiter", "ceilingDb": -1.0},
                    {"type": "echo", "delaySeconds": 0.25, "feedback": 0.3},
                    {"type": "chorus"},
                    {"type": "flanger"},
                    {"type": "tremolo", "rateHz": 5.0, "depth": 0.5}
                ]
            }),
        })
        .expect("filter plan");
        let filter = response.value["filterChain"].as_str().unwrap();
        for expected in [
            "atempo",
            "equalizer",
            "acompressor",
            "alimiter",
            "aecho",
            "chorus",
            "flanger",
            "tremolo",
        ] {
            assert!(filter.contains(expected), "missing {expected} in {filter}");
        }
    }

    #[test]
    fn split_join_and_edit_plans_are_preview_safe() {
        let split = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.io.splitPlan"),
            input: serde_json::json!({
                "input": "clip.wav",
                "outputDir": "segments",
                "segments": [{"startSeconds": 0.0, "endSeconds": 1.0}],
                "outputFormat": "wav"
            }),
        })
        .expect("split plan");
        assert_eq!(split.value["executed"], false);
        assert_eq!(
            split.value["segments"][0]["output"],
            "segments/segment_000.wav"
        );

        let join = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.io.joinPlan"),
            input: serde_json::json!({"inputs": ["a.wav", "b.wav"], "output": "joined.wav"}),
        })
        .expect("join plan");
        assert_eq!(join.value["filter"], "concat");

        let edit = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.io.editPlan"),
            input: serde_json::json!({"input": "a.wav", "output": "b.wav", "edit": {"effects": [{"type": "normalize"}]}}),
        })
        .expect("edit plan");
        assert_eq!(edit.value["executed"], false);
        assert!(edit.value["filterChain"]
            .as_str()
            .unwrap()
            .contains("loudnorm"));
    }
}
