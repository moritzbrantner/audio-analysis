//! Library-owned runtime surface for `audio-analysis-core`.

use runtime_core::{
    describe_surface_response, set_surface_operation_curation, structured_surface_response,
    surface_operation, PackageSurface, RuntimeCapabilities, SurfaceError, SurfaceOperation,
    SurfaceOperationCuration, SurfaceRequest, SurfaceResponse,
};

use crate::{
    mean_absolute, peak, rms, samples_to_seconds, seconds_to_samples, summarize_feature_series,
    windowed_level_series, zero_crossing_rate, FrameSpec, WindowFunction,
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
            operation(
                "describe",
                "Describe package",
                "Shared audio frame conversion, windowing, and streaming helpers for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.levels",
                "Audio levels",
                "Returns deterministic level metrics for normalized audio samples.",
                serde_json::json!({"samples": [0.0, 0.5, -0.5], "sampleRate": 48000, "channels": 1}),
            ),
            operation(
                "audio.frames",
                "Audio frames",
                "Summarizes fixed-size analysis frames over normalized samples.",
                serde_json::json!({"samples": [0.0, 0.5, -0.5, 0.25], "sampleRate": 48000, "channels": 1, "frameSize": 2, "hopSize": 1}),
            ),
            operation(
                "audio.timestamps",
                "Audio timestamps",
                "Converts between seconds, samples, and timestamp ticks for a sample rate.",
                serde_json::json!({"sampleRate": 48000, "seconds": 1.5, "samplesCount": 72000}),
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
    let mut operation = surface_operation(id, name, description, example_request);
    let curation = match id {
        "audio.levels" => SurfaceOperationCuration::workflow(10).primary(),
        "audio.frames" => SurfaceOperationCuration::workflow(20),
        "audio.timestamps" => SurfaceOperationCuration::debug(910),
        "describe" => SurfaceOperationCuration::debug(900),
        _ => SurfaceOperationCuration::from_operation_id(id),
    };
    set_surface_operation_curation(&mut operation, curation);
    if id == "audio.levels" {
        runtime_core::attach_landscape_contract(
            &mut operation,
            runtime_core::landscape::LandscapeOperationContract::new(
                runtime_core::landscape::LandscapeFunction::new(
                    "audio.core.levels",
                    env!("CARGO_PKG_NAME"),
                )
                .input(runtime_core::landscape::LandscapePort::new(
                    "frame",
                    runtime_core::landscape::well_known::audio_frame(),
                ))
                .output(runtime_core::landscape::LandscapePort::new(
                    "summary",
                    runtime_core::landscape::well_known::numbers_summary(),
                )),
            ),
        );
    }
    operation
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&package_surface(), request)),
        "audio.levels" => levels_value(request.input)?,
        "audio.frames" => frames_value(request.input)?,
        "audio.timestamps" => timestamps_value(request.input)?,
        operation => {
            return Err(
                SurfaceError::unsupported_operation(operation, env!("CARGO_PKG_NAME"))
                    .to_error_string(),
            );
        }
    };
    Ok(response(operation, value))
}

fn response(operation: runtime_core::OperationId, value: serde_json::Value) -> SurfaceResponse {
    let (title, message, summary) = match operation.as_str() {
        "audio.levels" => (
            "Audio level metrics",
            "Computed deterministic RMS, peak, and mean absolute level metrics for normalized audio samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "channels": value.get("channels").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "durationSeconds": value.get("durationSeconds").cloned().unwrap_or(serde_json::Value::Null),
                "rms": value.get("rms").cloned().unwrap_or(serde_json::Value::Null),
                "peak": value.get("peak").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.frames" => (
            "Audio frame summaries",
            "Segmented normalized samples into fixed-size analysis frames and summarized each previewed frame.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "channels": value.get("channels").cloned().unwrap_or(serde_json::Value::Null),
                "frameSize": value.get("frameSize").cloned().unwrap_or(serde_json::Value::Null),
                "hopSize": value.get("hopSize").cloned().unwrap_or(serde_json::Value::Null),
                "frameCount": value.get("frameCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.timestamps" => (
            "Audio timestamp conversion",
            "Converted between seconds, sample counts, and timestamp ticks for the requested sample rate.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "seconds": value.get("seconds").cloned().unwrap_or(serde_json::Value::Null),
                "samplesCount": value.get("samplesCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        _ => (
            "Audio operation result",
            "Completed the audio package surface operation.",
            serde_json::json!({}),
        ),
    };
    structured_surface_response(operation, title, message, summary, value)
}

fn levels_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let samples_per_channel = samples_per_channel(samples.len(), channels)?;
    let frame_size = positive_usize(
        &input,
        "frameSize",
        samples_per_channel.clamp(1, DEFAULT_PREVIEW_SAMPLES),
    )?;
    let hop_size = positive_usize(&input, "hopSize", frame_size)?;
    let series = windowed_level_series(
        &samples,
        sample_rate,
        FrameSpec::new(frame_size, hop_size).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let summary = summarize_feature_series(&series).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "channels": channels,
        "sampleCount": samples.len(),
        "samplesPerChannel": samples_per_channel,
        "durationSeconds": samples_to_seconds(samples_per_channel as u64, sample_rate).map_err(|error| error.to_string())?,
        "rms": rms(&samples),
        "peak": peak(&samples),
        "meanAbsolute": mean_absolute(&samples),
        "zeroCrossingRate": zero_crossing_rate(&samples),
        "featureSeries": series,
        "featureSummary": summary,
        "samplePreview": preview(&samples, input_limit(&input, "previewSamples", DEFAULT_PREVIEW_SAMPLES)?)
    }))
}

fn frames_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let frame_size = positive_usize(&input, "frameSize", 1024)?;
    let hop_size = positive_usize(&input, "hopSize", frame_size)?;
    let frame_spec = FrameSpec::new(frame_size, hop_size).map_err(|error| error.to_string())?;
    let window = window_name(input.get("window").and_then(serde_json::Value::as_str));
    let summaries = frame_spec
        .frames(&samples)
        .take(input_limit(&input, "maxFrames", 32)?)
        .map(|(start_sample, frame)| {
            let windowed = window.apply(frame);
            serde_json::json!({
                "startSample": start_sample,
                "len": frame.len(),
                "rms": rms(frame),
                "peak": peak(frame),
                "meanAbsolute": mean_absolute(frame),
                "zeroCrossingRate": zero_crossing_rate(frame),
                "windowedRms": rms(&windowed)
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "channels": channels,
        "sampleCount": samples.len(),
        "durationSeconds": samples_to_seconds(samples_per_channel(samples.len(), channels)? as u64, sample_rate).map_err(|error| error.to_string())?,
        "frameSize": frame_size,
        "hopSize": hop_size,
        "frameCount": frame_spec.frame_count(samples.len()),
        "frames": summaries
    }))
}

fn timestamps_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let sample_rate = sample_rate(&input)?;
    let seconds = input
        .get("seconds")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    if !seconds.is_finite() || seconds < 0.0 {
        return Err("seconds must be finite and non-negative".to_string());
    }
    let samples_count = input
        .get("samplesCount")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| seconds_to_samples(seconds, sample_rate).unwrap_or(0));
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "seconds": seconds,
        "samplesFromSeconds": seconds_to_samples(seconds, sample_rate).map_err(|error| error.to_string())?,
        "samplesCount": samples_count,
        "secondsFromSamples": samples_to_seconds(samples_count, sample_rate).map_err(|error| error.to_string())?,
        "timestamp": {
            "pts": samples_count,
            "timebase": {"num": 1, "den": sample_rate}
        }
    }))
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
    let mut samples = Vec::with_capacity(values.len());
    for value in values {
        let sample = value
            .as_f64()
            .ok_or_else(|| format!("{field} must contain only numbers"))?
            as f32;
        if !sample.is_finite() {
            return Err(format!("{field} must contain only finite numbers"));
        }
        samples.push(sample);
    }
    Ok(samples)
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

fn samples_per_channel(sample_count: usize, channels: u16) -> Result<usize, String> {
    let channels = usize::from(channels);
    if !sample_count.is_multiple_of(channels) {
        return Err("sample count must be divisible by channels".to_string());
    }
    Ok(sample_count / channels)
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

fn input_limit(
    input: &serde_json::Value,
    field: &str,
    default_value: usize,
) -> Result<usize, String> {
    positive_usize(input, field, default_value).map(|value| value.min(DEFAULT_PREVIEW_SAMPLES))
}

fn preview(samples: &[f32], limit: usize) -> Vec<f32> {
    samples.iter().copied().take(limit).collect()
}

fn window_name(name: Option<&str>) -> WindowFunction {
    match name {
        Some("hann" | "Hann") => WindowFunction::Hann,
        Some("hamming" | "Hamming") => WindowFunction::Hamming,
        _ => WindowFunction::Rectangular,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_core::OperationId;

    #[test]
    fn package_surface_lists_audio_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.levels"));
        assert!(ids.contains(&"audio.frames"));
        assert!(ids.contains(&"audio.timestamps"));
    }

    #[test]
    fn levels_operation_returns_summary() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.levels"),
            input: serde_json::json!({"samples": [0.0, 1.0, -1.0], "sampleRate": 3, "channels": 1}),
        })
        .expect("levels");
        assert_eq!(response.value["operation"], "audio.levels");
        assert!(response.value["title"].as_str().unwrap().contains("Audio"));
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert_eq!(response.value["sampleCount"], 3);
        assert!(response.value["rms"].as_f64().unwrap() > 0.0);
        assert_eq!(response.value["featureSummary"]["frame_count"], 1);
        assert!(response.value["featureSeries"]["points"].is_array());
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
            operation: OperationId::new("audio.levels"),
            input: serde_json::json!({"samples": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("samples"));
    }
}
