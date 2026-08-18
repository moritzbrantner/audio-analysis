//! Library-owned runtime surface for `audio-analysis-rhythm`.

use audio_analysis_core::FrameSpec;
use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{
    beat_grid, detect_onsets, estimate_tempo, onset_envelope, OnsetDetectorConfig,
    TempoEstimatorConfig,
};

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
                "Onset detection and tempo estimation for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.rhythm.onsets",
                "Detect onsets",
                "Computes an onset envelope and deterministic onset list.",
                serde_json::json!({"samples": [1.0, 0.0, 0.0, 1.0], "sampleRate": 1000, "frameSize": 2, "hopSize": 1}),
            ),
            operation(
                "audio.rhythm.tempo",
                "Estimate tempo",
                "Estimates BPM from detected onset intervals.",
                serde_json::json!({"samples": [1.0, 0.0, 0.0, 1.0], "sampleRate": 1000, "frameSize": 2, "hopSize": 1}),
            ),
            operation(
                "audio.rhythm.beatGrid",
                "Beat grid",
                "Creates a beat grid from start time, BPM, and beat count.",
                serde_json::json!({"startSeconds": 0.0, "bpm": 120.0, "beats": 4}),
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
        "audio.rhythm.onsets" => onsets_value(request.input)?,
        "audio.rhythm.tempo" => tempo_value(request.input)?,
        "audio.rhythm.beatGrid" => beat_grid_value(request.input)?,
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
            "Rhythm package metadata",
            "Inspected the onset, tempo, and beat-grid operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.rhythm.onsets" => (
            "Onset detection result",
            "Computed an onset envelope and deterministic onset list from normalized samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "envelopeFrameCount": value.get("envelopeFrameCount").cloned().unwrap_or(serde_json::Value::Null),
                "onsetCount": value.get("onsetCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.rhythm.tempo" => (
            "Tempo estimate result",
            "Estimated BPM from detected onset intervals.",
            serde_json::json!({
                "bpm": value.get("bpm").cloned().unwrap_or(serde_json::Value::Null),
                "onsetCount": value.get("onsetCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.rhythm.beatGrid" => (
            "Beat grid result",
            "Created a deterministic beat grid from start time, BPM, and beat count.",
            serde_json::json!({
                "bpm": value.get("bpm").cloned().unwrap_or(serde_json::Value::Null),
                "beatCount": value.get("grid").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        _ => (
            "Rhythm operation result",
            "Completed the rhythm package surface operation.",
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

fn onsets_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let (sample_rate, frame_spec, envelope, onsets) = detected_onsets(&input)?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "frameSize": frame_spec.frame_size,
        "hopSize": frame_spec.hop_size,
        "envelopeFrameCount": envelope.len(),
        "onsetCount": onsets.len(),
        "onsets": onsets.iter().take(64).map(|onset| serde_json::json!({
            "timestampSeconds": onset.timestamp_seconds,
            "strength": onset.strength
        })).collect::<Vec<_>>()
    }))
}

fn tempo_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let (sample_rate, frame_spec, _envelope, onsets) = detected_onsets(&input)?;
    let tempo = estimate_tempo(&onsets, TempoEstimatorConfig::default())
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "frameSize": frame_spec.frame_size,
        "hopSize": frame_spec.hop_size,
        "onsetCount": onsets.len(),
        "bpm": tempo.bpm,
        "confidence": tempo.confidence
    }))
}

fn beat_grid_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let start_seconds = finite_f64(&input, "startSeconds", 0.0)?;
    let bpm = finite_f64(&input, "bpm", 120.0)? as f32;
    let beats = positive_usize(&input, "beats", 4)?.min(1024);
    let grid = beat_grid(start_seconds, bpm, beats).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "startSeconds": start_seconds,
        "bpm": bpm,
        "beats": beats,
        "grid": grid
    }))
}

fn detected_onsets(
    input: &serde_json::Value,
) -> Result<(u32, FrameSpec, Vec<crate::OnsetStrength>, Vec<crate::Onset>), String> {
    let samples = sample_array(input, "samples")?;
    let sample_rate = sample_rate(input)?;
    let frame_size = positive_usize(input, "frameSize", 1024)?;
    let hop_size = positive_usize(input, "hopSize", frame_size / 2)?;
    let frame_spec = FrameSpec::new(frame_size, hop_size).map_err(|error| error.to_string())?;
    let envelope =
        onset_envelope(&samples, sample_rate, frame_spec).map_err(|error| error.to_string())?;
    let config = OnsetDetectorConfig {
        strength_threshold: finite_f64(input, "strengthThreshold", 0.05)? as f32,
        min_interval_seconds: finite_f64(input, "minIntervalSeconds", 0.05)?,
    };
    let onsets = detect_onsets(&envelope, config).map_err(|error| error.to_string())?;
    Ok((sample_rate, frame_spec, envelope, onsets))
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

fn finite_f64(input: &serde_json::Value, field: &str, default_value: f64) -> Result<f64, String> {
    let value = input
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default_value);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{field} must be finite"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_rhythm_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.rhythm.onsets"));
        assert!(ids.contains(&"audio.rhythm.beatGrid"));
    }

    #[test]
    fn beat_grid_operation_returns_grid() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.rhythm.beatGrid"),
            input: serde_json::json!({"startSeconds": 0.0, "bpm": 120.0, "beats": 4}),
        })
        .expect("beat grid");
        assert_eq!(response.value["operation"], "audio.rhythm.beatGrid");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert_eq!(response.value["grid"].as_array().unwrap().len(), 4);
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
            operation: OperationId::new("audio.rhythm.onsets"),
            input: serde_json::json!({"samples": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("samples"));
    }
}
