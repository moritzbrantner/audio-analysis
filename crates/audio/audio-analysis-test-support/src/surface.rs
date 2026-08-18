//! Library-owned runtime surface for `audio-analysis-test-support`.

use audio_contracts::AudioBuffer;
use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{click_track, impulse_train, owned_f32_frame, pink_noise, sine, white_noise};

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
                "Synthetic waveform fixtures and shared audio test helpers.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.fixtures.generate",
                "Generate fixture",
                "Generates deterministic in-memory sine, click, impulse, white-noise, or pink-noise samples.",
                serde_json::json!({"kind": "sine", "frequencyHz": 440.0, "sampleRate": 48000, "seconds": 0.1}),
            ),
            operation(
                "audio.fixtures.frame",
                "Generate frame",
                "Builds a deterministic OwnedAudioFrame summary from generated samples.",
                serde_json::json!({"kind": "sine", "frequencyHz": 440.0, "sampleRate": 48000, "seconds": 0.1}),
            ),
            operation(
                "audio.fixtures.catalog",
                "Inspect fixture catalog",
                "Inspects deterministic fixture generators available to tests and surface checks.",
                serde_json::json!({}),
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
        "audio.fixtures.generate" => generate_value(request.input)?,
        "audio.fixtures.frame" => frame_value(request.input)?,
        "audio.fixtures.catalog" => catalog_value(),
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
            "Audio fixture package metadata",
            "Inspected the deterministic audio fixture support operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.fixtures.generate" => (
            "Generated audio fixture",
            "Generated deterministic in-memory samples for tests and examples.",
            serde_json::json!({
                "kind": value.get("kind").cloned().unwrap_or(serde_json::Value::Null),
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "durationSeconds": value.get("durationSeconds").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.fixtures.frame" => (
            "Generated audio frame fixture",
            "Built a deterministic OwnedAudioFrame summary from generated samples.",
            serde_json::json!({
                "kind": value.get("kind").cloned().unwrap_or(serde_json::Value::Null),
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.fixtures.catalog" => (
            "Audio fixture catalog",
            "Inspected deterministic fixture generators without generating samples.",
            serde_json::json!({
                "kindCount": value.get("kinds").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        _ => (
            "Audio fixture operation result",
            "Completed the audio fixture support operation.",
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

fn generate_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let sample_rate = sample_rate(&input)?;
    let samples = fixture_samples(&input, sample_rate)?;
    Ok(serde_json::json!({
        "kind": kind(&input),
        "sampleRate": sample_rate,
        "channels": 1,
        "sampleCount": samples.len(),
        "durationSeconds": samples.len() as f64 / sample_rate as f64,
        "samplePreview": samples.iter().copied().take(DEFAULT_PREVIEW_SAMPLES).collect::<Vec<_>>()
    }))
}

fn frame_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let sample_rate = sample_rate(&input)?;
    let samples = fixture_samples(&input, sample_rate)?;
    let frame = owned_f32_frame(
        audio_contracts::Timestamp::new(0, audio_contracts::Timebase::new(1, sample_rate as i32)),
        sample_rate,
        1,
        samples,
    )
    .map_err(|error| error.to_string())?;
    let sample_count = match &frame.data {
        AudioBuffer::F32(samples) => samples.len(),
        _ => 0,
    };
    Ok(serde_json::json!({
        "kind": kind(&input),
        "sampleRate": frame.sample_rate,
        "channels": frame.channels,
        "sampleFormat": format!("{:?}", frame.sample_format()),
        "sampleCount": sample_count,
        "samplesPerChannel": frame.samples_per_channel()
    }))
}

fn catalog_value() -> serde_json::Value {
    serde_json::json!({
        "fixtures": [
            {"kind": "sine", "parameters": ["frequencyHz", "sampleRate", "seconds"]},
            {"kind": "click", "parameters": ["bpm", "sampleRate", "seconds"]},
            {"kind": "impulse", "parameters": ["bpm", "sampleRate", "seconds"]},
            {"kind": "whiteNoise", "parameters": ["seed", "samples"]},
            {"kind": "pinkNoise", "parameters": ["seed", "samples"]}
        ]
    })
}

fn fixture_samples(input: &serde_json::Value, sample_rate: u32) -> Result<Vec<f32>, String> {
    let seconds = finite_f32(input, "seconds", 0.1)?;
    let seed = input
        .get("seed")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);
    let sample_count = input
        .get("samples")
        .and_then(serde_json::Value::as_u64)
        .map(|value| value.min(192_000) as usize)
        .unwrap_or_else(|| (sample_rate as f32 * seconds).round().max(1.0) as usize);
    Ok(match kind(input).as_str() {
        "click" => click_track(sample_rate, finite_f32(input, "bpm", 120.0)?, seconds),
        "impulse" => impulse_train(sample_rate, finite_f32(input, "bpm", 120.0)?, seconds),
        "whiteNoise" | "white_noise" => white_noise(seed, sample_count),
        "pinkNoise" | "pink_noise" => pink_noise(seed, sample_count),
        _ => sine(
            finite_f32(input, "frequencyHz", 440.0)?,
            sample_rate,
            seconds,
        ),
    })
}

fn kind(input: &serde_json::Value) -> String {
    input
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("sine")
        .to_string()
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

fn finite_f32(input: &serde_json::Value, field: &str, default_value: f32) -> Result<f32, String> {
    let value = input
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default_value as f64) as f32;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!("{field} must be finite and positive"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_fixture_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.fixtures.generate"));
        assert!(ids.contains(&"audio.fixtures.catalog"));
    }

    #[test]
    fn generate_operation_returns_samples() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.fixtures.generate"),
            input: serde_json::json!({"kind": "sine", "frequencyHz": 440.0, "sampleRate": 1000, "seconds": 0.01}),
        })
        .expect("generate");
        assert_eq!(response.value["operation"], "audio.fixtures.generate");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert!(response.value["sampleCount"].as_u64().unwrap() > 0);
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
    fn invalid_fixture_returns_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.fixtures.generate"),
            input: serde_json::json!({"seconds": -1.0}),
        })
        .unwrap_err();
        assert!(error.contains("seconds"));
    }
}
