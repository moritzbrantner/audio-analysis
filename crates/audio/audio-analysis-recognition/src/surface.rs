//! Library-owned runtime surface for `audio-analysis-recognition`.

use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{
    AudioEmbeddingExtractor, AudioMatchOptions, AudioReferenceLibrary, SpectralAudioEmbedder,
    SpectralEmbeddingConfig,
};

const MAX_SAMPLES: usize = 192_000;
const DEFAULT_PREVIEW_VALUES: usize = 32;

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
                "Deterministic audio embeddings and similarity search for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.recognition.embed",
                "Embed audio",
                "Computes a deterministic spectral embedding for normalized samples.",
                serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 48000, "bands": 8}),
            ),
            operation(
                "audio.recognition.compare",
                "Compare audio",
                "Compares two in-memory sample arrays by cosine similarity.",
                serde_json::json!({"leftSamples": [0.0, 1.0, 0.0, -1.0], "rightSamples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 48000}),
            ),
            operation(
                "audio.recognition.search",
                "Search references",
                "Builds a transient sample-backed reference library and searches it.",
                serde_json::json!({"querySamples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 48000, "references": [{"id": "ref-1", "label": "Reference", "samples": [0.0, 1.0, 0.0, -1.0]}]}),
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
        "audio.recognition.embed" => embed_value(request.input)?,
        "audio.recognition.compare" => compare_value(request.input)?,
        "audio.recognition.search" => search_value(request.input)?,
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
            "Recognition package metadata",
            "Inspected the embedding, comparison, and reference-search operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.recognition.embed" => (
            "Audio embedding result",
            "Computed a deterministic spectral embedding for normalized audio samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "dimensions": value.get("dimensions").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.recognition.compare" => (
            "Audio comparison result",
            "Compared two in-memory sample arrays by cosine similarity.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "similarity": value.get("similarity").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.recognition.search" => (
            "Audio reference search result",
            "Built a transient sample-backed reference library and searched it with the query audio.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "referenceCount": value.get("referenceCount").cloned().unwrap_or(serde_json::Value::Null),
                "matchCount": value.get("matches").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        _ => (
            "Recognition operation result",
            "Completed the recognition package surface operation.",
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

fn embed_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let embedder = embedder_from_input(&input)?;
    let embedding = embedder
        .embed_samples(&samples, sample_rate)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "sampleCount": samples.len(),
        "dimensions": embedding.dimensions(),
        "valuesPreview": embedding.values().iter().copied().take(DEFAULT_PREVIEW_VALUES).collect::<Vec<_>>()
    }))
}

fn compare_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let left = sample_array(&input, "leftSamples")?;
    let right = sample_array(&input, "rightSamples")?;
    let sample_rate = sample_rate(&input)?;
    let embedder = embedder_from_input(&input)?;
    let left_embedding = embedder
        .embed_samples(&left, sample_rate)
        .map_err(|error| error.to_string())?;
    let right_embedding = embedder
        .embed_samples(&right, sample_rate)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "leftSampleCount": left.len(),
        "rightSampleCount": right.len(),
        "dimensions": left_embedding.dimensions(),
        "similarity": left_embedding.cosine_similarity(&right_embedding).map_err(|error| error.to_string())?
    }))
}

fn search_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let query = sample_array(&input, "querySamples")?;
    let sample_rate = sample_rate(&input)?;
    let embedder = embedder_from_input(&input)?;
    let query_embedding = embedder
        .embed_samples(&query, sample_rate)
        .map_err(|error| error.to_string())?;
    let references = input
        .get("references")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "references must be an array".to_string())?;
    let mut library = AudioReferenceLibrary::new();
    for reference in references {
        let id = reference
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "reference id must be a string".to_string())?;
        let label = reference
            .get("label")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(id);
        let samples = sample_array(reference, "samples")?;
        library
            .add_reference_samples(id, label, &samples, sample_rate, &embedder)
            .map_err(|error| error.to_string())?;
    }
    let options = AudioMatchOptions::new(
        input
            .get("minScore")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32,
    )
    .map_err(|error| error.to_string())?
    .max_results(positive_usize(&input, "topK", 5)?);
    let matches = library
        .search(&query_embedding, &options)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "querySampleCount": query.len(),
        "referenceCount": library.len(),
        "matches": matches.into_iter().map(|matched| serde_json::json!({
            "id": matched.reference_id,
            "label": matched.label,
            "score": matched.score
        })).collect::<Vec<_>>()
    }))
}

fn embedder_from_input(input: &serde_json::Value) -> Result<SpectralAudioEmbedder, String> {
    let fft_size = positive_usize(input, "fftSize", 512)?;
    let hop_size = positive_usize(input, "hopSize", fft_size / 2)?;
    let bands = positive_usize(input, "bands", 8)?;
    SpectralAudioEmbedder::new(
        SpectralEmbeddingConfig::new(fft_size, hop_size, bands)
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_recognition_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.recognition.embed"));
        assert!(ids.contains(&"audio.recognition.search"));
        assert!(!ids.contains(&"audio.recognition.transcribe"));
        assert!(!ids.contains(&"audio.recognition.transcribeImported"));
        assert!(!ids.contains(&"audio.recognition.transcriptionPlan"));
        assert!(!ids.contains(&"audio.recognition.transcriptionProviders"));
    }

    #[test]
    fn compare_operation_returns_similarity() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.recognition.compare"),
            input: serde_json::json!({
                "leftSamples": [0.0, 1.0, 0.0, -1.0],
                "rightSamples": [0.0, 1.0, 0.0, -1.0],
                "sampleRate": 4,
                "fftSize": 4,
                "hopSize": 2,
                "bands": 2
            }),
        })
        .expect("compare");
        assert_eq!(response.value["operation"], "audio.recognition.compare");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert!(response.value["similarity"].as_f64().unwrap() > 0.9);
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
            operation: OperationId::new("audio.recognition.embed"),
            input: serde_json::json!({"samples": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("samples"));
    }

    #[test]
    fn transcription_operations_are_not_package_surface_operations() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.recognition.transcribe"),
            input: serde_json::json!({}),
        })
        .unwrap_err();
        assert!(error.contains("unsupported operation"));
    }
}
