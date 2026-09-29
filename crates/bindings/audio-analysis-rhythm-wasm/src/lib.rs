//! WASM bindings for `audio-analysis-rhythm`.

use runtime_core::{OperationId, SurfaceRequest};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = packageSurface)]
pub fn package_surface() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&audio_analysis_rhythm::surface::package_surface())
        .map_err(into_js_error)
}

#[wasm_bindgen(js_name = runOperation)]
pub fn run_operation(request: JsValue) -> Result<JsValue, JsValue> {
    let request: SurfaceRequest = serde_wasm_bindgen::from_value(request).map_err(into_js_error)?;
    let response =
        audio_analysis_rhythm::surface::run_surface_operation(request).map_err(into_js_error)?;
    serde_wasm_bindgen::to_value(&response).map_err(into_js_error)
}

/// Runs whole-track rhythm analysis from a typed PCM slice while preserving the
/// same Rust-owned `audio.rhythm.analyze` surface contract used by other transports.
#[wasm_bindgen(js_name = analyzeTrack)]
pub fn analyze_track(
    samples: &[f32],
    sample_rate: u32,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    let options = if options.is_null() || options.is_undefined() {
        serde_json::json!({})
    } else {
        serde_wasm_bindgen::from_value(options).map_err(into_js_error)?
    };
    let input = track_request_input(samples, sample_rate, options).map_err(into_js_error)?;
    let response = audio_analysis_rhythm::surface::run_surface_operation(SurfaceRequest {
        operation: OperationId::new("audio.rhythm.analyze"),
        input,
    })
    .map_err(into_js_error)?;
    let result = response.value.get("result").cloned();
    let value = result.unwrap_or(response.value);
    serde_wasm_bindgen::to_value(&value).map_err(into_js_error)
}

fn track_request_input(
    samples: &[f32],
    sample_rate: u32,
    options: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let mut input = match options {
        serde_json::Value::Null => serde_json::Map::new(),
        serde_json::Value::Object(input) => input,
        _ => return Err("analyzeTrack options must be an object".to_string()),
    };
    input.insert(
        "samples".to_string(),
        serde_json::to_value(samples).map_err(|error| error.to_string())?,
    );
    input.insert("sampleRate".to_string(), serde_json::json!(sample_rate));
    Ok(serde_json::Value::Object(input))
}

fn into_js_error(error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapped_surface_has_operations() {
        let surface = audio_analysis_rhythm::surface::package_surface();
        assert_eq!(surface.library, "moenarch-audio-analysis-rhythm");
        assert!(!surface.operations.is_empty());
        let operation = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() != "describe")
            .unwrap();
        let response =
            audio_analysis_rhythm::surface::run_surface_operation(runtime_core::SurfaceRequest {
                operation: operation.id.clone(),
                input: operation.example_request.clone(),
            })
            .expect("run default wasm operation");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
    }

    #[test]
    fn typed_track_request_preserves_surface_options() {
        let input = track_request_input(
            &[0.25, -0.5],
            16_000,
            serde_json::json!({"minBpm": 45, "hopSize": 128}),
        )
        .expect("typed track request");

        assert_eq!(input["sampleRate"], 16_000);
        assert_eq!(input["minBpm"], 45);
        assert_eq!(input["hopSize"], 128);
        assert_eq!(input["samples"].as_array().unwrap().len(), 2);
    }
}
