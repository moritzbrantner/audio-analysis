//! WASM bindings for `audio-analysis-core`.

use runtime_core::SurfaceRequest;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = packageSurface)]
pub fn package_surface() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&audio_analysis_core::surface::package_surface())
        .map_err(into_js_error)
}

#[wasm_bindgen(js_name = runOperation)]
pub fn run_operation(request: JsValue) -> Result<JsValue, JsValue> {
    let request: SurfaceRequest = serde_wasm_bindgen::from_value(request).map_err(into_js_error)?;
    let response =
        audio_analysis_core::surface::run_surface_operation(request).map_err(into_js_error)?;
    serde_wasm_bindgen::to_value(&response).map_err(into_js_error)
}

/// Measures clipping, no-input and activity over interleaved samples (a `Float32Array`) without
/// the surface's sample-count limit, so whole captures can be measured. `options` is an optional
/// object with `frameSeconds`, `clipLevel`, `noInputRms` and `activityRms`.
#[wasm_bindgen(js_name = captureMetrics)]
pub fn capture_metrics(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    let config = if options.is_undefined() || options.is_null() {
        audio_analysis_core::CaptureMetricsConfig::default()
    } else {
        // Through a JSON value, so unknown option names are rejected rather than ignored.
        let options: serde_json::Value =
            serde_wasm_bindgen::from_value(options).map_err(into_js_error)?;
        serde_json::from_value(options)
            .map_err(|error| into_js_error(format!("options are invalid: {error}")))?
    };
    let metrics = audio_analysis_core::capture_metrics(samples, sample_rate, channels, &config)
        .map_err(into_js_error)?;
    metrics
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(into_js_error)
}

fn into_js_error(error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn wrapped_surface_has_operations() {
        let surface = audio_analysis_core::surface::package_surface();
        assert_eq!(surface.library, "moenarch-audio-analysis-core");
        assert!(!surface.operations.is_empty());
        let operation = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() != "describe")
            .unwrap();
        let response =
            audio_analysis_core::surface::run_surface_operation(runtime_core::SurfaceRequest {
                operation: operation.id.clone(),
                input: operation.example_request.clone(),
            })
            .expect("run default wasm operation");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
    }
}
