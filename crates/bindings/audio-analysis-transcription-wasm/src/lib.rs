//! WASM bindings for `audio-analysis-transcription`.

use runtime_core::SurfaceRequest;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = packageSurface)]
pub fn package_surface() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&audio_analysis_transcription::surface::package_surface())
        .map_err(into_js_error)
}

/// Reports the browser-local transcription contract separately from the
/// package surface. The existing package surface remains conservative until a
/// real WebGPU provider is wired to `audio.transcription.transcribe`.
#[wasm_bindgen(js_name = browserCapabilities)]
pub fn browser_capabilities() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&browser_capability_contract()).map_err(into_js_error)
}

#[wasm_bindgen(js_name = runOperation)]
pub fn run_operation(request: JsValue) -> Result<JsValue, JsValue> {
    let request: SurfaceRequest = serde_wasm_bindgen::from_value(request).map_err(into_js_error)?;
    let response = audio_analysis_transcription::surface::run_surface_operation(request)
        .map_err(into_js_error)?;
    serde_wasm_bindgen::to_value(&response).map_err(into_js_error)
}

fn browser_capability_contract() -> serde_json::Value {
    serde_json::json!({
        "target": "wasm32-unknown-unknown",
        "requiredAcceleration": "webgpu",
        "modelProvisioning": "caller",
        "input": {
            "sampleRateHz": 16_000,
            "channels": 1,
            "sampleFormat": "f32"
        },
        "features": {
            "transcription": "pending-webgpu-provider",
            "timedSegments": "pending-webgpu-provider",
            "alignment": false,
            "diarization": false,
            "translation": false
        },
        "fallbacks": {
            "server": false,
            "python": false,
            "cpu": false
        }
    })
}

fn into_js_error(error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

#[cfg(test)]
mod tests {
    use super::browser_capability_contract;

    #[test]
    fn wrapped_surface_has_operations() {
        let surface = audio_analysis_transcription::surface::package_surface();
        assert_eq!(surface.library, "moenarch-audio-analysis-transcription");
        assert!(!surface.operations.is_empty());
        assert!(surface
            .operations
            .iter()
            .any(|operation| operation.id.as_str() == "audio.transcription.transcribe"));
    }

    #[test]
    fn browser_capabilities_keep_webgpu_provider_boundary_explicit() {
        let capabilities = browser_capability_contract();
        assert_eq!(capabilities["target"], "wasm32-unknown-unknown");
        assert_eq!(capabilities["requiredAcceleration"], "webgpu");
        assert_eq!(capabilities["input"]["sampleRateHz"], 16_000);
        assert_eq!(capabilities["input"]["channels"], 1);
        assert_eq!(
            capabilities["features"]["transcription"],
            "pending-webgpu-provider"
        );
        assert_eq!(capabilities["fallbacks"]["server"], false);
        assert_eq!(capabilities["fallbacks"]["python"], false);
        assert_eq!(capabilities["fallbacks"]["cpu"], false);
    }
}
