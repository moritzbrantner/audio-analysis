//! WASM bindings for `audio-analysis-transcription`.

use runtime_core::SurfaceRequest;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = packageSurface)]
pub fn package_surface() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&audio_analysis_transcription::surface::package_surface())
        .map_err(into_js_error)
}

#[wasm_bindgen(js_name = runOperation)]
pub fn run_operation(request: JsValue) -> Result<JsValue, JsValue> {
    let request: SurfaceRequest = serde_wasm_bindgen::from_value(request).map_err(into_js_error)?;
    let response = audio_analysis_transcription::surface::run_surface_operation(request)
        .map_err(into_js_error)?;
    serde_wasm_bindgen::to_value(&response).map_err(into_js_error)
}

fn into_js_error(error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

#[cfg(test)]
mod tests {
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
}
