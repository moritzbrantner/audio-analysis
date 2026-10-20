//! WASM bindings for `audio-analysis-speakers`.

use runtime_core::SurfaceRequest;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = packageSurface)]
pub fn package_surface() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&audio_analysis_speakers::surface::package_surface())
        .map_err(into_js_error)
}

#[wasm_bindgen(js_name = runOperation)]
pub fn run_operation(request: JsValue) -> Result<JsValue, JsValue> {
    let request: SurfaceRequest = serde_wasm_bindgen::from_value(request).map_err(into_js_error)?;
    let response =
        audio_analysis_speakers::surface::run_surface_operation(request).map_err(into_js_error)?;
    serde_wasm_bindgen::to_value(&response).map_err(into_js_error)
}

fn into_js_error(error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn wrapped_surface_has_operations() {
        let surface = audio_analysis_speakers::surface::package_surface();
        assert_eq!(surface.library, "moenarch-audio-analysis-speakers");
        assert!(!surface.operations.is_empty());
        let operation = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() != "describe")
            .unwrap();
        let response =
            audio_analysis_speakers::surface::run_surface_operation(runtime_core::SurfaceRequest {
                operation: operation.id.clone(),
                input: operation.example_request.clone(),
            })
            .expect("run default wasm operation");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
    }
}

/// Owns one real-model community diarization postprocessing run.
#[wasm_bindgen]
pub struct CommunityDiarization {
    processor: Option<audio_analysis_speakers::pyannote::CommunityDiarizationProcessor>,
}

#[wasm_bindgen]
impl CommunityDiarization {
    #[wasm_bindgen(constructor)]
    pub fn new(
        transform: &[u8],
        model: &[u8],
        clustering: &[u8],
        sample_count: u32,
    ) -> Result<Self, JsValue> {
        let processor =
            audio_analysis_speakers::pyannote::CommunityDiarizationProcessor::from_resources(
                transform,
                model,
                clustering,
                sample_count as usize,
            )
            .map_err(into_js_error)?;
        Ok(Self {
            processor: Some(processor),
        })
    }

    #[wasm_bindgen(js_name = pushSegmentation)]
    pub fn push_segmentation(&mut self, scores: Vec<f32>) -> Result<JsValue, JsValue> {
        let requests = self
            .processor
            .as_mut()
            .ok_or_else(|| into_js_error("diarization already finished"))?
            .push_segmentation(scores)
            .map_err(into_js_error)?;
        serde_wasm_bindgen::to_value(&requests).map_err(into_js_error)
    }

    #[wasm_bindgen(js_name = pushEmbedding)]
    pub fn push_embedding(
        &mut self,
        chunk: u32,
        local_speaker: u32,
        values: Vec<f32>,
    ) -> Result<(), JsValue> {
        self.processor
            .as_mut()
            .ok_or_else(|| into_js_error("diarization already finished"))?
            .push_embedding(chunk as usize, local_speaker as usize, values)
            .map_err(into_js_error)
    }

    pub fn finish(&mut self) -> Result<JsValue, JsValue> {
        let result = self
            .processor
            .take()
            .ok_or_else(|| into_js_error("diarization already finished"))?
            .finish()
            .map_err(into_js_error)?;
        serde_wasm_bindgen::to_value(&result.response).map_err(into_js_error)
    }
}
