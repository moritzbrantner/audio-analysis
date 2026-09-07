//! WASM bindings for `audio-analysis-pitch`.

use audio_analysis_pitch::key::{
    analyze_key_track, HarmonicKeyConfig, KeyProfile, KeyTimelineConfig,
};
use runtime_core::SurfaceRequest;
use serde_json::{Map, Value};
use wasm_bindgen::prelude::*;

const MAX_TRACK_SECONDS: usize = 15 * 60;

#[wasm_bindgen(js_name = packageSurface)]
pub fn package_surface() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&audio_analysis_pitch::surface::package_surface())
        .map_err(into_js_error)
}

#[wasm_bindgen(js_name = runOperation)]
pub fn run_operation(request: JsValue) -> Result<JsValue, JsValue> {
    let request: SurfaceRequest = serde_wasm_bindgen::from_value(request).map_err(into_js_error)?;
    let response =
        audio_analysis_pitch::surface::run_surface_operation(request).map_err(into_js_error)?;
    serde_wasm_bindgen::to_value(&response).map_err(into_js_error)
}

/// Runs complete-track dominant-key and local key-window analysis from typed PCM.
///
/// This binding owns only typed transport, browser-specific input bounds, and
/// option decoding. Dominant-key selection, timeline windows, confidence
/// thresholding, and uncertainty semantics live in `audio-analysis-pitch`.
#[wasm_bindgen(js_name = analyzeTrackKey)]
pub fn analyze_track_key(
    samples: &[f32],
    sample_rate: u32,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    if sample_rate > 0 {
        let max_samples = (sample_rate as usize).saturating_mul(MAX_TRACK_SECONDS);
        if samples.len() > max_samples {
            return Err(into_js_error(format!(
                "track key samples must not exceed {MAX_TRACK_SECONDS} seconds"
            )));
        }
    }

    let options = options_object(options).map_err(into_js_error)?;
    let harmonic_config = harmonic_config(&options).map_err(into_js_error)?;
    let timeline_config = timeline_config(&options).map_err(into_js_error)?;
    let analysis = analyze_key_track(samples, sample_rate, harmonic_config, timeline_config)
        .map_err(into_js_error)?;
    serde_wasm_bindgen::to_value(&analysis).map_err(into_js_error)
}

fn options_object(options: JsValue) -> Result<Map<String, Value>, String> {
    if options.is_null() || options.is_undefined() {
        return Ok(Map::new());
    }
    match serde_wasm_bindgen::from_value::<Value>(options).map_err(|error| error.to_string())? {
        Value::Object(options) => Ok(options),
        _ => Err("analyzeTrackKey options must be an object".to_string()),
    }
}

fn harmonic_config(options: &Map<String, Value>) -> Result<HarmonicKeyConfig, String> {
    let mut config = HarmonicKeyConfig::default();
    config.fft_size = positive_usize(options, "fftSize", config.fft_size)?;
    config.hop_size = positive_usize(options, "hopSize", config.hop_size)?;
    config.min_frequency_hz =
        positive_f64(options, "minFrequencyHz", config.min_frequency_hz as f64)? as f32;
    config.max_frequency_hz =
        positive_f64(options, "maxFrequencyHz", config.max_frequency_hz as f64)? as f32;
    config.peak_threshold =
        unit_f64(options, "peakThreshold", config.peak_threshold as f64)? as f32;
    config.profile = match options
        .get("profile")
        .and_then(Value::as_str)
        .unwrap_or("ensemble")
    {
        "ensemble" => KeyProfile::Ensemble,
        "krumhansl" => KeyProfile::Krumhansl,
        "temperley" => KeyProfile::Temperley,
        value => {
            return Err(format!(
                "profile must be one of ensemble, krumhansl, or temperley; got `{value}`"
            ));
        }
    };
    config.validate().map_err(|error| error.to_string())?;
    Ok(config)
}

fn timeline_config(options: &Map<String, Value>) -> Result<KeyTimelineConfig, String> {
    let mut config = KeyTimelineConfig::default();
    config.window_seconds = positive_f64(
        options,
        "timelineWindowSeconds",
        config.window_seconds,
    )?;
    config.hop_seconds =
        positive_f64(options, "timelineHopSeconds", config.hop_seconds)?;
    config.min_confidence = unit_f64(
        options,
        "timelineMinConfidence",
        config.min_confidence as f64,
    )? as f32;
    if let Some(value) = options.get("timelineMaxWindows").and_then(Value::as_u64) {
        config.max_windows = usize::try_from(value)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| "timelineMaxWindows must be positive".to_string())?;
    }
    config.validate().map_err(|error| error.to_string())?;
    Ok(config)
}

fn positive_usize(
    options: &Map<String, Value>,
    field: &str,
    default_value: usize,
) -> Result<usize, String> {
    let value = options
        .get(field)
        .and_then(Value::as_u64)
        .unwrap_or(default_value as u64);
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{field} must be positive"))
}

fn positive_f64(
    options: &Map<String, Value>,
    field: &str,
    default_value: f64,
) -> Result<f64, String> {
    let value = options
        .get(field)
        .and_then(Value::as_f64)
        .unwrap_or(default_value);
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(format!("{field} must be finite and positive"))
    }
}

fn unit_f64(
    options: &Map<String, Value>,
    field: &str,
    default_value: f64,
) -> Result<f64, String> {
    let value = options
        .get(field)
        .and_then(Value::as_f64)
        .unwrap_or(default_value);
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(format!("{field} must be between 0 and 1"))
    }
}

fn into_js_error(error: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&error.to_string()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapped_surface_has_operations() {
        let surface = audio_analysis_pitch::surface::package_surface();
        assert_eq!(surface.library, "moenarch-audio-analysis-pitch");
        assert!(!surface.operations.is_empty());
        let operation = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() != "describe")
            .unwrap();
        let response =
            audio_analysis_pitch::surface::run_surface_operation(runtime_core::SurfaceRequest {
                operation: operation.id.clone(),
                input: operation.example_request.clone(),
            })
            .expect("run default wasm operation");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
    }

    #[test]
    fn adapter_options_map_to_reusable_timeline_config() {
        let options = serde_json::json!({
            "timelineWindowSeconds": 18.0,
            "timelineHopSeconds": 6.0,
            "timelineMinConfidence": 0.25,
            "timelineMaxWindows": 42
        })
        .as_object()
        .unwrap()
        .clone();
        let config = timeline_config(&options).expect("timeline config");
        assert_eq!(config.window_seconds, 18.0);
        assert_eq!(config.hop_seconds, 6.0);
        assert_eq!(config.min_confidence, 0.25);
        assert_eq!(config.max_windows, 42);
    }
}
