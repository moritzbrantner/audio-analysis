//! WASM bindings for `audio-analysis-pitch`.

use audio_analysis_pitch::key::{
    estimate_musical_key, HarmonicKeyConfig, KeyProfile, MusicalKeyEstimate, MusicalScale,
};
use runtime_core::SurfaceRequest;
use serde_json::{json, Map, Value};
use wasm_bindgen::prelude::*;

const MAX_TRACK_SECONDS: usize = 15 * 60;
const MAX_TIMELINE_WINDOWS: usize = 256;

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

/// Runs dominant-key and local key-window analysis directly from typed PCM.
///
/// The browser adapter owns only transport and bounded window orchestration;
/// every key decision is still made by the reusable Rust musical-key analyzer.
#[wasm_bindgen(js_name = analyzeTrackKey)]
pub fn analyze_track_key(
    samples: &[f32],
    sample_rate: u32,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    if sample_rate == 0 {
        return Err(into_js_error("sampleRate must be positive"));
    }
    if samples.is_empty() {
        return Err(into_js_error("track key samples must not be empty"));
    }
    let max_samples = (sample_rate as usize).saturating_mul(MAX_TRACK_SECONDS);
    if samples.len() > max_samples {
        return Err(into_js_error(format!(
            "track key samples must not exceed {MAX_TRACK_SECONDS} seconds"
        )));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(into_js_error(
            "track key samples must contain only finite values",
        ));
    }

    let options = options_object(options).map_err(into_js_error)?;
    let config = harmonic_config(&options).map_err(into_js_error)?;
    let timeline_window_seconds = positive_f64(&options, "timelineWindowSeconds", 24.0)
        .map_err(into_js_error)?;
    let timeline_hop_seconds = positive_f64(&options, "timelineHopSeconds", 8.0)
        .map_err(into_js_error)?;
    let timeline_min_confidence = unit_f64(&options, "timelineMinConfidence", 0.10)
        .map_err(into_js_error)? as f32;

    let dominant = estimate_musical_key(samples, sample_rate, config)
        .map_err(into_js_error)?
        .map(key_json);
    let timeline = key_timeline(
        samples,
        sample_rate,
        config,
        timeline_window_seconds,
        timeline_hop_seconds,
        timeline_min_confidence,
    )
    .map_err(into_js_error)?;

    serde_wasm_bindgen::to_value(&json!({
        "schemaVersion": "audio-analysis-key-track/v1",
        "sampleRate": sample_rate,
        "sampleCount": samples.len(),
        "durationSeconds": samples.len() as f64 / sample_rate as f64,
        "dominant": dominant,
        "timelineWindowSeconds": timeline_window_seconds,
        "timelineHopSeconds": timeline_hop_seconds,
        "timelineMinConfidence": timeline_min_confidence,
        "timeline": timeline,
    }))
    .map_err(into_js_error)
}

fn key_timeline(
    samples: &[f32],
    sample_rate: u32,
    config: HarmonicKeyConfig,
    window_seconds: f64,
    hop_seconds: f64,
    min_confidence: f32,
) -> Result<Vec<Value>, String> {
    let window_samples = ((window_seconds * sample_rate as f64).round() as usize)
        .max(config.fft_size)
        .min(samples.len());
    let hop_samples = ((hop_seconds * sample_rate as f64).round() as usize).max(1);
    if window_samples == 0 {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    let mut start = 0_usize;
    while start < samples.len() && result.len() < MAX_TIMELINE_WINDOWS {
        let end = start.saturating_add(window_samples).min(samples.len());
        if end.saturating_sub(start) < config.fft_size {
            break;
        }
        let estimate = estimate_musical_key(&samples[start..end], sample_rate, config)
            .map_err(|error| error.to_string())?
            .filter(|estimate| estimate.confidence >= min_confidence);
        let start_seconds = start as f64 / sample_rate as f64;
        let end_seconds = end as f64 / sample_rate as f64;
        result.push(json!({
            "startSeconds": start_seconds,
            "endSeconds": end_seconds,
            "centerSeconds": (start_seconds + end_seconds) * 0.5,
            "key": estimate.map(key_json),
        }));
        if end == samples.len() {
            break;
        }
        start = start.saturating_add(hop_samples);
    }
    Ok(result)
}

fn key_json(estimate: MusicalKeyEstimate) -> Value {
    json!({
        "label": estimate.label(),
        "tonic": estimate.tonic.as_str(),
        "scale": scale_name(estimate.scale),
        "strength": estimate.strength,
        "confidence": estimate.confidence,
        "runnerUp": {
            "tonic": estimate.runner_up.tonic.as_str(),
            "scale": scale_name(estimate.runner_up.scale),
            "correlation": estimate.runner_up.correlation,
        },
        "tuningCents": estimate.tuning_cents,
        "chroma": estimate.chroma.bins,
        "peakCount": estimate.peak_count,
    })
}

fn scale_name(scale: MusicalScale) -> &'static str {
    match scale {
        MusicalScale::Major => "major",
        MusicalScale::Minor => "minor",
    }
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
    config.min_frequency_hz = positive_f64(
        options,
        "minFrequencyHz",
        config.min_frequency_hz as f64,
    )? as f32;
    config.max_frequency_hz = positive_f64(
        options,
        "maxFrequencyHz",
        config.max_frequency_hz as f64,
    )? as f32;
    config.peak_threshold = unit_f64(options, "peakThreshold", config.peak_threshold as f64)? as f32;
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
    fn timeline_preserves_uncertain_windows() {
        let samples = vec![0.0; 16_000 * 3];
        let timeline = key_timeline(
            &samples,
            16_000,
            HarmonicKeyConfig {
                fft_size: 1024,
                hop_size: 512,
                ..HarmonicKeyConfig::default()
            },
            1.0,
            0.5,
            0.1,
        )
        .expect("timeline");
        assert!(!timeline.is_empty());
        assert!(timeline.iter().all(|window| window["key"].is_null()));
    }
}
