//! Library-owned runtime surface for `audio-analysis-fourier`.

use audio_analysis_core::WindowFunction;
use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{
    spectral_feature_frames, spectrogram, zero_crossing_rate, FourierTransform,
    SpectralFeatureOptions, StftConfig,
};

const MAX_SAMPLES: usize = 192_000;
const DEFAULT_PREVIEW_BINS: usize = 64;
const DEFAULT_PREVIEW_FRAMES: usize = 16;

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
                "FFT, STFT spectrograms, and spectral features for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.fourier.spectrum",
                "Spectrum",
                "Computes an FFT spectrum and returns dominant-frequency metadata.",
                serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 48000, "fftSize": 4}),
            ),
            operation(
                "audio.fourier.spectrogram",
                "Spectrogram",
                "Computes deterministic STFT frame summaries.",
                serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0, 0.0, 1.0], "sampleRate": 48000, "fftSize": 4, "hopSize": 2}),
            ),
            operation(
                "audio.fourier.features",
                "Spectral features",
                "Returns spectral centroid, bandwidth, rolloff, flatness, zero-crossing rate, and optional mel-style band features.",
                serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0], "sampleRate": 48000, "fftSize": 4, "hopSize": 2, "melBandCount": 4}),
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
        "audio.fourier.spectrum" => spectrum_value(request.input)?,
        "audio.fourier.spectrogram" => spectrogram_value(request.input)?,
        "audio.fourier.features" => features_value(request.input)?,
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
            "Fourier package metadata",
            "Inspected the FFT, STFT, and spectral feature operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.fourier.spectrum" => (
            "FFT spectrum result",
            "Computed an FFT spectrum and dominant-frequency metadata for normalized audio samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "fftSize": value.get("fftSize").cloned().unwrap_or(serde_json::Value::Null),
                "binCount": value.get("binCount").cloned().unwrap_or(serde_json::Value::Null),
                "dominantFrequencyHz": value.get("dominantFrequencyHz").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.fourier.spectrogram" => (
            "STFT spectrogram result",
            "Computed deterministic STFT frame summaries for normalized audio samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "fftSize": value.get("fftSize").cloned().unwrap_or(serde_json::Value::Null),
                "hopSize": value.get("hopSize").cloned().unwrap_or(serde_json::Value::Null),
                "frameCount": value.get("frameCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.fourier.features" => (
            "Spectral feature result",
            "Computed spectral centroid, bandwidth, rolloff, flatness, dominant frequency, zero-crossing rate, and optional mel-style band features.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "fftSize": value.get("fftSize").cloned().unwrap_or(serde_json::Value::Null),
                "centroidHz": value.get("centroidHz").cloned().unwrap_or(serde_json::Value::Null),
                "rolloffHz": value.get("rolloffHz").cloned().unwrap_or(serde_json::Value::Null),
                "dominantFrequencyHz": value.get("dominantFrequencyHz").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        _ => (
            "Fourier operation result",
            "Completed the Fourier package surface operation.",
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

fn spectrum_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let fft_size = positive_usize(&input, "fftSize", 1024)?;
    let transform = FourierTransform::with_window(fft_size, window_name(&input))
        .map_err(|error| error.to_string())?;
    let spectrum = transform
        .analyze_samples(&samples, sample_rate)
        .map_err(|error| error.to_string())?;
    let max_bins =
        positive_usize(&input, "maxBins", DEFAULT_PREVIEW_BINS)?.min(DEFAULT_PREVIEW_BINS);
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "sampleCount": samples.len(),
        "fftSize": spectrum.fft_size,
        "binCount": spectrum.bins.len(),
        "dominantFrequencyHz": spectrum.dominant_frequency_hz(),
        "bins": spectrum.bins.iter().take(max_bins).map(|bin| serde_json::json!({
            "index": bin.index,
            "frequencyHz": bin.frequency_hz,
            "magnitude": bin.magnitude,
            "power": bin.power
        })).collect::<Vec<_>>()
    }))
}

fn spectrogram_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let fft_size = positive_usize(&input, "fftSize", 1024)?;
    let hop_size = positive_usize(&input, "hopSize", fft_size / 2)?;
    let config = StftConfig::new(fft_size, hop_size)
        .map_err(|error| error.to_string())?
        .window(window_name(&input))
        .pad_final_frame(
            input
                .get("padFinalFrame")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        );
    let frames = spectrogram(&samples, sample_rate, &config).map_err(|error| error.to_string())?;
    let max_frames =
        positive_usize(&input, "maxFrames", DEFAULT_PREVIEW_FRAMES)?.min(DEFAULT_PREVIEW_FRAMES);
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "sampleCount": samples.len(),
        "fftSize": fft_size,
        "hopSize": hop_size,
        "frameCount": frames.len(),
        "frames": frames.iter().take(max_frames).map(|frame| serde_json::json!({
            "startSample": frame.start_sample,
            "startSeconds": frame.start_seconds,
            "dominantFrequencyHz": frame.spectrum.dominant_frequency_hz(),
            "centroidHz": frame.spectrum.features().centroid_hz
        })).collect::<Vec<_>>()
    }))
}

fn features_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let fft_size = positive_usize(&input, "fftSize", 1024)?;
    let hop_size = positive_usize(&input, "hopSize", fft_size / 2)?;
    let mel_band_count = input
        .get("melBandCount")
        .and_then(serde_json::Value::as_u64)
        .map(|value| usize::try_from(value).map_err(|_| "melBandCount must fit usize".to_string()))
        .transpose()?;
    let transform = FourierTransform::with_window(fft_size, window_name(&input))
        .map_err(|error| error.to_string())?;
    let spectrum = transform
        .analyze_samples(&samples, sample_rate)
        .map_err(|error| error.to_string())?;
    let features = spectrum.features();
    let options = SpectralFeatureOptions::new(fft_size, hop_size, sample_rate)
        .map_err(|error| error.to_string())?
        .mel_band_count(mel_band_count)
        .map_err(|error| error.to_string())?;
    let frames = spectral_feature_frames(&samples, options).map_err(|error| error.to_string())?;
    let max_frames =
        positive_usize(&input, "maxFrames", DEFAULT_PREVIEW_FRAMES)?.min(DEFAULT_PREVIEW_FRAMES);
    let mel_bands = if mel_band_count.is_some() && !frames.is_empty() {
        let band_count = frames[0].mel_bands.len();
        let mut bands = vec![0.0_f32; band_count];
        for frame in &frames {
            for (index, value) in frame.mel_bands.iter().enumerate() {
                bands[index] += *value;
            }
        }
        for band in &mut bands {
            *band /= frames.len() as f32;
        }
        bands
    } else {
        Vec::new()
    };
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "sampleCount": samples.len(),
        "fftSize": fft_size,
        "hopSize": hop_size,
        "centroidHz": features.centroid_hz,
        "bandwidthHz": features.bandwidth_hz,
        "rolloffHz": features.rolloff_hz,
        "flatness": features.flatness,
        "dominantFrequencyHz": features.dominant_frequency_hz,
        "zeroCrossingRate": zero_crossing_rate(&samples),
        "melBands": mel_bands,
        "summary": {
            "frameCount": frames.len(),
            "dominantFrequencyHz": features.dominant_frequency_hz,
            "centroidHz": features.centroid_hz,
            "rolloffHz": features.rolloff_hz
        },
        "frames": frames.iter().take(max_frames).map(|frame| serde_json::json!({
            "startSeconds": frame.start_seconds,
            "centroidHz": frame.centroid_hz,
            "bandwidthHz": frame.bandwidth_hz,
            "rolloffHz": frame.rolloff_hz,
            "flatness": frame.flatness,
            "energy": frame.energy,
            "melBands": frame.mel_bands
        })).collect::<Vec<_>>()
    }))
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

fn window_name(input: &serde_json::Value) -> WindowFunction {
    match input.get("window").and_then(serde_json::Value::as_str) {
        Some("rectangular" | "Rectangular") => WindowFunction::Rectangular,
        Some("hamming" | "Hamming") => WindowFunction::Hamming,
        Some("blackman" | "Blackman") => WindowFunction::Blackman,
        _ => WindowFunction::Hann,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_fourier_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.fourier.spectrum"));
        assert!(ids.contains(&"audio.fourier.features"));
    }

    #[test]
    fn spectrum_operation_returns_bins() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.fourier.spectrum"),
            input: serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 4, "fftSize": 4}),
        })
        .expect("spectrum");
        assert_eq!(response.value["operation"], "audio.fourier.spectrum");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert_eq!(response.value["fftSize"], 4);
        assert!(response.value["binCount"].as_u64().unwrap() > 0);
    }

    #[test]
    fn features_operation_returns_frames_and_mel_bands() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.fourier.features"),
            input: serde_json::json!({
                "samples": [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
                "sampleRate": 8,
                "fftSize": 4,
                "hopSize": 2,
                "melBandCount": 3
            }),
        })
        .expect("features");
        assert_eq!(response.value["operation"], "audio.fourier.features");
        assert_eq!(response.value["result"]["summary"]["frameCount"], 3);
        assert_eq!(response.value["melBands"].as_array().unwrap().len(), 3);
        assert_eq!(
            response.value["frames"][0]["melBands"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
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
            operation: OperationId::new("audio.fourier.features"),
            input: serde_json::json!({"samples": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("samples"));
    }
}
