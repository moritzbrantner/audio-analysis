//! Library-owned runtime surface for `audio-analysis-pitch`.

use audio_analysis_core::FrameSpec;
use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{
    frequency_to_midi_note, frequency_to_note_name, pitch_class_summary, segment_pitch_track,
    AutocorrelationPitchDetector, PitchDetectorConfig, PitchFrameEstimate,
};

const MAX_SAMPLES: usize = 192_000;

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
                "Autocorrelation pitch detection and note projection for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.pitch.estimate",
                "Estimate pitch",
                "Estimates one fundamental frequency from normalized samples.",
                serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 48000}),
            ),
            operation(
                "audio.pitch.track",
                "Pitch track",
                "Estimates pitch over fixed frames and groups contiguous note segments.",
                serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 48000, "frameSize": 2048, "hopSize": 512}),
            ),
            operation(
                "audio.pitch.noteName",
                "Inspect note name",
                "Inspects the MIDI note and scientific note name for a frequency in hertz.",
                serde_json::json!({"frequencyHz": 440.0}),
            ),
            operation(
                "audio.pitch.chroma",
                "Chroma",
                "Summarizes normalized samples into a 12-bin pitch-class chroma vector.",
                serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 48000}),
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
        "audio.pitch.estimate" => estimate_value(request.input)?,
        "audio.pitch.track" => track_value(request.input)?,
        "audio.pitch.noteName" => note_name_value(request.input)?,
        "audio.pitch.chroma" => chroma_value(request.input)?,
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
            "Pitch package metadata",
            "Inspected the pitch detection and note projection operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.pitch.estimate" => (
            "Pitch estimate result",
            "Estimated one fundamental frequency from normalized audio samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "frequencyHz": value.get("frequencyHz").cloned().unwrap_or(serde_json::Value::Null),
                "confidence": value.get("confidence").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.pitch.track" => (
            "Pitch track result",
            "Estimated pitch over fixed frames and grouped contiguous note segments.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "frameCount": value.get("frameCount").cloned().unwrap_or(serde_json::Value::Null),
                "segmentCount": value.get("segments").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.pitch.noteName" => (
            "Pitch note name",
            "Inspected the MIDI note and scientific note name for the supplied frequency.",
            serde_json::json!({
                "frequencyHz": value.get("frequencyHz").cloned().unwrap_or(serde_json::Value::Null),
                "midiNote": value.get("midiNote").cloned().unwrap_or(serde_json::Value::Null),
                "noteName": value.get("noteName").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.pitch.chroma" => (
            "Pitch chroma result",
            "Summarized normalized samples into a 12-bin pitch-class chroma vector.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "strongestPitchClass": value.get("strongestPitchClass").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        _ => (
            "Pitch operation result",
            "Completed the pitch package surface operation.",
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

fn estimate_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let detector = AutocorrelationPitchDetector::new(config_from_input(&input)?)
        .map_err(|error| error.to_string())?;
    let estimate = detector
        .estimate_samples(&samples, sample_rate)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "sampleCount": samples.len(),
        "frequencyHz": estimate.frequency_hz,
        "confidence": estimate.confidence,
        "midiNote": estimate.midi_note(),
        "noteName": estimate.note_name()
    }))
}

fn track_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let frame_size = positive_usize(&input, "frameSize", 2048)?;
    let hop_size = positive_usize(&input, "hopSize", frame_size / 2)?;
    let max_frames = positive_usize(&input, "maxFrames", 64)?.min(64);
    let frame_spec = FrameSpec::new(frame_size, hop_size).map_err(|error| error.to_string())?;
    let detector = AutocorrelationPitchDetector::new(config_from_input(&input)?)
        .map_err(|error| error.to_string())?;
    let mut estimates = Vec::new();
    for (start_sample, frame) in frame_spec.frames(&samples).take(max_frames) {
        let estimate = detector
            .estimate_samples(frame, sample_rate)
            .map_err(|error| error.to_string())?;
        let start_seconds = start_sample as f64 / sample_rate as f64;
        let end_seconds = start_seconds + frame.len() as f64 / sample_rate as f64;
        estimates.push(PitchFrameEstimate {
            start_seconds,
            end_seconds,
            frequency_hz: estimate.frequency_hz,
            confidence: estimate.confidence,
        });
    }
    let segments = segment_pitch_track(&estimates, 0.05, 0.0);
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "sampleCount": samples.len(),
        "frameSize": frame_size,
        "hopSize": hop_size,
        "frames": estimates.iter().map(|estimate| serde_json::json!({
            "startSeconds": estimate.start_seconds,
            "endSeconds": estimate.end_seconds,
            "frequencyHz": estimate.frequency_hz,
            "confidence": estimate.confidence,
            "noteName": estimate.note_name()
        })).collect::<Vec<_>>(),
        "segments": segments.iter().map(|segment| serde_json::json!({
            "startSeconds": segment.start_seconds,
            "endSeconds": segment.end_seconds,
            "frequencyHz": segment.frequency_hz,
            "midiNote": segment.midi_note,
            "noteName": segment.note_name,
            "confidence": segment.confidence,
            "frames": segment.frames
        })).collect::<Vec<_>>()
    }))
}

fn note_name_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let frequency_hz = input
        .get("frequencyHz")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| "frequencyHz must be a number".to_string())? as f32;
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err("frequencyHz must be finite and positive".to_string());
    }
    Ok(serde_json::json!({
        "frequencyHz": frequency_hz,
        "midiNote": frequency_to_midi_note(frequency_hz),
        "noteName": frequency_to_note_name(frequency_hz)
    }))
}

fn chroma_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let summary = pitch_class_summary(&samples, sample_rate).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "sampleCount": samples.len(),
        "chroma": summary.chroma.bins,
        "pitchClasses": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        "strongestPitchClass": summary.strongest_pitch_class.map(|note| note.as_str())
    }))
}

fn config_from_input(input: &serde_json::Value) -> Result<PitchDetectorConfig, String> {
    let mut config = PitchDetectorConfig::default();
    if let Some(value) = input
        .get("minFrequencyHz")
        .and_then(serde_json::Value::as_f64)
    {
        config.min_frequency_hz = value as f32;
    }
    if let Some(value) = input
        .get("maxFrequencyHz")
        .and_then(serde_json::Value::as_f64)
    {
        config.max_frequency_hz = value as f32;
    }
    if let Some(value) = input
        .get("confidenceThreshold")
        .and_then(serde_json::Value::as_f64)
    {
        config.confidence_threshold = value as f32;
    }
    config.validate().map_err(|error| error.to_string())?;
    Ok(config)
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
    fn package_surface_lists_pitch_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.pitch.estimate"));
        assert!(ids.contains(&"audio.pitch.noteName"));
        assert!(ids.contains(&"audio.pitch.chroma"));
    }

    #[test]
    fn note_name_operation_returns_a4() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.pitch.noteName"),
            input: serde_json::json!({"frequencyHz": 440.0}),
        })
        .expect("note");
        assert_eq!(response.value["operation"], "audio.pitch.noteName");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert_eq!(response.value["noteName"], "A4");
    }

    #[test]
    fn chroma_operation_returns_a_pitch_class() {
        let samples = (0..1024)
            .map(|index| {
                let t = index as f32 / 8_192.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect::<Vec<_>>();
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.pitch.chroma"),
            input: serde_json::json!({"samples": samples, "sampleRate": 8192}),
        })
        .expect("chroma");
        assert_eq!(response.value["operation"], "audio.pitch.chroma");
        assert_eq!(response.value["strongestPitchClass"], "A");
        assert_eq!(response.value["chroma"].as_array().unwrap().len(), 12);
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
            operation: OperationId::new("audio.pitch.estimate"),
            input: serde_json::json!({"samples": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("samples"));
    }
}
