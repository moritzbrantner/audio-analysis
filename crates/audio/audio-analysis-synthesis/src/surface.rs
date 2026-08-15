//! Library-owned runtime surface for `audio-analysis-synthesis`.

use audio_contracts::{AnalysisEvent, AudioBuffer, Timebase, Timestamp};
use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{
    event_to_tone_segment, synthesize_click_track, synthesize_timeline, synthesize_tone,
    AudioSynthesisConfig, ClippingPolicy, ToneSegment, ToneSpec, Waveform,
};

const DEFAULT_PREVIEW_SAMPLES: usize = 1024;

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
                "Deterministic inverse audio generation from symbolic and analyzed events.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.synthesis.tone",
                "Synthesize tone",
                "Generates an in-memory analytic tone frame.",
                serde_json::json!({"frequencyHz": 440.0, "durationSeconds": 0.1, "sampleRate": 48000, "channels": 1}),
            ),
            operation(
                "audio.synthesis.timeline",
                "Synthesize timeline",
                "Generates an in-memory tone timeline from segment specs.",
                serde_json::json!({"sampleRate": 48000, "segments": [{"startSeconds": 0.0, "frequencyHz": 440.0, "durationSeconds": 0.1}]}),
            ),
            operation(
                "audio.synthesis.fromEvents",
                "Synthesize from events",
                "Converts pitch/onset event labels into tone segments and synthesizes them.",
                serde_json::json!({"events": [{"label": "audio:pitch:440.00hz", "score": 0.8, "timestampSeconds": 0.0}], "defaultDurationSeconds": 0.1}),
            ),
            operation(
                "audio.synthesis.clickTrack",
                "Synthesize click track",
                "Generates a deterministic in-memory click track from BPM or explicit beat positions.",
                serde_json::json!({"bpm": 120.0, "durationSeconds": 1.0, "sampleRate": 1000, "clickFrequencyHz": 1800.0, "amplitude": 0.8}),
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
        "audio.synthesis.tone" => tone_value(request.input)?,
        "audio.synthesis.timeline" => timeline_value(request.input)?,
        "audio.synthesis.fromEvents" => from_events_value(request.input)?,
        "audio.synthesis.clickTrack" => click_track_value(request.input)?,
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
            "Synthesis package metadata",
            "Inspected the tone, timeline, and event-to-audio synthesis operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.synthesis.tone" => (
            "Synthesized tone result",
            "Generated an in-memory analytic tone frame.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "channels": value.get("channels").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "durationSeconds": value.get("durationSeconds").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.synthesis.timeline" => (
            "Synthesized timeline result",
            "Generated an in-memory tone timeline from segment specs.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "segmentCount": value.get("segmentCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.synthesis.fromEvents" => (
            "Synthesized event audio result",
            "Converted pitch/onset event labels into tone segments and synthesized them.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "eventCount": value.get("eventCount").cloned().unwrap_or(serde_json::Value::Null),
                "segmentCount": value.get("segmentCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.synthesis.clickTrack" => (
            "Synthesized click track result",
            "Generated a deterministic in-memory click track from BPM or explicit beat positions.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null),
                "beatCount": value.get("beatCount").cloned().unwrap_or(serde_json::Value::Null),
                "durationSeconds": value.get("durationSeconds").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        _ => (
            "Synthesis operation result",
            "Completed the synthesis package surface operation.",
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

fn tone_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let tone = tone_from_input(&input)?;
    let generated =
        synthesize_tone(tone, config_from_input(&input)?).map_err(|error| error.to_string())?;
    frame_value(generated.value, 1)
}

fn timeline_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let segments = input
        .get("segments")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "segments must be an array".to_string())?
        .iter()
        .map(segment_from_value)
        .collect::<Result<Vec<_>, _>>()?;
    let generated = synthesize_timeline(&segments, config_from_input(&input)?)
        .map_err(|error| error.to_string())?;
    frame_value(generated.value, segments.len())
}

fn from_events_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let default_duration_seconds = finite_f32(&input, "defaultDurationSeconds", 0.1)?;
    let events = input
        .get("events")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "events must be an array".to_string())?
        .iter()
        .map(event_from_value)
        .collect::<Result<Vec<_>, _>>()?;
    let segments = events
        .iter()
        .filter_map(|event| event_to_tone_segment(event, default_duration_seconds))
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return Err("events did not contain synthesizable pitch or onset labels".to_string());
    }
    let generated = synthesize_timeline(&segments, config_from_input(&input)?)
        .map_err(|error| error.to_string())?;
    frame_value(generated.value, segments.len())
}

fn click_track_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let duration_seconds = finite_f32(&input, "durationSeconds", 1.0)?;
    let beats = beat_seconds_from_input(&input, duration_seconds)?;
    let click_frequency_hz = finite_f32(&input, "clickFrequencyHz", 1_800.0)?;
    let click_duration_seconds = finite_f32(&input, "clickDurationSeconds", 0.02)?;
    let amplitude = finite_f32(&input, "amplitude", 0.8)?;
    let config = config_from_input(&input)?;
    let generated = synthesize_click_track(
        &beats,
        duration_seconds,
        click_frequency_hz,
        click_duration_seconds,
        amplitude,
        config,
    )
    .map_err(|error| error.to_string())?;
    let mut value = frame_value(generated.value, beats.len())?;
    value["beatCount"] = serde_json::json!(beats.len());
    value["beatSeconds"] = serde_json::json!(beats);
    value["clickFrequencyHz"] = serde_json::json!(click_frequency_hz);
    Ok(value)
}

fn beat_seconds_from_input(
    input: &serde_json::Value,
    duration_seconds: f32,
) -> Result<Vec<f32>, String> {
    if let Some(beat_grid) = input.get("beatGrid").and_then(serde_json::Value::as_array) {
        return beat_grid
            .iter()
            .map(|value| {
                let beat = value
                    .as_f64()
                    .ok_or_else(|| "beatGrid values must be numbers".to_string())?
                    as f32;
                if beat.is_finite() && beat >= 0.0 {
                    Ok(beat)
                } else {
                    Err("beatGrid values must be finite and non-negative".to_string())
                }
            })
            .collect();
    }
    let bpm = finite_f32(input, "bpm", 120.0)?;
    if bpm <= 0.0 {
        return Err("bpm must be greater than zero".to_string());
    }
    let step = 60.0 / bpm;
    let mut beats = Vec::new();
    let mut current = 0.0;
    while current < duration_seconds {
        beats.push(current);
        current += step;
    }
    Ok(beats)
}

fn frame_value(
    frame: audio_contracts::OwnedAudioFrame,
    segment_count: usize,
) -> Result<serde_json::Value, String> {
    let samples = match frame.data {
        AudioBuffer::F32(samples) => samples,
        _ => return Err("synthesis output was not f32".to_string()),
    };
    Ok(serde_json::json!({
        "sampleRate": frame.sample_rate,
        "channels": frame.channels,
        "sampleCount": samples.len(),
        "samplesPerChannel": samples.len() / usize::from(frame.channels),
        "durationSeconds": samples.len() as f64 / frame.sample_rate as f64 / f64::from(frame.channels),
        "segmentCount": segment_count,
        "samplePreview": samples.into_iter().take(DEFAULT_PREVIEW_SAMPLES).collect::<Vec<_>>()
    }))
}

fn tone_from_input(input: &serde_json::Value) -> Result<ToneSpec, String> {
    let tone = ToneSpec {
        frequency_hz: finite_f32(input, "frequencyHz", 440.0)?,
        duration_seconds: finite_f32(input, "durationSeconds", 0.1)?,
        amplitude: finite_f32(input, "amplitude", 0.5)?,
        waveform: waveform(input.get("waveform").and_then(serde_json::Value::as_str)),
    };
    tone.validate().map_err(|error| error.to_string())?;
    Ok(tone)
}

fn segment_from_value(value: &serde_json::Value) -> Result<ToneSegment, String> {
    let segment = ToneSegment {
        start_seconds: finite_f32(value, "startSeconds", 0.0)?,
        tone: tone_from_input(value)?,
    };
    segment.validate().map_err(|error| error.to_string())?;
    Ok(segment)
}

fn event_from_value(value: &serde_json::Value) -> Result<AnalysisEvent, String> {
    let label = value
        .get("label")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "event label must be a string".to_string())?;
    let mut event = AnalysisEvent::new("surface", label);
    event.score = value
        .get("score")
        .and_then(serde_json::Value::as_f64)
        .map(|score| score as f32);
    if let Some(seconds) = value
        .get("timestampSeconds")
        .and_then(serde_json::Value::as_f64)
    {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err("timestampSeconds must be finite and non-negative".to_string());
        }
        let sample_rate = value
            .get("sampleRate")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(48_000) as i32;
        event.timestamp = Some(Timestamp::new(
            (seconds * f64::from(sample_rate)).round() as i64,
            Timebase::new(1, sample_rate),
        ));
    }
    Ok(event)
}

fn config_from_input(input: &serde_json::Value) -> Result<AudioSynthesisConfig, String> {
    let sample_rate = input
        .get("sampleRate")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(48_000);
    let channels = input
        .get("channels")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);
    let config = AudioSynthesisConfig::new(sample_rate as u32, channels as u16)
        .map_err(|error| error.to_string())?;
    Ok(
        match input.get("clipping").and_then(serde_json::Value::as_str) {
            Some("normalize") => config.clipping_policy(ClippingPolicy::Normalize),
            _ => config,
        },
    )
}

fn waveform(name: Option<&str>) -> Waveform {
    match name {
        Some("square") => Waveform::Square,
        Some("saw") => Waveform::Saw,
        Some("triangle") => Waveform::Triangle,
        Some("pulse") => Waveform::Pulse,
        _ => Waveform::Sine,
    }
}

fn finite_f32(input: &serde_json::Value, field: &str, default_value: f32) -> Result<f32, String> {
    let value = input
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default_value as f64) as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{field} must be finite"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_synthesis_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.synthesis.tone"));
        assert!(ids.contains(&"audio.synthesis.timeline"));
        assert!(ids.contains(&"audio.synthesis.clickTrack"));
    }

    #[test]
    fn tone_operation_returns_samples() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.synthesis.tone"),
            input: serde_json::json!({"frequencyHz": 440.0, "durationSeconds": 0.01, "sampleRate": 1000}),
        })
        .expect("tone");
        assert_eq!(response.value["operation"], "audio.synthesis.tone");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert!(response.value["sampleCount"].as_u64().unwrap() > 0);
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
    fn invalid_tone_returns_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.synthesis.tone"),
            input: serde_json::json!({"frequencyHz": -1.0}),
        })
        .unwrap_err();
        assert!(error.contains("frequency"));
    }

    #[test]
    fn click_track_operation_places_beats() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.synthesis.clickTrack"),
            input: serde_json::json!({"beatGrid": [0.0, 0.5], "durationSeconds": 1.0, "sampleRate": 1000}),
        })
        .expect("click track");
        assert_eq!(response.value["operation"], "audio.synthesis.clickTrack");
        assert_eq!(response.value["sampleCount"], 1000);
        assert_eq!(response.value["beatCount"], 2);
        let preview = response.value["samplePreview"].as_array().unwrap();
        assert!(preview[0].as_f64().unwrap() > 0.0);
        assert!(preview[500].as_f64().unwrap() > 0.0);
    }
}
