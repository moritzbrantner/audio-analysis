//! Library-owned runtime surface for `audio-generation-midi`.

use audio_analysis_synthesis::{AudioSynthesisConfig, Waveform};
use audio_contracts::AudioBuffer;
use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{
    pitch_track_to_midi_notes, MidiNote, MidiNoteEvent, MidiSong, MidiTrack, NoteName,
    PitchTrackFrame, PitchTrackMidiOptions,
};

const DEFAULT_PREVIEW_BYTES: usize = 64;
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
                "MIDI-like note sequencing, Standard MIDI encoding, and deterministic audio rendering.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.midi.note",
                "Inspect MIDI note",
                "Inspects frequency metadata for a MIDI note number or note name.",
                serde_json::json!({"note": 69}),
            ),
            operation(
                "audio.midi.encode",
                "Encode MIDI",
                "Encodes a deterministic single-track MIDI byte stream and returns a byte summary.",
                serde_json::json!({"tempoBpm": 120.0, "notes": [{"note": 69, "startBeats": 0.0, "durationBeats": 1.0}]}),
            ),
            operation(
                "audio.midi.render",
                "Render MIDI",
                "Renders a MIDI-like note sequence into deterministic in-memory audio samples.",
                serde_json::json!({"tempoBpm": 120.0, "sampleRate": 48000, "notes": [{"note": 69, "startBeats": 0.0, "durationBeats": 1.0}]}),
            ),
            operation(
                "audio.midi.fromPitchTrack",
                "Convert pitch track",
                "Converts pitch-track frames into merged MIDI-like note events and a byte summary.",
                serde_json::json!({"tempoBpm": 120.0, "minNoteDurationSeconds": 0.05, "pitchTrack": [{"startSeconds": 0.0, "endSeconds": 0.5, "frequencyHz": 440.0, "confidence": 0.9}, {"startSeconds": 0.5, "endSeconds": 1.0, "frequencyHz": 440.0, "confidence": 0.9}]}),
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
        "audio.midi.note" => note_value(request.input)?,
        "audio.midi.encode" => encode_value(request.input)?,
        "audio.midi.render" => render_value(request.input)?,
        "audio.midi.fromPitchTrack" => from_pitch_track_value(request.input)?,
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
            "MIDI package metadata",
            "Inspected the MIDI note, encoding, and audio rendering operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.midi.note" => (
            "MIDI note metadata",
            "Inspected frequency metadata for the requested MIDI note.",
            serde_json::json!({
                "note": value.get("note").cloned().unwrap_or(serde_json::Value::Null),
                "frequencyHz": value.get("frequencyHz").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.midi.encode" => (
            "MIDI encoding result",
            "Encoded a deterministic single-track MIDI byte stream and returned a byte summary.",
            serde_json::json!({
                "tempoBpm": value.get("tempoBpm").cloned().unwrap_or(serde_json::Value::Null),
                "noteCount": value.get("noteCount").cloned().unwrap_or(serde_json::Value::Null),
                "byteCount": value.get("byteCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.midi.render" => (
            "MIDI audio render result",
            "Rendered a MIDI-like note sequence into deterministic in-memory audio samples.",
            serde_json::json!({
                "tempoBpm": value.get("tempoBpm").cloned().unwrap_or(serde_json::Value::Null),
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "sampleCount": value.get("sampleCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.midi.fromPitchTrack" => (
            "Pitch-track MIDI result",
            "Converted pitch-track frames into merged MIDI-like note events and a byte summary.",
            serde_json::json!({
                "tempoBpm": value.get("tempoBpm").cloned().unwrap_or(serde_json::Value::Null),
                "noteCount": value.get("noteCount").cloned().unwrap_or(serde_json::Value::Null),
                "byteLength": value.get("byteLength").cloned().unwrap_or(serde_json::Value::Null),
                "diagnosticCount": value.get("diagnostics").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        _ => (
            "MIDI operation result",
            "Completed the MIDI generation package surface operation.",
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

fn note_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let note = midi_note_from_value(&input)?;
    Ok(serde_json::json!({
        "note": note.value(),
        "frequencyHz": note.frequency_hz()
    }))
}

fn encode_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let song = song_from_input(&input)?;
    let bytes = song.to_midi_bytes().map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "tempoBpm": song.tempo_bpm,
        "trackCount": song.tracks().len(),
        "byteLength": bytes.len(),
        "bytePreview": bytes.iter().copied().take(DEFAULT_PREVIEW_BYTES).collect::<Vec<_>>()
    }))
}

fn render_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let song = song_from_input(&input)?;
    let sample_rate = input
        .get("sampleRate")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(48_000) as u32;
    let channels = input
        .get("channels")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1) as u16;
    let generated = song
        .render(
            AudioSynthesisConfig::new(sample_rate, channels).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let samples = match generated.value.data {
        AudioBuffer::F32(samples) => samples,
        _ => return Err("render output was not f32".to_string()),
    };
    Ok(serde_json::json!({
        "tempoBpm": song.tempo_bpm,
        "trackCount": song.tracks().len(),
        "sampleRate": generated.value.sample_rate,
        "channels": generated.value.channels,
        "sampleCount": samples.len(),
        "samplesPerChannel": samples.len() / usize::from(generated.value.channels),
        "samplePreview": samples.into_iter().take(DEFAULT_PREVIEW_SAMPLES).collect::<Vec<_>>()
    }))
}

fn from_pitch_track_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let options = PitchTrackMidiOptions {
        tempo_bpm: finite_f32(&input, "tempoBpm", 120.0)?,
        quantization_beats: input
            .get("quantizationBeats")
            .map(|_| finite_f32(&input, "quantizationBeats", 0.0))
            .transpose()?,
        min_note_duration_seconds: finite_f32(&input, "minNoteDurationSeconds", 0.05)?,
        velocity: input
            .get("velocity")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(100) as u8,
    };
    let frames = input
        .get("pitchTrack")
        .or_else(|| input.get("frames"))
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "pitchTrack must be an array".to_string())?
        .iter()
        .map(pitch_frame_from_value)
        .collect::<Result<Vec<_>, _>>()?;
    let result = pitch_track_to_midi_notes(&frames, options).map_err(|error| error.to_string())?;
    let mut track = MidiTrack::new("pitch-track");
    for note in &result.notes {
        track.push(*note).map_err(|error| error.to_string())?;
    }
    let song = MidiSong::new(options.tempo_bpm)
        .map_err(|error| error.to_string())?
        .with_track(track)
        .map_err(|error| error.to_string())?;
    let bytes = song.to_midi_bytes().map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "tempoBpm": options.tempo_bpm,
        "noteCount": result.notes.len(),
        "midiByteLength": bytes.len(),
        "byteLength": bytes.len(),
        "bytePreview": bytes.iter().copied().take(DEFAULT_PREVIEW_BYTES).collect::<Vec<_>>(),
        "notes": result.notes.iter().map(|note| serde_json::json!({
            "note": note.note.value(),
            "frequencyHz": note.note.frequency_hz(),
            "startBeats": note.start_beats,
            "durationBeats": note.duration_beats,
            "velocity": note.velocity,
            "channel": note.channel
        })).collect::<Vec<_>>(),
        "diagnostics": result.diagnostics
    }))
}

fn pitch_frame_from_value(input: &serde_json::Value) -> Result<PitchTrackFrame, String> {
    let frame = PitchTrackFrame {
        start_seconds: finite_f32(input, "startSeconds", 0.0)?,
        end_seconds: finite_f32(input, "endSeconds", 0.0)?,
        frequency_hz: finite_f32(input, "frequencyHz", 0.0)?,
        confidence: finite_f32(input, "confidence", 1.0)?,
    };
    frame.validate().map_err(|error| error.to_string())?;
    Ok(frame)
}

fn song_from_input(input: &serde_json::Value) -> Result<MidiSong, String> {
    let tempo_bpm = finite_f32(input, "tempoBpm", 120.0)?;
    let mut track = MidiTrack::new(
        input
            .get("trackName")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("surface"),
    )
    .waveform(waveform(
        input.get("waveform").and_then(serde_json::Value::as_str),
    ));
    let notes = input
        .get("notes")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "notes must be an array".to_string())?;
    for note in notes {
        track
            .push(note_event_from_value(note)?)
            .map_err(|error| error.to_string())?;
    }
    MidiSong::new(tempo_bpm)
        .map_err(|error| error.to_string())?
        .with_track(track)
        .map_err(|error| error.to_string())
}

fn note_event_from_value(input: &serde_json::Value) -> Result<MidiNoteEvent, String> {
    let note = midi_note_from_value(input)?;
    MidiNoteEvent::new(
        note,
        finite_f32(input, "startBeats", 0.0)?,
        finite_f32(input, "durationBeats", 1.0)?,
    )
    .map_err(|error| error.to_string())?
    .velocity(
        input
            .get("velocity")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(100) as u8,
    )
    .map_err(|error| error.to_string())
}

fn midi_note_from_value(input: &serde_json::Value) -> Result<MidiNote, String> {
    if let Some(value) = input.get("note").and_then(serde_json::Value::as_u64) {
        return MidiNote::new(value as u8).map_err(|error| error.to_string());
    }
    let name = input
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("A");
    let octave = input
        .get("octave")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(4) as i8;
    MidiNote::from_name(note_name(name)?, octave).map_err(|error| error.to_string())
}

fn note_name(name: &str) -> Result<NoteName, String> {
    match name {
        "C" => Ok(NoteName::C),
        "Cs" | "C#" | "Db" => Ok(NoteName::Cs),
        "D" => Ok(NoteName::D),
        "Ds" | "D#" | "Eb" => Ok(NoteName::Ds),
        "E" => Ok(NoteName::E),
        "F" => Ok(NoteName::F),
        "Fs" | "F#" | "Gb" => Ok(NoteName::Fs),
        "G" => Ok(NoteName::G),
        "Gs" | "G#" | "Ab" => Ok(NoteName::Gs),
        "A" => Ok(NoteName::A),
        "As" | "A#" | "Bb" => Ok(NoteName::As),
        "B" => Ok(NoteName::B),
        _ => Err("unsupported note name".to_string()),
    }
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
    fn package_surface_lists_midi_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.midi.note"));
        assert!(ids.contains(&"audio.midi.encode"));
        assert!(ids.contains(&"audio.midi.fromPitchTrack"));
    }

    #[test]
    fn note_operation_returns_a4_frequency() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.midi.note"),
            input: serde_json::json!({"note": 69}),
        })
        .expect("note");
        assert_eq!(response.value["operation"], "audio.midi.note");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert_eq!(response.value["frequencyHz"], 440.0);
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
    fn invalid_note_returns_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.midi.note"),
            input: serde_json::json!({"name": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("note"));
    }

    #[test]
    fn pitch_track_operation_merges_a4_frames() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.midi.fromPitchTrack"),
            input: serde_json::json!({
                "tempoBpm": 120.0,
                "pitchTrack": [
                    {"startSeconds": 0.0, "endSeconds": 0.5, "frequencyHz": 440.0, "confidence": 0.9},
                    {"startSeconds": 0.5, "endSeconds": 1.0, "frequencyHz": 440.0, "confidence": 0.9},
                    {"startSeconds": 1.0, "endSeconds": 1.01, "frequencyHz": 493.88, "confidence": 0.9}
                ]
            }),
        })
        .expect("pitch track");
        assert_eq!(response.value["operation"], "audio.midi.fromPitchTrack");
        assert_eq!(response.value["noteCount"], 1);
        assert_eq!(response.value["notes"][0]["note"], 69);
        assert!(response.value["diagnostics"].as_array().unwrap().len() >= 2);
    }
}
