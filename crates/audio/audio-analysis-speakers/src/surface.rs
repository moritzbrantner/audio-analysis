//! Library-owned runtime surface for `audio-analysis-speakers`.

use audio_analysis_recognition::{AudioRuntime, SpectralEmbeddingConfig};
use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{
    assign_speakers_to_transcript_with_policy, diarize_speakers, DiarizedSpeaker, EnergyVadConfig,
    EnergyVoiceActivityDetector, SpeakerAudio, SpeakerDiarizationRequest,
    SpeakerDiarizationResponse, SpeakerDiarizer, SpeakerEmbeddingExtractor, SpeakerId,
    SpeakerIdentificationOptions, SpeakerLabel, SpeakerLibrary, SpeakerSegmentPrediction,
    SpeakerTranscriptAssignmentPolicy, SpectralSpeakerEmbedder, TranscriptionContract,
    VoiceActivityDetector, WindowedSpeakerDiarizer,
};

const MAX_SAMPLES: usize = 192_000;
const DEFAULT_PREVIEW_VALUES: usize = 32;

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
                "Speaker embeddings, enrollment, identification, VAD, and diarization APIs for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.speakers.embed",
                "Embed speaker",
                "Computes a deterministic spectral speaker embedding from normalized samples.",
                serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 48000, "channels": 1}),
            ),
            operation(
                "audio.speakers.identify",
                "Identify speaker",
                "Builds a transient enrolled-speaker library and identifies a query embedding.",
                serde_json::json!({"querySamples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 48000, "speakers": [{"id": "speaker-1", "label": "Speaker 1", "samples": [0.0, 1.0, 0.0, -1.0]}]}),
            ),
            operation(
                "audio.speakers.assignTranscript",
                "Assign transcript speakers",
                "Applies diarization segments to an existing transcription contract.",
                serde_json::json!({"overlapPolicy": "majority", "transcript": {"segments": [{"index": 0, "text": "hello", "startSeconds": 0.0, "endSeconds": 1.0, "isFinal": true}]}, "diarization": {"accepted": true, "operation": "diarize", "modelId": "single-speaker-heuristic", "runtime": "heuristic", "segments": [{"speaker": "speaker_0", "startSeconds": 0.0, "endSeconds": 1.0, "score": 1.0}]}}),
            ),
            operation(
                "audio.speakers.vad",
                "Voice activity",
                "Detects speech-like regions with a deterministic RMS voice activity detector.",
                serde_json::json!({"samples": [0.0, 0.2, 0.2, 0.0], "sampleRate": 4, "channels": 1, "frameSize": 2, "hopSize": 1, "threshold": 0.01, "minSpeechSeconds": 0.0, "minSilenceSeconds": 0.0}),
            ),
            operation(
                "audio.speakers.diarize",
                "Diarize speakers",
                "Runs deterministic VAD/window/spectral speaker diarization or normalizes imported diarization segments.",
                serde_json::json!({"samples": [0.0, 0.2, 0.2, 0.0, -0.2, -0.2, 0.0], "sampleRate": 4, "channels": 1, "frameSize": 2, "hopSize": 1, "threshold": 0.01, "minSpeechSeconds": 0.0, "minSilenceSeconds": 0.0}),
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
        "audio.speakers.embed" => embed_value(request.input)?,
        "audio.speakers.identify" => identify_value(request.input)?,
        "audio.speakers.assignTranscript" => assign_transcript_value(request.input)?,
        "audio.speakers.vad" => vad_value(request.input)?,
        "audio.speakers.diarize" => diarize_value(request.input)?,
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
            "Speaker package metadata",
            "Inspected the speaker embedding, identification, and transcript assignment operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.speakers.embed" => (
            "Speaker embedding result",
            "Computed a deterministic spectral speaker embedding from normalized samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "channels": value.get("channels").cloned().unwrap_or(serde_json::Value::Null),
                "dimensions": value.pointer("/model/dimensions").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.speakers.identify" => (
            "Speaker identification result",
            "Built a transient enrolled-speaker library and identified the query embedding.",
            serde_json::json!({
                "speakerCount": value.get("speakerCount").cloned().unwrap_or(serde_json::Value::Null),
                "matchCount": value.get("matches").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.speakers.assignTranscript" => (
            "Transcript speaker assignment result",
            "Applied diarization segments to an existing transcription contract.",
            serde_json::json!({
                "segmentCount": value.get("segments").and_then(serde_json::Value::as_array).map_or(0, Vec::len),
                "accepted": value.get("accepted").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.speakers.vad" => (
            "Voice activity result",
            "Detected speech-like regions with a deterministic RMS voice activity detector.",
            serde_json::json!({
                "segmentCount": value.pointer("/summary/segmentCount").cloned().unwrap_or(serde_json::Value::Null),
                "speechSeconds": value.pointer("/summary/speechSeconds").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.speakers.diarize" => (
            "Speaker diarization result",
            "Ran deterministic VAD/window/spectral diarization or normalized imported diarization segments.",
            serde_json::json!({
                "speakerCount": value.get("speakerCount").cloned().unwrap_or(serde_json::Value::Null),
                "segmentCount": value.get("segments").and_then(serde_json::Value::as_array).map_or(0, Vec::len),
                "runtime": value.get("runtime").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        _ => (
            "Speaker operation result",
            "Completed the speaker package surface operation.",
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

fn embed_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let mut embedder = speaker_embedder(&input)?;
    let audio = SpeakerAudio::interleaved(&samples, sample_rate, channels)
        .map_err(|error| error.to_string())?;
    let embedding = embedder
        .embed_speaker(&audio)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "channels": channels,
        "sampleCount": samples.len(),
        "durationSeconds": audio.duration_seconds(),
        "model": {
            "name": embedding.model().name,
            "version": embedding.model().version,
            "dimensions": embedding.dimensions()
        },
        "valuesPreview": embedding.values().iter().copied().take(DEFAULT_PREVIEW_VALUES).collect::<Vec<_>>()
    }))
}

fn identify_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let query = sample_array(&input, "querySamples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let mut embedder = speaker_embedder(&input)?;
    let query_audio = SpeakerAudio::interleaved(&query, sample_rate, channels)
        .map_err(|error| error.to_string())?;
    let query_embedding = embedder
        .embed_speaker(&query_audio)
        .map_err(|error| error.to_string())?;
    let mut library = SpeakerLibrary::new();
    let speakers = input
        .get("speakers")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "speakers must be an array".to_string())?;
    for speaker in speakers {
        let id = speaker
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "speaker id must be a string".to_string())?;
        let label = speaker
            .get("label")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(id);
        let samples = sample_array(speaker, "samples")?;
        let audio = SpeakerAudio::interleaved(&samples, sample_rate, channels)
            .map_err(|error| error.to_string())?;
        library
            .enroll(
                SpeakerId::new(id).map_err(|error| error.to_string())?,
                SpeakerLabel::new(label).map_err(|error| error.to_string())?,
                &audio,
                &mut embedder,
            )
            .map_err(|error| error.to_string())?;
    }
    let result = library
        .identify(&query_embedding, &SpeakerIdentificationOptions::default())
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "speakerCount": library.len(),
        "unknown": result.unknown,
        "margin": result.margin,
        "bestMatch": result.best_match.as_ref().map(|matched| serde_json::json!({
            "speakerId": matched.speaker_id.as_str(),
            "label": matched.label.as_str(),
            "score": matched.score,
            "confidence": format!("{:?}", matched.confidence)
        })),
        "matches": result.ranked_matches.iter().map(|matched| serde_json::json!({
            "speakerId": matched.speaker_id.as_str(),
            "label": matched.label.as_str(),
            "score": matched.score,
            "margin": matched.margin,
            "confidence": format!("{:?}", matched.confidence)
        })).collect::<Vec<_>>()
    }))
}

fn assign_transcript_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let transcript: TranscriptionContract = serde_json::from_value(
        input
            .get("transcript")
            .cloned()
            .ok_or_else(|| "transcript is required".to_string())?,
    )
    .map_err(|error| format!("invalid transcript: {error}"))?;
    let diarization: SpeakerDiarizationResponse = serde_json::from_value(
        input
            .get("diarization")
            .cloned()
            .ok_or_else(|| "diarization is required".to_string())?,
    )
    .map_err(|error| format!("invalid diarization: {error}"))?;
    let policy = overlap_policy_from_input(&input)?;
    let assigned = assign_speakers_to_transcript_with_policy(&transcript, &diarization, policy)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "accepted": true,
        "overlapPolicy": policy,
        "segmentCount": assigned.segments.len(),
        "segments": assigned.segments,
        "transcript": assigned
    }))
}

fn overlap_policy_from_input(
    input: &serde_json::Value,
) -> Result<SpeakerTranscriptAssignmentPolicy, String> {
    let Some(value) = input.get("overlapPolicy") else {
        return Ok(SpeakerTranscriptAssignmentPolicy::Majority);
    };
    serde_json::from_value(value.clone()).map_err(|error| format!("invalid overlapPolicy: {error}"))
}

fn vad_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let audio = SpeakerAudio::interleaved(&samples, sample_rate, channels)
        .map_err(|error| error.to_string())?;
    let config = vad_config_from_input(&input, sample_rate)?;
    let mut vad = EnergyVoiceActivityDetector::new(config).map_err(|error| error.to_string())?;
    let spans = vad
        .detect_speech(&audio)
        .map_err(|error| error.to_string())?;
    let speech_seconds = spans
        .iter()
        .map(|span| span.duration_seconds())
        .sum::<f64>();
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "channels": channels,
        "sampleCount": samples.len(),
        "summary": {
            "segmentCount": spans.len(),
            "speechSeconds": speech_seconds
        },
        "segments": spans.iter().map(|span| serde_json::json!({
            "startSeconds": span.start_seconds,
            "endSeconds": span.end_seconds,
            "score": span.score
        })).collect::<Vec<_>>()
    }))
}

fn diarize_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    if input.get("samples").is_none() {
        let request: SpeakerDiarizationRequest = serde_json::from_value(input)
            .map_err(|error| format!("invalid diarization request: {error}"))?;
        let response = diarize_speakers(request).map_err(|error| error.to_string())?;
        let speaker_count = unique_speaker_count(&response.segments);
        return Ok(serde_json::json!({
            "accepted": response.accepted,
            "operation": response.operation,
            "modelId": response.model_id,
            "runtime": response.runtime,
            "speakerCount": speaker_count,
            "segments": response.segments,
            "diagnostics": ["Imported segments or heuristic fallback were normalized without native diarization."]
        }));
    }

    let samples = sample_array(&input, "samples")?;
    let sample_rate = sample_rate(&input)?;
    let channels = channels(&input)?;
    let audio = SpeakerAudio::interleaved(&samples, sample_rate, channels)
        .map_err(|error| error.to_string())?;
    let vad = EnergyVoiceActivityDetector::new(vad_config_from_input(&input, sample_rate)?)
        .map_err(|error| error.to_string())?;
    let embedder = speaker_embedder(&input)?;
    let mut diarizer = WindowedSpeakerDiarizer::new(embedder, vad);
    if let Some(threshold) = finite_f32(&input, "clusterThreshold")? {
        diarizer = diarizer
            .cluster_threshold(threshold)
            .map_err(|error| error.to_string())?;
    }
    let result = diarizer
        .diarize(&audio)
        .map_err(|error| error.to_string())?;
    let segments = result
        .segments
        .into_iter()
        .map(|segment| SpeakerSegmentPrediction {
            speaker: diarized_speaker_label(segment.speaker),
            start_seconds: segment.start_seconds as f32,
            end_seconds: segment.end_seconds as f32,
            score: Some(segment.score),
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "accepted": true,
        "operation": "diarize",
        "modelId": "spectral-speaker-baseline",
        "runtime": AudioRuntime::Heuristic,
        "speakerCount": unique_speaker_count(&segments),
        "segments": segments,
        "diagnostics": ["Deterministic baseline diarization uses RMS VAD and spectral speaker embeddings; it is intended for tests and prototypes."]
    }))
}

fn diarized_speaker_label(speaker: DiarizedSpeaker) -> String {
    match speaker {
        DiarizedSpeaker::Known(id) => id.as_str().to_string(),
        DiarizedSpeaker::Unknown(label) => label,
    }
}

fn unique_speaker_count(segments: &[SpeakerSegmentPrediction]) -> usize {
    let mut speakers = std::collections::BTreeSet::new();
    for segment in segments {
        speakers.insert(segment.speaker.as_str());
    }
    speakers.len()
}

fn speaker_embedder(input: &serde_json::Value) -> Result<SpectralSpeakerEmbedder, String> {
    let fft_size = positive_usize(input, "fftSize", 512)?;
    let hop_size = positive_usize(input, "hopSize", fft_size / 2)?;
    let bands = positive_usize(input, "bands", 8)?;
    SpectralSpeakerEmbedder::new(
        SpectralEmbeddingConfig::new(fft_size, hop_size, bands)
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn vad_config_from_input(
    input: &serde_json::Value,
    sample_rate: u32,
) -> Result<EnergyVadConfig, String> {
    let frame_size = positive_usize(
        input,
        "frameSize",
        ((sample_rate as f64) * EnergyVadConfig::default().frame_seconds).round() as usize,
    )?;
    let hop_size = positive_usize(
        input,
        "hopSize",
        ((sample_rate as f64) * EnergyVadConfig::default().hop_seconds).round() as usize,
    )?;
    let config = EnergyVadConfig {
        rms_threshold: finite_f32(input, "threshold")?
            .unwrap_or(EnergyVadConfig::default().rms_threshold),
        frame_seconds: frame_size as f64 / sample_rate as f64,
        hop_seconds: hop_size as f64 / sample_rate as f64,
        min_speech_seconds: finite_f64(input, "minSpeechSeconds")?
            .unwrap_or(EnergyVadConfig::default().min_speech_seconds),
        merge_gap_seconds: finite_f64(input, "minSilenceSeconds")?
            .unwrap_or(EnergyVadConfig::default().merge_gap_seconds),
    };
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

fn channels(input: &serde_json::Value) -> Result<u16, String> {
    let value = input
        .get("channels")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(1);
    u16::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| "channels must be a positive u16".to_string())
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

fn finite_f32(input: &serde_json::Value, field: &str) -> Result<Option<f32>, String> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    let value = value
        .as_f64()
        .ok_or_else(|| format!("{field} must be a number"))? as f32;
    if value.is_finite() {
        Ok(Some(value))
    } else {
        Err(format!("{field} must be finite"))
    }
}

fn finite_f64(input: &serde_json::Value, field: &str) -> Result<Option<f64>, String> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    let value = value
        .as_f64()
        .ok_or_else(|| format!("{field} must be a number"))?;
    if value.is_finite() {
        Ok(Some(value))
    } else {
        Err(format!("{field} must be finite"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_speaker_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.speakers.embed"));
        assert!(ids.contains(&"audio.speakers.assignTranscript"));
        assert!(ids.contains(&"audio.speakers.vad"));
        assert!(ids.contains(&"audio.speakers.diarize"));
    }

    #[test]
    fn embed_operation_returns_dimensions() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.speakers.embed"),
            input: serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 4, "fftSize": 4, "hopSize": 2, "bands": 2}),
        })
        .expect("embed");
        assert_eq!(response.value["operation"], "audio.speakers.embed");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert!(response.value["model"]["dimensions"].as_u64().unwrap() > 0);
    }

    #[test]
    fn vad_and_diarize_operations_return_segments() {
        let vad = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.speakers.vad"),
            input: serde_json::json!({
                "samples": [0.0, 0.2, 0.2, 0.0],
                "sampleRate": 4,
                "channels": 1,
                "frameSize": 2,
                "hopSize": 1,
                "threshold": 0.01,
                "minSpeechSeconds": 0.0,
                "minSilenceSeconds": 0.0
            }),
        })
        .expect("vad");
        assert_eq!(vad.value["operation"], "audio.speakers.vad");
        assert!(vad.value["summary"]["segmentCount"].as_u64().unwrap() > 0);

        let diarize = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.speakers.diarize"),
            input: serde_json::json!({
                "samples": [0.0, 0.2, 0.2, 0.0, -0.2, -0.2, 0.0],
                "sampleRate": 4,
                "channels": 1,
                "frameSize": 2,
                "hopSize": 1,
                "threshold": 0.01,
                "minSpeechSeconds": 0.0,
                "minSilenceSeconds": 0.0,
                "fftSize": 4,
                "hopSize": 2,
                "bands": 2
            }),
        })
        .expect("diarize");
        assert_eq!(diarize.value["operation"], "audio.speakers.diarize");
        assert!(!diarize.value["segments"].as_array().unwrap().is_empty());
    }

    #[test]
    fn assign_transcript_accepts_overlap_policy() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.speakers.assignTranscript"),
            input: serde_json::json!({
                "overlapPolicy": "strictContained",
                "transcript": {
                    "segments": [
                        {"index": 0, "text": "hello", "startSeconds": 0.0, "endSeconds": 2.0, "isFinal": true}
                    ]
                },
                "diarization": {
                    "accepted": true,
                    "operation": "diarize",
                    "modelId": "fixture",
                    "runtime": "imported",
                    "segments": [
                        {"speaker": "partial", "startSeconds": 0.5, "endSeconds": 1.5, "score": 1.0}
                    ]
                }
            }),
        })
        .expect("assign transcript");
        assert_eq!(
            response.value["operation"],
            "audio.speakers.assignTranscript"
        );
        assert_eq!(response.value["overlapPolicy"], "strictContained");
        assert_eq!(response.value["segments"][0]["speaker"], "unknown");
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
            operation: OperationId::new("audio.speakers.embed"),
            input: serde_json::json!({"samples": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("samples"));
    }
}
