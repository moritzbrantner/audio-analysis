//! Library-owned runtime surface for `audio-analysis-rhythm`.

use audio_analysis_core::FrameSpec;
use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::track::{analyze_rhythm_track, TrackRhythmConfig, TrackedBeat};
use crate::{
    beat_grid, detect_onsets, estimate_tempo, onset_envelope, OnsetDetectorConfig,
    TempoEstimatorConfig,
};

const MAX_SAMPLES: usize = 192_000;
const MAX_TRACK_SECONDS: usize = 15 * 60;
const MIN_SECTION_SECONDS: f64 = 8.0;
const SECTION_CHANGE_THRESHOLD: f32 = 0.18;

#[derive(Debug, Clone, Copy, PartialEq)]
struct StructuralSection {
    start_seconds: f64,
    end_seconds: f64,
    start_boundary_confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BeatContext {
    bar_index: usize,
    section_index: usize,
}

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
                "Onset detection, tempo estimation, and whole-track beat analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.rhythm.onsets",
                "Detect onsets",
                "Computes an onset envelope and deterministic onset list.",
                serde_json::json!({"samples": [1.0, 0.0, 0.0, 1.0], "sampleRate": 1000, "frameSize": 2, "hopSize": 1}),
            ),
            operation(
                "audio.rhythm.tempo",
                "Estimate tempo",
                "Estimates BPM from detected onset intervals.",
                serde_json::json!({"samples": [1.0, 0.0, 0.0, 1.0], "sampleRate": 1000, "frameSize": 2, "hopSize": 1}),
            ),
            operation(
                "audio.rhythm.beatGrid",
                "Beat grid",
                "Creates a beat grid from start time, BPM, and beat count.",
                serde_json::json!({"startSeconds": 0.0, "bpm": 120.0, "beats": 4}),
            ),
            operation(
                "audio.rhythm.analyze",
                "Analyze track rhythm",
                "Uses spectral flux, tempo autocorrelation, dynamic-programming beat tracking, bar-phase accents, and rhythmic change points to estimate BPM, beats, downbeats, and structural sections.",
                serde_json::json!({"samples": [1.0, 0.0, 0.0, 1.0], "sampleRate": 48000}),
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
        "audio.rhythm.onsets" => onsets_value(request.input)?,
        "audio.rhythm.tempo" => tempo_value(request.input)?,
        "audio.rhythm.beatGrid" => beat_grid_value(request.input)?,
        "audio.rhythm.analyze" => track_analysis_value(request.input)?,
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
            "Rhythm package metadata",
            "Inspected the onset, tempo, beat-grid, and whole-track rhythm operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.rhythm.onsets" => (
            "Onset detection result",
            "Computed an onset envelope and deterministic onset list from normalized samples.",
            serde_json::json!({
                "sampleRate": value.get("sampleRate").cloned().unwrap_or(serde_json::Value::Null),
                "envelopeFrameCount": value.get("envelopeFrameCount").cloned().unwrap_or(serde_json::Value::Null),
                "onsetCount": value.get("onsetCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.rhythm.tempo" => (
            "Tempo estimate result",
            "Estimated BPM from detected onset intervals.",
            serde_json::json!({
                "bpm": value.get("bpm").cloned().unwrap_or(serde_json::Value::Null),
                "onsetCount": value.get("onsetCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.rhythm.beatGrid" => (
            "Beat grid result",
            "Created a deterministic beat grid from start time, BPM, and beat count.",
            serde_json::json!({
                "bpm": value.get("bpm").cloned().unwrap_or(serde_json::Value::Null),
                "beatCount": value.get("grid").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.rhythm.analyze" => (
            "Track rhythm analysis",
            "Estimated whole-track tempo candidates, a globally consistent beat path, 4/4 downbeats, and rhythmic structural sections.",
            serde_json::json!({
                "bpm": value.get("bpm").cloned().unwrap_or(serde_json::Value::Null),
                "confidence": value.get("confidence").cloned().unwrap_or(serde_json::Value::Null),
                "beatCount": value.get("beats").and_then(serde_json::Value::as_array).map_or(0, Vec::len),
                "barCount": value.get("barCount").cloned().unwrap_or(serde_json::Value::Null),
                "downbeatCount": value.get("downbeats").and_then(serde_json::Value::as_array).map_or(0, Vec::len),
                "sectionCount": value.get("sections").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        _ => (
            "Rhythm operation result",
            "Completed the rhythm package surface operation.",
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

fn onsets_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let (sample_rate, frame_spec, envelope, onsets) = detected_onsets(&input)?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "frameSize": frame_spec.frame_size,
        "hopSize": frame_spec.hop_size,
        "envelopeFrameCount": envelope.len(),
        "onsetCount": onsets.len(),
        "onsets": onsets.iter().take(64).map(|onset| serde_json::json!({
            "timestampSeconds": onset.timestamp_seconds,
            "strength": onset.strength
        })).collect::<Vec<_>>()
    }))
}

fn tempo_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let (sample_rate, frame_spec, _envelope, onsets) = detected_onsets(&input)?;
    let tempo = estimate_tempo(&onsets, TempoEstimatorConfig::default())
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "sampleRate": sample_rate,
        "frameSize": frame_spec.frame_size,
        "hopSize": frame_spec.hop_size,
        "onsetCount": onsets.len(),
        "bpm": tempo.bpm,
        "confidence": tempo.confidence
    }))
}

fn beat_grid_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let start_seconds = finite_f64(&input, "startSeconds", 0.0)?;
    let bpm = finite_f64(&input, "bpm", 120.0)? as f32;
    let beats = positive_usize(&input, "beats", 4)?.min(1024);
    let grid = beat_grid(start_seconds, bpm, beats).map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "startSeconds": start_seconds,
        "bpm": bpm,
        "beats": beats,
        "grid": grid
    }))
}

fn track_analysis_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let sample_rate = sample_rate(&input)?;
    let samples = track_sample_array(&input, "samples", sample_rate)?;
    let time_offset_seconds = nonnegative_f64(&input, "timeOffsetSeconds", 0.0)?;
    let mut config = TrackRhythmConfig::default();
    config.min_bpm = finite_f64(&input, "minBpm", config.min_bpm as f64)? as f32;
    config.max_bpm = finite_f64(&input, "maxBpm", config.max_bpm as f64)? as f32;
    config.fft_size = positive_usize(&input, "fftSize", config.fft_size)?;
    config.hop_size = positive_usize(&input, "hopSize", config.hop_size)?;
    config.beats_per_bar = positive_usize(&input, "beatsPerBar", config.beats_per_bar)?;
    config.tempo_candidate_count =
        positive_usize(&input, "tempoCandidateCount", config.tempo_candidate_count)?.min(16);
    let analysis =
        analyze_rhythm_track(&samples, sample_rate, config).map_err(|error| error.to_string())?;
    let analysis_duration_seconds = samples.len() as f64 / sample_rate as f64;
    let analysis_end_seconds = time_offset_seconds + analysis_duration_seconds;
    let sections = structural_sections(&analysis.beats, analysis_duration_seconds);
    let beat_contexts = contextualize_beats(&analysis.beats, &sections);
    let bar_count = beat_contexts.last().map_or(0, |context| context.bar_index);

    Ok(serde_json::json!({
        "schemaVersion": "audio-analysis-song/v1",
        "sampleRate": sample_rate,
        "sampleCount": samples.len(),
        "analysisStartSeconds": time_offset_seconds,
        "analysisStartMs": timestamp_millis(time_offset_seconds),
        "analysisDurationSeconds": analysis_duration_seconds,
        "analysisEndSeconds": analysis_end_seconds,
        "analysisEndMs": timestamp_millis(analysis_end_seconds),
        "bpm": analysis.bpm,
        "confidence": analysis.confidence,
        "hopSeconds": analysis.hop_seconds,
        "beatsPerBar": config.beats_per_bar,
        "barCount": bar_count,
        "tempoCandidates": analysis.tempo_candidates.iter().map(|candidate| serde_json::json!({
            "bpm": candidate.bpm,
            "score": candidate.score
        })).collect::<Vec<_>>(),
        "beats": analysis.beats.iter().enumerate().map(|(index, beat)| {
            let context = beat_contexts[index];
            let timestamp_seconds = time_offset_seconds + beat.timestamp_seconds;
            let timestamp_ms = timestamp_millis(timestamp_seconds);
            serde_json::json!({
                "index": index + 1,
                "timestampSeconds": timestamp_seconds,
                "timestampMs": timestamp_ms,
                "timestamp": format_timestamp_millis(timestamp_ms),
                "strength": beat.strength,
                "beatInBar": beat.beat_in_bar,
                "barIndex": context.bar_index,
                "sectionIndex": context.section_index,
                "sectionLabel": format!("section-{}", context.section_index),
                "downbeat": beat.downbeat
            })
        }).collect::<Vec<_>>(),
        "downbeats": analysis.downbeats.iter().map(|timestamp| time_offset_seconds + timestamp).collect::<Vec<_>>(),
        "downbeatEvents": analysis.beats.iter().enumerate().filter(|(_, beat)| beat.downbeat).enumerate().map(|(downbeat_index, (beat_index, beat))| {
            let context = beat_contexts[beat_index];
            let timestamp_seconds = time_offset_seconds + beat.timestamp_seconds;
            let timestamp_ms = timestamp_millis(timestamp_seconds);
            serde_json::json!({
                "index": downbeat_index + 1,
                "beatIndex": beat_index + 1,
                "timestampSeconds": timestamp_seconds,
                "timestampMs": timestamp_ms,
                "timestamp": format_timestamp_millis(timestamp_ms),
                "barIndex": context.bar_index,
                "sectionIndex": context.section_index,
                "sectionLabel": format!("section-{}", context.section_index)
            })
        }).collect::<Vec<_>>(),
        "downbeatConfidence": analysis.downbeat_confidence,
        "sectionsMethod": "rhythmic-change-points-v1",
        "sections": sections.iter().enumerate().map(|(index, section)| {
            let start_seconds = time_offset_seconds + section.start_seconds;
            let end_seconds = time_offset_seconds + section.end_seconds;
            let start_ms = timestamp_millis(start_seconds);
            let end_ms = timestamp_millis(end_seconds);
            serde_json::json!({
                "index": index + 1,
                "label": format!("section-{}", index + 1),
                "startSeconds": start_seconds,
                "startMs": start_ms,
                "start": format_timestamp_millis(start_ms),
                "endSeconds": end_seconds,
                "endMs": end_ms,
                "end": format_timestamp_millis(end_ms),
                "durationSeconds": section.end_seconds - section.start_seconds,
                "startBoundaryConfidence": section.start_boundary_confidence
            })
        }).collect::<Vec<_>>()
    }))
}

fn structural_sections(beats: &[TrackedBeat], duration_seconds: f64) -> Vec<StructuralSection> {
    if beats.is_empty() || !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Vec::new();
    }

    let downbeat_indices = beats
        .iter()
        .enumerate()
        .filter_map(|(index, beat)| beat.downbeat.then_some(index))
        .collect::<Vec<_>>();
    if downbeat_indices.len() < 5 {
        return vec![StructuralSection {
            start_seconds: 0.0,
            end_seconds: duration_seconds,
            start_boundary_confidence: 1.0,
        }];
    }

    let mut selected = Vec::<(f64, f32)>::new();
    for position in 2..downbeat_indices.len().saturating_sub(2) {
        let boundary = downbeat_indices[position];
        let left_start = downbeat_indices[position - 2];
        let right_end = downbeat_indices[position + 2];
        let left_strength = mean_beat_strength(beats, left_start, boundary);
        let right_strength = mean_beat_strength(beats, boundary, right_end);
        let confidence = (right_strength - left_strength).abs().clamp(0.0, 1.0);
        let timestamp = beats[boundary].timestamp_seconds;
        if confidence < SECTION_CHANGE_THRESHOLD
            || timestamp < MIN_SECTION_SECONDS
            || duration_seconds - timestamp < MIN_SECTION_SECONDS
        {
            continue;
        }

        if let Some(last) = selected.last_mut() {
            if timestamp - last.0 < MIN_SECTION_SECONDS {
                if confidence > last.1 {
                    *last = (timestamp, confidence);
                }
                continue;
            }
        }
        selected.push((timestamp, confidence));
    }

    let mut boundaries = Vec::with_capacity(selected.len() + 2);
    boundaries.push((0.0, 1.0));
    boundaries.extend(selected);
    boundaries.push((duration_seconds, 1.0));
    boundaries
        .windows(2)
        .map(|pair| StructuralSection {
            start_seconds: pair[0].0,
            end_seconds: pair[1].0,
            start_boundary_confidence: pair[0].1,
        })
        .collect()
}

fn contextualize_beats(beats: &[TrackedBeat], sections: &[StructuralSection]) -> Vec<BeatContext> {
    let mut bar_index = 1;
    beats
        .iter()
        .enumerate()
        .map(|(index, beat)| {
            if index > 0 && beat.downbeat {
                bar_index += 1;
            }
            let section_index = sections
                .iter()
                .position(|section| {
                    beat.timestamp_seconds >= section.start_seconds
                        && beat.timestamp_seconds < section.end_seconds
                })
                .map(|index| index + 1)
                .unwrap_or_else(|| {
                    if sections.is_empty() {
                        0
                    } else {
                        sections.len()
                    }
                });
            BeatContext {
                bar_index,
                section_index,
            }
        })
        .collect()
}

fn mean_beat_strength(beats: &[TrackedBeat], start: usize, end: usize) -> f32 {
    if start >= end || start >= beats.len() {
        return 0.0;
    }
    let end = end.min(beats.len());
    beats[start..end]
        .iter()
        .map(|beat| beat.strength)
        .sum::<f32>()
        / (end - start) as f32
}

fn timestamp_millis(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * 1000.0).round() as u64
}

fn format_timestamp_millis(total_millis: u64) -> String {
    let hours = total_millis / 3_600_000;
    let minutes = (total_millis / 60_000) % 60;
    let seconds = (total_millis / 1_000) % 60;
    let millis = total_millis % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

fn detected_onsets(
    input: &serde_json::Value,
) -> Result<(u32, FrameSpec, Vec<crate::OnsetStrength>, Vec<crate::Onset>), String> {
    let samples = sample_array(input, "samples")?;
    let sample_rate = sample_rate(input)?;
    let frame_size = positive_usize(input, "frameSize", 1024)?;
    let hop_size = positive_usize(input, "hopSize", frame_size / 2)?;
    let frame_spec = FrameSpec::new(frame_size, hop_size).map_err(|error| error.to_string())?;
    let envelope =
        onset_envelope(&samples, sample_rate, frame_spec).map_err(|error| error.to_string())?;
    let config = OnsetDetectorConfig {
        strength_threshold: finite_f64(input, "strengthThreshold", 0.05)? as f32,
        min_interval_seconds: finite_f64(input, "minIntervalSeconds", 0.05)?,
    };
    let onsets = detect_onsets(&envelope, config).map_err(|error| error.to_string())?;
    Ok((sample_rate, frame_spec, envelope, onsets))
}

fn sample_array(input: &serde_json::Value, field: &str) -> Result<Vec<f32>, String> {
    sample_array_with_max(input, field, MAX_SAMPLES)
}

fn track_sample_array(
    input: &serde_json::Value,
    field: &str,
    sample_rate: u32,
) -> Result<Vec<f32>, String> {
    let max_samples = (sample_rate as usize).saturating_mul(MAX_TRACK_SECONDS);
    sample_array_with_max(input, field, max_samples)
}

fn sample_array_with_max(
    input: &serde_json::Value,
    field: &str,
    max_samples: usize,
) -> Result<Vec<f32>, String> {
    let values = input
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?;
    if values.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    if values.len() > max_samples {
        return Err(format!(
            "{field} must not contain more than {max_samples} samples"
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

fn finite_f64(input: &serde_json::Value, field: &str, default_value: f64) -> Result<f64, String> {
    let value = input
        .get(field)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default_value);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{field} must be finite"))
    }
}

fn nonnegative_f64(
    input: &serde_json::Value,
    field: &str,
    default_value: f64,
) -> Result<f64, String> {
    let value = finite_f64(input, field, default_value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(format!("{field} must be non-negative"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_rhythm_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.rhythm.onsets"));
        assert!(ids.contains(&"audio.rhythm.beatGrid"));
        assert!(ids.contains(&"audio.rhythm.analyze"));
    }

    #[test]
    fn beat_grid_operation_returns_grid() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.rhythm.beatGrid"),
            input: serde_json::json!({"startSeconds": 0.0, "bpm": 120.0, "beats": 4}),
        })
        .expect("beat grid");
        assert_eq!(response.value["operation"], "audio.rhythm.beatGrid");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert_eq!(response.value["grid"].as_array().unwrap().len(), 4);
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
    fn track_analysis_accepts_more_than_the_preview_sample_limit() {
        let samples = vec![0.0; MAX_SAMPLES + 1];
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.rhythm.analyze"),
            input: serde_json::json!({
                "samples": samples,
                "sampleRate": 8_000,
                "fftSize": 512,
                "hopSize": 512
            }),
        })
        .expect("whole-track rhythm analysis");

        assert_eq!(response.value["sampleCount"], MAX_SAMPLES + 1);
        assert_eq!(response.value["schemaVersion"], "audio-analysis-song/v1");
        assert_eq!(response.value["beatsPerBar"], 4);
        assert_eq!(response.value["barCount"], 0);
    }

    #[test]
    fn song_timestamps_include_milliseconds() {
        assert_eq!(timestamp_millis(195.022), 195_022);
        assert_eq!(format_timestamp_millis(195_022), "00:03:15.022");
    }

    #[test]
    fn structural_sections_detect_rhythmic_intensity_change() {
        let beats = (0..64)
            .map(|index| TrackedBeat {
                timestamp_seconds: index as f64 * 0.5,
                strength: if index < 32 { 0.2 } else { 0.9 },
                beat_in_bar: index % 4 + 1,
                downbeat: index % 4 == 0,
            })
            .collect::<Vec<_>>();

        let sections = structural_sections(&beats, 32.0);
        assert!(sections.len() >= 2);
        assert!(sections
            .iter()
            .any(|section| (section.start_seconds - 16.0).abs() <= 2.0));
    }

    #[test]
    fn beat_contexts_assign_tracked_bars_and_sections() {
        let beats = (0..12)
            .map(|index| TrackedBeat {
                timestamp_seconds: index as f64,
                strength: 0.5,
                beat_in_bar: index % 4 + 1,
                downbeat: index % 4 == 0,
            })
            .collect::<Vec<_>>();
        let sections = vec![
            StructuralSection {
                start_seconds: 0.0,
                end_seconds: 6.0,
                start_boundary_confidence: 1.0,
            },
            StructuralSection {
                start_seconds: 6.0,
                end_seconds: 12.0,
                start_boundary_confidence: 0.8,
            },
        ];

        let contexts = contextualize_beats(&beats, &sections);
        assert_eq!(contexts[0].bar_index, 1);
        assert_eq!(contexts[3].bar_index, 1);
        assert_eq!(contexts[4].bar_index, 2);
        assert_eq!(contexts[8].bar_index, 3);
        assert_eq!(contexts[5].section_index, 1);
        assert_eq!(contexts[6].section_index, 2);
        assert_eq!(contexts[11].section_index, 2);
    }

    #[test]
    fn onset_preview_keeps_the_existing_sample_limit() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.rhythm.onsets"),
            input: serde_json::json!({"samples": vec![0.0; MAX_SAMPLES + 1]}),
        })
        .expect_err("preview sample limit");

        assert!(error.contains("192000"));
    }

    #[test]
    fn invalid_samples_return_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.rhythm.onsets"),
            input: serde_json::json!({"samples": "bad"}),
        })
        .unwrap_err();
        assert!(error.contains("samples"));
    }
}
