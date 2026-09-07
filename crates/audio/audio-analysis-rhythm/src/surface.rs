//! Library-owned runtime surface for `audio-analysis-rhythm`.

use audio_analysis_core::FrameSpec;
use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::track::{
    analyze_rhythm_track, StructuralDescriptor, TrackRhythmAnalysis, TrackRhythmConfig,
    TrackedBeat,
};
use crate::{
    beat_grid, detect_onsets, estimate_tempo, onset_envelope, Onset, OnsetDetectorConfig,
    OnsetStrength, TempoEstimatorConfig,
};

const MAX_SAMPLES: usize = 192_000;
const MAX_TRACK_SECONDS: usize = 15 * 60;
const MIN_SECTION_SECONDS: f64 = 8.0;
const SECTION_CHANGE_THRESHOLD: f32 = 0.20;
const SECTION_IDENTITY_THRESHOLD: f32 = 0.18;

#[derive(Debug, Clone, PartialEq)]
struct StructuralSection {
    start_seconds: f64,
    end_seconds: f64,
    start_boundary_confidence: f32,
    identity: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BeatContext {
    bar_index: usize,
    section_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct DescriptorMean {
    onset: f32,
    low: f32,
    mid: f32,
    high: f32,
    chroma: [f32; 12],
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
                "Onset detection, tempo estimation, and whole-track rhythm analysis.",
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
                "Uses spectral flux, beat-path-rescored tempo candidates, elastic dynamic-programming beat tracking, bar-phase accents, and multi-descriptor musical change points.",
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
        input_schema: serde_json::json!({
            "type": "object",
            "additionalProperties": true,
            "xOperationCategory": runtime_core::operation_category(id)
        }),
        output_schema: serde_json::json!({
            "type": "object",
            "xOperationCategory": runtime_core::operation_category(id)
        }),
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
            "Estimated whole-track tempo candidates, an elastic beat path, 4/4 downbeats, and multi-descriptor structural sections.",
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
    let sections = structural_sections(&analysis, analysis_duration_seconds);
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
            "score": candidate.score,
            "autocorrelationScore": candidate.autocorrelation_score,
            "beatSupport": candidate.beat_support
        })).collect::<Vec<_>>(),
        "tempoMap": analysis.tempo_map.iter().map(|point| serde_json::json!({
            "timestampSeconds": time_offset_seconds + point.timestamp_seconds,
            "timestampMs": timestamp_millis(time_offset_seconds + point.timestamp_seconds),
            "bpm": point.bpm,
            "confidence": point.confidence
        })).collect::<Vec<_>>(),
        "beats": analysis.beats.iter().enumerate().map(|(index, beat)| {
            let context = beat_contexts[index];
            let timestamp_seconds = time_offset_seconds + beat.timestamp_seconds;
            let timestamp_ms = timestamp_millis(timestamp_seconds);
            let section_identity = context.section_index.checked_sub(1)
                .and_then(|section_index| sections.get(section_index))
                .map(|section| section.identity.as_str());
            serde_json::json!({
                "index": index + 1,
                "timestampSeconds": timestamp_seconds,
                "timestampMs": timestamp_ms,
                "timestamp": format_timestamp_millis(timestamp_ms),
                "strength": beat.strength,
                "localBpm": beat.local_bpm,
                "beatInBar": beat.beat_in_bar,
                "barIndex": context.bar_index,
                "sectionIndex": context.section_index,
                "sectionLabel": format!("section-{}", context.section_index),
                "sectionIdentity": section_identity,
                "downbeat": beat.downbeat
            })
        }).collect::<Vec<_>>(),
        "downbeats": analysis.downbeats.iter().map(|timestamp| time_offset_seconds + timestamp).collect::<Vec<_>>(),
        "downbeatEvents": analysis.beats.iter().enumerate().filter(|(_, beat)| beat.downbeat).enumerate().map(|(downbeat_index, (beat_index, beat))| {
            let context = beat_contexts[beat_index];
            let timestamp_seconds = time_offset_seconds + beat.timestamp_seconds;
            let timestamp_ms = timestamp_millis(timestamp_seconds);
            let section_identity = context.section_index.checked_sub(1)
                .and_then(|section_index| sections.get(section_index))
                .map(|section| section.identity.as_str());
            serde_json::json!({
                "index": downbeat_index + 1,
                "beatIndex": beat_index + 1,
                "timestampSeconds": timestamp_seconds,
                "timestampMs": timestamp_ms,
                "timestamp": format_timestamp_millis(timestamp_ms),
                "barIndex": context.bar_index,
                "sectionIndex": context.section_index,
                "sectionLabel": format!("section-{}", context.section_index),
                "sectionIdentity": section_identity
            })
        }).collect::<Vec<_>>(),
        "downbeatConfidence": analysis.downbeat_confidence,
        "sectionsMethod": "musical-change-points-v2",
        "sections": sections.iter().enumerate().map(|(index, section)| {
            let start_seconds = time_offset_seconds + section.start_seconds;
            let end_seconds = time_offset_seconds + section.end_seconds;
            let start_ms = timestamp_millis(start_seconds);
            let end_ms = timestamp_millis(end_seconds);
            serde_json::json!({
                "index": index + 1,
                "label": format!("section-{}", index + 1),
                "identity": section.identity,
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

fn structural_sections(
    analysis: &TrackRhythmAnalysis,
    duration_seconds: f64,
) -> Vec<StructuralSection> {
    if analysis.beats.is_empty() || !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Vec::new();
    }

    let downbeat_indices = analysis
        .beats
        .iter()
        .enumerate()
        .filter_map(|(index, beat)| beat.downbeat.then_some(index))
        .collect::<Vec<_>>();
    if downbeat_indices.len() < 5 {
        return vec![StructuralSection {
            start_seconds: 0.0,
            end_seconds: duration_seconds,
            start_boundary_confidence: 1.0,
            identity: "A".to_string(),
        }];
    }

    let mut selected = Vec::<(f64, f32)>::new();
    for position in 2..downbeat_indices.len().saturating_sub(2) {
        let boundary_index = downbeat_indices[position];
        let left_start_index = downbeat_indices[position - 2];
        let right_end_index = downbeat_indices[position + 2];
        let left_start = analysis.beats[left_start_index].timestamp_seconds;
        let boundary = analysis.beats[boundary_index].timestamp_seconds;
        let right_end = analysis.beats[right_end_index].timestamp_seconds;
        let Some(left) = mean_descriptor(&analysis.structural_descriptors, left_start, boundary) else {
            continue;
        };
        let Some(right) = mean_descriptor(&analysis.structural_descriptors, boundary, right_end) else {
            continue;
        };

        let direct_change = descriptor_distance(&left, &right);
        let repetition_change = mean_descriptor(
            &analysis.structural_descriptors,
            0.0,
            left_start.max(0.0),
        )
        .map(|history| {
            let left_similarity = 1.0 - descriptor_distance(&left, &history);
            let right_similarity = 1.0 - descriptor_distance(&right, &history);
            (left_similarity - right_similarity).abs()
        })
        .unwrap_or(0.0);
        let confidence = (0.82 * direct_change + 0.18 * repetition_change).clamp(0.0, 1.0);
        if confidence < SECTION_CHANGE_THRESHOLD
            || boundary < MIN_SECTION_SECONDS
            || duration_seconds - boundary < MIN_SECTION_SECONDS
        {
            continue;
        }

        if let Some(last) = selected.last_mut() {
            if boundary - last.0 < MIN_SECTION_SECONDS {
                if confidence > last.1 {
                    *last = (boundary, confidence);
                }
                continue;
            }
        }
        selected.push((boundary, confidence));
    }

    let mut boundaries = Vec::with_capacity(selected.len() + 2);
    boundaries.push((0.0, 1.0));
    boundaries.extend(selected);
    boundaries.push((duration_seconds, 1.0));
    let mut sections = boundaries
        .windows(2)
        .map(|pair| StructuralSection {
            start_seconds: pair[0].0,
            end_seconds: pair[1].0,
            start_boundary_confidence: pair[0].1,
            identity: String::new(),
        })
        .collect::<Vec<_>>();
    assign_section_identities(&mut sections, &analysis.structural_descriptors);
    sections
}

fn mean_descriptor(
    descriptors: &[StructuralDescriptor],
    start_seconds: f64,
    end_seconds: f64,
) -> Option<DescriptorMean> {
    if end_seconds <= start_seconds {
        return None;
    }
    let mut count = 0_usize;
    let mut onset = 0.0_f32;
    let mut low = 0.0_f32;
    let mut mid = 0.0_f32;
    let mut high = 0.0_f32;
    let mut chroma = [0.0_f32; 12];
    for descriptor in descriptors.iter().filter(|descriptor| {
        descriptor.timestamp_seconds >= start_seconds && descriptor.timestamp_seconds < end_seconds
    }) {
        count += 1;
        onset += descriptor.onset_novelty;
        low += descriptor.low_energy;
        mid += descriptor.mid_energy;
        high += descriptor.high_energy;
        for (target, value) in chroma.iter_mut().zip(descriptor.chroma) {
            *target += value;
        }
    }
    if count == 0 {
        return None;
    }
    let denominator = count as f32;
    for value in &mut chroma {
        *value /= denominator;
    }
    Some(DescriptorMean {
        onset: onset / denominator,
        low: low / denominator,
        mid: mid / denominator,
        high: high / denominator,
        chroma,
    })
}

fn descriptor_distance(left: &DescriptorMean, right: &DescriptorMean) -> f32 {
    let spectral = ((left.low - right.low).abs()
        + (left.mid - right.mid).abs()
        + (left.high - right.high).abs())
        / 3.0;
    let onset = (left.onset - right.onset).abs();
    let chroma = chroma_distance(&left.chroma, &right.chroma);
    (0.35 * spectral + 0.25 * onset + 0.40 * chroma).clamp(0.0, 1.0)
}

fn chroma_distance(left: &[f32; 12], right: &[f32; 12]) -> f32 {
    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        return 0.0;
    }
    (1.0 - dot / (left_norm * right_norm)).clamp(0.0, 1.0)
}

fn assign_section_identities(
    sections: &mut [StructuralSection],
    descriptors: &[StructuralDescriptor],
) {
    let mut prototypes: Vec<DescriptorMean> = Vec::new();
    for section in sections {
        let Some(descriptor) = mean_descriptor(descriptors, section.start_seconds, section.end_seconds)
        else {
            section.identity = section_identity_label(prototypes.len());
            continue;
        };
        let match_index = prototypes
            .iter()
            .enumerate()
            .map(|(index, prototype)| (index, descriptor_distance(&descriptor, prototype)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .filter(|(_, distance)| *distance <= SECTION_IDENTITY_THRESHOLD)
            .map(|(index, _)| index);
        let identity_index = match match_index {
            Some(index) => index,
            None => {
                prototypes.push(descriptor);
                prototypes.len() - 1
            }
        };
        section.identity = section_identity_label(identity_index);
    }
}

fn section_identity_label(index: usize) -> String {
    if index < 26 {
        ((b'A' + index as u8) as char).to_string()
    } else {
        format!("S{}", index + 1)
    }
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
                .unwrap_or_else(|| if sections.is_empty() { 0 } else { sections.len() });
            BeatContext {
                bar_index,
                section_index,
            }
        })
        .collect()
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
) -> Result<(u32, FrameSpec, Vec<OnsetStrength>, Vec<Onset>), String> {
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

fn finite_f64(
    input: &serde_json::Value,
    field: &str,
    default_value: f64,
) -> Result<f64, String> {
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
    fn section_identity_labels_are_stable() {
        assert_eq!(section_identity_label(0), "A");
        assert_eq!(section_identity_label(1), "B");
        assert_eq!(section_identity_label(25), "Z");
        assert_eq!(section_identity_label(26), "S27");
    }

    #[test]
    fn chroma_distance_separates_unrelated_pitch_classes() {
        let mut left = [0.0; 12];
        let mut right = [0.0; 12];
        left[0] = 1.0;
        right[6] = 1.0;
        assert!(chroma_distance(&left, &right) > 0.9);
        assert!(chroma_distance(&left, &left) < 0.01);
    }
}
