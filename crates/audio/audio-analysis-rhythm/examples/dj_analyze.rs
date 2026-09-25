use std::{error::Error, path::PathBuf};

use audio_analysis_io::{decode_audio_to_clip, AudioInput, AudioInputOptions};
use audio_analysis_pitch::key::{
    analyze_key_track_with_boundaries, HarmonicKeyConfig, KeyTimelineConfig, MusicalKeyEstimate,
    MusicalScale,
};
use audio_analysis_rhythm::track::{analyze_rhythm_track, TrackRhythmConfig};
use serde_json::json;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: dj_analyze <audio-file>")?;
    let (_metadata, clip) = decode_audio_to_clip(
        AudioInput::File(path.clone()),
        AudioInputOptions::recorded(),
    )?;
    let samples = downmix_to_mono(&clip.samples, clip.channels);

    let rhythm = analyze_rhythm_track(&samples, clip.sample_rate, TrackRhythmConfig::default())?;
    let duration_seconds = samples.len() as f64 / clip.sample_rate as f64;
    let key_boundaries = key_bar_boundaries(&rhythm.downbeats, duration_seconds);
    let key_analysis = analyze_key_track_with_boundaries(
        &samples,
        clip.sample_rate,
        HarmonicKeyConfig::default(),
        KeyTimelineConfig::default(),
        &key_boundaries,
    )?;

    let key_value = key_analysis.dominant.as_ref().map(key_estimate_json);
    let key_timeline = key_analysis
        .timeline
        .iter()
        .map(|window| {
            json!({
                "startSeconds": window.start_seconds,
                "endSeconds": window.end_seconds,
                "centerSeconds": window.center_seconds,
                "key": window.key.as_ref().map(key_estimate_json)
            })
        })
        .collect::<Vec<_>>();
    let key_segments = key_analysis
        .segments
        .iter()
        .map(|segment| {
            json!({
                "startSeconds": segment.start_seconds,
                "endSeconds": segment.end_seconds,
                "key": key_estimate_json(&segment.key),
                "confidence": segment.confidence,
                "windowCount": segment.window_count
            })
        })
        .collect::<Vec<_>>();

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "path": path,
            "sampleRate": clip.sample_rate,
            "channels": clip.channels,
            "durationSeconds": duration_seconds,
            "rhythm": {
                "bpm": rhythm.bpm,
                "confidence": rhythm.confidence,
                "tempoCandidates": rhythm.tempo_candidates.iter().map(|candidate| json!({
                    "bpm": candidate.bpm,
                    "score": candidate.score,
                    "autocorrelationScore": candidate.autocorrelation_score,
                    "beatSupport": candidate.beat_support
                })).collect::<Vec<_>>(),
                "beatCount": rhythm.beats.len(),
                "beats": rhythm.beats.iter().map(|beat| beat.timestamp_seconds).collect::<Vec<_>>(),
                "downbeats": rhythm.downbeats,
                "downbeatConfidence": rhythm.downbeat_confidence
            },
            "key": key_value,
            "keyTimeline": key_timeline,
            "keySegments": key_segments,
            "keyBoundaryAligned": key_analysis.boundary_aligned,
            "keyTimelineComplete": key_analysis.timeline_complete
        }))?
    );
    Ok(())
}

fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels as usize;
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}


fn key_estimate_json(estimate: &MusicalKeyEstimate) -> serde_json::Value {
    json!({
        "label": estimate.label(),
        "tonic": estimate.tonic.as_str(),
        "scale": match estimate.scale {
            MusicalScale::Major => "major",
            MusicalScale::Minor => "minor",
        },
        "strength": estimate.strength,
        "confidence": estimate.confidence,
        "tuningCents": estimate.tuning_cents,
        "chroma": estimate.chroma.bins,
        "runnerUp": {
            "tonic": estimate.runner_up.tonic.as_str(),
            "scale": match estimate.runner_up.scale {
                MusicalScale::Major => "major",
                MusicalScale::Minor => "minor",
            },
            "correlation": estimate.runner_up.correlation
        }
    })
}

fn key_bar_boundaries(downbeats: &[f64], duration_seconds: f64) -> Vec<f64> {
    if downbeats.len() < 2 || !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Vec::new();
    }

    let mut boundaries = downbeats
        .iter()
        .copied()
        .filter(|time| time.is_finite() && *time >= 0.0 && *time <= duration_seconds)
        .collect::<Vec<_>>();
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup();
    if boundaries.len() < 2 {
        return Vec::new();
    }
    if boundaries.first().is_some_and(|time| *time > 0.0) {
        boundaries.insert(0, 0.0);
    }
    if boundaries
        .last()
        .is_some_and(|time| *time < duration_seconds)
    {
        boundaries.push(duration_seconds);
    }
    boundaries
}
