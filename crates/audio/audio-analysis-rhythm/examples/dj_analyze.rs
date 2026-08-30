use std::{error::Error, path::PathBuf};

use audio_analysis_io::{decode_audio_to_clip, AudioInput, AudioInputOptions};
use audio_analysis_pitch::key::{estimate_musical_key, HarmonicKeyConfig, MusicalScale};
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
    let key = estimate_musical_key(&samples, clip.sample_rate, HarmonicKeyConfig::default())?;

    let key_value = key.map(|estimate| {
        let scale = match estimate.scale {
            MusicalScale::Major => "major",
            MusicalScale::Minor => "minor",
        };
        json!({
            "label": estimate.label(),
            "tonic": estimate.tonic.as_str(),
            "scale": scale,
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
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "path": path,
            "sampleRate": clip.sample_rate,
            "channels": clip.channels,
            "durationSeconds": samples.len() as f64 / clip.sample_rate as f64,
            "rhythm": {
                "bpm": rhythm.bpm,
                "confidence": rhythm.confidence,
                "tempoCandidates": rhythm.tempo_candidates.iter().map(|candidate| json!({
                    "bpm": candidate.bpm,
                    "score": candidate.score
                })).collect::<Vec<_>>(),
                "beatCount": rhythm.beats.len(),
                "beats": rhythm.beats.iter().map(|beat| beat.timestamp_seconds).collect::<Vec<_>>(),
                "downbeats": rhythm.downbeats,
                "downbeatConfidence": rhythm.downbeat_confidence
            },
            "key": key_value
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
