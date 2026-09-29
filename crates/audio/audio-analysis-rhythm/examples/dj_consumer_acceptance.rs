//! Authored consumer scenarios, not claims of observed downstream failures.
use audio_analysis_pitch::key::{estimate_musical_key, HarmonicKeyConfig};
use audio_analysis_processing::operations::playback::plan_beat_loop;
use audio_analysis_rhythm::track::{analyze_rhythm_track, TrackRhythmAnalysis, TrackRhythmConfig};
use serde_json::{json, Value};

const SAMPLE_RATE: u32 = 8_000;
const BEAT_COUNT: usize = 48;
const PHASE_SECONDS: f64 = 0.5;
const TIME_TOLERANCE: f64 = 0.07;

fn provenance() -> Value {
    serde_json::from_str(include_str!(
        "../tests/fixtures/consumer-scenarios.v1.json"
    ))
    .expect("consumer provenance manifest")
}

// Beat times and chord notes are authored independently of the estimators.
fn authored_track(bpm: f64, semitone_shift: Option<f64>) -> Vec<f32> {
    let period = 60.0 / bpm;
    let duration = PHASE_SECONDS + BEAT_COUNT as f64 * period;
    let mut samples = vec![0.0; (duration * SAMPLE_RATE as f64).ceil() as usize];
    for beat in 0..BEAT_COUNT {
        let start = ((PHASE_SECONDS + beat as f64 * period) * SAMPLE_RATE as f64).round() as usize;
        for offset in 0..80 {
            samples[start + offset] +=
                if beat % 4 == 0 { 1.0 } else { 0.72 } * (1.0 - offset as f32 / 80.0);
        }
    }
    if let Some(shift) = semitone_shift {
        // I-IV-V-I in C major; shifting seven semitones gives G major.
        let chords = [
            [261.6256, 329.6276, 391.9954],
            [174.6141, 220.0, 261.6256],
            [195.9977, 246.9417, 293.6648],
            [261.6256, 329.6276, 391.9954],
        ];
        for (index, sample) in samples.iter_mut().enumerate() {
            let time = index as f64 / SAMPLE_RATE as f64;
            if time < PHASE_SECONDS {
                continue;
            }
            let beat_position = (time - PHASE_SECONDS) / period;
            let chord = chords[(beat_position / 4.0) as usize % chords.len()];
            let within_chord = beat_position % 4.0;
            let envelope = (within_chord / 0.1).min(1.0) * ((4.0 - within_chord) / 0.1).min(1.0);
            *sample += (chord
                .iter()
                .map(|frequency| {
                    (std::f64::consts::TAU * frequency * 2.0_f64.powf(shift / 12.0) * time).sin()
                })
                .sum::<f64>()
                * 0.12
                * envelope) as f32;
        }
    }
    for sample in &mut samples {
        *sample /= 1.4;
    }
    samples
}

fn rhythm(samples: &[f32], expected_bpm: f64) -> TrackRhythmAnalysis {
    let analysis = analyze_rhythm_track(
        samples,
        SAMPLE_RATE,
        TrackRhythmConfig {
            fft_size: 512,
            hop_size: 128,
            ..TrackRhythmConfig::default()
        },
    )
    .expect("public rhythm analysis");
    let bpm = analysis.bpm.expect("authored pulse tempo");
    assert!(
        (bpm as f64 - expected_bpm).abs() / expected_bpm < 0.035,
        "bpm={bpm}"
    );
    assert!(analysis.confidence.is_finite() && (0.0..=1.0).contains(&analysis.confidence));
    assert!(analysis.confidence > 0.0);
    analysis
}

fn dj_party_loops() -> Value {
    let samples = authored_track(120.0, None);
    let analysis = rhythm(&samples, 120.0);
    let beats = analysis
        .beats
        .iter()
        .map(|beat| beat.timestamp_seconds)
        .collect::<Vec<_>>();
    let duration = samples.len() as f64 / SAMPLE_RATE as f64;
    let loops = [1, 2, 4, 8].map(|count| {
        let plan = plan_beat_loop(&beats, 1.82, count, duration).expect("analyzed loop");
        assert!((plan.start_seconds - 2.0).abs() < TIME_TOLERANCE);
        assert!((plan.end_seconds - (2.0 + count as f64 * 0.5)).abs() < TIME_TOLERANCE);
        json!({"beatCount": count, "startSeconds": plan.start_seconds, "endSeconds": plan.end_seconds})
    });
    assert!(plan_beat_loop(&beats, beats.last().unwrap() + 0.1, 1, duration).is_none());
    json!({"id": "dj-party-analyzed-quantized-loops", "bpm": analysis.bpm, "loops": loops})
}

fn media_player_transition() -> Value {
    let tracks = [(120.0, 0.0, "C major"), (128.0, 7.0, "G major")].map(|(bpm, shift, label)| {
        let samples = authored_track(bpm, Some(shift));
        let analysis = rhythm(&samples, bpm);
        let key = estimate_musical_key(
            &samples,
            SAMPLE_RATE,
            HarmonicKeyConfig {
                fft_size: 1024,
                hop_size: 256,
                max_frequency_hz: 3_000.0,
                ..HarmonicKeyConfig::default()
            },
        ).expect("public key analysis").expect("authored tonal progression");
        assert_eq!(key.label(), label);
        assert!(key.confidence.is_finite() && (0.0..=1.0).contains(&key.confidence));
        assert!(key.confidence > 0.0);
        // An authored four-beat interval, not a consumer overlap/queue decision.
        let start = PHASE_SECONDS + 8.0 * 60.0 / bpm;
        let end = PHASE_SECONDS + 12.0 * 60.0 / bpm;
        let anchors = [start, end].map(|expected| {
            let beat = analysis.beats.iter().min_by(|left, right| {
                (left.timestamp_seconds - expected).abs().total_cmp(&(right.timestamp_seconds - expected).abs())
            }).expect("transition beat anchor");
            assert!((beat.timestamp_seconds - expected).abs() < TIME_TOLERANCE);
            beat.timestamp_seconds
        });
        assert!(anchors[0] < anchors[1] && anchors[1] <= samples.len() as f64 / SAMPLE_RATE as f64);
        json!({"bpm": analysis.bpm, "rhythmConfidence": analysis.confidence, "key": key.label(), "keyConfidence": key.confidence, "transitionAnchorsSeconds": anchors})
    });
    let silence = vec![0.0; SAMPLE_RATE as usize * 4];
    let unavailable =
        analyze_rhythm_track(&silence, SAMPLE_RATE, TrackRhythmConfig::default()).unwrap();
    assert!(unavailable.bpm.is_none() && unavailable.beats.is_empty());
    assert!(
        estimate_musical_key(&silence, SAMPLE_RATE, HarmonicKeyConfig::default())
            .unwrap()
            .is_none()
    );
    json!({"id": "media-player-planned-two-track-transition", "tracks": tracks, "silenceHasNoEvidence": true})
}

fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "provenance": provenance(),
            "results": [dj_party_loops(), media_player_transition()]
        }))
        .unwrap()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dj_party_analyzed_quantized_loops() {
        assert_eq!(dj_party_loops()["id"], provenance()["cases"][0]["id"]);
    }

    #[test]
    fn media_player_planned_two_track_transition() {
        assert_eq!(
            media_player_transition()["id"],
            provenance()["cases"][1]["id"]
        );
    }
}

#[test]
fn media_player_g_major_with_clicks_retains_key() {
    let mut samples = authored_track(128.0, Some(7.0));
    samples.truncate(((PHASE_SECONDS + 16.0 * 60.0 / 128.0) * SAMPLE_RATE as f64) as usize);
    let key = estimate_musical_key(
        &samples,
        SAMPLE_RATE,
        HarmonicKeyConfig {
            fft_size: 1024,
            hop_size: 256,
            max_frequency_hz: 3000.0,
            ..HarmonicKeyConfig::default()
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(key.label(), "G major");
}
