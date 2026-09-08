use std::hint::black_box;

use audio_analysis_core::FrameSpec;
use audio_analysis_rhythm::{
    detect_onsets, estimate_tempo, onset_envelope, OnsetDetectorConfig, TempoEstimatorConfig,
};

const SAMPLE_RATE: u32 = 2_000;
const BPM: f32 = 120.0;
const SECONDS: f32 = 60.0;
const REPEATS: usize = 64;

fn main() {
    let samples = click_track(SAMPLE_RATE, BPM, SECONDS);
    let frame_spec = FrameSpec::new(80, 20).expect("profile frame spec");
    let onset_config = OnsetDetectorConfig {
        strength_threshold: 0.05,
        min_interval_seconds: 0.1,
    };
    let mut checksum = 0_usize;

    for _ in 0..REPEATS {
        let envelope = onset_envelope(black_box(&samples), SAMPLE_RATE, frame_spec)
            .expect("profile onset envelope");
        let onsets = detect_onsets(black_box(&envelope), onset_config).expect("profile onsets");
        let tempo = estimate_tempo(black_box(&onsets), TempoEstimatorConfig::default())
            .expect("profile tempo");

        assert!(!envelope.is_empty(), "profile workload must emit envelope frames");
        assert!(!onsets.is_empty(), "profile workload must detect deterministic clicks");
        assert!(tempo.bpm.is_some(), "profile workload must estimate a tempo");
        checksum ^= envelope.len() ^ onsets.len() ^ tempo.onset_count;
        black_box((&envelope, &onsets, tempo));
    }

    black_box(checksum);
}

fn click_track(sample_rate: u32, bpm: f32, seconds: f32) -> Vec<f32> {
    let len = (sample_rate as f32 * seconds) as usize;
    let interval = (sample_rate as f32 * 60.0 / bpm).max(1.0) as usize;
    let mut samples = vec![0.0; len];
    for start in (0..len).step_by(interval) {
        for sample in samples.iter_mut().skip(start).take(8) {
            *sample = 1.0;
        }
    }
    samples
}
