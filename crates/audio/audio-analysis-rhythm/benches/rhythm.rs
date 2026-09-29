use audio_analysis_core::FrameSpec;
use audio_analysis_rhythm::{
    detect_onsets, estimate_tempo, onset_envelope, OnsetDetectorConfig, TempoEstimatorConfig,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

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

fn bench_rhythm(c: &mut Criterion) {
    let samples = click_track(2_000, 120.0, 60.0);
    let frame_spec = FrameSpec::new(80, 20).unwrap();
    c.bench_function("onset_envelope_60s", |b| {
        b.iter(|| onset_envelope(black_box(&samples), 2_000, frame_spec).unwrap())
    });

    let envelope = onset_envelope(&samples, 2_000, frame_spec).unwrap();
    let onset_config = OnsetDetectorConfig {
        strength_threshold: 0.05,
        min_interval_seconds: 0.1,
    };
    c.bench_function("tempo_estimation_60s", |b| {
        b.iter(|| {
            let onsets = detect_onsets(black_box(&envelope), onset_config).unwrap();
            estimate_tempo(&onsets, TempoEstimatorConfig::default()).unwrap()
        })
    });
}

criterion_group!(benches, bench_rhythm);
criterion_main!(benches);
