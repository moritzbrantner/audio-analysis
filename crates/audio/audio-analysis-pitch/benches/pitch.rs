use audio_analysis_pitch::AutocorrelationPitchDetector;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn sine_len(freq_hz: f32, sample_rate: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

fn bench_pitch(c: &mut Criterion) {
    let detector = AutocorrelationPitchDetector::default();
    for len in [2048, 4096, 8192] {
        let samples = sine_len(440.0, 48_000, len);
        c.bench_function(&format!("autocorrelation_pitch_{len}"), |b| {
            b.iter(|| {
                detector
                    .estimate_samples(black_box(&samples), 48_000)
                    .unwrap()
            })
        });
    }
}

criterion_group!(benches, bench_pitch);
criterion_main!(benches);
