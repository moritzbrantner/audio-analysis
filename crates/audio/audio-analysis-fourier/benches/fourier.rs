use audio_analysis_core::WindowFunction;
use audio_analysis_fourier::{spectrogram, FourierTransform, StftConfig};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn sine_len(freq_hz: f32, sample_rate: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

fn bench_fourier(c: &mut Criterion) {
    for fft_size in [1024, 2048, 4096] {
        let samples = sine_len(440.0, 48_000, fft_size);
        let transform = FourierTransform::with_window(fft_size, WindowFunction::Hann).unwrap();
        c.bench_function(&format!("fft_{fft_size}"), |b| {
            b.iter(|| {
                transform
                    .analyze_samples(black_box(&samples), 48_000)
                    .unwrap()
            })
        });
    }

    let ten_seconds = sine_len(440.0, 48_000, 480_000);
    let config = StftConfig::new(2048, 512).unwrap();
    c.bench_function("stft_10s", |b| {
        b.iter(|| spectrogram(black_box(&ten_seconds), 48_000, &config).unwrap())
    });
}

criterion_group!(benches, bench_fourier);
criterion_main!(benches);
