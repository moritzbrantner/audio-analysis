use audio_analysis_recognition::{
    AudioEmbeddingExtractor, AudioMatchOptions, AudioReferenceLibrary, SpectralAudioEmbedder,
    SpectralEmbeddingConfig,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn sine(freq_hz: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
    let samples = (sample_rate as f32 * seconds) as usize;
    (0..samples)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

fn bench_recognition(c: &mut Criterion) {
    let extractor =
        SpectralAudioEmbedder::new(SpectralEmbeddingConfig::new(512, 256, 8).unwrap()).unwrap();
    let query_samples = sine(440.0, 8_000, 0.5);
    c.bench_function("embedding_extraction", |b| {
        b.iter(|| {
            extractor
                .embed_samples(black_box(&query_samples), 8_000)
                .unwrap()
        })
    });

    for size in [10, 100, 1000] {
        let mut library = AudioReferenceLibrary::new();
        for index in 0..size {
            let frequency = 200.0 + index as f32 * 3.0;
            library
                .add_reference_samples(
                    format!("ref-{index}"),
                    format!("Reference {index}"),
                    &sine(frequency, 8_000, 0.2),
                    8_000,
                    &extractor,
                )
                .unwrap();
        }
        let query = extractor.embed_samples(&query_samples, 8_000).unwrap();
        c.bench_function(&format!("library_search_{size}"), |b| {
            b.iter(|| {
                library
                    .search(
                        black_box(&query),
                        &AudioMatchOptions {
                            min_score: -1.0,
                            max_results: 5,
                        },
                    )
                    .unwrap()
            })
        });
    }
}

criterion_group!(benches, bench_recognition);
criterion_main!(benches);
