use audio_analysis_core::ChannelMix;
use audio_analysis_processing::{AudioProcessor, BiquadKind, BiquadSpec, NoiseGateSpec};
use audio_contracts::{AudioBuffer, OwnedAudioFrame, Timebase, Timestamp};
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

fn owned_f32_frame(
    timestamp: Timestamp,
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
) -> audio_contracts::Result<OwnedAudioFrame> {
    OwnedAudioFrame::new(timestamp, sample_rate, channels, AudioBuffer::F32(samples))
}

fn bench_processing(c: &mut Criterion) {
    let frame = owned_f32_frame(
        Timestamp::new(0, Timebase::new(1, 48_000)),
        48_000,
        1,
        sine(440.0, 48_000, 10.0),
    )
    .unwrap();

    c.bench_function("gain_only_10s", |b| {
        b.iter(|| {
            let mut processor = AudioProcessor::new().gain(0.5);
            processor.process_frame(black_box(frame.clone())).unwrap()
        })
    });

    c.bench_function("processing_chain_10s", |b| {
        b.iter(|| {
            let mut processor = AudioProcessor::new()
                .gain(0.5)
                .mono(ChannelMix::Average)
                .biquad(BiquadSpec {
                    kind: BiquadKind::LowPass,
                    cutoff_hz: 2_000.0,
                    q: 0.707,
                })
                .noise_gate(NoiseGateSpec {
                    threshold: 0.001,
                    attenuation: 0.0,
                });
            processor.process_frame(black_box(frame.clone())).unwrap()
        })
    });
}

criterion_group!(benches, bench_processing);
criterion_main!(benches);
