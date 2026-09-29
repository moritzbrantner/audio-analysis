use audio_analysis_pitch::{AutocorrelationPitchDetector, PitchSmoother};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 16_000;
    let samples = (0..2048)
        .map(|index| {
            let progress = index as f32 / 2048.0;
            let frequency = 220.0 + progress * 220.0;
            let t = index as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect::<Vec<_>>();

    let detector = AutocorrelationPitchDetector::default();
    let mut smoother = PitchSmoother::new(3)?;
    for chunk in samples.chunks(512) {
        let estimate = detector.estimate_samples(chunk, sample_rate)?;
        let smoothed = smoother.smooth(estimate);
        println!("{:?} {:?}", estimate.frequency_hz, smoothed.note_name());
    }
    Ok(())
}
