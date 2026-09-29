use audio_analysis_fourier::{spectral_flux, spectrogram, FourierTransform, StftConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 8_000;
    let samples = (0..1024)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect::<Vec<_>>();

    let frames = spectrogram(
        &samples,
        sample_rate,
        &StftConfig::new(256, 128)?.pad_final_frame(true),
    )?;
    let transform = FourierTransform::new(256)?;
    let full = transform.analyze_samples(&samples[..256], sample_rate)?;

    println!(
        "frames={} dominant={:?} first_flux={}",
        frames.len(),
        full.dominant_frequency_hz(),
        if frames.len() > 1 {
            spectral_flux(&frames[0].spectrum, &frames[1].spectrum)
        } else {
            0.0
        }
    );
    Ok(())
}
