use audio_analysis_synthesis::{
    synthesize_click, synthesize_noise_burst, synthesize_tone, AudioSynthesisConfig, ToneSpec,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AudioSynthesisConfig::new(8_000, 1)?;
    let tone = synthesize_tone(ToneSpec::sine(440.0, 0.1), config)?;
    let click = synthesize_click(0.0, 0.01, 0.8, AudioSynthesisConfig::new(8_000, 1)?)?;
    let noise = synthesize_noise_burst(0.05, 0.3, 42, AudioSynthesisConfig::new(8_000, 1)?)?;

    println!(
        "tone={} click={} noise={}",
        tone.value.samples_per_channel(),
        click.value.samples_per_channel(),
        noise.value.samples_per_channel()
    );
    Ok(())
}
