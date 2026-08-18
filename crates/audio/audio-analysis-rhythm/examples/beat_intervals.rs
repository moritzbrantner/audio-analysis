use audio_analysis_core::FrameSpec;
use audio_analysis_rhythm::{
    beat_grid, detect_onsets, estimate_tempo, inter_onset_intervals, onset_envelope,
    OnsetDetectorConfig, TempoEstimatorConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 2_000;
    let mut samples = vec![0.0_f32; sample_rate as usize * 2];
    for index in (0..samples.len()).step_by(500) {
        samples[index] = 1.0;
    }

    let envelope = onset_envelope(&samples, sample_rate, FrameSpec::new(80, 20)?)?;
    let onsets = detect_onsets(&envelope, OnsetDetectorConfig::default())?;
    let tempo = estimate_tempo(&onsets, TempoEstimatorConfig::default())?;
    let intervals = inter_onset_intervals(&onsets);
    let beats = beat_grid(
        onsets
            .first()
            .map(|onset| onset.timestamp_seconds)
            .unwrap_or(0.0),
        tempo.bpm.unwrap_or(120.0),
        4,
    )?;

    println!(
        "onsets={} intervals={intervals:?} beats={beats:?}",
        onsets.len()
    );
    Ok(())
}
