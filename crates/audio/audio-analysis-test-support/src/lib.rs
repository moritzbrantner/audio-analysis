//! Public API for the audio-analysis-test-support crate.

pub mod surface;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use audio_contracts::{AudioBuffer, OwnedAudioFrame, Result, Timestamp};

/// Generates a sine wave.
pub fn sine(freq_hz: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
    let samples = (sample_rate as f32 * seconds) as usize;
    sine_len(freq_hz, sample_rate, samples)
}

/// Generates a sine wave with a fixed sample count.
pub fn sine_len(freq_hz: f32, sample_rate: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            (2.0 * PI * freq_hz * t).sin()
        })
        .collect()
}

/// Returns impulse train.
pub fn impulse_train(sample_rate: u32, bpm: f32, seconds: f32) -> Vec<f32> {
    let len = (sample_rate as f32 * seconds) as usize;
    let interval = (sample_rate as f32 * 60.0 / bpm).max(1.0) as usize;
    let mut samples = vec![0.0; len];
    for index in (0..len).step_by(interval) {
        samples[index] = 1.0;
    }
    samples
}

/// Returns click track.
pub fn click_track(sample_rate: u32, bpm: f32, seconds: f32) -> Vec<f32> {
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

/// Returns white noise.
pub fn white_noise(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            value * 2.0 - 1.0
        })
        .collect()
}

/// Returns pink noise.
pub fn pink_noise(seed: u64, len: usize) -> Vec<f32> {
    let white = white_noise(seed, len.max(1));
    let mut acc = 0.0_f32;
    white
        .into_iter()
        .map(|sample| {
            acc = 0.98 * acc + 0.02 * sample;
            acc.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Returns chirp.
pub fn chirp(start_hz: f32, end_hz: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
    let samples = (sample_rate as f32 * seconds) as usize;
    (0..samples)
        .map(|index| {
            let progress = index as f32 / samples.max(1) as f32;
            let frequency = start_hz + (end_hz - start_hz) * progress;
            let t = index as f32 / sample_rate as f32;
            (2.0 * PI * frequency * t).sin()
        })
        .collect()
}

/// Returns stepped tones.
pub fn stepped_tones(tones: &[(f32, f32)], sample_rate: u32) -> Vec<f32> {
    tones
        .iter()
        .flat_map(|(frequency, seconds)| sine(*frequency, sample_rate, *seconds))
        .collect()
}

/// Returns mixed sources.
pub fn mixed_sources(tracks: &[Vec<f32>]) -> Vec<f32> {
    let len = tracks.iter().map(Vec::len).max().unwrap_or(0);
    let mut mixed = vec![0.0_f32; len];
    for track in tracks {
        for (out, sample) in mixed
            .iter_mut()
            .zip(track.iter().copied().chain(std::iter::repeat(0.0)))
        {
            *out = (*out + sample).clamp(-1.0, 1.0);
        }
    }
    mixed
}

/// Returns common sample rates.
pub fn common_sample_rates() -> &'static [u32] {
    &[8_000, 16_000, 44_100, 48_000]
}

/// Returns temp wav dir.
pub fn temp_wav_dir() -> io::Result<tempfile::TempDir> {
    tempfile::tempdir()
}

/// Returns interleaved stereo.
pub fn interleaved_stereo(left: &[f32], right: &[f32]) -> Vec<f32> {
    assert_eq!(left.len(), right.len(), "stereo channels must match");
    left.iter()
        .zip(right)
        .flat_map(|(left, right)| [*left, *right])
        .collect()
}

/// Returns owned f32 frame.
pub fn owned_f32_frame(
    timestamp: Timestamp,
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
) -> Result<OwnedAudioFrame> {
    OwnedAudioFrame::new(timestamp, sample_rate, channels, AudioBuffer::F32(samples))
}

/// Returns owned i16 frame.
pub fn owned_i16_frame(
    timestamp: Timestamp,
    sample_rate: u32,
    channels: u16,
    samples: Vec<i16>,
) -> Result<OwnedAudioFrame> {
    OwnedAudioFrame::new(timestamp, sample_rate, channels, AudioBuffer::I16(samples))
}

/// Writes pcm16 wav.
pub fn write_pcm16_wav(
    path: impl AsRef<Path>,
    sample_rate: u32,
    channels: u16,
    samples: &[f32],
) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    let data_len = samples.len() as u32 * 2;
    let byte_rate = sample_rate * channels as u32 * 2;
    let block_align = channels * 2;

    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_len).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16_u32.to_le_bytes())?;
    file.write_all(&1_u16.to_le_bytes())?;
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&16_u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

/// Asserts that two values are approximately equal.
pub fn assert_approx_eq(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {actual} to be within {tolerance} of {expected}"
    );
}

/// Asserts that two slices are approximately equal.
pub fn assert_approx_slice(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "slice lengths differ");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "index {index}: expected {actual} to be within {tolerance} of {expected}"
        );
    }
}

#[cfg(test)]
mod tests {
    use audio_contracts::Timebase;

    use super::*;

    #[test]
    fn generated_audio_fixtures_have_expected_shape() {
        let left = sine(440.0, 8_000, 0.01);
        let right = white_noise(1, left.len());
        let stereo = interleaved_stereo(&left, &right);

        assert_eq!(left.len(), 80);
        assert_eq!(stereo.len(), 160);
        assert_eq!(click_track(1_000, 120.0, 1.0)[0], 1.0);
        assert_eq!(impulse_train(1_000, 120.0, 1.0)[0], 1.0);
        assert_eq!(pink_noise(1, 80).len(), 80);
        assert_eq!(chirp(220.0, 440.0, 8_000, 0.01).len(), 80);
        assert_eq!(
            stepped_tones(&[(220.0, 0.01), (440.0, 0.01)], 8_000).len(),
            160
        );
        assert_eq!(mixed_sources(&[vec![0.5; 4], vec![0.25; 4]]), vec![0.75; 4]);
    }

    #[test]
    fn owned_frame_helpers_preserve_metadata() {
        let timestamp = Timestamp::new(2, Timebase::new(1, 8_000));
        let frame = owned_f32_frame(timestamp, 8_000, 1, vec![0.0, 0.5]).unwrap();

        assert_eq!(frame.timestamp, timestamp);
        assert_eq!(frame.sample_rate, 8_000);
        assert_eq!(frame.channels, 1);
        assert_eq!(frame.data.len(), 2);
    }
}
