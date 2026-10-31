//! Audio chunk windows and log-mel feature extraction.

use crate::{CandleWhisperWindowControls, SpeechSpan};
use candle_transformers::models::whisper;
use media_core::Result;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

#[derive(Debug, Clone)]
pub(super) struct ChunkWindow {
    pub(super) samples: Vec<f32>,
    pub(super) chunk_start_seconds: f64,
    pub(super) local_start_seconds: f64,
    pub(super) local_end_seconds: f64,
    pub(super) global_start_seconds: f64,
    pub(super) global_end_seconds: f64,
}

pub(super) fn chunk_windows(
    samples: &[f32],
    sample_rate: u32,
    chunk: &SpeechSpan,
    controls: &CandleWhisperWindowControls,
) -> Result<Vec<ChunkWindow>> {
    let duration = samples.len() as f64 / sample_rate as f64;
    let padded_start_seconds =
        (chunk.start_seconds - controls.leading_context_seconds).clamp(0.0, duration);
    let padded_end_seconds = (chunk.end_seconds + controls.trailing_context_seconds)
        .clamp(padded_start_seconds, duration);
    let start = seconds_to_index(padded_start_seconds, sample_rate, samples.len());
    let end = seconds_to_index(padded_end_seconds, sample_rate, samples.len()).max(start + 1);
    let max_window = whisper::N_SAMPLES;
    let mut windows = Vec::new();
    let mut cursor = start;
    while cursor < end {
        let window_end = (cursor + max_window).min(end);
        let local_start_seconds = (cursor - start) as f64 / sample_rate as f64;
        let local_end_seconds = (window_end - start) as f64 / sample_rate as f64;
        windows.push(ChunkWindow {
            samples: samples[cursor..window_end].to_vec(),
            chunk_start_seconds: padded_start_seconds,
            local_start_seconds,
            local_end_seconds,
            global_start_seconds: padded_start_seconds + local_start_seconds,
            global_end_seconds: padded_start_seconds + local_end_seconds,
        });
        cursor = window_end;
    }
    Ok(windows)
}

fn seconds_to_index(seconds: f64, sample_rate: u32, limit: usize) -> usize {
    (seconds * sample_rate as f64)
        .round()
        .clamp(0.0, limit as f64) as usize
}

pub(super) fn mel_filter_bank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(sample_rate as f32 / 2.0);
    let mel_points = (0..n_mels + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f32 / (n_mels + 1) as f32)
        .map(mel_to_hz)
        .collect::<Vec<_>>();
    let fft_freqs = (0..n_freqs)
        .map(|index| sample_rate as f32 * index as f32 / n_fft as f32)
        .collect::<Vec<_>>();
    let mut filters = vec![0.0; n_mels * n_freqs];
    for mel_index in 0..n_mels {
        let lower = mel_points[mel_index];
        let center = mel_points[mel_index + 1];
        let upper = mel_points[mel_index + 2];
        for (freq_index, freq) in fft_freqs.iter().enumerate() {
            let value = if *freq < lower || *freq > upper {
                0.0
            } else if *freq <= center {
                (*freq - lower) / (center - lower).max(f32::EPSILON)
            } else {
                (upper - *freq) / (upper - center).max(f32::EPSILON)
            };
            filters[mel_index * n_freqs + freq_index] = value.max(0.0);
        }
    }
    filters
}

pub(super) fn transformers_mel_filter_bank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let min_mel = whisper_slaney_hz_to_mel(0.0);
    let max_mel = whisper_slaney_hz_to_mel(sample_rate as f32 / 2.0);
    let mel_points = (0..n_mels + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f32 / (n_mels + 1) as f32)
        .map(whisper_slaney_mel_to_hz)
        .collect::<Vec<_>>();
    let fft_freqs = (0..n_freqs)
        .map(|index| sample_rate as f32 * index as f32 / n_fft as f32)
        .collect::<Vec<_>>();
    let mut filters = vec![0.0; n_mels * n_freqs];
    for mel_index in 0..n_mels {
        let lower = mel_points[mel_index];
        let center = mel_points[mel_index + 1];
        let upper = mel_points[mel_index + 2];
        for (freq_index, freq) in fft_freqs.iter().enumerate() {
            let value = if *freq < lower || *freq > upper {
                0.0
            } else if *freq <= center {
                (*freq - lower) / (center - lower).max(f32::EPSILON)
            } else {
                (upper - *freq) / (upper - center).max(f32::EPSILON)
            };
            // Transformers Whisper uses librosa/Slaney area normalization.
            // Keep the row-major layout expected by candle's pcm_to_mel.
            let area_normalization = 2.0 / (upper - lower).max(f32::EPSILON);
            filters[mel_index * n_freqs + freq_index] = value.max(0.0) * area_normalization;
        }
    }
    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10_f32.powf(mel / 2595.0) - 1.0)
}

pub(super) fn transformers_whisper_pcm_to_mel(
    config: &whisper::Config,
    samples: &[f32],
    filters: &[f32],
) -> Vec<f32> {
    let n_mel = config.num_mel_bins;
    let n_fft = whisper::N_FFT;
    let n_freqs = n_fft / 2 + 1;
    debug_assert_eq!(filters.len(), n_mel * n_freqs);

    // WhisperFeatureExtractor pads/truncates each window to 30 seconds before
    // its centered, reflect-padded STFT.
    let mut audio = vec![0.0_f32; whisper::N_SAMPLES];
    let copied = samples.len().min(audio.len());
    audio[..copied].copy_from_slice(&samples[..copied]);
    let hann = (0..n_fft)
        .map(|index| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * index as f32 / n_fft as f32).cos()))
        .collect::<Vec<_>>();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buffer = vec![Complex32::new(0.0, 0.0); n_fft];
    let mut mel = vec![0.0_f32; n_mel * whisper::N_FRAMES];
    let center = n_fft as isize / 2;

    // Center padding produces N_FRAMES + 1 frames. Whisper drops the final one.
    for frame in 0..whisper::N_FRAMES {
        let frame_start = frame as isize * whisper::HOP_LENGTH as isize - center;
        for (offset, value) in buffer.iter_mut().enumerate() {
            let index = reflect_index(frame_start + offset as isize, audio.len());
            *value = Complex32::new(audio[index] * hann[offset], 0.0);
        }
        fft.process(&mut buffer);
        for mel_index in 0..n_mel {
            let filter = &filters[mel_index * n_freqs..(mel_index + 1) * n_freqs];
            let energy = buffer[..n_freqs]
                .iter()
                .zip(filter)
                .map(|(value, weight)| value.norm_sqr() * weight)
                .sum::<f32>()
                .max(1e-10);
            mel[mel_index * whisper::N_FRAMES + frame] = energy.log10();
        }
    }

    let floor = mel.iter().copied().max_by(f32::total_cmp).unwrap_or(0.0) - 8.0;
    for value in &mut mel {
        *value = value.max(floor);
        *value = (*value + 4.0) / 4.0;
    }
    mel
}

fn reflect_index(index: isize, len: usize) -> usize {
    debug_assert!(len > 1);
    if index < 0 {
        (-index) as usize
    } else if index >= len as isize {
        (2 * len as isize - index - 2) as usize
    } else {
        index as usize
    }
}

fn whisper_slaney_hz_to_mel(hz: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1_000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    let linear = 3.0 * hz / 200.0;
    if hz < MIN_LOG_HZ {
        linear
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() * (27.0 / 6.4_f32.ln())
    }
}

fn whisper_slaney_mel_to_hz(mel: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1_000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    let linear = 200.0 * mel / 3.0;
    if mel < MIN_LOG_MEL {
        linear
    } else {
        MIN_LOG_HZ * ((6.4_f32.ln() / 27.0) * (mel - MIN_LOG_MEL)).exp()
    }
}
