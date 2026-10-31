//! The finite Silero timestamp port as it was before the segmenter (audio-analysis b80e3f9),
//! kept verbatim as the differential oracle for `SileroSegmenter`.

use media_core::{DetectError, Result};

use crate::silero_vad::{SileroTimestampDetector, SILERO_SAMPLE_RATE, SILERO_WINDOW_SAMPLES};
use crate::SpeechSpan;

pub(super) trait ReferenceTimestamps {
    fn reference_timestamps(
        &self,
        probabilities: &[f32],
        audio_len_samples: usize,
    ) -> Result<Vec<SpeechSpan>>;
}

impl ReferenceTimestamps for SileroTimestampDetector {
    fn reference_timestamps(
        &self,
        probabilities: &[f32],
        audio_len_samples: usize,
    ) -> Result<Vec<SpeechSpan>> {
        if probabilities
            .iter()
            .any(|probability| !probability.is_finite())
        {
            return Err(DetectError::InvalidArgument(
                "silero VAD probabilities must be finite".to_string(),
            ));
        }
        if probabilities.is_empty() || audio_len_samples == 0 {
            return Ok(Vec::new());
        }

        let threshold = self.threshold;
        let neg_threshold = (threshold - 0.15).max(0.01);
        let min_speech_samples =
            SILERO_SAMPLE_RATE as f64 * self.min_speech_duration_ms as f64 / 1000.0;
        let speech_pad_samples =
            (SILERO_SAMPLE_RATE as f64 * self.speech_pad_ms as f64 / 1000.0) as usize;
        let min_silence_samples =
            SILERO_SAMPLE_RATE as f64 * self.min_silence_duration_ms as f64 / 1000.0;
        let min_silence_at_max_speech = SILERO_SAMPLE_RATE as f64 * 98.0 / 1000.0;
        let max_speech_samples = (SILERO_SAMPLE_RATE as f64 * self.max_speech_duration_seconds
            - SILERO_WINDOW_SAMPLES as f64
            - 2.0 * speech_pad_samples as f64)
            .max(0.0);

        let mut triggered = false;
        let mut current_speech = RawSpeech::default();
        let mut speeches = Vec::<RawSpeech>::new();
        let mut temp_end = 0usize;
        let mut prev_end = 0usize;
        let mut next_start = 0usize;
        let mut possible_ends = Vec::<(usize, usize)>::new();

        for (index, speech_prob) in probabilities.iter().copied().enumerate() {
            let cur_sample = SILERO_WINDOW_SAMPLES * index;

            if speech_prob >= threshold && temp_end != 0 {
                let silence_duration = cur_sample.saturating_sub(temp_end);
                if silence_duration as f64 > min_silence_at_max_speech {
                    possible_ends.push((temp_end, silence_duration));
                }
                temp_end = 0;
                if next_start < prev_end {
                    next_start = cur_sample;
                }
            }

            if speech_prob >= threshold && !triggered {
                triggered = true;
                current_speech = RawSpeech {
                    start: cur_sample,
                    end: 0,
                };
                continue;
            }

            if triggered
                && (cur_sample.saturating_sub(current_speech.start)) as f64 > max_speech_samples
            {
                if let Some((best_end, duration)) = possible_ends
                    .iter()
                    .copied()
                    .max_by_key(|(_, duration)| *duration)
                {
                    current_speech.end = best_end;
                    speeches.push(current_speech);
                    current_speech = RawSpeech::default();
                    next_start = best_end + duration;
                    if next_start < best_end + cur_sample {
                        current_speech.start = next_start;
                    } else {
                        triggered = false;
                    }
                    prev_end = 0;
                    next_start = 0;
                    temp_end = 0;
                    possible_ends.clear();
                } else {
                    current_speech.end = cur_sample;
                    speeches.push(current_speech);
                    current_speech = RawSpeech::default();
                    prev_end = 0;
                    next_start = 0;
                    temp_end = 0;
                    triggered = false;
                    possible_ends.clear();
                }
                continue;
            }

            if speech_prob < neg_threshold && triggered {
                if temp_end == 0 {
                    temp_end = cur_sample;
                }
                let silence_duration = cur_sample.saturating_sub(temp_end);
                if (silence_duration as f64) < min_silence_samples {
                    continue;
                }
                current_speech.end = temp_end;
                if (current_speech.end.saturating_sub(current_speech.start)) as f64
                    > min_speech_samples
                {
                    speeches.push(current_speech);
                }
                current_speech = RawSpeech::default();
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                possible_ends.clear();
            }
        }

        if triggered
            && current_speech.start < audio_len_samples
            && audio_len_samples.saturating_sub(current_speech.start) as f64 > min_speech_samples
        {
            current_speech.end = audio_len_samples;
            speeches.push(current_speech);
        }

        apply_speech_padding(&mut speeches, speech_pad_samples, audio_len_samples);
        speeches
            .into_iter()
            .filter(|speech| speech.end > speech.start)
            .map(|speech| {
                let score = max_probability_for_span(probabilities, speech.start, speech.end);
                SpeechSpan::new(
                    speech.start as f64 / SILERO_SAMPLE_RATE as f64,
                    speech.end as f64 / SILERO_SAMPLE_RATE as f64,
                    score,
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RawSpeech {
    start: usize,
    end: usize,
}

fn apply_speech_padding(
    speeches: &mut [RawSpeech],
    speech_pad_samples: usize,
    audio_len_samples: usize,
) {
    for index in 0..speeches.len() {
        if index == 0 {
            speeches[index].start = speeches[index].start.saturating_sub(speech_pad_samples);
        }
        if index + 1 < speeches.len() {
            let silence_duration = speeches[index + 1]
                .start
                .saturating_sub(speeches[index].end);
            if silence_duration < 2 * speech_pad_samples {
                let half = silence_duration / 2;
                speeches[index].end += half;
                speeches[index + 1].start = speeches[index + 1]
                    .start
                    .saturating_sub(silence_duration - half);
            } else {
                speeches[index].end =
                    (speeches[index].end + speech_pad_samples).min(audio_len_samples);
                speeches[index + 1].start =
                    speeches[index + 1].start.saturating_sub(speech_pad_samples);
            }
        } else {
            speeches[index].end = (speeches[index].end + speech_pad_samples).min(audio_len_samples);
        }
    }
}

fn max_probability_for_span(probabilities: &[f32], start: usize, end: usize) -> f32 {
    probabilities
        .iter()
        .enumerate()
        .filter_map(|(index, probability)| {
            let window_start = index * SILERO_WINDOW_SAMPLES;
            let window_end = window_start + SILERO_WINDOW_SAMPLES;
            (window_end > start && window_start < end).then_some(*probability)
        })
        .fold(0.0, f32::max)
}
