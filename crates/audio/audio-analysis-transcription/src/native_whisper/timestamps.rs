//! Timestamp token decoding, logit suppression rules and segment projection.

use crate::{
    invalid_request, model_output_mismatch, CandleWhisperDecodeRequestConfig,
    CandleWhisperTimingMode,
};
use candle_transformers::models::whisper;
#[cfg(test)]
use media_core::TranscriptWordContract;
use media_core::{Result, TranscriptSegmentContract};
#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use tokenizers::Tokenizer;

use super::*;

const WHISPER_TIMESTAMP_SECONDS_PER_TOKEN: f64 = 0.02;
pub(super) const WHISPER_START_OF_PREV_TOKEN: &str = "<|startofprev|>";
const WHISPER_TIMESTAMP_TOKEN_COUNT: u32 =
    (whisper::CHUNK_LENGTH as f64 / WHISPER_TIMESTAMP_SECONDS_PER_TOKEN) as u32 + 1;

pub(super) fn decode_timestamp_window(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
) -> Result<Option<WhisperDecodedWindow>> {
    let spec = whisper_timestamp_spec(tokenizer)?;
    decode_timestamp_window_with_spec(tokenizer, token_ids, &spec)
}

fn decode_timestamp_window_with_spec(
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    spec: &WhisperTimestampSpec,
) -> Result<Option<WhisperDecodedWindow>> {
    let mut segments = Vec::new();
    let mut pending_text_tokens = Vec::new();
    let mut segment_start = None;
    let mut previous_timestamp = None;
    let mut saw_timestamp = false;

    for token_id in token_ids {
        if let Some(seconds) = timestamp_seconds(*token_id, spec) {
            saw_timestamp = true;
            if let Some(previous) = previous_timestamp {
                if seconds < previous {
                    return Err(model_output_mismatch(format!(
                        "Whisper timestamp tokens are not monotonic: {seconds:.2} after {previous:.2}"
                    )));
                }
            }
            if !pending_text_tokens.is_empty() {
                let start_seconds = segment_start.unwrap_or(seconds);
                if seconds < start_seconds {
                    return Err(model_output_mismatch(format!(
                        "Whisper timestamp segment ends before it starts: {seconds:.2} < {start_seconds:.2}"
                    )));
                }
                let text = decode_text_tokens(tokenizer, &pending_text_tokens)?;
                if !text.is_empty() {
                    segments.push(WhisperDecodedSegment {
                        text,
                        start_seconds,
                        end_seconds: seconds,
                        token_ids: std::mem::take(&mut pending_text_tokens),
                    });
                } else {
                    pending_text_tokens.clear();
                }
            }
            segment_start = Some(seconds);
            previous_timestamp = Some(seconds);
        } else {
            pending_text_tokens.push(*token_id);
        }
    }

    if !pending_text_tokens.is_empty() {
        return Ok(None);
    }
    if !saw_timestamp {
        return Ok(None);
    }

    let text = segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(Some(WhisperDecodedWindow { text, segments }))
}

pub(super) fn whisper_timestamp_spec(tokenizer: &Tokenizer) -> Result<WhisperTimestampSpec> {
    let begin_token_id = token_id(tokenizer, "<|0.00|>").ok_or_else(|| {
        invalid_request("Whisper tokenizer is missing timestamp token `<|0.00|>`")
    })?;
    let end_token_id = token_id(tokenizer, "<|30.00|>")
        .map(|token| token + 1)
        .unwrap_or(begin_token_id + WHISPER_TIMESTAMP_TOKEN_COUNT);
    if end_token_id <= begin_token_id {
        return Err(invalid_request(
            "Whisper timestamp token range is empty or malformed",
        ));
    }
    Ok(WhisperTimestampSpec {
        begin_token_id,
        end_token_id,
        seconds_per_token: WHISPER_TIMESTAMP_SECONDS_PER_TOKEN,
    })
}

fn optional_whisper_timestamp_spec(tokenizer: &Tokenizer) -> Result<Option<WhisperTimestampSpec>> {
    if token_id(tokenizer, "<|0.00|>").is_none() {
        return Ok(None);
    }
    whisper_timestamp_spec(tokenizer).map(Some)
}

pub(super) fn timestamp_spec_for_timing_mode(
    tokenizer: &Tokenizer,
    mode: CandleWhisperTimingMode,
) -> Result<Option<WhisperTimestampSpec>> {
    match mode {
        CandleWhisperTimingMode::Auto => optional_whisper_timestamp_spec(tokenizer),
        CandleWhisperTimingMode::NoTimestamps => Ok(None),
        CandleWhisperTimingMode::TimestampTokensRequired => {
            whisper_timestamp_spec(tokenizer).map(Some)
        }
    }
}

pub(super) fn timestamp_seconds(token_id: u32, spec: &WhisperTimestampSpec) -> Option<f64> {
    (spec.begin_token_id..spec.end_token_id)
        .contains(&token_id)
        .then(|| (token_id - spec.begin_token_id) as f64 * spec.seconds_per_token)
}

pub(super) fn apply_timestamp_logit_rules(
    logits: &mut [f32],
    generated: &[u32],
    spec: &WhisperTimestampSpec,
    eos: u32,
) -> Result<()> {
    let begin = spec.begin_token_id;
    let end = spec.end_token_id;
    let last_was_timestamp = generated
        .last()
        .is_some_and(|token| timestamp_seconds(*token, spec).is_some());
    let penultimate_was_timestamp = generated
        .get(generated.len().saturating_sub(2))
        .is_none_or(|token| timestamp_seconds(*token, spec).is_some());
    if last_was_timestamp {
        if penultimate_was_timestamp {
            suppress_range(logits, begin, end);
        } else {
            suppress_range(logits, 0, eos);
        }
    }
    if generated.is_empty() {
        let max_initial_timestamp = begin + (1.0 / spec.seconds_per_token).round() as u32;
        suppress_range(logits, max_initial_timestamp + 1, end);
    }
    if let Some(last_timestamp) = generated
        .iter()
        .rev()
        .find(|token| timestamp_seconds(**token, spec).is_some())
    {
        suppress_range(logits, begin, *last_timestamp);
    }
    let timestamp_logprob = logsumexp_range(logits, begin, end);
    let max_text_logprob = max_finite_range(logits, 0, begin);
    if let (Some(timestamp_logprob), Some(max_text_logprob)) = (timestamp_logprob, max_text_logprob)
    {
        if timestamp_logprob > max_text_logprob {
            suppress_range(logits, 0, begin);
        }
    }
    Ok(())
}

pub(super) fn suppress_token(logits: &mut [f32], token: u32) {
    if let Some(logit) = logits.get_mut(token as usize) {
        *logit = f32::NEG_INFINITY;
    }
}

pub(super) fn request_suppressed_token_ids(
    tokenizer: &Tokenizer,
    config: &CandleWhisperDecodeRequestConfig,
) -> BTreeSet<u32> {
    let mut suppressed = config
        .suppressed_token_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if config.suppress_numerals {
        suppressed.extend(
            tokenizer
                .get_vocab(true)
                .into_iter()
                .filter_map(|(token, token_id)| {
                    let special = token.starts_with("<|") && token.ends_with("|>");
                    (!special && token.chars().any(char::is_numeric)).then_some(token_id)
                }),
        );
    }
    suppressed
}

pub(super) fn suppress_range(logits: &mut [f32], start: u32, end: u32) {
    let start = start as usize;
    let end = (end as usize).min(logits.len());
    if start >= end {
        return;
    }
    for logit in &mut logits[start..end] {
        *logit = f32::NEG_INFINITY;
    }
}

pub(super) fn argmax_finite(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
}

fn max_finite_range(logits: &[f32], start: u32, end: u32) -> Option<f32> {
    let start = start as usize;
    let end = (end as usize).min(logits.len());
    if start >= end {
        return None;
    }
    logits[start..end]
        .iter()
        .copied()
        .filter(|logit| logit.is_finite())
        .max_by(f32::total_cmp)
}

fn logsumexp_range(logits: &[f32], start: u32, end: u32) -> Option<f32> {
    let start = start as usize;
    let end = (end as usize).min(logits.len());
    if start >= end {
        return None;
    }
    let max = logits[start..end]
        .iter()
        .copied()
        .filter(|logit| logit.is_finite())
        .max_by(f32::total_cmp)?;
    let sum = logits[start..end]
        .iter()
        .copied()
        .filter(|logit| logit.is_finite())
        .map(|logit| (logit - max).exp())
        .sum::<f32>();
    (sum > 0.0).then(|| max + sum.ln())
}

pub(super) fn has_stable_timestamp_segments(
    decoded: &WhisperDecodedWindow,
    samples: &[f32],
) -> bool {
    let audio_duration = samples.len() as f64 / whisper::SAMPLE_RATE as f64;
    let joined_text = decoded
        .segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    !joined_text.trim().is_empty()
        && joined_text
            .trim()
            .chars()
            .any(|character| character.is_alphanumeric())
        && decoded.segments.iter().all(|segment| {
            !segment.text.trim().is_empty()
                && segment.end_seconds > segment.start_seconds
                && segment.start_seconds >= 0.0
                && segment.end_seconds <= audio_duration + 0.5
        })
        && decoded
            .segments
            .last()
            .is_some_and(|segment| segment.end_seconds >= audio_duration * 0.85)
}

pub(super) fn decode_text_tokens(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(token_ids, true)
        .map(clean_decoded_text)
        .map_err(|error| model_output_mismatch(format!("failed to decode Whisper tokens: {error}")))
}

fn clean_decoded_text(text: String) -> String {
    text.replace("  ", " ").trim().to_string()
}

pub(super) fn window_fallback_segment(
    index: u64,
    text: String,
    start_seconds: f64,
    end_seconds: f64,
    language: Option<String>,
) -> Result<TranscriptSegmentContract> {
    let mut segment = TranscriptSegmentContract::new(index, text)
        .with_time_range(Some(start_seconds), Some(end_seconds))?;
    segment.language = language;
    segment
        .attributes
        .insert("provider".to_string(), "candle-whisper".to_string());
    segment
        .attributes
        .insert("timing".to_string(), "global".to_string());
    Ok(segment)
}

pub(super) fn decoded_window_to_contract_segments(
    decoded: WhisperDecodedWindow,
    next_index: &mut u64,
    window_start_seconds: f64,
    window_end_seconds: f64,
    language: Option<String>,
) -> Result<Vec<TranscriptSegmentContract>> {
    let mut segments = Vec::new();
    for decoded_segment in decoded.segments {
        let text = decoded_segment.text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        let global_start = (window_start_seconds + decoded_segment.start_seconds)
            .clamp(window_start_seconds, window_end_seconds);
        let global_end = (window_start_seconds + decoded_segment.end_seconds)
            .clamp(window_start_seconds, window_end_seconds);
        let mut segment = TranscriptSegmentContract::new(*next_index, text)
            .with_time_range(Some(global_start), Some(global_end))?;
        *next_index += 1;
        segment.language = language.clone();
        segment
            .attributes
            .insert("provider".to_string(), "candle-whisper".to_string());
        segment
            .attributes
            .insert("timing".to_string(), "global".to_string());
        segment.attributes.insert(
            "timingSource".to_string(),
            "whisperTimestampTokens".to_string(),
        );
        segments.push(segment);
    }
    Ok(segments)
}

#[cfg(test)]
pub(super) fn project_words_from_timestamp_segment(
    segment: &WhisperDecodedSegment,
) -> Vec<TranscriptWordContract> {
    let text = segment.text.trim();
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.is_empty() {
        return Vec::new();
    }

    let word_count = words.len();
    let weights = words
        .iter()
        .map(|word| word.chars().count())
        .collect::<Vec<_>>();
    let total_chars = weights.iter().sum::<usize>();
    if total_chars == 0 {
        return Vec::new();
    }

    let start = segment.start_seconds;
    let end = segment.end_seconds;
    let duration = (end - start).max(0.0);
    let mut cursor = start;
    words
        .into_iter()
        .zip(weights)
        .enumerate()
        .map(|(index, (word, weight))| {
            let word_start = cursor.clamp(start, end);
            let projected_end = if index + 1 == word_count {
                end
            } else {
                cursor + duration * weight as f64 / total_chars as f64
            };
            let word_end = projected_end.clamp(word_start, end);
            cursor = word_end;
            let mut contract = TranscriptWordContract::new(word)
                .with_time_range(Some(word_start), Some(word_end))
                .expect("projected word timing stays inside its decoded segment");
            contract.attributes = BTreeMap::from([(
                "timing".to_string(),
                "whisperTimestampProjection".to_string(),
            )]);
            contract
        })
        .collect()
}

pub(super) fn token_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer.token_to_id(token)
}
