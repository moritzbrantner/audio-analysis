//! Decode configuration, per-row state, fallback decisions and diagnostics.

use crate::{model_output_mismatch, CandleWhisperDecodeRequestConfig};
use candle_core::{DType, IndexOp, Tensor};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use media_core::Result;
use serde::Deserialize;
use std::io::Write;

use super::*;

#[derive(Debug, Clone, Deserialize)]
pub(super) struct GenerationConfig {
    #[serde(default)]
    pub(super) decoder_start_token_id: Option<u32>,
    #[serde(default)]
    pub(super) eos_token_id: Option<u32>,
    #[serde(default)]
    pub(super) forced_decoder_ids: Option<Vec<(usize, Option<u32>)>>,
    #[serde(default)]
    pub(super) max_length: Option<usize>,
    #[serde(default)]
    pub(super) lang_to_id: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    pub(super) task_to_id: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    pub(super) no_timestamps_token_id: Option<u32>,
    #[serde(default)]
    pub(super) suppress_tokens: Vec<u32>,
    #[serde(default)]
    pub(super) begin_suppress_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WhisperDecodeMode {
    WithoutTimestamps,
    TimestampTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WhisperWindowTiming {
    ChunkWindow,
    WhisperTimestampTokens,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperTimedWindow {
    pub(super) decoded: WhisperDecodedWindow,
    pub(super) timing: WhisperWindowTiming,
    pub(super) fallback_reason: Option<&'static str>,
    pub(super) diagnostics: WhisperDecodeDiagnostics,
    pub(super) conditioning_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WhisperRequestPromptState {
    initial_prompt_tokens: Vec<u32>,
    previous_text_tokens: Vec<u32>,
    condition_on_previous_text: bool,
}

impl WhisperRequestPromptState {
    pub(super) fn new(config: &CandleWhisperDecodeRequestConfig) -> Self {
        Self {
            initial_prompt_tokens: config.initial_prompt_tokens.clone(),
            previous_text_tokens: Vec::new(),
            condition_on_previous_text: config.condition_on_previous_text,
        }
    }

    pub(super) fn current_prompt_tokens(&self, max_prompt_tokens: usize) -> Vec<u32> {
        let initial_count = self.initial_prompt_tokens.len().min(max_prompt_tokens);
        let initial_start = self.initial_prompt_tokens.len() - initial_count;
        let mut tokens = self.initial_prompt_tokens[initial_start..].to_vec();
        if self.condition_on_previous_text {
            let previous_capacity = max_prompt_tokens.saturating_sub(tokens.len());
            let previous_count = self.previous_text_tokens.len().min(previous_capacity);
            let previous_start = self.previous_text_tokens.len() - previous_count;
            tokens.extend_from_slice(&self.previous_text_tokens[previous_start..]);
        }
        tokens
    }

    pub(super) fn record_generated_tokens(&mut self, tokens: &[u32], max_prompt_tokens: usize) {
        if !self.condition_on_previous_text {
            return;
        }
        self.previous_text_tokens.extend_from_slice(tokens);
        let keep = max_prompt_tokens
            .saturating_sub(self.initial_prompt_tokens.len().min(max_prompt_tokens));
        if self.previous_text_tokens.len() > keep {
            let remove = self.previous_text_tokens.len() - keep;
            self.previous_text_tokens.drain(..remove);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct WhisperDecodeDiagnostics {
    pub(super) timestamp_tokens_requested: bool,
    pub(super) timestamp_tokens_present: bool,
    pub(super) decoded_token_ids: Vec<u32>,
    pub(super) decoder_prompt_prefill_count: usize,
    pub(super) decoder_cached_token_step_count: usize,
    pub(super) decoder_input_token_count: usize,
    pub(super) generated_token_count: usize,
    pub(super) decoder_completed_row_count: usize,
    pub(super) decoder_max_active_row_batch_size: usize,
    pub(super) decoder_effective_active_batch_sizes: Vec<usize>,
    pub(super) decoder_active_row_compaction_count: usize,
    pub(super) decoder_self_attention_cache_reused: bool,
    pub(super) decoder_cross_attention_cache_reused: bool,
    pub(super) average_log_probability: f64,
    pub(super) no_speech_probability: Option<f64>,
    pub(super) compression_ratio: f64,
    pub(super) attempted_temperatures: Vec<f64>,
    pub(super) no_speech_rejected: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperTimestampSpec {
    pub(super) begin_token_id: u32,
    pub(super) end_token_id: u32,
    pub(super) seconds_per_token: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperDecodedWindow {
    pub(super) text: String,
    pub(super) segments: Vec<WhisperDecodedSegment>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperDecodedSegment {
    pub(super) text: String,
    pub(super) start_seconds: f64,
    pub(super) end_seconds: f64,
    pub(super) token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WhisperDecoderInputKind {
    PromptPrefill,
    CachedTokenStep,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WhisperDecoderInput {
    pub(super) token_ids: Vec<u32>,
    pub(super) position_offset: usize,
    pub(super) flush_cache: bool,
    pub(super) kind: WhisperDecoderInputKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WhisperAutoregressiveRow {
    tokens: Vec<u32>,
    prompt_len: usize,
    cache_position: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ActiveWhisperDecodeRow {
    pub(super) original_index: usize,
    pub(super) row: WhisperAutoregressiveRow,
    pub(super) stats: WhisperGenerationStats,
    pub(super) score: f64,
    pub(super) no_speech_probability: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperTokenDecodeResult {
    pub(super) token_ids: Vec<u32>,
    pub(super) stats: WhisperGenerationStats,
    pub(super) average_log_probability: f64,
    pub(super) no_speech_probability: Option<f64>,
    pub(super) attempted_temperatures: Vec<f64>,
    pub(super) no_speech_rejected: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WhisperInitialTokens {
    pub(super) token_ids: Vec<u32>,
    pub(super) sot_position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WhisperFallbackAttemptDecision {
    Accept,
    Retry,
    RejectNoSpeech,
}

pub(super) fn fallback_attempt_decision(
    config: &CandleWhisperDecodeRequestConfig,
    average_log_probability: f64,
    no_speech_probability: Option<f64>,
    compression_ratio: f64,
) -> WhisperFallbackAttemptDecision {
    let high_no_speech = config
        .max_no_speech_probability
        .zip(no_speech_probability)
        .is_some_and(|(maximum, observed)| observed > maximum);
    let low_log_probability = config
        .min_average_log_probability
        .is_some_and(|minimum| average_log_probability < minimum);
    if high_no_speech && (config.min_average_log_probability.is_none() || low_log_probability) {
        return WhisperFallbackAttemptDecision::RejectNoSpeech;
    }
    if low_log_probability
        || config
            .max_compression_ratio
            .is_some_and(|maximum| compression_ratio > maximum)
    {
        return WhisperFallbackAttemptDecision::Retry;
    }
    WhisperFallbackAttemptDecision::Accept
}

/// One completed decode at one temperature of the fallback schedule.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct WhisperFallbackAttempt {
    pub(super) token_ids: Vec<u32>,
    pub(super) temperature: f64,
    pub(super) average_log_probability: f64,
    pub(super) no_speech_probability: Option<f64>,
    pub(super) compression_ratio: f64,
    pub(super) decision: WhisperFallbackAttemptDecision,
}

/// Picks the decode result from the ordered attempts, as faster-whisper's
/// `generate_with_fallback` does: the last attempt when it passed (or was
/// rejected as silence); when every temperature still needed fallback, the
/// attempt with the highest average log probability among those under the
/// compression-ratio threshold, or among all attempts when none is.
pub(super) fn select_fallback_attempt(
    config: &CandleWhisperDecodeRequestConfig,
    mut attempts: Vec<WhisperFallbackAttempt>,
) -> WhisperFallbackAttempt {
    let last = attempts
        .pop()
        .expect("temperature fallback always records at least one attempt");
    if last.decision != WhisperFallbackAttemptDecision::Retry {
        return last;
    }
    attempts.push(last);
    let below_compression_threshold = |attempt: &WhisperFallbackAttempt| {
        config
            .max_compression_ratio
            .is_none_or(|maximum| attempt.compression_ratio <= maximum)
    };
    let pool_has_passing_ratio = attempts.iter().any(below_compression_threshold);
    attempts
        .into_iter()
        .filter(|attempt| !pool_has_passing_ratio || below_compression_threshold(attempt))
        .reduce(|best, attempt| {
            if attempt.average_log_probability > best.average_log_probability {
                attempt
            } else {
                best
            }
        })
        .expect("filtered pool keeps at least one attempt")
}

pub(super) fn finish_temperature_fallback(
    config: &CandleWhisperDecodeRequestConfig,
    attempts: Vec<WhisperFallbackAttempt>,
    mut stats: WhisperGenerationStats,
) -> WhisperTokenDecodeResult {
    let attempted_temperatures = attempts.iter().map(|attempt| attempt.temperature).collect();
    let mut selected = select_fallback_attempt(config, attempts);
    let no_speech_rejected = selected.decision == WhisperFallbackAttemptDecision::RejectNoSpeech;
    if no_speech_rejected {
        selected.token_ids.clear();
    }
    stats.record_completed_row();
    WhisperTokenDecodeResult {
        token_ids: selected.token_ids,
        stats,
        average_log_probability: selected.average_log_probability,
        no_speech_probability: selected.no_speech_probability,
        attempted_temperatures,
        no_speech_rejected,
    }
}

fn token_probability(logits: &[f32], token_id: u32) -> Option<f64> {
    let target = *logits.get(token_id as usize)? as f64;
    if !target.is_finite() {
        return Some(0.0);
    }
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)? as f64;
    let denominator = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| (value as f64 - max).exp())
        .sum::<f64>();
    Some((target - max).exp() / denominator)
}

pub(super) fn tensor_token_probability_at_position(
    logits: &Tensor,
    row: usize,
    position: usize,
    token_id: u32,
) -> Result<Option<f64>> {
    let position_logits = logits
        .i((row, position, ..))
        .and_then(|logits| logits.to_dtype(DType::F32))
        .and_then(|logits| logits.to_vec1::<f32>())
        .map_err(|error| {
            model_output_mismatch(format!(
                "failed to read Whisper position {position} logits: {error}"
            ))
        })?;
    Ok(token_probability(&position_logits, token_id))
}

pub(super) fn token_log_probability(logits: &[f32], token_id: u32) -> Option<f64> {
    token_probability(logits, token_id).map(f64::ln)
}

/// Whisper-compatible ratio: UTF-8 byte length divided by zlib-compressed size.
pub(super) fn text_compression_ratio(text: &str) -> Result<f64> {
    if text.is_empty() {
        return Ok(0.0);
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(text.as_bytes()).map_err(|error| {
        model_output_mismatch(format!("failed to compress Whisper output: {error}"))
    })?;
    let compressed = encoder.finish().map_err(|error| {
        model_output_mismatch(format!(
            "failed to finish Whisper output compression: {error}"
        ))
    })?;
    Ok(text.len() as f64 / compressed.len().max(1) as f64)
}

pub(super) fn average_log_probability(score: f64, generated_len: usize, completed: bool) -> f64 {
    score / (generated_len + usize::from(completed)).max(1) as f64
}

impl WhisperAutoregressiveRow {
    pub(super) fn new(prompt_tokens: Vec<u32>) -> Self {
        Self {
            tokens: prompt_tokens,
            prompt_len: 0,
            cache_position: 0,
        }
        .with_prompt_len()
    }

    fn with_prompt_len(mut self) -> Self {
        self.prompt_len = self.tokens.len();
        self
    }

    pub(super) fn next_decoder_input(&self) -> WhisperDecoderInput {
        if self.cache_position == 0 {
            return WhisperDecoderInput {
                token_ids: self.tokens.clone(),
                position_offset: 0,
                flush_cache: true,
                kind: WhisperDecoderInputKind::PromptPrefill,
            };
        }
        let last_token = self
            .tokens
            .last()
            .copied()
            .expect("autoregressive row must retain at least the prompt token");
        WhisperDecoderInput {
            token_ids: vec![last_token],
            position_offset: self.tokens.len() - 1,
            flush_cache: false,
            kind: WhisperDecoderInputKind::CachedTokenStep,
        }
    }

    pub(super) fn mark_forwarded(&mut self) {
        self.cache_position = self.tokens.len();
    }

    pub(super) fn generated_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len..]
    }

    pub(super) fn accept(&mut self, token: u32) {
        self.tokens.push(token);
    }

    pub(super) fn into_generated_tokens(self) -> Vec<u32> {
        self.tokens.into_iter().skip(self.prompt_len).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct WhisperGenerationStats {
    prompt_prefill_count: usize,
    cached_token_step_count: usize,
    pub(super) decoder_input_token_count: usize,
    pub(super) generated_token_count: usize,
    completed_row_count: usize,
    max_active_row_batch_size: usize,
    effective_active_batch_sizes: Vec<usize>,
    active_row_compaction_count: usize,
    decoder_self_attention_cache_reused: bool,
    decoder_cross_attention_cache_reused: bool,
}

impl WhisperGenerationStats {
    pub(super) fn record_input(&mut self, input: &WhisperDecoderInput) {
        match input.kind {
            WhisperDecoderInputKind::PromptPrefill => self.prompt_prefill_count += 1,
            WhisperDecoderInputKind::CachedTokenStep => self.cached_token_step_count += 1,
        }
        self.decoder_input_token_count += input.token_ids.len();
    }

    pub(super) fn record_generated_token(&mut self) {
        self.generated_token_count += 1;
    }

    pub(super) fn record_active_row_batch_size(&mut self, batch_size: usize) {
        self.max_active_row_batch_size = self.max_active_row_batch_size.max(batch_size);
        self.effective_active_batch_sizes.push(batch_size);
    }

    pub(super) fn record_active_row_compaction(&mut self) {
        self.active_row_compaction_count += 1;
    }

    pub(super) fn record_completed_row(&mut self) {
        self.completed_row_count = 1;
    }

    pub(super) fn record_decoder_stats(&mut self, stats: CachedWhisperDecoderStats) {
        self.decoder_self_attention_cache_reused |= stats.self_attention_cache_reused;
        self.decoder_cross_attention_cache_reused |= stats.cross_attention_cache_reused;
    }

    pub(super) fn extend(self, diagnostics: &mut WhisperDecodeDiagnostics) {
        diagnostics.decoder_prompt_prefill_count += self.prompt_prefill_count;
        diagnostics.decoder_cached_token_step_count += self.cached_token_step_count;
        diagnostics.decoder_input_token_count += self.decoder_input_token_count;
        diagnostics.generated_token_count += self.generated_token_count;
        diagnostics.decoder_completed_row_count += self.completed_row_count;
        diagnostics.decoder_max_active_row_batch_size = diagnostics
            .decoder_max_active_row_batch_size
            .max(self.max_active_row_batch_size);
        diagnostics
            .decoder_effective_active_batch_sizes
            .extend(self.effective_active_batch_sizes);
        diagnostics.decoder_active_row_compaction_count += self.active_row_compaction_count;
        diagnostics.decoder_self_attention_cache_reused |= self.decoder_self_attention_cache_reused;
        diagnostics.decoder_cross_attention_cache_reused |=
            self.decoder_cross_attention_cache_reused;
    }
}

impl WhisperDecodeDiagnostics {
    pub(super) fn add_generation_counts_from(&mut self, other: &Self) {
        self.decoder_prompt_prefill_count += other.decoder_prompt_prefill_count;
        self.decoder_cached_token_step_count += other.decoder_cached_token_step_count;
        self.decoder_input_token_count += other.decoder_input_token_count;
        self.generated_token_count += other.generated_token_count;
        self.decoder_completed_row_count += other.decoder_completed_row_count;
        self.decoder_max_active_row_batch_size = self
            .decoder_max_active_row_batch_size
            .max(other.decoder_max_active_row_batch_size);
        self.decoder_effective_active_batch_sizes
            .extend(other.decoder_effective_active_batch_sizes.iter().copied());
        self.decoder_active_row_compaction_count += other.decoder_active_row_compaction_count;
        self.decoder_self_attention_cache_reused |= other.decoder_self_attention_cache_reused;
        self.decoder_cross_attention_cache_reused |= other.decoder_cross_attention_cache_reused;
    }
}
