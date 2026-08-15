#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use candle_core::quantized::{gguf_file, GgmlDType};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    embedding, linear, linear_no_bias, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module,
    VarBuilder,
};
use candle_transformers::models::whisper::{self};
use flate2::{write::ZlibEncoder, Compression};
use media_core::Result;
use rustfft::{num_complex::Complex32, FftPlanner};
use serde::Deserialize;
use std::io::Write;
#[cfg(test)]
use text_transcripts::TranscriptWordContract;
use text_transcripts::{TranscriptSegmentContract, TranscriptionContract};
use tokenizers::Tokenizer;

use crate::native_device::{resolve_native_device, ResolvedNativeDevice};
use crate::native_whisper_quantized::CandleQ8WhisperModel;
use crate::{
    candle_batch_count, invalid_request, model_output_mismatch, setup_error, validate_asr_request,
    AsrRequest, AsrResponse, CandleWhisperComputeType, CandleWhisperDecodeRequestConfig,
    CandleWhisperDecodeRuntime, CandleWhisperOptions, CandleWhisperRuntimeControls,
    CandleWhisperTimingMode, CandleWhisperTranscriptionRequestConfig, CandleWhisperWindowControls,
    SpeechActivitySegment, TranscriptionTask,
};

const WHISPER_TIMESTAMP_SECONDS_PER_TOKEN: f64 = 0.02;
const WHISPER_START_OF_PREV_TOKEN: &str = "<|startofprev|>";
const WHISPER_TIMESTAMP_TOKEN_COUNT: u32 =
    (whisper::CHUNK_LENGTH as f64 / WHISPER_TIMESTAMP_SECONDS_PER_TOKEN) as u32 + 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperBundlePaths {
    pub root: PathBuf,
    pub config_json: PathBuf,
    pub generation_config_json: PathBuf,
    pub tokenizer_json: PathBuf,
    pub preprocessor_config_json: PathBuf,
    pub model_safetensors: PathBuf,
    pub model_q8_0_gguf: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperModelFormat {
    Safetensors,
    GgufQ8_0,
}

impl WhisperModelFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::GgufQ8_0 => "gguf-q8_0",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperRunSetup {
    model_id: String,
    task: TranscriptionTask,
    language: Option<String>,
    bundle: WhisperBundlePaths,
    model_source: &'static str,
    resolved_device: ResolvedNativeDevice,
    requested_compute_type: CandleWhisperComputeType,
    resolved_compute_type: CandleWhisperComputeType,
    model_weight_dtype: DType,
    model_format: WhisperModelFormat,
}

#[derive(Debug, Clone)]
struct ResolvedWhisperModel {
    model_id: String,
    bundle: WhisperBundlePaths,
    source: &'static str,
}

#[derive(Debug, Clone, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    decoder_start_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    forced_decoder_ids: Option<Vec<(usize, Option<u32>)>>,
    #[serde(default)]
    max_length: Option<usize>,
    #[serde(default)]
    lang_to_id: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    task_to_id: std::collections::BTreeMap<String, u32>,
    #[serde(default)]
    no_timestamps_token_id: Option<u32>,
    #[serde(default)]
    suppress_tokens: Vec<u32>,
    #[serde(default)]
    begin_suppress_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperDecodeMode {
    WithoutTimestamps,
    TimestampTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperWindowTiming {
    ChunkWindow,
    WhisperTimestampTokens,
}

#[derive(Debug, Clone, PartialEq)]
struct WhisperTimedWindow {
    decoded: WhisperDecodedWindow,
    timing: WhisperWindowTiming,
    fallback_reason: Option<&'static str>,
    diagnostics: WhisperDecodeDiagnostics,
    conditioning_token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperRequestPromptState {
    initial_prompt_tokens: Vec<u32>,
    previous_text_tokens: Vec<u32>,
    condition_on_previous_text: bool,
}

impl WhisperRequestPromptState {
    fn new(config: &CandleWhisperDecodeRequestConfig) -> Self {
        Self {
            initial_prompt_tokens: config.initial_prompt_tokens.clone(),
            previous_text_tokens: Vec::new(),
            condition_on_previous_text: config.condition_on_previous_text,
        }
    }

    fn current_prompt_tokens(&self, max_prompt_tokens: usize) -> Vec<u32> {
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

    fn record_generated_tokens(&mut self, tokens: &[u32], max_prompt_tokens: usize) {
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
struct WhisperDecodeDiagnostics {
    timestamp_tokens_requested: bool,
    timestamp_tokens_present: bool,
    decoded_token_ids: Vec<u32>,
    decoder_prompt_prefill_count: usize,
    decoder_cached_token_step_count: usize,
    decoder_input_token_count: usize,
    generated_token_count: usize,
    decoder_completed_row_count: usize,
    decoder_max_active_row_batch_size: usize,
    decoder_effective_active_batch_sizes: Vec<usize>,
    decoder_active_row_compaction_count: usize,
    decoder_self_attention_cache_reused: bool,
    decoder_cross_attention_cache_reused: bool,
    average_log_probability: f64,
    no_speech_probability: Option<f64>,
    compression_ratio: f64,
    attempted_temperatures: Vec<f64>,
    no_speech_rejected: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct WhisperTimestampSpec {
    begin_token_id: u32,
    end_token_id: u32,
    seconds_per_token: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct WhisperDecodedWindow {
    text: String,
    segments: Vec<WhisperDecodedSegment>,
}

#[derive(Debug, Clone, PartialEq)]
struct WhisperDecodedSegment {
    text: String,
    start_seconds: f64,
    end_seconds: f64,
    token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperDecoderInputKind {
    PromptPrefill,
    CachedTokenStep,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperDecoderInput {
    token_ids: Vec<u32>,
    position_offset: usize,
    flush_cache: bool,
    kind: WhisperDecoderInputKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperAutoregressiveRow {
    tokens: Vec<u32>,
    prompt_len: usize,
    cache_position: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct ActiveWhisperDecodeRow {
    original_index: usize,
    row: WhisperAutoregressiveRow,
    stats: WhisperGenerationStats,
    score: f64,
    no_speech_probability: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct WhisperTokenDecodeResult {
    token_ids: Vec<u32>,
    stats: WhisperGenerationStats,
    average_log_probability: f64,
    no_speech_probability: Option<f64>,
    attempted_temperatures: Vec<f64>,
    no_speech_rejected: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperInitialTokens {
    token_ids: Vec<u32>,
    sot_position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperFallbackAttemptDecision {
    Accept,
    Retry,
    RejectNoSpeech,
}

fn fallback_attempt_decision(
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

fn apply_no_speech_rejection(
    decision: WhisperFallbackAttemptDecision,
    token_ids: &mut Vec<u32>,
) -> bool {
    let rejected = decision == WhisperFallbackAttemptDecision::RejectNoSpeech;
    if rejected {
        token_ids.clear();
    }
    rejected
}

fn run_ordered_temperature_fallback<T>(
    temperatures: &[f64],
    mut attempt: impl FnMut(usize, f64) -> Result<T>,
    mut should_retry: impl FnMut(&T) -> bool,
) -> Result<(T, Vec<f64>)> {
    if temperatures.is_empty() {
        return Err(invalid_request(
            "Candle Whisper temperature_schedule must not be empty",
        ));
    }
    let mut attempted = Vec::new();
    for (index, temperature) in temperatures.iter().copied().enumerate() {
        let result = attempt(index, temperature)?;
        attempted.push(temperature);
        if !should_retry(&result) || index + 1 == temperatures.len() {
            return Ok((result, attempted));
        }
    }
    unreachable!("non-empty temperature schedule always returns its final attempt")
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

fn tensor_token_probability_at_position(
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

fn token_log_probability(logits: &[f32], token_id: u32) -> Option<f64> {
    token_probability(logits, token_id).map(f64::ln)
}

/// Whisper-compatible ratio: UTF-8 byte length divided by zlib-compressed size.
fn text_compression_ratio(text: &str) -> Result<f64> {
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

fn average_log_probability(score: f64, generated_len: usize, completed: bool) -> f64 {
    score / (generated_len + usize::from(completed)).max(1) as f64
}

impl WhisperAutoregressiveRow {
    fn new(prompt_tokens: Vec<u32>) -> Self {
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

    fn next_decoder_input(&self) -> WhisperDecoderInput {
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

    fn mark_forwarded(&mut self) {
        self.cache_position = self.tokens.len();
    }

    fn generated_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len..]
    }

    fn accept(&mut self, token: u32) {
        self.tokens.push(token);
    }

    fn into_generated_tokens(self) -> Vec<u32> {
        self.tokens.into_iter().skip(self.prompt_len).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct WhisperGenerationStats {
    prompt_prefill_count: usize,
    cached_token_step_count: usize,
    decoder_input_token_count: usize,
    generated_token_count: usize,
    completed_row_count: usize,
    max_active_row_batch_size: usize,
    effective_active_batch_sizes: Vec<usize>,
    active_row_compaction_count: usize,
    decoder_self_attention_cache_reused: bool,
    decoder_cross_attention_cache_reused: bool,
}

impl WhisperGenerationStats {
    fn record_input(&mut self, input: &WhisperDecoderInput) {
        match input.kind {
            WhisperDecoderInputKind::PromptPrefill => self.prompt_prefill_count += 1,
            WhisperDecoderInputKind::CachedTokenStep => self.cached_token_step_count += 1,
        }
        self.decoder_input_token_count += input.token_ids.len();
    }

    fn record_generated_token(&mut self) {
        self.generated_token_count += 1;
    }

    fn record_active_row_batch_size(&mut self, batch_size: usize) {
        self.max_active_row_batch_size = self.max_active_row_batch_size.max(batch_size);
        self.effective_active_batch_sizes.push(batch_size);
    }

    fn record_active_row_compaction(&mut self) {
        self.active_row_compaction_count += 1;
    }

    fn record_completed_row(&mut self) {
        self.completed_row_count = 1;
    }

    fn record_decoder_stats(&mut self, stats: CachedWhisperDecoderStats) {
        self.decoder_self_attention_cache_reused |= stats.self_attention_cache_reused;
        self.decoder_cross_attention_cache_reused |= stats.cross_attention_cache_reused;
    }

    fn extend(self, diagnostics: &mut WhisperDecodeDiagnostics) {
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
    fn add_generation_counts_from(&mut self, other: &Self) {
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

#[allow(dead_code)]
pub(crate) fn transcribe(
    options: &CandleWhisperOptions,
    request: AsrRequest,
) -> Result<AsrResponse> {
    transcribe_with_load_observer(
        options,
        &CandleWhisperTranscriptionRequestConfig::default(),
        request,
        |_| Ok(()),
    )
}

pub(crate) enum WhisperModelResolutionEvent {
    ResolutionStart,
    #[allow(dead_code)] // emitted only by the optional model-bundles resolver
    DownloadStart,
    #[allow(dead_code)] // emitted only by the optional model-bundles resolver
    DownloadEnd {
        duration_seconds: f64,
    },
    ResolutionEnd {
        source: &'static str,
    },
    LoadStart,
    LoadEnd {
        duration_seconds: f64,
    },
}

pub(crate) fn transcribe_with_load_observer(
    options: &CandleWhisperOptions,
    config: &CandleWhisperTranscriptionRequestConfig,
    request: AsrRequest,
    mut on_resolution: impl FnMut(WhisperModelResolutionEvent) -> Result<()>,
) -> Result<AsrResponse> {
    let setup = WhisperRunSetup::from_options_and_request_with_observer(
        options,
        &config.runtime,
        &request,
        &mut on_resolution,
    )?;
    on_resolution(WhisperModelResolutionEvent::LoadStart)?;
    let load_started = std::time::Instant::now();
    let mut session = CandleWhisperSession::load(setup)?;
    on_resolution(WhisperModelResolutionEvent::LoadEnd {
        duration_seconds: load_started.elapsed().as_secs_f64(),
    })?;
    let resolved_device = session.setup.resolved_device.clone();
    with_decoder_threads(config.runtime.decoder_threads, &resolved_device, || {
        session.transcribe_chunks(options, config, request)
    })
}

fn with_decoder_threads<T: Send>(
    decoder_threads: Option<usize>,
    resolved_device: &ResolvedNativeDevice,
    run: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    let Some(decoder_threads) = decoder_threads else {
        return run();
    };
    if decoder_threads == 0 {
        return Err(invalid_request(
            "Candle Whisper decoder_threads must be greater than zero",
        ));
    }
    if resolved_device.cuda_active() {
        return run();
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(decoder_threads)
        .build()
        .map_err(|error| {
            setup_error(format!(
                "failed to create a request-scoped Candle Whisper decoder pool with {decoder_threads} threads: {error}"
            ))
        })?;
    pool.install(run)
}

pub(crate) enum ReusableCandleWhisperSessionEvent {
    ResolutionStart,
    DownloadStart,
    DownloadEnd { duration_seconds: f64 },
    ResolutionEnd { source: &'static str },
    LoadStart,
    LoadEnd { duration_seconds: f64 },
    Reuse,
}

pub(crate) struct ReusableCandleWhisperSession {
    session: CandleWhisperSession,
}

impl ReusableCandleWhisperSession {
    pub(crate) fn transcribe(
        current: &mut Option<Self>,
        options: &CandleWhisperOptions,
        config: &CandleWhisperTranscriptionRequestConfig,
        request: AsrRequest,
        mut observe: impl FnMut(ReusableCandleWhisperSessionEvent) -> Result<()>,
    ) -> Result<AsrResponse> {
        let setup = WhisperRunSetup::from_options_and_request_with_observer(
            options,
            &config.runtime,
            &request,
            &mut |event| {
                observe(match event {
                    WhisperModelResolutionEvent::ResolutionStart => {
                        ReusableCandleWhisperSessionEvent::ResolutionStart
                    }
                    WhisperModelResolutionEvent::DownloadStart => {
                        ReusableCandleWhisperSessionEvent::DownloadStart
                    }
                    WhisperModelResolutionEvent::DownloadEnd { duration_seconds } => {
                        ReusableCandleWhisperSessionEvent::DownloadEnd { duration_seconds }
                    }
                    WhisperModelResolutionEvent::ResolutionEnd { source } => {
                        ReusableCandleWhisperSessionEvent::ResolutionEnd { source }
                    }
                    WhisperModelResolutionEvent::LoadEnd { .. } => {
                        unreachable!("setup resolution does not emit load completion")
                    }
                    WhisperModelResolutionEvent::LoadStart => {
                        unreachable!("setup resolution does not emit load start")
                    }
                })
            },
        )?;
        let session_reused = match current.as_ref() {
            Some(existing) if existing.session.setup == setup => true,
            Some(_) | None => {
                observe(ReusableCandleWhisperSessionEvent::LoadStart)?;
                let load_started = std::time::Instant::now();
                *current = Some(Self {
                    session: CandleWhisperSession::load(setup)?,
                });
                observe(ReusableCandleWhisperSessionEvent::LoadEnd {
                    duration_seconds: load_started.elapsed().as_secs_f64(),
                })?;
                false
            }
        };
        if session_reused {
            observe(ReusableCandleWhisperSessionEvent::Reuse)?;
        }
        let session = current
            .as_mut()
            .expect("reusable Candle Whisper session is loaded");
        let resolved_device = session.session.setup.resolved_device.clone();
        let mut response =
            with_decoder_threads(config.runtime.decoder_threads, &resolved_device, || {
                session.session.transcribe_chunks(options, config, request)
            })?;
        response.diagnostics.push(if session_reused {
            "asrModelSession=reused".to_string()
        } else {
            "asrModelSession=loaded".to_string()
        });
        Ok(response)
    }
}

impl WhisperRunSetup {
    #[allow(dead_code)]
    fn from_options_and_request(
        options: &CandleWhisperOptions,
        request: &AsrRequest,
    ) -> Result<Self> {
        Self::from_options_and_request_with_observer(
            options,
            &CandleWhisperRuntimeControls::default(),
            request,
            &mut |_| Ok(()),
        )
    }

    fn from_options_and_request_with_observer(
        options: &CandleWhisperOptions,
        controls: &CandleWhisperRuntimeControls,
        request: &AsrRequest,
        observe: &mut dyn FnMut(WhisperModelResolutionEvent) -> Result<()>,
    ) -> Result<Self> {
        validate_asr_request(request)?;
        if options.compute_type == CandleWhisperComputeType::Int8
            && options.device == crate::NativeDevicePreference::Cuda
        {
            return Err(setup_error(
                "native Candle Whisper compute type int8 is CPU-only; select device=cpu",
            ));
        }
        let resolved_device = resolve_native_device(options.device, controls.cuda_device_index)?;
        let resolved_compute_type = options
            .compute_type
            .resolve_for_device(resolved_device.cuda_active())?;
        let model = resolve_whisper_model_with_observer(
            options,
            &request.model_id,
            resolved_compute_type,
            observe,
        )?;
        let model_weight_dtype = candle_whisper_model_weight_dtype(resolved_compute_type);
        let model_format = if resolved_compute_type == CandleWhisperComputeType::Int8 {
            WhisperModelFormat::GgufQ8_0
        } else {
            WhisperModelFormat::Safetensors
        };
        Ok(Self {
            model_id: model.model_id,
            task: request.task,
            language: request
                .language
                .clone()
                .or_else(|| options.language.clone()),
            bundle: model.bundle,
            model_source: model.source,
            resolved_device,
            requested_compute_type: options.compute_type,
            resolved_compute_type,
            model_weight_dtype,
            model_format,
        })
    }
}

fn candle_whisper_model_weight_dtype(compute_type: CandleWhisperComputeType) -> DType {
    match compute_type {
        CandleWhisperComputeType::Automatic => unreachable!("compute type must be resolved first"),
        CandleWhisperComputeType::Fp16 => DType::F16,
        CandleWhisperComputeType::Fp32 => DType::F32,
        CandleWhisperComputeType::Int8 => DType::F32,
    }
}

fn candle_dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::F32 => "f32",
        _ => "other",
    }
}

#[allow(dead_code)]
fn resolve_whisper_model(
    options: &CandleWhisperOptions,
    requested_model_id: &str,
) -> Result<ResolvedWhisperModel> {
    let resolved_compute_type = options.compute_type.resolve_for_device(false)?;
    resolve_whisper_model_with_observer(
        options,
        requested_model_id,
        resolved_compute_type,
        &mut |_| Ok(()),
    )
}

fn resolve_whisper_model_with_observer(
    options: &CandleWhisperOptions,
    requested_model_id: &str,
    resolved_compute_type: CandleWhisperComputeType,
    observe: &mut dyn FnMut(WhisperModelResolutionEvent) -> Result<()>,
) -> Result<ResolvedWhisperModel> {
    observe(WhisperModelResolutionEvent::ResolutionStart)?;
    let model_id = canonical_whisper_model_id(requested_model_id)?;
    if let Some(bundle) = &options.model_bundle {
        let bundle = if resolved_compute_type == CandleWhisperComputeType::Int8 {
            resolve_q8_whisper_bundle_paths(bundle)?
        } else {
            resolve_whisper_bundle_paths(bundle)?
        };
        observe(WhisperModelResolutionEvent::ResolutionEnd {
            source: "explicit-bundle",
        })?;
        return Ok(ResolvedWhisperModel {
            model_id,
            bundle,
            source: "explicit-bundle",
        });
    }
    if resolved_compute_type == CandleWhisperComputeType::Int8 {
        return Err(setup_error(format!(
            "native Candle Whisper compute type int8 requires an explicit local Q8_0 bundle containing {}; automatic downloads and safetensors fallback are disabled",
            resolved_compute_type.required_bundle_files().join(", ")
        )));
    }

    #[cfg(feature = "model-bundles")]
    {
        if options.model_cache_only {
            let bundle = resolve_cached_whisper_model(&model_id, options.model_dir.as_deref())
                .ok_or_else(|| missing_whisper_model_error(&model_id, options))?;
            observe(WhisperModelResolutionEvent::ResolutionEnd {
                source: "hugging-face-cache",
            })?;
            return Ok(ResolvedWhisperModel {
                model_id,
                bundle,
                source: "hugging-face-cache",
            });
        }

        let mut downloader = model_runtime::HuggingFaceDownloader::new().progress(false);
        if let Some(model_dir) = &options.model_dir {
            downloader = downloader.cache_dir(model_dir.clone());
        }
        observe(WhisperModelResolutionEvent::DownloadStart)?;
        let download_started = std::time::Instant::now();
        let downloaded = downloader
            .download(&whisper_model_spec(&model_id))
            .map_err(|error| missing_whisper_model_error_with_source(&model_id, options, error))?;
        observe(WhisperModelResolutionEvent::DownloadEnd {
            duration_seconds: download_started.elapsed().as_secs_f64(),
        })?;
        let bundle = downloaded
            .model_dir()
            .ok_or_else(|| {
                setup_error(format!(
                    "native Candle Whisper model `{model_id}` resolved without a local model directory"
                ))
            })
            .and_then(resolve_whisper_bundle_paths)?;
        observe(WhisperModelResolutionEvent::ResolutionEnd {
            source: "hugging-face-cache",
        })?;
        Ok(ResolvedWhisperModel {
            model_id,
            bundle,
            source: "hugging-face-cache",
        })
    }

    #[cfg(not(feature = "model-bundles"))]
    {
        Err(setup_error(format!(
            "native Candle Whisper model `{model_id}` requires --whisper-bundle or the model-bundles feature for Hugging Face resolution"
        )))
    }
}

fn canonical_whisper_model_id(value: &str) -> Result<String> {
    match value {
        "tiny" => Ok("openai/whisper-tiny".to_string()),
        "tiny.en" => Ok("openai/whisper-tiny.en".to_string()),
        "base" => Ok("openai/whisper-base".to_string()),
        "base.en" => Ok("openai/whisper-base.en".to_string()),
        "small" => Ok("openai/whisper-small".to_string()),
        "small.en" => Ok("openai/whisper-small.en".to_string()),
        "medium" => Ok("openai/whisper-medium".to_string()),
        "medium.en" => Ok("openai/whisper-medium.en".to_string()),
        "large" => Ok("openai/whisper-large-v3".to_string()),
        "large-v1" => Ok("openai/whisper-large-v1".to_string()),
        "large-v2" => Ok("openai/whisper-large-v2".to_string()),
        "large-v3" => Ok("openai/whisper-large-v3".to_string()),
        "large-v3-turbo" => Ok("openai/whisper-large-v3-turbo".to_string()),
        other if looks_like_hf_repo_id(other) => Ok(other.to_string()),
        other => Err(setup_error(format!(
            "unsupported native Candle Whisper model alias `{other}`; native Candle Whisper requires a supported Whisper alias, a Hugging Face repo ID with Candle-compatible files, or --whisper-bundle"
        ))),
    }
}

fn looks_like_hf_repo_id(value: &str) -> bool {
    let mut parts = value.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(owner), Some(repo), None) if !owner.is_empty() && !repo.is_empty())
}

#[cfg(feature = "model-bundles")]
fn resolve_cached_whisper_model(
    model_id: &str,
    model_dir: Option<&Path>,
) -> Option<WhisperBundlePaths> {
    let mut roots = Vec::new();
    if let Some(model_dir) = model_dir {
        roots.push(model_dir.to_path_buf());
    } else if let Some(home) = std::env::var_os("HF_HOME") {
        roots.push(PathBuf::from(home).join("hub"));
    } else if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    for root in roots {
        for candidate in whisper_cache_candidates(&root, model_id) {
            if let Ok(paths) = resolve_whisper_bundle_paths(&candidate) {
                return Some(paths);
            }
        }
    }
    None
}

#[cfg(feature = "model-bundles")]
fn whisper_cache_candidates(root: &Path, model_id: &str) -> Vec<PathBuf> {
    let mut candidates = vec![root.to_path_buf(), root.join(model_id.replace('/', "--"))];
    let hf_repo_dir = root.join(format!("models--{}", model_id.replace('/', "--")));
    if let Ok(snapshot) = std::fs::read_to_string(hf_repo_dir.join("refs/main")) {
        candidates.push(hf_repo_dir.join("snapshots").join(snapshot.trim()));
    }
    if let Ok(entries) = std::fs::read_dir(hf_repo_dir.join("snapshots")) {
        for entry in entries.flatten() {
            candidates.push(entry.path());
        }
    }
    candidates
}

#[cfg(feature = "model-bundles")]
fn whisper_model_spec(model_id: &str) -> model_runtime::HuggingFaceModelSpec {
    let mut spec = model_runtime::HuggingFaceModelSpec::new(
        model_id.to_string(),
        model_runtime::ModelTask::SpeechRecognition,
    );
    spec.files = CandleWhisperComputeType::Automatic
        .required_bundle_files()
        .iter()
        .copied()
        .map(model_runtime::ModelFileRequest::required)
        .collect();
    spec
}

#[cfg(feature = "model-bundles")]
fn missing_whisper_model_error(
    model_id: &str,
    options: &CandleWhisperOptions,
) -> media_core::DetectError {
    setup_error(format!(
        "failed to resolve native Candle Whisper model `{model_id}`; required files: {}; --model-dir={}; cache-only={}",
        CandleWhisperComputeType::Automatic
            .required_bundle_files()
            .join(", "),
        options
            .model_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<default huggingface cache>".to_string()),
        options.model_cache_only
    ))
}

#[cfg(feature = "model-bundles")]
fn missing_whisper_model_error_with_source(
    model_id: &str,
    options: &CandleWhisperOptions,
    source: impl std::fmt::Display,
) -> media_core::DetectError {
    setup_error(format!(
        "failed to resolve native Candle Whisper model `{model_id}`; required files: {}; --model-dir={}; cache-only={}: {source}",
        CandleWhisperComputeType::Automatic
            .required_bundle_files()
            .join(", "),
        options
            .model_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<default huggingface cache>".to_string()),
        options.model_cache_only
    ))
}

fn whisper_setup_diagnostics(setup: &WhisperRunSetup) -> Vec<String> {
    vec![
        format!("asrModelResolved={}", setup.bundle.root.display()),
        format!("asrModelSource={}", setup.model_source),
        format!("asrModelId={}", setup.model_id),
        format!(
            "requestedComputeType={}",
            setup.requested_compute_type.as_str()
        ),
        format!(
            "resolvedComputeType={}",
            setup.resolved_compute_type.as_str()
        ),
        format!(
            "modelWeightDtype={}",
            candle_dtype_name(setup.model_weight_dtype)
        ),
        format!("computeType={}", setup.resolved_compute_type.as_str()),
        format!("modelFormat={}", setup.model_format.as_str()),
    ]
}

pub(crate) fn resolve_whisper_bundle_paths(bundle: &Path) -> Result<WhisperBundlePaths> {
    if !bundle.exists() {
        return Err(setup_error(format!(
            "required Candle Whisper model bundle `{}` is missing",
            bundle.display()
        )));
    }
    Ok(WhisperBundlePaths {
        root: bundle.to_path_buf(),
        config_json: crate::native_bundles::resolve_required_bundle_file(bundle, "config.json")?,
        generation_config_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "generation_config.json",
        )?,
        tokenizer_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "tokenizer.json",
        )?,
        preprocessor_config_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "preprocessor_config.json",
        )?,
        model_safetensors: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "model.safetensors",
        )?,
        model_q8_0_gguf: None,
    })
}

fn resolve_q8_whisper_bundle_paths(bundle: &Path) -> Result<WhisperBundlePaths> {
    if !bundle.exists() {
        return Err(setup_error(format!(
            "required Candle Whisper Q8_0 model bundle `{}` is missing; required files: {}",
            bundle.display(),
            CandleWhisperComputeType::Int8
                .required_bundle_files()
                .join(", ")
        )));
    }
    crate::native_bundles::validate_required_bundle_files(
        bundle,
        CandleWhisperComputeType::Int8.required_bundle_files(),
    )?;
    let paths = WhisperBundlePaths {
        root: bundle.to_path_buf(),
        config_json: crate::native_bundles::resolve_required_bundle_file(bundle, "config.json")?,
        generation_config_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "generation_config.json",
        )?,
        tokenizer_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "tokenizer.json",
        )?,
        preprocessor_config_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "preprocessor_config.json",
        )?,
        // Int8 deliberately leaves this path unresolved so model.safetensors can
        // never satisfy or mask the Q8 bundle contract.
        model_safetensors: bundle.join("model.safetensors"),
        model_q8_0_gguf: Some(crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "model.q8_0.gguf",
        )?),
    };
    validate_q8_whisper_bundle(&paths)?;
    Ok(paths)
}

fn validate_q8_whisper_bundle(paths: &WhisperBundlePaths) -> Result<()> {
    let config: whisper::Config = read_json(&paths.config_json, "config.json")?;
    let generation: GenerationConfig =
        read_json(&paths.generation_config_json, "generation_config.json")?;
    let _: serde_json::Value =
        read_json(&paths.preprocessor_config_json, "preprocessor_config.json")?;
    let tokenizer = Tokenizer::from_file(&paths.tokenizer_json).map_err(|error| {
        setup_error(format!(
            "failed to load Q8 Whisper tokenizer `{}`: {error}",
            paths.tokenizer_json.display()
        ))
    })?;
    validate_whisper_companion_metadata(&config, &generation, &tokenizer)?;

    let gguf_path = paths
        .model_q8_0_gguf
        .as_deref()
        .expect("Q8 bundle always has GGUF weights");
    let mut file = File::open(gguf_path).map_err(|error| {
        setup_error(format!(
            "failed to open Q8 Whisper GGUF `{}`: {error}",
            gguf_path.display()
        ))
    })?;
    let content = gguf_file::Content::read(&mut file).map_err(|error| {
        setup_error(format!(
            "invalid Q8 Whisper GGUF `{}`: {error}",
            gguf_path.display()
        ))
    })?;
    let architecture = content
        .metadata
        .get("general.architecture")
        .and_then(|value| match value {
            gguf_file::Value::String(value) => Some(value.as_str()),
            _ => None,
        })
        .ok_or_else(|| {
            setup_error("Q8 Whisper GGUF is missing string metadata `general.architecture`")
        })?;
    if architecture != "whisper" {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF metadata `general.architecture` must be `whisper`, got `{architecture}`"
        )));
    }
    let file_type = content
        .metadata
        .get("general.file_type")
        .and_then(|value| value.to_u64().ok())
        .ok_or_else(|| {
            setup_error("Q8 Whisper GGUF is missing integer metadata `general.file_type`")
        })?;
    if file_type != 7 {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF metadata `general.file_type` must identify Q8_0 (7), got {file_type}"
        )));
    }
    validate_q8_tensor(
        &content,
        "model.decoder.embed_tokens.weight",
        &[config.vocab_size, config.d_model],
    )?;
    validate_tensor_shape(
        &content,
        "model.encoder.conv1.weight",
        &[config.d_model, config.num_mel_bins, 3],
    )?;
    validate_tensor_shape(
        &content,
        "model.encoder.conv2.weight",
        &[config.d_model, config.d_model, 3],
    )?;
    for (name, info) in &content.tensor_infos {
        if q8_required_tensor_name(name) && info.ggml_dtype != GgmlDType::Q8_0 {
            return Err(setup_error(format!(
                "Q8 Whisper GGUF tensor `{name}` must use Q8_0, got {:?}",
                info.ggml_dtype
            )));
        }
    }
    Ok(())
}

fn validate_whisper_companion_metadata(
    config: &whisper::Config,
    generation: &GenerationConfig,
    tokenizer: &Tokenizer,
) -> Result<()> {
    if config.d_model == 0
        || config.encoder_layers == 0
        || config.decoder_layers == 0
        || config.encoder_attention_heads == 0
        || config.decoder_attention_heads == 0
        || !config
            .d_model
            .is_multiple_of(config.encoder_attention_heads)
        || !config
            .d_model
            .is_multiple_of(config.decoder_attention_heads)
    {
        return Err(setup_error(
            "Q8 Whisper config contains incompatible model or attention dimensions",
        ));
    }
    let tokenizer_vocab = tokenizer.get_vocab_size(true);
    if tokenizer_vocab != config.vocab_size {
        return Err(setup_error(format!(
            "Q8 Whisper tokenizer vocabulary size {tokenizer_vocab} does not match config vocab_size {}",
            config.vocab_size
        )));
    }
    for (name, token) in [
        ("decoder_start_token_id", generation.decoder_start_token_id),
        ("eos_token_id", generation.eos_token_id),
        ("no_timestamps_token_id", generation.no_timestamps_token_id),
    ] {
        if token.is_some_and(|token| token as usize >= config.vocab_size) {
            return Err(setup_error(format!(
                "Q8 Whisper generation metadata `{name}` is outside the tokenizer vocabulary"
            )));
        }
    }
    Ok(())
}

fn validate_q8_tensor(
    content: &gguf_file::Content,
    name: &str,
    expected_dims: &[usize],
) -> Result<()> {
    let info = content.tensor_infos.get(name).ok_or_else(|| {
        setup_error(format!(
            "Q8 Whisper GGUF is missing required tensor `{name}`"
        ))
    })?;
    if info.shape.dims() != expected_dims {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF tensor `{name}` has shape {:?}, expected {expected_dims:?}",
            info.shape.dims()
        )));
    }
    if info.ggml_dtype != GgmlDType::Q8_0 {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF tensor `{name}` must use Q8_0, got {:?}",
            info.ggml_dtype
        )));
    }
    Ok(())
}

fn validate_tensor_shape(
    content: &gguf_file::Content,
    name: &str,
    expected_dims: &[usize],
) -> Result<()> {
    let info = content.tensor_infos.get(name).ok_or_else(|| {
        setup_error(format!(
            "Q8 Whisper GGUF is missing required tensor `{name}`"
        ))
    })?;
    if info.shape.dims() != expected_dims {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF tensor `{name}` has shape {:?}, expected {expected_dims:?}",
            info.shape.dims()
        )));
    }
    Ok(())
}

fn q8_required_tensor_name(name: &str) -> bool {
    name.ends_with(".weight")
        && !name.contains(".conv")
        && !name.ends_with("embed_positions.weight")
        && !name.contains("layer_norm.weight")
}

#[derive(Debug, Clone)]
struct CachedWhisperAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl CachedWhisperAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            query: linear(n_state, n_state, vb.pp("q_proj"))?,
            key: linear_no_bias(n_state, n_state, vb.pp("k_proj"))?,
            value: linear(n_state, n_state, vb.pp("v_proj"))?,
            out: linear(n_state, n_state, vb.pp("out_proj"))?,
            n_head,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_cache: bool,
    ) -> candle_core::Result<(Tensor, bool)> {
        let q = self.query.forward(x)?;
        let (k, v, cache_reused) = match xa {
            None => {
                if flush_cache {
                    self.kv_cache = None;
                }
                let current_k = self.key.forward(x)?;
                let current_v = self.value.forward(x)?;
                if let Some((cached_k, cached_v)) = &self.kv_cache {
                    let k = Tensor::cat(&[cached_k, &current_k], 1)?;
                    let v = Tensor::cat(&[cached_v, &current_v], 1)?;
                    self.kv_cache = Some((k.clone(), v.clone()));
                    (k, v, true)
                } else {
                    self.kv_cache = Some((current_k.clone(), current_v.clone()));
                    (current_k, current_v, false)
                }
            }
            Some(x) => {
                if flush_cache {
                    self.kv_cache = None;
                }
                if let Some((k, v)) = &self.kv_cache {
                    (k.clone(), v.clone(), true)
                } else {
                    let k = self.key.forward(x)?;
                    let v = self.value.forward(x)?;
                    self.kv_cache = Some((k.clone(), v.clone()));
                    (k, v, false)
                }
            }
        };
        let wv = self.qkv_attention(&q, &k, &v, mask)?;
        Ok((self.out.forward(&wv)?, cache_reused))
    }

    fn reshape_head(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        x.reshape(target_dims)?.transpose(1, 2)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (_, _, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            qk = qk.broadcast_add(mask)?;
        }
        let w = candle_nn::ops::softmax_last_dim(&qk)?;
        w.matmul(&v)?.transpose(1, 2)?.flatten_from(2)
    }

    fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    fn select_kv_cache_rows(&mut self, row_indices: &Tensor) -> candle_core::Result<()> {
        if let Some((cached_k, cached_v)) = &self.kv_cache {
            self.kv_cache = Some((
                cached_k.index_select(row_indices, 0)?,
                cached_v.index_select(row_indices, 0)?,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct CachedWhisperBlockStats {
    self_cache_reused: bool,
    cross_cache_reused: bool,
}

#[derive(Debug, Clone)]
struct CachedWhisperBlock {
    attn: CachedWhisperAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(CachedWhisperAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl CachedWhisperBlock {
    fn load(
        n_state: usize,
        n_head: usize,
        cross_attention: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cross_attn = if cross_attention {
            Some((
                CachedWhisperAttention::load(n_state, n_head, vb.pp("encoder_attn"))?,
                layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            attn: CachedWhisperAttention::load(n_state, n_head, vb.pp("self_attn"))?,
            attn_ln: layer_norm(n_state, vb.pp("self_attn_layer_norm"))?,
            cross_attn,
            mlp_linear1: linear(n_state, n_state * 4, vb.pp("fc1"))?,
            mlp_linear2: linear(n_state * 4, n_state, vb.pp("fc2"))?,
            mlp_ln: layer_norm(n_state, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_kv_cache: bool,
    ) -> candle_core::Result<(Tensor, CachedWhisperBlockStats)> {
        let (attn, self_cache_reused) =
            self.attn
                .forward(&self.attn_ln.forward(x)?, None, mask, flush_kv_cache)?;
        let mut x = (x + attn)?;
        let mut stats = CachedWhisperBlockStats {
            self_cache_reused,
            cross_cache_reused: false,
        };
        if let Some((attn, ln)) = &mut self.cross_attn {
            let (cross, cross_cache_reused) =
                attn.forward(&ln.forward(&x)?, xa, None, flush_kv_cache)?;
            x = (&x + cross)?;
            stats.cross_cache_reused = cross_cache_reused;
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        Ok(((x + mlp)?, stats))
    }

    fn reset_kv_cache(&mut self) {
        self.attn.reset_kv_cache();
        if let Some((attn, _)) = &mut self.cross_attn {
            attn.reset_kv_cache();
        }
    }

    fn select_kv_cache_rows(&mut self, row_indices: &Tensor) -> candle_core::Result<()> {
        self.attn.select_kv_cache_rows(row_indices)?;
        if let Some((attn, _)) = &mut self.cross_attn {
            attn.select_kv_cache_rows(row_indices)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct CachedWhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<CachedWhisperBlock>,
    ln_post: LayerNorm,
}

impl CachedWhisperEncoder {
    fn load(vb: VarBuilder, cfg: &whisper::Config) -> candle_core::Result<Self> {
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let conv1 = conv1d(cfg.num_mel_bins, n_state, 3, cfg1, vb.pp("conv1"))?;
        let conv2 = conv1d(n_state, n_state, 3, cfg2, vb.pp("conv2"))?;
        let positional_embedding = sinusoids(cfg.max_source_positions, n_state, vb.device())?;
        let blocks = (0..cfg.encoder_layers)
            .map(|index| {
                CachedWhisperBlock::load(n_state, n_head, false, vb.pp(format!("layers.{index}")))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post: layer_norm(n_state, vb.pp("layer_norm"))?,
        })
    }

    fn forward(&mut self, x: &Tensor, flush_kv_cache: bool) -> candle_core::Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = x.transpose(1, 2)?;
        let (_, seq_len, _) = x.dims3()?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter_mut() {
            x = block.forward(&x, None, None, flush_kv_cache)?.0;
        }
        self.ln_post.forward(&x)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct CachedWhisperDecoderStats {
    self_attention_cache_reused: bool,
    cross_attention_cache_reused: bool,
}

impl CachedWhisperDecoderStats {
    fn merge_block(&mut self, block: CachedWhisperBlockStats) {
        self.self_attention_cache_reused |= block.self_cache_reused;
        self.cross_attention_cache_reused |= block.cross_cache_reused;
    }
}

#[derive(Debug, Clone)]
struct CachedWhisperDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<CachedWhisperBlock>,
    ln: LayerNorm,
}

impl CachedWhisperDecoder {
    fn load(vb: VarBuilder, cfg: &whisper::Config) -> candle_core::Result<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let token_embedding = embedding(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get(
            (cfg.max_target_positions, n_state),
            "embed_positions.weight",
        )?;
        let blocks = (0..cfg.decoder_layers)
            .map(|index| {
                CachedWhisperBlock::load(n_state, n_head, true, vb.pp(format!("layers.{index}")))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln: layer_norm(n_state, vb.pp("layer_norm"))?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        position_offset: usize,
        flush_kv_cache: bool,
    ) -> candle_core::Result<(Tensor, CachedWhisperDecoderStats)> {
        let token_count = x.dim(D::Minus1)?;
        let token_embedding = self.token_embedding.forward(x)?;
        let positional_embedding =
            self.positional_embedding
                .narrow(0, position_offset, token_count)?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        let mask = decoder_causal_mask(
            token_count,
            position_offset + token_count,
            position_offset,
            x.device(),
        )?;
        let mut stats = CachedWhisperDecoderStats::default();
        for block in self.blocks.iter_mut() {
            let (next, block_stats) = block.forward(&x, Some(xa), Some(&mask), flush_kv_cache)?;
            stats.merge_block(block_stats);
            x = next;
        }
        Ok((self.ln.forward(&x)?, stats))
    }

    fn final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let batch_size = x.dim(0)?;
        let weight = self
            .token_embedding
            .embeddings()
            .broadcast_left(batch_size)?;
        x.matmul(&weight.t()?)
    }

    fn reset_kv_cache(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_kv_cache();
        }
    }

    fn select_kv_cache_rows(&mut self, row_indices: &Tensor) -> candle_core::Result<()> {
        for block in self.blocks.iter_mut() {
            block.select_kv_cache_rows(row_indices)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct CachedWhisper {
    encoder: CachedWhisperEncoder,
    decoder: CachedWhisperDecoder,
    config: whisper::Config,
}

impl CachedWhisper {
    fn load(vb: &VarBuilder, config: whisper::Config) -> candle_core::Result<Self> {
        Ok(Self {
            encoder: CachedWhisperEncoder::load(vb.pp("model.encoder"), &config)?,
            decoder: CachedWhisperDecoder::load(vb.pp("model.decoder"), &config)?,
            config,
        })
    }

    fn reset_kv_cache(&mut self) {
        for block in self.encoder.blocks.iter_mut() {
            block.reset_kv_cache();
        }
        self.decoder.reset_kv_cache();
    }
}

#[derive(Debug, Clone)]
enum WhisperModel {
    Safetensors(CachedWhisper),
    Q8(CandleQ8WhisperModel),
}

impl WhisperModel {
    fn config(&self) -> &whisper::Config {
        match self {
            Self::Safetensors(model) => &model.config,
            Self::Q8(model) => &model.config,
        }
    }

    fn encode(&mut self, mel: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Safetensors(model) => model.encoder.forward(mel, true),
            Self::Q8(model) => model.encode(mel),
        }
    }

    fn decode(
        &mut self,
        tokens: &Tensor,
        encoder_features: &Tensor,
        position_offset: usize,
        reset_cache: bool,
    ) -> candle_core::Result<(Tensor, CachedWhisperDecoderStats)> {
        match self {
            Self::Safetensors(model) => {
                model
                    .decoder
                    .forward(tokens, encoder_features, position_offset, reset_cache)
            }
            Self::Q8(model) => {
                let output =
                    model.decode(tokens, encoder_features, position_offset, reset_cache)?;
                Ok((
                    output.activations,
                    CachedWhisperDecoderStats {
                        self_attention_cache_reused: output.diagnostics.self_attention_cache_reused,
                        cross_attention_cache_reused: output
                            .diagnostics
                            .cross_attention_cache_reused,
                    },
                ))
            }
        }
    }

    fn project_logits(&self, activations: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Safetensors(model) => model.decoder.final_linear(activations),
            Self::Q8(model) => model.project_logits(activations),
        }
    }

    fn reset_cache(&mut self) {
        match self {
            Self::Safetensors(model) => model.reset_kv_cache(),
            Self::Q8(model) => model.reset_cache(),
        }
    }

    fn select_cache_rows(&mut self, row_indices: &Tensor) -> candle_core::Result<()> {
        match self {
            Self::Safetensors(model) => model.decoder.select_kv_cache_rows(row_indices),
            Self::Q8(model) => model.select_cache_rows(row_indices),
        }
    }
}

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> candle_core::Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn layer_norm(size: usize, vb: VarBuilder) -> candle_core::Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

fn sinusoids(length: usize, channels: usize, device: &Device) -> candle_core::Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
    let shape = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(shape)? * inv_timescales.broadcast_as(shape)?)?;
    Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)
}

fn decoder_causal_mask(
    query_len: usize,
    key_len: usize,
    position_offset: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let values = (0..query_len)
        .flat_map(|query_index| {
            let absolute_query = position_offset + query_index;
            (0..key_len).map(move |key_index| {
                if key_index > absolute_query {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (query_len, key_len), device)
}

struct CandleWhisperSession {
    setup: WhisperRunSetup,
    device: Device,
    model: WhisperModel,
    tokenizer: Tokenizer,
    generation: GenerationConfig,
    mel_filters: Vec<f32>,
    transformers_mel_filters: Vec<f32>,
    encoder_duration_seconds: f64,
    decoder_duration_seconds: f64,
}

#[derive(Clone, Copy)]
enum WhisperFeatureExtractorMode {
    Legacy,
    Transformers,
}

impl CandleWhisperSession {
    fn load(setup: WhisperRunSetup) -> Result<Self> {
        let device = candle_device(&setup.resolved_device)?;
        let config: whisper::Config = read_json(&setup.bundle.config_json, "config.json")?;
        let generation: GenerationConfig = read_json(
            &setup.bundle.generation_config_json,
            "generation_config.json",
        )?;
        let _preprocessor: serde_json::Value = read_json(
            &setup.bundle.preprocessor_config_json,
            "preprocessor_config.json",
        )?;
        let tokenizer = Tokenizer::from_file(&setup.bundle.tokenizer_json).map_err(|error| {
            invalid_request(format!(
                "failed to load tokenizer `{}`: {error}",
                setup.bundle.tokenizer_json.display()
            ))
        })?;
        let model = match setup.model_format {
            WhisperModelFormat::Safetensors => {
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[setup.bundle.model_safetensors.as_path()],
                        setup.model_weight_dtype,
                        &device,
                    )
                }
                .map_err(|error| {
                    setup_error(format!(
                        "failed to load Candle Whisper weights `{}`: {error}",
                        setup.bundle.model_safetensors.display()
                    ))
                })?;
                WhisperModel::Safetensors(CachedWhisper::load(&vb, config.clone()).map_err(
                    |error| {
                        setup_error(format!(
                            "failed to construct Candle Whisper model from `{}`: {error}",
                            setup.bundle.root.display()
                        ))
                    },
                )?)
            }
            WhisperModelFormat::GgufQ8_0 => {
                let path = setup
                    .bundle
                    .model_q8_0_gguf
                    .as_deref()
                    .expect("validated Q8 setup includes GGUF weights");
                WhisperModel::Q8(
                    CandleQ8WhisperModel::from_gguf(path, config.clone(), &device).map_err(
                        |error| {
                            setup_error(format!(
                                "failed to construct Q8_0 Candle Whisper model from `{}`: {error}",
                                path.display()
                            ))
                        },
                    )?,
                )
            }
        };
        let mel_filters =
            mel_filter_bank(config.num_mel_bins, whisper::N_FFT, whisper::SAMPLE_RATE);
        let transformers_mel_filters =
            transformers_mel_filter_bank(config.num_mel_bins, whisper::N_FFT, whisper::SAMPLE_RATE);
        Ok(Self {
            setup,
            device,
            model,
            tokenizer,
            generation,
            mel_filters,
            transformers_mel_filters,
            encoder_duration_seconds: 0.0,
            decoder_duration_seconds: 0.0,
        })
    }

    fn transcribe_chunks(
        &mut self,
        options: &CandleWhisperOptions,
        config: &CandleWhisperTranscriptionRequestConfig,
        request: AsrRequest,
    ) -> Result<AsrResponse> {
        let controls = &config.runtime;
        let decode = &config.decode;
        let window_controls = &config.window;
        debug_assert!(
            decode.preserves_legacy_greedy_path() || !decode.search.temperature_schedule.is_empty()
        );
        let mut segments = Vec::new();
        let mut next_index = 0_u64;
        let mut used_timestamp_tokens = false;
        let mut used_timestamp_word_projection = false;
        let mut timestamp_tokens_requested = false;
        let mut timestamp_tokens_present = false;
        let mut rejected_timestamp_segments = false;
        let mut timing_fallbacks = Vec::new();
        let mut decoder_prompt_prefill_count = 0_usize;
        let mut decoder_cached_token_step_count = 0_usize;
        let mut decoder_input_token_count = 0_usize;
        let mut generated_token_count = 0_usize;
        let mut decoder_completed_row_count = 0_usize;
        let mut decoder_max_active_row_batch_size = 0_usize;
        let mut decoder_effective_active_batch_sizes = Vec::new();
        let mut decoder_active_row_compaction_count = 0_usize;
        let mut decoder_self_attention_cache_reused = false;
        let mut decoder_cross_attention_cache_reused = false;
        let mut average_log_probabilities = Vec::new();
        let mut no_speech_probabilities = Vec::new();
        let mut compression_ratios = Vec::new();
        let mut attempted_temperatures = Vec::new();
        let mut no_speech_rejected = false;
        let mut prompt_state = WhisperRequestPromptState::new(decode);
        self.encoder_duration_seconds = 0.0;
        self.decoder_duration_seconds = 0.0;
        let request_started = std::time::Instant::now();
        let max_prompt_tokens = (self.model.config().max_target_positions / 2).saturating_sub(1);
        let batch_size = candle_batch_size(options, request.chunks.len());
        let feature_extractor_mode =
            if window_controls.timing_mode == CandleWhisperTimingMode::NoTimestamps {
                WhisperFeatureExtractorMode::Transformers
            } else {
                WhisperFeatureExtractorMode::Legacy
            };
        for batch in request.chunks.chunks(batch_size) {
            let windows = collect_chunk_windows(
                &request.audio.samples,
                request.audio.sample_rate,
                batch,
                window_controls,
            )?;
            let timed_windows = match options.decode_runtime {
                CandleWhisperDecodeRuntime::AutoregressiveKvCache => {
                    let mut decoded = Vec::with_capacity(windows.len());
                    for window in &windows {
                        let mut window_decode = decode.clone();
                        window_decode.initial_prompt_tokens =
                            prompt_state.current_prompt_tokens(max_prompt_tokens);
                        let timed = self.decode_window_with_timing_mode(
                            &window.samples,
                            window_controls.timing_mode,
                            feature_extractor_mode,
                            &window_decode,
                        )?;
                        prompt_state.record_generated_tokens(
                            &timed.conditioning_token_ids,
                            max_prompt_tokens,
                        );
                        decoded.push(timed);
                    }
                    decoded
                }
                CandleWhisperDecodeRuntime::ActiveRowTensorBatch => self
                    .decode_windows_with_timing_mode(
                        &windows,
                        window_controls.timing_mode,
                        feature_extractor_mode,
                        decode,
                    )?,
            };
            for (window, timed) in windows.iter().zip(timed_windows) {
                debug_assert!(window.chunk_start_seconds <= window.global_start_seconds);
                debug_assert!(window.local_start_seconds <= window.local_end_seconds);
                timestamp_tokens_requested |= timed.diagnostics.timestamp_tokens_requested;
                timestamp_tokens_present |= timed.diagnostics.timestamp_tokens_present;
                decoder_prompt_prefill_count += timed.diagnostics.decoder_prompt_prefill_count;
                decoder_cached_token_step_count +=
                    timed.diagnostics.decoder_cached_token_step_count;
                decoder_input_token_count += timed.diagnostics.decoder_input_token_count;
                generated_token_count += timed.diagnostics.generated_token_count;
                decoder_completed_row_count += timed.diagnostics.decoder_completed_row_count;
                decoder_max_active_row_batch_size = decoder_max_active_row_batch_size
                    .max(timed.diagnostics.decoder_max_active_row_batch_size);
                decoder_effective_active_batch_sizes.extend(
                    timed
                        .diagnostics
                        .decoder_effective_active_batch_sizes
                        .iter()
                        .copied(),
                );
                decoder_active_row_compaction_count +=
                    timed.diagnostics.decoder_active_row_compaction_count;
                decoder_self_attention_cache_reused |=
                    timed.diagnostics.decoder_self_attention_cache_reused;
                decoder_cross_attention_cache_reused |=
                    timed.diagnostics.decoder_cross_attention_cache_reused;
                average_log_probabilities.push(timed.diagnostics.average_log_probability);
                if let Some(probability) = timed.diagnostics.no_speech_probability {
                    no_speech_probabilities.push(probability);
                }
                compression_ratios.push(timed.diagnostics.compression_ratio);
                attempted_temperatures
                    .extend(timed.diagnostics.attempted_temperatures.iter().copied());
                no_speech_rejected |= timed.diagnostics.no_speech_rejected;
                if let Some(reason) = timed.fallback_reason {
                    if !timing_fallbacks.contains(&reason) {
                        timing_fallbacks.push(reason);
                    }
                    rejected_timestamp_segments |= reason == "unstableTimestampSegments";
                }
                if timed.timing == WhisperWindowTiming::ChunkWindow {
                    if timed.decoded.text.trim().is_empty() {
                        continue;
                    }
                    segments.push(window_fallback_segment(
                        next_index,
                        timed.decoded.text,
                        window.global_start_seconds,
                        window.global_end_seconds,
                        self.setup.language.clone(),
                    ));
                    next_index += 1;
                } else {
                    used_timestamp_tokens = true;
                    let timestamp_segments = decoded_window_to_contract_segments(
                        timed.decoded,
                        &mut next_index,
                        window.global_start_seconds,
                        window.global_end_seconds,
                        self.setup.language.clone(),
                    );
                    if timestamp_segments
                        .iter()
                        .any(|segment| !segment.words.is_empty())
                    {
                        used_timestamp_word_projection = true;
                    }
                    segments.extend(timestamp_segments);
                }
            }
        }
        let transcript = TranscriptionContract::from_segments(
            request.audio.source,
            request.language.clone(),
            segments,
        )
        .map_err(|error| model_output_mismatch(error.to_string()))?;
        let device_label = device_label(&self.setup.resolved_device);
        let mut diagnostics = whisper_setup_diagnostics(&self.setup);
        diagnostics.extend([
            "provider=candle-whisper".to_string(),
            format!("device={device_label}"),
            format!(
                "decoderThreads={}",
                decoder_threads_diagnostic(controls, &self.setup.resolved_device)
            ),
            format!("timingMode={}", window_controls.timing_mode.as_str()),
            format!(
                "leadingContextSeconds={}",
                window_controls.leading_context_seconds
            ),
            format!(
                "trailingContextSeconds={}",
                window_controls.trailing_context_seconds
            ),
            format!("modelId={}", self.setup.model_id),
            format!("bundle={}", self.setup.bundle.root.display()),
            format!("cuda={}", device_is_cuda(&self.setup.resolved_device)),
            format!("asrTask={}", self.setup.task.as_whisper_task()),
            if used_timestamp_tokens {
                "timing=whisperTimestampTokens".to_string()
            } else {
                "timing=expandedVadWindow".to_string()
            },
        ]);
        let decode_strategy = if decode.preserves_legacy_greedy_path() {
            "greedy"
        } else if decode.search.beam_size > 1 {
            "beamSearch"
        } else {
            "temperatureSampling"
        };
        diagnostics.extend([
            format!("decodeStrategy={decode_strategy}"),
            format!(
                "temperatureSchedule={}",
                decode
                    .search
                    .temperature_schedule
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            format!("bestOf={}", decode.search.best_of),
            format!("beamSize={}", decode.search.beam_size),
            format!("beamPatience={}", decode.search.patience),
            format!("lengthPenalty={}", decode.search.length_penalty),
            format!("samplingSeed={}", decode.search.seed),
            format!(
                "averageLogProbability={}",
                average_log_probabilities.iter().sum::<f64>()
                    / average_log_probabilities.len().max(1) as f64
            ),
            format!(
                "noSpeechProbability={}",
                no_speech_probabilities
                    .iter()
                    .copied()
                    .max_by(f64::total_cmp)
                    .map_or_else(|| "unavailable".to_string(), |value| value.to_string())
            ),
            format!(
                "compressionRatio={}",
                compression_ratios
                    .iter()
                    .copied()
                    .max_by(f64::total_cmp)
                    .unwrap_or(0.0)
            ),
            format!(
                "temperatureFallbackAttempts={}",
                attempted_temperatures
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            format!("noSpeechRejected={no_speech_rejected}"),
        ]);
        if let Some(language) = &self.setup.language {
            diagnostics.push(format!("language={language}"));
        }
        let observed_batch_execution = observed_candle_batch_execution(
            options.decode_runtime,
            decoder_max_active_row_batch_size,
        );
        if used_timestamp_word_projection {
            diagnostics.push("wordTiming=whisperTimestampProjection".to_string());
        }
        diagnostics.push(format!(
            "timestampTokensRequested={timestamp_tokens_requested}"
        ));
        diagnostics.push(format!("timestampTokensPresent={timestamp_tokens_present}"));
        diagnostics.push(format!(
            "timestampSegmentsRejected={rejected_timestamp_segments}"
        ));
        diagnostics.extend([
            format!("batchExecution={observed_batch_execution}"),
            format!("generation={}", generation_label(observed_batch_execution)),
            format!("completedRowCount={decoder_completed_row_count}"),
            format!("effectiveActiveBatchSize={decoder_max_active_row_batch_size}"),
            format!(
                "effectiveActiveBatchSizes={}",
                format_effective_active_batch_sizes(&decoder_effective_active_batch_sizes)
            ),
            format!("effectiveMaxBatchSize={decoder_max_active_row_batch_size}"),
            format!(
                "activeRowCompaction={}",
                decoder_active_row_compaction_count > 0
            ),
            format!("activeRowCompactionCount={decoder_active_row_compaction_count}"),
            format!(
                "cacheReuse={}",
                format_cache_reuse(
                    decoder_self_attention_cache_reused,
                    decoder_cross_attention_cache_reused
                )
            ),
            format!("decoderPromptPrefillCount={decoder_prompt_prefill_count}"),
            format!("decoderCachedTokenStepCount={decoder_cached_token_step_count}"),
            format!("decoderInputTokenCount={decoder_input_token_count}"),
            format!("generatedTokenCount={generated_token_count}"),
            format!("decoderCompletedRowCount={decoder_completed_row_count}"),
            format!("decoderMaxActiveRowBatchSize={decoder_max_active_row_batch_size}"),
            format!(
                "decoderEffectiveActiveBatchSizes={}",
                format_effective_active_batch_sizes(&decoder_effective_active_batch_sizes)
            ),
            format!("decoderActiveRowCompactionCount={decoder_active_row_compaction_count}"),
            format!(
                "decoderActiveRowCompactionOccurred={}",
                decoder_active_row_compaction_count > 0
            ),
            format!(
                "decoderSelfAttentionCacheReused={}",
                decoder_self_attention_cache_reused
            ),
            format!(
                "decoderCrossAttentionCacheReused={}",
                decoder_cross_attention_cache_reused
            ),
            format!(
                "phaseTiming.encoderSeconds={}",
                self.encoder_duration_seconds
            ),
            format!(
                "phaseTiming.decoderSeconds={}",
                self.decoder_duration_seconds
            ),
            format!(
                "phaseTiming.asrSeconds={}",
                request_started.elapsed().as_secs_f64()
            ),
        ]);
        diagnostics.extend(
            timing_fallbacks
                .into_iter()
                .map(|reason| format!("timingFallback={reason}")),
        );
        Ok(AsrResponse {
            model_id: request.model_id,
            language: self
                .setup
                .task
                .output_language_hint()
                .map(str::to_string)
                .or_else(|| self.setup.language.clone()),
            transcript,
            diagnostics,
        })
    }

    fn decode_window_with_timing_mode(
        &mut self,
        samples: &[f32],
        mode: CandleWhisperTimingMode,
        feature_extractor_mode: WhisperFeatureExtractorMode,
        decode: &CandleWhisperDecodeRequestConfig,
    ) -> Result<WhisperTimedWindow> {
        match mode {
            CandleWhisperTimingMode::NoTimestamps => {
                let decoded = self.decode_window(
                    samples,
                    WhisperDecodeMode::WithoutTimestamps,
                    feature_extractor_mode,
                    decode,
                )?;
                Ok(WhisperTimedWindow {
                    conditioning_token_ids: decoded.diagnostics.decoded_token_ids.clone(),
                    decoded: decoded.window,
                    timing: WhisperWindowTiming::ChunkWindow,
                    fallback_reason: None,
                    diagnostics: decoded.diagnostics,
                })
            }
            CandleWhisperTimingMode::Auto => {
                if timestamp_spec_for_timing_mode(&self.tokenizer, mode)?.is_some() {
                    let decoded = self.decode_window(
                        samples,
                        WhisperDecodeMode::TimestampTokens,
                        feature_extractor_mode,
                        decode,
                    )?;
                    let diagnostics = decoded.diagnostics.clone();
                    if has_stable_timestamp_segments(&decoded.window, samples) {
                        return Ok(WhisperTimedWindow {
                            conditioning_token_ids: diagnostics.decoded_token_ids.clone(),
                            decoded: decoded.window,
                            timing: WhisperWindowTiming::WhisperTimestampTokens,
                            fallback_reason: None,
                            diagnostics,
                        });
                    }
                    let mut fallback = self.decode_window_with_timing_mode(
                        samples,
                        CandleWhisperTimingMode::NoTimestamps,
                        feature_extractor_mode,
                        decode,
                    )?;
                    fallback.fallback_reason = Some("unstableTimestampSegments");
                    fallback
                        .diagnostics
                        .add_generation_counts_from(&diagnostics);
                    fallback.diagnostics.timestamp_tokens_requested =
                        diagnostics.timestamp_tokens_requested;
                    fallback.diagnostics.timestamp_tokens_present =
                        diagnostics.timestamp_tokens_present;
                    fallback.diagnostics.decoded_token_ids = diagnostics.decoded_token_ids;
                    return Ok(fallback);
                }
                let mut fallback = self.decode_window_with_timing_mode(
                    samples,
                    CandleWhisperTimingMode::NoTimestamps,
                    feature_extractor_mode,
                    decode,
                )?;
                fallback.fallback_reason = Some("missingTimestampMetadata");
                Ok(fallback)
            }
            CandleWhisperTimingMode::TimestampTokensRequired => {
                timestamp_spec_for_timing_mode(&self.tokenizer, mode)?;
                let decoded = self.decode_window(
                    samples,
                    WhisperDecodeMode::TimestampTokens,
                    feature_extractor_mode,
                    decode,
                )?;
                let diagnostics = decoded.diagnostics.clone();
                if !has_stable_timestamp_segments(&decoded.window, samples) {
                    return Err(model_output_mismatch(
                        "Whisper timestamp-token decode produced no stable bounded text segments",
                    ));
                }
                Ok(WhisperTimedWindow {
                    conditioning_token_ids: diagnostics.decoded_token_ids.clone(),
                    decoded: decoded.window,
                    timing: WhisperWindowTiming::WhisperTimestampTokens,
                    fallback_reason: None,
                    diagnostics,
                })
            }
        }
    }

    fn decode_windows_with_timing_mode(
        &mut self,
        windows: &[ChunkWindow],
        mode: CandleWhisperTimingMode,
        feature_extractor_mode: WhisperFeatureExtractorMode,
        decode: &CandleWhisperDecodeRequestConfig,
    ) -> Result<Vec<WhisperTimedWindow>> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        match mode {
            CandleWhisperTimingMode::NoTimestamps => self
                .decode_window_batch(
                    windows,
                    WhisperDecodeMode::WithoutTimestamps,
                    feature_extractor_mode,
                    decode,
                )?
                .into_iter()
                .map(|decoded| {
                    Ok(WhisperTimedWindow {
                        conditioning_token_ids: decoded.diagnostics.decoded_token_ids.clone(),
                        decoded: decoded.window,
                        timing: WhisperWindowTiming::ChunkWindow,
                        fallback_reason: None,
                        diagnostics: decoded.diagnostics,
                    })
                })
                .collect(),
            CandleWhisperTimingMode::Auto => {
                if timestamp_spec_for_timing_mode(&self.tokenizer, mode)?.is_none() {
                    let mut fallback = self.decode_windows_with_timing_mode(
                        windows,
                        CandleWhisperTimingMode::NoTimestamps,
                        feature_extractor_mode,
                        decode,
                    )?;
                    for timed in &mut fallback {
                        timed.fallback_reason = Some("missingTimestampMetadata");
                    }
                    return Ok(fallback);
                }
                let timestamp_decoded = self.decode_window_batch(
                    windows,
                    WhisperDecodeMode::TimestampTokens,
                    feature_extractor_mode,
                    decode,
                )?;
                let mut results: Vec<Option<WhisperTimedWindow>> = vec![None; windows.len()];
                let mut fallback_indices = Vec::new();
                for (index, (window, decoded)) in windows.iter().zip(timestamp_decoded).enumerate()
                {
                    let diagnostics = decoded.diagnostics.clone();
                    if has_stable_timestamp_segments(&decoded.window, &window.samples) {
                        results[index] = Some(WhisperTimedWindow {
                            conditioning_token_ids: diagnostics.decoded_token_ids.clone(),
                            decoded: decoded.window,
                            timing: WhisperWindowTiming::WhisperTimestampTokens,
                            fallback_reason: None,
                            diagnostics,
                        });
                    } else {
                        fallback_indices.push((index, diagnostics));
                    }
                }
                if !fallback_indices.is_empty() {
                    let fallback_windows = fallback_indices
                        .iter()
                        .map(|(index, _)| windows[*index].clone())
                        .collect::<Vec<_>>();
                    let fallbacks = self.decode_windows_with_timing_mode(
                        &fallback_windows,
                        CandleWhisperTimingMode::NoTimestamps,
                        feature_extractor_mode,
                        decode,
                    )?;
                    for ((index, timestamp_diagnostics), mut fallback) in
                        fallback_indices.into_iter().zip(fallbacks)
                    {
                        fallback.fallback_reason = Some("unstableTimestampSegments");
                        fallback
                            .diagnostics
                            .add_generation_counts_from(&timestamp_diagnostics);
                        fallback.diagnostics.timestamp_tokens_requested =
                            timestamp_diagnostics.timestamp_tokens_requested;
                        fallback.diagnostics.timestamp_tokens_present =
                            timestamp_diagnostics.timestamp_tokens_present;
                        fallback.diagnostics.decoded_token_ids =
                            timestamp_diagnostics.decoded_token_ids;
                        results[index] = Some(fallback);
                    }
                }
                Ok(results
                    .into_iter()
                    .map(|result| result.expect("every batched Whisper window is decoded"))
                    .collect())
            }
            CandleWhisperTimingMode::TimestampTokensRequired => {
                timestamp_spec_for_timing_mode(&self.tokenizer, mode)?;
                let decoded = self.decode_window_batch(
                    windows,
                    WhisperDecodeMode::TimestampTokens,
                    feature_extractor_mode,
                    decode,
                )?;
                decoded
                    .into_iter()
                    .zip(windows)
                    .map(|(decoded, window)| {
                        let diagnostics = decoded.diagnostics.clone();
                        if !has_stable_timestamp_segments(&decoded.window, &window.samples) {
                            return Err(model_output_mismatch(
                                "Whisper timestamp-token decode produced no stable bounded text segments",
                            ));
                        }
                        Ok(WhisperTimedWindow {
                            conditioning_token_ids: diagnostics.decoded_token_ids.clone(),
                            decoded: decoded.window,
                            timing: WhisperWindowTiming::WhisperTimestampTokens,
                            fallback_reason: None,
                            diagnostics,
                        })
                    })
                    .collect()
            }
        }
    }

    fn decode_window(
        &mut self,
        samples: &[f32],
        mode: WhisperDecodeMode,
        feature_extractor_mode: WhisperFeatureExtractorMode,
        decode: &CandleWhisperDecodeRequestConfig,
    ) -> Result<WhisperDecodeOutput> {
        self.decode_window_batch(
            &[ChunkWindow {
                samples: samples.to_vec(),
                chunk_start_seconds: 0.0,
                local_start_seconds: 0.0,
                local_end_seconds: samples.len() as f64 / whisper::SAMPLE_RATE as f64,
                global_start_seconds: 0.0,
                global_end_seconds: samples.len() as f64 / whisper::SAMPLE_RATE as f64,
            }],
            mode,
            feature_extractor_mode,
            decode,
        )
        .map(|mut outputs| outputs.remove(0))
    }

    fn decode_window_batch(
        &mut self,
        windows: &[ChunkWindow],
        mode: WhisperDecodeMode,
        feature_extractor_mode: WhisperFeatureExtractorMode,
        decode: &CandleWhisperDecodeRequestConfig,
    ) -> Result<Vec<WhisperDecodeOutput>> {
        let audio_features = self.encode_window_batch(windows, feature_extractor_mode)?;
        let token_outputs =
            self.decode_tokens_batch(&audio_features, windows.len(), mode, decode)?;
        token_outputs
            .into_iter()
            .map(|decoded| self.tokens_to_decode_output(decoded, mode))
            .collect()
    }

    fn encode_window_batch(
        &mut self,
        windows: &[ChunkWindow],
        feature_extractor_mode: WhisperFeatureExtractorMode,
    ) -> Result<Tensor> {
        if should_microbatch_encoder(&self.setup.resolved_device, windows.len()) {
            return self.encode_windows_individually(windows, feature_extractor_mode);
        }
        let mel = self.mel_tensor_batch(windows, feature_extractor_mode)?;
        let started = std::time::Instant::now();
        let encoded = self
            .model
            .encode(&mel)
            .map_err(|error| model_output_mismatch(format!("Whisper encoder failed: {error}")))?;
        self.encoder_duration_seconds += started.elapsed().as_secs_f64();
        Ok(encoded)
    }

    fn encode_windows_individually(
        &mut self,
        windows: &[ChunkWindow],
        feature_extractor_mode: WhisperFeatureExtractorMode,
    ) -> Result<Tensor> {
        let mut encoded = Vec::with_capacity(windows.len());
        for window in windows {
            let mel =
                self.mel_tensor_batch(std::slice::from_ref(window), feature_extractor_mode)?;
            let started = std::time::Instant::now();
            let features = self.model.encode(&mel).map_err(|error| {
                model_output_mismatch(format!("Whisper encoder failed: {error}"))
            })?;
            self.encoder_duration_seconds += started.elapsed().as_secs_f64();
            encoded.push(features);
        }
        let encoded = encoded.iter().collect::<Vec<_>>();
        Tensor::cat(&encoded, 0).map_err(|error| {
            model_output_mismatch(format!("failed to stack Whisper encoder features: {error}"))
        })
    }

    fn mel_tensor_batch(
        &self,
        windows: &[ChunkWindow],
        feature_extractor_mode: WhisperFeatureExtractorMode,
    ) -> Result<Tensor> {
        let n_mel = self.model.config().num_mel_bins;
        let mut features = Vec::with_capacity(windows.len() * n_mel * whisper::N_FRAMES);
        for window in windows {
            match feature_extractor_mode {
                WhisperFeatureExtractorMode::Legacy => {
                    let mel = whisper::audio::pcm_to_mel(
                        self.model.config(),
                        &window.samples,
                        &self.mel_filters,
                    );
                    let mel_frames = mel.len() / n_mel;
                    for mel_index in 0..n_mel {
                        let row_start = mel_index * mel_frames;
                        let available = mel_frames.min(whisper::N_FRAMES);
                        features.extend_from_slice(&mel[row_start..row_start + available]);
                        if available < whisper::N_FRAMES {
                            features
                                .extend(std::iter::repeat_n(0.0, whisper::N_FRAMES - available));
                        }
                    }
                }
                WhisperFeatureExtractorMode::Transformers => {
                    features.extend(transformers_whisper_pcm_to_mel(
                        self.model.config(),
                        &window.samples,
                        &self.transformers_mel_filters,
                    ));
                }
            }
        }
        Tensor::from_vec(
            features,
            (windows.len(), n_mel, whisper::N_FRAMES),
            &self.device,
        )
        .map_err(|error| model_output_mismatch(format!("failed to build mel tensor: {error}")))
    }

    fn tokens_to_decode_output(
        &self,
        decoded: WhisperTokenDecodeResult,
        mode: WhisperDecodeMode,
    ) -> Result<WhisperDecodeOutput> {
        let WhisperTokenDecodeResult {
            token_ids,
            stats: generation_stats,
            average_log_probability,
            no_speech_probability,
            attempted_temperatures,
            no_speech_rejected,
        } = decoded;
        let decoded_text = decode_text_tokens(&self.tokenizer, &token_ids)?;
        let mut diagnostics = WhisperDecodeDiagnostics {
            timestamp_tokens_requested: mode == WhisperDecodeMode::TimestampTokens,
            timestamp_tokens_present: timestamp_spec_for_timing_mode(
                &self.tokenizer,
                CandleWhisperTimingMode::Auto,
            )?
            .is_some_and(|spec| {
                token_ids
                    .iter()
                    .any(|token| timestamp_seconds(*token, &spec).is_some())
            }),
            decoded_token_ids: token_ids.clone(),
            average_log_probability,
            no_speech_probability,
            compression_ratio: text_compression_ratio(&decoded_text)?,
            attempted_temperatures,
            no_speech_rejected,
            ..WhisperDecodeDiagnostics::default()
        };
        generation_stats.extend(&mut diagnostics);
        match mode {
            WhisperDecodeMode::WithoutTimestamps => Ok(WhisperDecodeOutput {
                window: WhisperDecodedWindow {
                    text: decoded_text,
                    segments: Vec::new(),
                },
                diagnostics,
            }),
            WhisperDecodeMode::TimestampTokens => {
                decode_timestamp_window(&self.tokenizer, &token_ids)?
                    .map(|window| {
                        Ok(WhisperDecodeOutput {
                            window,
                            diagnostics: diagnostics.clone(),
                        })
                    })
                    .unwrap_or_else(|| {
                        Ok(WhisperDecodeOutput {
                            window: WhisperDecodedWindow {
                                text: decoded_text,
                                segments: Vec::new(),
                            },
                            diagnostics,
                        })
                    })
            }
        }
    }

    fn decode_tokens_batch(
        &mut self,
        audio_features: &Tensor,
        row_count: usize,
        mode: WhisperDecodeMode,
        decode: &CandleWhisperDecodeRequestConfig,
    ) -> Result<Vec<WhisperTokenDecodeResult>> {
        if decode.preserves_legacy_greedy_path() {
            return self.decode_tokens_batch_greedy(audio_features, row_count, mode, decode);
        }

        let mut outputs = Vec::with_capacity(row_count);
        for row_index in 0..row_count {
            let row_features = audio_features
                .i(row_index)
                .and_then(|features| features.unsqueeze(0))
                .map_err(|error| {
                    model_output_mismatch(format!(
                        "failed to select Whisper search row {row_index}: {error}"
                    ))
                })?;
            outputs.push(self.decode_tokens_configured(&row_features, mode, decode)?);
        }
        Ok(outputs)
    }

    fn decode_tokens_configured(
        &mut self,
        audio_features: &Tensor,
        mode: WhisperDecodeMode,
        decode: &CandleWhisperDecodeRequestConfig,
    ) -> Result<WhisperTokenDecodeResult> {
        let eos = self.eos_token_id()?;
        let max_length = self
            .generation
            .max_length
            .unwrap_or(self.model.config().max_target_positions)
            .min(self.model.config().max_target_positions);
        let initial_tokens = self.initial_tokens(mode, &decode.initial_prompt_tokens)?;
        let max_generated_tokens = max_length.saturating_sub(initial_tokens.token_ids.len());
        let mut stats = WhisperGenerationStats::default();
        let temperatures = decode.search.temperature_schedule.clone();
        let (
            (mut search, average_log_probability, no_speech_probability, decision),
            attempted_temperatures,
        ) = run_ordered_temperature_fallback(
            &temperatures,
            |temperature_index, temperature| {
                let mut no_speech_probability = None;
                let search = crate::native_whisper_decode::decode_at_temperature(
                    &decode.search,
                    temperature_index,
                    temperature,
                    eos,
                    max_generated_tokens,
                    &mut |generated| {
                        self.configured_search_logits(
                            audio_features,
                            &initial_tokens.token_ids,
                            initial_tokens.sot_position,
                            generated,
                            mode,
                            decode,
                            &mut no_speech_probability,
                            &mut stats,
                        )
                    },
                )?;
                for _ in &search.token_ids {
                    stats.record_generated_token();
                }
                let average_log_probability = search.average_log_probability();
                let text = decode_text_tokens(&self.tokenizer, &search.token_ids)?;
                let compression_ratio = text_compression_ratio(&text)?;
                let decision = fallback_attempt_decision(
                    decode,
                    average_log_probability,
                    no_speech_probability,
                    compression_ratio,
                );
                Ok((
                    search,
                    average_log_probability,
                    no_speech_probability,
                    decision,
                ))
            },
            |attempt| attempt.3 == WhisperFallbackAttemptDecision::Retry,
        )?;
        let no_speech_rejected = apply_no_speech_rejection(decision, &mut search.token_ids);
        stats.record_completed_row();
        Ok(WhisperTokenDecodeResult {
            token_ids: search.token_ids,
            stats,
            average_log_probability,
            no_speech_probability,
            attempted_temperatures,
            no_speech_rejected,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn configured_search_logits(
        &mut self,
        audio_features: &Tensor,
        initial_tokens: &[u32],
        sot_position: usize,
        generated: &[u32],
        mode: WhisperDecodeMode,
        decode: &CandleWhisperDecodeRequestConfig,
        no_speech_probability: &mut Option<f64>,
        stats: &mut WhisperGenerationStats,
    ) -> Result<Vec<f32>> {
        self.model.reset_cache();
        let mut tokens = Vec::with_capacity(initial_tokens.len() + generated.len());
        tokens.extend_from_slice(initial_tokens);
        tokens.extend_from_slice(generated);
        let input = WhisperDecoderInput {
            token_ids: tokens.clone(),
            position_offset: 0,
            flush_cache: true,
            kind: WhisperDecoderInputKind::PromptPrefill,
        };
        stats.record_input(&input);
        stats.record_active_row_batch_size(1);
        let token_tensor = Tensor::new(tokens.as_slice(), &self.device)
            .and_then(|tokens| tokens.unsqueeze(0))
            .map_err(|error| {
                model_output_mismatch(format!(
                    "failed to build Whisper search token tensor: {error}"
                ))
            })?;
        let decoder_started = std::time::Instant::now();
        let (decoded, decoder_stats) = self
            .model
            .decode(&token_tensor, audio_features, 0, true)
            .map_err(|error| {
                model_output_mismatch(format!("Whisper search decoder failed: {error}"))
            })?;
        self.decoder_duration_seconds += decoder_started.elapsed().as_secs_f64();
        stats.record_decoder_stats(decoder_stats);
        let logits = self.model.project_logits(&decoded).map_err(|error| {
            model_output_mismatch(format!("Whisper search logits projection failed: {error}"))
        })?;
        if generated.is_empty() && no_speech_probability.is_none() {
            *no_speech_probability = token_id(&self.tokenizer, "<|nospeech|>")
                .map(|token| tensor_token_probability_at_position(&logits, 0, sot_position, token))
                .transpose()?
                .flatten();
        }
        let mut next_logits = logits
            .i((0, tokens.len() - 1, ..))
            .and_then(|logits| logits.to_dtype(DType::F32))
            .and_then(|logits| logits.to_vec1::<f32>())
            .map_err(|error| {
                model_output_mismatch(format!("Whisper search decode failed: {error}"))
            })?;
        self.apply_logit_filters(&mut next_logits, mode, generated, decode)?;
        Ok(next_logits)
    }

    fn decode_tokens_batch_greedy(
        &mut self,
        audio_features: &Tensor,
        row_count: usize,
        mode: WhisperDecodeMode,
        decode: &CandleWhisperDecodeRequestConfig,
    ) -> Result<Vec<WhisperTokenDecodeResult>> {
        self.model.reset_cache();
        let eos = self.eos_token_id()?;
        let max_length = self
            .generation
            .max_length
            .unwrap_or(self.model.config().max_target_positions)
            .min(self.model.config().max_target_positions);
        let initial_tokens = self.initial_tokens(mode, &decode.initial_prompt_tokens)?;
        let mut active_rows = (0..row_count)
            .map(|original_index| ActiveWhisperDecodeRow {
                original_index,
                row: WhisperAutoregressiveRow::new(initial_tokens.token_ids.clone()),
                stats: WhisperGenerationStats::default(),
                score: 0.0,
                no_speech_probability: None,
            })
            .collect::<Vec<_>>();
        let mut active_features = audio_features.clone();
        let mut completed: Vec<Option<WhisperTokenDecodeResult>> = vec![None; row_count];

        while !active_rows.is_empty() && active_rows[0].row.tokens.len() < max_length {
            let active_len_before_step = active_rows.len();
            let input = active_rows[0].row.next_decoder_input();
            debug_assert!(active_rows.iter().all(|active| active
                .row
                .next_decoder_input()
                .token_ids
                .len()
                == input.token_ids.len()));
            for active in &mut active_rows {
                active.stats.record_input(&input);
                active
                    .stats
                    .record_active_row_batch_size(active_len_before_step);
            }
            let token_ids = active_rows
                .iter()
                .flat_map(|active| active.row.next_decoder_input().token_ids)
                .collect::<Vec<_>>();
            let token_tensor = Tensor::from_vec(
                token_ids,
                (active_rows.len(), input.token_ids.len()),
                &self.device,
            )
            .map_err(|error| {
                model_output_mismatch(format!("failed to build batched token tensor: {error}"))
            })?;
            let decoder_started = std::time::Instant::now();
            let (decoded, decoder_stats) = self
                .model
                .decode(
                    &token_tensor,
                    &active_features,
                    input.position_offset,
                    input.flush_cache,
                )
                .map_err(|error| {
                    model_output_mismatch(format!("Whisper batched decoder failed: {error}"))
                })?;
            self.decoder_duration_seconds += decoder_started.elapsed().as_secs_f64();
            for active in &mut active_rows {
                active.stats.record_decoder_stats(decoder_stats);
                active.row.mark_forwarded();
            }
            let logits = self.model.project_logits(&decoded).map_err(|error| {
                model_output_mismatch(format!("Whisper batched logits projection failed: {error}"))
            })?;
            let seq_index = input.token_ids.len() - 1;
            let mut next_tokens = Vec::with_capacity(active_rows.len());
            for (active_index, mut active) in
                std::mem::take(&mut active_rows).into_iter().enumerate()
            {
                let mut next_logits = logits
                    .i((active_index, seq_index, ..))
                    .and_then(|logits| logits.to_dtype(DType::F32))
                    .and_then(|logits| logits.to_vec1::<f32>())
                    .map_err(|error| {
                        model_output_mismatch(format!(
                            "Whisper batched greedy decode failed: {error}"
                        ))
                    })?;
                if active.row.generated_tokens().is_empty() {
                    active.no_speech_probability = token_id(&self.tokenizer, "<|nospeech|>")
                        .map(|token| {
                            tensor_token_probability_at_position(
                                &logits,
                                active_index,
                                initial_tokens.sot_position,
                                token,
                            )
                        })
                        .transpose()?
                        .flatten();
                }
                self.apply_logit_filters(
                    &mut next_logits,
                    mode,
                    active.row.generated_tokens(),
                    decode,
                )?;
                let next = argmax_finite(&next_logits).ok_or_else(|| {
                    model_output_mismatch(
                        "Whisper logits were fully suppressed during batched decode",
                    )
                })? as u32;
                active.score +=
                    token_log_probability(&next_logits, next).unwrap_or(f64::NEG_INFINITY);
                next_tokens.push((active, next));
            }
            let (mut survivors, survivor_indices) =
                apply_active_row_decisions(next_tokens, eos, &mut completed)?;
            if survivors.is_empty() {
                break;
            }
            if survivors.len() < active_len_before_step {
                if let Some(survivor) = survivors.first_mut() {
                    survivor.stats.record_active_row_compaction();
                }
                let row_indices = Tensor::new(survivor_indices.as_slice(), &self.device)
                    .and_then(|indices| indices.to_dtype(DType::I64))
                    .map_err(|error| {
                        model_output_mismatch(format!(
                            "failed to build Whisper active-row index tensor: {error}"
                        ))
                    })?;
                active_features =
                    active_features
                        .index_select(&row_indices, 0)
                        .map_err(|error| {
                            model_output_mismatch(format!(
                                "failed to compact Whisper encoder features: {error}"
                            ))
                        })?;
                self.model
                    .select_cache_rows(&row_indices)
                    .map_err(|error| {
                        model_output_mismatch(format!(
                            "failed to compact Whisper decoder KV cache: {error}"
                        ))
                    })?;
            }
            active_rows = survivors;
        }

        for mut active in active_rows {
            active.stats.record_completed_row();
            let token_ids = active.row.into_generated_tokens();
            completed[active.original_index] = Some(WhisperTokenDecodeResult {
                average_log_probability: average_log_probability(
                    active.score,
                    token_ids.len(),
                    false,
                ),
                token_ids,
                stats: active.stats,
                no_speech_probability: active.no_speech_probability,
                attempted_temperatures: vec![0.0],
                no_speech_rejected: false,
            });
        }
        completed
            .into_iter()
            .map(|result| {
                result.ok_or_else(|| model_output_mismatch("missing Whisper batch row result"))
            })
            .collect()
    }

    fn apply_logit_filters(
        &self,
        logits: &mut [f32],
        mode: WhisperDecodeMode,
        generated: &[u32],
        decode: &CandleWhisperDecodeRequestConfig,
    ) -> Result<()> {
        for token in &self.generation.suppress_tokens {
            suppress_token(logits, *token);
        }
        if generated.is_empty() {
            for token in &self.generation.begin_suppress_tokens {
                suppress_token(logits, *token);
            }
        }
        for token in request_suppressed_token_ids(&self.tokenizer, decode) {
            suppress_token(logits, token);
        }
        let no_timestamps = self
            .generation
            .no_timestamps_token_id
            .or_else(|| token_id(&self.tokenizer, whisper::NO_TIMESTAMPS_TOKEN));
        if let Some(no_timestamps) = no_timestamps {
            suppress_token(logits, no_timestamps);
        }
        let Some(spec) =
            timestamp_spec_for_timing_mode(&self.tokenizer, CandleWhisperTimingMode::Auto)?
        else {
            return Ok(());
        };
        match mode {
            WhisperDecodeMode::WithoutTimestamps => {
                suppress_range(logits, spec.begin_token_id, spec.end_token_id);
            }
            WhisperDecodeMode::TimestampTokens => {
                apply_timestamp_logit_rules(logits, generated, &spec, self.eos_token_id()?)?;
            }
        }
        Ok(())
    }

    fn initial_tokens(
        &self,
        mode: WhisperDecodeMode,
        request_prompt: &[u32],
    ) -> Result<WhisperInitialTokens> {
        Self::build_initial_tokens(
            &self.generation,
            &self.tokenizer,
            self.setup.language.as_deref(),
            self.setup.task,
            mode,
            request_prompt,
            self.model.config().max_target_positions / 2,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_initial_tokens(
        generation: &GenerationConfig,
        tokenizer: &Tokenizer,
        language: Option<&str>,
        task: TranscriptionTask,
        mode: WhisperDecodeMode,
        request_prompt: &[u32],
        max_prompt_context: usize,
    ) -> Result<WhisperInitialTokens> {
        let controls =
            Self::initial_prompt_tokens_for_mode(generation, tokenizer, language, task, mode)?;
        for token_id in request_prompt {
            if tokenizer.id_to_token(*token_id).is_none() {
                return Err(invalid_request(format!(
                    "Whisper request prompt token id `{token_id}` is not in the tokenizer vocabulary"
                )));
            }
        }
        if request_prompt.is_empty() {
            return Ok(WhisperInitialTokens {
                token_ids: controls,
                sot_position: 0,
            });
        }
        let start_of_prev = token_id(tokenizer, WHISPER_START_OF_PREV_TOKEN).ok_or_else(|| {
            invalid_request("Whisper tokenizer is missing start-of-previous-text token")
        })?;
        let prompt_capacity = max_prompt_context.saturating_sub(1);
        let prompt_start = request_prompt.len().saturating_sub(prompt_capacity);
        let mut token_ids = Vec::with_capacity(1 + prompt_capacity + controls.len());
        token_ids.push(start_of_prev);
        token_ids.extend_from_slice(&request_prompt[prompt_start..]);
        let sot_position = token_ids.len();
        token_ids.extend(controls);
        Ok(WhisperInitialTokens {
            token_ids,
            sot_position,
        })
    }

    #[cfg(test)]
    fn initial_prompt_tokens(
        generation: &GenerationConfig,
        tokenizer: &Tokenizer,
        language: Option<&str>,
    ) -> Result<Vec<u32>> {
        Self::initial_prompt_tokens_for_mode(
            generation,
            tokenizer,
            language,
            TranscriptionTask::Transcribe,
            WhisperDecodeMode::WithoutTimestamps,
        )
    }

    #[cfg(test)]
    fn initial_prompt_tokens_for_task(
        generation: &GenerationConfig,
        tokenizer: &Tokenizer,
        language: Option<&str>,
        task: TranscriptionTask,
    ) -> Result<Vec<u32>> {
        Self::initial_prompt_tokens_for_mode(
            generation,
            tokenizer,
            language,
            task,
            WhisperDecodeMode::WithoutTimestamps,
        )
    }

    fn initial_prompt_tokens_for_mode(
        generation: &GenerationConfig,
        tokenizer: &Tokenizer,
        language: Option<&str>,
        task: TranscriptionTask,
        mode: WhisperDecodeMode,
    ) -> Result<Vec<u32>> {
        let decoder_start = Self::decoder_start_token_id(generation, tokenizer)?;
        let mut tokens = vec![decoder_start];
        if let Some(language) = language {
            let token = Self::language_token_id(generation, tokenizer, language).ok_or_else(|| {
                invalid_request(format!(
                    "Whisper generation config/tokenizer does not define language token `{language}`"
                ))
            })?;
            tokens.push(token);
        }
        let task_token = Self::task_token_id(generation, tokenizer, task.as_whisper_task())
            .ok_or_else(|| {
                invalid_request(format!(
                    "Whisper generation config/tokenizer is missing {} task token",
                    task.as_whisper_task()
                ))
            })?;
        tokens.push(task_token);
        let no_timestamps = token_id(tokenizer, whisper::NO_TIMESTAMPS_TOKEN);
        if mode == WhisperDecodeMode::WithoutTimestamps {
            tokens.push(no_timestamps.ok_or_else(|| {
                invalid_request("Whisper tokenizer is missing no-timestamps token")
            })?);
        }
        if let Some(forced) = &generation.forced_decoder_ids {
            for (position, token) in forced {
                let Some(token) = token else {
                    continue;
                };
                if mode == WhisperDecodeMode::TimestampTokens
                    && no_timestamps.is_some_and(|no_timestamps| no_timestamps == *token)
                {
                    continue;
                }
                if *position < tokens.len() {
                    tokens[*position] = *token;
                } else {
                    while tokens.len() < *position {
                        tokens.push(decoder_start);
                    }
                    tokens.push(*token);
                }
            }
        }
        Ok(tokens)
    }

    fn decoder_start_token_id(generation: &GenerationConfig, tokenizer: &Tokenizer) -> Result<u32> {
        generation
            .decoder_start_token_id
            .or_else(|| token_id(tokenizer, whisper::SOT_TOKEN))
            .ok_or_else(|| {
                invalid_request("Whisper generation config is missing decoder_start_token_id")
            })
    }

    fn eos_token_id(&self) -> Result<u32> {
        Self::resolve_eos_token_id(&self.generation, &self.tokenizer)
    }

    fn resolve_eos_token_id(generation: &GenerationConfig, tokenizer: &Tokenizer) -> Result<u32> {
        generation
            .eos_token_id
            .or_else(|| token_id(tokenizer, whisper::EOT_TOKEN))
            .ok_or_else(|| invalid_request("Whisper generation config is missing eos_token_id"))
    }

    fn language_token_id(
        generation: &GenerationConfig,
        tokenizer: &Tokenizer,
        language: &str,
    ) -> Option<u32> {
        let normalized = language.trim().to_lowercase();
        let wrapped = format!("<|{normalized}|>");
        generation
            .lang_to_id
            .get(&wrapped)
            .or_else(|| generation.lang_to_id.get(&normalized))
            .copied()
            .or_else(|| token_id(tokenizer, &wrapped))
    }

    fn task_token_id(
        generation: &GenerationConfig,
        tokenizer: &Tokenizer,
        task: &str,
    ) -> Option<u32> {
        let wrapped = format!("<|{task}|>");
        generation
            .task_to_id
            .get(&wrapped)
            .or_else(|| generation.task_to_id.get(task))
            .copied()
            .or_else(|| token_id(tokenizer, &wrapped))
    }
}

fn decoder_threads_diagnostic(
    controls: &CandleWhisperRuntimeControls,
    resolved_device: &ResolvedNativeDevice,
) -> String {
    match (controls.decoder_threads, resolved_device.cuda_active()) {
        (Some(_), true) => "ignored(cuda)".to_string(),
        (Some(threads), false) => threads.to_string(),
        (None, _) => "default".to_string(),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct WhisperDecodeOutput {
    window: WhisperDecodedWindow,
    diagnostics: WhisperDecodeDiagnostics,
}

fn candle_batch_size(options: &CandleWhisperOptions, chunk_count: usize) -> usize {
    match options.decode_runtime {
        CandleWhisperDecodeRuntime::AutoregressiveKvCache => {}
        CandleWhisperDecodeRuntime::ActiveRowTensorBatch => {
            return options.max_batch_size.unwrap_or(chunk_count.max(1)).max(1);
        }
    }
    if !options.batch_chunks {
        return 1;
    }
    if candle_batch_count(options, chunk_count) <= 1 {
        return chunk_count.max(1);
    }
    options.max_batch_size.unwrap_or(chunk_count.max(1)).max(1)
}

fn generation_label(observed_batch_execution: &str) -> &'static str {
    match observed_batch_execution {
        crate::CANDLE_WHISPER_ACTIVE_ROW_TENSOR_BATCH_EXECUTION => "active-row-tensor-batch",
        _ => "autoregressive-kv-cache",
    }
}

fn observed_candle_batch_execution(
    runtime: CandleWhisperDecodeRuntime,
    decoder_max_active_row_batch_size: usize,
) -> &'static str {
    if runtime == CandleWhisperDecodeRuntime::ActiveRowTensorBatch
        && decoder_max_active_row_batch_size > 1
    {
        return crate::CANDLE_WHISPER_ACTIVE_ROW_TENSOR_BATCH_EXECUTION;
    }
    crate::CANDLE_WHISPER_AUTOREGRESSIVE_KV_CACHE_EXECUTION
}

fn format_effective_active_batch_sizes(sizes: &[usize]) -> String {
    if sizes.is_empty() {
        return "none".to_string();
    }
    let mut sizes = sizes.to_vec();
    sizes.sort_unstable();
    sizes.dedup();
    sizes
        .into_iter()
        .map(|size| size.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn format_cache_reuse(self_attention: bool, cross_attention: bool) -> &'static str {
    match (self_attention, cross_attention) {
        (true, true) => "self-and-cross-attention",
        (true, false) => "self-attention",
        (false, true) => "cross-attention",
        (false, false) => "none",
    }
}

fn collect_chunk_windows(
    samples: &[f32],
    sample_rate: u32,
    chunks: &[SpeechActivitySegment],
    controls: &CandleWhisperWindowControls,
) -> Result<Vec<ChunkWindow>> {
    let mut windows = Vec::new();
    for chunk in chunks {
        windows.extend(chunk_windows(samples, sample_rate, chunk, controls)?);
    }
    Ok(windows)
}

fn apply_active_row_decisions(
    next_tokens: Vec<(ActiveWhisperDecodeRow, u32)>,
    eos: u32,
    completed: &mut [Option<WhisperTokenDecodeResult>],
) -> Result<(Vec<ActiveWhisperDecodeRow>, Vec<u32>)> {
    let mut survivors = Vec::new();
    let mut survivor_indices = Vec::new();
    for (active_index, (mut active, next)) in next_tokens.into_iter().enumerate() {
        if next == eos {
            let original_index = active.original_index;
            if completed.get(original_index).is_none() {
                return Err(model_output_mismatch(
                    "Whisper active row completed outside the result range",
                ));
            }
            active.stats.record_completed_row();
            let token_ids = active.row.into_generated_tokens();
            completed[original_index] = Some(WhisperTokenDecodeResult {
                average_log_probability: average_log_probability(
                    active.score,
                    token_ids.len(),
                    true,
                ),
                token_ids,
                stats: active.stats,
                no_speech_probability: active.no_speech_probability,
                attempted_temperatures: vec![0.0],
                no_speech_rejected: false,
            });
        } else {
            active.row.accept(next);
            active.stats.record_generated_token();
            survivor_indices.push(active_index as u32);
            survivors.push(active);
        }
    }
    Ok((survivors, survivor_indices))
}

fn decode_timestamp_window(
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

fn whisper_timestamp_spec(tokenizer: &Tokenizer) -> Result<WhisperTimestampSpec> {
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

fn timestamp_spec_for_timing_mode(
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

fn timestamp_seconds(token_id: u32, spec: &WhisperTimestampSpec) -> Option<f64> {
    (spec.begin_token_id..spec.end_token_id)
        .contains(&token_id)
        .then(|| (token_id - spec.begin_token_id) as f64 * spec.seconds_per_token)
}

fn apply_timestamp_logit_rules(
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

fn suppress_token(logits: &mut [f32], token: u32) {
    if let Some(logit) = logits.get_mut(token as usize) {
        *logit = f32::NEG_INFINITY;
    }
}

fn request_suppressed_token_ids(
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

fn suppress_range(logits: &mut [f32], start: u32, end: u32) {
    let start = start as usize;
    let end = (end as usize).min(logits.len());
    if start >= end {
        return;
    }
    for logit in &mut logits[start..end] {
        *logit = f32::NEG_INFINITY;
    }
}

fn argmax_finite(logits: &[f32]) -> Option<usize> {
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

fn has_stable_timestamp_segments(decoded: &WhisperDecodedWindow, samples: &[f32]) -> bool {
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

fn decode_text_tokens(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(token_ids, true)
        .map(clean_decoded_text)
        .map_err(|error| model_output_mismatch(format!("failed to decode Whisper tokens: {error}")))
}

fn clean_decoded_text(text: String) -> String {
    text.replace("  ", " ").trim().to_string()
}

fn window_fallback_segment(
    index: u64,
    text: String,
    start_seconds: f64,
    end_seconds: f64,
    language: Option<String>,
) -> TranscriptSegmentContract {
    let mut segment = TranscriptSegmentContract::new(index, text);
    segment.start_seconds = Some(start_seconds);
    segment.end_seconds = Some(end_seconds);
    segment.language = language;
    segment
        .attributes
        .insert("provider".to_string(), "candle-whisper".to_string());
    segment
        .attributes
        .insert("timing".to_string(), "global".to_string());
    segment
}

fn decoded_window_to_contract_segments(
    decoded: WhisperDecodedWindow,
    next_index: &mut u64,
    window_start_seconds: f64,
    window_end_seconds: f64,
    language: Option<String>,
) -> Vec<TranscriptSegmentContract> {
    decoded
        .segments
        .into_iter()
        .filter_map(|decoded_segment| {
            let text = decoded_segment.text.trim().to_string();
            if text.is_empty() {
                return None;
            }
            let mut segment = TranscriptSegmentContract::new(*next_index, text);
            *next_index += 1;
            let global_start = (window_start_seconds + decoded_segment.start_seconds)
                .clamp(window_start_seconds, window_end_seconds);
            let global_end = (window_start_seconds + decoded_segment.end_seconds)
                .clamp(window_start_seconds, window_end_seconds);
            segment.start_seconds = Some(global_start);
            segment.end_seconds = Some(global_end);
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
            Some(segment)
        })
        .collect()
}

#[cfg(test)]
fn project_words_from_timestamp_segment(
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
            TranscriptWordContract {
                text: word.to_string(),
                start_seconds: Some(word_start),
                end_seconds: Some(word_end),
                confidence: None,
                speaker: None,
                attributes: BTreeMap::from([(
                    "timing".to_string(),
                    "whisperTimestampProjection".to_string(),
                )]),
            }
        })
        .collect()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, label: &str) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|error| {
        setup_error(format!(
            "failed to read {label} `{}`: {error}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        invalid_request(format!(
            "failed to parse {label} `{}`: {error}",
            path.display()
        ))
    })
}

fn candle_device(resolved: &ResolvedNativeDevice) -> Result<Device> {
    resolved.candle_device()
}

fn device_label(resolved: &ResolvedNativeDevice) -> String {
    resolved.diagnostic_name()
}

fn device_is_cuda(resolved: &ResolvedNativeDevice) -> bool {
    resolved.cuda_active()
}

fn should_microbatch_encoder(resolved: &ResolvedNativeDevice, window_count: usize) -> bool {
    window_count > 1 && device_is_cuda(resolved)
}

#[derive(Debug, Clone)]
struct ChunkWindow {
    samples: Vec<f32>,
    chunk_start_seconds: f64,
    local_start_seconds: f64,
    local_end_seconds: f64,
    global_start_seconds: f64,
    global_end_seconds: f64,
}

fn chunk_windows(
    samples: &[f32],
    sample_rate: u32,
    chunk: &SpeechActivitySegment,
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

fn token_id(tokenizer: &Tokenizer, token: &str) -> Option<u32> {
    tokenizer.token_to_id(token)
}

fn mel_filter_bank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
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

fn transformers_mel_filter_bank(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
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

fn transformers_whisper_pcm_to_mel(
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QTensor;
    use sha2::{Digest, Sha256};
    use std::io::Cursor;
    use tokenizers::models::wordlevel::WordLevel;

    #[test]
    #[ignore = "requires pinned German audio, WhisperX JSON, Whisper-small bundle, and Transformers mel/logit oracles; run explicitly with --ignored"]
    fn german_no_align_fp32_matches_pinned_transformers_greedy_trace() {
        let bundle = std::env::var_os("CANDLE_WHISPER_SMALL_BUNDLE")
            .map(PathBuf::from)
            .expect("CANDLE_WHISPER_SMALL_BUNDLE must point at revision 973afd24965f72e36ca33b3055d56a652f456b4d");
        let audio_path = std::env::var_os("CANDLE_WHISPER_GERMAN_WAV")
            .map(PathBuf::from)
            .expect("CANDLE_WHISPER_GERMAN_WAV must point at the pinned five-second cache probe");
        const PINNED_REVISION: &str = "973afd24965f72e36ca33b3055d56a652f456b4d";
        assert_eq!(
            bundle.file_name().and_then(|name| name.to_str()),
            Some(PINNED_REVISION),
            "Whisper-small bundle must be the pinned Hugging Face snapshot"
        );
        let assert_sha256 = |path: &Path, expected: &str| {
            let bytes = std::fs::read(path).expect("pinned parity resource");
            let actual = format!("{:x}", Sha256::digest(bytes));
            assert_eq!(
                actual,
                expected,
                "pinned resource changed: {}",
                path.display()
            );
        };
        assert_sha256(
            &audio_path,
            "80df13c0cf5733684be7ccc9a243ce45637debfc5186be575a1d0608ad929ca6",
        );
        let whisperx_oracle_path =
            std::env::var_os("CANDLE_WHISPER_GERMAN_WHISPERX_ORACLE")
                .map(PathBuf::from)
                .expect(
                    "CANDLE_WHISPER_GERMAN_WHISPERX_ORACLE must point at the retained WhisperX 3.8.6 JSON",
                );
        let whisperx_oracle_bytes =
            std::fs::read(whisperx_oracle_path).expect("retained WhisperX 3.8.6 oracle");
        assert_eq!(
            format!("{:x}", Sha256::digest(&whisperx_oracle_bytes)),
            "1b38bea5200eea4c037eea0420ff71c73283c238f9820c978a60ca0c7612e095",
            "retained WhisperX 3.8.6 oracle changed"
        );
        let whisperx_oracle: serde_json::Value =
            serde_json::from_slice(&whisperx_oracle_bytes).expect("WhisperX JSON");
        let whisperx_segment = &whisperx_oracle["segments"][0];
        assert_eq!(whisperx_segment["start"], serde_json::json!(0.322));
        assert_eq!(whisperx_segment["end"], serde_json::json!(5.0));
        assert_eq!(
            whisperx_segment["text"]
                .as_str()
                .expect("WhisperX segment text")
                .trim(),
            "Das ist nicht dein Lieblingsverband."
        );
        for (name, expected) in [
            (
                "config.json",
                "e6a2b489da1b5aed65a8eb8d1e7466fa867ad5643a8bc138ba708bd56b2875c4",
            ),
            (
                "generation_config.json",
                "71565b8ef50d0bf7a1193ed4bbed195b94e70c18894d81bba2f1233dcec3ab53",
            ),
            (
                "tokenizer.json",
                "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
            ),
            (
                "preprocessor_config.json",
                "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666",
            ),
            (
                "model.safetensors",
                "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b",
            ),
        ] {
            assert_sha256(&bundle.join(name), expected);
        }
        let mut reader = hound::WavReader::open(&audio_path).expect("German PCM fixture");
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, whisper::SAMPLE_RATE as u32);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        let samples = reader
            .samples::<i16>()
            .map(|sample| sample.expect("PCM sample") as f32 / 32_768.0)
            .collect::<Vec<_>>();
        let duration = samples.len() as f64 / spec.sample_rate as f64;
        assert!((duration - 5.0).abs() <= f64::EPSILON);
        let request = AsrRequest {
            audio: crate::LoadedAudio {
                samples,
                sample_rate: spec.sample_rate,
                channels: spec.channels,
                source: Some("pinned-german-cache-probe".to_string()),
            },
            chunks: vec![SpeechActivitySegment::new(0.322, duration, 1.0)
                .expect("exact WhisperX 3.8.6 speech span")],
            task: TranscriptionTask::Transcribe,
            language: Some("de".to_string()),
            model_id: "openai/whisper-small".to_string(),
        };
        let options = CandleWhisperOptions {
            model_id: request.model_id.clone(),
            language: request.language.clone(),
            device: crate::NativeDevicePreference::Cpu,
            compute_type: CandleWhisperComputeType::Fp32,
            model_bundle: Some(bundle),
            model_cache_only: true,
            batch_chunks: false,
            max_batch_size: Some(1),
            ..CandleWhisperOptions::default()
        };
        let config = CandleWhisperTranscriptionRequestConfig {
            runtime: CandleWhisperRuntimeControls {
                decoder_threads: Some(8),
                ..CandleWhisperRuntimeControls::default()
            },
            decode: CandleWhisperDecodeRequestConfig::default(),
            window: CandleWhisperWindowControls {
                timing_mode: CandleWhisperTimingMode::NoTimestamps,
                leading_context_seconds: 0.0,
                trailing_context_seconds: 0.0,
            },
        };
        let setup =
            WhisperRunSetup::from_options_and_request(&options, &request).expect("run setup");
        let mut session = CandleWhisperSession::load(setup).expect("load Whisper-small");
        let windows = collect_chunk_windows(
            &request.audio.samples,
            request.audio.sample_rate,
            &request.chunks,
            &config.window,
        )
        .expect("zero-context windows");
        assert_eq!(windows.len(), 1);

        let mel = session
            .mel_tensor_batch(&windows, WhisperFeatureExtractorMode::Transformers)
            .expect("Candle mel");
        assert_eq!(mel.dims3().expect("mel shape"), (1, 80, 3000));
        let mel_values = mel
            .flatten_all()
            .and_then(|mel| mel.to_vec1::<f32>())
            .expect("f32 mel values");
        let mel_min = mel_values.iter().copied().fold(f32::INFINITY, f32::min);
        let mel_max = mel_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mel_oracle_path = std::env::var_os("CANDLE_WHISPER_GERMAN_MEL_ORACLE")
            .map(PathBuf::from)
            .expect(
                "CANDLE_WHISPER_GERMAN_MEL_ORACLE must point at the retained Transformers f32le tensor",
            );
        let mel_oracle_bytes =
            std::fs::read(mel_oracle_path).expect("retained Transformers mel oracle");
        assert_eq!(
            format!("{:x}", Sha256::digest(&mel_oracle_bytes)),
            "2e5702b857570249e43ab92804c28cabf8194f8f03c1fd48f97c3bfd9c091fbf",
            "retained Transformers mel oracle changed"
        );
        let mut oracle_chunks = mel_oracle_bytes.chunks_exact(std::mem::size_of::<f32>());
        let mel_oracle = oracle_chunks
            .by_ref()
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32 chunk")))
            .collect::<Vec<_>>();
        assert!(oracle_chunks.remainder().is_empty());
        assert_eq!(mel_oracle.len(), mel_values.len());
        let mel_max_abs_diff = mel_values
            .iter()
            .zip(&mel_oracle)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        let mel_root_mean_square_diff = (mel_values
            .iter()
            .zip(&mel_oracle)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>()
            / mel_values.len() as f64)
            .sqrt();
        assert!(
            (mel_min - -0.702_159_9).abs() <= 1e-4,
            "Candle/Transformers mel minimum diverged: {mel_min}"
        );
        assert!(
            (mel_max - 1.297_840_1).abs() <= 1e-4,
            "Candle/Transformers mel maximum diverged: {mel_max}"
        );
        assert!(
            mel_max_abs_diff <= 1e-4,
            "Candle/Transformers complete mel tensor diverged: maxAbsDiff={mel_max_abs_diff}, rmsDiff={mel_root_mean_square_diff}"
        );

        let initial = session
            .initial_tokens(WhisperDecodeMode::WithoutTimestamps, &[])
            .expect("German no-timestamps prompt");
        assert_eq!(initial.token_ids, vec![50258, 50261, 50359, 50363]);
        let features = session
            .encode_window_batch(&windows, WhisperFeatureExtractorMode::Transformers)
            .expect("encoder features");
        let mut no_speech_probability = None;
        let mut stats = WhisperGenerationStats::default();
        let first_logits = session
            .configured_search_logits(
                &features,
                &initial.token_ids,
                initial.sot_position,
                &[],
                WhisperDecodeMode::WithoutTimestamps,
                &config.decode,
                &mut no_speech_probability,
                &mut stats,
            )
            .expect("first filtered logits");
        assert_eq!(
            argmax_finite(&first_logits),
            Some(2846),
            "first Candle token diverged from Transformers"
        );
        let logits_oracle_path =
            std::env::var_os("CANDLE_WHISPER_GERMAN_LOGITS_ORACLE")
                .map(PathBuf::from)
                .expect(
                    "CANDLE_WHISPER_GERMAN_LOGITS_ORACLE must point at the retained filtered Transformers f32le vector",
                );
        let logits_oracle_bytes =
            std::fs::read(logits_oracle_path).expect("retained Transformers logits oracle");
        assert_eq!(
            format!("{:x}", Sha256::digest(&logits_oracle_bytes)),
            "751015116dcded3c45314938e471679439abd6ddc08ee5509bc16a66cd540412",
            "retained Transformers filtered-logits oracle changed"
        );
        let mut logits_oracle_chunks = logits_oracle_bytes.chunks_exact(std::mem::size_of::<f32>());
        let logits_oracle = logits_oracle_chunks
            .by_ref()
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32 chunk")))
            .collect::<Vec<_>>();
        assert!(logits_oracle_chunks.remainder().is_empty());
        assert_eq!(logits_oracle.len(), first_logits.len());

        // Candle and PyTorch use different CPU kernels. Compare every filtered
        // vocabulary logit with a fixed numerical bound, and require the exact
        // suppression mask, greedy argmax, and complete generated sequence.
        let mut logits_max_abs_diff = 0.0_f32;
        for (token_id, (actual, expected)) in first_logits.iter().zip(&logits_oracle).enumerate() {
            if expected.is_finite() {
                assert!(
                    actual.is_finite(),
                    "Candle suppressed finite Transformers token {token_id}"
                );
                logits_max_abs_diff = logits_max_abs_diff.max((actual - expected).abs());
            } else {
                assert_eq!(
                    *actual,
                    f32::NEG_INFINITY,
                    "Candle/Transformers suppression mask diverged for token {token_id}"
                );
            }
        }
        assert!(
            logits_max_abs_diff <= 5e-2,
            "complete first-step Candle/Transformers logits diverged: maxAbsDiff={logits_max_abs_diff}"
        );

        let decoded = session
            .decode_tokens_configured(
                &features,
                WhisperDecodeMode::WithoutTimestamps,
                &config.decode,
            )
            .expect("greedy decode");
        assert_eq!(
            decoded.token_ids,
            vec![2846, 1418, 1979, 25641, 11197, 5199, 1109, 331, 4235, 13,]
        );
        let greedy_text =
            decode_text_tokens(&session.tokenizer, &decoded.token_ids).expect("greedy text");
        assert_eq!(greedy_text.trim(), "Das ist nicht dein Lieblingsverband.");

        // The exact greedy oracle above is the prerequisite for exercising the
        // bounded beam-size-five path on the same encoded German window.
        let mut beam_decode = config.decode.clone();
        beam_decode.search.beam_size = 5;
        let beam_token_bound = 1;
        session.generation.max_length = Some(initial.token_ids.len() + beam_token_bound);
        let beam = session
            .decode_tokens_configured(
                &features,
                WhisperDecodeMode::WithoutTimestamps,
                &beam_decode,
            )
            .expect("bounded beam-size-five decode");
        assert!(!beam.token_ids.is_empty(), "beam decode produced no tokens");
        assert!(
            beam.token_ids.len() <= beam_token_bound,
            "beam decode exceeded the explicit test token bound"
        );
        let beam_text = decode_text_tokens(&session.tokenizer, &beam.token_ids).expect("beam text");
        assert!(!beam_text.trim().is_empty(), "beam decode produced no text");
    }

    #[test]
    fn model_resolution_observer_can_cancel_before_resolution_work() {
        let mut observed = 0;

        let error = resolve_whisper_model_with_observer(
            &CandleWhisperOptions::default(),
            "openai/whisper-tiny",
            CandleWhisperComputeType::Fp32,
            &mut |_| {
                observed += 1;
                Err(media_core::DetectError::InvalidArgument(
                    "transcription cancelled at a model boundary".to_string(),
                ))
            },
        )
        .expect_err("observer cancellation should stop model resolution");

        assert_eq!(observed, 1);
        assert!(error.to_string().contains("cancelled"));
    }

    #[test]
    fn model_resolution_observer_reports_explicit_bundle_boundaries_in_order() {
        let explicit = tempfile::tempdir().unwrap();
        create_fake_whisper_bundle(explicit.path());
        let options = CandleWhisperOptions {
            model_bundle: Some(explicit.path().to_path_buf()),
            ..CandleWhisperOptions::default()
        };
        let mut events = Vec::new();

        resolve_whisper_model_with_observer(
            &options,
            "tiny.en",
            CandleWhisperComputeType::Fp32,
            &mut |event| {
                events.push(match event {
                    WhisperModelResolutionEvent::ResolutionStart => "resolution-start",
                    WhisperModelResolutionEvent::ResolutionEnd { source } => source,
                    WhisperModelResolutionEvent::DownloadStart => "download-start",
                    WhisperModelResolutionEvent::DownloadEnd { .. } => "download-end",
                    WhisperModelResolutionEvent::LoadStart => "load-start",
                    WhisperModelResolutionEvent::LoadEnd { .. } => "load-end",
                });
                Ok(())
            },
        )
        .expect("explicit bundle should resolve");

        assert_eq!(events, ["resolution-start", "explicit-bundle"]);
    }

    fn create_fake_whisper_bundle(root: &Path) {
        for file in CandleWhisperComputeType::Automatic.required_bundle_files() {
            std::fs::write(root.join(file), "").unwrap();
        }
    }

    fn create_q8_companion_files(root: &Path, vocab_size: usize) {
        std::fs::write(
            root.join("config.json"),
            serde_json::json!({
                "num_mel_bins": 2,
                "max_source_positions": 4,
                "d_model": 32,
                "encoder_attention_heads": 4,
                "encoder_layers": 1,
                "vocab_size": vocab_size,
                "max_target_positions": 8,
                "decoder_attention_heads": 4,
                "decoder_layers": 1,
                "suppress_tokens": []
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            root.join("generation_config.json"),
            serde_json::json!({
                "decoder_start_token_id": 1,
                "eos_token_id": 2,
                "no_timestamps_token_id": 7,
                "max_length": 8
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(root.join("preprocessor_config.json"), "{}").unwrap();
        test_tokenizer()
            .save(root.join("tokenizer.json"), false)
            .unwrap();
    }

    fn q8_test_tensor(shape: impl Into<candle_core::Shape>, dtype: GgmlDType) -> QTensor {
        let shape = shape.into();
        let values = (0..shape.elem_count())
            .map(|index| index as f32 * 0.001)
            .collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values, shape.clone(), &Device::Cpu).unwrap();
        QTensor::quantize(&tensor, dtype).unwrap()
    }

    fn write_q8_validation_gguf(path: &Path, dtype: GgmlDType) {
        let tensors = [
            (
                "model.decoder.embed_tokens.weight",
                q8_test_tensor((15, 32), dtype),
            ),
            (
                "model.encoder.conv1.weight",
                q8_test_tensor((32, 2, 3), GgmlDType::F32),
            ),
            (
                "model.encoder.conv2.weight",
                q8_test_tensor((32, 32, 3), GgmlDType::F32),
            ),
        ];
        let metadata = [
            (
                "general.architecture",
                gguf_file::Value::String("whisper".to_string()),
            ),
            ("general.file_type", gguf_file::Value::U32(7)),
        ];
        let metadata_refs = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        let tensor_refs = tensors
            .iter()
            .map(|(name, tensor)| (*name, tensor))
            .collect::<Vec<_>>();
        let mut cursor = Cursor::new(Vec::new());
        gguf_file::write(&mut cursor, &metadata_refs, &tensor_refs).unwrap();
        std::fs::write(path, cursor.into_inner()).unwrap();
    }

    fn minimal_asr_request(model_id: &str) -> AsrRequest {
        AsrRequest {
            audio: crate::LoadedAudio {
                samples: vec![0.0; 16_000],
                sample_rate: 16_000,
                channels: 1,
                source: None,
            },
            chunks: vec![SpeechActivitySegment::new(0.0, 1.0, 0.5).unwrap()],
            task: TranscriptionTask::Transcribe,
            language: Some("en".to_string()),
            model_id: model_id.to_string(),
        }
    }

    #[test]
    fn decoder_thread_pools_are_request_scoped_under_concurrency() {
        let environment_before = std::env::var_os("RAYON_NUM_THREADS");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let run = |decoder_threads| {
            let barrier = std::sync::Arc::clone(&barrier);
            std::thread::spawn(move || {
                with_decoder_threads(
                    Some(decoder_threads),
                    &ResolvedNativeDevice::Cpu,
                    move || {
                        barrier.wait();
                        Ok(rayon::current_num_threads())
                    },
                )
                .unwrap()
            })
        };

        let one_thread = run(1);
        let three_threads = run(3);
        barrier.wait();

        assert_eq!(one_thread.join().unwrap(), 1);
        assert_eq!(three_threads.join().unwrap(), 3);
        assert_eq!(std::env::var_os("RAYON_NUM_THREADS"), environment_before);
    }

    #[test]
    fn omitted_decoder_threads_preserve_the_callers_default_rayon_runtime() {
        let outer_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let observed = outer_pool.install(|| {
            with_decoder_threads(None, &ResolvedNativeDevice::Cpu, || {
                Ok(rayon::current_num_threads())
            })
        });

        assert_eq!(observed.unwrap(), 2);
    }

    #[test]
    fn decoder_thread_diagnostics_report_default_and_cpu_application() {
        assert_eq!(
            decoder_threads_diagnostic(
                &CandleWhisperRuntimeControls::default(),
                &ResolvedNativeDevice::Cpu,
            ),
            "default"
        );
        assert_eq!(
            decoder_threads_diagnostic(
                &CandleWhisperRuntimeControls {
                    decoder_threads: Some(3),
                    ..CandleWhisperRuntimeControls::default()
                },
                &ResolvedNativeDevice::Cpu,
            ),
            "3"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn decoder_thread_diagnostics_report_cuda_controls_as_ignored() {
        assert_eq!(
            decoder_threads_diagnostic(
                &CandleWhisperRuntimeControls {
                    decoder_threads: Some(3),
                    ..CandleWhisperRuntimeControls::default()
                },
                &ResolvedNativeDevice::Cuda(1),
            ),
            "ignored(cuda)"
        );
    }

    fn test_generation() -> GenerationConfig {
        GenerationConfig {
            decoder_start_token_id: Some(1),
            eos_token_id: Some(2),
            forced_decoder_ids: None,
            max_length: Some(8),
            lang_to_id: [("<|en|>".to_string(), 3), ("<|de|>".to_string(), 4)]
                .into_iter()
                .collect(),
            task_to_id: [("transcribe".to_string(), 5), ("translate".to_string(), 6)]
                .into_iter()
                .collect(),
            no_timestamps_token_id: Some(7),
            suppress_tokens: Vec::new(),
            begin_suppress_tokens: Vec::new(),
        }
    }

    fn test_tokenizer() -> Tokenizer {
        let temp = tempfile::tempdir().unwrap();
        let vocab = temp.path().join("vocab.json");
        std::fs::write(
            &vocab,
            serde_json::json!({
                "<unk>": 0,
                whisper::SOT_TOKEN: 1,
                whisper::EOT_TOKEN: 2,
                "<|en|>": 3,
                "<|de|>": 4,
                "<|transcribe|>": 5,
                "<|translate|>": 6,
                whisper::NO_TIMESTAMPS_TOKEN: 7,
                "123": 8,
                "room2": 9,
                "<|nospeech|>": 10,
                "<|startofprev|>": 11,
                "hello": 12,
                "world": 13,
                "again": 14
            })
            .to_string(),
        )
        .unwrap();
        let model = WordLevel::from_file(vocab.to_str().unwrap(), "<unk>".to_string()).unwrap();
        Tokenizer::new(model)
    }

    fn timestamp_test_tokenizer() -> Tokenizer {
        let temp = tempfile::tempdir().unwrap();
        let vocab = temp.path().join("vocab.json");
        std::fs::write(
            &vocab,
            serde_json::json!({
                "<unk>": 0,
                "hello": 10,
                "world": 11,
                "again": 12,
                "123": 13,
                "<|0.00|>": 100,
                "<|1.00|>": 150,
                "<|2.00|>": 200,
                "<|3.00|>": 250,
                "<|30.00|>": 1600
            })
            .to_string(),
        )
        .unwrap();
        let model = WordLevel::from_file(vocab.to_str().unwrap(), "<unk>".to_string()).unwrap();
        Tokenizer::new(model)
    }

    fn decoded_segment(text: &str, start_seconds: f64, end_seconds: f64) -> WhisperDecodedSegment {
        WhisperDecodedSegment {
            text: text.to_string(),
            start_seconds,
            end_seconds,
            token_ids: Vec::new(),
        }
    }

    fn assert_approx_eq(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn autoregressive_row_prefills_prompt_then_steps_with_last_token_position() {
        let mut row = WhisperAutoregressiveRow::new(vec![1, 3, 5, 7]);

        let prefill = row.next_decoder_input();
        assert_eq!(prefill.token_ids, vec![1, 3, 5, 7]);
        assert_eq!(prefill.position_offset, 0);
        assert!(prefill.flush_cache);
        assert_eq!(prefill.kind, WhisperDecoderInputKind::PromptPrefill);

        row.mark_forwarded();
        row.accept(42);
        let step = row.next_decoder_input();

        assert_eq!(step.token_ids, vec![42]);
        assert_eq!(step.position_offset, 4);
        assert!(!step.flush_cache);
        assert_eq!(step.kind, WhisperDecoderInputKind::CachedTokenStep);
        assert_eq!(row.generated_tokens(), &[42]);
    }

    #[test]
    fn request_prompt_prefill_puts_start_of_prev_and_text_before_whisper_controls() {
        let generation = test_generation();
        let tokenizer = test_tokenizer();
        let prompt = CandleWhisperSession::build_initial_tokens(
            &generation,
            &tokenizer,
            Some("en"),
            TranscriptionTask::Transcribe,
            WhisperDecodeMode::WithoutTimestamps,
            &[12, 13],
            8,
        )
        .unwrap();

        assert_eq!(prompt.token_ids, vec![11, 12, 13, 1, 3, 5, 7]);
        assert_eq!(prompt.sot_position, 3);
    }

    #[test]
    fn request_prompt_prefill_truncates_the_oldest_tokens_to_context_budget() {
        let prompt = CandleWhisperSession::build_initial_tokens(
            &test_generation(),
            &test_tokenizer(),
            Some("en"),
            TranscriptionTask::Transcribe,
            WhisperDecodeMode::WithoutTimestamps,
            &[12, 13, 14],
            3,
        )
        .unwrap();

        assert_eq!(prompt.token_ids, vec![11, 13, 14, 1, 3, 5, 7]);
        assert_eq!(prompt.sot_position, 3);
    }

    #[test]
    fn request_prompt_prefill_rejects_token_ids_outside_the_vocabulary() {
        let error = CandleWhisperSession::build_initial_tokens(
            &test_generation(),
            &test_tokenizer(),
            Some("en"),
            TranscriptionTask::Transcribe,
            WhisperDecodeMode::WithoutTimestamps,
            &[999],
            8,
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("not in the tokenizer vocabulary"));
    }

    #[test]
    fn no_speech_probability_reads_the_sot_position_not_the_last_prompt_position() {
        let prompt = CandleWhisperSession::build_initial_tokens(
            &test_generation(),
            &test_tokenizer(),
            Some("en"),
            TranscriptionTask::Transcribe,
            WhisperDecodeMode::WithoutTimestamps,
            &[12, 13],
            8,
        )
        .unwrap();
        let mut values = vec![0.0_f32; prompt.token_ids.len() * 15];
        values[prompt.sot_position * 15 + 10] = 8.0;
        values[(prompt.token_ids.len() - 1) * 15 + 10] = -8.0;
        let logits =
            Tensor::from_vec(values, (1, prompt.token_ids.len(), 15), &Device::Cpu).unwrap();

        let probability = tensor_token_probability_at_position(&logits, 0, prompt.sot_position, 10)
            .unwrap()
            .unwrap();
        assert!(probability > 0.99);
    }

    #[test]
    fn request_prompt_state_carries_previous_text_and_truncates_to_the_boundary() {
        let config = CandleWhisperDecodeRequestConfig {
            initial_prompt_tokens: vec![90, 91],
            condition_on_previous_text: true,
            ..CandleWhisperDecodeRequestConfig::default()
        };
        let mut state = WhisperRequestPromptState::new(&config);
        state.record_generated_tokens(&[1, 2, 3, 4], 5);

        assert_eq!(state.current_prompt_tokens(5), vec![90, 91, 2, 3, 4]);

        let independent_request = WhisperRequestPromptState::new(&config);
        assert_eq!(independent_request.current_prompt_tokens(5), vec![90, 91]);
    }

    #[test]
    fn request_suppression_combines_explicit_ids_and_tokenizer_aware_numerals() {
        let tokenizer = timestamp_test_tokenizer();
        let config = CandleWhisperDecodeRequestConfig {
            suppressed_token_ids: vec![11],
            suppress_numerals: true,
            ..CandleWhisperDecodeRequestConfig::default()
        };

        let suppressed = request_suppressed_token_ids(&tokenizer, &config);
        assert!(suppressed.contains(&11));
        assert!(suppressed.contains(&13));
        assert!(
            !suppressed.contains(&150),
            "timestamp controls remain available"
        );
    }

    #[test]
    fn no_speech_threshold_rejects_before_temperature_retry() {
        let config = CandleWhisperDecodeRequestConfig {
            max_no_speech_probability: Some(0.6),
            min_average_log_probability: Some(-1.0),
            ..CandleWhisperDecodeRequestConfig::default()
        };

        let decision = fallback_attempt_decision(&config, -5.0, Some(0.8), 1.0);
        let mut tokens = vec![10, 11];
        assert!(apply_no_speech_rejection(decision, &mut tokens));
        assert!(tokens.is_empty());
    }

    #[test]
    fn high_no_speech_probability_accepts_high_confidence_text() {
        let config = CandleWhisperDecodeRequestConfig {
            max_no_speech_probability: Some(0.6),
            min_average_log_probability: Some(-1.0),
            ..CandleWhisperDecodeRequestConfig::default()
        };

        assert_eq!(
            fallback_attempt_decision(&config, -0.2, Some(0.8), 1.0),
            WhisperFallbackAttemptDecision::Accept
        );
    }

    #[test]
    fn threshold_fallback_visits_temperatures_in_declared_order_until_acceptance() {
        let observations = [
            WhisperFallbackAttemptDecision::Retry,
            WhisperFallbackAttemptDecision::Retry,
            WhisperFallbackAttemptDecision::Accept,
        ];
        let mut called = Vec::new();
        let (selected, attempted) = run_ordered_temperature_fallback(
            &[0.0, 0.4, 0.8, 1.0],
            |index, temperature| {
                called.push(temperature);
                Ok(observations[index])
            },
            |decision| *decision == WhisperFallbackAttemptDecision::Retry,
        )
        .unwrap();

        assert_eq!(selected, WhisperFallbackAttemptDecision::Accept);
        assert_eq!(called, vec![0.0, 0.4, 0.8]);
        assert_eq!(attempted, vec![0.0, 0.4, 0.8]);
    }

    #[test]
    fn compression_ratio_is_zero_for_empty_output_and_detects_repetition() {
        assert_eq!(text_compression_ratio("").unwrap(), 0.0);
        let repeated = "the quick brown fox jumps over the lazy dog ".repeat(20);
        assert!(text_compression_ratio(&repeated).unwrap() > 2.4);
        assert!(text_compression_ratio("ordinary short text").unwrap() < 2.4);
    }

    #[test]
    fn default_request_config_preserves_the_exact_legacy_greedy_path() {
        assert!(CandleWhisperDecodeRequestConfig::default().preserves_legacy_greedy_path());
    }

    #[test]
    fn generation_stats_report_actual_decoder_cache_reuse() {
        let mut stats = WhisperGenerationStats::default();
        let prefill = WhisperDecoderInput {
            token_ids: vec![1, 3, 5, 7],
            position_offset: 0,
            flush_cache: true,
            kind: WhisperDecoderInputKind::PromptPrefill,
        };
        let step = WhisperDecoderInput {
            token_ids: vec![42],
            position_offset: 4,
            flush_cache: false,
            kind: WhisperDecoderInputKind::CachedTokenStep,
        };
        stats.record_input(&prefill);
        stats.record_input(&step);
        stats.record_generated_token();
        stats.record_active_row_batch_size(3);
        stats.record_active_row_compaction();
        stats.record_completed_row();
        stats.record_decoder_stats(CachedWhisperDecoderStats {
            self_attention_cache_reused: true,
            cross_attention_cache_reused: true,
        });

        let mut diagnostics = WhisperDecodeDiagnostics::default();
        stats.extend(&mut diagnostics);

        assert_eq!(diagnostics.decoder_prompt_prefill_count, 1);
        assert_eq!(diagnostics.decoder_cached_token_step_count, 1);
        assert_eq!(diagnostics.decoder_input_token_count, 5);
        assert_eq!(diagnostics.generated_token_count, 1);
        assert_eq!(diagnostics.decoder_completed_row_count, 1);
        assert_eq!(diagnostics.decoder_max_active_row_batch_size, 3);
        assert_eq!(diagnostics.decoder_effective_active_batch_sizes, vec![3]);
        assert_eq!(diagnostics.decoder_active_row_compaction_count, 1);
        assert!(diagnostics.decoder_self_attention_cache_reused);
        assert!(diagnostics.decoder_cross_attention_cache_reused);
    }

    #[test]
    fn observed_batch_execution_requires_real_multi_row_decoder_call() {
        assert_eq!(
            observed_candle_batch_execution(CandleWhisperDecodeRuntime::ActiveRowTensorBatch, 3),
            crate::CANDLE_WHISPER_ACTIVE_ROW_TENSOR_BATCH_EXECUTION
        );
        assert_eq!(
            observed_candle_batch_execution(CandleWhisperDecodeRuntime::ActiveRowTensorBatch, 1),
            crate::CANDLE_WHISPER_AUTOREGRESSIVE_KV_CACHE_EXECUTION
        );
        assert_eq!(
            observed_candle_batch_execution(CandleWhisperDecodeRuntime::AutoregressiveKvCache, 3),
            crate::CANDLE_WHISPER_AUTOREGRESSIVE_KV_CACHE_EXECUTION
        );
    }

    #[test]
    fn encoder_microbatching_is_cuda_only_for_multi_window_batches() {
        assert!(!should_microbatch_encoder(&ResolvedNativeDevice::Cpu, 1));
        assert!(!should_microbatch_encoder(&ResolvedNativeDevice::Cpu, 2));

        #[cfg(feature = "cuda")]
        {
            assert!(!should_microbatch_encoder(
                &ResolvedNativeDevice::Cuda(0),
                1
            ));
            assert!(should_microbatch_encoder(&ResolvedNativeDevice::Cuda(0), 2));
        }
    }

    #[test]
    fn effective_active_batch_sizes_are_sorted_and_deduplicated() {
        assert_eq!(format_effective_active_batch_sizes(&[]), "none");
        assert_eq!(
            format_effective_active_batch_sizes(&[3, 2, 3, 1, 2]),
            "1,2,3"
        );
    }

    #[test]
    fn cache_reuse_diagnostic_names_observed_cache_modes() {
        assert_eq!(format_cache_reuse(true, true), "self-and-cross-attention");
        assert_eq!(format_cache_reuse(true, false), "self-attention");
        assert_eq!(format_cache_reuse(false, true), "cross-attention");
        assert_eq!(format_cache_reuse(false, false), "none");
    }

    #[test]
    fn active_row_decisions_compact_finished_rows_and_preserve_original_order() {
        let eos = 2;
        let rows = (0..3)
            .map(|original_index| ActiveWhisperDecodeRow {
                original_index,
                row: WhisperAutoregressiveRow::new(vec![1]),
                stats: WhisperGenerationStats::default(),
                score: 0.0,
                no_speech_probability: None,
            })
            .collect::<Vec<_>>();
        let mut completed = vec![None, None, None];

        let (survivors, survivor_indices) = apply_active_row_decisions(
            rows.into_iter().zip([10, eos, 12]).collect(),
            eos,
            &mut completed,
        )
        .unwrap();
        assert_eq!(survivor_indices, vec![0, 2]);
        assert_eq!(
            survivors
                .iter()
                .map(|row| row.original_index)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(completed[1].as_ref().unwrap().token_ids, Vec::<u32>::new());

        let (survivors, survivor_indices) = apply_active_row_decisions(
            survivors.into_iter().zip([eos, 14]).collect(),
            eos,
            &mut completed,
        )
        .unwrap();
        assert_eq!(survivor_indices, vec![1]);
        assert_eq!(survivors[0].original_index, 2);
        assert_eq!(completed[0].as_ref().unwrap().token_ids, vec![10]);

        let (survivors, survivor_indices) = apply_active_row_decisions(
            survivors.into_iter().zip([eos]).collect(),
            eos,
            &mut completed,
        )
        .unwrap();
        assert!(survivors.is_empty());
        assert!(survivor_indices.is_empty());

        let completed_tokens = completed
            .into_iter()
            .map(|row| row.unwrap().token_ids)
            .collect::<Vec<_>>();
        assert_eq!(completed_tokens, vec![vec![10], vec![], vec![12, 14]]);
    }

    #[test]
    fn active_row_decisions_keep_per_row_generation_stats_isolated() {
        let eos = 2;
        let mut rows = (0..2)
            .map(|original_index| ActiveWhisperDecodeRow {
                original_index,
                row: WhisperAutoregressiveRow::new(vec![1]),
                stats: WhisperGenerationStats::default(),
                score: 0.0,
                no_speech_probability: None,
            })
            .collect::<Vec<_>>();
        rows[0].stats.decoder_input_token_count = 3;
        rows[1].stats.decoder_input_token_count = 7;
        let mut completed = vec![None, None];

        let (survivors, _) = apply_active_row_decisions(
            rows.into_iter().zip([42, eos]).collect(),
            eos,
            &mut completed,
        )
        .unwrap();
        assert_eq!(survivors[0].stats.decoder_input_token_count, 3);
        assert_eq!(survivors[0].stats.generated_token_count, 1);
        assert_eq!(
            completed[1]
                .as_ref()
                .unwrap()
                .stats
                .decoder_input_token_count,
            7
        );
        assert_eq!(
            completed[1].as_ref().unwrap().stats.generated_token_count,
            0
        );
    }

    #[test]
    fn fallback_diagnostics_keep_timestamp_state_and_retry_generation_counts() {
        let mut fallback = WhisperDecodeDiagnostics {
            decoder_prompt_prefill_count: 1,
            decoder_cached_token_step_count: 2,
            decoder_input_token_count: 6,
            generated_token_count: 2,
            decoder_self_attention_cache_reused: true,
            decoder_cross_attention_cache_reused: true,
            ..WhisperDecodeDiagnostics::default()
        };
        let timestamp_attempt = WhisperDecodeDiagnostics {
            timestamp_tokens_requested: true,
            timestamp_tokens_present: true,
            decoded_token_ids: vec![100, 10, 150],
            decoder_prompt_prefill_count: 1,
            decoder_cached_token_step_count: 1,
            decoder_input_token_count: 5,
            generated_token_count: 1,
            decoder_completed_row_count: 0,
            decoder_max_active_row_batch_size: 2,
            decoder_effective_active_batch_sizes: vec![2],
            decoder_active_row_compaction_count: 0,
            decoder_self_attention_cache_reused: true,
            decoder_cross_attention_cache_reused: true,
            ..WhisperDecodeDiagnostics::default()
        };

        fallback.add_generation_counts_from(&timestamp_attempt);
        fallback.timestamp_tokens_requested = timestamp_attempt.timestamp_tokens_requested;
        fallback.timestamp_tokens_present = timestamp_attempt.timestamp_tokens_present;
        fallback.decoded_token_ids = timestamp_attempt.decoded_token_ids;

        assert!(fallback.timestamp_tokens_requested);
        assert!(fallback.timestamp_tokens_present);
        assert_eq!(fallback.decoded_token_ids, vec![100, 10, 150]);
        assert_eq!(fallback.decoder_prompt_prefill_count, 2);
        assert_eq!(fallback.decoder_cached_token_step_count, 3);
        assert_eq!(fallback.decoder_input_token_count, 11);
        assert_eq!(fallback.generated_token_count, 3);
        assert_eq!(fallback.decoder_completed_row_count, 0);
        assert_eq!(fallback.decoder_max_active_row_batch_size, 2);
        assert_eq!(fallback.decoder_effective_active_batch_sizes, vec![2]);
        assert_eq!(fallback.decoder_active_row_compaction_count, 0);
        assert!(fallback.decoder_self_attention_cache_reused);
        assert!(fallback.decoder_cross_attention_cache_reused);
    }

    #[test]
    fn initial_prompt_uses_requested_language_and_transcribe_task() {
        let tokens = CandleWhisperSession::initial_prompt_tokens(
            &test_generation(),
            &test_tokenizer(),
            Some("en"),
        )
        .unwrap();
        assert_eq!(tokens, vec![1, 3, 5, 7]);
        assert!(!tokens.contains(&6));
    }

    #[test]
    fn initial_prompt_uses_requested_language_and_translate_task() {
        let tokens = CandleWhisperSession::initial_prompt_tokens_for_task(
            &test_generation(),
            &test_tokenizer(),
            Some("en"),
            TranscriptionTask::Translate,
        )
        .unwrap();
        assert_eq!(tokens, vec![1, 3, 6, 7]);
        assert!(!tokens.contains(&5));
    }

    #[test]
    fn initial_prompt_uses_option_language_when_request_language_absent() {
        let setup = WhisperRunSetup {
            model_id: "openai/whisper-tiny".to_string(),
            task: TranscriptionTask::Transcribe,
            language: Some("de".to_string()),
            bundle: WhisperBundlePaths {
                root: PathBuf::from("bundle"),
                config_json: PathBuf::from("config.json"),
                generation_config_json: PathBuf::from("generation_config.json"),
                tokenizer_json: PathBuf::from("tokenizer.json"),
                preprocessor_config_json: PathBuf::from("preprocessor_config.json"),
                model_safetensors: PathBuf::from("model.safetensors"),
                model_q8_0_gguf: None,
            },
            model_source: "explicit-bundle",
            resolved_device: ResolvedNativeDevice::Cpu,
            requested_compute_type: CandleWhisperComputeType::Automatic,
            resolved_compute_type: CandleWhisperComputeType::Fp32,
            model_weight_dtype: DType::F32,
            model_format: WhisperModelFormat::Safetensors,
        };
        let tokens = CandleWhisperSession::initial_prompt_tokens(
            &test_generation(),
            &test_tokenizer(),
            setup.language.as_deref(),
        )
        .unwrap();
        assert_eq!(tokens[1], 4);
    }

    #[test]
    fn request_language_wins_before_prompt_construction() {
        let request = AsrRequest {
            audio: crate::LoadedAudio {
                samples: vec![0.0; 16_000],
                sample_rate: 16_000,
                channels: 1,
                source: None,
            },
            chunks: vec![SpeechActivitySegment::new(0.0, 1.0, 0.5).unwrap()],
            task: TranscriptionTask::Transcribe,
            language: Some("en".to_string()),
            model_id: "openai/whisper-tiny".to_string(),
        };
        let options = CandleWhisperOptions {
            language: Some("de".to_string()),
            model_bundle: Some(PathBuf::from("missing")),
            ..CandleWhisperOptions::default()
        };
        let language = request
            .language
            .clone()
            .or_else(|| options.language.clone())
            .unwrap();
        assert_eq!(language, "en");
    }

    #[test]
    fn whisper_aliases_canonicalize_to_hugging_face_ids() {
        assert_eq!(
            canonical_whisper_model_id("small").unwrap(),
            "openai/whisper-small"
        );
        assert_eq!(
            canonical_whisper_model_id("tiny.en").unwrap(),
            "openai/whisper-tiny.en"
        );
        assert_eq!(
            canonical_whisper_model_id("large").unwrap(),
            "openai/whisper-large-v3"
        );
        assert_eq!(
            canonical_whisper_model_id("openai/whisper-small").unwrap(),
            "openai/whisper-small"
        );
        let error = canonical_whisper_model_id("unknown")
            .unwrap_err()
            .to_string();
        assert!(error.contains("setup_error"));
        assert!(error.contains("unsupported native Candle Whisper model alias"));
    }

    #[test]
    fn whisper_bundle_priority_wins_over_model_dir() {
        let explicit = tempfile::tempdir().unwrap();
        create_fake_whisper_bundle(explicit.path());
        let cache = tempfile::tempdir().unwrap();
        let options = CandleWhisperOptions {
            model_id: "tiny.en".to_string(),
            model_bundle: Some(explicit.path().to_path_buf()),
            model_dir: Some(cache.path().to_path_buf()),
            model_cache_only: true,
            ..CandleWhisperOptions::default()
        };
        let resolved = resolve_whisper_model(&options, "tiny.en").unwrap();
        assert_eq!(resolved.source, "explicit-bundle");
        assert_eq!(resolved.model_id, "openai/whisper-tiny.en");
        assert_eq!(resolved.bundle.root, explicit.path());
    }

    #[cfg(feature = "model-bundles")]
    #[test]
    fn whisper_cache_only_resolves_fake_hf_snapshot() {
        let temp = tempfile::tempdir().unwrap();
        let snapshot = temp
            .path()
            .join("models--openai--whisper-tiny.en/snapshots/abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        create_fake_whisper_bundle(&snapshot);
        let options = CandleWhisperOptions {
            model_dir: Some(temp.path().to_path_buf()),
            model_cache_only: true,
            ..CandleWhisperOptions::default()
        };
        let resolved = resolve_whisper_model(&options, "tiny.en").unwrap();
        assert_eq!(resolved.source, "hugging-face-cache");
        assert_eq!(resolved.model_id, "openai/whisper-tiny.en");
        assert_eq!(resolved.bundle.root, snapshot);
    }

    #[cfg(feature = "model-bundles")]
    #[test]
    fn whisper_cache_only_missing_model_reports_required_files() {
        let temp = tempfile::tempdir().unwrap();
        let options = CandleWhisperOptions {
            model_dir: Some(temp.path().to_path_buf()),
            model_cache_only: true,
            ..CandleWhisperOptions::default()
        };
        let error = resolve_whisper_model(&options, "tiny.en")
            .unwrap_err()
            .to_string();
        assert!(error.contains("setup_error"));
        assert!(error.contains("openai/whisper-tiny.en"));
        assert!(error.contains("config.json"));
        assert!(error.contains("generation_config.json"));
        assert!(error.contains("tokenizer.json"));
        assert!(error.contains("preprocessor_config.json"));
        assert!(error.contains("model.safetensors"));
        assert!(error.contains("cache-only=true"));
    }

    #[cfg(feature = "model-bundles")]
    #[test]
    fn whisper_model_spec_requests_required_candle_files() {
        let spec = whisper_model_spec("openai/whisper-tiny.en");
        assert_eq!(spec.repo_id_value(), Some("openai/whisper-tiny.en"));
        let rendered = format!("{:?}", spec.files);
        for file in CandleWhisperComputeType::Automatic.required_bundle_files() {
            assert!(rendered.contains(file));
        }
    }

    #[test]
    fn whisper_setup_reports_model_resolution_diagnostics() {
        let explicit = tempfile::tempdir().unwrap();
        create_fake_whisper_bundle(explicit.path());
        let options = CandleWhisperOptions {
            model_bundle: Some(explicit.path().to_path_buf()),
            ..CandleWhisperOptions::default()
        };
        let setup =
            WhisperRunSetup::from_options_and_request(&options, &minimal_asr_request("tiny.en"))
                .unwrap();
        let diagnostics = whisper_setup_diagnostics(&setup);
        assert!(diagnostics
            .iter()
            .any(|item| item == "asrModelSource=explicit-bundle"));
        assert!(diagnostics
            .iter()
            .any(|item| item == "asrModelId=openai/whisper-tiny.en"));
        assert!(diagnostics
            .iter()
            .any(|item| item.starts_with("asrModelResolved=")));
        assert!(diagnostics
            .iter()
            .any(|item| item == "requestedComputeType=automatic"));
        assert!(diagnostics
            .iter()
            .any(|item| item == "resolvedComputeType=fp32"));
        assert!(diagnostics
            .iter()
            .any(|item| item == "modelWeightDtype=f32"));
    }

    #[test]
    fn int8_resolves_only_for_cpu_with_typed_cuda_guidance() {
        assert_eq!(
            CandleWhisperComputeType::Int8
                .resolve_for_device(false)
                .unwrap(),
            CandleWhisperComputeType::Int8
        );
        let error = CandleWhisperComputeType::Int8
            .resolve_for_device(true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("setup_error"));
        assert!(error.contains("CPU-only"));
        assert!(error.contains("device=cpu"));

        let setup_error = WhisperRunSetup::from_options_and_request(
            &CandleWhisperOptions {
                compute_type: CandleWhisperComputeType::Int8,
                device: crate::NativeDevicePreference::Cuda,
                ..CandleWhisperOptions::default()
            },
            &minimal_asr_request("tiny.en"),
        )
        .unwrap_err()
        .to_string();
        assert!(setup_error.contains("setup_error"));
        assert!(setup_error.contains("CPU-only"));
        assert!(setup_error.contains("device=cpu"));
    }

    #[test]
    fn int8_requires_q8_gguf_and_never_uses_safetensors_fallback() {
        let bundle = tempfile::tempdir().unwrap();
        create_q8_companion_files(bundle.path(), 15);
        std::fs::write(bundle.path().join("model.safetensors"), b"not used").unwrap();
        let options = CandleWhisperOptions {
            compute_type: CandleWhisperComputeType::Int8,
            model_bundle: Some(bundle.path().to_path_buf()),
            device: crate::NativeDevicePreference::Cpu,
            ..CandleWhisperOptions::default()
        };

        let error = resolve_whisper_model(&options, "tiny.en")
            .unwrap_err()
            .to_string();
        assert!(error.contains("model.q8_0.gguf"));
        assert!(!error.contains("failed to load Candle Whisper weights"));
    }

    #[test]
    fn int8_without_an_explicit_bundle_reports_the_public_required_file_list() {
        let options = CandleWhisperOptions {
            compute_type: CandleWhisperComputeType::Int8,
            device: crate::NativeDevicePreference::Cpu,
            ..CandleWhisperOptions::default()
        };

        let error = resolve_whisper_model(&options, "tiny.en")
            .unwrap_err()
            .to_string();
        for file in CandleWhisperComputeType::Int8.required_bundle_files() {
            assert!(error.contains(file), "missing `{file}` in `{error}`");
        }
    }

    #[test]
    fn int8_missing_explicit_bundle_reports_the_public_required_file_list() {
        let bundle = tempfile::tempdir().unwrap().path().join("missing");
        let error = resolve_q8_whisper_bundle_paths(&bundle)
            .unwrap_err()
            .to_string();

        for file in CandleWhisperComputeType::Int8.required_bundle_files() {
            assert!(error.contains(file), "missing `{file}` in `{error}`");
        }
    }

    #[test]
    fn int8_rejects_invalid_gguf_before_asr() {
        let bundle = tempfile::tempdir().unwrap();
        create_q8_companion_files(bundle.path(), 15);
        std::fs::write(bundle.path().join("model.q8_0.gguf"), b"not-a-gguf").unwrap();
        let options = CandleWhisperOptions {
            compute_type: CandleWhisperComputeType::Int8,
            model_bundle: Some(bundle.path().to_path_buf()),
            device: crate::NativeDevicePreference::Cpu,
            ..CandleWhisperOptions::default()
        };

        let error = resolve_whisper_model(&options, "tiny.en")
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid Q8 Whisper GGUF"));
    }

    #[test]
    fn int8_rejects_non_q8_tensor_quantization_before_asr() {
        let bundle = tempfile::tempdir().unwrap();
        create_q8_companion_files(bundle.path(), 15);
        write_q8_validation_gguf(&bundle.path().join("model.q8_0.gguf"), GgmlDType::F32);
        let options = CandleWhisperOptions {
            compute_type: CandleWhisperComputeType::Int8,
            model_bundle: Some(bundle.path().to_path_buf()),
            device: crate::NativeDevicePreference::Cpu,
            ..CandleWhisperOptions::default()
        };

        let error = resolve_whisper_model(&options, "tiny.en")
            .unwrap_err()
            .to_string();
        assert!(error.contains("must use Q8_0"));
    }

    #[test]
    fn int8_rejects_incompatible_companion_dimensions_before_asr() {
        let bundle = tempfile::tempdir().unwrap();
        create_q8_companion_files(bundle.path(), 14);
        write_q8_validation_gguf(&bundle.path().join("model.q8_0.gguf"), GgmlDType::Q8_0);
        let options = CandleWhisperOptions {
            compute_type: CandleWhisperComputeType::Int8,
            model_bundle: Some(bundle.path().to_path_buf()),
            device: crate::NativeDevicePreference::Cpu,
            ..CandleWhisperOptions::default()
        };

        let error = resolve_whisper_model(&options, "tiny.en")
            .unwrap_err()
            .to_string();
        assert!(error.contains("tokenizer vocabulary size"));
        assert!(error.contains("config vocab_size"));
    }

    #[test]
    fn q8_setup_diagnostics_report_compute_format_and_cache_contract() {
        let setup = WhisperRunSetup {
            model_id: "openai/whisper-tiny".to_string(),
            task: TranscriptionTask::Transcribe,
            language: Some("en".to_string()),
            bundle: WhisperBundlePaths {
                root: PathBuf::from("bundle"),
                config_json: PathBuf::from("config.json"),
                generation_config_json: PathBuf::from("generation_config.json"),
                tokenizer_json: PathBuf::from("tokenizer.json"),
                preprocessor_config_json: PathBuf::from("preprocessor_config.json"),
                model_safetensors: PathBuf::from("model.safetensors"),
                model_q8_0_gguf: Some(PathBuf::from("model.q8_0.gguf")),
            },
            model_source: "explicit-bundle",
            resolved_device: ResolvedNativeDevice::Cpu,
            requested_compute_type: CandleWhisperComputeType::Int8,
            resolved_compute_type: CandleWhisperComputeType::Int8,
            model_weight_dtype: DType::F32,
            model_format: WhisperModelFormat::GgufQ8_0,
        };
        let diagnostics = whisper_setup_diagnostics(&setup);
        assert!(diagnostics.iter().any(|item| item == "computeType=int8"));
        assert!(diagnostics
            .iter()
            .any(|item| item == "modelFormat=gguf-q8_0"));
        assert_eq!(format_cache_reuse(true, true), "self-and-cross-attention");
    }

    #[test]
    fn whisper_setup_resolves_cuda_automatic_to_fp16_weights_without_cuda_runtime() {
        assert_eq!(
            candle_whisper_model_weight_dtype(
                CandleWhisperComputeType::Automatic
                    .resolve_for_device(true)
                    .unwrap()
            ),
            DType::F16
        );
    }

    #[test]
    fn whisper_setup_resolves_explicit_fp32_to_fp32_weights_on_cuda() {
        assert_eq!(
            candle_whisper_model_weight_dtype(
                CandleWhisperComputeType::Fp32
                    .resolve_for_device(true)
                    .unwrap()
            ),
            DType::F32
        );
    }

    #[test]
    fn nullable_forced_decoder_ids_are_skipped() {
        let mut generation = test_generation();
        generation.forced_decoder_ids = Some(vec![(1, None), (2, Some(5)), (3, Some(7))]);
        let tokens =
            CandleWhisperSession::initial_prompt_tokens(&generation, &test_tokenizer(), Some("en"))
                .unwrap();
        assert_eq!(tokens, vec![1, 3, 5, 7]);
    }

    #[test]
    fn timestamp_prompt_omits_no_timestamps_token() {
        let mut generation = test_generation();
        generation.forced_decoder_ids = Some(vec![(3, Some(7))]);
        let tokens = CandleWhisperSession::initial_prompt_tokens_for_mode(
            &generation,
            &test_tokenizer(),
            Some("en"),
            TranscriptionTask::Transcribe,
            WhisperDecodeMode::TimestampTokens,
        )
        .unwrap();
        assert_eq!(tokens, vec![1, 3, 5]);
        assert!(!tokens.contains(&7));
    }

    #[test]
    fn no_timestamps_prompt_keeps_forced_no_timestamps_token() {
        let mut generation = test_generation();
        generation.forced_decoder_ids = Some(vec![(3, Some(7))]);
        let tokens = CandleWhisperSession::initial_prompt_tokens_for_mode(
            &generation,
            &test_tokenizer(),
            Some("en"),
            TranscriptionTask::Transcribe,
            WhisperDecodeMode::WithoutTimestamps,
        )
        .unwrap();
        assert_eq!(tokens, vec![1, 3, 5, 7]);
    }

    #[test]
    fn timestamp_token_detection_uses_whisper_zero_token() {
        let spec = whisper_timestamp_spec(&timestamp_test_tokenizer()).unwrap();
        assert_eq!(spec.begin_token_id, 100);
        assert_eq!(spec.end_token_id, 1601);
        assert_eq!(timestamp_seconds(150, &spec), Some(1.0));
        assert_eq!(timestamp_seconds(99, &spec), None);
    }

    #[test]
    fn missing_timestamp_metadata_returns_invalid_request() {
        let error = whisper_timestamp_spec(&test_tokenizer())
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid_request"));
        assert!(error.contains("<|0.00|>"));
    }

    #[test]
    fn auto_timing_allows_missing_timestamp_metadata() {
        let spec = timestamp_spec_for_timing_mode(&test_tokenizer(), CandleWhisperTimingMode::Auto)
            .unwrap();
        assert!(spec.is_none());
    }

    #[test]
    fn required_timing_rejects_missing_timestamp_metadata() {
        let error = timestamp_spec_for_timing_mode(
            &test_tokenizer(),
            CandleWhisperTimingMode::TimestampTokensRequired,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("invalid_request"));
        assert!(error.contains("<|0.00|>"));
    }

    #[test]
    fn no_timestamps_timing_does_not_require_timestamp_metadata() {
        let spec = timestamp_spec_for_timing_mode(
            &test_tokenizer(),
            CandleWhisperTimingMode::NoTimestamps,
        )
        .unwrap();
        assert!(spec.is_none());
    }

    #[test]
    fn timestamp_decode_reads_one_bounded_segment() {
        let tokenizer = timestamp_test_tokenizer();
        let decoded = decode_timestamp_window(&tokenizer, &[100, 10, 11, 150])
            .unwrap()
            .unwrap();
        assert_eq!(decoded.text, "hello world");
        assert_eq!(decoded.segments.len(), 1);
        assert_eq!(decoded.segments[0].text, "hello world");
        assert_eq!(decoded.segments[0].start_seconds, 0.0);
        assert_eq!(decoded.segments[0].end_seconds, 1.0);
        assert_eq!(decoded.segments[0].token_ids, vec![10, 11]);
    }

    #[test]
    fn timestamp_decode_reads_multiple_bounded_segments() {
        let tokenizer = timestamp_test_tokenizer();
        let decoded = decode_timestamp_window(&tokenizer, &[100, 10, 150, 150, 11, 200])
            .unwrap()
            .unwrap();
        assert_eq!(decoded.text, "hello world");
        assert_eq!(decoded.segments.len(), 2);
        assert_eq!(decoded.segments[0].text, "hello");
        assert_eq!(decoded.segments[0].start_seconds, 0.0);
        assert_eq!(decoded.segments[0].end_seconds, 1.0);
        assert_eq!(decoded.segments[1].text, "world");
        assert_eq!(decoded.segments[1].start_seconds, 1.0);
        assert_eq!(decoded.segments[1].end_seconds, 2.0);
    }

    #[test]
    fn timestamp_decode_missing_end_timestamp_falls_back() {
        let tokenizer = timestamp_test_tokenizer();
        let decoded = decode_timestamp_window(&tokenizer, &[100, 10, 11]).unwrap();
        assert!(decoded.is_none());
    }

    #[test]
    fn timestamp_decode_without_timestamp_tokens_falls_back() {
        let tokenizer = timestamp_test_tokenizer();
        let decoded = decode_timestamp_window(&tokenizer, &[10, 11]).unwrap();
        assert!(decoded.is_none());
    }

    #[test]
    fn timestamp_decode_rejects_non_monotonic_timestamps() {
        let tokenizer = timestamp_test_tokenizer();
        let error = decode_timestamp_window(&tokenizer, &[150, 10, 100])
            .unwrap_err()
            .to_string();
        assert!(error.contains("model_output_mismatch"));
        assert!(error.contains("not monotonic"));
    }

    #[test]
    fn timestamp_decode_uses_text_between_timestamp_pairs() {
        let tokenizer = timestamp_test_tokenizer();
        let decoded = decode_timestamp_window(&tokenizer, &[100, 150, 10, 200, 250])
            .unwrap()
            .unwrap();
        assert_eq!(decoded.segments.len(), 1);
        assert_eq!(decoded.segments[0].text, "hello");
        assert_eq!(decoded.segments[0].start_seconds, 1.0);
        assert_eq!(decoded.segments[0].end_seconds, 2.0);
    }

    #[test]
    fn timestamp_logit_rules_select_timestamp_when_timestamp_mass_wins() {
        let spec = WhisperTimestampSpec {
            begin_token_id: 10,
            end_token_id: 13,
            seconds_per_token: 0.02,
        };
        let mut logits = vec![0.0; 13];
        logits[3] = 2.0;
        logits[10] = 1.8;
        logits[11] = 1.8;
        apply_timestamp_logit_rules(&mut logits, &[], &spec, 2).unwrap();

        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
        let selected = argmax_finite(&logits).unwrap();
        assert!((10..13).contains(&selected));
    }

    #[test]
    fn projected_single_word_receives_full_segment_duration() {
        let words = project_words_from_timestamp_segment(&decoded_segment("hello", 10.0, 12.0));

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[0].start_seconds, Some(10.0));
        assert_eq!(words[0].end_seconds, Some(12.0));
        assert_eq!(
            words[0].attributes.get("timing").map(String::as_str),
            Some("whisperTimestampProjection")
        );
    }

    #[test]
    fn projected_words_split_by_character_weight() {
        let words =
            project_words_from_timestamp_segment(&decoded_segment("hello rustaceans", 0.0, 3.0));

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "rustaceans");
        assert_approx_eq(words[0].start_seconds.unwrap(), 0.0);
        assert_approx_eq(words[0].end_seconds.unwrap(), 1.0);
        assert_approx_eq(words[1].start_seconds.unwrap(), 1.0);
        assert_approx_eq(words[1].end_seconds.unwrap(), 3.0);
    }

    #[test]
    fn projected_words_keep_punctuation_attached() {
        let words =
            project_words_from_timestamp_segment(&decoded_segment("hello, world!", 0.0, 2.0));

        assert_eq!(
            words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["hello,", "world!"]
        );
    }

    #[test]
    fn projected_words_ignore_empty_or_whitespace_text() {
        let words = project_words_from_timestamp_segment(&decoded_segment("   \n\t  ", 0.0, 2.0));

        assert!(words.is_empty());
    }

    #[test]
    fn projected_words_stay_inside_parent_segment() {
        let words =
            project_words_from_timestamp_segment(&decoded_segment("hello rust world", 5.0, 5.1));

        assert!(!words.is_empty());
        for word in words {
            let start = word.start_seconds.unwrap();
            let end = word.end_seconds.unwrap();
            assert!((5.0..=5.1).contains(&start));
            assert!((5.0..=5.1).contains(&end));
            assert!(end >= start);
        }
    }

    #[test]
    fn timestamp_decoded_segments_do_not_project_words() {
        let decoded = WhisperDecodedWindow {
            text: "hello world".to_string(),
            segments: vec![decoded_segment("hello world", 0.5, 1.5)],
        };
        let mut next_index = 0;
        let segments = decoded_window_to_contract_segments(
            decoded,
            &mut next_index,
            10.0,
            12.0,
            Some("en".to_string()),
        );

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_seconds, Some(10.5));
        assert_eq!(segments[0].end_seconds, Some(11.5));
        assert!(segments[0].words.is_empty());
        assert!(!segments[0].attributes.contains_key("wordTiming"));
    }

    #[test]
    fn timestamp_decoded_multiple_segments_keep_word_timing_empty() {
        let decoded = WhisperDecodedWindow {
            text: "hello world rustaceans".to_string(),
            segments: vec![
                decoded_segment("hello world", 0.0, 1.0),
                decoded_segment("rustaceans unite", 1.0, 2.0),
            ],
        };
        let mut next_index = 0;
        let segments =
            decoded_window_to_contract_segments(decoded, &mut next_index, 10.0, 12.0, None);

        assert_eq!(segments.len(), 2);
        for segment in &segments {
            let segment_start = segment.start_seconds.unwrap();
            let segment_end = segment.end_seconds.unwrap();
            assert!(segment_start >= 10.0);
            assert!(segment_end <= 12.0);
            assert!(segment.words.is_empty());
        }
    }

    #[test]
    fn timestamp_decoded_segments_map_to_transcript_contracts() {
        let decoded = WhisperDecodedWindow {
            text: "hello world".to_string(),
            segments: vec![
                WhisperDecodedSegment {
                    text: "hello".to_string(),
                    start_seconds: 0.5,
                    end_seconds: 1.25,
                    token_ids: vec![10],
                },
                WhisperDecodedSegment {
                    text: "world".to_string(),
                    start_seconds: 1.25,
                    end_seconds: 1.75,
                    token_ids: vec![11],
                },
            ],
        };
        let mut next_index = 7;
        let segments = decoded_window_to_contract_segments(
            decoded,
            &mut next_index,
            10.0,
            12.0,
            Some("en".to_string()),
        );
        assert_eq!(next_index, 9);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].index, 7);
        assert_eq!(segments[0].text, "hello");
        assert_eq!(segments[0].start_seconds, Some(10.5));
        assert_eq!(segments[0].end_seconds, Some(11.25));
        assert_eq!(segments[1].index, 8);
        assert_eq!(segments[1].text, "world");
        assert_eq!(segments[1].start_seconds, Some(11.25));
        assert_eq!(segments[1].end_seconds, Some(11.75));
        assert_eq!(segments[0].language.as_deref(), Some("en"));
        assert_eq!(
            segments[0].attributes.get("timing").map(String::as_str),
            Some("global")
        );
        assert_eq!(
            segments[0]
                .attributes
                .get("timingSource")
                .map(String::as_str),
            Some("whisperTimestampTokens")
        );
        TranscriptionContract::from_segments(None, Some("en".to_string()), segments).unwrap();
    }

    #[test]
    fn projected_timestamp_words_pass_strict_transcript_validation() {
        let decoded = WhisperDecodedWindow {
            text: "hello world".to_string(),
            segments: vec![decoded_segment("hello world", 0.25, 1.25)],
        };
        let mut next_index = 0;
        let segments = decoded_window_to_contract_segments(
            decoded,
            &mut next_index,
            3.0,
            5.0,
            Some("en".to_string()),
        );
        let transcript =
            TranscriptionContract::from_segments(None, Some("en".to_string()), segments).unwrap();

        transcript.validate_strict().unwrap();
    }

    #[test]
    fn window_fallback_segment_uses_global_timing() {
        let segment =
            window_fallback_segment(3, "hello".to_string(), 4.0, 5.5, Some("en".to_string()));
        assert_eq!(segment.start_seconds, Some(4.0));
        assert_eq!(segment.end_seconds, Some(5.5));
        assert_eq!(
            segment.attributes.get("provider").map(String::as_str),
            Some("candle-whisper")
        );
        assert_eq!(
            segment.attributes.get("timing").map(String::as_str),
            Some("global")
        );
    }

    #[test]
    fn fallback_chunk_window_segment_does_not_project_words() {
        let segment = window_fallback_segment(
            3,
            "hello world".to_string(),
            4.0,
            5.5,
            Some("en".to_string()),
        );

        assert!(segment.words.is_empty());
        assert!(!segment.attributes.contains_key("wordTiming"));
    }

    #[test]
    fn chunk_windows_carry_local_and_global_timing() {
        let chunk = SpeechActivitySegment::new(1.0, 2.0, 0.8).unwrap();
        let windows = chunk_windows(
            &vec![0.0; 48_000],
            16_000,
            &chunk,
            &CandleWhisperWindowControls::default(),
        )
        .unwrap();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].chunk_start_seconds, 0.75);
        assert_eq!(windows[0].local_start_seconds, 0.0);
        assert_eq!(windows[0].local_end_seconds, 1.29);
        assert_eq!(windows[0].global_start_seconds, 0.75);
        assert_eq!(windows[0].global_end_seconds, 2.04);
    }

    #[test]
    fn chunk_windows_use_request_scoped_context() {
        let chunk = SpeechActivitySegment::new(1.0, 2.0, 0.8).unwrap();
        let controls = CandleWhisperWindowControls {
            leading_context_seconds: 0.1,
            trailing_context_seconds: 0.2,
            ..CandleWhisperWindowControls::default()
        };
        let windows = chunk_windows(&vec![0.0; 48_000], 16_000, &chunk, &controls).unwrap();

        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].chunk_start_seconds, 0.9);
        assert_eq!(windows[0].local_end_seconds, 1.3);
        assert_eq!(windows[0].global_start_seconds, 0.9);
        assert_eq!(windows[0].global_end_seconds, 2.2);
    }

    #[test]
    fn unknown_explicit_language_returns_invalid_request() {
        let error = CandleWhisperSession::initial_prompt_tokens(
            &test_generation(),
            &test_tokenizer(),
            Some("zz"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("invalid_request"));
        assert!(error.contains("language token"));
    }

    #[test]
    fn missing_eos_returns_invalid_request() {
        let mut generation = test_generation();
        generation.eos_token_id = None;
        let tokenizer = test_tokenizer();
        assert!(CandleWhisperSession::resolve_eos_token_id(&generation, &tokenizer).is_ok());

        let temp = tempfile::tempdir().unwrap();
        let vocab = temp.path().join("vocab.json");
        std::fs::write(
            &vocab,
            serde_json::json!({
                "<unk>": 0,
                whisper::SOT_TOKEN: 1,
                "<|en|>": 3,
                "<|transcribe|>": 5,
                whisper::NO_TIMESTAMPS_TOKEN: 7
            })
            .to_string(),
        )
        .unwrap();
        let tokenizer = Tokenizer::new(
            WordLevel::from_file(vocab.to_str().unwrap(), "<unk>".to_string()).unwrap(),
        );
        let error = CandleWhisperSession::resolve_eos_token_id(&generation, &tokenizer)
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid_request"));
        assert!(error.contains("eos_token_id"));
    }

    #[test]
    fn whisper_bundle_resolution_accepts_direct_and_files_layouts() {
        for nested in [false, true] {
            let temp = tempfile::tempdir().unwrap();
            let root = temp.path();
            let file_root = if nested {
                std::fs::create_dir(root.join("files")).unwrap();
                root.join("files")
            } else {
                root.to_path_buf()
            };
            for file in [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "preprocessor_config.json",
                "model.safetensors",
            ] {
                std::fs::write(file_root.join(file), "").unwrap();
            }
            let paths = resolve_whisper_bundle_paths(root).unwrap();
            assert!(paths.config_json.exists());
            assert!(paths.generation_config_json.exists());
            assert!(paths.tokenizer_json.exists());
            assert!(paths.preprocessor_config_json.exists());
            assert!(paths.model_safetensors.exists());
        }
    }

    #[cfg(feature = "model-bundles")]
    #[test]
    fn whisper_bundle_resolution_accepts_manifest_layout() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::create_dir(root.join("snapshots")).unwrap();
        for file in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "preprocessor_config.json",
            "model.safetensors",
        ] {
            std::fs::write(root.join("snapshots").join(file), "").unwrap();
        }
        std::fs::write(
            root.join("manifest.json"),
            serde_json::json!({
                "schema_version": 1,
                "name": "whisper-test",
                "repo_id": "openai/whisper-tiny",
                "revision": "main",
                "task": "speech_recognition",
                "files": {
                    "config.json": {"remote_path": "config.json", "local_path": "snapshots/config.json", "size_bytes": 0},
                    "generation_config.json": {"remote_path": "generation_config.json", "local_path": "snapshots/generation_config.json", "size_bytes": 0},
                    "tokenizer.json": {"remote_path": "tokenizer.json", "local_path": "snapshots/tokenizer.json", "size_bytes": 0},
                    "preprocessor_config.json": {"remote_path": "preprocessor_config.json", "local_path": "snapshots/preprocessor_config.json", "size_bytes": 0},
                    "model.safetensors": {"remote_path": "model.safetensors", "local_path": "snapshots/model.safetensors", "size_bytes": 0}
                }
            })
            .to_string(),
        )
        .unwrap();
        let paths = resolve_whisper_bundle_paths(root).unwrap();
        assert_eq!(
            paths.model_safetensors,
            root.join("snapshots/model.safetensors")
        );
    }
}
