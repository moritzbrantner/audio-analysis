//! Candle Whisper session: model load, window transcription and encoding.

use crate::native_device::ResolvedNativeDevice;
use crate::native_whisper_quantized::CandleQ8WhisperModel;
use crate::{
    candle_batch_count, invalid_request, model_output_mismatch, setup_error, AsrRequest,
    AsrResponse, CandleWhisperDecodeRequestConfig, CandleWhisperDecodeRuntime,
    CandleWhisperOptions, CandleWhisperRuntimeControls, CandleWhisperTimingMode,
    CandleWhisperTranscriptionRequestConfig,
};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper;
use media_core::Result;
use tokenizers::Tokenizer;

use super::*;

pub(super) struct CandleWhisperSession {
    pub(super) setup: WhisperRunSetup,
    /// Decode language of the current request: the requested one, or the one
    /// detected from the first window when none was requested.
    pub(super) language: Option<String>,
    pub(super) device: Device,
    pub(super) model: WhisperModel,
    pub(super) tokenizer: Tokenizer,
    pub(super) generation: GenerationConfig,
    mel_filters: Vec<f32>,
    transformers_mel_filters: Vec<f32>,
    encoder_duration_seconds: f64,
    pub(super) decoder_duration_seconds: f64,
}

#[derive(Clone, Copy)]
pub(super) enum WhisperFeatureExtractorMode {
    Legacy,
    Transformers,
}

pub(super) fn whisper_max_generated_tokens(
    generation: &GenerationConfig,
    config: &whisper::Config,
    prompt_length: usize,
) -> usize {
    let total_max_length = generation
        .max_length
        .unwrap_or(config.max_target_positions)
        .min(config.max_target_positions);
    (total_max_length / 2).min(total_max_length.saturating_sub(prompt_length))
}

impl CandleWhisperSession {
    pub(super) fn load(setup: WhisperRunSetup) -> Result<Self> {
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
            language: setup.language.clone(),
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

    pub(super) fn transcribe_chunks(
        &mut self,
        options: &CandleWhisperOptions,
        config: &CandleWhisperTranscriptionRequestConfig,
        request: AsrRequest,
    ) -> Result<AsrResponse> {
        let controls = &config.runtime;
        let decode = self.resolve_request_decode_config(&config.decode)?;
        let decode = &decode;
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
        let detected_language = match &self.setup.language {
            Some(language) => {
                self.language = Some(language.clone());
                None
            }
            None => {
                let detected = self.detect_language(&request.audio.samples)?;
                self.language = detected.as_ref().map(|(language, _)| language.clone());
                detected
            }
        };
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
                let (clip_start, clip_end) = window.transcript_bounds();
                if timed.timing == WhisperWindowTiming::ChunkWindow {
                    if timed.decoded.text.trim().is_empty() || clip_end <= clip_start {
                        continue;
                    }
                    segments.push(window_fallback_segment(
                        next_index,
                        timed.decoded.text,
                        clip_start,
                        clip_end,
                        self.language.clone(),
                    )?);
                    next_index += 1;
                } else {
                    used_timestamp_tokens = true;
                    let timestamp_segments = decoded_window_to_contract_segments(
                        timed.decoded,
                        &mut next_index,
                        window.global_start_seconds,
                        (clip_start, clip_end),
                        self.language.clone(),
                    )?;
                    if timestamp_segments
                        .iter()
                        .any(|segment| !segment.words().is_empty())
                    {
                        used_timestamp_word_projection = true;
                    }
                    segments.extend(timestamp_segments);
                }
            }
        }
        let transcript = crate::transcription_from_segments(
            request.audio.source,
            self.language.clone(),
            segments,
        )
        .map_err(|error| model_output_mismatch(error.to_string()))?;
        let device_label = device_label(&self.setup.resolved_device);
        let mut diagnostics = whisper_setup_diagnostics(&self.setup);
        if let WhisperModel::Q8(model) = &self.model {
            diagnostics.extend(q8_execution_diagnostics(
                model.execution_route(),
                model.kernel_route().as_str(),
            ));
        }
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
                "initialPromptTokenCount={}",
                decode.initial_prompt_tokens.len()
            ),
            format!("suppressedTokenCount={}", decode.suppressed_token_ids.len()),
            format!("suppressNumerals={}", decode.suppress_numerals),
            format!(
                "conditionOnPreviousText={}",
                decode.condition_on_previous_text
            ),
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
        if let Some(language) = &self.language {
            diagnostics.push(format!("language={language}"));
        }
        if let Some((_, probability)) = &detected_language {
            diagnostics.push("languageDetection=whisperFirstWindow".to_string());
            diagnostics.push(format!("languageProbability={probability}"));
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
                .or_else(|| self.language.clone()),
            transcript,
            diagnostics,
        })
    }

    /// Whisper language identification, as in OpenAI Whisper and WhisperX:
    /// encode the first 30 s of the request audio, run the decoder on the
    /// start-of-transcript token alone and take the most probable language
    /// token. English-only models define no language tokens and return `None`.
    pub(super) fn detect_language(&mut self, samples: &[f32]) -> Result<Option<(String, f64)>> {
        let candidates = language_candidates(&self.generation);
        if candidates.is_empty() || samples.is_empty() {
            return Ok(None);
        }
        let samples = &samples[..samples.len().min(whisper::N_SAMPLES)];
        let seconds = samples.len() as f64 / whisper::SAMPLE_RATE as f64;
        let window = ChunkWindow {
            samples: samples.to_vec(),
            chunk_start_seconds: 0.0,
            local_start_seconds: 0.0,
            local_end_seconds: seconds,
            global_start_seconds: 0.0,
            global_end_seconds: seconds,
            speech_start_seconds: 0.0,
            speech_end_seconds: seconds,
        };
        let features = self.encode_window_batch(
            std::slice::from_ref(&window),
            WhisperFeatureExtractorMode::Legacy,
        )?;
        let start = Self::decoder_start_token_id(&self.generation, &self.tokenizer)?;
        let failed = |error: candle_core::Error| {
            model_output_mismatch(format!("Whisper language detection failed: {error}"))
        };
        let tokens = Tensor::new(&[start], &self.device)
            .and_then(|tokens| tokens.unsqueeze(0))
            .map_err(failed)?;
        self.model.reset_cache();
        let decoder_started = std::time::Instant::now();
        let (decoded, _) = self
            .model
            .decode(&tokens, &features, 0, true)
            .map_err(failed)?;
        self.decoder_duration_seconds += decoder_started.elapsed().as_secs_f64();
        let logits = self
            .model
            .project_logits(&decoded)
            .and_then(|logits| logits.i((0, 0, ..)))
            .and_then(|logits| logits.to_dtype(DType::F32))
            .and_then(|logits| logits.to_vec1::<f32>())
            .map_err(failed)?;
        Ok(pick_language(&logits, &candidates))
    }

    fn resolve_request_decode_config(
        &self,
        config: &CandleWhisperDecodeRequestConfig,
    ) -> Result<CandleWhisperDecodeRequestConfig> {
        let mut resolved = config.clone();
        if let Some(prompt) = config
            .initial_prompt
            .as_deref()
            .filter(|prompt| !prompt.is_empty())
        {
            let encoding = self.tokenizer.encode(prompt, false).map_err(|error| {
                invalid_request(format!(
                    "Candle Whisper could not tokenize initial_prompt: {error}"
                ))
            })?;
            resolved
                .initial_prompt_tokens
                .extend_from_slice(encoding.get_ids());
        }
        for token_id in &resolved.suppressed_token_ids {
            if self.tokenizer.id_to_token(*token_id).is_none() {
                return Err(invalid_request(format!(
                    "Whisper suppressed token id `{token_id}` is not in the tokenizer vocabulary"
                )));
            }
        }
        Ok(resolved)
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
                speech_start_seconds: 0.0,
                speech_end_seconds: samples.len() as f64 / whisper::SAMPLE_RATE as f64,
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

    pub(super) fn encode_window_batch(
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

    pub(super) fn mel_tensor_batch(
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
        let mel = Tensor::from_vec(
            features,
            (windows.len(), n_mel, whisper::N_FRAMES),
            &self.device,
        )
        .map_err(|error| model_output_mismatch(format!("failed to build mel tensor: {error}")))?;
        prepare_whisper_encoder_input(mel, self.setup.model_weight_dtype).map_err(|error| {
            model_output_mismatch(format!(
                "failed to convert mel tensor to {} for Whisper encoder: {error}",
                candle_dtype_name(self.setup.model_weight_dtype)
            ))
        })
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
}

pub(super) fn decoder_threads_diagnostic(
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

pub(super) fn observed_candle_batch_execution(
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

pub(super) fn format_effective_active_batch_sizes(sizes: &[usize]) -> String {
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

/// Language codes and token ids from `generation_config.json`'s `lang_to_id`
/// (empty for English-only models). Using the configured ids keeps detection
/// correct for both the pre-v3 (99) and v3 (100) language vocabularies.
pub(super) fn language_candidates(generation: &GenerationConfig) -> Vec<(String, u32)> {
    generation
        .lang_to_id
        .iter()
        .map(|(token, id)| {
            let code = token.trim_start_matches("<|").trim_end_matches("|>");
            (code.to_string(), *id)
        })
        .collect()
}

/// Softmax over the language-token logits only, then the most probable
/// language with its probability.
pub(super) fn pick_language(logits: &[f32], candidates: &[(String, u32)]) -> Option<(String, f64)> {
    let scored = candidates
        .iter()
        .filter_map(|(code, id)| {
            let logit = f64::from(*logits.get(*id as usize)?);
            logit.is_finite().then_some((code, logit))
        })
        .collect::<Vec<_>>();
    let (code, max) = scored
        .iter()
        .copied()
        .max_by(|left, right| left.1.total_cmp(&right.1))?;
    let total = scored
        .iter()
        .map(|(_, logit)| (logit - max).exp())
        .sum::<f64>();
    Some((code.clone(), 1.0 / total))
}

pub(super) fn format_cache_reuse(self_attention: bool, cross_attention: bool) -> &'static str {
    match (self_attention, cross_attention) {
        (true, true) => "self-and-cross-attention",
        (true, false) => "self-attention",
        (false, true) => "cross-attention",
        (false, false) => "none",
    }
}
