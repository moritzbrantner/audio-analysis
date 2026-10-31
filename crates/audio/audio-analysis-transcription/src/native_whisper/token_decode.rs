//! Autoregressive token decoding, logit filtering and initial prompt tokens.

use crate::{
    invalid_request, model_output_mismatch, CandleWhisperDecodeRequestConfig,
    CandleWhisperTimingMode, CandleWhisperWindowControls, SpeechSpan, TranscriptionTask,
};
use candle_core::{DType, IndexOp, Tensor};
use candle_transformers::models::whisper;
use media_core::Result;
use tokenizers::Tokenizer;

use super::*;

impl CandleWhisperSession {
    pub(super) fn decode_tokens_batch(
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

    pub(super) fn decode_tokens_configured(
        &mut self,
        audio_features: &Tensor,
        mode: WhisperDecodeMode,
        decode: &CandleWhisperDecodeRequestConfig,
    ) -> Result<WhisperTokenDecodeResult> {
        let eos = self.eos_token_id()?;
        let initial_tokens = self.initial_tokens(mode, &decode.initial_prompt_tokens)?;
        let max_generated_tokens = whisper_max_generated_tokens(
            &self.generation,
            self.model.config(),
            initial_tokens.token_ids.len(),
        );
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
    pub(super) fn configured_search_logits(
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
        let initial_tokens = self.initial_tokens(mode, &decode.initial_prompt_tokens)?;
        let max_generated_tokens = whisper_max_generated_tokens(
            &self.generation,
            self.model.config(),
            initial_tokens.token_ids.len(),
        );
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

        while !active_rows.is_empty()
            && active_rows[0].row.generated_tokens().len() < max_generated_tokens
        {
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

    pub(super) fn initial_tokens(
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
    pub(super) fn build_initial_tokens(
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
    pub(super) fn initial_prompt_tokens(
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
    pub(super) fn initial_prompt_tokens_for_task(
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

    pub(super) fn initial_prompt_tokens_for_mode(
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

    pub(super) fn resolve_eos_token_id(
        generation: &GenerationConfig,
        tokenizer: &Tokenizer,
    ) -> Result<u32> {
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

pub(super) fn collect_chunk_windows(
    samples: &[f32],
    sample_rate: u32,
    chunks: &[SpeechSpan],
    controls: &CandleWhisperWindowControls,
) -> Result<Vec<ChunkWindow>> {
    let mut windows = Vec::new();
    for chunk in chunks {
        windows.extend(chunk_windows(samples, sample_rate, chunk, controls)?);
    }
    Ok(windows)
}

pub(super) fn apply_active_row_decisions(
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
