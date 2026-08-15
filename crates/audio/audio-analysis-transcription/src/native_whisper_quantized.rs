use std::path::Path;

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm};
use candle_transformers::models::whisper;
use candle_transformers::quantized_nn::{layer_norm, linear, linear_no_bias, Embedding, Linear};
use candle_transformers::quantized_var_builder::VarBuilder;

/// Observable cache behavior for one Q8 Whisper decoder operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandleQ8WhisperDecoderDiagnostics {
    /// Whether this operation cleared all decoder state before evaluating tokens.
    pub cache_reset: bool,
    /// Absolute position assigned to the first supplied token.
    pub position_offset: usize,
    /// Number of token positions supplied by this operation.
    pub input_token_count: usize,
    /// Number of self-attention token positions retained before this operation.
    pub self_attention_cache_tokens_before: usize,
    /// Number of self-attention token positions retained after this operation.
    pub self_attention_cache_tokens_after: usize,
    /// Whether the operation appended to existing self-attention keys and values.
    pub self_attention_cache_reused: bool,
    /// Whether encoder keys and values were projected during this operation.
    pub cross_attention_cache_computed: bool,
    /// Whether existing encoder keys and values were reused during this operation.
    pub cross_attention_cache_reused: bool,
    /// Number of decoder layers that projected encoder keys and values.
    pub cross_attention_projection_count: usize,
}

/// Activations and cache diagnostics returned by a Q8 Whisper decoder operation.
#[derive(Debug, Clone)]
pub struct CandleQ8WhisperDecoderOutput {
    pub activations: Tensor,
    pub diagnostics: CandleQ8WhisperDecoderDiagnostics,
}

/// Aggregate behavior observed while greedily decoding an active-row Q8 batch.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CandleQ8WhisperBatchDiagnostics {
    /// Active row count supplied to each decoder operation, in execution order.
    pub effective_active_row_batch_sizes: Vec<usize>,
    /// Number of decoder/encoder row compactions after partial batch completion.
    pub active_row_compaction_count: usize,
    /// Number of decoder operations that appended to self-attention state.
    pub self_attention_cache_reuse_count: usize,
    /// Number of decoder operations that reused encoder cross-attention state.
    pub cross_attention_cache_reuse_count: usize,
    /// Number of encoder key/value projections performed across decoder layers.
    pub cross_attention_projection_count: usize,
    /// Number of non-EOS transcript tokens generated across all rows.
    pub generated_token_count: usize,
}

/// Ordered transcript token IDs and diagnostics from a greedy Q8 batch decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleQ8WhisperBatchOutput {
    /// Per-row generated token IDs in the same order as the encoder feature rows.
    pub token_ids: Vec<Vec<u32>>,
    pub diagnostics: CandleQ8WhisperBatchDiagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct AttentionCacheStats {
    cache_computed: bool,
    cache_reused: bool,
}

#[derive(Debug, Clone)]
struct Q8WhisperAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Q8WhisperAttention {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        require_q8_weight(&vb, (n_state, n_state), "q_proj.weight")?;
        require_q8_weight(&vb, (n_state, n_state), "k_proj.weight")?;
        require_q8_weight(&vb, (n_state, n_state), "v_proj.weight")?;
        require_q8_weight(&vb, (n_state, n_state), "out_proj.weight")?;
        Ok(Self {
            query: linear(n_state, n_state, vb.pp("q_proj"))?,
            key: linear_no_bias(n_state, n_state, vb.pp("k_proj"))?,
            value: linear(n_state, n_state, vb.pp("v_proj"))?,
            out: linear(n_state, n_state, vb.pp("out_proj"))?,
            n_head,
            kv_cache: None,
        })
    }

    fn forward_self(&mut self, x: &Tensor, mask: &Tensor) -> Result<(Tensor, AttentionCacheStats)> {
        let q = self.query.forward(x)?;
        let current_k = self.key.forward(x)?;
        let current_v = self.value.forward(x)?;
        let cache_reused = self.kv_cache.is_some();
        let (k, v) = match &self.kv_cache {
            Some((cached_k, cached_v)) => (
                Tensor::cat(&[cached_k, &current_k], 1)?,
                Tensor::cat(&[cached_v, &current_v], 1)?,
            ),
            None => (current_k, current_v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        let output = self
            .out
            .forward(&self.qkv_attention(&q, &k, &v, Some(mask))?)?;
        Ok((
            output,
            AttentionCacheStats {
                cache_computed: true,
                cache_reused,
            },
        ))
    }

    fn forward_cross(
        &mut self,
        x: &Tensor,
        encoder_features: &Tensor,
    ) -> Result<(Tensor, AttentionCacheStats)> {
        let q = self.query.forward(x)?;
        let (k, v, stats) = match &self.kv_cache {
            Some((k, v)) => (
                k.clone(),
                v.clone(),
                AttentionCacheStats {
                    cache_computed: false,
                    cache_reused: true,
                },
            ),
            None => {
                let k = self.key.forward(encoder_features)?;
                let v = self.value.forward(encoder_features)?;
                self.kv_cache = Some((k.clone(), v.clone()));
                (
                    k,
                    v,
                    AttentionCacheStats {
                        cache_computed: true,
                        cache_reused: false,
                    },
                )
            }
        };
        let output = self.out.forward(&self.qkv_attention(&q, &k, &v, None)?)?;
        Ok((output, stats))
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, _, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            qk = qk.broadcast_add(mask)?;
        }
        candle_nn::ops::softmax_last_dim(&qk)?
            .matmul(&v)?
            .transpose(1, 2)?
            .flatten_from(2)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, sequence, state) = x.dims3()?;
        x.reshape((batch, sequence, self.n_head, state / self.n_head))?
            .transpose(1, 2)
    }

    fn reset_cache(&mut self) {
        self.kv_cache = None;
    }

    fn select_cache_rows(&mut self, row_indices: &Tensor) -> Result<()> {
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
struct Q8WhisperBlockStats {
    self_cache_reused: bool,
    cross_cache_computed: bool,
    cross_cache_reused: bool,
}

#[derive(Debug, Clone)]
struct Q8WhisperBlock {
    self_attention: Q8WhisperAttention,
    self_attention_layer_norm: LayerNorm,
    cross_attention: Q8WhisperAttention,
    cross_attention_layer_norm: LayerNorm,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    final_layer_norm: LayerNorm,
}

#[derive(Debug, Clone)]
struct Q8WhisperEncoderBlock {
    self_attention: Q8WhisperAttention,
    self_attention_layer_norm: LayerNorm,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    final_layer_norm: LayerNorm,
}

impl Q8WhisperEncoderBlock {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        require_q8_weight(&vb, (n_state * 4, n_state), "fc1.weight")?;
        require_q8_weight(&vb, (n_state, n_state * 4), "fc2.weight")?;
        Ok(Self {
            self_attention: Q8WhisperAttention::load(n_state, n_head, vb.pp("self_attn"))?,
            self_attention_layer_norm: layer_norm(n_state, 1e-5, vb.pp("self_attn_layer_norm"))?,
            mlp_linear1: linear(n_state, n_state * 4, vb.pp("fc1"))?,
            mlp_linear2: linear(n_state * 4, n_state, vb.pp("fc2"))?,
            final_layer_norm: layer_norm(n_state, 1e-5, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.self_attention.reset_cache();
        let (attention, _) = self
            .self_attention
            .forward_self(&self.self_attention_layer_norm.forward(x)?, mask)?;
        let x = (x + attention)?;
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.final_layer_norm.forward(&x)?)?
                .gelu()?,
        )?;
        x + mlp
    }
}

#[derive(Debug, Clone)]
struct CandleQ8WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<Q8WhisperEncoderBlock>,
    layer_norm: LayerNorm,
}

impl CandleQ8WhisperEncoder {
    fn load(builder: &VarBuilder, config: &whisper::Config) -> Result<Self> {
        let vb = builder.pp("model.encoder");
        let conv1 = q8_conv1d(
            config.num_mel_bins,
            config.d_model,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = q8_conv1d(
            config.d_model,
            config.d_model,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let positional_embedding = sinusoids(
            config.max_source_positions,
            config.d_model,
            builder.device(),
        )?;
        let blocks = (0..config.encoder_layers)
            .map(|index| {
                Q8WhisperEncoderBlock::load(
                    config.d_model,
                    config.encoder_attention_heads,
                    vb.pp(format!("layers.{index}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            layer_norm: layer_norm(config.d_model, 1e-5, vb.pp("layer_norm"))?,
        })
    }

    fn forward(&mut self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(mel)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?.transpose(1, 2)?;
        let (_, sequence_len, _) = x.dims3()?;
        let positions = self.positional_embedding.narrow(0, 0, sequence_len)?;
        let mut x = x.broadcast_add(&positions)?;
        let mask = Tensor::zeros((sequence_len, sequence_len), DType::F32, x.device())?;
        for block in &mut self.blocks {
            x = block.forward(&x, &mask)?;
        }
        self.layer_norm.forward(&x)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CandleQ8WhisperModel {
    encoder: CandleQ8WhisperEncoder,
    decoder: CandleQ8WhisperDecoder,
    pub(crate) config: whisper::Config,
}

impl CandleQ8WhisperModel {
    pub(crate) fn from_gguf(path: &Path, config: whisper::Config, device: &Device) -> Result<Self> {
        let builder = VarBuilder::from_gguf(path, device)?;
        let encoder = CandleQ8WhisperEncoder::load(&builder, &config)?;
        let decoder = CandleQ8WhisperDecoder::from_var_builder(&builder, config.clone())?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub(crate) fn encode(&mut self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward(mel)
    }

    pub(crate) fn decode(
        &mut self,
        tokens: &Tensor,
        encoder_features: &Tensor,
        position_offset: usize,
        reset_cache: bool,
    ) -> Result<CandleQ8WhisperDecoderOutput> {
        self.decoder
            .forward_incremental(tokens, encoder_features, position_offset, reset_cache)
    }

    pub(crate) fn project_logits(&self, activations: &Tensor) -> Result<Tensor> {
        self.decoder.project_logits(activations)
    }

    pub(crate) fn reset_cache(&mut self) {
        self.decoder.reset_cache();
    }

    pub(crate) fn select_cache_rows(&mut self, row_indices: &Tensor) -> Result<()> {
        self.decoder.select_cache_rows(row_indices)
    }
}

fn q8_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb
        .get((out_channels, in_channels, kernel_size), "weight")?
        .dequantize(vb.device())?;
    let bias = vb.get(out_channels, "bias")?.dequantize(vb.device())?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn sinusoids(length: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let max_timescale = 10_000_f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inverse_timescales = (0..channels / 2)
        .map(|index| (index as f32 * -log_timescale_increment).exp())
        .collect::<Vec<_>>();
    let inverse_timescales = Tensor::new(inverse_timescales.as_slice(), device)?.unsqueeze(0)?;
    let positions = Tensor::arange(0, length as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;
    let shape = (length, channels / 2);
    let scaled = (positions.broadcast_as(shape)? * inverse_timescales.broadcast_as(shape)?)?;
    Tensor::cat(&[scaled.sin()?, scaled.cos()?], 1)
}

impl Q8WhisperBlock {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        require_q8_weight(&vb, (n_state * 4, n_state), "fc1.weight")?;
        require_q8_weight(&vb, (n_state, n_state * 4), "fc2.weight")?;
        Ok(Self {
            self_attention: Q8WhisperAttention::load(n_state, n_head, vb.pp("self_attn"))?,
            self_attention_layer_norm: layer_norm(n_state, 1e-5, vb.pp("self_attn_layer_norm"))?,
            cross_attention: Q8WhisperAttention::load(n_state, n_head, vb.pp("encoder_attn"))?,
            cross_attention_layer_norm: layer_norm(
                n_state,
                1e-5,
                vb.pp("encoder_attn_layer_norm"),
            )?,
            mlp_linear1: linear(n_state, n_state * 4, vb.pp("fc1"))?,
            mlp_linear2: linear(n_state * 4, n_state, vb.pp("fc2"))?,
            final_layer_norm: layer_norm(n_state, 1e-5, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        encoder_features: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Q8WhisperBlockStats)> {
        let (self_attention, self_stats) = self
            .self_attention
            .forward_self(&self.self_attention_layer_norm.forward(x)?, mask)?;
        let x = (x + self_attention)?;
        let (cross_attention, cross_stats) = self.cross_attention.forward_cross(
            &self.cross_attention_layer_norm.forward(&x)?,
            encoder_features,
        )?;
        let x = (&x + cross_attention)?;
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.final_layer_norm.forward(&x)?)?
                .gelu()?,
        )?;
        Ok((
            (x + mlp)?,
            Q8WhisperBlockStats {
                self_cache_reused: self_stats.cache_reused,
                cross_cache_computed: cross_stats.cache_computed,
                cross_cache_reused: cross_stats.cache_reused,
            },
        ))
    }

    fn reset_cache(&mut self) {
        self.self_attention.reset_cache();
        self.cross_attention.reset_cache();
    }

    fn select_cache_rows(&mut self, row_indices: &Tensor) -> Result<()> {
        self.self_attention.select_cache_rows(row_indices)?;
        self.cross_attention.select_cache_rows(row_indices)
    }
}

/// Position-aware incremental decoder for a Q8_0 Candle Whisper model.
///
/// The decoder owns cache state for one fixed audio window. Call
/// [`Self::forward_incremental`] with `reset_cache = true` for the prompt
/// prefill of a new window, then pass only newly generated tokens with their
/// absolute position offsets.
#[derive(Debug, Clone)]
pub struct CandleQ8WhisperDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<Q8WhisperBlock>,
    layer_norm: LayerNorm,
    max_target_positions: usize,
    cache_start_position: Option<usize>,
    cached_token_count: usize,
}

impl CandleQ8WhisperDecoder {
    /// Loads only the Whisper text decoder tensors from a GGUF file.
    pub fn from_gguf(
        path: impl AsRef<Path>,
        config: whisper::Config,
        device: &Device,
    ) -> Result<Self> {
        let builder = VarBuilder::from_gguf(path, device)?;
        Self::from_var_builder(&builder, config)
    }

    /// Loads only the Whisper text decoder tensors from an in-memory GGUF fixture.
    pub fn from_gguf_buffer(
        buffer: &[u8],
        config: whisper::Config,
        device: &Device,
    ) -> Result<Self> {
        let builder = VarBuilder::from_gguf_buffer(buffer, device)?;
        Self::from_var_builder(&builder, config)
    }

    /// Constructs a decoder from an already-loaded quantized variable builder.
    pub fn from_var_builder(builder: &VarBuilder, config: whisper::Config) -> Result<Self> {
        let vb = builder.pp("model.decoder");
        require_q8_weight(
            &vb,
            (config.vocab_size, config.d_model),
            "embed_tokens.weight",
        )?;
        let token_embedding =
            Embedding::new(config.vocab_size, config.d_model, vb.pp("embed_tokens"))?;
        let positional_embedding = vb
            .get(
                (config.max_target_positions, config.d_model),
                "embed_positions.weight",
            )?
            .dequantize(vb.device())?;
        let blocks = (0..config.decoder_layers)
            .map(|index| {
                Q8WhisperBlock::load(
                    config.d_model,
                    config.decoder_attention_heads,
                    vb.pp(format!("layers.{index}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let layer_norm = layer_norm(config.d_model, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            layer_norm,
            max_target_positions: config.max_target_positions,
            cache_start_position: None,
            cached_token_count: 0,
        })
    }

    /// Decodes newly supplied token positions while retaining attention state.
    pub fn forward_incremental(
        &mut self,
        tokens: &Tensor,
        encoder_features: &Tensor,
        position_offset: usize,
        reset_cache: bool,
    ) -> Result<CandleQ8WhisperDecoderOutput> {
        let input_token_count = tokens.dim(D::Minus1)?;
        if input_token_count == 0 {
            candle_core::bail!("Q8 Whisper decoder requires at least one token")
        }
        let position_end = position_offset
            .checked_add(input_token_count)
            .ok_or_else(|| {
                candle_core::Error::msg("Q8 Whisper decoder position range overflowed")
            })?;
        if position_end > self.max_target_positions {
            candle_core::bail!(
                "Q8 Whisper decoder positions {position_offset}..{position_end} exceed max target positions {}",
                self.max_target_positions
            )
        }

        if reset_cache {
            self.reset_cache();
        } else if let Some(cache_start) = self.cache_start_position {
            let expected_offset = cache_start + self.cached_token_count;
            if position_offset != expected_offset {
                candle_core::bail!(
                    "Q8 Whisper decoder expected absolute position offset {expected_offset}, got {position_offset}; reset the cache before starting a new token range"
                )
            }
        }

        let cache_tokens_before = self.cached_token_count;
        let cache_start = self.cache_start_position.unwrap_or(position_offset);
        let token_embedding = self.token_embedding.forward(tokens)?;
        let positional_embedding =
            self.positional_embedding
                .narrow(0, position_offset, input_token_count)?;
        let mut activations = token_embedding.broadcast_add(&positional_embedding)?;
        let mask = causal_mask(
            input_token_count,
            cache_tokens_before + input_token_count,
            position_offset,
            cache_start,
            activations.device(),
        )?;
        let mut self_cache_reused = false;
        let mut cross_cache_computed = false;
        let mut cross_cache_reused = false;
        let mut cross_attention_projection_count = 0;
        for block in &mut self.blocks {
            let (next, stats) = block.forward(&activations, encoder_features, &mask)?;
            activations = next;
            self_cache_reused |= stats.self_cache_reused;
            cross_cache_computed |= stats.cross_cache_computed;
            cross_cache_reused |= stats.cross_cache_reused;
            cross_attention_projection_count += usize::from(stats.cross_cache_computed);
        }
        let activations = self.layer_norm.forward(&activations)?;
        self.cache_start_position = Some(cache_start);
        self.cached_token_count += input_token_count;

        Ok(CandleQ8WhisperDecoderOutput {
            activations,
            diagnostics: CandleQ8WhisperDecoderDiagnostics {
                cache_reset: reset_cache,
                position_offset,
                input_token_count,
                self_attention_cache_tokens_before: cache_tokens_before,
                self_attention_cache_tokens_after: self.cached_token_count,
                self_attention_cache_reused: self_cache_reused,
                cross_attention_cache_computed: cross_cache_computed,
                cross_attention_cache_reused: cross_cache_reused,
                cross_attention_projection_count,
            },
        })
    }

    /// Projects decoder activations to vocabulary logits with tied embeddings.
    pub fn project_logits(&self, activations: &Tensor) -> Result<Tensor> {
        let batch_size = activations.dim(0)?;
        let weight = self
            .token_embedding
            .embeddings()
            .broadcast_left(batch_size)?;
        activations.matmul(&weight.t()?)
    }

    /// Greedily decodes one or more encoder rows through the incremental cache path.
    ///
    /// All rows share the same Whisper prompt and maximum generated-token count.
    /// Rows that emit `eos_token_id` are removed immediately while surviving
    /// encoder features and decoder caches retain their original relative order.
    pub fn decode_greedy_batch(
        &mut self,
        prompt_tokens: &[u32],
        encoder_features: &Tensor,
        eos_token_id: u32,
        max_generated_tokens: usize,
    ) -> Result<CandleQ8WhisperBatchOutput> {
        if prompt_tokens.is_empty() {
            candle_core::bail!("Q8 Whisper batch decode requires at least one prompt token")
        }
        if max_generated_tokens == 0 {
            candle_core::bail!("Q8 Whisper batch decode requires at least one generated token")
        }
        let row_count = encoder_features.dim(0)?;
        if row_count == 0 {
            candle_core::bail!("Q8 Whisper batch decode requires at least one encoder row")
        }
        if prompt_tokens.len() >= self.max_target_positions {
            candle_core::bail!(
                "Q8 Whisper prompt length {} leaves no generated-token position within max target positions {}",
                prompt_tokens.len(),
                self.max_target_positions
            )
        }
        let generation_limit =
            max_generated_tokens.min(self.max_target_positions - prompt_tokens.len());
        let mut active_rows = (0..row_count)
            .map(|original_index| (original_index, Vec::<u32>::new()))
            .collect::<Vec<_>>();
        let mut active_features = encoder_features.clone();
        let mut completed = vec![None; row_count];
        let mut diagnostics = CandleQ8WhisperBatchDiagnostics::default();

        while !active_rows.is_empty() {
            let active_count = active_rows.len();
            diagnostics
                .effective_active_row_batch_sizes
                .push(active_count);
            let (input_tokens, input_len, position_offset, reset_cache) =
                if active_rows[0].1.is_empty() {
                    (
                        prompt_tokens.repeat(active_count),
                        prompt_tokens.len(),
                        0,
                        true,
                    )
                } else {
                    (
                        active_rows
                            .iter()
                            .map(|(_, generated)| {
                                *generated
                                    .last()
                                    .expect("active Q8 row has a generated token after prefill")
                            })
                            .collect(),
                        1,
                        prompt_tokens.len() + active_rows[0].1.len() - 1,
                        false,
                    )
                };
            debug_assert!(active_rows
                .iter()
                .all(|(_, generated)| generated.len() == active_rows[0].1.len()));
            let tokens = Tensor::from_vec(
                input_tokens,
                (active_count, input_len),
                encoder_features.device(),
            )?;
            let decoded =
                self.forward_incremental(&tokens, &active_features, position_offset, reset_cache)?;
            diagnostics.self_attention_cache_reuse_count +=
                usize::from(decoded.diagnostics.self_attention_cache_reused);
            diagnostics.cross_attention_cache_reuse_count +=
                usize::from(decoded.diagnostics.cross_attention_cache_reused);
            diagnostics.cross_attention_projection_count +=
                decoded.diagnostics.cross_attention_projection_count;
            let logits = self.project_logits(&decoded.activations)?;
            let sequence_index = input_len - 1;
            let mut survivors = Vec::with_capacity(active_count);
            let mut survivor_indices = Vec::with_capacity(active_count);
            for (active_index, (original_index, mut generated)) in
                active_rows.into_iter().enumerate()
            {
                let row_logits = logits
                    .i((active_index, sequence_index, ..))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                let next_token = row_logits
                    .iter()
                    .enumerate()
                    .filter(|(_, value)| value.is_finite())
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(token, _)| token as u32)
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "Q8 Whisper batch decode produced no finite token logits",
                        )
                    })?;
                if next_token == eos_token_id {
                    completed[original_index] = Some(generated);
                    continue;
                }
                generated.push(next_token);
                diagnostics.generated_token_count += 1;
                if generated.len() >= generation_limit {
                    completed[original_index] = Some(generated);
                } else {
                    survivor_indices.push(active_index as u32);
                    survivors.push((original_index, generated));
                }
            }
            if survivors.is_empty() {
                break;
            }
            if survivors.len() < active_count {
                diagnostics.active_row_compaction_count += 1;
                let row_indices =
                    Tensor::new(survivor_indices.as_slice(), encoder_features.device())?
                        .to_dtype(DType::I64)?;
                active_features = active_features.index_select(&row_indices, 0)?;
                self.select_cache_rows(&row_indices)?;
            }
            active_rows = survivors;
        }

        let token_ids = completed
            .into_iter()
            .map(|tokens| {
                tokens.ok_or_else(|| {
                    candle_core::Error::msg("Q8 Whisper batch decode lost an output row")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(CandleQ8WhisperBatchOutput {
            token_ids,
            diagnostics,
        })
    }

    /// Clears self- and cross-attention state for a new audio window.
    pub fn reset_cache(&mut self) {
        for block in &mut self.blocks {
            block.reset_cache();
        }
        self.cache_start_position = None;
        self.cached_token_count = 0;
    }

    fn select_cache_rows(&mut self, row_indices: &Tensor) -> Result<()> {
        for block in &mut self.blocks {
            block.select_cache_rows(row_indices)?;
        }
        Ok(())
    }
}

fn require_q8_weight(
    builder: &VarBuilder,
    shape: impl Into<candle_core::Shape>,
    name: &str,
) -> Result<()> {
    let tensor = builder.get(shape, name)?;
    if tensor.dtype() != GgmlDType::Q8_0 {
        candle_core::bail!(
            "Q8 Whisper decoder tensor `{name}` must use Q8_0, got {:?}",
            tensor.dtype()
        )
    }
    Ok(())
}

fn causal_mask(
    query_len: usize,
    key_len: usize,
    query_position_offset: usize,
    key_position_offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let values = (0..query_len)
        .flat_map(|query_index| {
            let query_position = query_position_offset + query_index;
            (0..key_len).map(move |key_index| {
                let key_position = key_position_offset + key_index;
                if key_position > query_position {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (query_len, key_len), device)
}
