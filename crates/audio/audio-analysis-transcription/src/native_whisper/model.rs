//! Cached Whisper encoder/decoder graph and the model format dispatch.

use crate::native_whisper_quantized::CandleQ8WhisperModel;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    embedding, linear, linear_no_bias, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module,
    VarBuilder,
};
use candle_transformers::models::whisper;

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
        let positional_embedding =
            sinusoids(cfg.max_source_positions, n_state, vb.device(), vb.dtype())?;
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
pub(super) struct CachedWhisperDecoderStats {
    pub(super) self_attention_cache_reused: bool,
    pub(super) cross_attention_cache_reused: bool,
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
            x.dtype(),
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
pub(super) struct CachedWhisper {
    encoder: CachedWhisperEncoder,
    decoder: CachedWhisperDecoder,
    config: whisper::Config,
}

impl CachedWhisper {
    pub(super) fn load(vb: &VarBuilder, config: whisper::Config) -> candle_core::Result<Self> {
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
pub(super) enum WhisperModel {
    Safetensors(CachedWhisper),
    Q8(CandleQ8WhisperModel),
}

impl WhisperModel {
    pub(super) fn config(&self) -> &whisper::Config {
        match self {
            Self::Safetensors(model) => &model.config,
            Self::Q8(model) => &model.config,
        }
    }

    pub(super) fn encode(&mut self, mel: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Safetensors(model) => model.encoder.forward(mel, true),
            Self::Q8(model) => model.encode(mel),
        }
    }

    pub(super) fn decode(
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

    pub(super) fn project_logits(&self, activations: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Safetensors(model) => model.decoder.final_linear(activations),
            Self::Q8(model) => model.project_logits(activations),
        }
    }

    pub(super) fn reset_cache(&mut self) {
        match self {
            Self::Safetensors(model) => model.reset_kv_cache(),
            Self::Q8(model) => model.reset_cache(),
        }
    }

    pub(super) fn select_cache_rows(&mut self, row_indices: &Tensor) -> candle_core::Result<()> {
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

pub(super) fn layer_norm(size: usize, vb: VarBuilder) -> candle_core::Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

pub(super) fn sinusoids(
    length: usize,
    channels: usize,
    device: &Device,
    dtype: DType,
) -> candle_core::Result<Tensor> {
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
    Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?.to_dtype(dtype)
}

pub(super) fn decoder_causal_mask(
    query_len: usize,
    key_len: usize,
    position_offset: usize,
    device: &Device,
    dtype: DType,
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
    Tensor::from_vec(values, (query_len, key_len), device)?.to_dtype(dtype)
}
