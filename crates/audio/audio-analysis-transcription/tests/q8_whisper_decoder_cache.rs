#![cfg(feature = "candle")]

use std::io::Cursor;

use audio_analysis_transcription::CandleQ8WhisperDecoder;
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::whisper;

fn tiny_config() -> whisper::Config {
    whisper::Config {
        num_mel_bins: 2,
        max_source_positions: 4,
        d_model: 32,
        encoder_attention_heads: 4,
        encoder_layers: 1,
        vocab_size: 32,
        max_target_positions: 8,
        decoder_attention_heads: 4,
        decoder_layers: 1,
        suppress_tokens: Vec::new(),
    }
}

fn values(element_count: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..element_count)
        .map(|index| {
            let centered = ((index * 17 + seed * 29) % 97) as f32 - 48.0;
            centered * scale
        })
        .collect()
}

fn qtensor(
    shape: impl Into<candle_core::Shape>,
    seed: usize,
    scale: f32,
    dtype: GgmlDType,
) -> QTensor {
    let shape = shape.into();
    let tensor = Tensor::from_vec(
        values(shape.elem_count(), seed, scale),
        shape.clone(),
        &Device::Cpu,
    )
    .unwrap();
    QTensor::quantize(&tensor, dtype).unwrap()
}

fn constant_qtensor(shape: impl Into<candle_core::Shape>, value: f32, dtype: GgmlDType) -> QTensor {
    let shape = shape.into();
    let tensor =
        Tensor::from_vec(vec![value; shape.elem_count()], shape.clone(), &Device::Cpu).unwrap();
    QTensor::quantize(&tensor, dtype).unwrap()
}

fn tiny_q8_decoder_gguf() -> Vec<u8> {
    let config = tiny_config();
    let d_model = config.d_model;
    let d_mlp = d_model * 4;
    let mut tensors = Vec::<(String, QTensor)>::new();
    let mut push = |name: &str, tensor: QTensor| tensors.push((name.to_string(), tensor));

    push(
        "model.decoder.embed_tokens.weight",
        qtensor((config.vocab_size, d_model), 1, 0.004, GgmlDType::Q8_0),
    );
    push(
        "model.decoder.embed_positions.weight",
        qtensor(
            (config.max_target_positions, d_model),
            2,
            0.006,
            GgmlDType::F32,
        ),
    );

    let layer = "model.decoder.layers.0";
    for (attention_index, attention) in ["self_attn", "encoder_attn"].into_iter().enumerate() {
        for (projection_index, projection) in
            ["q_proj", "v_proj", "out_proj"].into_iter().enumerate()
        {
            push(
                &format!("{layer}.{attention}.{projection}.weight"),
                qtensor(
                    (d_model, d_model),
                    10 + attention_index * 10 + projection_index,
                    0.003,
                    GgmlDType::Q8_0,
                ),
            );
            push(
                &format!("{layer}.{attention}.{projection}.bias"),
                qtensor(
                    d_model,
                    40 + attention_index * 10 + projection_index,
                    0.001,
                    GgmlDType::F32,
                ),
            );
        }
        push(
            &format!("{layer}.{attention}.k_proj.weight"),
            qtensor(
                (d_model, d_model),
                30 + attention_index,
                0.003,
                GgmlDType::Q8_0,
            ),
        );
    }

    for (index, name) in [
        "self_attn_layer_norm",
        "encoder_attn_layer_norm",
        "final_layer_norm",
    ]
    .into_iter()
    .enumerate()
    {
        push(
            &format!("{layer}.{name}.weight"),
            constant_qtensor(d_model, 1.0, GgmlDType::F32),
        );
        push(
            &format!("{layer}.{name}.bias"),
            qtensor(d_model, 60 + index, 0.001, GgmlDType::F32),
        );
    }

    push(
        &format!("{layer}.fc1.weight"),
        qtensor((d_mlp, d_model), 70, 0.002, GgmlDType::Q8_0),
    );
    push(
        &format!("{layer}.fc1.bias"),
        qtensor(d_mlp, 71, 0.001, GgmlDType::F32),
    );
    push(
        &format!("{layer}.fc2.weight"),
        qtensor((d_model, d_mlp), 72, 0.002, GgmlDType::Q8_0),
    );
    push(
        &format!("{layer}.fc2.bias"),
        qtensor(d_model, 73, 0.001, GgmlDType::F32),
    );
    push(
        "model.decoder.layer_norm.weight",
        constant_qtensor(d_model, 1.0, GgmlDType::F32),
    );
    push(
        "model.decoder.layer_norm.bias",
        qtensor(d_model, 80, 0.001, GgmlDType::F32),
    );

    let refs = tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect::<Vec<_>>();
    let mut buffer = Cursor::new(Vec::new());
    gguf_file::write(&mut buffer, &[], &refs).unwrap();
    buffer.into_inner()
}

fn encoder_features() -> Tensor {
    encoder_features_with_seed(91)
}

fn encoder_features_with_seed(seed: usize) -> Tensor {
    Tensor::from_vec(values(3 * 32, seed, 0.01), (1, 3, 32), &Device::Cpu).unwrap()
}

fn last_logits(
    decoder: &CandleQ8WhisperDecoder,
    activations: &Tensor,
) -> candle_core::Result<Vec<f32>> {
    let logits = decoder.project_logits(activations)?;
    let sequence_len = logits.dim(1)?;
    logits.i((0, sequence_len - 1, ..))?.to_vec1()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let max_difference = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_difference < 1e-4,
        "maximum logit difference was {max_difference}"
    );
}

#[test]
fn q8_cached_token_steps_match_full_prefix_logits_and_report_cache_use() {
    let config = tiny_config();
    let fixture = tiny_q8_decoder_gguf();
    let encoder = encoder_features();
    let mut full =
        CandleQ8WhisperDecoder::from_gguf_buffer(&fixture, config.clone(), &Device::Cpu).unwrap();
    let mut cached =
        CandleQ8WhisperDecoder::from_gguf_buffer(&fixture, config, &Device::Cpu).unwrap();

    let prefix_two = Tensor::new(&[[1_u32, 2]], &Device::Cpu).unwrap();
    let prefill = cached
        .forward_incremental(&prefix_two, &encoder, 0, true)
        .unwrap();
    assert_eq!(prefill.diagnostics.position_offset, 0);
    assert_eq!(prefill.diagnostics.input_token_count, 2);
    assert_eq!(prefill.diagnostics.self_attention_cache_tokens_before, 0);
    assert_eq!(prefill.diagnostics.self_attention_cache_tokens_after, 2);
    assert!(!prefill.diagnostics.self_attention_cache_reused);
    assert!(prefill.diagnostics.cross_attention_cache_computed);
    assert!(!prefill.diagnostics.cross_attention_cache_reused);
    assert_eq!(prefill.diagnostics.cross_attention_projection_count, 1);

    let token_three = Tensor::new(&[[3_u32]], &Device::Cpu).unwrap();
    let cached_three = cached
        .forward_incremental(&token_three, &encoder, 2, false)
        .unwrap();
    let prefix_three = Tensor::new(&[[1_u32, 2, 3]], &Device::Cpu).unwrap();
    let full_three = full
        .forward_incremental(&prefix_three, &encoder, 0, true)
        .unwrap();
    assert_close(
        &last_logits(&cached, &cached_three.activations).unwrap(),
        &last_logits(&full, &full_three.activations).unwrap(),
    );
    assert_eq!(
        cached_three.diagnostics.self_attention_cache_tokens_before,
        2
    );
    assert_eq!(cached_three.diagnostics.position_offset, 2);
    assert_eq!(cached_three.diagnostics.input_token_count, 1);
    assert_eq!(
        cached_three.diagnostics.self_attention_cache_tokens_after,
        3
    );
    assert!(cached_three.diagnostics.self_attention_cache_reused);
    assert!(!cached_three.diagnostics.cross_attention_cache_computed);
    assert!(cached_three.diagnostics.cross_attention_cache_reused);
    assert_eq!(cached_three.diagnostics.cross_attention_projection_count, 0);

    let token_four = Tensor::new(&[[4_u32]], &Device::Cpu).unwrap();
    let cached_four = cached
        .forward_incremental(&token_four, &encoder, 3, false)
        .unwrap();
    let prefix_four = Tensor::new(&[[1_u32, 2, 3, 4]], &Device::Cpu).unwrap();
    let full_four = full
        .forward_incremental(&prefix_four, &encoder, 0, true)
        .unwrap();
    assert_close(
        &last_logits(&cached, &cached_four.activations).unwrap(),
        &last_logits(&full, &full_four.activations).unwrap(),
    );

    let reset_prefill = cached
        .forward_incremental(&prefix_two, &encoder, 0, true)
        .unwrap();
    assert_close(
        &last_logits(&cached, &reset_prefill.activations).unwrap(),
        &last_logits(&cached, &prefill.activations).unwrap(),
    );
    assert_eq!(
        reset_prefill.diagnostics.self_attention_cache_tokens_before,
        0
    );
    assert_eq!(
        reset_prefill.diagnostics.self_attention_cache_tokens_after,
        2
    );
    assert!(reset_prefill.diagnostics.cache_reset);
    assert!(reset_prefill.diagnostics.cross_attention_cache_computed);
}

#[test]
fn q8_cached_decode_rejects_a_discontinuous_absolute_position() {
    let fixture = tiny_q8_decoder_gguf();
    let mut decoder =
        CandleQ8WhisperDecoder::from_gguf_buffer(&fixture, tiny_config(), &Device::Cpu).unwrap();
    let encoder = encoder_features();
    decoder
        .forward_incremental(
            &Tensor::new(&[[1_u32, 2]], &Device::Cpu).unwrap(),
            &encoder,
            0,
            true,
        )
        .unwrap();

    let error = decoder
        .forward_incremental(
            &Tensor::new(&[[3_u32]], &Device::Cpu).unwrap(),
            &encoder,
            4,
            false,
        )
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("expected absolute position offset 2, got 4"),
        "{error}"
    );
}

#[test]
fn q8_one_row_greedy_decode_uses_incremental_self_and_cross_attention_caches() {
    let fixture = tiny_q8_decoder_gguf();
    let mut decoder =
        CandleQ8WhisperDecoder::from_gguf_buffer(&fixture, tiny_config(), &Device::Cpu).unwrap();

    let output = decoder
        .decode_greedy_batch(&[1, 2], &encoder_features(), u32::MAX, 2)
        .unwrap();

    assert_eq!(output.token_ids.len(), 1);
    assert_eq!(output.token_ids[0].len(), 2);
    assert_eq!(
        output.diagnostics.effective_active_row_batch_sizes,
        vec![1, 1]
    );
    assert_eq!(output.diagnostics.active_row_compaction_count, 0);
    assert_eq!(output.diagnostics.self_attention_cache_reuse_count, 1);
    assert_eq!(output.diagnostics.cross_attention_cache_reuse_count, 1);
    assert_eq!(output.diagnostics.generated_token_count, 2);
}

#[test]
fn q8_active_row_batch_matches_sequential_order_after_unequal_completion() {
    let fixture = tiny_q8_decoder_gguf();
    let prompt = [1_u32, 2];
    let max_generated_tokens = 4;
    let mut discovery =
        CandleQ8WhisperDecoder::from_gguf_buffer(&fixture, tiny_config(), &Device::Cpu).unwrap();
    let candidates = (0..32)
        .map(|seed| {
            let features = encoder_features_with_seed(seed);
            let tokens = discovery
                .decode_greedy_batch(&prompt, &features, u32::MAX, max_generated_tokens)
                .unwrap()
                .token_ids
                .remove(0);
            (features, tokens)
        })
        .collect::<Vec<_>>();
    let (first_index, second_index, eos) = candidates
        .iter()
        .enumerate()
        .find_map(|(first_index, (_, first_tokens))| {
            first_tokens.iter().copied().find_map(|eos| {
                let first_completion = first_tokens.iter().position(|token| *token == eos)?;
                candidates
                    .iter()
                    .enumerate()
                    .find_map(|(second_index, (_, second_tokens))| {
                        let second_completion = second_tokens
                            .iter()
                            .position(|token| *token == eos)
                            .unwrap_or(max_generated_tokens);
                        (first_completion != second_completion).then_some((
                            first_index,
                            second_index,
                            eos,
                        ))
                    })
            })
        })
        .expect("deterministic Q8 fixture should expose unequal EOS completion");
    let encoder_batch = Tensor::cat(
        &[&candidates[first_index].0, &candidates[second_index].0],
        0,
    )
    .unwrap();

    let sequential = [first_index, second_index]
        .into_iter()
        .map(|index| {
            CandleQ8WhisperDecoder::from_gguf_buffer(&fixture, tiny_config(), &Device::Cpu)
                .unwrap()
                .decode_greedy_batch(&prompt, &candidates[index].0, eos, max_generated_tokens)
                .unwrap()
                .token_ids
                .remove(0)
        })
        .collect::<Vec<_>>();
    assert_ne!(sequential[0].len(), sequential[1].len());

    let mut batched =
        CandleQ8WhisperDecoder::from_gguf_buffer(&fixture, tiny_config(), &Device::Cpu).unwrap();
    let output = batched
        .decode_greedy_batch(&prompt, &encoder_batch, eos, max_generated_tokens)
        .unwrap();

    assert_eq!(output.token_ids, sequential);
    assert_eq!(output.diagnostics.effective_active_row_batch_sizes[0], 2);
    assert!(output
        .diagnostics
        .effective_active_row_batch_sizes
        .contains(&1));
    assert_eq!(output.diagnostics.active_row_compaction_count, 1);
    assert_eq!(
        output.diagnostics.generated_token_count,
        output.token_ids.iter().map(Vec::len).sum::<usize>()
    );
    assert!(output.diagnostics.self_attention_cache_reuse_count > 0);
    assert!(output.diagnostics.cross_attention_cache_reuse_count > 0);
}
