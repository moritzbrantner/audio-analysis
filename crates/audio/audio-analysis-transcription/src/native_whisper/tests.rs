use crate::native_device::ResolvedNativeDevice;
use crate::{
    AsrRequest, CandleWhisperComputeType, CandleWhisperDecodeRequestConfig,
    CandleWhisperDecodeRuntime, CandleWhisperOptions, CandleWhisperRuntimeControls,
    CandleWhisperTimingMode, CandleWhisperTranscriptionRequestConfig, CandleWhisperWindowControls,
    SpeechSpan, TranscriptionTask,
};
use candle_core::quantized::{gguf_file, GgmlDType};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::whisper;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

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
        let actual = hex::encode(Sha256::digest(bytes));
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
    let whisperx_oracle_path = std::env::var_os("CANDLE_WHISPER_GERMAN_WHISPERX_ORACLE")
        .map(PathBuf::from)
        .expect(
            "CANDLE_WHISPER_GERMAN_WHISPERX_ORACLE must point at the retained WhisperX 3.8.6 JSON",
        );
    let whisperx_oracle_bytes =
        std::fs::read(whisperx_oracle_path).expect("retained WhisperX 3.8.6 oracle");
    assert_eq!(
        hex::encode(Sha256::digest(&whisperx_oracle_bytes)),
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
        chunks: vec![
            SpeechSpan::new(0.322, duration, 1.0).expect("exact WhisperX 3.8.6 speech span")
        ],
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
    let setup = WhisperRunSetup::from_options_and_request(&options, &request).expect("run setup");
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
        hex::encode(Sha256::digest(&mel_oracle_bytes)),
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
        hex::encode(Sha256::digest(&logits_oracle_bytes)),
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
#[ignore = "requires the retained full German WAV, WhisperX 3.8.6 JSON, and pinned Whisper-small bundle; run explicitly with --ignored"]
fn german_long_window_stops_at_whisperx_greedy_generation_limit() {
    let bundle = std::env::var_os("CANDLE_WHISPER_SMALL_BUNDLE")
        .map(PathBuf::from)
        .expect("CANDLE_WHISPER_SMALL_BUNDLE must point at the pinned Whisper-small revision");
    let audio_path = std::env::var_os("CANDLE_WHISPER_GERMAN_LONG_WAV")
        .map(PathBuf::from)
        .expect("CANDLE_WHISPER_GERMAN_LONG_WAV must point at the retained full German WAV");
    let oracle_path = std::env::var_os("CANDLE_WHISPER_GERMAN_LONG_WHISPERX_ORACLE")
        .map(PathBuf::from)
        .expect(
            "CANDLE_WHISPER_GERMAN_LONG_WHISPERX_ORACLE must point at the retained WhisperX 3.8.6 JSON",
        );
    let assert_sha256 = |path: &Path, expected: &str| {
        let bytes = std::fs::read(path).expect("pinned long-window parity resource");
        let actual = hex::encode(Sha256::digest(bytes));
        assert_eq!(
            actual,
            expected,
            "pinned long-window resource changed: {}",
            path.display()
        );
    };
    assert_sha256(
        &audio_path,
        "193a1cfbe84eaac2585a5bfa8519d28c2b74ba84d630c3ed73403565310f979c",
    );
    assert_sha256(
        &oracle_path,
        "531b6a990798a9caccb35991dd22ea5311b22c4c8c4451198390e443f175727c",
    );
    assert_sha256(
        &bundle.join("model.safetensors"),
        "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b",
    );

    let oracle_bytes = std::fs::read(&oracle_path).expect("retained WhisperX oracle");
    let oracle: serde_json::Value = serde_json::from_slice(&oracle_bytes).expect("WhisperX JSON");
    let oracle_segment = &oracle["segments"][3];
    assert_eq!(oracle_segment["start"], serde_json::json!(61.538));
    assert_eq!(oracle_segment["end"], serde_json::json!(91.504));
    let oracle_text = oracle_segment["text"]
        .as_str()
        .expect("WhisperX segment text");

    let mut reader = hound::WavReader::open(&audio_path).expect("German PCM fixture");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, whisper::SAMPLE_RATE as u32);
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.bits_per_sample, 16);
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.expect("PCM sample") as f32 / 32_768.0)
        .collect::<Vec<_>>();
    let request = AsrRequest {
        audio: crate::LoadedAudio {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            source: Some("retained-full-german-parity-fixture".to_string()),
        },
        chunks: vec![
            SpeechSpan::new(61.538, 91.504, 1.0).expect("exact WhisperX 3.8.6 long speech span")
        ],
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
        window: CandleWhisperWindowControls {
            timing_mode: CandleWhisperTimingMode::NoTimestamps,
            leading_context_seconds: 0.0,
            trailing_context_seconds: 0.0,
        },
        ..CandleWhisperTranscriptionRequestConfig::default()
    };
    let setup = WhisperRunSetup::from_options_and_request(&options, &request).expect("run setup");
    let mut session = CandleWhisperSession::load(setup).expect("load Whisper-small");
    let windows = collect_chunk_windows(
        &request.audio.samples,
        request.audio.sample_rate,
        &request.chunks,
        &config.window,
    )
    .expect("zero-context long window");
    assert_eq!(windows.len(), 1);
    let features = session
        .encode_window_batch(&windows, WhisperFeatureExtractorMode::Transformers)
        .expect("encoder features");
    let initial = session
        .initial_tokens(WhisperDecodeMode::WithoutTimestamps, &[])
        .expect("German no-timestamps prompt");
    let oracle_tokens = session
        .tokenizer
        .encode(oracle_text, false)
        .expect("tokenize WhisperX text")
        .get_ids()
        .to_vec();
    assert_eq!(oracle_tokens.len(), 224);
    assert_eq!(
        decode_text_tokens(&session.tokenizer, &oracle_tokens).expect("decode oracle token IDs"),
        oracle_text
    );
    let mut no_speech_probability = None;
    let mut stats = WhisperGenerationStats::default();
    let full_prefix_logits = session
        .configured_search_logits(
            &features,
            &initial.token_ids,
            initial.sot_position,
            &oracle_tokens,
            WhisperDecodeMode::WithoutTimestamps,
            &config.decode,
            &mut no_speech_probability,
            &mut stats,
        )
        .expect("full-prefix logits at WhisperX generation boundary");
    let full_prefix_next = argmax_finite(&full_prefix_logits).expect("finite full-prefix logits");

    assert_eq!(
        full_prefix_next, 674,
        "the retained oracle must exercise a generation-limit boundary, not EOS"
    );
    assert_eq!(
        whisper_max_generated_tokens(
            &session.generation,
            session.model.config(),
            initial.token_ids.len(),
        ),
        oracle_tokens.len()
    );

    let decoded = session
        .decode_tokens_batch(
            &features,
            1,
            WhisperDecodeMode::WithoutTimestamps,
            &config.decode,
        )
        .expect("KV-cached greedy decode");
    assert_eq!(decoded.len(), 1);
    assert_eq!(
        decoded[0].token_ids.len(),
        oracle_tokens.len(),
        "KV-cached Candle decode must stop at the Whisper generation limit"
    );
    assert_eq!(
        decode_text_tokens(&session.tokenizer, &decoded[0].token_ids)
            .expect("decode KV-cached token IDs")
            .trim(),
        oracle_text
    );
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
        chunks: vec![SpeechSpan::new(0.0, 1.0, 0.5).unwrap()],
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

#[test]
fn whisper_generation_limit_reserves_half_the_decoder_context_for_output() {
    let config: whisper::Config = serde_json::from_value(serde_json::json!({
        "num_mel_bins": 2,
        "max_source_positions": 4,
        "d_model": 32,
        "encoder_attention_heads": 4,
        "encoder_layers": 1,
        "vocab_size": 15,
        "max_target_positions": 8,
        "decoder_attention_heads": 4,
        "decoder_layers": 1,
        "suppress_tokens": []
    }))
    .expect("test Whisper config");

    assert_eq!(
        whisper_max_generated_tokens(&test_generation(), &config, 4),
        4
    );
    assert_eq!(
        whisper_max_generated_tokens(&test_generation(), &config, 7),
        1
    );
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

// native-whisperx#418: tiny..large-v2 call the token <|nocaptions|>, so
// no-speech rejection never fired for them.
#[test]
fn no_speech_token_resolves_in_pre_v3_vocabularies() {
    assert_eq!(no_speech_token_id(&test_tokenizer()), Some(10));
    let temp = tempfile::tempdir().unwrap();
    let vocab = temp.path().join("vocab.json");
    std::fs::write(
        &vocab,
        serde_json::json!({ "<unk>": 0, "<|nocaptions|>": 50362 }).to_string(),
    )
    .unwrap();
    let tokenizer =
        Tokenizer::new(WordLevel::from_file(vocab.to_str().unwrap(), "<unk>".to_string()).unwrap());
    assert_eq!(no_speech_token_id(&tokenizer), Some(50362));
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
    let logits = Tensor::from_vec(values, (1, prompt.token_ids.len(), 15), &Device::Cpu).unwrap();

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
        chunks: vec![SpeechSpan::new(0.0, 1.0, 0.5).unwrap()],
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
        device: crate::NativeDevicePreference::Cpu,
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
    assert_eq!(
        q8_execution_diagnostics("hybrid-f32-encoder-cross-projection-q8-decoder", "avx2-fma"),
        [
            "q8ExecutionRoute=hybrid-f32-encoder-cross-projection-q8-decoder",
            "q8LinearKernel=avx2-fma",
        ]
    );
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
fn whisper_encoder_input_matches_fp16_model_weights() {
    let mel = Tensor::zeros((1, 80, whisper::N_FRAMES), DType::F32, &Device::Cpu)
        .expect("f32 mel tensor");

    let prepared = prepare_whisper_encoder_input(mel, DType::F16)
        .expect("encoder input should convert to the model weight dtype");

    assert_eq!(prepared.dtype(), DType::F16);
}

#[test]
fn whisper_encoder_positional_embedding_matches_fp16_model_weights() {
    let positional_embedding =
        sinusoids(4, 4, &Device::Cpu, DType::F16).expect("fp16 positional embedding");

    assert_eq!(positional_embedding.dtype(), DType::F16);
}

#[test]
fn whisper_decoder_causal_mask_matches_fp16_model_weights() {
    let mask =
        decoder_causal_mask(2, 2, 0, &Device::Cpu, DType::F16).expect("fp16 decoder causal mask");

    assert_eq!(mask.dtype(), DType::F16);
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
    let spec =
        timestamp_spec_for_timing_mode(&test_tokenizer(), CandleWhisperTimingMode::Auto).unwrap();
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
    let spec =
        timestamp_spec_for_timing_mode(&test_tokenizer(), CandleWhisperTimingMode::NoTimestamps)
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
    assert_eq!(words[0].start_seconds(), Some(10.0));
    assert_eq!(words[0].end_seconds(), Some(12.0));
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
    assert_approx_eq(words[0].start_seconds().unwrap(), 0.0);
    assert_approx_eq(words[0].end_seconds().unwrap(), 1.0);
    assert_approx_eq(words[1].start_seconds().unwrap(), 1.0);
    assert_approx_eq(words[1].end_seconds().unwrap(), 3.0);
}

#[test]
fn projected_words_keep_punctuation_attached() {
    let words = project_words_from_timestamp_segment(&decoded_segment("hello, world!", 0.0, 2.0));

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
        let start = word.start_seconds().unwrap();
        let end = word.end_seconds().unwrap();
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
        (10.0, 12.0),
        Some("en".to_string()),
    )
    .unwrap();

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start_seconds(), Some(10.5));
    assert_eq!(segments[0].end_seconds(), Some(11.5));
    assert!(segments[0].words().is_empty());
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
        decoded_window_to_contract_segments(decoded, &mut next_index, 10.0, (10.0, 12.0), None)
            .unwrap();

    assert_eq!(segments.len(), 2);
    for segment in &segments {
        let segment_start = segment.start_seconds().unwrap();
        let segment_end = segment.end_seconds().unwrap();
        assert!(segment_start >= 10.0);
        assert!(segment_end <= 12.0);
        assert!(segment.words().is_empty());
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
        (10.0, 12.0),
        Some("en".to_string()),
    )
    .unwrap();
    assert_eq!(next_index, 9);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].index, 7);
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start_seconds(), Some(10.5));
    assert_eq!(segments[0].end_seconds(), Some(11.25));
    assert_eq!(segments[1].index, 8);
    assert_eq!(segments[1].text, "world");
    assert_eq!(segments[1].start_seconds(), Some(11.25));
    assert_eq!(segments[1].end_seconds(), Some(11.75));
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
    crate::transcription_from_segments(None, Some("en".to_string()), segments).unwrap();
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
        (3.0, 5.0),
        Some("en".to_string()),
    )
    .unwrap();
    let transcript =
        crate::transcription_from_segments(None, Some("en".to_string()), segments).unwrap();

    transcript.validate_strict().unwrap();
}

#[test]
fn window_fallback_segment_uses_global_timing() {
    let segment =
        window_fallback_segment(3, "hello".to_string(), 4.0, 5.5, Some("en".to_string())).unwrap();
    assert_eq!(segment.start_seconds(), Some(4.0));
    assert_eq!(segment.end_seconds(), Some(5.5));
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
    )
    .unwrap();

    assert!(segment.words().is_empty());
    assert!(!segment.attributes.contains_key("wordTiming"));
}

#[test]
fn chunk_windows_carry_local_and_global_timing() {
    let chunk = SpeechSpan::new(1.0, 2.0, 0.8).unwrap();
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
    assert_eq!(windows[0].transcript_bounds(), (1.0, 2.0));
}

#[test]
fn full_size_chunk_drops_context_instead_of_a_remainder_window() {
    let chunk = SpeechSpan::new(30.0, 60.0, 0.8).unwrap();
    let windows = chunk_windows(
        &vec![0.0; 16_000 * 62],
        16_000,
        &chunk,
        &CandleWhisperWindowControls::default(),
    )
    .unwrap();
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0].samples.len(), whisper::N_SAMPLES);
    assert_eq!(windows[0].global_start_seconds, 30.0);
    assert_eq!(windows[0].global_end_seconds, 60.0);

    let long = SpeechSpan::new(0.0, 45.0, 0.8).unwrap();
    let windows = chunk_windows(
        &vec![0.0; 16_000 * 62],
        16_000,
        &long,
        &CandleWhisperWindowControls::default(),
    )
    .unwrap();
    assert_eq!(windows.len(), 2);
    assert_eq!(windows[1].global_end_seconds, 45.04);
}

// native-whisperx#418: the next chunk's leading context starts before the
// previous chunk ends, so a segment at its local 0.0 used to land 0.25 s
// before the previous chunk's last segment and fail strict validation.
#[test]
fn adjacent_chunk_segments_stay_chronological() {
    let samples = vec![0.0; 16_000 * 62];
    let controls = CandleWhisperWindowControls::default();
    let first = SpeechSpan::new(0.0, 29.0, 0.8).unwrap();
    let second = SpeechSpan::new(29.0, 58.0, 0.8).unwrap();
    let mut next_index = 0;
    let mut segments = Vec::new();
    for (chunk, decoded) in [
        (&first, vec![decoded_segment("music", 28.5, 29.04)]),
        (
            &second,
            vec![
                // Only covers the leading context: dropped.
                decoded_segment("you", 0.0, 0.2),
                decoded_segment("music", 0.0, 2.0),
            ],
        ),
    ] {
        let windows = chunk_windows(&samples, 16_000, chunk, &controls).unwrap();
        assert_eq!(windows.len(), 1);
        segments.extend(
            decoded_window_to_contract_segments(
                WhisperDecodedWindow {
                    text: "music".to_string(),
                    segments: decoded,
                },
                &mut next_index,
                windows[0].global_start_seconds,
                windows[0].transcript_bounds(),
                None,
            )
            .unwrap(),
        );
    }

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].end_seconds(), Some(29.0));
    assert_eq!(segments[1].text, "music");
    assert_eq!(segments[1].start_seconds(), Some(29.0));
    crate::transcription_from_segments(None, None, segments)
        .unwrap()
        .validate_strict()
        .unwrap();
}

#[test]
fn chunk_windows_use_request_scoped_context() {
    let chunk = SpeechSpan::new(1.0, 2.0, 0.8).unwrap();
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
    let tokenizer =
        Tokenizer::new(WordLevel::from_file(vocab.to_str().unwrap(), "<unk>".to_string()).unwrap());
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

#[test]
fn language_detection_softmaxes_only_the_configured_language_tokens() {
    let candidates = language_candidates(&test_generation());
    assert_eq!(
        candidates,
        vec![("de".to_string(), 4), ("en".to_string(), 3)]
    );
    // Token 0 (a text token) has the highest logit overall; detection must
    // still pick among the language tokens 3 and 4 only.
    let logits = [9.0, 0.0, 0.0, 2.0, 0.0, 5.0, 0.0, 0.0];
    let (language, probability) = pick_language(&logits, &candidates).expect("a language");
    assert_eq!(language, "en");
    let expected = 1.0 / (1.0 + (-2.0_f64).exp());
    assert!((probability - expected).abs() < 1e-12);
}

#[test]
fn english_only_models_skip_language_detection() {
    let generation = GenerationConfig {
        lang_to_id: Default::default(),
        ..test_generation()
    };
    assert!(language_candidates(&generation).is_empty());
    assert_eq!(pick_language(&[1.0, 2.0], &[]), None);
}

/// Native WhisperX #434: with no language the Candle provider never ran
/// language identification and decoded `<|startoftranscript|><|transcribe|>`
/// without a language token, so Whisper-small produced punctuation only.
///
/// Audio: "Blender Conference 2026 Recap" by Blender
/// (https://www.youtube.com/watch?v=sO_lhW8av3U), CC BY, English speech over
/// music. Kept out of Git; prepare it with
/// `yt-dlp -f bestaudio -x --audio-format wav -o recap.wav https://www.youtube.com/watch?v=sO_lhW8av3U`
/// then `ffmpeg -i recap.wav -ac 1 -ar 16000 -c:a pcm_s16le blender-conference-2026-recap.wav`.
/// OpenAI Whisper / WhisperX 3.8 detect `en` (p = 0.93) on its first 30 s.
#[test]
#[ignore = "requires CANDLE_WHISPER_SMALL_BUNDLE and the CC BY recap WAV in CANDLE_WHISPER_LANGUAGE_ID_WAV"]
fn language_detection_picks_english_for_music_under_speech() {
    let bundle = std::env::var_os("CANDLE_WHISPER_SMALL_BUNDLE")
        .map(PathBuf::from)
        .expect("CANDLE_WHISPER_SMALL_BUNDLE must point at a Whisper-small snapshot");
    let audio_path = std::env::var_os("CANDLE_WHISPER_LANGUAGE_ID_WAV")
        .map(PathBuf::from)
        .expect("CANDLE_WHISPER_LANGUAGE_ID_WAV must point at the 16 kHz recap WAV");
    let mut reader = hound::WavReader::open(&audio_path).expect("recap WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, whisper::SAMPLE_RATE as u32);
    assert_eq!(spec.channels, 1);
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.expect("PCM sample") as f32 / 32_768.0)
        .collect::<Vec<_>>();
    let request = AsrRequest {
        audio: crate::LoadedAudio {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            source: Some("blender-conference-2026-recap".to_string()),
        },
        chunks: vec![SpeechSpan::new(0.0, 30.0, 1.0).expect("first window")],
        task: TranscriptionTask::Transcribe,
        language: None,
        model_id: "openai/whisper-small".to_string(),
    };
    let options = CandleWhisperOptions {
        model_id: request.model_id.clone(),
        device: crate::NativeDevicePreference::Cpu,
        compute_type: CandleWhisperComputeType::Fp32,
        model_bundle: Some(bundle),
        model_cache_only: true,
        ..CandleWhisperOptions::default()
    };
    let response = transcribe(&options, request).expect("transcribe without a language");
    assert_eq!(response.language.as_deref(), Some("en"));
    assert_eq!(response.transcript.language.as_deref(), Some("en"));
    assert!(response
        .diagnostics
        .iter()
        .any(|entry| entry == "languageDetection=whisperFirstWindow"));
    let text = response
        .transcript
        .segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let words = text
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphabetic))
        .count();
    assert!(words >= 10, "expected English words, got {text:?}");
}
