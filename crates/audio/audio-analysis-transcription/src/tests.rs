use super::*;
use media_core::TranscriptSegmentContract;

#[test]
fn whisperx_import_preserves_provider_fields_and_nested_attributes() {
    let transcript = import_whisperx_json(
        br#"{
            "text": "hi there",
            "language": "en",
            "segment_count": 1,
            "providerRevision": "fixture-v1",
            "segments": [{
                "id": 7,
                "start": 1.0,
                "end": 2.0,
                "text": " hi there ",
                "words": [
                    {"word": "hi", "start": 1.1, "end": 1.3, "score": 0.9, "speaker": "S0", "token": 4}
                ],
                "chars": [
                    {"char": "h", "start": 1.1, "end": 1.2, "score": 0.8, "extra": "kept"}
                ]
            }]
        }"#,
    )
    .unwrap();

    assert_eq!(transcript.segments[0].index, 7);
    assert_eq!(transcript.segments[0].text, "hi there");
    assert_eq!(transcript.segments[0].language.as_deref(), Some("en"));
    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("S0"));
    assert_eq!(
        transcript
            .attributes
            .get("providerRevision")
            .map(String::as_str),
        Some("fixture-v1")
    );
    assert_eq!(
        transcript.segments[0].words()[0]
            .attributes
            .get("token")
            .map(String::as_str),
        Some("4")
    );
    assert_eq!(
        transcript.segments[0].chars()[0]
            .attributes
            .get("extra")
            .map(String::as_str),
        Some("kept")
    );
}

#[test]
fn whisperx_import_attaches_flat_words_and_requires_segments() {
    let transcript = import_whisperx_json(
        br#"{
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "word_segments": [{"word": "hello", "start": 0.2, "end": 0.8, "speakerLabel": "S1"}]
        }"#,
    )
    .unwrap();

    assert_eq!(transcript.segments[0].words().len(), 1);
    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("S1"));
    let error = import_whisperx_json(br#"{"text":"missing segments"}"#)
        .unwrap_err()
        .to_string();
    assert!(error.contains("segments array"));
}

#[derive(Default)]
struct MockAsrProvider;

impl AudioTranscriptionProvider for MockAsrProvider {
    fn provider_id(&self) -> &str {
        "mock-asr"
    }

    fn transcribe(&mut self, request: AsrRequest) -> Result<AsrResponse> {
        let segments = request
            .chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                TranscriptSegmentContract::new(index as u64, " hello ")
                    .with_time_range(Some(0.0), Some(chunk.end_seconds - chunk.start_seconds))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(AsrResponse {
            model_id: request.model_id,
            language: request
                .task
                .output_language_hint()
                .map(str::to_string)
                .or(request.language),
            transcript: transcription_from_segments(
                request.audio.source,
                Some("en".to_string()),
                segments,
            )
            .map_err(|error| DetectError::InvalidArgument(error.to_string()))?,
            diagnostics: vec!["mock ASR completed".to_string()],
        })
    }
}

#[derive(Clone)]
struct FixedVadProvider {
    segments: Vec<SpeechSpan>,
}

impl VoiceActivityDetector for FixedVadProvider {
    fn provider_id(&self) -> &str {
        "fixed-vad"
    }

    fn detect_speech(
        &mut self,
        _audio: &SpeakerAudio<'_>,
        _options: &VadOptions,
    ) -> Result<VadResponse> {
        Ok(VadResponse {
            segments: self.segments.clone(),
            diagnostics: vec!["fixed VAD completed".to_string()],
        })
    }
}

#[derive(Default)]
struct MockAlignmentProvider;

impl ForcedAlignmentProvider for MockAlignmentProvider {
    fn provider_id(&self) -> &str {
        "mock-aligner"
    }

    fn align(&mut self, request: AlignmentRequest) -> Result<AlignmentResponse> {
        Ok(AlignmentResponse {
            model_id: request.model_id,
            words: vec![AlignedWord {
                segment_index: 0,
                word_index: 0,
                text: "hello".to_string(),
                start_seconds: 0.05,
                end_seconds: 0.35,
                confidence: Some(0.91),
            }],
            chars: Vec::new(),
            diagnostics: vec!["mock alignment completed".to_string()],
        })
    }
}

#[derive(Default)]
struct RecordingObserver {
    events: Vec<TranscriptionPipelineEvent>,
}

impl TranscriptionPipelineObserver for RecordingObserver {
    fn observe(&mut self, event: TranscriptionPipelineEvent) {
        self.events.push(event);
    }
}

#[derive(Default)]
struct CancelAtAlignmentObserver {
    cancelled: bool,
}

impl TranscriptionPipelineObserver for CancelAtAlignmentObserver {
    fn observe(&mut self, event: TranscriptionPipelineEvent) {
        if matches!(event, TranscriptionPipelineEvent::AlignmentStart { .. }) {
            self.cancelled = true;
        }
    }

    fn cancellation_requested(&self) -> bool {
        self.cancelled
    }
}

struct ObservingAsrProvider;

impl AudioTranscriptionProvider for ObservingAsrProvider {
    fn provider_id(&self) -> &str {
        "observing-asr"
    }

    fn transcribe(&mut self, request: AsrRequest) -> Result<AsrResponse> {
        MockAsrProvider.transcribe(request)
    }

    fn transcribe_with_observer(
        &mut self,
        request: AsrRequest,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AsrResponse> {
        observer.observe(TranscriptionPipelineEvent::ModelLoadStart {
            stage: "asr".to_string(),
            provider: self.provider_id().to_string(),
            model_id: request.model_id.clone(),
        });
        observer.observe(TranscriptionPipelineEvent::ModelLoadEnd {
            stage: "asr".to_string(),
            provider: self.provider_id().to_string(),
            model_id: request.model_id.clone(),
            duration_seconds: 0.125,
        });
        observer.observe(TranscriptionPipelineEvent::ModelReuse {
            stage: "asr".to_string(),
            provider: self.provider_id().to_string(),
            model_id: request.model_id.clone(),
        });
        self.transcribe(request)
    }
}

struct ObservingAlignmentProvider;

impl ForcedAlignmentProvider for ObservingAlignmentProvider {
    fn provider_id(&self) -> &str {
        "observing-aligner"
    }

    fn align(&mut self, request: AlignmentRequest) -> Result<AlignmentResponse> {
        MockAlignmentProvider.align(request)
    }

    fn align_with_observer(
        &mut self,
        request: AlignmentRequest,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AlignmentResponse> {
        observer.observe(TranscriptionPipelineEvent::ModelLoadStart {
            stage: "alignment".to_string(),
            provider: self.provider_id().to_string(),
            model_id: request.model_id.clone(),
        });
        observer.observe(TranscriptionPipelineEvent::ModelLoadEnd {
            stage: "alignment".to_string(),
            provider: self.provider_id().to_string(),
            model_id: request.model_id.clone(),
            duration_seconds: 0.25,
        });
        self.align(request)
    }
}

struct MockDiarizationProvider;

impl TranscriptDiarizationProvider for MockDiarizationProvider {
    fn provider_id(&self) -> &str {
        "mock-diarization"
    }

    fn diarize(
        &mut self,
        _audio: LoadedAudio,
        _transcript: &TranscriptionContract,
        options: &DiarizationOptions,
    ) -> Result<SpeakerDiarizationResponse> {
        Ok(SpeakerDiarizationResponse {
            accepted: true,
            operation: "audio.speakers.diarize".to_string(),
            model_id: options.model_id.clone(),
            runtime: audio_analysis_speakers::AudioRuntime::Imported,
            segments: vec![SpeakerSegmentPrediction {
                speaker: "SPEAKER_00".to_string(),
                start_seconds: 0.0,
                end_seconds: 1.0,
                score: Some(0.9),
            }],
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        })
    }
}

struct PanickingDiarizationProvider {
    called: bool,
}

#[cfg(feature = "diarization")]
struct MockOnnxDiarizationProvider;

#[cfg(feature = "diarization")]
impl TranscriptDiarizationProvider for MockOnnxDiarizationProvider {
    fn provider_id(&self) -> &str {
        "mock-onnx-diarization"
    }

    fn diarize(
        &mut self,
        _audio: LoadedAudio,
        _transcript: &TranscriptionContract,
        options: &DiarizationOptions,
    ) -> Result<SpeakerDiarizationResponse> {
        Ok(SpeakerDiarizationResponse {
            accepted: true,
            operation: "audio.speakers.diarize".to_string(),
            model_id: options.model_id.clone(),
            runtime: audio_analysis_speakers::AudioRuntime::Onnx,
            segments: vec![audio_analysis_speakers::SpeakerSegmentPrediction {
                speaker: "SPEAKER_ONNX".to_string(),
                start_seconds: 0.0,
                end_seconds: 1.0,
                score: Some(0.9),
            }],
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        })
    }
}

impl TranscriptDiarizationProvider for PanickingDiarizationProvider {
    fn provider_id(&self) -> &str {
        "panicking-diarization"
    }

    fn diarize(
        &mut self,
        _audio: LoadedAudio,
        _transcript: &TranscriptionContract,
        _options: &DiarizationOptions,
    ) -> Result<SpeakerDiarizationResponse> {
        self.called = true;
        panic!("diarization provider should not be called for invalid options");
    }
}

fn sample_request() -> TranscriptionPipelineRequest {
    let mut samples = vec![0.0; 16_000];
    for sample in &mut samples[1_000..5_000] {
        *sample = 0.1;
    }
    TranscriptionPipelineRequest {
        source: TranscriptionSource::Samples {
            samples,
            sample_rate: 16_000,
            channels: 1,
            source: Some("synthetic".to_string()),
        },
        provider: TranscriptionProviderSelection::CandleWhisper(CandleWhisperOptions::default()),
        vad: TranscriptionVadOptions {
            detection: VadOptions {
                min_speech_seconds: 0.01,
                ..VadOptions::default()
            },
            ..TranscriptionVadOptions::default()
        },
        alignment: AlignmentOptions::default(),
        diarization: DiarizationOptions::default(),
        output: TranscriptionOutputOptions::default(),
    }
}

#[test]
fn cooperative_cancellation_stops_before_alignment_provider_work() {
    let mut request = sample_request();
    request.alignment.enabled = true;
    let mut vad = FixedVadProvider {
        segments: vec![SpeechSpan::new(0.0, 1.0, 1.0).unwrap()],
    };
    let mut asr = MockAsrProvider;
    let mut aligner = MockAlignmentProvider;
    let mut observer = CancelAtAlignmentObserver::default();

    let error = run_transcription_pipeline_with_observer(
        request,
        &mut vad,
        &mut asr,
        Some(&mut aligner),
        None,
        &mut observer,
    )
    .expect_err("cancellation should win before alignment provider work");

    assert!(error.to_string().contains("cancelled"));
}

fn batch_test_chunks() -> Vec<SpeechSpan> {
    vec![
        SpeechSpan::new(0.0, 0.20, 1.0).unwrap(),
        SpeechSpan::new(0.20, 0.40, 1.0).unwrap(),
        SpeechSpan::new(0.40, 0.60, 1.0).unwrap(),
    ]
}

fn batch_test_request(options: CandleWhisperOptions) -> TranscriptionPipelineRequest {
    TranscriptionPipelineRequest {
        provider: TranscriptionProviderSelection::CandleWhisper(options),
        ..sample_request()
    }
}

#[test]
fn reusable_session_keeps_compatible_stack_and_replaces_incompatible_options() {
    let compatible = batch_test_request(CandleWhisperOptions {
        model_id: "openai/whisper-tiny.en".to_string(),
        ..CandleWhisperOptions::default()
    });
    let mut session = ReusableTranscriptionSession::new(
        ReusableTranscriptionSessionOptions::from_request(&compatible),
    )
    .expect("reusable session");

    session
        .prepare_for_request(&compatible)
        .expect("compatible request must retain the owned stack");
    assert_eq!(session.asr_provider.provider_id(), "candle-whisper");

    let incompatible = batch_test_request(CandleWhisperOptions {
        model_id: "openai/whisper-base.en".to_string(),
        ..CandleWhisperOptions::default()
    });
    session
        .prepare_for_request(&incompatible)
        .expect("incompatible provider configuration must replace the owned stack");
    assert_eq!(
        session.options.provider, incompatible.provider,
        "the session must retain the replacement provider configuration"
    );
}

#[test]
fn pipeline_observer_receives_model_load_and_reuse_events() {
    let mut request = sample_request();
    request.vad = TranscriptionVadOptions {
        enabled: false,
        ..TranscriptionVadOptions::default()
    };
    request.alignment = AlignmentOptions {
        enabled: true,
        model_id: "facebook/wav2vec2-base-960h".to_string(),
        ..AlignmentOptions::default()
    };
    let mut vad = FixedVadProvider {
        segments: vec![SpeechSpan::new(0.0, 0.5, 1.0).unwrap()],
    };
    let mut asr = ObservingAsrProvider;
    let mut aligner = ObservingAlignmentProvider;
    let mut observer = RecordingObserver::default();

    let response = run_transcription_pipeline_with_observer(
        request,
        &mut vad,
        &mut asr,
        Some(&mut aligner),
        None,
        &mut observer,
    )
    .expect("pipeline should run with observing providers");

    assert!(response.accepted);
    assert!(observer
        .events
        .contains(&TranscriptionPipelineEvent::ModelLoadStart {
            stage: "asr".to_string(),
            provider: "observing-asr".to_string(),
            model_id: "openai/whisper-large-v3-turbo".to_string(),
        }));
    assert!(observer
        .events
        .contains(&TranscriptionPipelineEvent::ModelLoadEnd {
            stage: "asr".to_string(),
            provider: "observing-asr".to_string(),
            model_id: "openai/whisper-large-v3-turbo".to_string(),
            duration_seconds: 0.125,
        }));
    assert!(observer
        .events
        .contains(&TranscriptionPipelineEvent::ModelReuse {
            stage: "asr".to_string(),
            provider: "observing-asr".to_string(),
            model_id: "openai/whisper-large-v3-turbo".to_string(),
        }));
    assert!(observer
        .events
        .contains(&TranscriptionPipelineEvent::ModelLoadStart {
            stage: "alignment".to_string(),
            provider: "observing-aligner".to_string(),
            model_id: "facebook/wav2vec2-base-960h".to_string(),
        }));
    assert!(observer
        .events
        .contains(&TranscriptionPipelineEvent::ModelLoadEnd {
            stage: "alignment".to_string(),
            provider: "observing-aligner".to_string(),
            model_id: "facebook/wav2vec2-base-960h".to_string(),
            duration_seconds: 0.25,
        }));
}

#[test]
fn candle_whisper_options_default_to_automatic_compute_type() {
    let options = CandleWhisperOptions::default();
    assert_eq!(options.compute_type, CandleWhisperComputeType::Automatic);
    assert_eq!(options.device, NativeDevicePreference::Auto);
    assert_eq!(CandleWhisperRuntimeControls::default().cuda_device_index, 0);
    assert_eq!(
        CandleWhisperRuntimeControls::default().decoder_threads,
        None
    );
}

#[test]
fn candle_whisper_decode_validation_rejects_invalid_search_combinations() {
    let invalid = [
        CandleWhisperDecodeConfig {
            temperature_schedule: vec![f64::NAN],
            ..CandleWhisperDecodeConfig::default()
        },
        CandleWhisperDecodeConfig {
            best_of: 0,
            ..CandleWhisperDecodeConfig::default()
        },
        CandleWhisperDecodeConfig {
            beam_size: 0,
            ..CandleWhisperDecodeConfig::default()
        },
        CandleWhisperDecodeConfig {
            temperature_schedule: vec![0.4],
            beam_size: 2,
            ..CandleWhisperDecodeConfig::default()
        },
        CandleWhisperDecodeConfig {
            best_of: 2,
            ..CandleWhisperDecodeConfig::default()
        },
        CandleWhisperDecodeConfig {
            beam_size: 2,
            best_of: 2,
            ..CandleWhisperDecodeConfig::default()
        },
        CandleWhisperDecodeConfig {
            patience: 2.0,
            ..CandleWhisperDecodeConfig::default()
        },
        CandleWhisperDecodeConfig {
            length_penalty: -1.0,
            ..CandleWhisperDecodeConfig::default()
        },
    ];

    for config in invalid {
        assert!(matches!(
            config.validate(),
            Err(DetectError::InvalidArgument(_))
        ));
    }
}

#[test]
fn candle_whisper_default_decode_config_selects_the_legacy_greedy_path() {
    assert!(CandleWhisperDecodeConfig::default().uses_default_greedy_path());
    assert!(!CandleWhisperDecodeConfig {
        temperature_schedule: vec![0.2],
        ..CandleWhisperDecodeConfig::default()
    }
    .uses_default_greedy_path());
    assert!(!CandleWhisperDecodeConfig {
        beam_size: 2,
        ..CandleWhisperDecodeConfig::default()
    }
    .uses_default_greedy_path());
}

#[test]
fn candle_whisper_decode_request_thresholds_fail_closed() {
    for config in [
        CandleWhisperDecodeRequestConfig {
            min_average_log_probability: Some(f64::NAN),
            ..CandleWhisperDecodeRequestConfig::default()
        },
        CandleWhisperDecodeRequestConfig {
            max_no_speech_probability: Some(1.1),
            ..CandleWhisperDecodeRequestConfig::default()
        },
        CandleWhisperDecodeRequestConfig {
            max_compression_ratio: Some(0.0),
            ..CandleWhisperDecodeRequestConfig::default()
        },
    ] {
        assert!(matches!(
            config.validate(),
            Err(DetectError::InvalidArgument(_))
        ));
    }
}

#[test]
fn previous_text_conditioning_rejects_independent_active_row_batches() {
    let config = CandleWhisperDecodeRequestConfig {
        condition_on_previous_text: true,
        ..CandleWhisperDecodeRequestConfig::default()
    };
    let options = CandleWhisperOptions {
        decode_runtime: CandleWhisperDecodeRuntime::ActiveRowTensorBatch,
        batch_chunks: true,
        max_batch_size: Some(2),
        ..CandleWhisperOptions::default()
    };

    assert!(matches!(
        config.validate_runtime(&options),
        Err(DetectError::InvalidArgument(_))
    ));
}

#[cfg(feature = "candle")]
#[test]
fn real_tiny_whisper_bundle_runs_greedy_sampling_and_beam_paths_when_requested() {
    if std::env::var("RUN_CANDLE_WHISPER_DECODE_TESTS").as_deref() != Ok("1") {
        eprintln!(
            "skipping native Whisper decode smoke; set RUN_CANDLE_WHISPER_DECODE_TESTS=1"
        );
        return;
    }
    let bundle = std::env::var_os("CANDLE_WHISPER_TINY_BUNDLE")
        .map(PathBuf::from)
        .expect("CANDLE_WHISPER_TINY_BUNDLE is required for the decode smoke");
    let mut provider = ReusableCandleWhisperTranscriber::new(CandleWhisperOptions {
        model_id: "openai/whisper-tiny.en".to_string(),
        language: Some("en".to_string()),
        device: NativeDevicePreference::Cpu,
        compute_type: CandleWhisperComputeType::Fp32,
        model_bundle: Some(bundle),
        model_cache_only: true,
        batch_chunks: false,
        max_batch_size: Some(1),
        ..CandleWhisperOptions::default()
    });
    let request = || AsrRequest {
        audio: LoadedAudio {
            samples: vec![0.0; 16_000],
            sample_rate: 16_000,
            channels: 1,
            source: Some("decode-smoke".to_string()),
        },
        chunks: vec![SpeechSpan::new(0.0, 1.0, 1.0).unwrap()],
        task: TranscriptionTask::Transcribe,
        language: Some("en".to_string()),
        model_id: "openai/whisper-tiny.en".to_string(),
    };
    let paths = [
        (
            CandleWhisperDecodeConfig::default(),
            "decodeStrategy=greedy",
        ),
        (
            CandleWhisperDecodeConfig {
                temperature_schedule: vec![0.2],
                best_of: 2,
                seed: 7,
                ..CandleWhisperDecodeConfig::default()
            },
            "decodeStrategy=temperatureSampling",
        ),
        (
            CandleWhisperDecodeConfig {
                beam_size: 2,
                patience: 1.0,
                length_penalty: 1.0,
                ..CandleWhisperDecodeConfig::default()
            },
            "decodeStrategy=beamSearch",
        ),
    ];

    for (decode, expected_diagnostic) in paths {
        let response = provider
            .transcribe_with_decode_config(request(), decode)
            .expect("configured native Whisper execution path should run");
        assert!(response
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic == expected_diagnostic));
    }
}

#[cfg(feature = "candle")]
#[test]
#[ignore = "requires caller-owned CANDLE_WHISPER_Q8_BUNDLE and CANDLE_WHISPER_Q8_WAV resources"]
fn real_q8_whisper_bundle_transcribes_short_fixture_with_valid_contract() {
    let bundle = std::env::var_os("CANDLE_WHISPER_Q8_BUNDLE")
        .map(PathBuf::from)
        .expect("CANDLE_WHISPER_Q8_BUNDLE is required for the Q8 smoke");
    let audio_path = std::env::var_os("CANDLE_WHISPER_Q8_WAV")
        .map(PathBuf::from)
        .expect("CANDLE_WHISPER_Q8_WAV is required for the Q8 smoke");
    let audio =
        LoadedAudio::mono_16khz_from_source(&TranscriptionSource::Path { path: audio_path })
            .expect("Q8 smoke fixture must be readable mono-compatible WAV audio");
    let duration = audio.duration_seconds().min(10.0);
    let mut provider = ReusableCandleWhisperTranscriber::new(CandleWhisperOptions {
        model_id: "openai/whisper-tiny.en".to_string(),
        language: Some("en".to_string()),
        device: NativeDevicePreference::Cpu,
        compute_type: CandleWhisperComputeType::Int8,
        model_bundle: Some(bundle),
        model_cache_only: true,
        batch_chunks: true,
        max_batch_size: Some(2),
        decode_runtime: CandleWhisperDecodeRuntime::ActiveRowTensorBatch,
        ..CandleWhisperOptions::default()
    });
    let response = provider
        .transcribe(AsrRequest {
            audio,
            chunks: vec![SpeechSpan::new(0.0, duration, 1.0).unwrap()],
            task: TranscriptionTask::Transcribe,
            language: Some("en".to_string()),
            model_id: "openai/whisper-tiny.en".to_string(),
        })
        .expect("Q8 Whisper smoke transcription should run");

    response
        .transcript
        .validate_strict()
        .expect("Q8 Whisper smoke must return a valid transcript contract");
    for expected in ["computeType=int8", "modelFormat=gguf-q8_0"] {
        assert!(response
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic == expected));
    }
    assert!(response
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.starts_with("cacheReuse=")));
    assert!(response
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.starts_with("phaseTiming.asrSeconds=")));
}

#[test]
fn candle_whisper_controlled_observer_entrypoints_are_public_and_validate_before_runtime() {
    let request = || AsrRequest {
        audio: LoadedAudio {
            samples: vec![0.0; 16],
            sample_rate: 16_000,
            channels: 1,
            source: Some("controlled-observer".to_string()),
        },
        chunks: vec![SpeechSpan::new(0.0, 0.001, 1.0).unwrap()],
        task: TranscriptionTask::Transcribe,
        language: Some("en".to_string()),
        model_id: "openai/whisper-tiny.en".to_string(),
    };
    let controls = CandleWhisperRuntimeControls {
        decoder_threads: Some(0),
        ..CandleWhisperRuntimeControls::default()
    };
    let decode = CandleWhisperDecodeRequestConfig::default();
    let mut observer = RecordingObserver::default();
    let mut single = CandleWhisperTranscriber::new(CandleWhisperOptions::default());
    let mut reusable = ReusableCandleWhisperTranscriber::new(CandleWhisperOptions::default());

    let single_error = single
        .transcribe_with_runtime_controls_and_decode_request_config_and_observer(
            request(),
            controls,
            decode.clone(),
            &mut observer,
        )
        .unwrap_err();
    let reusable_error = reusable
        .transcribe_with_runtime_controls_and_decode_request_config_and_observer(
            request(),
            controls,
            decode,
            &mut observer,
        )
        .unwrap_err();

    assert!(single_error
        .to_string()
        .contains("decoder_threads must be greater than zero"));
    assert!(reusable_error
        .to_string()
        .contains("decoder_threads must be greater than zero"));
}

#[test]
fn candle_whisper_runtime_controls_serialize_as_public_request_fields() {
    let controls = CandleWhisperRuntimeControls {
        cuda_device_index: 2,
        decoder_threads: Some(3),
    };

    let encoded = serde_json::to_value(controls).unwrap();

    assert_eq!(encoded["cudaDeviceIndex"], 2);
    assert_eq!(encoded["decoderThreads"], 3);
}

#[test]
fn candle_whisper_rejects_negative_cuda_device_indices_during_request_decode() {
    let error = serde_json::from_value::<CandleWhisperRuntimeControls>(serde_json::json!({
        "cudaDeviceIndex": -1
    }))
    .unwrap_err();

    assert!(error.to_string().contains("invalid value"));
}

#[test]
fn candle_whisper_rejects_zero_decoder_threads() {
    let error = validate_candle_runtime_controls(
        &CandleWhisperOptions::default(),
        &CandleWhisperRuntimeControls {
            decoder_threads: Some(0),
            ..CandleWhisperRuntimeControls::default()
        },
    )
    .unwrap_err();

    assert!(matches!(error, DetectError::InvalidArgument(_)));
    assert!(error
        .to_string()
        .contains("decoder_threads must be greater than zero"));
}

#[test]
fn candle_whisper_cpu_rejects_nonzero_cuda_device_index() {
    let error = validate_candle_runtime_controls(
        &CandleWhisperOptions {
            device: NativeDevicePreference::Cpu,
            ..CandleWhisperOptions::default()
        },
        &CandleWhisperRuntimeControls {
            cuda_device_index: 1,
            ..CandleWhisperRuntimeControls::default()
        },
    )
    .unwrap_err();

    assert!(matches!(error, DetectError::InvalidArgument(_)));
    assert!(error
        .to_string()
        .contains("cuda_device_index 1 cannot be used with device=cpu"));
}

#[test]
fn candle_whisper_compute_type_serializes_public_values_and_aliases() {
    let options = CandleWhisperOptions {
        compute_type: CandleWhisperComputeType::Fp16,
        ..CandleWhisperOptions::default()
    };
    let encoded = serde_json::to_value(&options).unwrap();
    assert_eq!(encoded["computeType"], "fp16");

    let decoded: CandleWhisperOptions =
        serde_json::from_value(serde_json::json!({"computeType": "float32"})).unwrap();
    assert_eq!(decoded.compute_type, CandleWhisperComputeType::Fp32);

    let decoded: CandleWhisperOptions =
        serde_json::from_value(serde_json::json!({"computeType": "auto"})).unwrap();
    assert_eq!(decoded.compute_type, CandleWhisperComputeType::Automatic);

    let decoded: CandleWhisperOptions =
        serde_json::from_value(serde_json::json!({"computeType": "int8"})).unwrap();
    assert_eq!(decoded.compute_type, CandleWhisperComputeType::Int8);
    assert_eq!(
        serde_json::to_value(decoded).unwrap()["computeType"],
        "int8"
    );
}

#[test]
fn candle_whisper_compute_type_resolves_by_device() {
    assert_eq!(
        CandleWhisperComputeType::Automatic
            .resolve_for_device(true)
            .unwrap(),
        CandleWhisperComputeType::Fp16
    );
    assert_eq!(
        CandleWhisperComputeType::Fp16
            .resolve_for_device(true)
            .unwrap(),
        CandleWhisperComputeType::Fp16
    );
    assert_eq!(
        CandleWhisperComputeType::Fp32
            .resolve_for_device(true)
            .unwrap(),
        CandleWhisperComputeType::Fp32
    );
    assert_eq!(
        CandleWhisperComputeType::Automatic
            .resolve_for_device(false)
            .unwrap(),
        CandleWhisperComputeType::Fp32
    );
    assert_eq!(
        CandleWhisperComputeType::Fp32
            .resolve_for_device(false)
            .unwrap(),
        CandleWhisperComputeType::Fp32
    );
    assert_eq!(
        CandleWhisperComputeType::Int8
            .resolve_for_device(false)
            .unwrap(),
        CandleWhisperComputeType::Int8
    );
    assert!(CandleWhisperComputeType::Int8
        .resolve_for_device(true)
        .is_err());
}

#[test]
fn int8_exposes_the_exact_required_q8_bundle_files() {
    assert_eq!(
        CandleWhisperComputeType::Int8.required_bundle_files(),
        &[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "preprocessor_config.json",
            "model.q8_0.gguf",
        ]
    );
}

#[test]
fn full_precision_compute_types_still_require_safetensors_weights() {
    for compute_type in [
        CandleWhisperComputeType::Automatic,
        CandleWhisperComputeType::Fp16,
        CandleWhisperComputeType::Fp32,
    ] {
        assert!(compute_type
            .required_bundle_files()
            .contains(&"model.safetensors"));
        assert!(!compute_type
            .required_bundle_files()
            .contains(&"model.q8_0.gguf"));
    }
}

#[cfg(feature = "candle")]
#[test]
fn candle_setup_validates_int8_against_the_q8_bundle_contract() {
    let bundle = tempfile::tempdir().unwrap();
    for file in CandleWhisperComputeType::Int8.required_bundle_files() {
        std::fs::write(bundle.path().join(file), []).unwrap();
    }
    let options = CandleWhisperOptions {
        device: NativeDevicePreference::Cpu,
        compute_type: CandleWhisperComputeType::Int8,
        model_bundle: Some(bundle.path().to_path_buf()),
        ..CandleWhisperOptions::default()
    };

    validate_candle_setup(&options, &CandleWhisperRuntimeControls::default()).unwrap();
}

#[test]
fn candle_whisper_cpu_fp16_is_rejected_clearly() {
    let error = CandleWhisperComputeType::Fp16
        .resolve_for_device(false)
        .unwrap_err()
        .to_string();

    assert!(error.contains("setup_error"));
    assert!(error.contains("fp16 requires a CUDA device"));
}

#[test]
fn only_automatic_candle_compute_type_is_setup_fallback_eligible() {
    assert!(CandleWhisperComputeType::Automatic.setup_fallback_eligible());
    assert!(!CandleWhisperComputeType::Fp16.setup_fallback_eligible());
    assert!(!CandleWhisperComputeType::Fp32.setup_fallback_eligible());
    assert!(!CandleWhisperComputeType::Int8.setup_fallback_eligible());
}

fn diarization_response_for_tests(
    segments: Vec<SpeakerSegmentPrediction>,
) -> SpeakerDiarizationResponse {
    SpeakerDiarizationResponse {
        accepted: true,
        operation: "audio.speakers.diarize".to_string(),
        model_id: "test-speakers".to_string(),
        runtime: audio_analysis_speakers::AudioRuntime::Imported,
        segments,
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    }
}

fn transcript_with_words(
    words: Vec<(&str, f64, f64)>,
) -> std::result::Result<TranscriptionContract, Box<dyn std::error::Error>> {
    let segment_end = words
        .iter()
        .map(|(_, _, end_seconds)| *end_seconds)
        .fold(2.0_f64, f64::max);
    let mut segment = TranscriptSegmentContract::new(
        0,
        words
            .iter()
            .map(|(word, _, _)| *word)
            .collect::<Vec<_>>()
            .join(" "),
    )
    .with_time_range(Some(0.0), Some(segment_end))?;
    for (text, start_seconds, end_seconds) in words {
        segment.push_word(
            TranscriptWordContract::new(text)
                .with_time_range(Some(start_seconds), Some(end_seconds))?,
        )?;
    }
    Ok(transcription_from_segments(
        None,
        Some("en".to_string()),
        vec![segment],
    )?)
}

fn assign_speakers_from_diarization(
    transcript: &mut TranscriptionContract,
    diarization: &SpeakerDiarizationResponse,
    policy: SpeakerAssignmentPolicy,
) -> Result<()> {
    *transcript = audio_analysis_speakers::assign_speakers_to_transcript_with_policy(
        transcript,
        diarization,
        policy,
    )?;
    Ok(())
}

#[cfg(feature = "diarization")]
fn sine_into(samples: &mut [f32], sample_rate: u32, start_seconds: f32, freq_hz: f32) {
    for (offset, sample) in samples.iter_mut().enumerate() {
        let t = start_seconds + offset as f32 / sample_rate as f32;
        *sample = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5;
    }
}

#[cfg(feature = "diarization")]
fn two_profile_loaded_audio() -> LoadedAudio {
    let sample_rate = 16_000;
    let mut samples = vec![0.0_f32; sample_rate as usize * 2];
    let first_start = (0.20 * sample_rate as f32) as usize;
    let first_end = (0.50 * sample_rate as f32) as usize;
    let second_start = (1.00 * sample_rate as f32) as usize;
    let second_end = (1.40 * sample_rate as f32) as usize;
    sine_into(
        &mut samples[first_start..first_end],
        sample_rate,
        0.20,
        220.0,
    );
    sine_into(
        &mut samples[second_start..second_end],
        sample_rate,
        1.00,
        1_200.0,
    );
    LoadedAudio {
        samples,
        sample_rate,
        channels: 1,
        source: Some("synthetic-two-speaker".to_string()),
    }
}

#[cfg(all(feature = "diarization", feature = "onnx"))]
fn local_16khz_wav_samples(
    path: &Path,
) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 {
        return Err(format!(
            "DIARIZATION_AUDIO_PATH must be 16 kHz WAV, got {} Hz",
            spec.sample_rate
        )
        .into());
    }
    if spec.channels == 0 {
        return Err("DIARIZATION_AUDIO_PATH WAV channel count must be non-zero".into());
    }
    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                return Err("float DIARIZATION_AUDIO_PATH WAV must be 32-bit".into());
            }
            reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / 32_768.0))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(spec.bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    let channels = spec.channels as usize;
    let samples = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect()
    };
    if samples.is_empty() {
        return Err("DIARIZATION_AUDIO_PATH WAV must contain samples".into());
    }
    if !samples.iter().all(|sample| sample.is_finite()) {
        return Err("DIARIZATION_AUDIO_PATH WAV samples must be finite".into());
    }
    Ok(samples)
}

#[test]
fn diarization_options_reject_invalid_speaker_bounds() {
    let mut options = DiarizationOptions {
        speaker: SpeakerDiarizationOptions {
            min_speakers: Some(0),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    assert!(validate_diarization_options(&options)
        .unwrap_err()
        .to_string()
        .contains("invalid_request"));

    options = DiarizationOptions {
        speaker: SpeakerDiarizationOptions {
            max_speakers: Some(0),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    assert!(validate_diarization_options(&options)
        .unwrap_err()
        .to_string()
        .contains("invalid_request"));

    options = DiarizationOptions {
        speaker: SpeakerDiarizationOptions {
            min_speakers: Some(3),
            max_speakers: Some(2),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    assert!(validate_diarization_options(&options)
        .unwrap_err()
        .to_string()
        .contains("invalid_request"));

    options = DiarizationOptions {
        speaker: SpeakerDiarizationOptions {
            min_speakers: Some(1),
            max_speakers: Some(2),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    validate_diarization_options(&options).unwrap();
}

#[test]
fn diarization_options_keep_flat_json_shape_with_speaker_owned_contract() {
    let options: DiarizationOptions = serde_json::from_value(serde_json::json!({
        "enabled": true,
        "modelId": "native-speakers",
        "speakerEmbeddingModelBundle": "/models/speaker",
        "speakerEmbeddingModelFile": "speaker.onnx",
        "speakerEmbeddingInputName": "waveform",
        "speakerEmbeddingOutputName": "embedding",
        "speakerEmbeddingDimension": 192,
        "speakerEmbeddingSampleRate": 16000,
        "returnSpeakerEmbeddings": true,
        "minSpeakers": 1,
        "maxSpeakers": 2,
        "assignmentPolicy": "strictContained"
    }))
    .unwrap();

    assert!(options.enabled);
    assert_eq!(options.model_id, "native-speakers");
    assert_eq!(
        options.speaker_embedding_model_bundle.as_deref(),
        Some(Path::new("/models/speaker"))
    );
    assert_eq!(
        options.speaker_embedding_model_file.as_deref(),
        Some("speaker.onnx")
    );
    assert_eq!(
        options.speaker_embedding_input_name.as_deref(),
        Some("waveform")
    );
    assert_eq!(
        options.speaker_embedding_output_name.as_deref(),
        Some("embedding")
    );
    assert_eq!(options.speaker_embedding_dimension, Some(192));
    assert_eq!(options.speaker_embedding_sample_rate, Some(16_000));
    assert!(options.return_speaker_embeddings);
    assert_eq!(options.min_speakers, Some(1));
    assert_eq!(options.max_speakers, Some(2));
    assert_eq!(
        options.assignment_policy,
        SpeakerAssignmentPolicy::StrictContained
    );

    let value = serde_json::to_value(&options).unwrap();
    assert_eq!(value["enabled"], true);
    assert_eq!(value["modelId"], "native-speakers");
    assert_eq!(value["assignmentPolicy"], "strictContained");
    assert!(value.get("speaker").is_none());
}

#[test]
fn default_build_exposes_speakers_owned_diarization_response_types() {
    fn accepts_speakers_response(_: audio_analysis_speakers::SpeakerDiarizationResponse) {}

    let response = SpeakerDiarizationResponse {
        accepted: true,
        operation: "audio.speakers.diarize".to_string(),
        model_id: "fixture".to_string(),
        runtime: audio_analysis_speakers::AudioRuntime::Imported,
        segments: vec![SpeakerSegmentPrediction {
            speaker: "speaker_0".to_string(),
            start_seconds: 0.0,
            end_seconds: 1.0,
            score: None,
        }],
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    };

    accepts_speakers_response(response);
}

#[test]
fn batch_options_reject_zero_max_batch_size() {
    let mut vad = FixedVadProvider {
        segments: batch_test_chunks(),
    };
    let mut asr = MockAsrProvider;
    let result = run_transcription_pipeline(
        batch_test_request(CandleWhisperOptions {
            max_batch_size: Some(0),
            ..CandleWhisperOptions::default()
        }),
        &mut vad,
        &mut asr,
        None,
        None,
    );

    let error = result.unwrap_err().to_string();
    assert!(error.contains("invalid_request"));
    assert!(error.contains("max_batch_size"));
}

#[test]
fn candle_decode_runtime_defaults_to_autoregressive_kv_cache() {
    let options = CandleWhisperOptions::default();

    assert_eq!(
        options.decode_runtime,
        CandleWhisperDecodeRuntime::AutoregressiveKvCache
    );
    assert_eq!(
        options.decode_runtime.execution_id(),
        "candle-whisper-autoregressive-kv-cache"
    );
    validate_candle_batch_options(&options).unwrap();
}

#[test]
fn candle_decode_runtime_deserializes_active_row_tensor_batch() {
    let options: CandleWhisperOptions = serde_json::from_value(serde_json::json!({
        "decodeRuntime": "activeRowTensorBatch",
        "batchChunks": true,
        "maxBatchSize": 4
    }))
    .unwrap();

    assert_eq!(
        options.decode_runtime,
        CandleWhisperDecodeRuntime::ActiveRowTensorBatch
    );
    assert_eq!(
        options.decode_runtime.execution_id(),
        "candle-whisper-active-row-tensor-batch"
    );
}

#[test]
fn candle_active_row_decode_runtime_is_supported_when_batching_is_enabled() {
    let options = CandleWhisperOptions {
        decode_runtime: CandleWhisperDecodeRuntime::ActiveRowTensorBatch,
        batch_chunks: true,
        max_batch_size: Some(4),
        ..CandleWhisperOptions::default()
    };

    validate_candle_batch_options(&options).unwrap();
    assert!(options.decode_runtime.is_supported());
}

#[test]
fn candle_active_row_decode_runtime_requires_chunk_batching() {
    let options = CandleWhisperOptions {
        decode_runtime: CandleWhisperDecodeRuntime::ActiveRowTensorBatch,
        batch_chunks: false,
        max_batch_size: Some(4),
        ..CandleWhisperOptions::default()
    };

    let error = validate_candle_batch_options(&options)
        .unwrap_err()
        .to_string();
    assert!(error.contains("invalid_request"));
    assert!(error.contains("requires batch_chunks=true"));
}

#[test]
fn candle_active_row_decode_runtime_rejects_single_row_batching() {
    let options = CandleWhisperOptions {
        decode_runtime: CandleWhisperDecodeRuntime::ActiveRowTensorBatch,
        batch_chunks: true,
        max_batch_size: Some(1),
        ..CandleWhisperOptions::default()
    };

    let error = validate_candle_batch_options(&options)
        .unwrap_err()
        .to_string();
    assert!(error.contains("invalid_request"));
    assert!(error.contains("max_batch_size greater than one"));
}

#[test]
fn batch_chunking_preserves_transcript_order() {
    let mut vad = FixedVadProvider {
        segments: batch_test_chunks(),
    };
    let mut asr = MockAsrProvider;
    let response = run_transcription_pipeline(
        batch_test_request(CandleWhisperOptions {
            batch_chunks: true,
            max_batch_size: Some(2),
            ..CandleWhisperOptions::default()
        }),
        &mut vad,
        &mut asr,
        None,
        None,
    )
    .unwrap();

    let starts = response
        .transcript
        .segments
        .iter()
        .map(|segment| segment.start_seconds().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(starts, vec![0.0, 0.20, 0.40]);
}

#[test]
fn batch_chunking_reports_batch_diagnostics() {
    let mut vad = FixedVadProvider {
        segments: batch_test_chunks(),
    };
    let mut asr = MockAsrProvider;
    let response = run_transcription_pipeline(
        batch_test_request(CandleWhisperOptions {
            batch_chunks: true,
            max_batch_size: Some(2),
            ..CandleWhisperOptions::default()
        }),
        &mut vad,
        &mut asr,
        None,
        None,
    )
    .unwrap();

    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "chunkCount=3"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "batchChunks=true"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "maxBatchSize=2"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "batchCount=2"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "batchExecution=candle-whisper-autoregressive-kv-cache"));
}

#[test]
fn requested_active_row_runtime_reports_fallback_execution_without_native_proof() {
    let mut vad = FixedVadProvider {
        segments: batch_test_chunks(),
    };
    let mut asr = MockAsrProvider;
    let response = run_transcription_pipeline(
        batch_test_request(CandleWhisperOptions {
            decode_runtime: CandleWhisperDecodeRuntime::ActiveRowTensorBatch,
            batch_chunks: true,
            max_batch_size: Some(3),
            ..CandleWhisperOptions::default()
        }),
        &mut vad,
        &mut asr,
        None,
        None,
    )
    .unwrap();

    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "batchExecution=candle-whisper-autoregressive-kv-cache"));
    assert!(!response
        .diagnostics
        .iter()
        .any(|item| item == "batchExecution=candle-whisper-active-row-tensor-batch"));
}

#[test]
fn public_batch_diagnostics_do_not_claim_active_row_execution_by_request() {
    let diagnostics = candle_batch_diagnostics(
        &CandleWhisperOptions {
            decode_runtime: CandleWhisperDecodeRuntime::ActiveRowTensorBatch,
            batch_chunks: true,
            max_batch_size: Some(4),
            ..CandleWhisperOptions::default()
        },
        3,
    );

    assert!(diagnostics
        .iter()
        .any(|item| item == "batchExecution=candle-whisper-autoregressive-kv-cache"));
    assert!(!diagnostics
        .iter()
        .any(|item| item == "batchExecution=candle-whisper-active-row-tensor-batch"));
}

#[test]
fn batch_chunking_reports_unbounded_batch_diagnostics() {
    let mut vad = FixedVadProvider {
        segments: batch_test_chunks(),
    };
    let mut asr = MockAsrProvider;
    let response = run_transcription_pipeline(
        batch_test_request(CandleWhisperOptions {
            batch_chunks: true,
            max_batch_size: None,
            ..CandleWhisperOptions::default()
        }),
        &mut vad,
        &mut asr,
        None,
        None,
    )
    .unwrap();

    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "maxBatchSize=unbounded"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "batchCount=1"));
}

#[test]
fn batch_disabled_reports_sequential_diagnostics() {
    let mut vad = FixedVadProvider {
        segments: batch_test_chunks(),
    };
    let mut asr = MockAsrProvider;
    let response = run_transcription_pipeline(
        batch_test_request(CandleWhisperOptions {
            batch_chunks: false,
            max_batch_size: Some(2),
            ..CandleWhisperOptions::default()
        }),
        &mut vad,
        &mut asr,
        None,
        None,
    )
    .unwrap();

    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "batchChunks=false"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "batchCount=3"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "batchExecution=candle-whisper-autoregressive-kv-cache"));
}

#[test]
#[cfg(feature = "diarization")]
fn speech_spans_from_transcript_prefers_aligned_words(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let transcript = transcript_with_words(vec![("hello", 0.2, 0.5), ("world", 1.0, 1.4)])?;

    let spans = speech_spans_from_transcript(&transcript, 2.0)?;

    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].start_seconds, 0.2);
    assert_eq!(spans[0].end_seconds, 0.5);
    assert_eq!(spans[1].start_seconds, 1.0);
    assert_eq!(spans[1].end_seconds, 1.4);
    assert!(spans[0].start_seconds < spans[1].start_seconds);
    assert!(spans
        .iter()
        .all(|span| span.score.is_finite() && span.score > 0.0));
    Ok(())
}

#[test]
#[cfg(feature = "diarization")]
fn speech_spans_from_transcript_falls_back_to_segments(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let first =
        TranscriptSegmentContract::new(0, "hello").with_time_range(Some(0.2), Some(0.5))?;
    let second =
        TranscriptSegmentContract::new(1, "world").with_time_range(Some(1.0), Some(1.4))?;
    let transcript =
        transcription_from_segments(None, Some("en".to_string()), vec![second, first])?;

    let spans = speech_spans_from_transcript(&transcript, 2.0)?;

    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].start_seconds, 0.2);
    assert_eq!(spans[0].end_seconds, 0.5);
    assert_eq!(spans[1].start_seconds, 1.0);
    assert_eq!(spans[1].end_seconds, 1.4);
    Ok(())
}

#[test]
#[cfg(feature = "diarization")]
fn speech_spans_from_transcript_rejects_out_of_range_timing(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let transcript = transcript_with_words(vec![("hello", 0.2, 2.000_01)])?;

    let error = speech_spans_from_transcript(&transcript, 2.0)
        .unwrap_err()
        .to_string();

    assert!(error.contains("invalid_request"));
    Ok(())
}

#[cfg(feature = "alignment")]
struct AlignmentAwareDiarizationProvider {
    saw_aligned_words: bool,
}

#[cfg(feature = "alignment")]
impl TranscriptDiarizationProvider for AlignmentAwareDiarizationProvider {
    fn provider_id(&self) -> &str {
        "alignment-aware-diarization"
    }

    fn diarize(
        &mut self,
        _audio: LoadedAudio,
        transcript: &TranscriptionContract,
        options: &DiarizationOptions,
    ) -> Result<SpeakerDiarizationResponse> {
        self.saw_aligned_words = transcript.segments.iter().any(|segment| {
            segment.words().iter().any(|word| {
                word.start_seconds().is_some()
                    && word.end_seconds().is_some()
                    && word.confidence().is_some()
            })
        });
        assert!(
            self.saw_aligned_words,
            "diarization should receive transcript word timings from alignment"
        );
        let mut response = diarization_response_for_tests(vec![SpeakerSegmentPrediction {
            speaker: "SPEAKER_ALIGNED".to_string(),
            start_seconds: 0.0,
            end_seconds: 1.0,
            score: Some(0.95),
        }]);
        response.model_id = options.model_id.clone();
        Ok(response)
    }
}

#[cfg(all(feature = "alignment", feature = "candle"))]
fn write_tiny_wav2vec2_bundle(root: &Path) {
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    fs::write(
        root.join("config.json"),
        serde_json::json!({
            "model_type": "wav2vec2",
            "architectures": ["Wav2Vec2ForCTC"],
            "vocab_size": 10,
            "word_delimiter_token": "|",
            "hidden_size": 1,
            "num_hidden_layers": 0,
            "num_attention_heads": 1,
            "intermediate_size": 1,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-5,
            "feat_extract_activation": "gelu",
            "conv_dim": [1],
            "conv_stride": [1],
            "conv_kernel": [1],
            "conv_bias": false,
            "num_conv_pos_embeddings": 0,
            "num_conv_pos_embedding_groups": 1
        })
        .to_string(),
    )
    .unwrap();
    fs::write(
        root.join("tokenizer.json"),
        serde_json::json!({
            "version": "1.0",
            "word_delimiter_token": "|",
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[PAD]": 0,
                    "H": 1,
                    "E": 2,
                    "L": 3,
                    "O": 4,
                    "|": 5,
                    "W": 6,
                    "R": 7,
                    "D": 8,
                    "<unk>": 9
                },
                "unk_token": "<unk>"
            }
        })
        .to_string(),
    )
    .unwrap();
    fs::write(
        root.join("preprocessor_config.json"),
        serde_json::json!({
            "sampling_rate": 16000,
            "do_normalize": false,
            "return_attention_mask": false
        })
        .to_string(),
    )
    .unwrap();

    let device = Device::Cpu;
    let mut tensors = HashMap::new();
    tensors.insert(
        "wav2vec2.feature_extractor.conv_layers.0.conv.weight".to_string(),
        Tensor::new(&[1.0f32], &device)
            .unwrap()
            .reshape((1, 1, 1))
            .unwrap(),
    );
    tensors.insert(
        "wav2vec2.feature_projection.layer_norm.weight".to_string(),
        Tensor::new(&[1.0f32], &device).unwrap(),
    );
    tensors.insert(
        "wav2vec2.feature_projection.layer_norm.bias".to_string(),
        Tensor::new(&[0.0f32], &device).unwrap(),
    );
    tensors.insert(
        "wav2vec2.feature_projection.projection.weight".to_string(),
        Tensor::new(&[1.0f32], &device)
            .unwrap()
            .reshape((1, 1))
            .unwrap(),
    );
    tensors.insert(
        "wav2vec2.feature_projection.projection.bias".to_string(),
        Tensor::new(&[0.0f32], &device).unwrap(),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        Tensor::new(&[0.0f32; 10], &device)
            .unwrap()
            .reshape((10, 1))
            .unwrap(),
    );
    tensors.insert(
        "lm_head.bias".to_string(),
        Tensor::new(&[0.0f32; 10], &device).unwrap(),
    );
    candle_core::safetensors::save(&tensors, root.join("model.safetensors")).unwrap();
}

#[cfg(all(feature = "alignment", feature = "candle", feature = "model-bundles"))]
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

#[cfg(all(feature = "alignment", feature = "candle", feature = "model-bundles"))]
fn write_default_alignment_smoke_wav(path: &Path) -> std::result::Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for index in 0..16_000 {
        let sample = if (1_000..8_000).contains(&index) {
            16_384i16
        } else {
            0i16
        };
        writer.write_sample(sample)?;
    }
    writer.finalize()
}

#[cfg(all(feature = "alignment", feature = "candle", feature = "model-bundles"))]
fn default_alignment_smoke_audio_path(
    temp: &Path,
) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) =
        env_path("ALIGNMENT_AUDIO_PATH").or_else(|| env_path("TRANSCRIPTION_AUDIO_PATH"))
    {
        return Ok(path);
    }

    let repo_sample = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../vendor/whisper.cpp/samples/jfk.wav");
    if repo_sample.exists() {
        return Ok(repo_sample);
    }

    let generated = temp.join("default-alignment-smoke.wav");
    write_default_alignment_smoke_wav(&generated)?;
    Ok(generated)
}

#[cfg(all(feature = "alignment", feature = "candle", feature = "model-bundles"))]
fn resolve_alignment_smoke_bundle_candidate(path: &Path) -> Option<PathBuf> {
    let candidates = [
        path.to_path_buf(),
        path.join("main"),
        path.join("wav2vec2-base-960h"),
        path.join("wav2vec2-base-960h/main"),
        path.join("facebook--wav2vec2-base-960h"),
        path.join("facebook--wav2vec2-base-960h/main"),
        path.join("models--facebook--wav2vec2-base-960h"),
    ];
    candidates
        .into_iter()
        .find(|candidate| native_wav2vec2::resolve_wav2vec2_bundle_paths(candidate).is_ok())
}

#[cfg(all(feature = "alignment", feature = "candle", feature = "model-bundles"))]
fn default_alignment_smoke_bundle(
    temp: &Path,
) -> std::result::Result<(PathBuf, &'static str), Box<dyn std::error::Error>> {
    if let Some(bundle) = env_path("ALIGNMENT_MODEL_BUNDLE") {
        return Ok((bundle, "ALIGNMENT_MODEL_BUNDLE"));
    }

    if let Some(model_dir) = env_path("ALIGNMENT_MODEL_DIR") {
        if let Some(bundle) = resolve_alignment_smoke_bundle_candidate(&model_dir) {
            return Ok((bundle, "ALIGNMENT_MODEL_DIR"));
        }
        return Err(format!(
            "ALIGNMENT_MODEL_DIR did not contain a supported wav2vec2 bundle: {}",
            model_dir.display()
        )
        .into());
    }

    if let Some(xdg_data_home) = env_path("XDG_DATA_HOME") {
        let smoke_models = xdg_data_home.join("video-analysis-smoke/models");
        if let Some(bundle) = resolve_alignment_smoke_bundle_candidate(&smoke_models) {
            return Ok((bundle, "default-xdg-data-home"));
        }
    }

    if let Some(home) = env_path("HOME") {
        let smoke_models = home.join(".local/share/video-analysis-smoke/models");
        if let Some(bundle) = resolve_alignment_smoke_bundle_candidate(&smoke_models) {
            return Ok((bundle, "default-home-local-share"));
        }
    }

    let generated = temp.join("default-wav2vec2-bundle");
    fs::create_dir_all(&generated)?;
    write_tiny_wav2vec2_bundle(&generated);
    Ok((generated, "generated-tiny-bundle"))
}

#[test]
fn majority_overlap_assigns_word_and_segment_speaker(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut transcript = transcript_with_words(vec![("hello", 0.1, 0.7), ("world", 0.8, 1.2)])?;
    let diarization = diarization_response_for_tests(vec![
        SpeakerSegmentPrediction {
            speaker: "speaker_0".to_string(),
            start_seconds: 0.0,
            end_seconds: 0.75,
            score: Some(0.9),
        },
        SpeakerSegmentPrediction {
            speaker: "speaker_1".to_string(),
            start_seconds: 0.75,
            end_seconds: 1.5,
            score: Some(0.8),
        },
    ]);

    assign_speakers_from_diarization(
        &mut transcript,
        &diarization,
        SpeakerAssignmentPolicy::Majority,
    )?;

    assert_eq!(
        transcript.segments[0].words()[0].speaker.as_deref(),
        Some("speaker_0")
    );
    assert_eq!(
        transcript.segments[0].words()[1].speaker.as_deref(),
        Some("speaker_1")
    );
    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("speaker_0"));
    Ok(())
}

#[test]
fn nearest_start_policy_assigns_nearest_speaker(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut transcript = transcript_with_words(vec![("hello", 0.72, 0.9)])?;
    let diarization = diarization_response_for_tests(vec![
        SpeakerSegmentPrediction {
            speaker: "speaker_0".to_string(),
            start_seconds: 0.0,
            end_seconds: 0.4,
            score: Some(0.9),
        },
        SpeakerSegmentPrediction {
            speaker: "speaker_1".to_string(),
            start_seconds: 0.7,
            end_seconds: 1.0,
            score: Some(0.9),
        },
    ]);

    assign_speakers_from_diarization(
        &mut transcript,
        &diarization,
        SpeakerAssignmentPolicy::NearestStart,
    )?;

    assert_eq!(
        transcript.segments[0].words()[0].speaker.as_deref(),
        Some("speaker_1")
    );
    Ok(())
}

#[test]
fn strict_contained_policy_leaves_uncontained_words_unassigned(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut transcript = transcript_with_words(vec![("hello", 0.2, 0.8)])?;
    let diarization = diarization_response_for_tests(vec![SpeakerSegmentPrediction {
        speaker: "speaker_0".to_string(),
        start_seconds: 0.3,
        end_seconds: 0.7,
        score: Some(0.9),
    }]);

    assign_speakers_from_diarization(
        &mut transcript,
        &diarization,
        SpeakerAssignmentPolicy::StrictContained,
    )?;

    assert!(transcript.segments[0].words()[0].speaker.is_none());
    assert!(transcript.segments[0].speaker.is_none());
    Ok(())
}

#[test]
fn existing_segment_speaker_is_not_overwritten(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut transcript = transcript_with_words(vec![("hello", 0.1, 0.4)])?;
    transcript.segments[0].speaker = Some("manual".to_string());
    let diarization = diarization_response_for_tests(vec![SpeakerSegmentPrediction {
        speaker: "speaker_0".to_string(),
        start_seconds: 0.0,
        end_seconds: 1.0,
        score: Some(0.9),
    }]);

    assign_speakers_from_diarization(
        &mut transcript,
        &diarization,
        SpeakerAssignmentPolicy::Majority,
    )?;

    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("manual"));
    assert_eq!(
        transcript.segments[0].words()[0].speaker.as_deref(),
        Some("speaker_0")
    );
    Ok(())
}

#[test]
fn segment_speaker_uses_majority_word_speaker_when_words_are_present(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut transcript = transcript_with_words(vec![
        ("one", 0.0, 0.3),
        ("two", 0.3, 0.6),
        ("three", 0.6, 0.9),
    ])?;
    let diarization = diarization_response_for_tests(vec![
        SpeakerSegmentPrediction {
            speaker: "speaker_0".to_string(),
            start_seconds: 0.0,
            end_seconds: 0.65,
            score: Some(0.9),
        },
        SpeakerSegmentPrediction {
            speaker: "speaker_1".to_string(),
            start_seconds: 0.65,
            end_seconds: 1.0,
            score: Some(0.9),
        },
    ]);

    assign_speakers_from_diarization(
        &mut transcript,
        &diarization,
        SpeakerAssignmentPolicy::Majority,
    )?;

    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("speaker_0"));
    Ok(())
}

#[test]
fn segment_without_words_uses_policy_fallback(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let segment =
        TranscriptSegmentContract::new(0, "hello").with_time_range(Some(0.5), Some(0.8))?;
    let mut transcript =
        transcription_from_segments(None, Some("en".to_string()), vec![segment])?;
    let diarization = diarization_response_for_tests(vec![
        SpeakerSegmentPrediction {
            speaker: "speaker_0".to_string(),
            start_seconds: 0.0,
            end_seconds: 0.2,
            score: Some(0.9),
        },
        SpeakerSegmentPrediction {
            speaker: "speaker_1".to_string(),
            start_seconds: 0.45,
            end_seconds: 1.0,
            score: Some(0.9),
        },
    ]);

    assign_speakers_from_diarization(
        &mut transcript,
        &diarization,
        SpeakerAssignmentPolicy::StrictContained,
    )?;

    assert_eq!(transcript.segments[0].speaker.as_deref(), Some("speaker_1"));
    Ok(())
}

#[test]
fn offset_chunk_local_segments_skips_global_timing() -> Result<()> {
    let mut segment =
        TranscriptSegmentContract::new(0, "hello").with_time_range(Some(10.0), Some(10.5))?;
    segment
        .attributes
        .insert("timing".to_string(), "global".to_string());
    segment.push_word(
        TranscriptWordContract::new("hello").with_time_range(Some(10.0), Some(10.5))?,
    )?;
    let mut transcript =
        transcription_from_segments(None, Some("en".to_string()), vec![segment])
            .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
    let chunks = vec![SpeechSpan::new(5.0, 6.0, 0.8)?];

    offset_chunk_local_segments(&mut transcript, &chunks)?;

    assert_eq!(transcript.segments[0].start_seconds(), Some(10.0));
    assert_eq!(transcript.segments[0].end_seconds(), Some(10.5));
    assert_eq!(
        transcript.segments[0].words()[0].start_seconds(),
        Some(10.0)
    );
    assert_eq!(transcript.segments[0].words()[0].end_seconds(), Some(10.5));
    Ok(())
}

#[test]
fn offset_chunk_local_segments_offsets_local_timing() -> Result<()> {
    let segment =
        TranscriptSegmentContract::new(0, "hello").with_time_range(Some(0.0), Some(0.5))?;
    let mut transcript =
        transcription_from_segments(None, Some("en".to_string()), vec![segment])
            .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
    let chunks = vec![SpeechSpan::new(5.0, 6.0, 0.8)?];

    offset_chunk_local_segments(&mut transcript, &chunks)?;

    assert_eq!(transcript.segments[0].start_seconds(), Some(5.0));
    assert_eq!(transcript.segments[0].end_seconds(), Some(5.5));
    assert_eq!(
        transcript.segments[0]
            .attributes
            .get("timing")
            .map(String::as_str),
        Some("global")
    );
    Ok(())
}

#[test]
fn alignment_overwrites_projected_word_timings() -> Result<()> {
    let mut segment =
        TranscriptSegmentContract::new(0, "hello").with_time_range(Some(0.0), Some(1.0))?;
    let mut word =
        TranscriptWordContract::new("hello").with_time_range(Some(0.0), Some(0.5))?;
    word.attributes = BTreeMap::from([(
        "timing".to_string(),
        "whisperTimestampProjection".to_string(),
    )]);
    segment.push_word(word)?;
    let mut transcript =
        transcription_from_segments(None, Some("en".to_string()), vec![segment])
            .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;

    apply_alignment_words(
        &mut transcript,
        &[AlignedWord {
            segment_index: 0,
            word_index: 0,
            text: "hello".to_string(),
            start_seconds: 0.1,
            end_seconds: 0.4,
            confidence: Some(0.9),
        }],
    )?;

    let word = &transcript.segments[0].words()[0];
    assert_eq!(word.text, "hello");
    assert_eq!(word.start_seconds(), Some(0.1));
    assert_eq!(word.end_seconds(), Some(0.4));
    assert_eq!(word.confidence(), Some(0.9));
    assert_eq!(
        word.attributes.get("timing").map(String::as_str),
        Some("whisperTimestampProjection")
    );
    assert_eq!(transcript.segments[0].start_seconds(), Some(0.1));
    assert_eq!(transcript.segments[0].end_seconds(), Some(0.4));
    Ok(())
}

#[test]
fn provider_plan_reports_candle_primary_native_provider() {
    let plans = transcription_provider_plans();
    let candle = plans
        .iter()
        .find(|plan| plan.provider_id == "candle-whisper")
        .unwrap();
    assert!(candle.primary);
    assert!(!candle.external_runtime);
    assert!(plans
        .iter()
        .any(|plan| plan.provider_id == "whisperx-command" && !plan.primary));
}

#[test]
fn cuda_request_without_feature_returns_setup_error() {
    let mut provider = CandleWhisperTranscriber::new(CandleWhisperOptions {
        device: NativeDevicePreference::Cuda,
        ..CandleWhisperOptions::default()
    });
    let result = provider.transcribe(AsrRequest {
        audio: LoadedAudio {
            samples: vec![0.0; 16],
            sample_rate: 16_000,
            channels: 1,
            source: None,
        },
        chunks: vec![SpeechSpan::new(0.0, 0.001, 0.0).unwrap()],
        task: TranscriptionTask::Transcribe,
        language: None,
        model_id: "openai/whisper-large-v3".to_string(),
    });
    let error = result.unwrap_err().to_string();
    assert!(error.contains("setup_error") || cfg!(feature = "cuda"));
}

#[test]
fn missing_model_bundle_returns_setup_error() {
    let temp = tempfile::tempdir().unwrap();
    let mut provider = CandleWhisperTranscriber::new(CandleWhisperOptions {
        model_dir: Some(temp.path().to_path_buf()),
        model_cache_only: true,
        ..CandleWhisperOptions::default()
    });
    let result = provider.transcribe(AsrRequest {
        audio: LoadedAudio {
            samples: vec![0.0; 16],
            sample_rate: 16_000,
            channels: 1,
            source: None,
        },
        chunks: vec![SpeechSpan::new(0.0, 0.001, 0.0).unwrap()],
        task: TranscriptionTask::Transcribe,
        language: None,
        model_id: "openai/whisper-large-v3".to_string(),
    });
    let error = result.unwrap_err().to_string();
    assert!(error.contains("setup_error") || error.contains("unsupported_runtime"));
    assert!(error.contains("cache-only=true") || error.contains("model-bundles"));
}

#[test]
fn empty_audio_returns_invalid_request() {
    let mut provider = CandleWhisperTranscriber::default();
    let result = provider.transcribe(AsrRequest {
        audio: LoadedAudio {
            samples: Vec::new(),
            sample_rate: 16_000,
            channels: 1,
            source: None,
        },
        chunks: vec![SpeechSpan::new(0.0, 0.001, 0.0).unwrap()],
        task: TranscriptionTask::Transcribe,
        language: None,
        model_id: "openai/whisper-large-v3".to_string(),
    });
    let error = result.unwrap_err().to_string();
    assert!(error.contains("invalid_request"));
    assert!(error.contains("empty audio"));
}

#[test]
fn non_finite_audio_returns_invalid_request() {
    let mut provider = CandleWhisperTranscriber::default();
    let result = provider.transcribe(AsrRequest {
        audio: LoadedAudio {
            samples: vec![0.0, f32::NAN],
            sample_rate: 16_000,
            channels: 1,
            source: None,
        },
        chunks: vec![SpeechSpan::new(0.0, 0.001, 0.0).unwrap()],
        task: TranscriptionTask::Transcribe,
        language: None,
        model_id: "openai/whisper-large-v3".to_string(),
    });
    let error = result.unwrap_err().to_string();
    assert!(error.contains("invalid_request"));
    assert!(error.contains("finite"));
}

#[cfg(not(feature = "audio-io"))]
#[test]
fn path_non_wav_returns_unsupported_runtime() {
    let result = LoadedAudio::mono_16khz_from_source(&TranscriptionSource::Path {
        path: PathBuf::from("clip.mp4"),
    });
    let error = result.unwrap_err().to_string();
    assert!(error.contains("unsupported_runtime"));
    assert!(error.contains("WAV"));
}

#[test]
fn wav_path_decodes_to_mono_16khz() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("stereo-8khz.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 8_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)?;
    for _ in 0..8_000 {
        writer.write_sample::<i16>(16_384)?;
        writer.write_sample::<i16>(0)?;
    }
    writer.finalize()?;

    let audio = LoadedAudio::mono_16khz_from_source(&TranscriptionSource::Path { path })?;
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples.len(), 16_000);
    assert!(audio.samples.iter().all(|sample| sample.is_finite()));
    assert!(audio.samples.iter().any(|sample| *sample > 0.20));
    Ok(())
}

#[test]
fn decode_diagnostics_report_direct_samples_route() {
    let (audio, diagnostics) =
        native_audio::mono_16khz_from_source_with_diagnostics(&TranscriptionSource::Samples {
            samples: vec![0.0, 1.0],
            sample_rate: 8_000,
            channels: 1,
            source: Some("inline".to_string()),
        })
        .unwrap();

    assert_eq!(diagnostics.decode_route, "direct-samples");
    assert_eq!(diagnostics.source_path_extension, None);
    assert_eq!(diagnostics.input_sample_rate, Some(8_000));
    assert_eq!(diagnostics.output_sample_rate, 16_000);
    assert_eq!(diagnostics.output_channels, 1);
    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.channels, 1);
}

#[test]
fn decode_diagnostics_report_wav_route() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("diagnostic.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 8_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)?;
    writer.write_sample::<i16>(16_384)?;
    writer.finalize()?;

    let (_audio, diagnostics) =
        native_audio::mono_16khz_from_source_with_diagnostics(&TranscriptionSource::Path {
            path,
        })?;

    assert_eq!(diagnostics.decode_route, "native-wav-reader");
    assert_eq!(diagnostics.source_path_extension.as_deref(), Some("wav"));
    assert_eq!(diagnostics.input_sample_rate, Some(8_000));
    assert_eq!(diagnostics.output_sample_rate, 16_000);
    assert_eq!(diagnostics.output_channels, 1);
    Ok(())
}

#[test]
fn wav_path_still_uses_native_reader_without_audio_io(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("native-reader.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)?;
    writer.write_sample::<i16>(16_384)?;
    writer.write_sample::<i16>(-16_384)?;
    writer.finalize()?;

    let audio =
        LoadedAudio::mono_16khz_from_source(&TranscriptionSource::Path { path: path.clone() })?;

    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.channels, 1);
    assert_eq!(
        audio.source.as_deref(),
        Some(path.to_string_lossy().as_ref())
    );
    assert_eq!(audio.samples.len(), 2);
    assert!(audio.samples[0] > 0.49);
    assert!(audio.samples[1] < -0.49);
    Ok(())
}

#[cfg(feature = "audio-io")]
#[test]
#[ignore = "requires RUN_NATIVE_MEDIA_DECODE_TESTS=1 and a local FFmpeg-decodable media file"]
fn native_media_decode_when_requested() -> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var("RUN_NATIVE_MEDIA_DECODE_TESTS")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("set RUN_NATIVE_MEDIA_DECODE_TESTS=1 to run native media decode smoke");
        return Ok(());
    }
    let path = std::env::var_os("TRANSCRIPTION_MEDIA_PATH")
        .map(PathBuf::from)
        .map(resolve_smoke_path)
        .ok_or("TRANSCRIPTION_MEDIA_PATH must point to a local media file")?;

    let (audio, diagnostics) =
        native_audio::mono_16khz_from_source_with_diagnostics(&TranscriptionSource::Path {
            path: path.clone(),
        })?;

    assert_eq!(audio.sample_rate, 16_000);
    assert_eq!(audio.channels, 1);
    assert!(!audio.samples.is_empty());
    assert!(audio.samples.iter().all(|sample| sample.is_finite()));
    assert_eq!(diagnostics.decode_route, "audio-io-media-decode");
    assert_eq!(diagnostics.output_sample_rate, 16_000);
    assert_eq!(diagnostics.output_channels, 1);
    assert!(diagnostics.input_sample_rate.is_some());
    assert_eq!(
        audio.source.as_deref(),
        Some(path.to_string_lossy().as_ref())
    );
    Ok(())
}

#[cfg(feature = "audio-io")]
#[test]
fn selected_media_pipeline_transcribes_distinguishable_tracks_and_preserves_default() {
    use audio_analysis_io::{
        is_ffmpeg_available, is_ffprobe_available, write_two_audio_stream_test_media,
    };

    if !(is_ffmpeg_available() && is_ffprobe_available()) {
        eprintln!("skipping selected-media test because ffmpeg/ffprobe is unavailable");
        return;
    }

    #[derive(Default)]
    struct FrequencyAsr;

    impl AudioTranscriptionProvider for FrequencyAsr {
        fn provider_id(&self) -> &str {
            "frequency-asr"
        }

        fn transcribe(&mut self, request: AsrRequest) -> Result<AsrResponse> {
            let crossings = request
                .audio
                .samples
                .windows(2)
                .filter(|window| window[0] <= 0.0 && window[1] > 0.0)
                .count();
            let label = if crossings > 120 { "high" } else { "low" };
            let segment = TranscriptSegmentContract::new(0, label)
                .with_time_range(Some(0.0), Some(request.audio.duration_seconds()))
                .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
            Ok(AsrResponse {
                model_id: request.model_id,
                language: Some("en".to_string()),
                transcript: transcription_from_segments(
                    request.audio.source,
                    Some("en".to_string()),
                    vec![segment],
                )
                .map_err(|error| DetectError::InvalidArgument(error.to_string()))?,
                diagnostics: Vec::new(),
            })
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two-audio-streams.mkv");
    write_two_audio_stream_test_media(&path).unwrap();

    let run = |selection| {
        let mut request = sample_request();
        request.source = match selection {
            Some(audio_stream) => TranscriptionSource::Media {
                path: path.clone(),
                audio_stream,
            },
            None => TranscriptionSource::Path { path: path.clone() },
        };
        request.vad.enabled = false;
        let mut vad = EnergyVoiceActivityDetector;
        let mut asr = FrequencyAsr;
        let mut observer = RecordingObserver::default();
        run_transcription_pipeline_with_observer(
            request,
            &mut vad,
            &mut asr,
            None,
            None,
            &mut observer,
        )
        .unwrap()
        .transcript
        .text
        .unwrap()
    };

    assert_eq!(run(None), "low");
    assert_eq!(run(Some(0)), "low");
    assert_eq!(run(Some(1)), "high");
}

#[cfg(feature = "audio-io")]
#[test]
fn invalid_selected_media_is_typed_before_progress_or_provider_work() {
    use audio_analysis_io::{
        is_ffmpeg_available, is_ffprobe_available, write_two_audio_stream_test_media,
        AudioStreamSelection, AudioStreamSelectionErrorReason, FfmpegError,
    };

    if !(is_ffmpeg_available() && is_ffprobe_available()) {
        eprintln!("skipping selected-media test because ffmpeg/ffprobe is unavailable");
        return;
    }

    struct PanickingAsr;
    impl AudioTranscriptionProvider for PanickingAsr {
        fn provider_id(&self) -> &str {
            "panicking-asr"
        }

        fn transcribe(&mut self, _request: AsrRequest) -> Result<AsrResponse> {
            panic!("invalid selection must be rejected before provider work")
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two-audio-streams.mkv");
    write_two_audio_stream_test_media(&path).unwrap();
    let mut request = sample_request();
    request.source = TranscriptionSource::Path { path };
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = PanickingAsr;
    let mut observer = RecordingObserver::default();

    let error = run_transcription_pipeline_from_selected_media_with_observer(
        request,
        Some(2),
        &mut vad,
        &mut asr,
        None,
        None,
        &mut observer,
    )
    .unwrap_err();

    let SelectedMediaTranscriptionError::Decode(audio_analysis_io::AudioIoError::Ffmpeg(
        FfmpegError::InvalidAudioStreamSelection {
            selection,
            reason,
            available_streams,
        },
    )) = error
    else {
        panic!("expected typed selected-stream error");
    };
    assert_eq!(selection, AudioStreamSelection::AudioOrdinal(2));
    assert_eq!(reason, AudioStreamSelectionErrorReason::OutOfRange);
    assert_eq!(available_streams.streams.len(), 3);
    assert!(observer.events.is_empty());
}

#[cfg(feature = "audio-io")]
fn resolve_smoke_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() || path.exists() {
        return path;
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Some(workspace_root) = manifest_dir.ancestors().nth(3) else {
        return path;
    };
    let workspace_path = workspace_root.join(&path);
    if workspace_path.exists() {
        workspace_path
    } else {
        path
    }
}

#[test]
fn vad_splits_deterministic_synthetic_speech() {
    let request = sample_request();
    let audio = LoadedAudio::mono_16khz_from_source(&request.source).unwrap();
    let mut vad = EnergyVoiceActivityDetector;
    let response = vad
        .detect_speech(&audio.speaker_audio().unwrap(), &request.vad.detection)
        .unwrap();
    assert_eq!(response.segments.len(), 1);
    assert!(response.segments[0].start_seconds < 0.07);
    assert!(response.segments[0].end_seconds > 0.30);
}

#[test]
fn pipeline_preserves_aligned_segment_associations_in_chronological_order() {
    struct OverlappingAsr;
    impl AudioTranscriptionProvider for OverlappingAsr {
        fn provider_id(&self) -> &str {
            "frozen-overlapping-asr"
        }
        fn transcribe(&mut self, request: AsrRequest) -> Result<AsrResponse> {
            let mut segments = Vec::new();
            for (index, text, start, end) in [
                (164, "And...", 0.0, 0.5),
                (165, "It's probably just a-", 0.26, 1.3),
            ] {
                let mut segment = TranscriptSegmentContract::new(index, text)
                    .with_time_range(Some(start), Some(end))?;
                segment.attributes.insert("timing".into(), "global".into());
                segment.speaker = Some(format!("speaker-{index}"));
                segments.push(segment);
            }
            Ok(AsrResponse {
                model_id: request.model_id,
                language: Some("en".into()),
                transcript: transcription_from_segments(
                    request.audio.source,
                    Some("en".into()),
                    segments,
                )?,
                diagnostics: Vec::new(),
            })
        }
    }
    struct OverlappingAlignment;
    impl ForcedAlignmentProvider for OverlappingAlignment {
        fn provider_id(&self) -> &str {
            "frozen-ctc"
        }
        fn align(&mut self, request: AlignmentRequest) -> Result<AlignmentResponse> {
            // Relative timings from Native WhisperX #372's minimized real CTC result.
            Ok(AlignmentResponse {
                model_id: request.model_id,
                words: vec![
                    AlignedWord {
                        segment_index: 164,
                        word_index: 0,
                        text: "And...".into(),
                        start_seconds: 0.35416666666669,
                        end_seconds: 0.47916666666669,
                        confidence: None,
                    },
                    AlignedWord {
                        segment_index: 165,
                        word_index: 0,
                        text: "It's".into(),
                        start_seconds: 0.34156862745098,
                        end_seconds: 0.45,
                        confidence: None,
                    },
                ],
                chars: vec![AlignedChar {
                    segment_index: 164,
                    char_index: 0,
                    character: "A".into(),
                    start_seconds: Some(0.35416666666669),
                    end_seconds: Some(0.4),
                    confidence: None,
                }],
                diagnostics: Vec::new(),
            })
        }
    }
    let mut request = sample_request();
    request.source = TranscriptionSource::Samples {
        samples: vec![0.1; 32_000],
        sample_rate: 16_000,
        channels: 1,
        source: Some("overlap.wav".into()),
    };
    request.alignment.enabled = true;
    let mut vad = FixedVadProvider {
        segments: vec![SpeechSpan::new(0.0, 2.0, 1.0).unwrap()],
    };
    let result = run_transcription_pipeline(
        request,
        &mut vad,
        &mut OverlappingAsr,
        Some(&mut OverlappingAlignment),
        None,
    )
    .expect("overlapping alignment must produce valid chronological segments");
    let transcript = result.transcript;
    transcript.validate_strict().unwrap();
    assert_eq!(
        transcript
            .segments
            .iter()
            .map(|segment| segment.index)
            .collect::<Vec<_>>(),
        vec![165, 164]
    );
    assert_eq!(transcript.segments[0].text, "It's probably just a-");
    assert_eq!(transcript.segments[0].words()[0].text, "It's");
    assert_eq!(
        transcript.segments[0].start_seconds(),
        Some(0.34156862745098)
    );
    assert_eq!(
        transcript.segments[0].speaker.as_deref(),
        Some("speaker-165")
    );
    assert_eq!(transcript.segments[1].text, "And...");
    assert_eq!(transcript.segments[1].words()[0].text, "And...");
    assert_eq!(transcript.segments[1].chars()[0].character, "A");
    assert_eq!(
        transcript.segments[1].start_seconds(),
        Some(0.35416666666669)
    );
    assert_eq!(
        transcript.segments[1].speaker.as_deref(),
        Some("speaker-164")
    );
    assert_eq!(
        transcript.text.as_deref(),
        Some("It's probably just a- And...")
    );
}

#[test]
fn mock_pipeline_normalizes_offsets_alignment_and_diarization() {
    let mut request = sample_request();
    request.alignment.enabled = true;
    request.diarization.enabled = true;
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;
    let mut aligner = MockAlignmentProvider;
    let mut diarizer = MockDiarizationProvider;
    let response = run_transcription_pipeline(
        request,
        &mut vad,
        &mut asr,
        Some(&mut aligner),
        Some(&mut diarizer),
    )
    .unwrap();
    assert!(response.accepted);
    assert_eq!(response.provider, "candle-whisper");
    assert_eq!(response.alignment.as_ref().unwrap().word_count, 1);
    assert_eq!(
        response.transcript.segments[0].words()[0]
            .speaker
            .as_deref(),
        Some("SPEAKER_00")
    );
    assert_eq!(
        response.transcript.segments[0].speaker.as_deref(),
        Some("SPEAKER_00")
    );
}

#[test]
fn native_pipeline_reports_diarization_diagnostics() {
    let mut request = sample_request();
    request.diarization = DiarizationOptions {
        enabled: true,
        speaker: SpeakerDiarizationOptions {
            min_speakers: Some(2),
            max_speakers: Some(3),
            ..SpeakerDiarizationOptions::default()
        },
    };
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;
    let mut diarizer = MockDiarizationProvider;

    let response =
        run_transcription_pipeline(request, &mut vad, &mut asr, None, Some(&mut diarizer))
            .unwrap();

    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationProvider=mock-diarization"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationSegmentCount=1"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationSpeakerCount=1"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationMinSpeakers=2"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationSpeakerBoundsApplied=true"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationSpeakerCountBelowRequestedMin=1/2"));
}

#[cfg(feature = "diarization")]
#[test]
fn native_pipeline_reports_onnx_diarization_diagnostics() {
    let mut request = sample_request();
    request.diarization = DiarizationOptions {
        enabled: true,
        speaker: SpeakerDiarizationOptions {
            speaker_embedding_model_bundle: Some(PathBuf::from("speaker-model")),
            speaker_embedding_dimension: Some(2),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;
    let mut diarizer = MockOnnxDiarizationProvider;

    let response =
        run_transcription_pipeline(request, &mut vad, &mut asr, None, Some(&mut diarizer))
            .unwrap();

    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationRuntime=onnx"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerEmbeddingProvider=onnx"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerEmbeddingDimension=2"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationBaseline=false"));
    assert!(!response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationBaseline=heuristic-native"));
    assert_eq!(
        response.transcript.segments[0].speaker.as_deref(),
        Some("SPEAKER_ONNX")
    );
}

#[cfg(feature = "diarization")]
#[test]
fn native_onnx_diarization_missing_bundle_returns_setup_error() {
    let mut request = sample_request();
    request.diarization = DiarizationOptions {
        enabled: true,
        speaker: SpeakerDiarizationOptions {
            speaker_embedding_model_bundle: Some(PathBuf::from(
                "/definitely/missing/onnx-speaker-model",
            )),
            speaker_embedding_dimension: Some(2),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;
    let mut diarizer = NativeSpeakerDiarizationProvider;

    let error =
        run_native_transcription_pipeline(request, &mut vad, &mut asr, Some(&mut diarizer))
            .unwrap_err()
            .to_string();

    assert!(error.contains("setup_error"));
    assert!(error.contains("ONNX speaker embedding"));
}

#[cfg(all(feature = "diarization", feature = "onnx"))]
#[test]
#[ignore = "requires RUN_NATIVE_SPEAKER_MODEL_TESTS=1 and caller-owned local ONNX speaker model/audio"]
fn native_onnx_diarization_smoke_when_requested(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var("RUN_NATIVE_SPEAKER_MODEL_TESTS").as_deref() != Ok("1") {
        eprintln!(
            "skipping native ONNX diarization smoke; set RUN_NATIVE_SPEAKER_MODEL_TESTS=1"
        );
        return Ok(());
    }

    let bundle_path = std::env::var_os("SPEAKER_EMBEDDING_MODEL_BUNDLE")
        .map(PathBuf::from)
        .ok_or("SPEAKER_EMBEDDING_MODEL_BUNDLE is required")?;
    let audio_path = std::env::var_os("DIARIZATION_AUDIO_PATH")
        .map(PathBuf::from)
        .ok_or("DIARIZATION_AUDIO_PATH is required")?;
    let embedding_dimension = std::env::var("SPEAKER_EMBEDDING_DIMENSION")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(192);
    let model_file =
        optional_smoke_env_value(std::env::var("SPEAKER_EMBEDDING_MODEL_FILE").ok());
    let input_name =
        optional_smoke_env_value(std::env::var("SPEAKER_EMBEDDING_INPUT_NAME").ok());
    let output_name =
        optional_smoke_env_value(std::env::var("SPEAKER_EMBEDDING_OUTPUT_NAME").ok());
    let model_path = resolve_onnx_smoke_model_path(&bundle_path, model_file.as_deref())?;
    eprintln!("speakerEmbeddingResolvedModelPath={}", model_path.display());
    eprintln!("speakerEmbeddingExpectedDimension={embedding_dimension}");
    eprintln!(
        "speakerEmbeddingConfiguredInputName={}",
        input_name.as_deref().unwrap_or("<auto>")
    );
    eprintln!(
        "speakerEmbeddingConfiguredOutputName={}",
        output_name.as_deref().unwrap_or("<auto>")
    );
    let static_metadata = runtime_onnx::inspect_model_metadata(&model_path)?;
    eprintln!("onnxStaticMetadata=ok");
    for diagnostic in runtime_onnx::inspect_model_graph_diagnostics(&model_path)? {
        eprintln!("{diagnostic}");
    }
    eprintln!(
        "onnxLoadMode={}",
        std::env::var("ONNX_RUNTIME_LOAD_MODE").unwrap_or_else(|_| "file".to_string())
    );
    eprintln!(
        "onnxRuntimeDylib={}",
        if std::env::var_os("ORT_DYLIB_PATH").is_some() {
            "set"
        } else {
            "unset"
        }
    );
    if let Some(input) = static_metadata.inputs.first() {
        eprintln!("speakerEmbeddingStaticInputName={}", input.name);
        eprintln!(
            "speakerEmbeddingStaticInputDimensions={}",
            format_onnx_smoke_dimensions(&input.dimensions)
        );
    }
    if let Some(output) = static_metadata.outputs.first() {
        eprintln!("speakerEmbeddingStaticOutputName={}", output.name);
        eprintln!(
            "speakerEmbeddingStaticOutputDimensions={}",
            format_onnx_smoke_dimensions(&output.dimensions)
        );
    }
    eprintln!(
        "onnxSessionOptions=cpu-single-threaded,no-memory-pattern,graph-optimization-disabled"
    );
    let samples = local_16khz_wav_samples(&audio_path)?;
    let duration_seconds = samples.len() as f64 / 16_000.0;
    let midpoint = (duration_seconds / 2.0).clamp(1.0 / 16_000.0, duration_seconds);
    let first_end = midpoint.min(duration_seconds);
    let second_start = first_end;
    let second_end = duration_seconds;

    let mut request = TranscriptionPipelineRequest {
        source: TranscriptionSource::Samples {
            samples,
            sample_rate: 16_000,
            channels: 1,
            source: Some(audio_path.to_string_lossy().into_owned()),
        },
        provider: TranscriptionProviderSelection::CandleWhisper(CandleWhisperOptions::default()),
        vad: TranscriptionVadOptions::default(),
        alignment: AlignmentOptions::default(),
        diarization: DiarizationOptions {
            enabled: true,
            speaker: SpeakerDiarizationOptions {
                speaker_embedding_model_bundle: Some(bundle_path),
                speaker_embedding_model_file: model_file,
                speaker_embedding_input_name: input_name,
                speaker_embedding_output_name: output_name,
                speaker_embedding_dimension: Some(embedding_dimension),
                speaker_embedding_sample_rate: Some(16_000),
                assignment_policy: SpeakerAssignmentPolicy::StrictContained,
                ..SpeakerDiarizationOptions::default()
            },
            ..DiarizationOptions::default()
        },
        output: TranscriptionOutputOptions::default(),
    };
    if second_end > second_start {
        request.vad.enabled = true;
    }

    let segments = if second_end > second_start {
        vec![
            SpeechSpan::new(0.0, first_end, 1.0)?,
            SpeechSpan::new(second_start, second_end, 1.0)?,
        ]
    } else {
        vec![SpeechSpan::new(0.0, first_end, 1.0)?]
    };
    let mut vad = FixedVadProvider { segments };
    let mut asr = MockAsrProvider;
    let mut diarizer = NativeSpeakerDiarizationProvider;

    let response =
        run_native_transcription_pipeline(request, &mut vad, &mut asr, Some(&mut diarizer))?;
    eprintln!("{}", response.diagnostics.join("\n"));

    assert!(response.accepted);
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationRuntime=onnx"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerEmbeddingProvider=onnx"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == &format!("speakerEmbeddingDimension={embedding_dimension}")));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "diarizationBaseline=false"));
    assert!(response.transcript.segments.iter().all(|segment| segment
        .speaker
        .as_deref()
        .is_some_and(|speaker| { !speaker.trim().is_empty() })));
    normalize_transcription_contract(response.transcript.clone())
        .map_err(|error| format!("transcript speaker assignment must validate: {error}"))?;
    Ok(())
}

#[cfg(all(feature = "diarization", feature = "onnx"))]
fn optional_smoke_env_value(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
}

#[cfg(all(feature = "diarization", feature = "onnx"))]
fn resolve_onnx_smoke_model_path(
    bundle_path: &Path,
    model_file: Option<&str>,
) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
    let model_file = model_file.unwrap_or("model.onnx");
    if bundle_path.is_file() {
        return Ok(bundle_path.to_path_buf());
    }
    #[cfg(feature = "model-bundles")]
    {
        let manifest_path = bundle_path.join("manifest.json");
        if manifest_path.is_file() {
            let bundle = model_runtime::ModelBundle::load(&manifest_path)?;
            for file in bundle.manifest.files.values() {
                if file.remote_path == model_file
                    || file.remote_path.ends_with(model_file)
                    || file.local_path.ends_with(model_file)
                {
                    if let Some(path) = bundle.file_path(&file.remote_path) {
                        return Ok(path);
                    }
                }
            }
        }
    }
    Ok(bundle_path.join(model_file))
}

#[cfg(all(feature = "diarization", feature = "onnx"))]
fn format_onnx_smoke_dimensions(dimensions: &[runtime_onnx::OnnxDimension]) -> String {
    let values = dimensions
        .iter()
        .map(|dimension| match dimension {
            runtime_onnx::OnnxDimension::Fixed(value) => value.to_string(),
            runtime_onnx::OnnxDimension::Symbolic(value) => value.clone(),
            runtime_onnx::OnnxDimension::Unknown => "unknown".to_string(),
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{values}]")
}

#[test]
fn native_pipeline_diarization_invalid_bounds_errors_before_provider() {
    let mut request = sample_request();
    request.diarization = DiarizationOptions {
        enabled: true,
        speaker: SpeakerDiarizationOptions {
            min_speakers: Some(3),
            max_speakers: Some(2),
            ..SpeakerDiarizationOptions::default()
        },
    };
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;
    let mut diarizer = PanickingDiarizationProvider { called: false };

    let error =
        run_transcription_pipeline(request, &mut vad, &mut asr, None, Some(&mut diarizer))
            .unwrap_err()
            .to_string();

    assert!(error.contains("invalid_request"));
    assert!(!diarizer.called);
}

#[test]
#[cfg(not(feature = "diarization"))]
fn native_diarization_without_feature_still_reports_no_provider() {
    let mut request = sample_request();
    request.diarization.enabled = true;
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;

    let error = run_transcription_pipeline(request, &mut vad, &mut asr, None, None)
        .unwrap_err()
        .to_string();

    assert!(error.contains("setup_error"));
    assert!(error.contains("no diarization provider is available"));
}

#[test]
#[cfg(feature = "diarization")]
fn native_speaker_diarization_provider_uses_transcript_spans_when_available(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let samples = (0..32_000)
        .map(|index| if index % 80 < 40 { 0.2 } else { -0.2 })
        .collect::<Vec<_>>();
    let audio = LoadedAudio {
        samples,
        sample_rate: 16_000,
        channels: 1,
        source: Some("synthetic".to_string()),
    };
    let transcript = transcript_with_words(vec![("hello", 0.20, 0.50), ("world", 1.00, 1.40)])?;
    let options = DiarizationOptions {
        enabled: true,
        speaker: SpeakerDiarizationOptions {
            model_id: "requested-native-speakers".to_string(),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    let mut provider = NativeSpeakerDiarizationProvider;

    let response = provider.diarize(audio, &transcript, &options)?;

    assert_eq!(response.model_id, "requested-native-speakers");
    assert_eq!(
        response.runtime,
        audio_analysis_speakers::AudioRuntime::Heuristic
    );
    assert_eq!(response.segments.len(), 2);
    assert!((response.segments[0].start_seconds - 0.20).abs() < 0.001);
    assert!((response.segments[0].end_seconds - 0.50).abs() < 0.001);
    assert!((response.segments[1].start_seconds - 1.00).abs() < 0.001);
    assert!((response.segments[1].end_seconds - 1.40).abs() < 0.001);
    assert!(response
        .segments
        .iter()
        .all(|segment| segment.speaker.starts_with("speaker_")));
    Ok(())
}

#[test]
#[cfg(feature = "diarization")]
fn native_speaker_diarization_provider_applies_exact_speaker_bounds(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let audio = two_profile_loaded_audio();
    let transcript = transcript_with_words(vec![("hello", 0.20, 0.50), ("world", 1.00, 1.40)])?;
    let options = DiarizationOptions {
        enabled: true,
        speaker: SpeakerDiarizationOptions {
            min_speakers: Some(2),
            max_speakers: Some(2),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    let mut provider = NativeSpeakerDiarizationProvider;

    let response = provider.diarize(audio, &transcript, &options)?;
    let speakers = response
        .segments
        .iter()
        .map(|segment| segment.speaker.clone())
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(speakers.len(), 2, "{:?}", response.segments);
    assert!(speakers.contains("speaker_0"));
    assert!(speakers.contains("speaker_1"));
    Ok(())
}

#[test]
#[cfg(feature = "diarization")]
fn native_speaker_diarization_provider_applies_max_one_bound(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let audio = two_profile_loaded_audio();
    let transcript = transcript_with_words(vec![("hello", 0.20, 0.50), ("world", 1.00, 1.40)])?;
    let options = DiarizationOptions {
        enabled: true,
        speaker: SpeakerDiarizationOptions {
            max_speakers: Some(1),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    let mut provider = NativeSpeakerDiarizationProvider;

    let response = provider.diarize(audio, &transcript, &options)?;
    let speakers = response
        .segments
        .iter()
        .map(|segment| segment.speaker.clone())
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(speakers.len(), 1, "{:?}", response.segments);
    assert!(speakers.contains("speaker_0"));
    Ok(())
}

#[test]
#[cfg(feature = "diarization")]
fn native_speaker_diarization_provider_falls_back_without_transcript_timing(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let samples = (0..16_000)
        .map(|index| {
            if (1_000..8_000).contains(&index) {
                0.2
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let audio = LoadedAudio {
        samples,
        sample_rate: 16_000,
        channels: 1,
        source: Some("synthetic".to_string()),
    };
    let segment = TranscriptSegmentContract::new(0, "hello without timing");
    let transcript = transcription_from_segments(None, Some("en".to_string()), vec![segment])?;
    let options = DiarizationOptions {
        enabled: true,
        speaker: SpeakerDiarizationOptions {
            model_id: "fallback-native-speakers".to_string(),
            ..SpeakerDiarizationOptions::default()
        },
        ..DiarizationOptions::default()
    };
    let mut provider = NativeSpeakerDiarizationProvider;

    let response = provider.diarize(audio, &transcript, &options)?;

    assert_eq!(response.model_id, "fallback-native-speakers");
    assert_eq!(response.operation, "audio.speakers.diarize");
    assert_eq!(
        response.runtime,
        audio_analysis_speakers::AudioRuntime::Heuristic
    );
    assert!(!response.segments.is_empty());
    Ok(())
}

#[test]
fn alignment_options_default_device_is_cpu() {
    assert_eq!(
        AlignmentOptions::default().device,
        NativeDevicePreference::Cpu
    );
}

#[test]
fn alignment_options_deserializes_cuda_device() {
    let options: AlignmentOptions =
        serde_json::from_str(r#"{"device":"cuda"}"#).expect("alignment options should parse");

    assert_eq!(options.device, NativeDevicePreference::Cuda);
}

#[test]
#[cfg(feature = "alignment")]
fn native_pipeline_supplies_alignment_provider_when_enabled(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    write_tiny_wav2vec2_bundle(temp.path());
    let mut request = sample_request();
    request.alignment = AlignmentOptions {
        enabled: true,
        model_bundle: Some(temp.path().to_path_buf()),
        ..AlignmentOptions::default()
    };
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;

    let response = run_native_transcription_pipeline(request, &mut vad, &mut asr, None)?;

    let alignment = response.alignment.as_ref().unwrap();
    assert_eq!(alignment.provider, "ctc-forced-aligner");
    assert_eq!(alignment.model_id, default_alignment_model());
    assert_eq!(alignment.word_count, 1);
    let word = &response.transcript.segments[0].words()[0];
    assert_eq!(word.text, "hello");
    assert!(word.start_seconds().is_some());
    assert!(word.end_seconds().is_some());
    assert!(word
        .confidence()
        .is_some_and(|confidence| confidence.is_finite()));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "alignmentModelSource=explicit-bundle"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "alignmentDevice=cpu"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "alignmentCuda=false"));
    response.transcript.validate_strict()?;
    Ok(())
}

#[test]
fn native_pipeline_leaves_alignment_absent_when_disabled() {
    let request = sample_request();
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;

    let response =
        run_native_transcription_pipeline(request, &mut vad, &mut asr, None).unwrap();

    assert!(response.alignment.is_none());
    assert!(response
        .diagnostics
        .iter()
        .all(|item| !item.to_lowercase().contains("alignment")));
    response
        .transcript
        .validate_strict()
        .expect("native pipeline response should be strictly valid");
}

#[test]
fn native_pipeline_passes_translate_task_to_asr_provider() {
    let mut request = sample_request();
    request.provider = TranscriptionProviderSelection::CandleWhisper(CandleWhisperOptions {
        task: TranscriptionTask::Translate,
        ..CandleWhisperOptions::default()
    });
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;

    let response =
        run_native_transcription_pipeline(request, &mut vad, &mut asr, None).unwrap();

    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "asrTask=translate"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "translationRuntime=whisper-task"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "translationTargetLanguage=en"));
    assert_eq!(response.transcript.language.as_deref(), Some("en"));
    assert!(response.alignment.is_none());
}

#[test]
fn native_translate_rejects_ctc_alignment() {
    let mut request = sample_request();
    request.provider = TranscriptionProviderSelection::CandleWhisper(CandleWhisperOptions {
        task: TranscriptionTask::Translate,
        ..CandleWhisperOptions::default()
    });
    request.alignment.enabled = true;
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;

    let error = run_native_transcription_pipeline(request, &mut vad, &mut asr, None)
        .unwrap_err()
        .to_string();

    assert!(error.contains("invalid_request"));
    assert!(error.contains("translation output cannot be wav2vec2/CTC-aligned"));
}

#[test]
fn native_translate_allows_diarization_with_segment_timings() {
    let mut request = sample_request();
    request.provider = TranscriptionProviderSelection::CandleWhisper(CandleWhisperOptions {
        task: TranscriptionTask::Translate,
        ..CandleWhisperOptions::default()
    });
    request.diarization.enabled = true;
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;
    let mut diarizer = MockDiarizationProvider;

    let response =
        run_native_transcription_pipeline(request, &mut vad, &mut asr, Some(&mut diarizer))
            .unwrap();

    assert!(response.diarization.is_some());
    assert_eq!(
        response.transcript.segments[0].speaker.as_deref(),
        Some("SPEAKER_00")
    );
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "asrTask=translate"));
}

#[test]
#[cfg(feature = "alignment")]
fn native_pipeline_runs_alignment_before_diarization() {
    let temp = tempfile::tempdir().expect("tempdir");
    write_tiny_wav2vec2_bundle(temp.path());
    let mut request = sample_request();
    request.alignment = AlignmentOptions {
        enabled: true,
        model_bundle: Some(temp.path().to_path_buf()),
        ..AlignmentOptions::default()
    };
    request.diarization.enabled = true;
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;
    let mut diarizer = AlignmentAwareDiarizationProvider {
        saw_aligned_words: false,
    };

    let response =
        run_native_transcription_pipeline(request, &mut vad, &mut asr, Some(&mut diarizer))
            .unwrap();

    assert!(diarizer.saw_aligned_words);
    assert!(response.diarization.is_some());
    let word = &response.transcript.segments[0].words()[0];
    assert!(word.start_seconds().is_some());
    assert!(word.end_seconds().is_some());
    assert_eq!(word.speaker.as_deref(), Some("SPEAKER_ALIGNED"));
    assert_eq!(
        response.transcript.segments[0].speaker.as_deref(),
        Some("SPEAKER_ALIGNED")
    );
}

#[test]
#[cfg(not(feature = "alignment"))]
fn native_pipeline_alignment_without_feature_reports_alignment_unsupported_runtime() {
    let mut request = sample_request();
    request.alignment.enabled = true;
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;

    let error = run_native_transcription_pipeline(request, &mut vad, &mut asr, None)
        .unwrap_err()
        .to_string();

    assert!(error.contains("unsupported_runtime"));
    assert!(error.contains("CTC alignment"));
    assert!(error.contains("alignment"));
    assert!(!error.contains("no alignment provider is available"));
}

#[test]
#[cfg(all(feature = "alignment", feature = "candle"))]
fn native_pipeline_with_tiny_wav2vec2_bundle_runs_model_backed_alignment(
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    write_tiny_wav2vec2_bundle(temp.path());
    let mut request = sample_request();
    request.alignment = AlignmentOptions {
        enabled: true,
        model_bundle: Some(temp.path().to_path_buf()),
        ..AlignmentOptions::default()
    };
    let mut vad = EnergyVoiceActivityDetector;
    let mut asr = MockAsrProvider;

    let response = run_native_transcription_pipeline(request, &mut vad, &mut asr, None)?;

    let alignment = response.alignment.as_ref().unwrap();
    assert_eq!(alignment.provider, "ctc-forced-aligner");
    assert_eq!(alignment.word_count, 1);
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "alignmentModelExecution=candle-wav2vec2"));
    let word = &response.transcript.segments[0].words()[0];
    assert_eq!(word.text, "hello");
    assert!(word.start_seconds().is_some());
    assert!(word.end_seconds().is_some());
    assert!(word.confidence().is_some());
    response.transcript.validate_strict()?;
    Ok(())
}

#[test]
fn missing_command_returns_setup_error() {
    let result = transcribe(TranscriptionPipelineRequest {
        source: TranscriptionSource::Path {
            path: PathBuf::from("missing.wav"),
        },
        provider: TranscriptionProviderSelection::ExternalWhisperX(WhisperXCommandOptions {
            command: PathBuf::from("definitely-missing-whisperx-command"),
            ..WhisperXCommandOptions::default()
        }),
        vad: TranscriptionVadOptions::default(),
        alignment: AlignmentOptions::default(),
        diarization: DiarizationOptions::default(),
        output: TranscriptionOutputOptions::default(),
    });

    let error = result.unwrap_err().to_string();
    assert!(error.contains("setup_error"));
    assert!(error.contains("not found"));
}

#[test]
fn whisperx_json_bytes_prefers_current_source_stem() {
    let temp = tempfile::tempdir().expect("tempdir");
    fs::write(temp.path().join("a.json"), br#"{"text":"first"}"#).expect("a json");
    fs::write(temp.path().join("b.json"), br#"{"text":"second"}"#).expect("b json");

    let (path, bytes) =
        whisperx_json_bytes(Path::new("audio/b.wav"), temp.path(), b"{}").expect("json");

    assert_eq!(path, Some(temp.path().join("b.json")));
    assert_eq!(bytes, br#"{"text":"second"}"#);
}

#[test]
fn reusable_candle_provider_reports_unsupported_without_candle_feature() {
    #[cfg(not(feature = "candle"))]
    {
        let mut provider = ReusableCandleWhisperTranscriber::new(CandleWhisperOptions {
            model_bundle: Some(PathBuf::from("bundle")),
            ..CandleWhisperOptions::default()
        });
        let error = provider
            .transcribe(AsrRequest {
                audio: LoadedAudio {
                    samples: vec![0.0; 16],
                    sample_rate: 16_000,
                    channels: 1,
                    source: None,
                },
                chunks: vec![SpeechSpan::new(0.0, 0.001, 1.0).unwrap()],
                task: TranscriptionTask::Transcribe,
                language: Some("en".to_string()),
                model_id: "tiny.en".to_string(),
            })
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("unsupported_runtime") || error.contains("setup_error"),
            "{error}"
        );
        assert!(
            error.contains("candle") || error.contains("bundle"),
            "{error}"
        );
    }
}

#[test]
fn diarization_requires_token_before_spawn() {
    let result = transcribe(TranscriptionPipelineRequest {
        source: TranscriptionSource::Path {
            path: PathBuf::from("missing.wav"),
        },
        provider: TranscriptionProviderSelection::ExternalWhisperX(WhisperXCommandOptions {
            command: PathBuf::from("definitely-missing-whisperx-command"),
            diarize: true,
            hf_token_env: Some("VIDEO_ANALYSIS_TEST_MISSING_HF_TOKEN".to_string()),
            ..WhisperXCommandOptions::default()
        }),
        vad: TranscriptionVadOptions::default(),
        alignment: AlignmentOptions::default(),
        diarization: DiarizationOptions::default(),
        output: TranscriptionOutputOptions::default(),
    });

    let error = result.unwrap_err().to_string();
    assert!(error.contains("diarization requires"));
    assert!(!error.contains("not found"));
}

#[test]
fn mock_command_output_round_trips() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let command = temp.path().join("mock-whisperx.sh");
    let output_dir = temp.path().join("out");
    fs::write(
        &command,
        format!(
            "#!/usr/bin/env bash\nmkdir -p \"{}\"\nprintf 'Transcript: [0.29 --> 1.47]  hello\\n'\ncat > \"{}/sample.json\" <<'JSON'\n{{\"segments\":[{{\"start\":0.0,\"end\":1.0,\"text\":\" hello \",\"speaker\":\"SPEAKER_00\",\"words\":[{{\"word\":\"hello\",\"start\":0.0,\"end\":0.8,\"score\":0.9,\"speaker\":\"SPEAKER_00\"}}]}}]}}\nJSON\n",
            output_dir.display(),
            output_dir.display()
        ),
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&command)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&command, permissions)?;
    }

    let response = transcribe(TranscriptionPipelineRequest {
        source: TranscriptionSource::Path {
            path: PathBuf::from("speech.wav"),
        },
        provider: TranscriptionProviderSelection::ExternalWhisperX(WhisperXCommandOptions {
            command,
            output_dir: Some(output_dir),
            ..WhisperXCommandOptions::default()
        }),
        vad: TranscriptionVadOptions::default(),
        alignment: AlignmentOptions::default(),
        diarization: DiarizationOptions::default(),
        output: TranscriptionOutputOptions::default(),
    })?;

    assert!(response.accepted);
    assert_eq!(response.transcript.text.as_deref(), Some("hello"));
    assert_eq!(
        response.transcript.segments[0].words()[0]
            .speaker
            .as_deref(),
        Some("SPEAKER_00")
    );
    assert_eq!(response.vad_segments.len(), 1);
    assert_eq!(response.vad_segments[0].start_seconds, 0.29);
    assert_eq!(response.vad_segments[0].end_seconds, 1.47);
    Ok(())
}

#[test]
fn whisperx_stdout_vad_segments_parses_transcript_lines() {
    let segments = whisperx_stdout_vad_segments(
        b"noise\nTranscript: [0.29 --> 1.47]  This is a test.\nTranscript: [2.00 --> 3.25]  More speech.\n",
    );

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].start_seconds, 0.29);
    assert_eq!(segments[0].end_seconds, 1.47);
    assert_eq!(segments[1].start_seconds, 2.0);
    assert_eq!(segments[1].end_seconds, 3.25);
}

#[test]
fn whisperx_args_include_alignment_parity_flags() {
    let args = whisperx_args(
        Path::new("speech.wav"),
        Path::new("out"),
        &WhisperXCommandOptions {
            align_model: Some("facebook/wav2vec2-base-960h".to_string()),
            model_dir: Some(PathBuf::from("models")),
            model_cache_only: true,
            no_align: true,
            interpolate_method: AlignmentInterpolationMethod::Linear,
            return_char_alignments: true,
            ..WhisperXCommandOptions::default()
        },
    );

    assert!(args.iter().any(|arg| arg == "--no_align"));
    assert!(args
        .windows(2)
        .any(|pair| pair[0] == "--align_model" && pair[1] == "facebook/wav2vec2-base-960h"));
    assert!(args
        .windows(2)
        .any(|pair| pair[0] == "--model_dir" && pair[1] == "models"));
    assert!(args.iter().any(|arg| arg == "--model_cache_only"));
    assert!(args
        .windows(2)
        .any(|pair| pair[0] == "--interpolate_method" && pair[1] == "linear"));
    assert!(args.iter().any(|arg| arg == "--return_char_alignments"));
}

#[test]
fn whisperx_args_keep_hugging_face_credentials_out_of_process_arguments() {
    let command = whisperx_command(
        Path::new("speech.wav"),
        Path::new("out"),
        &WhisperXCommandOptions {
            diarize: true,
            ..WhisperXCommandOptions::default()
        },
        Some("secret-token"),
    );
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();

    assert!(!args.iter().any(|arg| arg == "--hf_token"));
    assert!(!args.iter().any(|arg| arg == "secret-token"));
    assert!(command.get_envs().any(|(name, value)| {
        name == "HF_TOKEN" && value.is_some_and(|value| value == "secret-token")
    }));
}

#[test]
fn whisperx_args_include_task_translate() {
    let args = whisperx_args(
        Path::new("speech.wav"),
        Path::new("out"),
        &WhisperXCommandOptions {
            task: TranscriptionTask::Translate,
            ..WhisperXCommandOptions::default()
        },
    );

    assert!(args
        .windows(2)
        .any(|pair| pair[0] == "--task" && pair[1] == "translate"));
}

#[test]
fn timeout_returns_typed_error() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let command = temp.path().join("slow-whisperx.sh");
    fs::write(&command, "#!/usr/bin/env bash\nsleep 2\n")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&command)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&command, permissions)?;
    }

    let result = transcribe(TranscriptionPipelineRequest {
        source: TranscriptionSource::Path {
            path: PathBuf::from("speech.wav"),
        },
        provider: TranscriptionProviderSelection::ExternalWhisperX(WhisperXCommandOptions {
            command,
            timeout_seconds: Some(1),
            ..WhisperXCommandOptions::default()
        }),
        vad: TranscriptionVadOptions::default(),
        alignment: AlignmentOptions::default(),
        diarization: DiarizationOptions::default(),
        output: TranscriptionOutputOptions::default(),
    });
    assert!(result.unwrap_err().to_string().contains("timeout"));
    Ok(())
}

#[test]
#[ignore]
fn native_whisper_hf_cache_smoke() {
    if std::env::var("RUN_NATIVE_WHISPER_MODEL_CACHE_TESTS").as_deref() != Ok("1") {
        eprintln!(
            "skipping native Whisper HF cache smoke; set RUN_NATIVE_WHISPER_MODEL_CACHE_TESTS=1"
        );
        return;
    }
    #[cfg(not(all(feature = "candle", feature = "model-bundles")))]
    panic!("native Whisper HF cache smoke requires candle,model-bundles features");

    #[cfg(all(feature = "candle", feature = "model-bundles"))]
    {
        let audio_path = std::env::var_os("TRANSCRIPTION_AUDIO_PATH")
            .map(PathBuf::from)
            .expect("TRANSCRIPTION_AUDIO_PATH is required");
        let model_dir = std::env::var_os("TRANSCRIPTION_MODEL_DIR")
            .map(PathBuf::from)
            .expect("TRANSCRIPTION_MODEL_DIR is required");
        let model_id =
            std::env::var("TRANSCRIPTION_MODEL_ID").unwrap_or_else(|_| "tiny.en".to_string());
        let response = transcribe(TranscriptionPipelineRequest {
            source: TranscriptionSource::Path { path: audio_path },
            provider: TranscriptionProviderSelection::CandleWhisper(CandleWhisperOptions {
                model_id,
                device: NativeDevicePreference::Cpu,
                language: Some("en".to_string()),
                model_dir: Some(model_dir),
                model_cache_only: true,
                ..CandleWhisperOptions::default()
            }),
            vad: TranscriptionVadOptions::default(),
            alignment: AlignmentOptions {
                enabled: false,
                ..AlignmentOptions::default()
            },
            diarization: DiarizationOptions::default(),
            output: TranscriptionOutputOptions::default(),
        })
        .expect("native Candle Whisper HF cache transcription should run");
        eprintln!("{}", response.diagnostics.join("\n"));
        assert!(response.accepted);
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "asrModelSource=hugging-face-cache"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item.starts_with("asrModelResolved=")));
    }
}

#[test]
#[ignore]
fn candle_whisper_cuda_smoke_when_requested() {
    if std::env::var("RUN_NATIVE_TRANSCRIPTION_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping native transcription smoke; set RUN_NATIVE_TRANSCRIPTION_TESTS=1");
        return;
    }
    #[cfg(not(all(feature = "candle", feature = "cuda", feature = "model-bundles")))]
    panic!("native transcription smoke requires candle,cuda,model-bundles features");

    #[cfg(all(feature = "candle", feature = "cuda", feature = "model-bundles"))]
    {
        let bundle = std::env::var_os("TRANSCRIPTION_MODEL_BUNDLE")
            .map(PathBuf::from)
            .expect("TRANSCRIPTION_MODEL_BUNDLE is required");
        let audio_path = std::env::var_os("TRANSCRIPTION_AUDIO_PATH")
            .map(PathBuf::from)
            .expect("TRANSCRIPTION_AUDIO_PATH is required");
        let response = transcribe(TranscriptionPipelineRequest {
            source: TranscriptionSource::Path { path: audio_path },
            provider: TranscriptionProviderSelection::CandleWhisper(CandleWhisperOptions {
                model_id: "openai/whisper-tiny".to_string(),
                device: NativeDevicePreference::Cuda,
                language: Some("en".to_string()),
                model_bundle: Some(bundle),
                ..CandleWhisperOptions::default()
            }),
            vad: TranscriptionVadOptions::default(),
            alignment: AlignmentOptions::default(),
            diarization: DiarizationOptions::default(),
            output: TranscriptionOutputOptions::default(),
        })
        .expect("native Candle Whisper CUDA transcription should run");
        eprintln!("{}", response.diagnostics.join("\n"));
        assert!(response.accepted);
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "provider=candle-whisper"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "device=cuda:0"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item.starts_with("modelId=")));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item.starts_with("bundle=")));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item.starts_with("chunkCount=")));
        assert!(response.diagnostics.iter().any(|item| item == "cuda=true"));
        assert!(
            response
                .transcript
                .text
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty())
                || !response.transcript.segments.is_empty()
        );
    }
}

#[test]
#[ignore]
fn candle_whisper_cuda_translate_smoke_when_requested() {
    if std::env::var("RUN_NATIVE_TRANSLATION_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping native translation smoke; set RUN_NATIVE_TRANSLATION_TESTS=1");
        return;
    }
    #[cfg(not(all(feature = "candle", feature = "cuda", feature = "model-bundles")))]
    panic!("native translation smoke requires candle,cuda,model-bundles features");

    #[cfg(all(feature = "candle", feature = "cuda", feature = "model-bundles"))]
    {
        let bundle = std::env::var_os("TRANSCRIPTION_MODEL_BUNDLE")
            .map(PathBuf::from)
            .expect("TRANSCRIPTION_MODEL_BUNDLE is required");
        let audio_path = std::env::var_os("TRANSCRIPTION_AUDIO_PATH")
            .map(PathBuf::from)
            .expect("TRANSCRIPTION_AUDIO_PATH is required");
        let response = transcribe(TranscriptionPipelineRequest {
            source: TranscriptionSource::Path { path: audio_path },
            provider: TranscriptionProviderSelection::CandleWhisper(CandleWhisperOptions {
                model_id: "openai/whisper-tiny".to_string(),
                task: TranscriptionTask::Translate,
                device: NativeDevicePreference::Cuda,
                model_bundle: Some(bundle),
                ..CandleWhisperOptions::default()
            }),
            vad: TranscriptionVadOptions::default(),
            alignment: AlignmentOptions {
                enabled: false,
                ..AlignmentOptions::default()
            },
            diarization: DiarizationOptions::default(),
            output: TranscriptionOutputOptions::default(),
        })
        .expect("native Candle Whisper CUDA translation should run");
        eprintln!("{}", response.diagnostics.join("\n"));
        assert!(response.accepted);
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "provider=candle-whisper"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "device=cuda:0"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "asrTask=translate"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "translationRuntime=whisper-task"));
        assert_eq!(response.transcript.language.as_deref(), Some("en"));
        assert!(
            response
                .transcript
                .text
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty())
                || !response.transcript.segments.is_empty()
        );
    }
}

#[test]
#[ignore]
fn ctc_alignment_wav2vec2_smoke_when_requested() {
    if matches!(
        std::env::var("RUN_NATIVE_ALIGNMENT_TESTS").as_deref(),
        Ok("0" | "false" | "FALSE")
    ) {
        eprintln!("skipping native alignment smoke; RUN_NATIVE_ALIGNMENT_TESTS disables it");
        return;
    }
    #[cfg(not(all(feature = "candle", feature = "alignment", feature = "model-bundles")))]
    panic!("native alignment smoke requires candle,alignment,model-bundles features");

    #[cfg(all(feature = "candle", feature = "alignment", feature = "model-bundles"))]
    {
        let temp = tempfile::tempdir().expect("alignment smoke tempdir should be created");
        let (bundle, bundle_source) = default_alignment_smoke_bundle(temp.path())
            .expect("alignment smoke should resolve a default wav2vec2 bundle");
        let audio_path = default_alignment_smoke_audio_path(temp.path())
            .expect("alignment smoke should resolve a default WAV path");
        let layout = native_wav2vec2::inspect_wav2vec2_bundle_layout(&bundle)
            .expect("alignment smoke should inspect wav2vec2 bundle layout");
        eprintln!(
            "alignment smoke defaults: bundleSource={bundle_source} bundle={} audio={}",
            bundle.display(),
            audio_path.display()
        );
        eprintln!("wav2vec2 layout report: {layout:#?}");
        let transcript_text = std::env::var("ALIGNMENT_TRANSCRIPT_TEXT")
            .unwrap_or_else(|_| "hello world".to_string());
        let audio = LoadedAudio::mono_16khz_from_source(&TranscriptionSource::Path {
            path: audio_path,
        })
        .expect("alignment smoke requires readable WAV audio");
        let segment = TranscriptSegmentContract::new(0, transcript_text.clone())
            .with_time_range(
                Some(0.0),
                Some(audio.duration_seconds().clamp(1.0 / 16_000.0, 1.0)),
            )
            .expect("alignment smoke transcript timing should validate");
        let transcript =
            transcription_from_segments(None, Some("en".to_string()), vec![segment])
                .expect("alignment smoke transcript should validate");
        let request = AlignmentRequest {
            audio,
            transcript,
            language: Some("en".to_string()),
            model_id: default_alignment_model(),
        };
        let mut aligner = CtcForcedAligner {
            options: AlignmentOptions {
                enabled: true,
                model_bundle: Some(bundle),
                return_char_alignments: true,
                ..AlignmentOptions::default()
            },
        };
        let response = aligner
            .align(request)
            .expect("native wav2vec2/CTC alignment should run");
        eprintln!("{}", response.diagnostics.join("\n"));
        assert!(!response.words.is_empty());
        assert!(response
            .words
            .iter()
            .all(|word| word.end_seconds >= word.start_seconds));
        assert!(response.words.iter().all(|word| word.confidence.is_some()));
        assert!(!response.chars.is_empty());
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "alignmentProvider=ctc-forced-aligner"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "alignmentModelExecution=candle-wav2vec2"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "alignmentModelSource=explicit-bundle"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "alignmentInterpolateMethod=nearest"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "returnCharAlignments=true"));
    }
}
