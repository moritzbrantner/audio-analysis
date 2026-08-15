use audio_analysis_transcription::{
    AsrRequest, CandleWhisperOptions, CandleWhisperTimingMode, CandleWhisperTranscriber,
    CandleWhisperTranscriptionRequestConfig, CandleWhisperWindowControls, LoadedAudio,
    NoopTranscriptionPipelineObserver, ReusableCandleWhisperTranscriber, SpeechActivitySegment,
    TranscriptionTask,
};

#[test]
fn candle_request_controls_preserve_existing_window_defaults() {
    let controls = CandleWhisperWindowControls::default();
    let request = CandleWhisperTranscriptionRequestConfig::default();

    assert_eq!(controls.timing_mode, CandleWhisperTimingMode::Auto);
    assert_eq!(controls.leading_context_seconds, 0.25);
    assert_eq!(controls.trailing_context_seconds, 0.04);
    assert_eq!(request.window, controls);

    let encoded = serde_json::to_value(request).unwrap();
    assert_eq!(encoded["window"]["timingMode"], "auto");
    assert_eq!(encoded["window"]["leadingContextSeconds"], 0.25);
    assert_eq!(encoded["window"]["trailingContextSeconds"], 0.04);

    let decoded: CandleWhisperTranscriptionRequestConfig =
        serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(decoded, CandleWhisperTranscriptionRequestConfig::default());
}

#[test]
fn candle_timing_modes_have_stable_request_values() {
    assert_eq!(
        serde_json::to_value(CandleWhisperTimingMode::Auto).unwrap(),
        "auto"
    );
    assert_eq!(
        serde_json::to_value(CandleWhisperTimingMode::NoTimestamps).unwrap(),
        "noTimestamps"
    );
    assert_eq!(
        serde_json::to_value(CandleWhisperTimingMode::TimestampTokensRequired).unwrap(),
        "timestampTokensRequired"
    );
}

#[test]
fn candle_request_controls_reject_non_finite_or_negative_context() {
    let request = || AsrRequest {
        audio: LoadedAudio {
            samples: vec![0.0; 16],
            sample_rate: 16_000,
            channels: 1,
            source: Some("window-controls-contract".to_string()),
        },
        chunks: vec![SpeechActivitySegment::new(0.0, 0.001, 1.0).unwrap()],
        task: TranscriptionTask::Transcribe,
        language: Some("en".to_string()),
        model_id: "openai/whisper-tiny.en".to_string(),
    };
    let invalid_windows = [
        CandleWhisperWindowControls {
            leading_context_seconds: f64::NAN,
            ..CandleWhisperWindowControls::default()
        },
        CandleWhisperWindowControls {
            leading_context_seconds: -0.01,
            ..CandleWhisperWindowControls::default()
        },
        CandleWhisperWindowControls {
            trailing_context_seconds: f64::INFINITY,
            ..CandleWhisperWindowControls::default()
        },
        CandleWhisperWindowControls {
            trailing_context_seconds: -0.01,
            ..CandleWhisperWindowControls::default()
        },
    ];
    let mut provider = CandleWhisperTranscriber::new(CandleWhisperOptions::default());

    for window in invalid_windows {
        let error = provider
            .transcribe_with_request_config(
                request(),
                CandleWhisperTranscriptionRequestConfig {
                    window,
                    ..CandleWhisperTranscriptionRequestConfig::default()
                },
            )
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("context_seconds must be finite and greater than or equal to zero"));
    }
}

#[test]
fn canonical_observer_entrypoints_validate_request_controls_before_runtime() {
    let request = || AsrRequest {
        audio: LoadedAudio {
            samples: vec![0.0; 16],
            sample_rate: 16_000,
            channels: 1,
            source: Some("canonical-observer-contract".to_string()),
        },
        chunks: vec![SpeechActivitySegment::new(0.0, 0.001, 1.0).unwrap()],
        task: TranscriptionTask::Transcribe,
        language: Some("en".to_string()),
        model_id: "openai/whisper-tiny.en".to_string(),
    };
    let config = CandleWhisperTranscriptionRequestConfig {
        runtime: audio_analysis_transcription::CandleWhisperRuntimeControls {
            decoder_threads: Some(0),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut observer = NoopTranscriptionPipelineObserver;
    let mut single = CandleWhisperTranscriber::new(CandleWhisperOptions::default());
    let mut reusable = ReusableCandleWhisperTranscriber::new(CandleWhisperOptions::default());

    let single_error = single
        .transcribe_with_request_config_and_observer(request(), config.clone(), &mut observer)
        .unwrap_err();
    let reusable_error = reusable
        .transcribe_with_request_config_and_observer(request(), config, &mut observer)
        .unwrap_err();

    assert!(single_error
        .to_string()
        .contains("decoder_threads must be greater than zero"));
    assert!(reusable_error
        .to_string()
        .contains("decoder_threads must be greater than zero"));
}
