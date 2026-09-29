#![cfg(feature = "candle")]

use std::path::PathBuf;

use audio_analysis_transcription::{
    AsrRequest, CandleWhisperComputeType, CandleWhisperDecodeConfig,
    CandleWhisperDecodeRequestConfig, CandleWhisperOptions, CandleWhisperRuntimeControls,
    CandleWhisperTimingMode, CandleWhisperTranscriptionRequestConfig, CandleWhisperWindowControls,
    LoadedAudio, NativeDevicePreference, ReusableCandleWhisperTranscriber, SpeechActivitySegment,
    TranscriptionPipelineEvent, TranscriptionPipelineObserver, TranscriptionTask,
};

#[derive(Default)]
struct RecordingObserver {
    events: Vec<TranscriptionPipelineEvent>,
    resolution_starts: usize,
    resolution_ends: usize,
    cancel_on_reuse: bool,
    cancellation_requested: bool,
}

impl TranscriptionPipelineObserver for RecordingObserver {
    fn observe(&mut self, event: TranscriptionPipelineEvent) {
        if self.cancel_on_reuse && matches!(event, TranscriptionPipelineEvent::ModelReuse { .. }) {
            self.cancellation_requested = true;
        }
        self.events.push(event);
    }

    fn model_resolution_start(&mut self, stage: &str, provider: &str, model_id: &str) {
        assert_eq!(stage, "asr");
        assert_eq!(provider, "candle-whisper");
        assert_eq!(model_id, "openai/whisper-tiny.en");
        self.resolution_starts += 1;
    }

    fn model_resolution_end(&mut self, stage: &str, provider: &str, model_id: &str, _source: &str) {
        assert_eq!(stage, "asr");
        assert_eq!(provider, "candle-whisper");
        assert_eq!(model_id, "openai/whisper-tiny.en");
        self.resolution_ends += 1;
    }

    fn cancellation_requested(&self) -> bool {
        self.cancellation_requested
    }
}

fn request() -> AsrRequest {
    AsrRequest {
        audio: LoadedAudio {
            samples: vec![0.0; 16_000],
            sample_rate: 16_000,
            channels: 1,
            source: Some("controlled-observer-contract".to_string()),
        },
        chunks: vec![SpeechActivitySegment::new(0.0, 1.0, 1.0).unwrap()],
        task: TranscriptionTask::Transcribe,
        language: Some("en".to_string()),
        model_id: "openai/whisper-tiny.en".to_string(),
    }
}

#[test]
#[ignore = "requires the pinned local Whisper bundle; run explicitly with --ignored"]
fn controlled_reusable_candle_operation_reports_reuse_and_honors_cancellation() {
    let bundle = std::env::var_os("CANDLE_WHISPER_TINY_BUNDLE")
        .map(PathBuf::from)
        .expect("CANDLE_WHISPER_TINY_BUNDLE is required for the public observer contract");
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
    let controls = CandleWhisperRuntimeControls {
        cuda_device_index: 0,
        decoder_threads: Some(1),
    };
    let decode = CandleWhisperDecodeRequestConfig {
        search: CandleWhisperDecodeConfig {
            seed: 7,
            ..CandleWhisperDecodeConfig::default()
        },
        ..CandleWhisperDecodeRequestConfig::default()
    };
    let config = CandleWhisperTranscriptionRequestConfig {
        runtime: controls,
        decode: decode.clone(),
        window: CandleWhisperWindowControls {
            timing_mode: CandleWhisperTimingMode::NoTimestamps,
            leading_context_seconds: 0.0,
            trailing_context_seconds: 0.0,
        },
    };
    let mut observer = RecordingObserver::default();

    let first = provider
        .transcribe_with_request_config_and_observer(request(), config.clone(), &mut observer)
        .expect("the first controlled request should load and run the model");

    assert!(first
        .diagnostics
        .iter()
        .any(|item| item == "asrModelSession=loaded"));
    assert!(first
        .diagnostics
        .iter()
        .any(|item| item == "timingMode=noTimestamps"));
    assert!(first
        .diagnostics
        .iter()
        .any(|item| item == "leadingContextSeconds=0"));
    assert!(first
        .diagnostics
        .iter()
        .any(|item| item == "trailingContextSeconds=0"));
    assert_eq!(observer.resolution_starts, 1);
    assert_eq!(observer.resolution_ends, 1);
    assert!(observer
        .events
        .contains(&TranscriptionPipelineEvent::ModelLoadStart {
            stage: "asr".to_string(),
            provider: "candle-whisper".to_string(),
            model_id: "openai/whisper-tiny.en".to_string(),
        }));
    assert!(observer.events.iter().any(|event| matches!(
        event,
        TranscriptionPipelineEvent::ModelLoadEnd {
            stage,
            provider,
            model_id,
            ..
        } if stage == "asr"
            && provider == "candle-whisper"
            && model_id == "openai/whisper-tiny.en"
    )));

    let second_request_events_start = observer.events.len();
    observer.cancel_on_reuse = true;
    let error = provider
        .transcribe_with_request_config_and_observer(request(), config, &mut observer)
        .expect_err("cancellation should stop the reused request at the observer safe check");

    assert!(error.to_string().contains("cancelled"));
    assert_eq!(observer.resolution_starts, 2);
    assert_eq!(observer.resolution_ends, 2);
    assert!(observer.events[second_request_events_start..].contains(
        &TranscriptionPipelineEvent::ModelReuse {
            stage: "asr".to_string(),
            provider: "candle-whisper".to_string(),
            model_id: "openai/whisper-tiny.en".to_string(),
        }
    ));
    assert!(observer.cancellation_requested);
}
