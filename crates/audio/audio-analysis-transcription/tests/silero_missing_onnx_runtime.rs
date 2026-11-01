#![cfg(feature = "silero-vad")]

// Its own test binary: ONNX Runtime is loaded once per process (vox#159).
#[test]
fn silero_without_onnx_runtime_reports_unavailable_instead_of_hanging() {
    let dir = std::env::temp_dir().join(format!("silero-missing-ort-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("silero_vad.onnx");
    std::fs::write(&model, b"not a model").unwrap();
    std::env::set_var("ORT_DYLIB_PATH", dir.join("libonnxruntime-missing.so"));

    let Err(error) = audio_analysis_transcription::SileroVadStream::from_options(
        audio_analysis_transcription::SileroVadOptions {
            model_path: model,
            input_name: None,
            output_name: None,
            threshold: 0.5,
            max_speech_duration_seconds: 30.0,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
        },
    ) else {
        panic!("Silero must fail without an ONNX Runtime dylib");
    };

    let message = error.to_string();
    assert!(message.contains("ONNX Runtime unavailable"), "{message}");
    assert!(message.contains("ORT_DYLIB_PATH"), "{message}");
    std::fs::remove_dir_all(&dir).ok();
}
