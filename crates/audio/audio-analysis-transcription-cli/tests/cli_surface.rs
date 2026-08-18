#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_analysis_transcription_cli::LIBRARY_CRATE,
        "audio-analysis-transcription"
    );
    let surface = audio_analysis_transcription_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-analysis-transcription");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_analysis_transcription_cli::run_operation(
        "audio.transcription.importWhisperX",
        serde_json::json!({
            "content": "{\"segments\":[{\"start\":0.0,\"end\":1.0,\"text\":\"Hello.\"}]}"
        }),
    )
    .expect("run operation");
    assert_eq!(
        response.operation.as_str(),
        "audio.transcription.importWhisperX"
    );
    assert!(response.value["text"].is_string());
    assert!(response.value["segments"].is_array());
}
