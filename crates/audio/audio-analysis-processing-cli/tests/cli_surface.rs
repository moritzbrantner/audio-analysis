#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_analysis_processing_cli::LIBRARY_CRATE,
        "audio-analysis-processing"
    );
    let surface = audio_analysis_processing_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-analysis-processing");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_analysis_processing_cli::run_operation(
        "audio.processing.energy",
        serde_json::json!({"samples": [0.0, 1.0, -1.0], "sampleRate": 3, "channels": 1}),
    )
    .expect("run operation");
    assert_eq!(response.operation.as_str(), "audio.processing.energy");
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}
