#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_analysis_synthesis_cli::LIBRARY_CRATE,
        "audio-analysis-synthesis"
    );
    let surface = audio_analysis_synthesis_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-analysis-synthesis");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_analysis_synthesis_cli::run_operation(
        "audio.synthesis.tone",
        serde_json::json!({"frequencyHz": 440.0, "durationSeconds": 0.01, "sampleRate": 1000}),
    )
    .expect("run operation");
    assert_eq!(response.operation.as_str(), "audio.synthesis.tone");
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}
