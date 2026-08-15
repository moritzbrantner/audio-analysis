#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(audio_analysis_io_cli::LIBRARY_CRATE, "audio-analysis-io");
    let surface = audio_analysis_io_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-analysis-io");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_analysis_io_cli::run_operation(
        "audio.io.inputPlan",
        serde_json::json!({"source": "clip.wav"}),
    )
    .expect("run operation");
    assert_eq!(response.operation.as_str(), "audio.io.inputPlan");
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}
