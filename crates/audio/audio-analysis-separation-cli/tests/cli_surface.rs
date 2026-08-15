#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_analysis_separation_cli::LIBRARY_CRATE,
        "audio-analysis-separation"
    );
    let surface = audio_analysis_separation_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-analysis-separation");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_analysis_separation_cli::run_operation(
        "audio.separation.expectedStems",
        serde_json::json!({"model": "htdemucs", "format": "wav"}),
    )
    .expect("run operation");
    assert_eq!(
        response.operation.as_str(),
        "audio.separation.expectedStems"
    );
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}
