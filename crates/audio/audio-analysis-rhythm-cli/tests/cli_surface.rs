#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_analysis_rhythm_cli::LIBRARY_CRATE,
        "audio-analysis-rhythm"
    );
    let surface = audio_analysis_rhythm_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-analysis-rhythm");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_analysis_rhythm_cli::run_operation(
        "audio.rhythm.beatGrid",
        serde_json::json!({"startSeconds": 0.0, "bpm": 120.0, "beats": 4}),
    )
    .expect("run operation");
    assert_eq!(response.operation.as_str(), "audio.rhythm.beatGrid");
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}
