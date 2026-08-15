#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_analysis_pitch_cli::LIBRARY_CRATE,
        "audio-analysis-pitch"
    );
    let surface = audio_analysis_pitch_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-analysis-pitch");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_analysis_pitch_cli::run_operation(
        "audio.pitch.noteName",
        serde_json::json!({"frequencyHz": 440.0}),
    )
    .expect("run operation");
    assert_eq!(response.operation.as_str(), "audio.pitch.noteName");
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}
