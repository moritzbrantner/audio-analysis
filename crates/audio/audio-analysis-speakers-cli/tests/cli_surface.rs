#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_analysis_speakers_cli::LIBRARY_CRATE,
        "audio-analysis-speakers"
    );
    let surface = audio_analysis_speakers_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-analysis-speakers");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_analysis_speakers_cli::run_operation(
        "audio.speakers.embed",
        serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 4, "fftSize": 4, "hopSize": 2, "bands": 2}),
    )
    .expect("run operation");
    assert_eq!(response.operation.as_str(), "audio.speakers.embed");
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}
