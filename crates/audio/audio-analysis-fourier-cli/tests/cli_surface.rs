#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_analysis_fourier_cli::LIBRARY_CRATE,
        "audio-analysis-fourier"
    );
    let surface = audio_analysis_fourier_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-analysis-fourier");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_analysis_fourier_cli::run_operation(
        "audio.fourier.spectrum",
        serde_json::json!({"samples": [0.0, 1.0, 0.0, -1.0], "sampleRate": 4, "fftSize": 4}),
    )
    .expect("run operation");
    assert_eq!(response.operation.as_str(), "audio.fourier.spectrum");
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}
