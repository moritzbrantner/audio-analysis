#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_generation_midi_cli::LIBRARY_CRATE,
        "audio-generation-midi"
    );
    let surface = audio_generation_midi_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-generation-midi");
    assert!(!surface.operations.is_empty());
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_generation_midi_cli::run_operation(
        "audio.midi.note",
        serde_json::json!({"note": 69}),
    )
    .expect("run operation");
    assert_eq!(response.operation.as_str(), "audio.midi.note");
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}
