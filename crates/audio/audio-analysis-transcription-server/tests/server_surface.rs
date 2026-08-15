#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = audio_analysis_transcription_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio-analysis-transcription"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = audio_analysis_transcription_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"describe","input":{"includeOperations":true}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains(r#""operation""#));
}

#[test]
fn run_endpoint_calls_representative_operation() {
    let response = audio_analysis_transcription_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"audio.transcription.importWhisperX","input":{"content":"{\"segments\":[{\"start\":0.0,\"end\":1.0,\"text\":\"Hello.\"}]}"}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio.transcription.importWhisperX"));
    assert!(response.body.contains("\"segments\""));
}
