#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = audio_analysis_synthesis_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio-analysis-synthesis"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = audio_analysis_synthesis_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"describe","input":{"includeOperations":true}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains(r#""operation""#));
}

#[test]
fn run_endpoint_calls_representative_operation() {
    let response = audio_analysis_synthesis_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"audio.synthesis.tone","input":{"frequencyHz":440.0,"durationSeconds":0.01,"sampleRate":1000}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio.synthesis.tone"));
    assert!(response.body.contains("\"title\""));
    assert!(response.body.contains("\"summary\""));
}
