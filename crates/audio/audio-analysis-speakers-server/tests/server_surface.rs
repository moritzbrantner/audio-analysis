#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = audio_analysis_speakers_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio-analysis-speakers"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = audio_analysis_speakers_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"describe","input":{"includeOperations":true}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains(r#""operation""#));
}

#[test]
fn run_endpoint_calls_representative_operation() {
    let response = audio_analysis_speakers_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"audio.speakers.embed","input":{"samples":[0.0,1.0,0.0,-1.0],"sampleRate":4,"fftSize":4,"hopSize":2,"bands":2}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio.speakers.embed"));
    assert!(response.body.contains("\"title\""));
    assert!(response.body.contains("\"summary\""));
}
