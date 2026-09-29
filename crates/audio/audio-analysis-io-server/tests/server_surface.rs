#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = audio_analysis_io_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio-analysis-io"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = audio_analysis_io_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"describe","input":{"includeOperations":true}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains(r#""operation""#));
}

#[test]
fn run_endpoint_calls_representative_operation() {
    let response = audio_analysis_io_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"audio.io.inputPlan","input":{"source":"clip.wav"}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio.io.inputPlan"));
    assert!(response.body.contains("\"title\""));
    assert!(response.body.contains("\"summary\""));
}
