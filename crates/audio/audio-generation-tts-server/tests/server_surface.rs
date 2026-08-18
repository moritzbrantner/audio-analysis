#[test]
fn package_endpoint_reports_wrapped_library() {
    let response = audio_generation_tts_server::response_for("GET", "/api/package", "");
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio-generation-tts"));
}

#[test]
fn run_endpoint_calls_library_surface() {
    let response = audio_generation_tts_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"describe","input":{"includeOperations":true}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains(r#""operation""#));
}

#[test]
fn run_endpoint_calls_representative_operation() {
    let response = audio_generation_tts_server::response_for(
        "POST",
        "/api/run",
        r#"{"operation":"audio.tts.synthesize","input":{"text":"Hello from server adapter."}}"#,
    );
    assert_eq!(response.status_code, 200);
    assert!(response.body.contains("audio.tts.synthesize"));
    assert!(response.body.contains("\"title\""));
    assert!(response.body.contains("\"summary\""));
}

#[cfg(all(feature = "candle", feature = "external-tests"))]
#[test]
#[ignore = "requires F5_BUNDLE and VOCOS_BUNDLE pointing at local compatible bundles"]
fn server_run_endpoint_runs_native_f5_vocos_synthesis_when_requested() {
    let f5_bundle =
        std::env::var("F5_BUNDLE").expect("set F5_BUNDLE to a local compatible F5 bundle");
    let vocos_bundle =
        std::env::var("VOCOS_BUNDLE").expect("set VOCOS_BUNDLE to a local compatible Vocos bundle");
    let body = serde_json::json!({
        "operation": "audio.tts.synthesize",
        "input": {
            "text": "Native server smoke.",
            "referenceVoicePrompt": {
                "audio": {
                    "sampleRateHz": 24000,
                    "channels": 1,
                    "samples": [0.0, 0.01, -0.01, 0.0]
                },
                "transcript": "Reference voice prompt text."
            },
            "provider": {
                "providerId": "f5",
                "modelId": std::env::var("F5_MODEL_ID").unwrap_or_else(|_| "f5-tts-v1-base".to_string()),
                "native": true,
                "device": "cpu",
                "modelBundle": {
                    "bundlePath": f5_bundle
                },
                "vocoder": {
                    "providerId": "vocos",
                    "modelId": std::env::var("VOCOS_MODEL_ID").unwrap_or_else(|_| "vocos-mel-24khz".to_string()),
                    "modelBundle": {
                        "bundlePath": vocos_bundle
                    }
                }
            },
            "options": {
                "maxDurationSeconds": 0.02
            }
        }
    });
    let response = audio_generation_tts_server::response_for("POST", "/api/run", &body.to_string());

    assert_eq!(response.status_code, 200);
    assert!(response.body.contains(r#""status":"ready""#));
    assert!(response.body.contains(r#""audioGenerated":true"#));
    assert!(response.body.contains(r#""provider":"f5""#));
    assert!(response.body.contains(r#""vocoder":"vocos""#));
}
