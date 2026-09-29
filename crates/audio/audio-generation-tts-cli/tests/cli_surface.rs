#[test]
fn cli_adapter_reports_wrapped_library() {
    assert_eq!(
        audio_generation_tts_cli::LIBRARY_CRATE,
        "audio-generation-tts"
    );
    let surface = audio_generation_tts_cli::package_surface();
    assert_eq!(surface.library, "moenarch-audio-generation-tts");
    assert!(surface
        .operations
        .iter()
        .any(|operation| operation.id.as_str() == "audio.tts.synthesize"));
}

#[test]
fn cli_adapter_runs_representative_operation() {
    let response = audio_generation_tts_cli::run_operation(
        "audio.tts.synthesize",
        serde_json::json!({"text":"Hello from the CLI adapter."}),
    )
    .expect("run operation");
    assert_eq!(response.operation.as_str(), "audio.tts.synthesize");
    assert_eq!(response.value["result"]["audioGenerated"], false);
    assert!(response.value["title"].is_string());
    assert!(response.value["summary"].is_object());
}

#[cfg(all(feature = "candle", feature = "external-tests"))]
#[test]
#[ignore = "requires F5_BUNDLE and VOCOS_BUNDLE pointing at local compatible bundles"]
fn cli_adapter_runs_native_f5_vocos_synthesis_when_requested() {
    let f5_bundle =
        std::env::var("F5_BUNDLE").expect("set F5_BUNDLE to a local compatible F5 bundle");
    let vocos_bundle =
        std::env::var("VOCOS_BUNDLE").expect("set VOCOS_BUNDLE to a local compatible Vocos bundle");
    let response = audio_generation_tts_cli::run_operation(
        "audio.tts.synthesize",
        serde_json::json!({
            "text": "Native CLI smoke.",
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
        }),
    )
    .expect("native cli synthesize");

    assert_eq!(response.value["result"]["status"], "ready");
    assert_eq!(response.value["result"]["audioGenerated"], true);
    assert_eq!(
        response.value["result"]["nativeDiagnostics"]["provider"],
        "f5"
    );
    assert_eq!(
        response.value["result"]["nativeDiagnostics"]["vocoder"],
        "vocos"
    );
}
