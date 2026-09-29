#![cfg(all(feature = "native", feature = "external-tests"))]

use std::path::PathBuf;

use audio_analysis_transcription::{
    whisper_cpp_catalog, NativeWhisperCppTranscriber, WhisperCppConfig, WhisperCppModel,
    WhisperCppModelStore,
};

#[test]
#[ignore = "validates opt-in whisper.cpp model store and only runs when local models exist"]
fn whisper_catalog_reports_cached_models() {
    let catalog = whisper_cpp_catalog();
    assert_eq!(catalog.default_model, WhisperCppModel::default());
    assert!(!catalog.models.is_empty());
    for status in catalog.models {
        let _ = status.cached;
        let _ = status.model.file_name();
    }
}

#[test]
#[ignore = "runs native whisper.cpp only when RUN_NATIVE_WHISPER_TESTS=1 and a local fixture/model are provided"]
fn native_whisper_cpp_smoke_when_requested() {
    if std::env::var("RUN_NATIVE_WHISPER_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping native Whisper smoke test; set RUN_NATIVE_WHISPER_TESTS=1 to enable");
        return;
    }

    let input = std::env::var_os("NATIVE_WHISPER_AUDIO_PATH")
        .map(PathBuf::from)
        .expect("set NATIVE_WHISPER_AUDIO_PATH to a local 16 kHz mono WAV fixture");
    assert!(
        input.is_file(),
        "NATIVE_WHISPER_AUDIO_PATH must point to an existing local audio file"
    );

    let store = std::env::var_os("WHISPER_CPP_MODEL_STORE")
        .map(PathBuf::from)
        .map(WhisperCppModelStore::new)
        .unwrap_or_default();
    let model = WhisperCppModel::TinyEn;
    assert!(
        store.model_path(model).is_file(),
        "missing cached whisper.cpp model `{}` at `{}`; run the explicit model-runtime setup before enabling RUN_NATIVE_WHISPER_TESTS",
        model,
        store.model_path(model).display()
    );

    let mut transcriber = NativeWhisperCppTranscriber::new(WhisperCppConfig {
        model,
        language: Some("en".to_string()),
        translate: false,
        threads: Some(1),
    })
    .with_model_store(store);
    let transcript = transcriber
        .transcribe_file(&input)
        .expect("native whisper.cpp transcription should succeed");

    assert!(!transcript.segments.is_empty() || transcript.text.is_some());
}
