use audio_analysis_transcription::{
    inspect_alignment_resources, inspect_candle_whisper_resources, AlignmentOptions,
    CandleWhisperOptions, CandleWhisperRuntimeControls,
};

#[cfg(feature = "candle")]
use audio_analysis_transcription::NativeDevicePreference;

#[cfg(feature = "candle")]
fn whisper_bundle(root: &std::path::Path) {
    for name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "model.safetensors",
    ] {
        std::fs::write(root.join(name), b"deliberately not model data").unwrap();
    }
}

#[cfg(feature = "candle")]
#[test]
fn whisper_inspects_explicit_files_without_loading_them() {
    let dir = tempfile::tempdir().unwrap();
    whisper_bundle(dir.path());
    let report = inspect_candle_whisper_resources(
        &CandleWhisperOptions {
            model_id: "tiny.en".into(),
            device: NativeDevicePreference::Cpu,
            model_bundle: Some(dir.path().into()),
            ..Default::default()
        },
        CandleWhisperRuntimeControls::default(),
    )
    .unwrap();
    assert_eq!(report.model.model_id, "openai/whisper-tiny.en");
    assert_eq!(report.model.bundle_path, dir.path().canonicalize().unwrap());
    assert_eq!(report.model.required_files.len(), 5);
    assert!(report
        .model
        .required_files
        .iter()
        .all(|file| file.is_absolute() && file.is_file()));
}

#[cfg(feature = "candle")]
#[test]
fn whisper_rejects_a_directory_in_place_of_weights() {
    let dir = tempfile::tempdir().unwrap();
    whisper_bundle(dir.path());
    std::fs::remove_file(dir.path().join("model.safetensors")).unwrap();
    std::fs::create_dir(dir.path().join("model.safetensors")).unwrap();
    let error = inspect_candle_whisper_resources(
        &CandleWhisperOptions {
            device: NativeDevicePreference::Cpu,
            model_bundle: Some(dir.path().into()),
            ..Default::default()
        },
        CandleWhisperRuntimeControls::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("model.safetensors"), "{error}");
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
#[test]
fn cache_inspection_never_creates_or_downloads_even_when_options_allow_it() {
    let dir = tempfile::tempdir().unwrap();
    let absent_cache = dir.path().join("missing");
    let error = inspect_candle_whisper_resources(
        &CandleWhisperOptions {
            device: NativeDevicePreference::Cpu,
            model_dir: Some(absent_cache.clone()),
            model_cache_only: false,
            ..Default::default()
        },
        CandleWhisperRuntimeControls::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("cache"), "{error}");
    assert!(!absent_cache.exists());
    whisper_bundle(dir.path());
    let report = inspect_candle_whisper_resources(
        &CandleWhisperOptions {
            device: NativeDevicePreference::Cpu,
            model_dir: Some(dir.path().into()),
            ..Default::default()
        },
        CandleWhisperRuntimeControls::default(),
    )
    .unwrap();
    assert_eq!(report.model.bundle_path, dir.path().canonicalize().unwrap());
}

#[cfg(feature = "alignment")]
#[test]
fn alignment_inspects_files_without_opening_weight_tensors() {
    let dir = tempfile::tempdir().unwrap();
    for name in [
        "config.json",
        "vocab.json",
        "preprocessor_config.json",
        "model.safetensors",
    ] {
        std::fs::write(dir.path().join(name), b"not a tensor or config").unwrap();
    }
    let report = inspect_alignment_resources(&AlignmentOptions {
        enabled: true,
        model_id: "WAV2VEC2_ASR_BASE_960H".into(),
        model_bundle: Some(dir.path().into()),
        ..Default::default()
    })
    .unwrap()
    .unwrap();
    assert_eq!(report.model_id, "facebook/wav2vec2-base-960h");
    assert_eq!(report.required_files.len(), 4);
}

#[test]
fn disabled_alignment_needs_no_feature_or_resource() {
    assert!(inspect_alignment_resources(&AlignmentOptions::default())
        .unwrap()
        .is_none());
}

#[cfg(not(feature = "candle"))]
#[test]
fn missing_candle_feature_is_explicit() {
    let error = inspect_candle_whisper_resources(
        &CandleWhisperOptions::default(),
        CandleWhisperRuntimeControls::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("candle"), "{error}");
}

#[cfg(not(feature = "alignment"))]
#[test]
fn enabled_alignment_reports_missing_feature() {
    let error = inspect_alignment_resources(&AlignmentOptions {
        enabled: true,
        ..Default::default()
    })
    .unwrap_err();
    assert!(error.to_string().contains("alignment"), "{error}");
}

#[cfg(all(feature = "candle", not(feature = "cuda")))]
#[test]
fn explicit_cuda_never_falls_back_to_cpu() {
    let error = inspect_candle_whisper_resources(
        &CandleWhisperOptions {
            device: NativeDevicePreference::Cuda,
            ..Default::default()
        },
        CandleWhisperRuntimeControls::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("CUDA"), "{error}");
}

#[cfg(feature = "candle")]
#[test]
fn q8_inspection_requires_gguf_and_does_not_substitute_safetensors() {
    use audio_analysis_transcription::CandleWhisperComputeType;
    let dir = tempfile::tempdir().unwrap();
    whisper_bundle(dir.path());
    let options = CandleWhisperOptions {
        device: NativeDevicePreference::Cpu,
        compute_type: CandleWhisperComputeType::Int8,
        model_bundle: Some(dir.path().into()),
        ..Default::default()
    };
    let error = inspect_candle_whisper_resources(&options, Default::default()).unwrap_err();
    assert!(error.to_string().contains("model.q8_0.gguf"), "{error}");
    std::fs::write(dir.path().join("model.q8_0.gguf"), b"not gguf").unwrap();
    let report = inspect_candle_whisper_resources(&options, Default::default()).unwrap();
    assert_eq!(report.compute_type, CandleWhisperComputeType::Int8);
    assert!(report
        .model
        .required_files
        .contains(&dir.path().join("model.q8_0.gguf")));
    assert!(!report
        .model
        .required_files
        .contains(&dir.path().join("model.safetensors")));
}

#[cfg(all(feature = "alignment", feature = "model-bundles"))]
#[test]
fn alignment_cache_miss_is_read_only_even_if_downloads_are_allowed() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing-cache");
    let error = inspect_alignment_resources(&AlignmentOptions {
        enabled: true,
        model_dir: Some(missing.clone()),
        model_cache_only: false,
        ..Default::default()
    })
    .unwrap_err();
    assert!(error.to_string().contains("cache"), "{error}");
    assert!(!missing.exists());
}

#[cfg(feature = "candle")]
#[test]
fn invalid_cpu_device_index_is_not_ignored() {
    let error = inspect_candle_whisper_resources(
        &CandleWhisperOptions {
            device: NativeDevicePreference::Cpu,
            ..Default::default()
        },
        CandleWhisperRuntimeControls {
            cuda_device_index: 1,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("cuda_device_index"), "{error}");
}
