//! Library-owned runtime surface for `audio-generation-tts`.

use model_runtime::{ModelFileRequest, ModelPreset, ModelSpec};
use runtime_core::{
    set_surface_operation_curation, structured_surface_response, OperationId, PackageSurface,
    RuntimeCapabilities, SurfaceOperation, SurfaceOperationCuration, SurfaceRequest,
    SurfaceResponse,
};

use crate::{
    run_f5_mel_diagnostic, run_vocos_vocoder_diagnostic, speech_synthesis_text_chunking_report,
    synthesize, NativeF5MelDiagnosticRequest, NativeTtsDevicePreference,
    NativeVocosVocoderDiagnosticRequest, PcmAudio, ReferenceVoicePrompt, ReferenceVoicePromptAudio,
    SpeechSynthesisRequest, SpeechSynthesisStatus, TtsModelBundleSelection,
};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust().with_requirement(
            "native-tts-provider",
            "Native F5 + Vocos synthesis requires explicit native selection, local bundles, and the candle feature.",
            cfg!(feature = "candle"),
        ),
        operations: vec![
            operation(
                "describe",
                "Describe package",
                "Generic and speaker-conditioned TTS contracts, validation, and setup diagnostics.",
                serde_json::json!({"includeOperations": true}),
                SurfaceOperationCuration::debug(900),
            ),
            operation(
                "audio.tts.synthesize",
                "Synthesize speech",
                "Validates a TTS request and runs explicit native F5 + Vocos synthesis when local setup is available.",
                example_synthesis_request(),
                SurfaceOperationCuration::workflow(10).primary(),
            ),
            operation(
                "audio.tts.plan",
                "Preview synthesis plan",
                "Previews provider, runtime, and output requirements without synthesizing audio.",
                example_synthesis_request(),
                SurfaceOperationCuration::debug(910),
            ),
            operation(
                "audio.tts.models",
                "Inspect TTS models",
                "Inspects the current side-effect-free TTS model inventory.",
                serde_json::json!({}),
                SurfaceOperationCuration::debug(920),
            ),
            operation(
                "audio.tts.referencePromptPlan",
                "Inspect reference prompt plan",
                "Inspects Reference Voice Prompt readiness for speaker-conditioned TTS.",
                serde_json::json!({"referenceVoicePrompt": example_reference_prompt()}),
                SurfaceOperationCuration::debug(930),
            ),
            operation(
                "audio.tts.debug.f5Mel",
                "Debug F5 mel generation",
                "Validates a local F5 bundle and runs a mel-level native diagnostic without vocoding audio.",
                example_f5_debug_request(),
                SurfaceOperationCuration::debug(940),
            ),
            operation(
                "audio.tts.debug.vocosVocoder",
                "Debug Vocos vocoder",
                "Validates a local Vocos bundle and vocodes a constrained mel diagnostic into PCM audio.",
                example_vocos_debug_request(),
                SurfaceOperationCuration::debug(950),
            ),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
    curation: SurfaceOperationCuration,
) -> SurfaceOperation {
    let mut operation = runtime_core::surface_operation(id, name, description, example_request);
    set_surface_operation_curation(&mut operation, curation);
    operation
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "audio.tts.synthesize" => synthesize_value(request.input)?,
        "audio.tts.plan" => plan_value(request.input)?,
        "audio.tts.models" => models_value(),
        "audio.tts.referencePromptPlan" => reference_prompt_plan_value(request.input)?,
        "audio.tts.debug.f5Mel" => f5_mel_debug_value(request.input)?,
        "audio.tts.debug.vocosVocoder" => vocos_vocoder_debug_value(request.input)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ));
        }
    };
    Ok(response(operation, value))
}

fn response(operation: OperationId, value: serde_json::Value) -> SurfaceResponse {
    let (title, message, summary) = match operation.as_str() {
        "describe" => (
            "TTS package metadata",
            "Inspected the generic and speaker-conditioned TTS operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.tts.synthesize" => (
            "TTS synthesis setup result",
            "Validated the synthesis request and returned audio or explicit setup diagnostics.",
            serde_json::json!({
                "status": value.get("status").cloned().unwrap_or(serde_json::Value::Null),
                "audioGenerated": value.get("audioGenerated").cloned().unwrap_or(serde_json::Value::Null),
                "nativeRuntime": value.get("nativeDiagnostics").and_then(|diagnostics| diagnostics.get("runtime")).cloned().unwrap_or(serde_json::Value::Null),
                "diagnosticCount": value.get("diagnostics").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.tts.plan" => (
            "TTS synthesis plan",
            "Previewed provider, runtime, and output requirements without synthesizing audio.",
            serde_json::json!({
                "willSynthesize": value.get("willSynthesize").cloned().unwrap_or(serde_json::Value::Null),
                "status": value.get("status").cloned().unwrap_or(serde_json::Value::Null),
                "speakerConditioned": value.get("speakerConditioned").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.tts.models" => (
            "TTS model inventory",
            "Inspected current TTS model inventory state without selecting or downloading a model.",
            serde_json::json!({
                "modelCount": value.get("models").and_then(serde_json::Value::as_array).map_or(0, Vec::len),
                "defaultModelSelected": value.get("defaultModelSelected").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.tts.referencePromptPlan" => (
            "Reference prompt plan",
            "Inspected Reference Voice Prompt readiness for speaker-conditioned TTS.",
            serde_json::json!({
                "provided": value.get("provided").cloned().unwrap_or(serde_json::Value::Null),
                "transcriptProvided": value.get("transcriptProvided").cloned().unwrap_or(serde_json::Value::Null),
                "action": value.get("action").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.tts.debug.f5Mel" => (
            "F5 mel diagnostic",
            "Validated the F5 provider path and returned a mel-level diagnostic without vocoding audio.",
            serde_json::json!({
                "status": value.get("status").cloned().unwrap_or(serde_json::Value::Null),
                "modelId": value.get("modelId").cloned().unwrap_or(serde_json::Value::Null),
                "audioGenerated": value.get("audioGenerated").cloned().unwrap_or(serde_json::Value::Null),
                "vocoderRequired": value.get("vocoderRequired").cloned().unwrap_or(serde_json::Value::Null),
                "diagnosticCount": value.get("diagnostics").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.tts.debug.vocosVocoder" => (
            "Vocos vocoder diagnostic",
            "Validated the Vocos provider path and returned a constrained PCM audio diagnostic when setup is available.",
            serde_json::json!({
                "status": value.get("status").cloned().unwrap_or(serde_json::Value::Null),
                "modelId": value.get("modelId").cloned().unwrap_or(serde_json::Value::Null),
                "audioGenerated": value.get("audioGenerated").cloned().unwrap_or(serde_json::Value::Null),
                "sampleRateHz": value.get("frame").and_then(|frame| frame.get("sampleRateHz")).cloned().unwrap_or(serde_json::Value::Null),
                "channels": value.get("frame").and_then(|frame| frame.get("channels")).cloned().unwrap_or(serde_json::Value::Null),
                "diagnosticCount": value.get("diagnostics").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        _ => (
            "TTS operation result",
            "Completed the TTS package surface operation.",
            serde_json::json!({}),
        ),
    };
    structured_surface_response(operation, title, message, summary, value)
}

fn describe_value(input: serde_json::Value) -> serde_json::Value {
    let surface = package_surface();
    serde_json::json!({
        "library": surface.library,
        "version": surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface.operations.iter().map(|operation| operation.id.as_str()).collect::<Vec<_>>(),
        "input": input
    })
}

fn synthesize_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let request = request_from_value(input)?;
    let output = synthesize(&request)?;
    Ok(serde_json::json!({
        "status": status_string(&output.status),
        "provider": output.provider,
        "audioGenerated": output.audio.is_some(),
        "audio": output.audio,
        "nativeDiagnostics": output.native_diagnostics,
        "diagnostics": output.diagnostics,
        "plan": plan_for_request(&request, &output.status),
    }))
}

fn plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let request = request_from_value(input)?;
    request.validate()?;
    Ok(plan_for_request(&request, &planned_status(&request)))
}

fn models_value() -> serde_json::Value {
    let models = tts_model_presets()
        .into_iter()
        .map(tts_model_json)
        .collect::<Vec<_>>();
    serde_json::json!({
        "defaultModelSelected": false,
        "models": models,
        "nativeProvidersImplemented": true,
        "nativeSynthesisAvailable": cfg!(feature = "candle"),
        "featureFlags": feature_flags_json(),
        "message": "No TTS model preset is selected by default. F5 + Vocos native synthesis requires explicit provider selection and local bundles.",
        "requirements": [
            {
                "id": "native-tts-provider",
                "requiredFor": ["audio.tts.synthesize"],
                "available": cfg!(feature = "candle")
            }
        ]
    })
}

fn tts_model_presets() -> Vec<ModelPreset> {
    ModelPreset::ALL
        .iter()
        .copied()
        .filter(|preset| {
            matches!(
                preset,
                ModelPreset::F5TtsV1Base
                    | ModelPreset::F5TtsBase
                    | ModelPreset::E2TtsBase
                    | ModelPreset::VocosMel24Khz
            )
        })
        .collect()
}

fn tts_model_json(preset: ModelPreset) -> serde_json::Value {
    let spec = preset.spec();
    serde_json::json!({
        "id": preset.as_str(),
        "name": spec.name.as_str(),
        "displayName": spec.metadata.get("displayName"),
        "task": spec.task.as_protocol_str(),
        "repoId": spec.repo_id_value(),
        "revision": spec.revision_value(),
        "requiredFiles": required_files(&spec),
        "requestedFiles": file_requests_json(&spec.files),
        "license": license_json(&spec),
        "explicitOptIn": spec.metadata.get("explicitOptIn").is_some_and(|value| value == "true"),
        "metadata": spec.metadata,
        "runtime": {
            "downloadsModels": false,
            "runsInference": false,
            "sideEffects": []
        }
    })
}

fn required_files(spec: &ModelSpec) -> Vec<&str> {
    spec.files
        .iter()
        .filter_map(|request| match request {
            ModelFileRequest::Required(path) => Some(path.as_str()),
            ModelFileRequest::Optional(_) | ModelFileRequest::FirstAvailable(_) => None,
        })
        .collect()
}

fn file_requests_json(files: &[ModelFileRequest]) -> Vec<serde_json::Value> {
    files
        .iter()
        .map(|request| match request {
            ModelFileRequest::Required(path) => {
                serde_json::json!({"kind": "required", "path": path})
            }
            ModelFileRequest::Optional(path) => {
                serde_json::json!({"kind": "optional", "path": path})
            }
            ModelFileRequest::FirstAvailable(paths) => {
                serde_json::json!({"kind": "firstAvailable", "paths": paths})
            }
        })
        .collect()
}

fn license_json(spec: &ModelSpec) -> serde_json::Value {
    serde_json::json!({
        "id": spec.metadata.get("license"),
        "name": spec.metadata.get("licenseName"),
        "url": spec.metadata.get("licenseUrl"),
        "scope": spec.metadata.get("licenseScope"),
    })
}

fn reference_prompt_plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let prompt = input
        .get("referenceVoicePrompt")
        .cloned()
        .or_else(|| input.get("referencePrompt").cloned());
    match prompt {
        Some(prompt) => {
            let prompt: ReferenceVoicePrompt = serde_json::from_value(prompt)
                .map_err(|error| format!("invalid request: referenceVoicePrompt {error}"))?;
            prompt.validate_source_and_hints()?;
            Ok(reference_prompt_plan(Some(&prompt)))
        }
        None => Ok(reference_prompt_plan(None)),
    }
}

fn f5_mel_debug_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let request: NativeF5MelDiagnosticRequest =
        serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))?;
    serde_json::to_value(run_f5_mel_diagnostic(&request)?)
        .map_err(|error| format!("failed to encode F5 mel diagnostic: {error}"))
}

fn vocos_vocoder_debug_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let request: NativeVocosVocoderDiagnosticRequest =
        serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))?;
    serde_json::to_value(run_vocos_vocoder_diagnostic(&request)?)
        .map_err(|error| format!("failed to encode Vocos vocoder diagnostic: {error}"))
}

fn request_from_value(input: serde_json::Value) -> Result<SpeechSynthesisRequest, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

fn plan_for_request(
    request: &SpeechSynthesisRequest,
    status: &SpeechSynthesisStatus,
) -> serde_json::Value {
    serde_json::json!({
        "status": status_string(status),
        "willSynthesize": can_attempt_native_synthesis(request),
        "speakerConditioned": request.is_speaker_conditioned(),
        "provider": request.provider,
        "device": device_plan(request.provider.device),
        "modelBundle": model_bundle_plan(request),
        "vocoder": vocoder_plan(request),
        "requestedOutput": {
            "sampleRateHz": request.options.sample_rate_hz,
            "channels": request.options.channels,
            "format": "pcm-f32-interleaved"
        },
        "textChunking": speech_synthesis_text_chunking_report(&request.text),
        "runtime": {
            "nativeProvidersImplemented": true,
            "nativeSynthesisAvailable": cfg!(feature = "candle"),
            "downloadsModels": false,
            "runsInference": can_attempt_native_synthesis(request),
            "sideEffects": []
        },
        "featureFlags": feature_flags_json(),
        "referencePrompt": reference_prompt_plan(request.reference_voice_prompt.as_ref()),
        "requirements": [
            {
                "id": "native-tts-provider",
                "available": cfg!(feature = "candle"),
                "message": "Native F5 + Vocos synthesis requires explicit local bundle paths and the candle feature."
            }
        ]
    })
}

fn can_attempt_native_synthesis(request: &SpeechSynthesisRequest) -> bool {
    let reference_prompt_ready = request
        .reference_voice_prompt
        .as_ref()
        .is_some_and(ReferenceVoicePrompt::has_transcript);
    let vocoder_ready = request.provider.vocoder.as_ref().is_some_and(|vocoder| {
        vocoder.provider_id == "vocos" && vocoder.model_bundle.bundle_path.is_some()
    });
    request.provider.native
        && request.provider.provider_id == "f5"
        && cfg!(feature = "candle")
        && reference_prompt_ready
        && request.provider.model_bundle.bundle_path.is_some()
        && vocoder_ready
}

fn planned_status(request: &SpeechSynthesisRequest) -> SpeechSynthesisStatus {
    if request.provider.native || request_reference_prompt_asr_unavailable(request) {
        SpeechSynthesisStatus::SetupRequired
    } else {
        SpeechSynthesisStatus::UnsupportedRuntime
    }
}

fn request_reference_prompt_asr_unavailable(request: &SpeechSynthesisRequest) -> bool {
    request
        .reference_voice_prompt
        .as_ref()
        .is_some_and(|prompt| {
            !prompt.has_transcript() && prompt.asr_fallback.is_some() && !cfg!(feature = "asr")
        })
}

fn device_plan(preference: NativeTtsDevicePreference) -> serde_json::Value {
    let (selection, auto_behavior, message) = match preference {
        NativeTtsDevicePreference::Auto => (
            if cfg!(feature = "cuda") {
                "cuda-if-available-else-cpu"
            } else {
                "cpu-without-cuda-feature"
            },
            "cudaPreferredWhenAvailable",
            "Auto is CUDA-preferred when this crate is built with the cuda feature and a CUDA device is available; otherwise CPU is used.",
        ),
        NativeTtsDevicePreference::Cpu => (
            "cpu",
            "notApplicable",
            "CPU was explicitly requested.",
        ),
        NativeTtsDevicePreference::Cuda => (
            "cuda",
            "notApplicable",
            "CUDA was explicitly requested and requires the cuda feature plus an available CUDA device in later native providers.",
        ),
    };

    serde_json::json!({
        "preference": preference.as_str(),
        "selection": selection,
        "cudaFeatureEnabled": cfg!(feature = "cuda"),
        "autoBehavior": auto_behavior,
        "willProbeHardware": false,
        "message": message
    })
}

fn model_bundle_plan(request: &SpeechSynthesisRequest) -> serde_json::Value {
    let model_id = request.provider.model_id.as_deref();
    let bundle = &request.provider.model_bundle;
    let preset = model_id.and_then(model_preset_by_id);
    let resolution = bundle_resolution(model_id, preset, bundle);
    let download_allowed =
        cfg!(feature = "model-bundles") && bundle.auto_download && !bundle.cache_only;
    let required_files = preset.map(|preset| {
        let spec = preset.spec();
        required_files(&spec)
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    });

    serde_json::json!({
        "modelId": model_id,
        "modelKnown": preset.is_some(),
        "bundlePath": bundle.bundle_path,
        "resolution": resolution,
        "requiredFiles": required_files,
        "modelBundlesFeatureEnabled": cfg!(feature = "model-bundles"),
        "autoDownloadRequested": bundle.auto_download,
        "cacheOnly": bundle.cache_only,
        "downloadAllowed": download_allowed,
        "downloadPolicy": download_policy(bundle),
        "willResolveBundle": false,
        "willDownload": false,
        "message": model_bundle_message(model_id, bundle, download_allowed)
    })
}

fn vocoder_plan(request: &SpeechSynthesisRequest) -> serde_json::Value {
    let vocoder = request.provider.vocoder.clone().unwrap_or_default();
    let model_id = Some(vocoder.model_id.as_str());
    let preset = model_id.and_then(model_preset_by_id);
    let required_files = preset.map(|preset| {
        let spec = preset.spec();
        required_files(&spec)
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    serde_json::json!({
        "providerId": vocoder.provider_id,
        "modelId": vocoder.model_id,
        "modelKnown": preset.is_some(),
        "bundlePath": vocoder.model_bundle.bundle_path,
        "resolution": bundle_resolution(model_id, preset, &vocoder.model_bundle),
        "requiredFiles": required_files,
        "modelBundlesFeatureEnabled": cfg!(feature = "model-bundles"),
        "autoDownloadRequested": vocoder.model_bundle.auto_download,
        "cacheOnly": vocoder.model_bundle.cache_only,
        "downloadAllowed": cfg!(feature = "model-bundles")
            && vocoder.model_bundle.auto_download
            && !vocoder.model_bundle.cache_only,
        "downloadPolicy": download_policy(&vocoder.model_bundle),
        "willResolveBundle": false,
        "willDownload": false
    })
}

fn model_preset_by_id(id: &str) -> Option<ModelPreset> {
    tts_model_presets()
        .into_iter()
        .find(|preset| preset.as_str() == id)
}

fn bundle_resolution(
    model_id: Option<&str>,
    preset: Option<ModelPreset>,
    bundle: &TtsModelBundleSelection,
) -> &'static str {
    if bundle.bundle_path.is_some() {
        "explicitBundlePath"
    } else if preset.is_some() && cfg!(feature = "model-bundles") {
        "modelRuntimePreset"
    } else if model_id.is_some() {
        "requiresModelBundlesFeatureOrExplicitBundle"
    } else {
        "notRequested"
    }
}

fn download_policy(bundle: &TtsModelBundleSelection) -> &'static str {
    if bundle.cache_only {
        "cacheOnly"
    } else if bundle.auto_download && cfg!(feature = "model-bundles") {
        "autoDownloadAllowedByModelBundlesFeature"
    } else if bundle.auto_download {
        "autoDownloadRequiresModelBundlesFeature"
    } else {
        "manualBundleOnly"
    }
}

fn model_bundle_message(
    model_id: Option<&str>,
    bundle: &TtsModelBundleSelection,
    download_allowed: bool,
) -> &'static str {
    if bundle.cache_only && bundle.auto_download {
        "Cache-only mode forbids downloads even though autoDownload was requested."
    } else if download_allowed {
        "A later native provider may download missing files because model-bundles and autoDownload are enabled."
    } else if bundle.auto_download {
        "autoDownload was requested, but downloads require the model-bundles feature."
    } else if bundle.bundle_path.is_some() {
        "Planning records the explicit bundle path without checking the filesystem."
    } else if model_id.is_some() {
        "Planning records the model preset requirement without resolving or downloading files."
    } else {
        "No native model bundle was requested."
    }
}

fn feature_flags_json() -> Vec<serde_json::Value> {
    vec![
        feature_flag(
            "candle",
            cfg!(feature = "candle"),
            "Enables native Candle tensor/model execution for F5 + Vocos TTS providers.",
        ),
        feature_flag(
            "cuda",
            cfg!(feature = "cuda"),
            "Enables CUDA device planning and later native CUDA execution.",
        ),
        feature_flag(
            "model-bundles",
            cfg!(feature = "model-bundles"),
            "Enables explicit model bundle functionality, including optional auto-download planning.",
        ),
        feature_flag(
            "audio-io",
            cfg!(feature = "audio-io"),
            "Reserved for native reference-audio IO integration in later slices.",
        ),
        feature_flag(
            "asr",
            cfg!(feature = "asr"),
            "Reserved for reference prompt transcript fallback planning in later slices.",
        ),
        feature_flag(
            "external-tests",
            cfg!(feature = "external-tests"),
            "Enables opt-in external/native smoke coverage.",
        ),
    ]
}

fn feature_flag(name: &str, enabled: bool, purpose: &str) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "enabled": enabled,
        "purpose": purpose
    })
}

fn reference_prompt_plan(prompt: Option<&ReferenceVoicePrompt>) -> serde_json::Value {
    match prompt {
        Some(prompt) => {
            serde_json::json!({
                "provided": true,
                "transcriptProvided": prompt.has_transcript(),
                "languageHint": prompt.language,
                "source": reference_audio_source_json(&prompt.audio),
                "action": if prompt.has_transcript() {
                    "readyForProviderValidation"
                } else if prompt.asr_fallback.is_some() {
                    "planAsrFallback"
                } else {
                    "needsTranscriptOrAsrFallback"
                },
                "asrFallback": asr_fallback_plan(prompt),
                "message": if prompt.has_transcript() {
                    "Reference Voice Prompt includes audio and transcript."
                } else if prompt.asr_fallback.is_some() {
                    "Reference Voice Prompt is missing a transcript and will require ASR fallback setup before provider validation."
                } else {
                    "Reference Voice Prompt includes audio but no transcript; configure referenceVoicePrompt.asrFallback to plan ASR setup."
                }
            })
        }
        None => serde_json::json!({
            "provided": false,
            "transcriptProvided": false,
            "action": "notRequiredForGenericTts",
            "message": "No Reference Voice Prompt was supplied."
        }),
    }
}

fn asr_fallback_plan(prompt: &ReferenceVoicePrompt) -> serde_json::Value {
    let Some(fallback) = &prompt.asr_fallback else {
        return serde_json::json!({
            "configured": false,
            "available": false,
            "asrFeatureEnabled": cfg!(feature = "asr")
        });
    };
    let language_hint = fallback.language.as_deref().or(prompt.language.as_deref());

    #[cfg(feature = "asr")]
    {
        let source = prompt.audio.to_transcription_source();
        let provider_plan = audio_analysis_transcription::transcription_provider_plans()
            .into_iter()
            .find(|plan| plan.provider_id == fallback.provider_id);
        let provider_known = provider_plan.is_some();

        serde_json::json!({
            "configured": true,
            "available": provider_known,
            "asrFeatureEnabled": true,
            "providerKnown": provider_known,
            "providerId": fallback.provider_id,
            "modelId": fallback.model_id,
            "languageHint": language_hint,
            "sourceKind": transcription_source_kind(&source),
            "willRunAsr": false,
            "transcriptionProviderPlan": provider_plan,
            "message": if provider_known {
                "ASR fallback is planned through audio-analysis-transcription; this operation does not run transcription."
            } else {
                "ASR fallback provider is not known to audio-analysis-transcription."
            }
        })
    }

    #[cfg(not(feature = "asr"))]
    {
        serde_json::json!({
            "configured": true,
            "available": false,
            "asrFeatureEnabled": false,
            "providerKnown": serde_json::Value::Null,
            "providerId": fallback.provider_id,
            "modelId": fallback.model_id,
            "languageHint": language_hint,
            "sourceKind": prompt.audio.kind(),
            "willRunAsr": false,
            "setup": [
                "Build audio-generation-tts with the `asr` feature to plan fallback through audio-analysis-transcription."
            ],
            "message": "ASR fallback is configured but unavailable in this build."
        })
    }
}

#[cfg(feature = "asr")]
fn transcription_source_kind(
    source: &audio_analysis_transcription::TranscriptionSource,
) -> &'static str {
    match source {
        audio_analysis_transcription::TranscriptionSource::Samples { .. } => "samples",
        audio_analysis_transcription::TranscriptionSource::Path { .. } => "path",
    }
}

fn reference_audio_source_json(audio: &ReferenceVoicePromptAudio) -> serde_json::Value {
    match audio {
        ReferenceVoicePromptAudio::Samples(audio) => serde_json::json!({
            "kind": "samples",
            "sampleRateHz": audio.sample_rate_hz,
            "channels": audio.channels,
            "sampleCount": audio.samples.len()
        }),
        ReferenceVoicePromptAudio::Path { path } => serde_json::json!({
            "kind": "path",
            "path": path
        }),
    }
}

fn status_string(status: &SpeechSynthesisStatus) -> &'static str {
    match status {
        SpeechSynthesisStatus::Ready => "ready",
        SpeechSynthesisStatus::SetupRequired => "setupRequired",
        SpeechSynthesisStatus::UnsupportedRuntime => "unsupportedRuntime",
    }
}

fn example_synthesis_request() -> serde_json::Value {
    serde_json::json!({
        "text": "Hello from the TTS package surface.",
        "provider": {
            "providerId": "generic",
            "native": false
        },
        "options": {
            "sampleRateHz": 24000,
            "channels": 1,
            "seed": 42,
            "speed": 1.0,
            "removeSilence": false
        }
    })
}

fn example_reference_prompt() -> serde_json::Value {
    serde_json::json!({
        "audio": example_pcm_audio(),
        "transcript": "Reference voice prompt text.",
        "language": "en",
        "metadata": {
            "source": "inline-example"
        }
    })
}

fn example_f5_debug_request() -> serde_json::Value {
    serde_json::json!({
        "text": "Run an F5 mel diagnostic.",
        "modelId": "f5-tts-v1-base",
        "bundlePath": ".model-runtime/f5-tts-v1-base/main",
        "device": "cpu",
        "options": {
            "maxDurationSeconds": 0.25,
            "steps": 1,
            "cfgStrength": 1.0,
            "speed": 1.0
        }
    })
}

fn example_vocos_debug_request() -> serde_json::Value {
    serde_json::json!({
        "modelId": "vocos-mel-24khz",
        "bundlePath": ".model-runtime/vocos-mel-24khz/main",
        "device": "cpu",
        "mel": {
            "frames": 4,
            "channels": 100
        },
        "options": {
            "maxDurationSeconds": 0.05
        }
    })
}

fn example_pcm_audio() -> PcmAudio {
    PcmAudio {
        sample_rate_hz: 24_000,
        channels: 1,
        samples: vec![0.0, 0.01, -0.01, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_core::SurfaceOperationRole;

    #[test]
    fn package_surface_exposes_tts_operations_with_roles() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "describe",
                "audio.tts.synthesize",
                "audio.tts.plan",
                "audio.tts.models",
                "audio.tts.referencePromptPlan",
                "audio.tts.debug.f5Mel",
                "audio.tts.debug.vocosVocoder"
            ]
        );
        let synthesize = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() == "audio.tts.synthesize")
            .expect("synthesize");
        assert_eq!(synthesize.curation.role, SurfaceOperationRole::Workflow);
        assert!(synthesize.curation.primary);
        let plan = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() == "audio.tts.plan")
            .expect("plan");
        assert_eq!(plan.curation.role, SurfaceOperationRole::Debug);
        let f5_debug = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() == "audio.tts.debug.f5Mel")
            .expect("f5 mel debug");
        assert_eq!(f5_debug.curation.role, SurfaceOperationRole::Debug);
        let vocos_debug = surface
            .operations
            .iter()
            .find(|operation| operation.id.as_str() == "audio.tts.debug.vocosVocoder")
            .expect("vocos vocoder debug");
        assert_eq!(vocos_debug.curation.role, SurfaceOperationRole::Debug);
    }

    #[test]
    fn synthesize_surface_returns_setup_diagnostics() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.synthesize".into(),
            input: example_synthesis_request(),
        })
        .expect("synthesize response");
        assert_eq!(response.value["result"]["status"], "unsupportedRuntime");
        assert_eq!(response.value["result"]["audioGenerated"], false);
        assert!(response.value["result"]["diagnostics"].is_array());
    }

    #[test]
    fn reference_prompt_plan_accepts_transcript_present_path_source() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.referencePromptPlan".into(),
            input: serde_json::json!({
                "referenceVoicePrompt": {
                    "audio": {"path": "fixtures/reference.wav"},
                    "transcript": "The reference speaker reads this sentence.",
                    "language": "en"
                }
            }),
        })
        .expect("reference prompt plan");

        let result = &response.value["result"];
        assert_eq!(result["provided"], true);
        assert_eq!(result["source"]["kind"], "path");
        assert_eq!(result["source"]["path"], "fixtures/reference.wav");
        assert_eq!(result["transcriptProvided"], true);
        assert_eq!(result["languageHint"], "en");
        assert_eq!(result["action"], "readyForProviderValidation");
    }

    #[test]
    fn reference_prompt_plan_reports_missing_transcript() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.referencePromptPlan".into(),
            input: serde_json::json!({
                "referenceVoicePrompt": {
                    "audio": example_pcm_audio()
                }
            }),
        })
        .expect("reference prompt plan");
        assert_eq!(
            response.value["result"]["action"],
            "needsTranscriptOrAsrFallback"
        );
    }

    #[test]
    fn reference_prompt_plan_reports_configured_asr_fallback_unavailable_by_default() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.referencePromptPlan".into(),
            input: serde_json::json!({
                "referenceVoicePrompt": {
                    "audio": example_pcm_audio(),
                    "language": "en",
                    "asrFallback": {
                        "providerId": "candle-whisper",
                        "modelId": "openai/whisper-large-v3-turbo"
                    }
                }
            }),
        })
        .expect("reference prompt plan");

        let result = &response.value["result"];
        assert_eq!(result["transcriptProvided"], false);
        assert_eq!(result["action"], "planAsrFallback");
        assert_eq!(result["asrFallback"]["configured"], true);
        assert_eq!(result["asrFallback"]["available"], false);
        assert_eq!(result["asrFallback"]["providerId"], "candle-whisper");
        assert_eq!(result["asrFallback"]["languageHint"], "en");
    }

    #[test]
    fn plan_surface_reports_setup_required_for_unavailable_asr_fallback() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.plan".into(),
            input: serde_json::json!({
                "text": "Plan speaker-conditioned TTS.",
                "provider": {"providerId": "speaker-conditioned"},
                "referenceVoicePrompt": {
                    "audio": example_pcm_audio(),
                    "language": "en",
                    "asrFallback": {
                        "providerId": "candle-whisper"
                    }
                }
            }),
        })
        .expect("plan response");

        let result = &response.value["result"];
        assert_eq!(result["status"], "setupRequired");
        assert_eq!(result["referencePrompt"]["action"], "planAsrFallback");
        assert_eq!(result["referencePrompt"]["asrFallback"]["available"], false);
    }

    #[cfg(feature = "asr")]
    #[test]
    fn reference_prompt_plan_uses_transcription_provider_plan_when_asr_feature_enabled() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.referencePromptPlan".into(),
            input: serde_json::json!({
                "referenceVoicePrompt": {
                    "audio": {"path": "fixtures/reference.wav"},
                    "language": "en",
                    "asrFallback": {
                        "providerId": "candle-whisper",
                        "modelId": "openai/whisper-large-v3-turbo"
                    }
                }
            }),
        })
        .expect("reference prompt plan");

        let fallback = &response.value["result"]["asrFallback"];
        assert_eq!(fallback["configured"], true);
        assert_eq!(fallback["available"], true);
        assert_eq!(fallback["asrFeatureEnabled"], true);
        assert_eq!(fallback["sourceKind"], "path");
        assert_eq!(
            fallback["transcriptionProviderPlan"]["providerId"],
            "candle-whisper"
        );
    }

    #[test]
    fn plan_surface_explains_native_bundle_and_device_choice_without_side_effects() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.plan".into(),
            input: serde_json::json!({
                "text": "Plan native TTS.",
                "provider": {
                    "providerId": "f5",
                    "modelId": "f5-tts-v1-base",
                    "native": true,
                    "device": "auto",
                    "modelBundle": {
                        "autoDownload": true,
                        "cacheOnly": true
                    }
                }
            }),
        })
        .expect("plan response");

        let result = &response.value["result"];
        assert_eq!(result["willSynthesize"], false);
        assert_eq!(result["runtime"]["runsInference"], false);
        assert_eq!(result["runtime"]["downloadsModels"], false);
        assert_eq!(result["device"]["preference"], "auto");
        assert_eq!(
            result["device"]["autoBehavior"],
            "cudaPreferredWhenAvailable"
        );
        assert_eq!(result["modelBundle"]["modelId"], "f5-tts-v1-base");
        assert_eq!(result["modelBundle"]["cacheOnly"], true);
        assert_eq!(result["modelBundle"]["autoDownloadRequested"], true);
        assert_eq!(result["modelBundle"]["downloadAllowed"], false);
    }

    #[test]
    fn plan_and_synthesize_report_long_text_chunking_without_audio_chunks() {
        let long_text = [
            "The first sentence establishes the speaker-conditioned synthesis request.",
            "The second sentence is intentionally long enough to force chunk planning at the package surface boundary.",
            "The final sentence verifies that chunking metadata stays separate from the core in-memory PCM output contract.",
        ]
        .join(" ");

        let plan = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.plan".into(),
            input: serde_json::json!({"text": long_text}),
        })
        .expect("plan response");
        let chunking = &plan.value["result"]["textChunking"];

        assert_eq!(chunking["strategy"], "sentence-boundary");
        assert!(chunking["chunkCount"].as_u64().expect("chunk count") > 1);
        assert_eq!(chunking["preservesInMemoryOutputContract"], true);
        assert!(chunking["chunks"].as_array().expect("chunks").len() > 1);

        let synthesis = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.synthesize".into(),
            input: serde_json::json!({"text": long_text}),
        })
        .expect("synthesis response");
        let result = &synthesis.value["result"];

        assert_eq!(result["audioGenerated"], false);
        assert!(result.get("audioChunks").is_none());
        assert_eq!(
            result["plan"]["textChunking"]["preservesInMemoryOutputContract"],
            true
        );
    }

    #[test]
    fn models_surface_lists_explicit_tts_presets_with_license_metadata() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.models".into(),
            input: serde_json::json!({}),
        })
        .expect("models response");

        assert_eq!(response.value["result"]["defaultModelSelected"], false);
        let models = response.value["result"]["models"]
            .as_array()
            .expect("models array");
        let ids = models
            .iter()
            .map(|model| model["id"].as_str().expect("model id"))
            .collect::<Vec<_>>();
        assert!(ids.contains(&"f5-tts-v1-base"));
        assert!(ids.contains(&"f5-tts-base"));
        assert!(ids.contains(&"e2-tts-base"));
        assert!(ids.contains(&"vocos-mel-24khz"));

        let f5 = models
            .iter()
            .find(|model| model["id"] == "f5-tts-v1-base")
            .expect("f5 v1 preset");
        assert_eq!(f5["repoId"], "SWivid/F5-TTS");
        assert_eq!(f5["task"], "speaker_conditioned_tts");
        assert_eq!(f5["license"]["id"], "cc-by-nc-4.0");
        assert_eq!(f5["explicitOptIn"], true);
        assert!(f5["requiredFiles"]
            .as_array()
            .expect("required files")
            .contains(&serde_json::json!(
                "F5TTS_v1_Base/model_1250000.safetensors"
            )));

        let e2 = models
            .iter()
            .find(|model| model["id"] == "e2-tts-base")
            .expect("e2 preset");
        assert_eq!(e2["repoId"], "SWivid/E2-TTS");
        assert_eq!(e2["license"]["id"], "cc-by-nc-4.0");

        let vocos = models
            .iter()
            .find(|model| model["id"] == "vocos-mel-24khz")
            .expect("vocos preset");
        assert_eq!(vocos["repoId"], "charactr/vocos-mel-24khz");
        assert_eq!(vocos["license"]["id"], "mit");
    }

    #[test]
    fn models_surface_reports_native_tts_feature_flags() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.models".into(),
            input: serde_json::json!({}),
        })
        .expect("models response");

        let feature_flags = response.value["result"]["featureFlags"]
            .as_array()
            .expect("feature flags");
        let names = feature_flags
            .iter()
            .map(|feature| feature["name"].as_str().expect("feature name"))
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "candle",
                "cuda",
                "model-bundles",
                "audio-io",
                "asr",
                "external-tests"
            ]
        );
    }

    #[test]
    fn vocos_debug_surface_reports_missing_bundle_setup_error() {
        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.debug.vocosVocoder".into(),
            input: serde_json::json!({
                "modelId": "vocos-mel-24khz",
                "device": "cpu"
            }),
        })
        .expect("vocos vocoder diagnostic");

        let result = &response.value["result"];
        assert_eq!(result["status"], "setupRequired");
        assert_eq!(result["audioGenerated"], false);
        assert_eq!(result["device"]["selected"], "cpu");
        assert!(result["diagnostics"]
            .as_array()
            .expect("diagnostics")
            .iter()
            .any(|diagnostic| diagnostic["code"] == "vocos_bundle_missing"));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn f5_debug_surface_returns_mel_diagnostic_for_local_bundle() {
        let temp = tempfile::tempdir().expect("tempdir");
        let bundle = temp.path();
        crate::native_f5::test_support::write_test_f5_bundle(
            bundle,
            serde_json::json!({
                "model_type": "f5-tts",
                "architectures": ["F5TTS"],
                "n_mel_channels": 4,
                "sample_rate": 24000,
                "hop_length": 256
            }),
        );

        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.debug.f5Mel".into(),
            input: serde_json::json!({
                "text": "diagnose f5",
                "bundlePath": bundle,
                "modelId": "f5-tts-v1-base",
                "device": "cpu",
                "options": {
                    "maxDurationSeconds": 0.05
                }
            }),
        })
        .expect("f5 mel diagnostic");

        let result = &response.value["result"];
        assert_eq!(result["status"], "ready");
        assert_eq!(result["vocoderRequired"], true);
        assert_eq!(result["audioGenerated"], false);
        assert_eq!(result["device"]["selected"], "cpu");
        assert_eq!(result["bundle"]["vocabEntries"], 2);
        assert_eq!(result["bundle"]["tensorCount"], 1);
        assert_eq!(result["mel"]["channels"], 4);
        assert!(result["diagnostics"]
            .as_array()
            .expect("diagnostics")
            .is_empty());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn synthesize_surface_runs_f5_and_vocos_for_local_native_bundles() {
        let temp = tempfile::tempdir().expect("tempdir");
        let f5_bundle = temp.path().join("f5");
        let vocos_bundle = temp.path().join("vocos");
        crate::native_f5::test_support::write_test_f5_bundle(
            &f5_bundle,
            serde_json::json!({
                "model_type": "f5-tts",
                "architectures": ["F5TTS"],
                "n_mel_channels": 4,
                "sample_rate": 24000,
                "hop_length": 256
            }),
        );
        crate::native_vocos::test_support::write_test_vocos_bundle(
            &vocos_bundle,
            &crate::native_vocos::test_support::test_config(4, 24_000, 256),
            &[1, 2, 3, 4],
        );

        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.synthesize".into(),
            input: serde_json::json!({
                "text": "Synthesize with the reference speaker.",
                "referenceVoicePrompt": {
                    "audio": example_pcm_audio(),
                    "transcript": "Reference voice prompt text.",
                    "language": "en"
                },
                "provider": {
                    "providerId": "f5",
                    "modelId": "f5-tts-v1-base",
                    "native": true,
                    "device": "cpu",
                    "modelBundle": {
                        "bundlePath": f5_bundle
                    },
                    "vocoder": {
                        "providerId": "vocos",
                        "modelId": "vocos-mel-24khz",
                        "modelBundle": {
                            "bundlePath": vocos_bundle
                        }
                    }
                },
                "options": {
                    "seed": 41,
                    "steps": 3,
                    "cfgStrength": 1.25,
                    "speed": 1.5,
                    "maxDurationSeconds": 0.02,
                    "removeSilence": true
                }
            }),
        })
        .expect("native synthesize");

        let result = &response.value["result"];
        assert_eq!(result["status"], "ready");
        assert_eq!(result["audioGenerated"], true);
        assert_eq!(result["audio"]["sampleRateHz"], 24_000);
        assert!(
            result["audio"]["samples"]
                .as_array()
                .expect("samples")
                .len()
                > 0
        );
        assert_eq!(result["nativeDiagnostics"]["provider"], "f5");
        assert_eq!(result["nativeDiagnostics"]["modelId"], "f5-tts-v1-base");
        assert_eq!(result["nativeDiagnostics"]["vocoder"], "vocos");
        assert_eq!(
            result["nativeDiagnostics"]["vocoderModelId"],
            "vocos-mel-24khz"
        );
        assert_eq!(result["nativeDiagnostics"]["runtime"], "candle");
        assert_eq!(result["nativeDiagnostics"]["device"], "cpu");
        assert_eq!(result["nativeDiagnostics"]["inference"]["seed"], 41);
        assert_eq!(result["nativeDiagnostics"]["inference"]["steps"], 3);
        assert_eq!(
            result["nativeDiagnostics"]["inference"]["cfgStrength"],
            1.25
        );
        assert_eq!(result["nativeDiagnostics"]["inference"]["speed"], 1.5);
        let max_duration = result["nativeDiagnostics"]["inference"]["maxDurationSeconds"]
            .as_f64()
            .expect("max duration");
        assert!((max_duration - 0.02).abs() < 1.0e-6);
        assert_eq!(
            result["nativeDiagnostics"]["inference"]["removeSilence"],
            true
        );
        assert_eq!(
            result["nativeDiagnostics"]["bundleSource"],
            "explicitBundlePath"
        );
        assert!(result["diagnostics"]
            .as_array()
            .expect("diagnostics")
            .is_empty());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn synthesize_surface_concatenates_long_text_chunks_into_one_pcm_output() {
        let temp = tempfile::tempdir().expect("tempdir");
        let f5_bundle = temp.path().join("f5");
        let vocos_bundle = temp.path().join("vocos");
        crate::native_f5::test_support::write_test_f5_bundle(
            &f5_bundle,
            serde_json::json!({
                "model_type": "f5-tts",
                "architectures": ["F5TTS"],
                "n_mel_channels": 4,
                "sample_rate": 24000,
                "hop_length": 256
            }),
        );
        crate::native_vocos::test_support::write_test_vocos_bundle(
            &vocos_bundle,
            &crate::native_vocos::test_support::test_config(4, 24_000, 256),
            &[1, 2, 3, 4],
        );
        let long_text = [
            "The first sentence establishes the native synthesis request for a reference speaker.",
            "The second sentence is long enough to require chunk planning while using the same local diagnostic bundles.",
            "The third sentence verifies that chunked synthesis still returns one in-memory PCM audio object.",
        ]
        .join(" ");

        let response = run_surface_operation(SurfaceRequest {
            operation: "audio.tts.synthesize".into(),
            input: serde_json::json!({
                "text": long_text,
                "referenceVoicePrompt": {
                    "audio": example_pcm_audio(),
                    "transcript": "Reference voice prompt text.",
                    "language": "en"
                },
                "provider": {
                    "providerId": "f5",
                    "modelId": "f5-tts-v1-base",
                    "native": true,
                    "device": "cpu",
                    "modelBundle": {
                        "bundlePath": f5_bundle
                    },
                    "vocoder": {
                        "providerId": "vocos",
                        "modelId": "vocos-mel-24khz",
                        "modelBundle": {
                            "bundlePath": vocos_bundle
                        }
                    }
                },
                "options": {
                    "maxDurationSeconds": 0.02,
                    "steps": 1
                }
            }),
        })
        .expect("native synthesize");
        let result = &response.value["result"];
        let chunk_count = result["nativeDiagnostics"]["textChunking"]["chunkCount"]
            .as_u64()
            .expect("chunk count");
        let samples = result["audio"]["samples"].as_array().expect("samples");

        assert_eq!(result["status"], "ready");
        assert_eq!(result["audioGenerated"], true);
        assert!(chunk_count > 1);
        assert!(samples.len() > chunk_count as usize);
        assert!(result.get("audioChunks").is_none());
        assert_eq!(
            result["nativeDiagnostics"]["textChunking"]["preservesInMemoryOutputContract"],
            true
        );
    }
}
