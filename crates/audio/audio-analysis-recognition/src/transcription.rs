//! Generic transcription contracts and deterministic providers.
#![allow(deprecated)]

use audio_contracts::{DetectError, Result};
use model_runtime::{FallbackPolicy, ModelRuntimeBackend, ModelSpec, ModelTask};
use serde::{Deserialize, Serialize};
use text_transcripts::{TranscriptSegmentContract, TranscriptionContract};

use crate::{AudioRuntime, AudioRuntimeSelection, SpeechRecognitionRequest};

/// Input accepted by generic audio transcription workflows.
#[deprecated(
    note = "use audio-analysis-transcription for real ASR or text-transcripts for imported normalization"
)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum TranscriptionInput {
    /// Caller-supplied transcript segments.
    #[serde(alias = "imported_segments")]
    ImportedSegments {
        /// Transcript segments to normalize.
        segments: Vec<TranscriptSegmentContract>,
    },
    /// Caller-supplied full transcript contract.
    #[serde(alias = "imported_contract")]
    ImportedContract {
        /// Transcript contract to normalize.
        transcript: TranscriptionContract,
    },
    /// Local source path for a non-default native or external provider.
    #[serde(alias = "source_path")]
    SourcePath {
        /// Audio source path.
        path: String,
    },
}

/// Runtime selection for generic audio transcription.
#[deprecated(note = "use audio-analysis-transcription provider options for real ASR")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionRuntimeSelection {
    /// Optional preset or compatibility model identifier.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Optional generic model spec.
    #[serde(default)]
    pub model: Option<ModelSpec>,
    /// Optional preferred model runtime backend.
    #[serde(default)]
    pub backend: Option<ModelRuntimeBackend>,
    /// Fallback behavior for unsupported native paths.
    #[serde(default)]
    pub fallback_policy: FallbackPolicy,
}

/// Generic request for audio transcription.
#[deprecated(note = "use audio-analysis-transcription::TranscriptionPipelineRequest for real ASR")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionRequest {
    /// Optional source identifier.
    #[serde(default)]
    pub source: Option<String>,
    /// Optional language hint.
    #[serde(default)]
    pub language: Option<String>,
    /// Transcription input.
    pub input: TranscriptionInput,
    /// Runtime selection.
    #[serde(default)]
    pub runtime: TranscriptionRuntimeSelection,
}

/// Generic response for audio transcription.
#[deprecated(note = "use audio-analysis-transcription::TranscriptionPipelineResponse for real ASR")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Selected model id.
    pub model_id: String,
    /// Runtime backend used.
    pub backend: ModelRuntimeBackend,
    /// Transcript contract returned by the provider.
    pub transcript: TranscriptionContract,
    /// Provider diagnostics.
    pub diagnostics: Vec<String>,
}

/// Transcription provider families.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionProviderKind {
    /// Caller-supplied transcript provider.
    Imported,
    /// whisper.cpp provider owned by text-transcripts.
    WhisperCpp,
    /// External command provider.
    ExternalCommand,
    /// Generic model-runtime provider.
    ModelRuntime,
    /// Caller-defined provider.
    Custom(String),
}

/// Metadata-only transcription backend plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionBackendPlan {
    /// Selected model id.
    pub selected_model_id: String,
    /// Provider family.
    pub provider: TranscriptionProviderKind,
    /// Runtime backend.
    pub backend: ModelRuntimeBackend,
    /// Model task.
    pub task: ModelTask,
    /// Whether this plan executes in the default audio surface.
    pub will_execute: bool,
    /// Optional required feature.
    pub requires_feature: Option<String>,
    /// Crate or caller responsible for execution.
    pub execution_owner: String,
    /// Setup notes or commands.
    pub setup: Vec<String>,
    /// Plan diagnostics.
    pub diagnostics: Vec<String>,
}

/// Compatibility details for the existing whisper.cpp native path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WhisperCppTranscriptionPlan {
    /// Selected Whisper model id.
    pub model_id: String,
    /// Native runtime name.
    pub native_runtime: String,
    /// Crate that owns execution.
    pub execution_owner: String,
    /// Required feature in the execution-owning crate.
    pub required_feature: String,
    /// Whether this plan executes in the default audio surface.
    pub will_execute: bool,
}

impl WhisperCppTranscriptionPlan {
    /// Creates a whisper.cpp compatibility plan.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            native_runtime: "whisper.cpp".to_string(),
            execution_owner: "moenarch-text-transcripts".to_string(),
            required_feature: "native".to_string(),
            will_execute: false,
        }
    }
}

/// Trait for audio transcription provider implementations.
pub trait AudioTranscriptionProvider {
    /// Provider identifier.
    fn provider_id(&self) -> &str;

    /// Runtime backend used by this provider.
    fn backend(&self) -> ModelRuntimeBackend;

    /// Transcribes audio or normalizes imported transcript input.
    fn transcribe(&mut self, request: TranscriptionRequest) -> Result<TranscriptionResponse>;
}

/// Provider for caller-supplied transcript data.
#[derive(Debug, Clone, Default)]
pub struct ImportedTranscriptionProvider;

impl AudioTranscriptionProvider for ImportedTranscriptionProvider {
    fn provider_id(&self) -> &str {
        "imported-transcript"
    }

    fn backend(&self) -> ModelRuntimeBackend {
        ModelRuntimeBackend::Imported
    }

    fn transcribe(&mut self, request: TranscriptionRequest) -> Result<TranscriptionResponse> {
        let model_id = selected_transcription_model_id(&request.runtime);
        let transcript = match request.input {
            TranscriptionInput::ImportedSegments { segments } => {
                text_transcripts::normalize_imported_segments(
                    request.source,
                    request.language,
                    segments,
                )
                .map_err(|error| DetectError::InvalidArgument(error.to_string()))?
            }
            TranscriptionInput::ImportedContract { mut transcript } => {
                if transcript.source.is_none() {
                    transcript.source = request.source;
                }
                if transcript.language.is_none() {
                    transcript.language = request.language;
                }
                text_transcripts::normalize_transcription_contract(transcript)
                    .map_err(|error| DetectError::InvalidArgument(error.to_string()))?
            }
            TranscriptionInput::SourcePath { path } => {
                return Err(DetectError::InvalidArgument(format!(
                    "unsupported_runtime: source-path transcription for `{path}` requires an explicit native, external, or model-runtime provider; default builds only normalize imported transcript data"
                )));
            }
        };
        if transcript.segments.is_empty() {
            return Err(DetectError::InvalidArgument(
                "transcription request must include imported segments, an imported transcript contract, or a runnable provider"
                    .to_string(),
            ));
        }
        transcript
            .validate_strict()
            .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
        Ok(TranscriptionResponse {
            accepted: true,
            operation: "transcribe".to_string(),
            model_id,
            backend: ModelRuntimeBackend::Imported,
            transcript,
            diagnostics: vec![
                "normalized caller-supplied transcript data without native ASR".to_string(),
            ],
        })
    }
}

impl From<AudioRuntimeSelection> for TranscriptionRuntimeSelection {
    fn from(value: AudioRuntimeSelection) -> Self {
        Self {
            model_id: value.model_id,
            model: value.model,
            backend: value.backend.map(audio_runtime_to_model_backend),
            fallback_policy: audio_fallback_to_model_fallback(value.fallback_policy),
        }
    }
}

impl From<SpeechRecognitionRequest> for TranscriptionRequest {
    fn from(value: SpeechRecognitionRequest) -> Self {
        Self {
            source: value.source,
            language: value.language,
            input: TranscriptionInput::ImportedSegments {
                segments: value.imported_segments,
            },
            runtime: value.model.into(),
        }
    }
}

/// Runs a generic transcription request.
#[deprecated(
    note = "use audio-analysis-transcription for real ASR or text_transcripts::normalize_imported_segments for imported transcript normalization"
)]
pub fn transcribe(request: TranscriptionRequest) -> Result<TranscriptionResponse> {
    match &request.input {
        TranscriptionInput::ImportedSegments { .. }
        | TranscriptionInput::ImportedContract { .. } => {
            let mut provider = ImportedTranscriptionProvider;
            provider.transcribe(request)
        }
        TranscriptionInput::SourcePath { .. } => {
            let plan = transcription_plan(&request.runtime);
            Err(DetectError::InvalidArgument(format!(
                "unsupported_runtime: {} cannot execute in the default audio recognition surface; {}",
                plan.selected_model_id,
                plan.diagnostics.join(" ")
            )))
        }
    }
}

/// Builds a metadata-only transcription runtime plan.
pub fn transcription_plan(selection: &TranscriptionRuntimeSelection) -> TranscriptionBackendPlan {
    let selected_model_id = selected_transcription_model_id(selection);
    let backend = selection
        .backend
        .clone()
        .unwrap_or(ModelRuntimeBackend::WhisperCpp);
    let provider = provider_kind_for_backend(&backend);
    match backend {
        ModelRuntimeBackend::Imported => TranscriptionBackendPlan {
            selected_model_id,
            provider,
            backend,
            task: ModelTask::SpeechRecognition,
            will_execute: true,
            requires_feature: None,
            execution_owner: "moenarch-audio-analysis-recognition".to_string(),
            setup: vec!["Pass imported transcript segments or a transcript contract.".to_string()],
            diagnostics: vec![
                "default audio recognition builds normalize imported transcript data in memory"
                    .to_string(),
            ],
        },
        ModelRuntimeBackend::WhisperCpp => TranscriptionBackendPlan {
            selected_model_id,
            provider,
            backend,
            task: ModelTask::SpeechRecognition,
            will_execute: false,
            requires_feature: Some("native".to_string()),
            execution_owner: "moenarch-text-transcripts".to_string(),
            setup: vec![
                "Enable and configure the text-transcripts native whisper.cpp path.".to_string(),
                "Default audio recognition surfaces do not download models or run native ASR."
                    .to_string(),
            ],
            diagnostics: vec![
                "whisper.cpp is one transcription provider, not the audio crate's core abstraction"
                    .to_string(),
                "native Whisper execution remains owned by text-transcripts".to_string(),
            ],
        },
        ModelRuntimeBackend::External => TranscriptionBackendPlan {
            selected_model_id,
            provider,
            backend,
            task: ModelTask::SpeechRecognition,
            will_execute: false,
            requires_feature: None,
            execution_owner: "caller".to_string(),
            setup: vec![
                "Provide an explicit external transcription adapter outside default package-surface execution."
                    .to_string(),
            ],
            diagnostics: vec![
                "external command transcription is planned metadata in default builds".to_string(),
            ],
        },
        _ => TranscriptionBackendPlan {
            selected_model_id,
            provider,
            backend,
            task: ModelTask::SpeechRecognition,
            will_execute: false,
            requires_feature: None,
            execution_owner: "caller".to_string(),
            setup: vec![
                "Provide a concrete model-runtime transcription provider before execution."
                    .to_string(),
            ],
            diagnostics: vec![
                "model-runtime transcription execution is not implemented by the default audio surface"
                    .to_string(),
            ],
        },
    }
}

/// Returns the model-runtime backend that corresponds to an audio runtime.
pub fn audio_runtime_to_model_backend(runtime: AudioRuntime) -> ModelRuntimeBackend {
    match runtime {
        AudioRuntime::Onnx => ModelRuntimeBackend::Onnx,
        AudioRuntime::Candle => ModelRuntimeBackend::Candle,
        AudioRuntime::WhisperCpp => ModelRuntimeBackend::WhisperCpp,
        AudioRuntime::Demucs => ModelRuntimeBackend::Demucs,
        AudioRuntime::External => ModelRuntimeBackend::External,
        AudioRuntime::Spectral => ModelRuntimeBackend::Heuristic,
        AudioRuntime::Heuristic => ModelRuntimeBackend::Heuristic,
        AudioRuntime::Imported => ModelRuntimeBackend::Imported,
    }
}

pub(crate) fn model_backend_to_audio_runtime(backend: &ModelRuntimeBackend) -> AudioRuntime {
    match backend {
        ModelRuntimeBackend::Onnx => AudioRuntime::Onnx,
        ModelRuntimeBackend::Candle => AudioRuntime::Candle,
        ModelRuntimeBackend::WhisperCpp => AudioRuntime::WhisperCpp,
        ModelRuntimeBackend::Demucs => AudioRuntime::Demucs,
        ModelRuntimeBackend::External => AudioRuntime::External,
        ModelRuntimeBackend::Imported => AudioRuntime::Imported,
        _ => AudioRuntime::Heuristic,
    }
}

fn provider_kind_for_backend(backend: &ModelRuntimeBackend) -> TranscriptionProviderKind {
    match backend {
        ModelRuntimeBackend::Imported => TranscriptionProviderKind::Imported,
        ModelRuntimeBackend::WhisperCpp => TranscriptionProviderKind::WhisperCpp,
        ModelRuntimeBackend::External => TranscriptionProviderKind::ExternalCommand,
        ModelRuntimeBackend::Custom(value) => TranscriptionProviderKind::Custom(value.clone()),
        _ => TranscriptionProviderKind::ModelRuntime,
    }
}

fn selected_transcription_model_id(selection: &TranscriptionRuntimeSelection) -> String {
    selection
        .model_id
        .clone()
        .or_else(|| selection.model.as_ref().map(|model| model.name.clone()))
        .unwrap_or_else(|| "whisper-tiny-en".to_string())
}

fn audio_fallback_to_model_fallback(value: crate::FallbackPolicy) -> FallbackPolicy {
    match value {
        crate::FallbackPolicy::Error => FallbackPolicy::Error,
        crate::FallbackPolicy::FastFallback => FallbackPolicy::FastFallback,
        crate::FallbackPolicy::HeuristicFallback => FallbackPolicy::HeuristicFallback,
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn transcribe_accepts_imported_segments_with_generic_request() {
        let response = transcribe(TranscriptionRequest {
            source: Some("clip.wav".to_string()),
            language: Some("en".to_string()),
            input: TranscriptionInput::ImportedSegments {
                segments: vec![TranscriptSegmentContract {
                    index: 0,
                    start_seconds: Some(0.0),
                    end_seconds: Some(1.0),
                    text: " hello ".to_string(),
                    language: None,
                    speaker: None,
                    confidence: Some(2.0),
                    is_final: true,
                    words: Vec::new(),
                    chars: Vec::new(),
                    attributes: BTreeMap::new(),
                }],
            },
            runtime: TranscriptionRuntimeSelection::default(),
        })
        .unwrap();

        assert_eq!(response.backend, ModelRuntimeBackend::Imported);
        assert_eq!(response.transcript.text.as_deref(), Some("hello"));
        assert_eq!(response.transcript.source.as_deref(), Some("clip.wav"));
        assert_eq!(response.transcript.language.as_deref(), Some("en"));
        assert_eq!(response.transcript.segments[0].confidence, Some(1.0));
    }

    #[test]
    fn transcribe_accepts_imported_contract() {
        let response = transcribe(TranscriptionRequest {
            source: Some("fallback.wav".to_string()),
            language: Some("en".to_string()),
            input: TranscriptionInput::ImportedContract {
                transcript: TranscriptionContract::new(vec![TranscriptSegmentContract::new(
                    0,
                    "hello world",
                )]),
            },
            runtime: TranscriptionRuntimeSelection::default(),
        })
        .unwrap();

        assert_eq!(response.transcript.text.as_deref(), Some("hello world"));
        assert_eq!(response.transcript.source.as_deref(), Some("fallback.wav"));
    }

    #[test]
    fn transcribe_rejects_source_path_without_runnable_provider() {
        let error = transcribe(TranscriptionRequest {
            source: None,
            language: None,
            input: TranscriptionInput::SourcePath {
                path: "clip.wav".to_string(),
            },
            runtime: TranscriptionRuntimeSelection::default(),
        })
        .unwrap_err();

        assert!(error.to_string().contains("unsupported_runtime"));
    }

    #[test]
    fn transcription_plan_for_whisper_cpp_is_non_executing_and_names_text_transcripts() {
        let plan = transcription_plan(&TranscriptionRuntimeSelection {
            backend: Some(ModelRuntimeBackend::WhisperCpp),
            ..TranscriptionRuntimeSelection::default()
        });

        assert_eq!(plan.provider, TranscriptionProviderKind::WhisperCpp);
        assert_eq!(plan.execution_owner, "moenarch-text-transcripts");
        assert!(!plan.will_execute);
    }

    #[test]
    fn audio_runtime_selection_maps_to_model_runtime_backend() {
        let selection = AudioRuntimeSelection {
            backend: Some(AudioRuntime::WhisperCpp),
            ..AudioRuntimeSelection::default()
        };
        let generic = TranscriptionRuntimeSelection::from(selection);
        assert_eq!(generic.backend, Some(ModelRuntimeBackend::WhisperCpp));
    }
}
