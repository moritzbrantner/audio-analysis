//! Speaker diarization request/response contract and the request-level entry points.

use std::collections::BTreeMap;
use std::path::PathBuf;

use audio_analysis_recognition::{AudioRuntime, AudioRuntimeSelection, FallbackPolicy};
use audio_contracts::{DetectError, Result};

use crate::assignment::normalize_speaker_segment;
use crate::errors::{invalid_request, setup_error};
use crate::{
    DiarizationSegment, DiarizedSpeaker, EnergyVoiceActivityDetector, OnnxSpeakerEmbeddingConfig,
    SpeakerAudio, SpeakerDiarizer, SpeakerEmbedding, SpeakerTranscriptAssignmentPolicy,
    SpectralSpeakerEmbedder, WindowedSpeakerDiarizer,
};

/// Request for speaker diarization.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerDiarizationRequest {
    /// Optional source identifier.
    #[serde(default)]
    pub source: Option<String>,
    /// Optional duration for heuristic fallback.
    #[serde(default)]
    pub duration_seconds: Option<f32>,
    /// Runtime selection.
    #[serde(default)]
    pub model: AudioRuntimeSelection,
    /// Caller-supplied speaker segments.
    #[serde(default)]
    pub imported_segments: Vec<SpeakerSegmentPrediction>,
}

/// One speaker segment.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerSegmentPrediction {
    /// Speaker label.
    pub speaker: String,
    /// Start time in seconds.
    pub start_seconds: f32,
    /// End time in seconds.
    pub end_seconds: f32,
    /// Confidence score.
    #[serde(default)]
    pub score: Option<f32>,
}

/// Response for speaker diarization.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerDiarizationResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Selected model id.
    pub model_id: String,
    /// Runtime used.
    pub runtime: AudioRuntime,
    /// Speaker segments.
    pub segments: Vec<SpeakerSegmentPrediction>,
    /// Optional speaker embeddings keyed by emitted speaker label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_embeddings: Option<BTreeMap<String, SpeakerEmbedding>>,
    /// Provider diagnostics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<String>,
}

/// Speaker-owned diarization options shared by transcription and speaker surfaces.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerDiarizationOptions {
    /// Selected diarization model id.
    #[serde(default = "default_speaker_diarization_model")]
    pub model_id: String,
    /// ONNX speaker embedding model bundle or direct model path.
    #[serde(default)]
    pub speaker_embedding_model_bundle: Option<PathBuf>,
    /// ONNX speaker embedding model filename.
    #[serde(default)]
    pub speaker_embedding_model_file: Option<String>,
    /// ONNX speaker embedding input name.
    #[serde(default)]
    pub speaker_embedding_input_name: Option<String>,
    /// ONNX speaker embedding output name.
    #[serde(default)]
    pub speaker_embedding_output_name: Option<String>,
    /// ONNX speaker embedding vector dimension.
    #[serde(default)]
    pub speaker_embedding_dimension: Option<usize>,
    /// ONNX speaker embedding expected sample rate.
    #[serde(default)]
    pub speaker_embedding_sample_rate: Option<u32>,
    /// pyannote diarization bundle path.
    #[serde(default)]
    pub pyannote_model_bundle: Option<PathBuf>,
    /// pyannote manifest filename.
    #[serde(default)]
    pub pyannote_manifest_file: Option<String>,
    /// pyannote segmentation model filename.
    #[serde(default)]
    pub pyannote_segmentation_model_file: Option<String>,
    /// pyannote embedding model filename.
    #[serde(default)]
    pub pyannote_embedding_model_file: Option<String>,
    /// pyannote PLDA transform filename.
    #[serde(default)]
    pub pyannote_plda_transform_file: Option<String>,
    /// pyannote PLDA model filename.
    #[serde(default)]
    pub pyannote_plda_model_file: Option<String>,
    /// pyannote clustering config filename.
    #[serde(default)]
    pub pyannote_clustering_config_file: Option<String>,
    /// Whether provider responses should include speaker embeddings when available.
    #[serde(default)]
    pub return_speaker_embeddings: bool,
    /// Requested minimum number of speakers.
    #[serde(default)]
    pub min_speakers: Option<usize>,
    /// Requested maximum number of speakers.
    #[serde(default)]
    pub max_speakers: Option<usize>,
    /// Transcript Speaker Assignment policy.
    #[serde(default)]
    pub assignment_policy: SpeakerTranscriptAssignmentPolicy,
}

impl SpeakerDiarizationOptions {
    /// Validates the speaker-owned diarization contract.
    pub fn validate(&self) -> Result<()> {
        if self.min_speakers == Some(0) {
            return Err(invalid_request(
                "diarization min_speakers must be greater than zero",
            ));
        }
        if self.max_speakers == Some(0) {
            return Err(invalid_request(
                "diarization max_speakers must be greater than zero",
            ));
        }
        if let (Some(min), Some(max)) = (self.min_speakers, self.max_speakers) {
            if min > max {
                return Err(invalid_request(
                    "diarization min_speakers must be less than or equal to max_speakers",
                ));
            }
        }
        if self.speaker_embedding_dimension == Some(0) {
            return Err(invalid_request(
                "diarization speaker_embedding_dimension must be greater than zero",
            ));
        }
        if self.speaker_embedding_sample_rate == Some(0) {
            return Err(invalid_request(
                "diarization speaker_embedding_sample_rate must be greater than zero",
            ));
        }
        if is_pyannote_diarization_model(&self.model_id)
            && self.speaker_embedding_model_bundle.is_some()
        {
            return Err(invalid_request(
                "pyannote diarization uses pyannote_model_bundle, not speaker_embedding_model_bundle",
            ));
        }
        Ok(())
    }

    /// Returns true when the selected model id targets the pyannote provider.
    pub fn is_pyannote_model(&self) -> bool {
        is_pyannote_diarization_model(&self.model_id)
    }

    /// Returns true when ONNX speaker embedding execution is explicitly configured.
    pub fn uses_onnx_speaker_embedding(&self) -> bool {
        self.speaker_embedding_model_bundle.is_some()
    }

    /// Builds an ONNX speaker embedding config from this contract.
    pub fn onnx_speaker_embedding_config(&self) -> Result<OnnxSpeakerEmbeddingConfig> {
        let bundle_path = self
            .speaker_embedding_model_bundle
            .clone()
            .ok_or_else(|| setup_error("ONNX speaker embedding model bundle is required"))?;
        let embedding_dimension = self.speaker_embedding_dimension.ok_or_else(|| {
            invalid_request("ONNX speaker embedding dimension must be configured explicitly")
        })?;
        Ok(OnnxSpeakerEmbeddingConfig {
            bundle_path,
            model_file: self.speaker_embedding_model_file.clone(),
            input_name: self.speaker_embedding_input_name.clone(),
            output_name: self.speaker_embedding_output_name.clone(),
            embedding_dimension,
            sample_rate: self.speaker_embedding_sample_rate.unwrap_or(16_000),
        })
    }
}

impl Default for SpeakerDiarizationOptions {
    fn default() -> Self {
        Self {
            model_id: default_speaker_diarization_model(),
            speaker_embedding_model_bundle: None,
            speaker_embedding_model_file: None,
            speaker_embedding_input_name: None,
            speaker_embedding_output_name: None,
            speaker_embedding_dimension: None,
            speaker_embedding_sample_rate: None,
            pyannote_model_bundle: None,
            pyannote_manifest_file: None,
            pyannote_segmentation_model_file: None,
            pyannote_embedding_model_file: None,
            pyannote_plda_transform_file: None,
            pyannote_plda_model_file: None,
            pyannote_clustering_config_file: None,
            return_speaker_embeddings: false,
            min_speakers: None,
            max_speakers: None,
            assignment_policy: SpeakerTranscriptAssignmentPolicy::Majority,
        }
    }
}

/// Returns the default native speaker diarization model id.
pub fn default_speaker_diarization_model() -> String {
    "native-spectral-speaker-baseline".to_string()
}

/// Returns true when a diarization model id targets the pyannote provider.
pub fn is_pyannote_diarization_model(model_id: &str) -> bool {
    matches!(
        model_id,
        "pyannote/speaker-diarization-community-1" | "pyannote-community-1"
    )
}

/// Handles speaker diarization from imported segments or one-speaker fallback.
pub fn diarize_speakers(request: SpeakerDiarizationRequest) -> Result<SpeakerDiarizationResponse> {
    let model_id = request
        .model
        .model_id
        .clone()
        .or_else(|| request.model.model.as_ref().map(|model| model.name.clone()))
        .unwrap_or_else(|| "pyannote-speaker-diarization-3.1".to_string());

    if !request.imported_segments.is_empty() {
        let segments = request
            .imported_segments
            .into_iter()
            .map(normalize_speaker_segment)
            .collect::<Result<Vec<_>>>()?;
        return Ok(SpeakerDiarizationResponse {
            accepted: true,
            operation: "diarize".to_string(),
            model_id,
            runtime: AudioRuntime::Imported,
            segments,
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        });
    }

    match request.model.fallback_policy {
        FallbackPolicy::FastFallback | FallbackPolicy::HeuristicFallback => {
            let duration = request.duration_seconds.unwrap_or(0.0).max(0.0);
            Ok(SpeakerDiarizationResponse {
                accepted: true,
                operation: "diarize".to_string(),
                model_id,
                runtime: AudioRuntime::Heuristic,
                segments: vec![SpeakerSegmentPrediction {
                    speaker: "speaker_0".to_string(),
                    start_seconds: 0.0,
                    end_seconds: duration,
                    score: Some(1.0),
                }],
                speaker_embeddings: None,
                diagnostics: Vec::new(),
            })
        }
        FallbackPolicy::Error => Err(DetectError::InvalidArgument(
            "unsupported_runtime: native speaker diarization requires imported segments or an explicit fallback policy"
                .to_string(),
        )),
    }
}

/// Runs the deterministic native speaker diarization baseline over mono audio.
pub fn diarize_speaker_audio_baseline(
    samples: &[f32],
    sample_rate: u32,
) -> Result<SpeakerDiarizationResponse> {
    let audio = SpeakerAudio::mono(samples, sample_rate)?;
    let embedder = SpectralSpeakerEmbedder::default();
    // Heuristic baseline: attribute every span, including very short ones.
    let mut diarizer = WindowedSpeakerDiarizer::new(embedder, EnergyVoiceActivityDetector)
        .cluster_threshold(0.95)?
        .min_speaker_evidence_seconds(0.0)?;
    let merge_gap_seconds = diarizer.vad_options.merge_gap_seconds as f32;
    let result = diarizer.diarize(&audio)?;
    let segments = stable_speaker_predictions(result.segments, merge_gap_seconds)?;
    Ok(SpeakerDiarizationResponse {
        accepted: true,
        operation: "audio.speakers.diarize".to_string(),
        model_id: "native-spectral-speaker-baseline".to_string(),
        runtime: AudioRuntime::Heuristic,
        segments,
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    })
}

fn stable_speaker_predictions(
    segments: Vec<DiarizationSegment>,
    merge_gap_seconds: f32,
) -> Result<Vec<SpeakerSegmentPrediction>> {
    let mut unknown_labels: Vec<(String, String)> = Vec::new();
    let mut predictions = Vec::new();
    for segment in segments {
        let speaker = match segment.speaker {
            DiarizedSpeaker::Unresolved => continue,
            DiarizedSpeaker::Known(id) => id.as_str().to_string(),
            DiarizedSpeaker::Unknown(label) => {
                if let Some((_, stable)) = unknown_labels
                    .iter()
                    .find(|(existing, _)| existing == &label)
                {
                    stable.clone()
                } else {
                    let stable = format!("speaker_{}", unknown_labels.len());
                    unknown_labels.push((label, stable.clone()));
                    stable
                }
            }
        };
        predictions.push(normalize_speaker_segment(SpeakerSegmentPrediction {
            speaker,
            start_seconds: segment.start_seconds as f32,
            end_seconds: segment.end_seconds as f32,
            score: Some(segment.score),
        })?);
    }
    merge_speaker_predictions(predictions, merge_gap_seconds)
}

pub(crate) fn merge_speaker_predictions(
    segments: Vec<SpeakerSegmentPrediction>,
    merge_gap_seconds: f32,
) -> Result<Vec<SpeakerSegmentPrediction>> {
    let mut merged: Vec<SpeakerSegmentPrediction> = Vec::new();
    for segment in segments {
        if let Some(last) = merged.last_mut() {
            if last.speaker == segment.speaker
                && segment.start_seconds - last.end_seconds <= merge_gap_seconds
            {
                let last_duration = (last.end_seconds - last.start_seconds).max(0.0);
                let segment_duration = (segment.end_seconds - segment.start_seconds).max(0.0);
                let total = last_duration + segment_duration;
                last.end_seconds = segment.end_seconds;
                last.score = match (last.score, segment.score) {
                    (Some(left), Some(right)) if total > f32::EPSILON => {
                        Some(((left * last_duration) + (right * segment_duration)) / total)
                    }
                    (Some(left), Some(right)) => Some(left.max(right)),
                    (Some(left), None) => Some(left),
                    (None, Some(right)) => Some(right),
                    (None, None) => None,
                };
                continue;
            }
        }
        merged.push(segment);
    }
    Ok(merged)
}
