#![doc = include_str!("../README.md")]

#[cfg(feature = "pyannote-diarization")]
pub mod pyannote;
pub mod surface;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use audio_analysis_core::{interleaved_to_mono, rms, zero_pad_to, ChannelMix, FrameSpec};
pub use audio_analysis_recognition::AudioRuntime;
use audio_analysis_recognition::{
    AudioEmbeddingExtractor, AudioRuntimeSelection, FallbackPolicy, SpectralAudioEmbedder,
    SpectralEmbeddingConfig, TranscriptSegmentContract, TranscriptionContract,
};
use audio_contracts::{AudioBuffer, AudioFrame, DetectError, Result};

const SNAPSHOT_VERSION: u32 = 1;

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

#[cfg(feature = "pyannote-diarization")]
pub use pyannote::{
    PyannoteCommunityDiarizationConfig, PyannoteCommunityDiarizationResult,
    PyannoteCommunityDiarizer,
};

/// Speaker assignment policy for transcript segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SpeakerTranscriptAssignmentPolicy {
    /// Assign the speaker with the largest positive time overlap.
    #[default]
    Majority,
    /// Assign the speaker whose segment start is nearest the transcript start.
    NearestStart,
    /// Assign only when the diarization segment fully contains the transcript segment.
    StrictContained,
}

/// Backward-compatible alias for the speaker transcript assignment policy.
pub type TranscriptSpeakerOverlapPolicy = SpeakerTranscriptAssignmentPolicy;

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
/// Identifier for an enrolled speaker.
pub struct SpeakerId(String);

impl SpeakerId {
    /// Creates a new value.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "speaker id must not be empty".to_string(),
            ));
        }
        Ok(Self(value))
    }

    /// Returns this value as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for SpeakerId {
    type Error = DetectError;

    fn try_from(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<String> for SpeakerId {
    type Error = DetectError;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Human-readable speaker label.
pub struct SpeakerLabel(String);

impl SpeakerLabel {
    /// Creates a new value.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "speaker label must not be empty".to_string(),
            ));
        }
        Ok(Self(value))
    }

    /// Returns this value as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for SpeakerLabel {
    type Error = DetectError;

    fn try_from(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<String> for SpeakerLabel {
    type Error = DetectError;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Speaker embedding model metadata.
pub struct SpeakerEmbeddingModel {
    /// Model family or implementation class.
    pub family: SpeakerEmbeddingModelFamily,
    /// Stable model name.
    pub name: String,
    /// Stable model version.
    pub version: String,
    /// Embedding dimensionality.
    pub dimensions: usize,
}

impl SpeakerEmbeddingModel {
    /// Creates a new value.
    pub fn new(
        family: SpeakerEmbeddingModelFamily,
        name: impl Into<String>,
        version: impl Into<String>,
        dimensions: usize,
    ) -> Result<Self> {
        let model = Self {
            family,
            name: name.into(),
            version: version.into(),
            dimensions,
        };
        model.validate()?;
        Ok(model)
    }

    /// Returns a model descriptor for the deterministic spectral baseline.
    pub fn spectral_baseline(dimensions: usize) -> Result<Self> {
        Self::new(
            SpeakerEmbeddingModelFamily::SpectralBaseline,
            "spectral-speaker-baseline",
            "1",
            dimensions,
        )
    }

    fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "speaker embedding model name must not be empty".to_string(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "speaker embedding model version must not be empty".to_string(),
            ));
        }
        if self.dimensions == 0 {
            return Err(DetectError::InvalidArgument(
                "speaker embedding model dimensions must be greater than zero".to_string(),
            ));
        }
        if let SpeakerEmbeddingModelFamily::Custom(value) = &self.family {
            if value.trim().is_empty() {
                return Err(DetectError::InvalidArgument(
                    "custom speaker embedding model family must not be empty".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Known speaker embedding model families.
pub enum SpeakerEmbeddingModelFamily {
    /// Deterministic spectral baseline for tests and prototypes.
    SpectralBaseline,
    /// ECAPA-TDNN speaker embeddings.
    EcapaTdnn,
    /// x-vector speaker embeddings.
    XVector,
    /// pyannote-style speaker embeddings.
    Pyannote,
    /// SpeechBrain-compatible speaker verification embeddings.
    SpeechBrain,
    /// User-provided model family.
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Speaker embedding values with model identity and source sample rate.
pub struct SpeakerEmbedding {
    values: Vec<f32>,
    model: SpeakerEmbeddingModel,
    sample_rate: u32,
}

impl SpeakerEmbedding {
    /// Creates a new value.
    pub fn new(
        values: impl Into<Vec<f32>>,
        model: SpeakerEmbeddingModel,
        sample_rate: u32,
    ) -> Result<Self> {
        model.validate()?;
        let mut values = values.into();
        normalize_values(&mut values)?;
        let embedding = Self {
            values,
            model,
            sample_rate,
        };
        embedding.validate_stored()?;
        Ok(embedding)
    }

    /// Returns values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns model metadata.
    pub fn model(&self) -> &SpeakerEmbeddingModel {
        &self.model
    }

    /// Returns source sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns dimensions.
    pub fn dimensions(&self) -> usize {
        self.values.len()
    }

    /// Returns cosine similarity.
    pub fn cosine_similarity(&self, other: &Self) -> Result<f32> {
        self.validate_stored()?;
        other.validate_stored()?;
        validate_model_compatible(&self.model, &other.model)?;
        if self.dimensions() != other.dimensions() {
            return Err(DetectError::InvalidArgument(format!(
                "speaker embedding dimensions differ: {} != {}",
                self.dimensions(),
                other.dimensions()
            )));
        }
        Ok(self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(left, right)| left * right)
            .sum())
    }

    fn validate_stored(&self) -> Result<()> {
        self.model.validate()?;
        if self.values.len() != self.model.dimensions {
            return Err(DetectError::InvalidArgument(format!(
                "speaker embedding dimensions differ from model: {} != {}",
                self.values.len(),
                self.model.dimensions
            )));
        }
        if self.sample_rate == 0 {
            return Err(DetectError::InvalidAudioFormat {
                sample_rate: self.sample_rate,
                channels: 1,
            });
        }
        if self.values.is_empty() {
            return Err(DetectError::InvalidArgument(
                "speaker embedding must not be empty".to_string(),
            ));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(DetectError::InvalidArgument(
                "speaker embedding values must be finite".to_string(),
            ));
        }
        let norm = self
            .values
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm <= f32::EPSILON || !norm.is_finite() {
            return Err(DetectError::InvalidArgument(
                "speaker embedding norm must be finite and non-zero".to_string(),
            ));
        }
        if (norm - 1.0).abs() > 1.0e-3 {
            return Err(DetectError::InvalidArgument(
                "stored speaker embedding values must be normalized".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Borrowed speaker audio with validation suitable for speaker embedding.
pub struct SpeakerAudio<'a> {
    samples: &'a [f32],
    sample_rate: u32,
    channels: u16,
    speech_spans: Vec<SpeechSpan>,
    min_duration_seconds: Option<f64>,
    max_duration_seconds: Option<f64>,
}

impl<'a> SpeakerAudio<'a> {
    /// Creates mono speaker audio.
    pub fn mono(samples: &'a [f32], sample_rate: u32) -> Result<Self> {
        Self::interleaved(samples, sample_rate, 1)
    }

    /// Creates speaker audio from interleaved samples with a known channel count.
    pub fn interleaved(samples: &'a [f32], sample_rate: u32, channels: u16) -> Result<Self> {
        let audio = Self {
            samples,
            sample_rate,
            channels,
            speech_spans: Vec::new(),
            min_duration_seconds: None,
            max_duration_seconds: None,
        };
        audio.validate()?;
        Ok(audio)
    }

    /// Returns this value with allowed duration bounds.
    pub fn duration_bounds(
        mut self,
        min_duration_seconds: Option<f64>,
        max_duration_seconds: Option<f64>,
    ) -> Result<Self> {
        self.min_duration_seconds = min_duration_seconds;
        self.max_duration_seconds = max_duration_seconds;
        self.validate()?;
        Ok(self)
    }

    /// Returns this value with known speech activity spans.
    pub fn with_speech_spans(mut self, speech_spans: Vec<SpeechSpan>) -> Result<Self> {
        for span in &speech_spans {
            span.validate()?;
            if span.end_seconds > self.duration_seconds() {
                return Err(DetectError::InvalidArgument(
                    "speech span exceeds audio duration".to_string(),
                ));
            }
        }
        self.speech_spans = speech_spans;
        Ok(self)
    }

    /// Builds owned mono samples from an audio frame.
    pub fn mono_samples_from_frame(frame: &AudioFrame<'_>) -> Result<Vec<f32>> {
        interleaved_to_mono(frame.data, frame.channels, ChannelMix::Average)
    }

    /// Returns sample values.
    pub fn samples(&self) -> &'a [f32] {
        self.samples
    }

    /// Returns sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns channel count.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Returns speech spans.
    pub fn speech_spans(&self) -> &[SpeechSpan] {
        &self.speech_spans
    }

    /// Returns samples per channel.
    pub fn samples_per_channel(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Returns duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.samples_per_channel() as f64 / self.sample_rate as f64
    }

    /// Returns mono samples.
    pub fn to_mono(&self) -> Result<Vec<f32>> {
        if self.channels == 1 {
            return Ok(self.samples.to_vec());
        }
        interleaved_to_mono(
            &AudioBuffer::F32(self.samples.to_vec()),
            self.channels,
            ChannelMix::Average,
        )
    }

    fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 || self.channels == 0 {
            return Err(DetectError::InvalidAudioFormat {
                sample_rate: self.sample_rate,
                channels: self.channels,
            });
        }
        if self.samples.is_empty() {
            return Err(DetectError::InvalidArgument(
                "speaker audio samples must not be empty".to_string(),
            ));
        }
        if !self.samples.len().is_multiple_of(self.channels as usize) {
            return Err(DetectError::InvalidArgument(
                "speaker audio samples must contain complete interleaved frames".to_string(),
            ));
        }
        if self.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(DetectError::InvalidArgument(
                "speaker audio samples must be finite".to_string(),
            ));
        }
        let duration = self.duration_seconds();
        if let Some(minimum) = self.min_duration_seconds {
            if !minimum.is_finite() || minimum < 0.0 {
                return Err(DetectError::InvalidArgument(
                    "minimum speaker audio duration must be finite and non-negative".to_string(),
                ));
            }
            if duration < minimum {
                return Err(DetectError::InvalidArgument(format!(
                    "speaker audio duration {duration:.3}s is below minimum {minimum:.3}s"
                )));
            }
        }
        if let Some(maximum) = self.max_duration_seconds {
            if !maximum.is_finite() || maximum <= 0.0 {
                return Err(DetectError::InvalidArgument(
                    "maximum speaker audio duration must be finite and positive".to_string(),
                ));
            }
            if duration > maximum {
                return Err(DetectError::InvalidArgument(format!(
                    "speaker audio duration {duration:.3}s exceeds maximum {maximum:.3}s"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Enrolled speaker profile.
pub struct SpeakerProfile {
    id: SpeakerId,
    label: SpeakerLabel,
    embeddings: Vec<SpeakerEmbedding>,
    metadata: BTreeMap<String, String>,
}

impl SpeakerProfile {
    /// Creates a new value.
    pub fn new(id: SpeakerId, label: SpeakerLabel) -> Self {
        Self {
            id,
            label,
            embeddings: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Returns this value with embedding.
    pub fn with_embedding(mut self, embedding: SpeakerEmbedding) -> Result<Self> {
        self.add_embedding(embedding)?;
        Ok(self)
    }

    /// Returns this value with metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Adds an embedding.
    pub fn add_embedding(&mut self, embedding: SpeakerEmbedding) -> Result<()> {
        embedding.validate_stored()?;
        if let Some(model) = self.embedding_model() {
            validate_model_compatible(model, embedding.model())?;
        }
        self.embeddings.push(embedding);
        Ok(())
    }

    /// Returns id.
    pub fn id(&self) -> &SpeakerId {
        &self.id
    }

    /// Returns label.
    pub fn label(&self) -> &SpeakerLabel {
        &self.label
    }

    /// Returns embeddings.
    pub fn embeddings(&self) -> &[SpeakerEmbedding] {
        &self.embeddings
    }

    /// Returns metadata.
    pub fn metadata_map(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Returns embedding model.
    pub fn embedding_model(&self) -> Option<&SpeakerEmbeddingModel> {
        self.embeddings.first().map(SpeakerEmbedding::model)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Speaker match result.
pub struct SpeakerMatch {
    /// Matched speaker id.
    pub speaker_id: SpeakerId,
    /// Matched speaker label.
    pub label: SpeakerLabel,
    /// Cosine similarity score.
    pub score: f32,
    /// Difference between this score and the next-best score.
    pub margin: Option<f32>,
    /// Confidence bucket derived from score and margin.
    pub confidence: SpeakerConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Speaker match confidence bucket.
pub enum SpeakerConfidence {
    /// Score and margin are comfortably above common thresholds.
    High,
    /// Score passes threshold but leaves less separation.
    Medium,
    /// Score is weak or accepted only by permissive settings.
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Policy for accepted unknown results.
pub enum UnknownSpeakerPolicy {
    /// Return a result with `unknown` set when thresholds fail.
    #[default]
    ReturnUnknown,
    /// Return no best match when thresholds fail.
    NoMatch,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Speaker identification thresholds and result options.
pub struct SpeakerIdentificationOptions {
    /// Minimum accepted best score.
    pub min_score: f32,
    /// Optional minimum best-vs-second margin.
    pub min_margin: Option<f32>,
    /// Maximum ranked results to return.
    pub max_results: usize,
    /// Unknown handling policy.
    pub unknown_policy: UnknownSpeakerPolicy,
}

impl SpeakerIdentificationOptions {
    /// Creates a new value.
    pub fn new(min_score: f32) -> Result<Self> {
        let options = Self {
            min_score,
            ..Self::default()
        };
        options.validate()?;
        Ok(options)
    }

    /// Returns this value with a minimum margin.
    pub fn min_margin(mut self, min_margin: f32) -> Result<Self> {
        self.min_margin = Some(min_margin);
        self.validate()?;
        Ok(self)
    }

    /// Returns this value with max results.
    pub fn max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    /// Returns this value with unknown policy.
    pub fn unknown_policy(mut self, unknown_policy: UnknownSpeakerPolicy) -> Self {
        self.unknown_policy = unknown_policy;
        self
    }

    fn validate(&self) -> Result<()> {
        if !self.min_score.is_finite() || !(-1.0..=1.0).contains(&self.min_score) {
            return Err(DetectError::InvalidArgument(
                "minimum speaker identification score must be finite and between -1.0 and 1.0"
                    .to_string(),
            ));
        }
        if let Some(margin) = self.min_margin {
            if !margin.is_finite() || margin < 0.0 {
                return Err(DetectError::InvalidArgument(
                    "minimum speaker identification margin must be finite and non-negative"
                        .to_string(),
                ));
            }
        }
        if self.max_results == 0 {
            return Err(DetectError::InvalidArgument(
                "max speaker identification results must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for SpeakerIdentificationOptions {
    fn default() -> Self {
        Self {
            min_score: 0.82,
            min_margin: Some(0.05),
            max_results: 5,
            unknown_policy: UnknownSpeakerPolicy::ReturnUnknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Speaker identification result.
pub struct SpeakerIdentificationResult {
    /// Accepted best match, if any.
    pub best_match: Option<SpeakerMatch>,
    /// Ranked candidate matches.
    pub ranked_matches: Vec<SpeakerMatch>,
    /// Whether the query should be treated as unknown.
    pub unknown: bool,
    /// Best-vs-second score margin.
    pub margin: Option<f32>,
    /// Query embedding model metadata.
    pub embedding_model: SpeakerEmbeddingModel,
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Speaker profile library.
pub struct SpeakerLibrary {
    profiles: BTreeMap<SpeakerId, SpeakerProfile>,
    embedding_model: Option<SpeakerEmbeddingModel>,
}

impl SpeakerLibrary {
    /// Creates a new value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enrolls a speaker from audio.
    pub fn enroll<E: SpeakerEmbeddingExtractor>(
        &mut self,
        id: SpeakerId,
        label: SpeakerLabel,
        audio: &SpeakerAudio<'_>,
        extractor: &mut E,
    ) -> Result<SpeakerProfile> {
        let embedding = extractor.embed_speaker(audio)?;
        let profile = SpeakerProfile::new(id, label).with_embedding(embedding)?;
        self.add_profile(profile.clone())?;
        Ok(profile)
    }

    /// Adds a profile.
    pub fn add_profile(&mut self, profile: SpeakerProfile) -> Result<()> {
        validate_profile_header(&profile)?;
        if profile.embeddings().is_empty() {
            return Err(DetectError::InvalidArgument(format!(
                "speaker profile `{}` must contain at least one embedding",
                profile.id().as_str()
            )));
        }
        let model = profile.embedding_model().cloned().ok_or_else(|| {
            DetectError::InvalidArgument("speaker profile must contain a model".to_string())
        })?;
        for embedding in profile.embeddings() {
            embedding.validate_stored()?;
            validate_model_compatible(&model, embedding.model())?;
        }
        self.validate_model(&model)?;
        if self.profiles.contains_key(profile.id()) {
            return Err(DetectError::InvalidArgument(
                "duplicate speaker id".to_string(),
            ));
        }
        self.embedding_model = Some(model);
        self.profiles.insert(profile.id().clone(), profile);
        Ok(())
    }

    /// Adds an embedding to an existing speaker profile.
    pub fn add_embedding(&mut self, id: &SpeakerId, embedding: SpeakerEmbedding) -> Result<()> {
        self.validate_model(embedding.model())?;
        let profile = self.profiles.get_mut(id).ok_or_else(|| {
            DetectError::InvalidArgument(format!("unknown speaker id `{}`", id.as_str()))
        })?;
        profile.add_embedding(embedding)
    }

    /// Returns profile.
    pub fn profile(&self, id: &SpeakerId) -> Option<&SpeakerProfile> {
        self.profiles.get(id)
    }

    /// Returns profiles.
    pub fn profiles(&self) -> impl Iterator<Item = &SpeakerProfile> {
        self.profiles.values()
    }

    /// Returns len.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Returns whether is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Returns embedding model.
    pub fn embedding_model(&self) -> Option<&SpeakerEmbeddingModel> {
        self.embedding_model.as_ref()
    }

    /// Identifies a speaker embedding.
    pub fn identify(
        &self,
        embedding: &SpeakerEmbedding,
        options: &SpeakerIdentificationOptions,
    ) -> Result<SpeakerIdentificationResult> {
        options.validate()?;
        self.validate_model(embedding.model())?;

        let mut ranked_matches = Vec::new();
        for profile in self.profiles.values() {
            let Some(score) = best_profile_score(profile, embedding)? else {
                continue;
            };
            ranked_matches.push(SpeakerMatch {
                speaker_id: profile.id().clone(),
                label: profile.label().clone(),
                score,
                margin: None,
                confidence: SpeakerConfidence::Low,
            });
        }
        ranked_matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.speaker_id.cmp(&right.speaker_id))
        });

        let margin = match (ranked_matches.first(), ranked_matches.get(1)) {
            (Some(best), Some(second)) => Some(best.score - second.score),
            (Some(_), None) => None,
            _ => None,
        };
        for index in 0..ranked_matches.len() {
            let next_score = ranked_matches.get(index + 1).map(|value| value.score);
            ranked_matches[index].margin =
                next_score.map(|score| ranked_matches[index].score - score);
            ranked_matches[index].confidence =
                confidence_for(ranked_matches[index].score, ranked_matches[index].margin);
        }

        let accepted = ranked_matches.first().is_some_and(|best| {
            best.score >= options.min_score
                && options
                    .min_margin
                    .is_none_or(|minimum| margin.is_some_and(|value| value >= minimum))
        });
        let best_match = if accepted {
            ranked_matches.first().cloned()
        } else {
            None
        };
        let unknown = !accepted && options.unknown_policy == UnknownSpeakerPolicy::ReturnUnknown;

        ranked_matches.truncate(options.max_results);
        Ok(SpeakerIdentificationResult {
            best_match,
            ranked_matches,
            unknown,
            margin,
            embedding_model: embedding.model().clone(),
        })
    }

    /// Converts this value to snapshot.
    pub fn to_snapshot(&self) -> Result<SpeakerLibrarySnapshot> {
        let embedding_model = self.embedding_model.clone().ok_or_else(|| {
            DetectError::InvalidArgument(
                "speaker library snapshot requires at least one profile".to_string(),
            )
        })?;
        Ok(SpeakerLibrarySnapshot {
            version: SNAPSHOT_VERSION,
            embedding_model,
            profiles: self.profiles.values().cloned().collect(),
        })
    }

    /// Builds this value from snapshot.
    pub fn from_snapshot(snapshot: SpeakerLibrarySnapshot) -> Result<Self> {
        if snapshot.version != SNAPSHOT_VERSION {
            return Err(DetectError::InvalidArgument(format!(
                "unsupported speaker library snapshot version `{}`",
                snapshot.version
            )));
        }
        snapshot.embedding_model.validate()?;
        let mut library = Self::new();
        for profile in snapshot.profiles {
            for embedding in profile.embeddings() {
                validate_model_compatible(&snapshot.embedding_model, embedding.model())?;
                if embedding.dimensions() != snapshot.embedding_model.dimensions {
                    return Err(DetectError::InvalidArgument(
                        "speaker library snapshot embedding dimensions did not match model"
                            .to_string(),
                    ));
                }
            }
            library.add_profile(profile)?;
        }
        if library.embedding_model.as_ref() != Some(&snapshot.embedding_model) {
            return Err(DetectError::InvalidArgument(
                "speaker library snapshot model did not match profiles".to_string(),
            ));
        }
        Ok(library)
    }

    /// Converts this value to JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.to_snapshot()?)
            .map_err(|err| DetectError::Source(err.to_string()))
    }

    /// Builds this value from JSON str.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let snapshot = serde_json::from_str::<SpeakerLibrarySnapshot>(json)
            .map_err(|err| DetectError::Source(err.to_string()))?;
        Self::from_snapshot(snapshot)
    }

    fn validate_model(&self, model: &SpeakerEmbeddingModel) -> Result<()> {
        if let Some(expected) = &self.embedding_model {
            validate_model_compatible(expected, model)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Stable serialized speaker library format.
pub struct SpeakerLibrarySnapshot {
    /// Snapshot format version.
    pub version: u32,
    /// Embedding model used by all profiles.
    pub embedding_model: SpeakerEmbeddingModel,
    /// Speaker profiles.
    pub profiles: Vec<SpeakerProfile>,
}

/// Trait for speaker embedding extractor implementations.
pub trait SpeakerEmbeddingExtractor {
    /// Returns model metadata.
    fn model_info(&self) -> SpeakerEmbeddingModel;

    /// Embeds speaker audio.
    fn embed_speaker(&mut self, audio: &SpeakerAudio<'_>) -> Result<SpeakerEmbedding>;
}

/// Request for a model-backed speaker embedding provider.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerEmbeddingRequest<'a> {
    /// Borrowed audio to embed.
    pub audio: SpeakerAudio<'a>,
}

/// Response from a model-backed speaker embedding provider.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerEmbeddingResponse {
    /// Stable provider model identifier.
    pub model_id: String,
    /// Runtime used to produce the embedding.
    pub runtime: AudioRuntime,
    /// Normalized speaker embedding.
    pub embedding: SpeakerEmbedding,
    /// Provider diagnostics.
    pub diagnostics: Vec<String>,
}

/// Trait for speaker embedding providers that can expose runtime diagnostics.
pub trait SpeakerEmbeddingProvider {
    /// Returns provider identifier.
    fn provider_id(&self) -> &str;

    /// Embeds speaker audio and returns runtime metadata.
    fn embed(&mut self, request: SpeakerEmbeddingRequest<'_>) -> Result<SpeakerEmbeddingResponse>;
}

#[derive(Debug, Clone, PartialEq)]
/// Deterministic spectral speaker embedder.
///
/// This is a deterministic baseline useful for tests and prototypes. It is not
/// production-grade speaker verification.
pub struct SpectralSpeakerEmbedder {
    inner: SpectralAudioEmbedder,
    model: SpeakerEmbeddingModel,
}

impl SpectralSpeakerEmbedder {
    /// Creates a new value.
    pub fn new(config: SpectralEmbeddingConfig) -> Result<Self> {
        let inner = SpectralAudioEmbedder::new(config.clone())?;
        let model = SpeakerEmbeddingModel::spectral_baseline(8 + config.bands)?;
        Ok(Self { inner, model })
    }

    /// Returns the inner generic spectral embedder.
    pub fn inner(&self) -> &SpectralAudioEmbedder {
        &self.inner
    }
}

impl Default for SpectralSpeakerEmbedder {
    fn default() -> Self {
        Self::new(SpectralEmbeddingConfig::default()).expect("valid default spectral config")
    }
}

impl SpeakerEmbeddingExtractor for SpectralSpeakerEmbedder {
    fn model_info(&self) -> SpeakerEmbeddingModel {
        self.model.clone()
    }

    fn embed_speaker(&mut self, audio: &SpeakerAudio<'_>) -> Result<SpeakerEmbedding> {
        let samples = audio.to_mono()?;
        let embedding = self.inner.embed_samples(&samples, audio.sample_rate())?;
        SpeakerEmbedding::new(
            embedding.values().to_vec(),
            self.model_info(),
            audio.sample_rate(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Configuration for ONNX-backed speaker embeddings.
pub struct OnnxSpeakerEmbeddingConfig {
    /// Path to a direct `.onnx` file, a directory containing the ONNX file, or a
    /// model-runtime bundle when `model-bundles` is enabled.
    pub bundle_path: PathBuf,
    /// Model filename or bundle remote/local suffix. Defaults to `model.onnx`.
    pub model_file: Option<String>,
    /// Optional ONNX input name. Defaults to the sole model input.
    pub input_name: Option<String>,
    /// Optional ONNX output name. Defaults to the sole model output.
    pub output_name: Option<String>,
    /// Expected embedding dimensionality.
    pub embedding_dimension: usize,
    /// Required request sample rate. Defaults to 16 kHz when zero in
    /// `Default`, but explicit zero is rejected during validation.
    pub sample_rate: u32,
}

impl OnnxSpeakerEmbeddingConfig {
    /// Returns the model filename, defaulting to `model.onnx`.
    pub fn model_file_name(&self) -> &str {
        self.model_file.as_deref().unwrap_or("model.onnx")
    }

    /// Returns the configured sample rate.
    pub fn effective_sample_rate(&self) -> u32 {
        if self.sample_rate == 0 {
            16_000
        } else {
            self.sample_rate
        }
    }

    fn validate(&self) -> Result<()> {
        if self.bundle_path.as_os_str().is_empty() {
            return Err(setup_error(
                "ONNX speaker embedding bundle path is required",
            ));
        }
        if let Some(model_file) = &self.model_file {
            if model_file.trim().is_empty() {
                return Err(invalid_request(
                    "ONNX speaker embedding model_file must not be empty",
                ));
            }
        }
        if let Some(input_name) = &self.input_name {
            if input_name.trim().is_empty() {
                return Err(invalid_request(
                    "ONNX speaker embedding input_name must not be empty",
                ));
            }
        }
        if let Some(output_name) = &self.output_name {
            if output_name.trim().is_empty() {
                return Err(invalid_request(
                    "ONNX speaker embedding output_name must not be empty",
                ));
            }
        }
        if self.embedding_dimension == 0 {
            return Err(invalid_request(
                "ONNX speaker embedding dimension must be greater than zero",
            ));
        }
        if self.sample_rate == 0 {
            return Err(invalid_request(
                "ONNX speaker embedding sample_rate must be greater than zero",
            ));
        }
        Ok(())
    }
}

impl Default for OnnxSpeakerEmbeddingConfig {
    fn default() -> Self {
        Self {
            bundle_path: PathBuf::new(),
            model_file: None,
            input_name: None,
            output_name: None,
            embedding_dimension: 0,
            sample_rate: 16_000,
        }
    }
}

/// ONNX-backed speaker embedding provider.
pub struct OnnxSpeakerEmbedder {
    config: OnnxSpeakerEmbeddingConfig,
    model_path: PathBuf,
    model: SpeakerEmbeddingModel,
    #[cfg(feature = "onnx")]
    runner: Option<Box<dyn runtime_onnx::OnnxRunner + Send>>,
}

impl std::fmt::Debug for OnnxSpeakerEmbedder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OnnxSpeakerEmbedder")
            .field("config", &self.config)
            .field("model_path", &self.model_path)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl OnnxSpeakerEmbedder {
    /// Creates a new model descriptor.
    pub fn new(model_path: impl AsRef<Path>, model: SpeakerEmbeddingModel) -> Result<Self> {
        model.validate()?;
        Ok(Self {
            config: OnnxSpeakerEmbeddingConfig {
                bundle_path: model_path.as_ref().to_path_buf(),
                embedding_dimension: model.dimensions,
                sample_rate: 16_000,
                ..OnnxSpeakerEmbeddingConfig::default()
            },
            model_path: model_path.as_ref().to_path_buf(),
            model,
            #[cfg(feature = "onnx")]
            runner: None,
        })
    }

    /// Creates an executable ONNX speaker embedder from config.
    pub fn from_config(config: OnnxSpeakerEmbeddingConfig) -> Result<Self> {
        let model_path = resolve_onnx_speaker_model_path(&config)?;
        let model = SpeakerEmbeddingModel::new(
            SpeakerEmbeddingModelFamily::Custom("onnx-speaker-embedding".to_string()),
            model_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("onnx-speaker-embedding"),
            "1",
            config.embedding_dimension,
        )?;
        Self::from_resolved_model(config, model_path, model)
    }

    #[cfg(feature = "onnx")]
    fn from_resolved_model(
        config: OnnxSpeakerEmbeddingConfig,
        model_path: PathBuf,
        model: SpeakerEmbeddingModel,
    ) -> Result<Self> {
        let runner = runtime_onnx::from_file_cpu_single_threaded(&model_path)
            .map_err(map_onnx_session_error)?;
        Ok(Self {
            config,
            model_path,
            model,
            runner: Some(Box::new(runner)),
        })
    }

    #[cfg(not(feature = "onnx"))]
    fn from_resolved_model(
        config: OnnxSpeakerEmbeddingConfig,
        model_path: PathBuf,
        model: SpeakerEmbeddingModel,
    ) -> Result<Self> {
        let _ = (config, model_path, model);
        Err(unsupported_runtime(
            "ONNX speaker embedding requires the `onnx` feature",
        ))
    }

    /// Creates an ONNX speaker embedder from a test or custom runner.
    #[cfg(feature = "onnx")]
    pub fn from_runner(
        config: OnnxSpeakerEmbeddingConfig,
        runner: impl runtime_onnx::OnnxRunner + Send + 'static,
    ) -> Result<Self> {
        let model_path = resolve_onnx_speaker_model_path(&config)?;
        let model = SpeakerEmbeddingModel::new(
            SpeakerEmbeddingModelFamily::Custom("onnx-speaker-embedding".to_string()),
            model_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("onnx-speaker-embedding"),
            "1",
            config.embedding_dimension,
        )?;
        Ok(Self {
            config,
            model_path,
            model,
            runner: Some(Box::new(runner)),
        })
    }

    /// Returns model path.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Returns config.
    pub fn config(&self) -> &OnnxSpeakerEmbeddingConfig {
        &self.config
    }
}

impl SpeakerEmbeddingExtractor for OnnxSpeakerEmbedder {
    fn model_info(&self) -> SpeakerEmbeddingModel {
        self.model.clone()
    }

    fn embed_speaker(&mut self, audio: &SpeakerAudio<'_>) -> Result<SpeakerEmbedding> {
        Ok(self
            .embed(SpeakerEmbeddingRequest {
                audio: audio.clone(),
            })?
            .embedding)
    }
}

impl SpeakerEmbeddingProvider for OnnxSpeakerEmbedder {
    fn provider_id(&self) -> &str {
        "onnx-speaker-embedding"
    }

    fn embed(&mut self, request: SpeakerEmbeddingRequest<'_>) -> Result<SpeakerEmbeddingResponse> {
        self.config.validate()?;
        if request.audio.sample_rate() != self.config.sample_rate {
            return Err(invalid_request(format!(
                "ONNX speaker embedding requires {} Hz audio, got {} Hz",
                self.config.sample_rate,
                request.audio.sample_rate()
            )));
        }
        let mono = request.audio.to_mono()?;
        #[cfg(feature = "onnx")]
        let metadata_diagnostics = self.onnx_runtime_metadata_diagnostics()?;
        #[cfg(not(feature = "onnx"))]
        let metadata_diagnostics = Vec::new();
        let (embedding, input_diagnostics) =
            self.embed_mono_with_onnx(&mono, request.audio.sample_rate())?;
        let mut diagnostics = vec![
            "speakerEmbeddingProvider=onnx".to_string(),
            "speakerEmbeddingRuntime=onnx".to_string(),
            format!("speakerEmbeddingModelPath={}", self.model_path.display()),
            format!(
                "speakerEmbeddingDimension={}",
                self.config.embedding_dimension
            ),
            format!(
                "speakerEmbeddingInputName={}",
                self.config.input_name.as_deref().unwrap_or("<auto>")
            ),
            format!(
                "speakerEmbeddingOutputName={}",
                self.config.output_name.as_deref().unwrap_or("<auto>")
            ),
        ];
        diagnostics.extend(metadata_diagnostics);
        diagnostics.extend(input_diagnostics);
        Ok(SpeakerEmbeddingResponse {
            model_id: self.model.name.clone(),
            runtime: AudioRuntime::Onnx,
            embedding,
            diagnostics,
        })
    }
}

impl OnnxSpeakerEmbedder {
    #[cfg(feature = "onnx")]
    fn onnx_runtime_metadata_diagnostics(&mut self) -> Result<Vec<String>> {
        let runner = self.runner.as_mut().ok_or_else(|| {
            unsupported_runtime("ONNX speaker embedding runner is not initialized")
        })?;
        let metadata = runner.metadata().map_err(map_onnx_runtime_error)?;
        let input = select_onnx_speaker_input(&metadata, self.config.input_name.as_deref())?;
        let output = select_onnx_speaker_output(&metadata, self.config.output_name.as_deref())?;
        Ok(vec![
            format!("speakerEmbeddingSelectedInputName={}", input.name),
            format!(
                "speakerEmbeddingSelectedInputRank={}",
                input.dimensions.len()
            ),
            format!(
                "speakerEmbeddingSelectedInputDimensions={}",
                format_onnx_dimensions(&input.dimensions)
            ),
            format!("speakerEmbeddingSelectedOutputName={}", output.name),
            format!(
                "speakerEmbeddingSelectedOutputRank={}",
                output.dimensions.len()
            ),
            format!(
                "speakerEmbeddingSelectedOutputDimensions={}",
                format_onnx_dimensions(&output.dimensions)
            ),
            format!(
                "speakerEmbeddingExpectedDimension={}",
                self.config.embedding_dimension
            ),
        ])
    }

    #[cfg(feature = "onnx")]
    fn embed_mono_with_onnx(
        &mut self,
        mono: &[f32],
        sample_rate: u32,
    ) -> Result<(SpeakerEmbedding, Vec<String>)> {
        let runner = self.runner.as_mut().ok_or_else(|| {
            unsupported_runtime("ONNX speaker embedding runner is not initialized")
        })?;
        let metadata = runner.metadata().map_err(map_onnx_runtime_error)?;
        let input = select_onnx_speaker_input(&metadata, self.config.input_name.as_deref())?;
        let output = select_onnx_speaker_output(&metadata, self.config.output_name.as_deref())?;
        let (shape, values, input_diagnostics) =
            onnx_speaker_input_tensor(&input, mono, sample_rate)?;
        let outputs = runner
            .run(vec![runtime_onnx::single_f32_input(
                input.name.clone(),
                shape,
                values,
            )
            .map_err(map_onnx_runtime_error)?])
            .map_err(map_onnx_runtime_error)?;
        let tensor =
            runtime_onnx::f32_output_by_name_or_index(&outputs, &output.name, output.index)
                .map_err(map_onnx_runtime_error)?;
        let values = onnx_speaker_embedding_values(tensor, self.config.embedding_dimension)?;
        Ok((
            SpeakerEmbedding::new(values, self.model_info(), sample_rate)?,
            input_diagnostics,
        ))
    }

    #[cfg(not(feature = "onnx"))]
    fn embed_mono_with_onnx(
        &mut self,
        _mono: &[f32],
        _sample_rate: u32,
    ) -> Result<(SpeakerEmbedding, Vec<String>)> {
        Err(unsupported_runtime(
            "ONNX speaker embedding requires the `onnx` feature",
        ))
    }
}

/// Trait for voice activity detector implementations.
pub trait VoiceActivityDetector {
    /// Detects speech spans.
    fn detect_speech(&mut self, audio: &SpeakerAudio<'_>) -> Result<Vec<SpeechSpan>>;
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
/// Speech activity span.
pub struct SpeechSpan {
    /// Start in seconds.
    pub start_seconds: f64,
    /// End in seconds.
    pub end_seconds: f64,
    /// Speech confidence or energy score.
    pub score: f32,
}

impl SpeechSpan {
    /// Creates a new value.
    pub fn new(start_seconds: f64, end_seconds: f64, score: f32) -> Result<Self> {
        let span = Self {
            start_seconds,
            end_seconds,
            score,
        };
        span.validate()?;
        Ok(span)
    }

    /// Returns duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.end_seconds - self.start_seconds
    }

    fn validate(&self) -> Result<()> {
        if !self.start_seconds.is_finite()
            || !self.end_seconds.is_finite()
            || !self.score.is_finite()
        {
            return Err(DetectError::InvalidArgument(
                "speech span values must be finite".to_string(),
            ));
        }
        if self.start_seconds < 0.0 || self.end_seconds <= self.start_seconds {
            return Err(DetectError::InvalidArgument(
                "speech span must have non-negative start and positive duration".to_string(),
            ));
        }
        if self.score < 0.0 {
            return Err(DetectError::InvalidArgument(
                "speech span score must be non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Energy/RMS VAD configuration.
pub struct EnergyVadConfig {
    /// RMS threshold above which a frame is speech.
    pub rms_threshold: f32,
    /// Frame duration in seconds.
    pub frame_seconds: f64,
    /// Hop duration in seconds.
    pub hop_seconds: f64,
    /// Minimum accepted speech span duration.
    pub min_speech_seconds: f64,
    /// Merge speech spans separated by less than this gap.
    pub merge_gap_seconds: f64,
}

impl EnergyVadConfig {
    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if !self.rms_threshold.is_finite() || self.rms_threshold < 0.0 {
            return Err(DetectError::InvalidArgument(
                "VAD RMS threshold must be finite and non-negative".to_string(),
            ));
        }
        if !self.frame_seconds.is_finite() || self.frame_seconds <= 0.0 {
            return Err(DetectError::InvalidArgument(
                "VAD frame duration must be finite and positive".to_string(),
            ));
        }
        if !self.hop_seconds.is_finite() || self.hop_seconds <= 0.0 {
            return Err(DetectError::InvalidArgument(
                "VAD hop duration must be finite and positive".to_string(),
            ));
        }
        if !self.min_speech_seconds.is_finite() || self.min_speech_seconds < 0.0 {
            return Err(DetectError::InvalidArgument(
                "VAD minimum speech duration must be finite and non-negative".to_string(),
            ));
        }
        if !self.merge_gap_seconds.is_finite() || self.merge_gap_seconds < 0.0 {
            return Err(DetectError::InvalidArgument(
                "VAD merge gap must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for EnergyVadConfig {
    fn default() -> Self {
        Self {
            rms_threshold: 0.01,
            frame_seconds: 0.03,
            hop_seconds: 0.01,
            min_speech_seconds: 0.08,
            merge_gap_seconds: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Simple energy/RMS voice activity detector.
pub struct EnergyVoiceActivityDetector {
    /// The config value.
    pub config: EnergyVadConfig,
}

impl EnergyVoiceActivityDetector {
    /// Creates a new value.
    pub fn new(config: EnergyVadConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }
}

impl Default for EnergyVoiceActivityDetector {
    fn default() -> Self {
        Self::new(EnergyVadConfig::default()).expect("valid default VAD config")
    }
}

impl VoiceActivityDetector for EnergyVoiceActivityDetector {
    fn detect_speech(&mut self, audio: &SpeakerAudio<'_>) -> Result<Vec<SpeechSpan>> {
        self.config.validate()?;
        let samples = audio.to_mono()?;
        let frame_size = seconds_to_samples(self.config.frame_seconds, audio.sample_rate())?;
        let hop_size = seconds_to_samples(self.config.hop_seconds, audio.sample_rate())?;
        let frame_spec = FrameSpec::new(frame_size, hop_size)?;
        let mut active = Vec::new();
        let frames: Vec<(usize, Vec<f32>)> = if samples.len() < frame_size {
            vec![(0, zero_pad_to(samples.clone(), frame_size))]
        } else {
            frame_spec
                .frames(&samples)
                .map(|(start, frame)| (start, frame.to_vec()))
                .collect()
        };
        for (start, frame) in frames {
            let score = rms(&frame);
            if score >= self.config.rms_threshold {
                let start_seconds = start as f64 / audio.sample_rate() as f64;
                let end_seconds =
                    ((start + frame_size).min(samples.len())) as f64 / audio.sample_rate() as f64;
                active.push(SpeechSpan::new(start_seconds, end_seconds, score)?);
            }
        }
        Ok(filter_short_spans(
            merge_spans(active, self.config.merge_gap_seconds)?,
            self.config.min_speech_seconds,
        ))
    }
}

/// Trait for speaker diarization implementations.
pub trait SpeakerDiarizer {
    /// Diarizes speaker activity.
    fn diarize(&mut self, audio: &SpeakerAudio<'_>) -> Result<DiarizationResult>;
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Speaker diarization result.
pub struct DiarizationResult {
    /// Diarization segments.
    pub segments: Vec<DiarizationSegment>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Diarization segment.
pub struct DiarizationSegment {
    /// Start in seconds.
    pub start_seconds: f64,
    /// End in seconds.
    pub end_seconds: f64,
    /// Speaker assignment.
    pub speaker: DiarizedSpeaker,
    /// Segment confidence or cluster score.
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Diarized speaker assignment.
pub enum DiarizedSpeaker {
    /// Enrolled speaker.
    Known(SpeakerId),
    /// Unenrolled cluster label.
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq)]
/// Simple VAD/window/embed/cluster diarizer.
pub struct WindowedSpeakerDiarizer<E, V> {
    /// Speaker embedding extractor.
    pub extractor: E,
    /// Voice activity detector.
    pub vad: V,
    /// Optional enrolled speaker library.
    pub library: Option<SpeakerLibrary>,
    /// Identification options for mapping clusters to known speakers.
    pub identification_options: SpeakerIdentificationOptions,
    /// Minimum cosine similarity for assigning a window to an existing cluster.
    pub cluster_threshold: f32,
    /// Minimum number of unknown speaker clusters to produce when enough
    /// unknown speech windows are available.
    pub min_speakers: Option<usize>,
    /// Maximum number of unknown speaker clusters to produce.
    pub max_speakers: Option<usize>,
}

impl<E, V> WindowedSpeakerDiarizer<E, V> {
    /// Creates a new value.
    pub fn new(extractor: E, vad: V) -> Self {
        Self {
            extractor,
            vad,
            library: None,
            identification_options: SpeakerIdentificationOptions::default(),
            cluster_threshold: 0.72,
            min_speakers: None,
            max_speakers: None,
        }
    }

    /// Returns this value with a speaker library.
    pub fn library(mut self, library: SpeakerLibrary) -> Self {
        self.library = Some(library);
        self
    }

    /// Returns this value with cluster threshold.
    pub fn cluster_threshold(mut self, cluster_threshold: f32) -> Result<Self> {
        if !cluster_threshold.is_finite() || !(-1.0..=1.0).contains(&cluster_threshold) {
            return Err(DetectError::InvalidArgument(
                "speaker diarization cluster threshold must be finite and between -1.0 and 1.0"
                    .to_string(),
            ));
        }
        self.cluster_threshold = cluster_threshold;
        Ok(self)
    }

    /// Returns this value with unknown-speaker cluster bounds.
    pub fn speaker_bounds(
        mut self,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<Self> {
        validate_speaker_bounds(min_speakers, max_speakers)?;
        self.min_speakers = min_speakers;
        self.max_speakers = max_speakers;
        Ok(self)
    }
}

impl<E, V> SpeakerDiarizer for WindowedSpeakerDiarizer<E, V>
where
    E: SpeakerEmbeddingExtractor,
    V: VoiceActivityDetector,
{
    fn diarize(&mut self, audio: &SpeakerAudio<'_>) -> Result<DiarizationResult> {
        let spans = self.vad.detect_speech(audio)?;
        let mono = audio.to_mono()?;
        let mut windows = Vec::new();

        for span in spans {
            let start = (span.start_seconds * audio.sample_rate() as f64).round() as usize;
            let end = (span.end_seconds * audio.sample_rate() as f64).round() as usize;
            let segment_samples = &mono[start.min(mono.len())..end.min(mono.len())];
            if segment_samples.is_empty() {
                continue;
            }
            let segment_audio = SpeakerAudio::mono(segment_samples, audio.sample_rate())?;
            let embedding = self.extractor.embed_speaker(&segment_audio)?;

            let known = self
                .library
                .as_ref()
                .map(|library| library.identify(&embedding, &self.identification_options))
                .transpose()?
                .and_then(|result| result.best_match.map(|value| value.speaker_id));

            windows.push(DiarizationWindow {
                span,
                embedding,
                known,
            });
        }

        let unknown_labels = cluster_unknown_windows_with_bounds(
            &windows,
            self.cluster_threshold,
            self.min_speakers,
            self.max_speakers,
        )?;
        let segments = windows
            .into_iter()
            .enumerate()
            .map(|(index, window)| {
                let speaker = if let Some(speaker_id) = window.known {
                    DiarizedSpeaker::Known(speaker_id)
                } else {
                    DiarizedSpeaker::Unknown(
                        unknown_labels[index]
                            .clone()
                            .expect("unknown window has clustered label"),
                    )
                };
                DiarizationSegment {
                    start_seconds: window.span.start_seconds,
                    end_seconds: window.span.end_seconds,
                    speaker,
                    score: window.span.score,
                }
            })
            .collect();

        Ok(DiarizationResult { segments })
    }
}

#[derive(Debug, Clone)]
struct Cluster {
    centroid: SpeakerEmbedding,
    members: Vec<usize>,
}

#[derive(Debug, Clone)]
struct DiarizationWindow {
    span: SpeechSpan,
    embedding: SpeakerEmbedding,
    known: Option<SpeakerId>,
}

fn validate_speaker_bounds(min_speakers: Option<usize>, max_speakers: Option<usize>) -> Result<()> {
    if min_speakers == Some(0) {
        return Err(DetectError::InvalidArgument(
            "speaker diarization min_speakers must be greater than zero".to_string(),
        ));
    }
    if max_speakers == Some(0) {
        return Err(DetectError::InvalidArgument(
            "speaker diarization max_speakers must be greater than zero".to_string(),
        ));
    }
    if let (Some(min), Some(max)) = (min_speakers, max_speakers) {
        if min > max {
            return Err(DetectError::InvalidArgument(
                "speaker diarization min_speakers must be less than or equal to max_speakers"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

fn cluster_unknown_windows_with_bounds(
    windows: &[DiarizationWindow],
    threshold: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
) -> Result<Vec<Option<String>>> {
    validate_speaker_bounds(min_speakers, max_speakers)?;
    let mut clusters: Vec<Cluster> = Vec::new();
    for (index, window) in windows.iter().enumerate() {
        if window.known.is_some() {
            continue;
        }
        assign_unknown_window(&mut clusters, windows, index, threshold, max_speakers)?;
    }

    let target_min = min_speakers.unwrap_or(0).min(
        windows
            .iter()
            .filter(|window| window.known.is_none())
            .count(),
    );
    while clusters.len() < target_min {
        if !split_most_dispersed_cluster(&mut clusters, windows)? {
            break;
        }
    }

    if let Some(max) = max_speakers {
        while clusters.len() > max {
            if !merge_closest_clusters(&mut clusters, windows)? {
                break;
            }
        }
    }

    let mut cluster_for_window = vec![None; windows.len()];
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        for &member in &cluster.members {
            cluster_for_window[member] = Some(cluster_index);
        }
    }

    let mut first_seen_clusters = Vec::new();
    let mut labels = vec![None; windows.len()];
    for (index, window) in windows.iter().enumerate() {
        if window.known.is_some() {
            continue;
        }
        let cluster_index = cluster_for_window[index].expect("unknown window belongs to cluster");
        let label_index = if let Some(existing) = first_seen_clusters
            .iter()
            .position(|value| *value == cluster_index)
        {
            existing
        } else {
            first_seen_clusters.push(cluster_index);
            first_seen_clusters.len() - 1
        };
        labels[index] = Some(format!("speaker_{label_index}"));
    }
    Ok(labels)
}

fn assign_unknown_window(
    clusters: &mut Vec<Cluster>,
    windows: &[DiarizationWindow],
    window_index: usize,
    threshold: f32,
    max_speakers: Option<usize>,
) -> Result<()> {
    let embedding = &windows[window_index].embedding;
    let mut best_index = None;
    let mut best_score = f32::NEG_INFINITY;
    for (index, cluster) in clusters.iter().enumerate() {
        let score = cluster.centroid.cosine_similarity(embedding)?;
        if score > best_score {
            best_score = score;
            best_index = Some(index);
        }
    }
    if best_score >= threshold
        || (max_speakers.is_some_and(|max| clusters.len() >= max) && best_index.is_some())
    {
        let cluster_index = best_index.expect("best index exists when best score is finite");
        clusters[cluster_index].members.push(window_index);
        refresh_centroid(&mut clusters[cluster_index], windows)?;
        return Ok(());
    }

    clusters.push(Cluster {
        centroid: embedding.clone(),
        members: vec![window_index],
    });
    Ok(())
}

fn split_most_dispersed_cluster(
    clusters: &mut Vec<Cluster>,
    windows: &[DiarizationWindow],
) -> Result<bool> {
    let mut selected = None;
    let mut selected_dispersion = f32::NEG_INFINITY;
    let mut selected_member = None;
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        if cluster.members.len() < 2 {
            continue;
        }
        let mut cluster_dispersion = 0.0_f32;
        let mut farthest_member = cluster.members[0];
        let mut farthest_distance = f32::NEG_INFINITY;
        for &member in &cluster.members {
            let similarity = cluster
                .centroid
                .cosine_similarity(&windows[member].embedding)?;
            let distance = 1.0 - similarity;
            cluster_dispersion += distance;
            if distance > farthest_distance {
                farthest_distance = distance;
                farthest_member = member;
            }
        }
        cluster_dispersion /= cluster.members.len() as f32;
        if cluster_dispersion > selected_dispersion {
            selected_dispersion = cluster_dispersion;
            selected = Some(cluster_index);
            selected_member = Some(farthest_member);
        }
    }

    let Some(cluster_index) = selected else {
        return Ok(false);
    };
    let member = selected_member.expect("selected cluster has a farthest member");
    clusters[cluster_index]
        .members
        .retain(|value| *value != member);
    refresh_centroid(&mut clusters[cluster_index], windows)?;
    clusters.push(Cluster {
        centroid: windows[member].embedding.clone(),
        members: vec![member],
    });
    Ok(true)
}

fn merge_closest_clusters(
    clusters: &mut Vec<Cluster>,
    windows: &[DiarizationWindow],
) -> Result<bool> {
    if clusters.len() < 2 {
        return Ok(false);
    }
    let mut best_pair = (0, 1);
    let mut best_score = f32::NEG_INFINITY;
    for left in 0..clusters.len() {
        for right in (left + 1)..clusters.len() {
            let score = clusters[left]
                .centroid
                .cosine_similarity(&clusters[right].centroid)?;
            if score > best_score {
                best_score = score;
                best_pair = (left, right);
            }
        }
    }
    let (left, right) = best_pair;
    let mut right_cluster = clusters.remove(right);
    clusters[left].members.append(&mut right_cluster.members);
    clusters[left].members.sort_unstable();
    refresh_centroid(&mut clusters[left], windows)?;
    Ok(true)
}

fn refresh_centroid(cluster: &mut Cluster, windows: &[DiarizationWindow]) -> Result<()> {
    if cluster.members.len() == 1 {
        cluster.centroid = windows[cluster.members[0]].embedding.clone();
        return Ok(());
    }
    let dimensions = cluster.centroid.dimensions();
    let mut values = vec![0.0_f32; dimensions];
    for &member in &cluster.members {
        for (index, value) in windows[member].embedding.values().iter().enumerate() {
            values[index] += *value;
        }
    }
    for value in &mut values {
        *value /= cluster.members.len() as f32;
    }
    cluster.centroid = SpeakerEmbedding::new(
        values,
        cluster.centroid.model().clone(),
        cluster.centroid.sample_rate(),
    )
    .unwrap_or_else(|_| windows[cluster.members[0]].embedding.clone());
    Ok(())
}

fn best_profile_score(
    profile: &SpeakerProfile,
    embedding: &SpeakerEmbedding,
) -> Result<Option<f32>> {
    let mut best = None;
    for profile_embedding in profile.embeddings() {
        let score = profile_embedding.cosine_similarity(embedding)?;
        best = Some(best.map(|value: f32| value.max(score)).unwrap_or(score));
    }
    Ok(best)
}

fn validate_profile_header(profile: &SpeakerProfile) -> Result<()> {
    if profile.id().as_str().trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "speaker profile id must not be empty".to_string(),
        ));
    }
    if profile.label().as_str().trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "speaker profile label must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_model_compatible(
    expected: &SpeakerEmbeddingModel,
    actual: &SpeakerEmbeddingModel,
) -> Result<()> {
    if expected != actual {
        return Err(DetectError::InvalidArgument(format!(
            "speaker embedding model mismatch: expected {}@{} ({} dims), got {}@{} ({} dims)",
            expected.name,
            expected.version,
            expected.dimensions,
            actual.name,
            actual.version,
            actual.dimensions
        )));
    }
    Ok(())
}

fn resolve_onnx_speaker_model_path(config: &OnnxSpeakerEmbeddingConfig) -> Result<PathBuf> {
    config.validate()?;
    let bundle_path = &config.bundle_path;
    let model_file = config.model_file_name();
    if bundle_path.is_file() {
        if bundle_path.extension().and_then(|value| value.to_str()) == Some("onnx") {
            return Ok(bundle_path.clone());
        }
        return Err(setup_error(format!(
            "ONNX speaker embedding path `{}` is not an .onnx file",
            bundle_path.display()
        )));
    }
    if !bundle_path.exists() {
        return Err(setup_error(format!(
            "ONNX speaker embedding bundle `{}` does not exist",
            bundle_path.display()
        )));
    }
    if !bundle_path.is_dir() {
        return Err(setup_error(format!(
            "ONNX speaker embedding bundle `{}` is not a file or directory",
            bundle_path.display()
        )));
    }

    #[cfg(feature = "model-bundles")]
    {
        let manifest_path = bundle_path.join("manifest.json");
        if manifest_path.is_file() {
            let bundle = model_runtime::ModelBundle::load(&manifest_path).map_err(|error| {
                invalid_request(format!(
                    "failed to parse ONNX speaker embedding bundle manifest `{}`: {error}",
                    manifest_path.display()
                ))
            })?;
            for file in bundle.manifest.files.values() {
                if file.remote_path == model_file
                    || file.remote_path.ends_with(model_file)
                    || file.local_path.ends_with(model_file)
                {
                    if let Some(path) = bundle.file_path(&file.remote_path) {
                        if path.is_file() {
                            return Ok(path);
                        }
                        return Err(setup_error(format!(
                            "ONNX speaker embedding model file `{}` from bundle manifest does not exist",
                            path.display()
                        )));
                    }
                }
            }
            return Err(setup_error(format!(
                "ONNX speaker embedding bundle `{}` is missing model file `{model_file}`",
                bundle_path.display()
            )));
        }
    }

    let model_path = bundle_path.join(model_file);
    if model_path.is_file() {
        Ok(model_path)
    } else {
        Err(setup_error(format!(
            "ONNX speaker embedding model file `{}` does not exist",
            model_path.display()
        )))
    }
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
struct SelectedOnnxIo {
    name: String,
    index: usize,
    dimensions: Vec<runtime_onnx::OnnxDimension>,
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnnxSpeakerInputKind {
    Waveform,
    Feature {
        mel_bins: usize,
        fixed_frames: Option<usize>,
    },
}

#[cfg(feature = "onnx")]
fn select_onnx_speaker_input(
    metadata: &runtime_onnx::OnnxSessionMetadata,
    requested_name: Option<&str>,
) -> Result<SelectedOnnxIo> {
    let (index, input) = select_onnx_io(&metadata.inputs, requested_name, "input")?;
    if input.element_type != Some(runtime_onnx::OnnxTensorElementType::F32) {
        return Err(unsupported_runtime(format!(
            "ONNX speaker embedding input `{}` must be f32",
            input.name
        )));
    }
    if !matches!(input.dimensions.len(), 2 | 3) {
        return Err(unsupported_runtime(format!(
            "ONNX speaker embedding input `{}` must have rank 2 or 3, got rank {}",
            input.name,
            input.dimensions.len()
        )));
    }
    validate_fixed_dimension(&input.dimensions[0], 1, "batch")?;
    let selected = SelectedOnnxIo {
        name: input.name.clone(),
        index,
        dimensions: input.dimensions.clone(),
    };
    let kind = onnx_speaker_input_kind(&selected)?;
    if matches!(kind, OnnxSpeakerInputKind::Waveform) && input.dimensions.len() == 3 {
        validate_fixed_dimension(&input.dimensions[1], 1, "channel")?;
    }
    Ok(SelectedOnnxIo {
        name: input.name.clone(),
        index,
        dimensions: input.dimensions.clone(),
    })
}

#[cfg(feature = "onnx")]
fn select_onnx_speaker_output(
    metadata: &runtime_onnx::OnnxSessionMetadata,
    requested_name: Option<&str>,
) -> Result<SelectedOnnxIo> {
    let (index, output) = select_onnx_io(&metadata.outputs, requested_name, "output")?;
    if output.element_type != Some(runtime_onnx::OnnxTensorElementType::F32) {
        return Err(model_output_mismatch(format!(
            "ONNX speaker embedding output `{}` must be f32",
            output.name
        )));
    }
    if !matches!(output.dimensions.len(), 1 | 2) {
        return Err(model_output_mismatch(format!(
            "ONNX speaker embedding output `{}` must have rank 1 or 2, got rank {}",
            output.name,
            output.dimensions.len()
        )));
    }
    if output.dimensions.len() == 2 {
        validate_output_fixed_dimension(&output.dimensions[0], 1, "batch")?;
    }
    Ok(SelectedOnnxIo {
        name: output.name.clone(),
        index,
        dimensions: output.dimensions.clone(),
    })
}

#[cfg(feature = "onnx")]
fn select_onnx_io<'a>(
    values: &'a [runtime_onnx::OnnxIoInfo],
    requested_name: Option<&str>,
    role: &str,
) -> Result<(usize, &'a runtime_onnx::OnnxIoInfo)> {
    if let Some(name) = requested_name {
        return values
            .iter()
            .enumerate()
            .find(|(_, value)| value.name == name)
            .ok_or_else(|| {
                unsupported_runtime(format!("ONNX speaker embedding {role} `{name}` is missing"))
            });
    }
    if values.len() != 1 {
        return Err(unsupported_runtime(format!(
            "ONNX speaker embedding requires exactly one {role} when no {role}_name is configured, got {}",
            values.len()
        )));
    }
    Ok((0, &values[0]))
}

#[cfg(feature = "onnx")]
fn validate_fixed_dimension(
    dimension: &runtime_onnx::OnnxDimension,
    expected: usize,
    role: &str,
) -> Result<()> {
    if matches!(dimension, runtime_onnx::OnnxDimension::Fixed(value) if *value != expected) {
        return Err(unsupported_runtime(format!(
            "ONNX speaker embedding fixed {role} dimension must be {expected}"
        )));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn validate_output_fixed_dimension(
    dimension: &runtime_onnx::OnnxDimension,
    expected: usize,
    role: &str,
) -> Result<()> {
    if matches!(dimension, runtime_onnx::OnnxDimension::Fixed(value) if *value != expected) {
        return Err(model_output_mismatch(format!(
            "ONNX speaker embedding fixed output {role} dimension must be {expected}"
        )));
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn format_onnx_dimensions(dimensions: &[runtime_onnx::OnnxDimension]) -> String {
    let values = dimensions
        .iter()
        .map(|dimension| match dimension {
            runtime_onnx::OnnxDimension::Fixed(value) => value.to_string(),
            runtime_onnx::OnnxDimension::Symbolic(value) => value.clone(),
            runtime_onnx::OnnxDimension::Unknown => "unknown".to_string(),
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{values}]")
}

#[cfg(feature = "onnx")]
fn onnx_speaker_input_kind(input: &SelectedOnnxIo) -> Result<OnnxSpeakerInputKind> {
    match input.dimensions.as_slice() {
        [_, _] => Ok(OnnxSpeakerInputKind::Waveform),
        [_, runtime_onnx::OnnxDimension::Fixed(1), _] => {
            Ok(OnnxSpeakerInputKind::Waveform)
        }
        [_, frames, bins] => match bins {
            runtime_onnx::OnnxDimension::Fixed(mel_bins) if *mel_bins > 1 && *mel_bins <= 512 => {
                let fixed_frames = match frames {
                    runtime_onnx::OnnxDimension::Fixed(value) => {
                        if *value == 0 {
                            return Err(unsupported_runtime(
                                "ONNX speaker embedding fixed feature frame dimension must be positive",
                            ));
                        }
                        Some(*value)
                    }
                    runtime_onnx::OnnxDimension::Symbolic(_)
                    | runtime_onnx::OnnxDimension::Unknown => None,
                };
                Ok(OnnxSpeakerInputKind::Feature {
                    mel_bins: *mel_bins,
                    fixed_frames,
                })
            }
            _ => Err(unsupported_runtime(format!(
                "ONNX speaker embedding input `{}` rank 3 dimensions {} must be waveform [batch,1,samples] or feature [batch,frames,mel_bins]",
                input.name,
                format_onnx_dimensions(&input.dimensions)
            ))),
        },
        _ => Err(unsupported_runtime(format!(
            "ONNX speaker embedding input `{}` must have rank 2 or 3",
            input.name
        ))),
    }
}

#[cfg(feature = "onnx")]
fn onnx_speaker_input_tensor(
    input: &SelectedOnnxIo,
    mono: &[f32],
    sample_rate: u32,
) -> Result<(Vec<usize>, Vec<f32>, Vec<String>)> {
    match onnx_speaker_input_kind(input)? {
        OnnxSpeakerInputKind::Waveform => onnx_speaker_waveform_input_tensor(input, mono),
        OnnxSpeakerInputKind::Feature {
            mel_bins,
            fixed_frames,
        } => onnx_speaker_feature_input_tensor(mono, sample_rate, mel_bins, fixed_frames),
    }
}

#[cfg(feature = "onnx")]
fn onnx_speaker_waveform_input_tensor(
    input: &SelectedOnnxIo,
    mono: &[f32],
) -> Result<(Vec<usize>, Vec<f32>, Vec<String>)> {
    let sample_dimension = input.dimensions.last().ok_or_else(|| {
        unsupported_runtime("ONNX speaker embedding input is missing sample dimension")
    })?;
    let samples = match sample_dimension {
        runtime_onnx::OnnxDimension::Fixed(value) => *value,
        runtime_onnx::OnnxDimension::Symbolic(_) | runtime_onnx::OnnxDimension::Unknown => {
            mono.len()
        }
    };
    let mut values = vec![0.0; samples];
    let copy_len = mono.len().min(samples);
    values[..copy_len].copy_from_slice(&mono[..copy_len]);
    let shape = if input.dimensions.len() == 2 {
        vec![1, samples]
    } else {
        vec![1, 1, samples]
    };
    Ok((
        shape,
        values,
        vec!["speakerEmbeddingInputKind=waveform".to_string()],
    ))
}

#[cfg(feature = "onnx")]
fn onnx_speaker_feature_input_tensor(
    mono: &[f32],
    sample_rate: u32,
    mel_bins: usize,
    fixed_frames: Option<usize>,
) -> Result<(Vec<usize>, Vec<f32>, Vec<String>)> {
    let (frame_count, values) = speaker_logmel_features(mono, sample_rate, mel_bins, fixed_frames)?;
    Ok((
        vec![1, frame_count, mel_bins],
        values,
        vec![
            "speakerEmbeddingInputKind=feature".to_string(),
            "speakerFeatureExtractor=fbank-logmel-cpu".to_string(),
            format!("speakerFeatureSampleRate={SPEAKER_FEATURE_SAMPLE_RATE}"),
            format!("speakerFeatureFrameLengthSamples={SPEAKER_FEATURE_FRAME_LENGTH}"),
            format!("speakerFeatureHopSamples={SPEAKER_FEATURE_HOP}"),
            format!("speakerFeatureFftSize={SPEAKER_FEATURE_FFT_SIZE}"),
            format!("speakerFeatureBins={mel_bins}"),
            format!("speakerFeatureFrameCount={frame_count}"),
            "speakerFeatureCmvn=mean".to_string(),
        ],
    ))
}

#[cfg(feature = "onnx")]
const SPEAKER_FEATURE_SAMPLE_RATE: u32 = 16_000;
#[cfg(feature = "onnx")]
const SPEAKER_FEATURE_FRAME_LENGTH: usize = 400;
#[cfg(feature = "onnx")]
const SPEAKER_FEATURE_HOP: usize = 160;
#[cfg(feature = "onnx")]
const SPEAKER_FEATURE_FFT_SIZE: usize = 512;
#[cfg(feature = "onnx")]
const SPEAKER_FEATURE_PREEMPHASIS: f32 = 0.97;
#[cfg(feature = "onnx")]
const SPEAKER_FEATURE_LOWER_HZ: f32 = 20.0;
#[cfg(feature = "onnx")]
const SPEAKER_FEATURE_LOG_FLOOR: f32 = 1.0e-10;

#[cfg(feature = "onnx")]
fn speaker_logmel_features(
    mono: &[f32],
    sample_rate: u32,
    mel_bins: usize,
    fixed_frames: Option<usize>,
) -> Result<(usize, Vec<f32>)> {
    if sample_rate != SPEAKER_FEATURE_SAMPLE_RATE {
        return Err(invalid_request(format!(
            "ONNX feature speaker embedding requires {SPEAKER_FEATURE_SAMPLE_RATE} Hz audio, got {sample_rate} Hz"
        )));
    }
    if mel_bins == 0 {
        return Err(unsupported_runtime(
            "ONNX feature speaker embedding mel bin count must be positive",
        ));
    }
    let mut frames = logmel_frames(mono, mel_bins)?;
    let target_frames = fixed_frames.unwrap_or(frames.len()).max(1);
    if frames.len() > target_frames {
        frames.truncate(target_frames);
    }
    mean_normalize_frames(&mut frames, mel_bins);
    while frames.len() < target_frames {
        frames.push(vec![0.0; mel_bins]);
    }
    let values = frames.into_iter().flatten().collect::<Vec<_>>();
    Ok((target_frames, values))
}

#[cfg(feature = "onnx")]
fn logmel_frames(mono: &[f32], mel_bins: usize) -> Result<Vec<Vec<f32>>> {
    let filters = mel_filterbank(
        mel_bins,
        SPEAKER_FEATURE_FFT_SIZE,
        SPEAKER_FEATURE_SAMPLE_RATE,
        SPEAKER_FEATURE_LOWER_HZ,
        SPEAKER_FEATURE_SAMPLE_RATE as f32 / 2.0,
    );
    let window = hamming_window(SPEAKER_FEATURE_FRAME_LENGTH);
    let frame_starts = if mono.len() <= SPEAKER_FEATURE_FRAME_LENGTH {
        vec![0]
    } else {
        (0..=(mono.len() - SPEAKER_FEATURE_FRAME_LENGTH))
            .step_by(SPEAKER_FEATURE_HOP)
            .collect::<Vec<_>>()
    };
    let mut frames = Vec::with_capacity(frame_starts.len().max(1));
    for start in frame_starts {
        let mut frame = vec![0.0_f32; SPEAKER_FEATURE_FFT_SIZE];
        for index in 0..SPEAKER_FEATURE_FRAME_LENGTH {
            let sample_index = start + index;
            let current = mono.get(sample_index).copied().unwrap_or(0.0);
            let previous = if sample_index == 0 {
                0.0
            } else {
                mono.get(sample_index - 1).copied().unwrap_or(0.0)
            };
            frame[index] = (current - SPEAKER_FEATURE_PREEMPHASIS * previous) * window[index];
        }
        let power = power_spectrum(&frame);
        let mut mel = Vec::with_capacity(mel_bins);
        for filter in &filters {
            let energy = filter
                .iter()
                .enumerate()
                .map(|(bin, weight)| power[bin] * weight)
                .sum::<f32>()
                .max(SPEAKER_FEATURE_LOG_FLOOR);
            mel.push(energy.ln());
        }
        frames.push(mel);
    }
    if frames.is_empty() {
        frames.push(vec![SPEAKER_FEATURE_LOG_FLOOR.ln(); mel_bins]);
    }
    Ok(frames)
}

#[cfg(feature = "onnx")]
fn hamming_window(length: usize) -> Vec<f32> {
    if length <= 1 {
        return vec![1.0; length];
    }
    (0..length)
        .map(|index| {
            0.54 - 0.46 * ((2.0 * std::f32::consts::PI * index as f32) / (length - 1) as f32).cos()
        })
        .collect()
}

#[cfg(feature = "onnx")]
fn power_spectrum(frame: &[f32]) -> Vec<f32> {
    let bins = SPEAKER_FEATURE_FFT_SIZE / 2 + 1;
    let mut power = vec![0.0_f32; bins];
    for (bin, value) in power.iter_mut().enumerate() {
        let mut real = 0.0_f32;
        let mut imag = 0.0_f32;
        for (index, sample) in frame.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * bin as f32 * index as f32
                / SPEAKER_FEATURE_FFT_SIZE as f32;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }
        *value = real.mul_add(real, imag * imag);
    }
    power
}

#[cfg(feature = "onnx")]
fn mel_filterbank(
    mel_bins: usize,
    fft_size: usize,
    sample_rate: u32,
    lower_hz: f32,
    upper_hz: f32,
) -> Vec<Vec<f32>> {
    let lower_mel = hz_to_mel(lower_hz);
    let upper_mel = hz_to_mel(upper_hz);
    let mel_points = (0..(mel_bins + 2))
        .map(|index| lower_mel + (upper_mel - lower_mel) * index as f32 / (mel_bins + 1) as f32)
        .map(mel_to_hz)
        .collect::<Vec<_>>();
    let fft_bins = mel_points
        .iter()
        .map(|hz| {
            (((fft_size + 1) as f32 * hz / sample_rate as f32).floor() as usize).min(fft_size / 2)
        })
        .collect::<Vec<_>>();
    let mut filters = vec![vec![0.0_f32; fft_size / 2 + 1]; mel_bins];
    for mel in 0..mel_bins {
        let left = fft_bins[mel];
        let center = fft_bins[mel + 1].max(left + 1);
        let right = fft_bins[mel + 2].max(center + 1);
        for bin in left..center.min(filters[mel].len()) {
            filters[mel][bin] = (bin - left) as f32 / (center - left) as f32;
        }
        for bin in center..right.min(filters[mel].len()) {
            filters[mel][bin] = (right - bin) as f32 / (right - center) as f32;
        }
    }
    filters
}

#[cfg(feature = "onnx")]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

#[cfg(feature = "onnx")]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(feature = "onnx")]
fn mean_normalize_frames(frames: &mut [Vec<f32>], mel_bins: usize) {
    if frames.is_empty() {
        return;
    }
    let mut means = vec![0.0_f32; mel_bins];
    for frame in frames.iter() {
        for (index, value) in frame.iter().enumerate().take(mel_bins) {
            means[index] += *value;
        }
    }
    for mean in &mut means {
        *mean /= frames.len() as f32;
    }
    for frame in frames {
        for (index, value) in frame.iter_mut().enumerate().take(mel_bins) {
            *value -= means[index];
        }
    }
}

#[cfg(feature = "onnx")]
fn onnx_speaker_embedding_values(
    tensor: &runtime_onnx::OnnxF32Tensor,
    embedding_dimension: usize,
) -> Result<Vec<f32>> {
    match tensor.shape.as_slice() {
        [dim] if *dim == embedding_dimension => Ok(tensor.values.clone()),
        [1, dim] if *dim == embedding_dimension => Ok(tensor.values.clone()),
        shape => Err(model_output_mismatch(format!(
            "ONNX speaker embedding output shape {shape:?} does not match expected embedding dimension {embedding_dimension}"
        ))),
    }
}

#[cfg(feature = "onnx")]
fn map_onnx_session_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::Unavailable => {
            unsupported_runtime("ONNX speaker embedding runtime is unavailable")
        }
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message)
            if message.contains("does not exist") =>
        {
            setup_error(message)
        }
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message) => invalid_request(message),
        runtime_onnx::OnnxRuntimeError::Io(error) => setup_error(error.to_string()),
        other => unsupported_runtime(other.to_string()),
    }
}

#[cfg(feature = "onnx")]
fn map_onnx_runtime_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::Unavailable => {
            unsupported_runtime("ONNX speaker embedding runtime is unavailable")
        }
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message) => invalid_request(message),
        runtime_onnx::OnnxRuntimeError::InvalidTensorShape(message)
        | runtime_onnx::OnnxRuntimeError::UnsupportedTensorType(message)
        | runtime_onnx::OnnxRuntimeError::Source(message) => model_output_mismatch(message),
        runtime_onnx::OnnxRuntimeError::Io(error) => setup_error(error.to_string()),
    }
}

fn setup_error(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("setup_error: {}", message.into()))
}

fn invalid_request(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("invalid_request: {}", message.into()))
}

fn unsupported_runtime(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("unsupported_runtime: {}", message.into()))
}

#[cfg(feature = "onnx")]
fn model_output_mismatch(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("model_output_mismatch: {}", message.into()))
}

fn confidence_for(score: f32, margin: Option<f32>) -> SpeakerConfidence {
    if score >= 0.90 && margin.is_none_or(|value| value >= 0.10) {
        SpeakerConfidence::High
    } else if score >= 0.82 && margin.is_none_or(|value| value >= 0.05) {
        SpeakerConfidence::Medium
    } else {
        SpeakerConfidence::Low
    }
}

fn normalize_values(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Err(DetectError::InvalidArgument(
            "speaker embedding must not be empty".to_string(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "speaker embedding values must be finite".to_string(),
        ));
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::EPSILON || !norm.is_finite() {
        return Err(DetectError::InvalidArgument(
            "speaker embedding norm must be finite and non-zero".to_string(),
        ));
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}

fn seconds_to_samples(seconds: f64, sample_rate: u32) -> Result<usize> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(DetectError::InvalidArgument(
            "duration must be finite and positive".to_string(),
        ));
    }
    Ok(((seconds * sample_rate as f64).round() as usize).max(1))
}

fn merge_spans(spans: Vec<SpeechSpan>, merge_gap_seconds: f64) -> Result<Vec<SpeechSpan>> {
    let mut merged: Vec<SpeechSpan> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut() {
            if span.start_seconds - last.end_seconds <= merge_gap_seconds {
                let total_duration = last.duration_seconds() + span.duration_seconds();
                let score = if total_duration <= f64::EPSILON {
                    last.score.max(span.score) as f64
                } else {
                    ((last.score as f64 * last.duration_seconds())
                        + (span.score as f64 * span.duration_seconds()))
                        / total_duration
                } as f32;
                last.end_seconds = span.end_seconds;
                last.score = score;
                last.validate()?;
                continue;
            }
        }
        merged.push(span);
    }
    Ok(merged)
}

fn filter_short_spans(spans: Vec<SpeechSpan>, min_speech_seconds: f64) -> Vec<SpeechSpan> {
    spans
        .into_iter()
        .filter(|span| span.duration_seconds() >= min_speech_seconds)
        .collect()
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
    let vad_config = EnergyVadConfig::default();
    let vad = EnergyVoiceActivityDetector::new(vad_config)?;
    let merge_gap_seconds = vad_config.merge_gap_seconds as f32;
    let mut diarizer = WindowedSpeakerDiarizer::new(embedder, vad).cluster_threshold(0.95)?;
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

fn merge_speaker_predictions(
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

/// Assigns speaker diarization output onto transcript segments by time overlap.
pub fn assign_speakers_to_transcript(
    transcript: &TranscriptionContract,
    diarization: &SpeakerDiarizationResponse,
) -> Result<TranscriptionContract> {
    assign_speakers_to_transcript_with_policy(
        transcript,
        diarization,
        SpeakerTranscriptAssignmentPolicy::Majority,
    )
}

/// Assigns speaker diarization output onto transcript words and segments using an explicit policy.
pub fn assign_speakers_to_transcript_with_policy(
    transcript: &TranscriptionContract,
    diarization: &SpeakerDiarizationResponse,
    policy: SpeakerTranscriptAssignmentPolicy,
) -> Result<TranscriptionContract> {
    let mut transcript = transcript
        .clone()
        .normalized()
        .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
    let diarization_segments = diarization
        .segments
        .iter()
        .cloned()
        .map(normalize_speaker_segment)
        .collect::<Result<Vec<_>>>()?;

    for segment in &mut transcript.segments {
        for word in &mut segment.words {
            let Some((start, end)) = word.start_seconds.zip(word.end_seconds) else {
                continue;
            };
            word.speaker = speaker_for_range(start, end, &diarization_segments, policy)
                .map(|speaker_segment| speaker_segment.speaker.clone());
        }
        if segment.speaker.is_some() {
            continue;
        }
        if !segment.words.is_empty() {
            segment.speaker = majority_word_speaker(segment);
            continue;
        }
        let Some(best) = best_speaker_match(segment, &diarization_segments, policy) else {
            if policy == SpeakerTranscriptAssignmentPolicy::StrictContained
                && segment.speaker.is_none()
            {
                segment.speaker = Some("unknown".to_string());
            }
            continue;
        };
        segment.speaker = Some(best.speaker.clone());
        if let Some(score) = best.score {
            segment
                .attributes
                .insert("speakerScore".to_string(), score.to_string());
        }
    }

    transcript
        .validate_strict()
        .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
    Ok(transcript)
}

fn majority_word_speaker(segment: &TranscriptSegmentContract) -> Option<String> {
    let mut counts = BTreeMap::<String, usize>::new();
    for speaker in segment
        .words
        .iter()
        .filter_map(|word| word.speaker.as_ref())
    {
        *counts.entry(speaker.clone()).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(speaker, _)| speaker)
}

fn normalize_speaker_segment(
    mut segment: SpeakerSegmentPrediction,
) -> Result<SpeakerSegmentPrediction> {
    segment.speaker = segment.speaker.trim().to_string();
    if segment.speaker.is_empty() {
        return Err(DetectError::InvalidArgument(
            "speaker label must not be empty".to_string(),
        ));
    }
    if !segment.start_seconds.is_finite() || !segment.end_seconds.is_finite() {
        return Err(DetectError::InvalidArgument(
            "speaker segment timestamps must be finite".to_string(),
        ));
    }
    if segment.end_seconds < segment.start_seconds {
        return Err(DetectError::InvalidArgument(
            "speaker segment end_seconds must be greater than or equal to start_seconds"
                .to_string(),
        ));
    }
    segment.score = segment
        .score
        .and_then(|score| score.is_finite().then(|| score.clamp(0.0, 1.0)));
    Ok(segment)
}

fn best_speaker_match<'a>(
    transcript_segment: &TranscriptSegmentContract,
    diarization_segments: &'a [SpeakerSegmentPrediction],
    policy: SpeakerTranscriptAssignmentPolicy,
) -> Option<&'a SpeakerSegmentPrediction> {
    let (Some(start), Some(end)) = (
        transcript_segment.start_seconds,
        transcript_segment.end_seconds,
    ) else {
        return match policy {
            SpeakerTranscriptAssignmentPolicy::NearestStart => {
                let start = transcript_segment.midpoint_seconds()?;
                speaker_for_range(start, start, diarization_segments, policy)
            }
            _ => None,
        };
    };
    speaker_for_range(start, end, diarization_segments, policy)
}

fn speaker_for_range(
    start: f64,
    end: f64,
    diarization_segments: &[SpeakerSegmentPrediction],
    policy: SpeakerTranscriptAssignmentPolicy,
) -> Option<&SpeakerSegmentPrediction> {
    match policy {
        SpeakerTranscriptAssignmentPolicy::Majority => {
            best_majority_speaker_match(start, end, (start + end) / 2.0, diarization_segments)
        }
        SpeakerTranscriptAssignmentPolicy::NearestStart => {
            nearest_start_speaker_match(start, diarization_segments)
        }
        SpeakerTranscriptAssignmentPolicy::StrictContained => {
            strict_contained_speaker_match(start, end, diarization_segments)
        }
    }
}

fn best_majority_speaker_match(
    transcript_start: f64,
    transcript_end: f64,
    transcript_midpoint: f64,
    diarization_segments: &[SpeakerSegmentPrediction],
) -> Option<&SpeakerSegmentPrediction> {
    let positive_overlap = diarization_segments
        .iter()
        .filter_map(|candidate| {
            let overlap = speaker_overlap_seconds(transcript_start, transcript_end, candidate);
            (overlap > 0.0).then_some((candidate, overlap))
        })
        .max_by(compare_speaker_candidates);
    if let Some((candidate, _)) = positive_overlap {
        return Some(candidate);
    }

    diarization_segments
        .iter()
        .filter(|candidate| {
            let start = candidate.start_seconds as f64;
            let end = candidate.end_seconds as f64;
            start <= transcript_midpoint && transcript_midpoint <= end
        })
        .map(|candidate| (candidate, 0.0))
        .max_by(compare_speaker_candidates)
        .map(|(candidate, _)| candidate)
}

fn nearest_start_speaker_match(
    transcript_start: f64,
    diarization_segments: &[SpeakerSegmentPrediction],
) -> Option<&SpeakerSegmentPrediction> {
    diarization_segments
        .iter()
        .map(|candidate| {
            (
                candidate,
                (candidate.start_seconds as f64 - transcript_start).abs(),
            )
        })
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| speaker_score(right.0).total_cmp(&speaker_score(left.0)))
                .then_with(|| left.0.start_seconds.total_cmp(&right.0.start_seconds))
                .then_with(|| left.0.speaker.cmp(&right.0.speaker))
        })
        .map(|(candidate, _)| candidate)
}

fn strict_contained_speaker_match(
    transcript_start: f64,
    transcript_end: f64,
    diarization_segments: &[SpeakerSegmentPrediction],
) -> Option<&SpeakerSegmentPrediction> {
    diarization_segments
        .iter()
        .filter(|candidate| {
            candidate.start_seconds as f64 <= transcript_start
                && candidate.end_seconds as f64 >= transcript_end
        })
        .map(|candidate| (candidate, (transcript_end - transcript_start).max(0.0)))
        .max_by(compare_speaker_candidates)
        .map(|(candidate, _)| candidate)
}

fn speaker_overlap_seconds(
    transcript_start: f64,
    transcript_end: f64,
    speaker_segment: &SpeakerSegmentPrediction,
) -> f64 {
    let start = transcript_start.max(speaker_segment.start_seconds as f64);
    let end = transcript_end.min(speaker_segment.end_seconds as f64);
    (end - start).max(0.0)
}

fn compare_speaker_candidates(
    left: &(&SpeakerSegmentPrediction, f64),
    right: &(&SpeakerSegmentPrediction, f64),
) -> std::cmp::Ordering {
    left.1
        .total_cmp(&right.1)
        .then_with(|| speaker_score(left.0).total_cmp(&speaker_score(right.0)))
        .then_with(|| right.0.start_seconds.total_cmp(&left.0.start_seconds))
        .then_with(|| right.0.speaker.cmp(&left.0.speaker))
}

fn speaker_score(segment: &SpeakerSegmentPrediction) -> f32 {
    segment.score.unwrap_or(f32::NEG_INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_diarization_returns_imported_segments() {
        let response = diarize_speakers(SpeakerDiarizationRequest {
            source: None,
            duration_seconds: None,
            model: AudioRuntimeSelection::default(),
            imported_segments: vec![SpeakerSegmentPrediction {
                speaker: "speaker_a".to_string(),
                start_seconds: 0.0,
                end_seconds: 1.5,
                score: Some(0.8),
            }],
        })
        .unwrap();

        assert_eq!(response.runtime, AudioRuntime::Imported);
        assert_eq!(response.segments[0].speaker, "speaker_a");
    }

    #[test]
    fn task_diarization_heuristic_returns_single_segment() {
        let response = diarize_speakers(SpeakerDiarizationRequest {
            source: None,
            duration_seconds: Some(2.5),
            model: AudioRuntimeSelection {
                fallback_policy: FallbackPolicy::HeuristicFallback,
                ..AudioRuntimeSelection::default()
            },
            imported_segments: Vec::new(),
        })
        .unwrap();

        assert_eq!(response.runtime, AudioRuntime::Heuristic);
        assert_eq!(response.segments[0].speaker, "speaker_0");
        assert_eq!(response.segments[0].end_seconds, 2.5);
    }

    #[test]
    fn task_diarization_rejects_invalid_imported_segments() {
        let empty_speaker = diarize_speakers(SpeakerDiarizationRequest {
            source: None,
            duration_seconds: None,
            model: AudioRuntimeSelection::default(),
            imported_segments: vec![SpeakerSegmentPrediction {
                speaker: "   ".to_string(),
                start_seconds: 0.0,
                end_seconds: 1.0,
                score: Some(0.8),
            }],
        });
        assert!(empty_speaker.is_err());

        let reversed = diarize_speakers(SpeakerDiarizationRequest {
            source: None,
            duration_seconds: None,
            model: AudioRuntimeSelection::default(),
            imported_segments: vec![SpeakerSegmentPrediction {
                speaker: "speaker_a".to_string(),
                start_seconds: 2.0,
                end_seconds: 1.0,
                score: Some(0.8),
            }],
        });
        assert!(reversed.is_err());
    }

    #[test]
    fn task_diarization_clamps_imported_scores() {
        let response = diarize_speakers(SpeakerDiarizationRequest {
            source: None,
            duration_seconds: None,
            model: AudioRuntimeSelection::default(),
            imported_segments: vec![SpeakerSegmentPrediction {
                speaker: " speaker_a ".to_string(),
                start_seconds: 0.0,
                end_seconds: 1.0,
                score: Some(2.0),
            }],
        })
        .unwrap();

        assert_eq!(response.segments[0].speaker, "speaker_a");
        assert_eq!(response.segments[0].score, Some(1.0));
    }

    #[test]
    fn assigns_speakers_to_transcript_by_overlap() {
        let mut first = TranscriptSegmentContract::new(0, "hello");
        first.start_seconds = Some(0.0);
        first.end_seconds = Some(1.0);
        let mut second = TranscriptSegmentContract::new(1, "world");
        second.start_seconds = Some(1.0);
        second.end_seconds = Some(2.0);
        let transcript = TranscriptionContract::new(vec![first, second]);
        let diarization = SpeakerDiarizationResponse {
            accepted: true,
            operation: "diarize".to_string(),
            model_id: "fixture".to_string(),
            runtime: AudioRuntime::Imported,
            segments: vec![
                SpeakerSegmentPrediction {
                    speaker: "speaker_b".to_string(),
                    start_seconds: 0.5,
                    end_seconds: 2.0,
                    score: Some(0.7),
                },
                SpeakerSegmentPrediction {
                    speaker: "speaker_a".to_string(),
                    start_seconds: 0.0,
                    end_seconds: 1.1,
                    score: Some(0.9),
                },
            ],
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        };

        let assigned = assign_speakers_to_transcript(&transcript, &diarization).unwrap();

        assert_eq!(assigned.segments[0].speaker.as_deref(), Some("speaker_a"));
        assert_eq!(assigned.segments[1].speaker.as_deref(), Some("speaker_b"));
        assert_eq!(
            assigned.segments[0]
                .attributes
                .get("speakerScore")
                .map(String::as_str),
            Some("0.9")
        );
    }

    #[test]
    fn speaker_assignment_preserves_existing_speaker_without_match() {
        let mut segment = TranscriptSegmentContract::new(0, "hello");
        segment.start_seconds = Some(0.0);
        segment.end_seconds = Some(1.0);
        segment.speaker = Some("original".to_string());
        let transcript = TranscriptionContract::new(vec![segment]);
        let diarization = SpeakerDiarizationResponse {
            accepted: true,
            operation: "diarize".to_string(),
            model_id: "fixture".to_string(),
            runtime: AudioRuntime::Imported,
            segments: vec![SpeakerSegmentPrediction {
                speaker: "other".to_string(),
                start_seconds: 3.0,
                end_seconds: 4.0,
                score: Some(1.0),
            }],
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        };

        let assigned = assign_speakers_to_transcript(&transcript, &diarization).unwrap();

        assert_eq!(assigned.segments[0].speaker.as_deref(), Some("original"));
        assert!(!assigned.segments[0].attributes.contains_key("speakerScore"));
    }

    #[test]
    fn speaker_assignment_supports_nearest_start_policy() {
        let mut segment = TranscriptSegmentContract::new(0, "hello");
        segment.start_seconds = Some(5.0);
        segment.end_seconds = Some(6.0);
        let transcript = TranscriptionContract::new(vec![segment]);
        let diarization = SpeakerDiarizationResponse {
            accepted: true,
            operation: "diarize".to_string(),
            model_id: "fixture".to_string(),
            runtime: AudioRuntime::Imported,
            segments: vec![
                SpeakerSegmentPrediction {
                    speaker: "speaker_far".to_string(),
                    start_seconds: 0.0,
                    end_seconds: 1.0,
                    score: Some(1.0),
                },
                SpeakerSegmentPrediction {
                    speaker: "speaker_near".to_string(),
                    start_seconds: 4.9,
                    end_seconds: 5.1,
                    score: Some(0.1),
                },
            ],
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        };

        let assigned = assign_speakers_to_transcript_with_policy(
            &transcript,
            &diarization,
            SpeakerTranscriptAssignmentPolicy::NearestStart,
        )
        .unwrap();

        assert_eq!(
            assigned.segments[0].speaker.as_deref(),
            Some("speaker_near")
        );
    }

    #[test]
    fn speaker_assignment_strict_contained_uses_unknown_without_match() {
        let mut segment = TranscriptSegmentContract::new(0, "hello");
        segment.start_seconds = Some(0.0);
        segment.end_seconds = Some(2.0);
        let transcript = TranscriptionContract::new(vec![segment]);
        let diarization = SpeakerDiarizationResponse {
            accepted: true,
            operation: "diarize".to_string(),
            model_id: "fixture".to_string(),
            runtime: AudioRuntime::Imported,
            segments: vec![SpeakerSegmentPrediction {
                speaker: "partial".to_string(),
                start_seconds: 0.5,
                end_seconds: 1.5,
                score: Some(1.0),
            }],
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        };

        let assigned = assign_speakers_to_transcript_with_policy(
            &transcript,
            &diarization,
            SpeakerTranscriptAssignmentPolicy::StrictContained,
        )
        .unwrap();

        assert_eq!(assigned.segments[0].speaker.as_deref(), Some("unknown"));
    }

    #[test]
    fn speaker_diarization_options_accept_flat_camel_case_json() {
        let options: SpeakerDiarizationOptions = serde_json::from_value(serde_json::json!({
            "modelId": "pyannote-community-1",
            "pyannoteModelBundle": "/models/pyannote",
            "pyannoteManifestFile": "manifest.json",
            "returnSpeakerEmbeddings": true,
            "minSpeakers": 1,
            "maxSpeakers": 3,
            "assignmentPolicy": "nearestStart"
        }))
        .unwrap();

        assert_eq!(options.model_id, "pyannote-community-1");
        assert!(options.is_pyannote_model());
        assert_eq!(
            options.pyannote_model_bundle.as_deref(),
            Some(Path::new("/models/pyannote"))
        );
        assert_eq!(
            options.pyannote_manifest_file.as_deref(),
            Some("manifest.json")
        );
        assert!(options.return_speaker_embeddings);
        assert_eq!(options.min_speakers, Some(1));
        assert_eq!(options.max_speakers, Some(3));
        assert_eq!(
            options.assignment_policy,
            SpeakerTranscriptAssignmentPolicy::NearestStart
        );
        options.validate().unwrap();
    }

    #[test]
    fn speaker_diarization_options_validate_owned_bounds_and_model_options() {
        let invalid_bounds = SpeakerDiarizationOptions {
            min_speakers: Some(3),
            max_speakers: Some(2),
            ..SpeakerDiarizationOptions::default()
        };
        assert!(invalid_bounds
            .validate()
            .unwrap_err()
            .to_string()
            .contains("invalid_request"));

        let invalid_embedding_dimension = SpeakerDiarizationOptions {
            speaker_embedding_dimension: Some(0),
            ..SpeakerDiarizationOptions::default()
        };
        assert!(invalid_embedding_dimension
            .validate()
            .unwrap_err()
            .to_string()
            .contains("speaker_embedding_dimension"));

        let pyannote_with_onnx_bundle = SpeakerDiarizationOptions {
            model_id: "pyannote/speaker-diarization-community-1".to_string(),
            speaker_embedding_model_bundle: Some(PathBuf::from("speaker-model")),
            ..SpeakerDiarizationOptions::default()
        };
        assert!(pyannote_with_onnx_bundle
            .validate()
            .unwrap_err()
            .to_string()
            .contains("pyannote_model_bundle"));
    }

    #[test]
    fn speaker_assignment_assigns_words_and_majority_segment_speaker() {
        let mut segment = TranscriptSegmentContract::new(0, "one two three");
        segment.start_seconds = Some(0.0);
        segment.end_seconds = Some(0.9);
        for (text, start, end) in [("one", 0.0, 0.3), ("two", 0.3, 0.6), ("three", 0.6, 0.9)] {
            segment.words.push(
                serde_json::from_value(serde_json::json!({
                    "text": text,
                    "startSeconds": start,
                    "endSeconds": end
                }))
                .unwrap(),
            );
        }
        let transcript = TranscriptionContract::new(vec![segment]);
        let diarization = SpeakerDiarizationResponse {
            accepted: true,
            operation: "diarize".to_string(),
            model_id: "fixture".to_string(),
            runtime: AudioRuntime::Imported,
            segments: vec![
                SpeakerSegmentPrediction {
                    speaker: "speaker_0".to_string(),
                    start_seconds: 0.0,
                    end_seconds: 0.65,
                    score: Some(0.9),
                },
                SpeakerSegmentPrediction {
                    speaker: "speaker_1".to_string(),
                    start_seconds: 0.65,
                    end_seconds: 1.0,
                    score: Some(0.9),
                },
            ],
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        };

        let assigned = assign_speakers_to_transcript_with_policy(
            &transcript,
            &diarization,
            SpeakerTranscriptAssignmentPolicy::Majority,
        )
        .unwrap();

        assert_eq!(
            assigned.segments[0].words[0].speaker.as_deref(),
            Some("speaker_0")
        );
        assert_eq!(
            assigned.segments[0].words[1].speaker.as_deref(),
            Some("speaker_0")
        );
        assert_eq!(
            assigned.segments[0].words[2].speaker.as_deref(),
            Some("speaker_1")
        );
        assert_eq!(assigned.segments[0].speaker.as_deref(), Some("speaker_0"));
    }

    #[test]
    fn speaker_assignment_word_policy_variants_match_transcription_contract() {
        let mut segment = TranscriptSegmentContract::new(0, "hello");
        segment.start_seconds = Some(0.2);
        segment.end_seconds = Some(0.8);
        segment.words.push(
            serde_json::from_value(serde_json::json!({
                "text": "hello",
                "startSeconds": 0.2,
                "endSeconds": 0.8
            }))
            .unwrap(),
        );
        let transcript = TranscriptionContract::new(vec![segment]);
        let diarization = SpeakerDiarizationResponse {
            accepted: true,
            operation: "diarize".to_string(),
            model_id: "fixture".to_string(),
            runtime: AudioRuntime::Imported,
            segments: vec![
                SpeakerSegmentPrediction {
                    speaker: "speaker_far".to_string(),
                    start_seconds: 0.0,
                    end_seconds: 0.1,
                    score: Some(0.9),
                },
                SpeakerSegmentPrediction {
                    speaker: "speaker_near".to_string(),
                    start_seconds: 0.18,
                    end_seconds: 0.3,
                    score: Some(0.9),
                },
            ],
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        };

        let nearest = assign_speakers_to_transcript_with_policy(
            &transcript,
            &diarization,
            SpeakerTranscriptAssignmentPolicy::NearestStart,
        )
        .unwrap();
        assert_eq!(
            nearest.segments[0].words[0].speaker.as_deref(),
            Some("speaker_near")
        );

        let strict = assign_speakers_to_transcript_with_policy(
            &transcript,
            &diarization,
            SpeakerTranscriptAssignmentPolicy::StrictContained,
        )
        .unwrap();
        assert!(strict.segments[0].words[0].speaker.is_none());
        assert!(strict.segments[0].speaker.is_none());
    }

    #[derive(Debug, Clone)]
    struct MeanSignSpeakerEmbedder {
        model: SpeakerEmbeddingModel,
    }

    impl MeanSignSpeakerEmbedder {
        fn new() -> Self {
            Self {
                model: test_model(),
            }
        }
    }

    impl SpeakerEmbeddingExtractor for MeanSignSpeakerEmbedder {
        fn model_info(&self) -> SpeakerEmbeddingModel {
            self.model.clone()
        }

        fn embed_speaker(&mut self, audio: &SpeakerAudio<'_>) -> Result<SpeakerEmbedding> {
            let mean = audio.samples().iter().sum::<f32>() / audio.samples().len() as f32;
            if mean >= 0.0 {
                SpeakerEmbedding::new(vec![1.0, 0.0], self.model_info(), audio.sample_rate())
            } else {
                SpeakerEmbedding::new(vec![0.0, 1.0], self.model_info(), audio.sample_rate())
            }
        }
    }

    #[derive(Debug, Clone)]
    struct FixedVad {
        spans: Vec<SpeechSpan>,
    }

    impl VoiceActivityDetector for FixedVad {
        fn detect_speech(&mut self, _audio: &SpeakerAudio<'_>) -> Result<Vec<SpeechSpan>> {
            Ok(self.spans.clone())
        }
    }

    fn sine(freq_hz: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let samples = (sample_rate as f32 * seconds) as usize;
        (0..samples)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5
            })
            .collect()
    }

    fn two_profile_speech_fixture(sample_rate: u32) -> Vec<f32> {
        let mut samples = sine(220.0, sample_rate, 0.35);
        samples.extend(vec![0.0; (sample_rate as f32 * 0.12) as usize]);
        samples.extend(sine(1_200.0, sample_rate, 0.35));
        samples
    }

    fn speaker_segment(
        speaker: &str,
        start_seconds: f32,
        end_seconds: f32,
    ) -> SpeakerSegmentPrediction {
        SpeakerSegmentPrediction {
            speaker: speaker.to_string(),
            start_seconds,
            end_seconds,
            score: Some(0.8),
        }
    }

    fn test_model() -> SpeakerEmbeddingModel {
        SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::SpeechBrain, "spkrec", "1", 2)
            .unwrap()
    }

    fn embedding(values: impl Into<Vec<f32>>) -> SpeakerEmbedding {
        SpeakerEmbedding::new(values, test_model(), 16_000).unwrap()
    }

    fn id(value: &str) -> SpeakerId {
        SpeakerId::new(value).unwrap()
    }

    fn label(value: &str) -> SpeakerLabel {
        SpeakerLabel::new(value).unwrap()
    }

    #[test]
    fn speaker_audio_validates_samples_rate_channels_and_duration() {
        assert!(SpeakerAudio::mono(&[], 16_000).is_err());
        assert!(SpeakerAudio::mono(&[0.0], 0).is_err());
        assert!(SpeakerAudio::mono(&[f32::NAN], 16_000).is_err());
        assert!(SpeakerAudio::interleaved(&[0.0, 1.0, 0.0], 16_000, 2).is_err());
        assert!(SpeakerAudio::mono(&[0.0; 160], 16_000)
            .unwrap()
            .duration_bounds(Some(0.02), None)
            .is_err());
    }

    #[test]
    fn speaker_audio_accepts_valid_speech_spans_and_frame_mono_conversion() {
        let samples = [0.1, -0.1, 0.2, -0.2];
        let audio = SpeakerAudio::interleaved(&samples, 16_000, 2)
            .unwrap()
            .with_speech_spans(vec![SpeechSpan::new(0.0, 0.0001, 0.9).unwrap()])
            .unwrap();

        assert_eq!(audio.channels(), 2);
        assert_eq!(audio.speech_spans().len(), 1);
        assert_eq!(audio.to_mono().unwrap(), vec![0.0, 0.0]);
        assert!(SpeakerAudio::mono(&[0.0; 100], 1_000)
            .unwrap()
            .with_speech_spans(vec![SpeechSpan::new(0.0, 1.0, 0.5).unwrap()])
            .is_err());
    }

    #[test]
    fn ids_labels_models_spans_and_options_validate_inputs() {
        assert!(SpeakerId::new(" ").is_err());
        assert!(SpeakerLabel::new("").is_err());
        assert!(SpeakerEmbeddingModel::new(
            SpeakerEmbeddingModelFamily::Custom(" ".to_string()),
            "custom",
            "1",
            2,
        )
        .is_err());
        assert!(
            SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::EcapaTdnn, "", "1", 2).is_err()
        );
        assert!(SpeechSpan::new(0.2, 0.1, 0.5).is_err());
        assert!(SpeechSpan::new(0.0, 0.1, f32::NAN).is_err());
        assert!(SpeakerIdentificationOptions::new(f32::NAN).is_err());
        assert!(SpeakerIdentificationOptions::new(0.8)
            .unwrap()
            .min_margin(f32::NAN)
            .is_err());
        assert!(SpeakerIdentificationOptions {
            max_results: 0,
            ..SpeakerIdentificationOptions::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn speaker_embedding_carries_model_identity() {
        let model = SpeakerEmbeddingModel::new(
            SpeakerEmbeddingModelFamily::EcapaTdnn,
            "ecapa",
            "2026-01",
            2,
        )
        .unwrap();
        let embedding = SpeakerEmbedding::new(vec![2.0, 0.0], model.clone(), 16_000).unwrap();

        assert_eq!(embedding.model(), &model);
        assert_eq!(embedding.values(), &[1.0, 0.0]);
        assert!(SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], model, 16_000).is_err());
    }

    #[derive(Debug, Clone)]
    struct MockSpeakerEmbeddingProvider {
        model: SpeakerEmbeddingModel,
    }

    impl SpeakerEmbeddingProvider for MockSpeakerEmbeddingProvider {
        fn provider_id(&self) -> &str {
            "mock-speaker-embedding"
        }

        fn embed(
            &mut self,
            request: SpeakerEmbeddingRequest<'_>,
        ) -> Result<SpeakerEmbeddingResponse> {
            Ok(SpeakerEmbeddingResponse {
                model_id: self.model.name.clone(),
                runtime: AudioRuntime::Onnx,
                embedding: SpeakerEmbedding::new(
                    vec![3.0, 4.0],
                    self.model.clone(),
                    request.audio.sample_rate(),
                )?,
                diagnostics: vec!["speakerEmbeddingProvider=mock".to_string()],
            })
        }
    }

    #[test]
    fn speaker_embedding_provider_trait_returns_runtime_response() {
        let samples = [0.1_f32; 160];
        let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();
        let mut provider = MockSpeakerEmbeddingProvider {
            model: test_model(),
        };

        let response = provider
            .embed(SpeakerEmbeddingRequest { audio })
            .expect("mock embedding");

        assert_eq!(provider.provider_id(), "mock-speaker-embedding");
        assert_eq!(response.model_id, "spkrec");
        assert_eq!(response.runtime, AudioRuntime::Onnx);
        assert_eq!(response.diagnostics, ["speakerEmbeddingProvider=mock"]);
        assert_eq!(response.embedding.values(), &[0.6, 0.8]);
    }

    #[test]
    fn speaker_embedding_rejects_model_mismatch_and_invalid_stored_values() {
        let left = embedding([1.0, 0.0]);
        let other_model =
            SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::XVector, "xvector", "1", 2)
                .unwrap();
        let right = SpeakerEmbedding::new(vec![1.0, 0.0], other_model, 16_000).unwrap();

        assert!(left.cosine_similarity(&right).is_err());

        let mut stored = left.clone();
        stored.values = vec![2.0, 0.0];
        assert!(stored.validate_stored().is_err());
        stored.values = vec![f32::NAN, 0.0];
        assert!(stored.validate_stored().is_err());
        stored.values = vec![1.0, 0.0];
        stored.sample_rate = 0;
        assert!(stored.validate_stored().is_err());
    }

    #[test]
    fn library_identifies_with_score_and_margin_thresholds() {
        let model = test_model();
        let alice = SpeakerEmbedding::new(vec![1.0, 0.0], model.clone(), 16_000).unwrap();
        let bob = SpeakerEmbedding::new(vec![0.0, 1.0], model.clone(), 16_000).unwrap();
        let query = SpeakerEmbedding::new(vec![0.99, 0.01], model, 16_000).unwrap();
        let mut library = SpeakerLibrary::new();
        library
            .add_profile(
                SpeakerProfile::new(id("alice"), label("Alice"))
                    .with_embedding(alice)
                    .unwrap(),
            )
            .unwrap();
        library
            .add_profile(
                SpeakerProfile::new(id("bob"), label("Bob"))
                    .with_embedding(bob)
                    .unwrap(),
            )
            .unwrap();

        let result = library
            .identify(&query, &SpeakerIdentificationOptions::default())
            .unwrap();

        assert_eq!(result.best_match.unwrap().speaker_id.as_str(), "alice");
        assert!(!result.unknown);
        assert!(result.margin.unwrap() > 0.9);
    }

    #[test]
    fn library_orders_matches_applies_max_results_and_confidence() {
        let mut library = SpeakerLibrary::new();
        for (speaker_id, values) in [
            ("c", vec![0.0, 1.0]),
            ("a", vec![1.0, 0.0]),
            ("b", vec![1.0, 0.0]),
        ] {
            library
                .add_profile(
                    SpeakerProfile::new(id(speaker_id), label(speaker_id))
                        .with_embedding(embedding(values))
                        .unwrap(),
                )
                .unwrap();
        }
        let options = SpeakerIdentificationOptions {
            min_score: -1.0,
            min_margin: None,
            max_results: 2,
            unknown_policy: UnknownSpeakerPolicy::ReturnUnknown,
        };

        let result = library.identify(&embedding([1.0, 0.0]), &options).unwrap();

        assert_eq!(result.ranked_matches.len(), 2);
        assert_eq!(result.ranked_matches[0].speaker_id.as_str(), "a");
        assert_eq!(result.ranked_matches[1].speaker_id.as_str(), "b");
        assert_eq!(result.ranked_matches[0].confidence, SpeakerConfidence::Low);
        assert_eq!(result.margin, Some(0.0));
    }

    #[test]
    fn library_marks_unknown_when_margin_fails() {
        let model =
            SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::XVector, "xvector", "1", 2)
                .unwrap();
        let mut library = SpeakerLibrary::new();
        library
            .add_profile(
                SpeakerProfile::new(id("a"), label("A"))
                    .with_embedding(
                        SpeakerEmbedding::new(vec![1.0, 0.0], model.clone(), 16_000).unwrap(),
                    )
                    .unwrap(),
            )
            .unwrap();
        library
            .add_profile(
                SpeakerProfile::new(id("b"), label("B"))
                    .with_embedding(
                        SpeakerEmbedding::new(vec![0.99, 0.01], model.clone(), 16_000).unwrap(),
                    )
                    .unwrap(),
            )
            .unwrap();
        let query = SpeakerEmbedding::new(vec![1.0, 0.0], model, 16_000).unwrap();

        let result = library
            .identify(&query, &SpeakerIdentificationOptions::default())
            .unwrap();

        assert!(result.best_match.is_none());
        assert!(result.unknown);
    }

    #[test]
    fn library_no_match_policy_suppresses_unknown_flag() {
        let mut library = SpeakerLibrary::new();
        library
            .add_profile(
                SpeakerProfile::new(id("alice"), label("Alice"))
                    .with_embedding(embedding([1.0, 0.0]))
                    .unwrap(),
            )
            .unwrap();
        let options = SpeakerIdentificationOptions {
            min_score: 1.0,
            min_margin: Some(0.1),
            max_results: 1,
            unknown_policy: UnknownSpeakerPolicy::NoMatch,
        };

        let result = library.identify(&embedding([1.0, 0.0]), &options).unwrap();

        assert!(result.best_match.is_none());
        assert!(!result.unknown);
    }

    #[test]
    fn library_rejects_duplicate_ids_unknown_ids_empty_profiles_and_mixed_models() {
        let mut library = SpeakerLibrary::new();
        library
            .add_profile(
                SpeakerProfile::new(id("alice"), label("Alice"))
                    .with_embedding(embedding([1.0, 0.0]))
                    .unwrap(),
            )
            .unwrap();
        assert!(library
            .add_profile(
                SpeakerProfile::new(id("alice"), label("Duplicate"))
                    .with_embedding(embedding([0.0, 1.0]))
                    .unwrap()
            )
            .is_err());
        assert!(library
            .add_profile(SpeakerProfile::new(id("empty"), label("Empty")))
            .is_err());
        assert!(library
            .add_embedding(&id("missing"), embedding([1.0, 0.0]))
            .is_err());

        let other_model =
            SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::Pyannote, "pyannote", "1", 2)
                .unwrap();
        assert!(library
            .add_embedding(
                &id("alice"),
                SpeakerEmbedding::new(vec![1.0, 0.0], other_model, 16_000).unwrap(),
            )
            .is_err());
    }

    #[test]
    fn spectral_speaker_embedder_enrolls_and_round_trips_snapshot() {
        let sample_rate = 8_000;
        let audio = sine(220.0, sample_rate, 0.4);
        let speaker_audio = SpeakerAudio::mono(&audio, sample_rate).unwrap();
        let mut embedder =
            SpectralSpeakerEmbedder::new(SpectralEmbeddingConfig::new(512, 256, 8).unwrap())
                .unwrap();
        let mut library = SpeakerLibrary::new();
        library
            .enroll(
                id("speaker-a"),
                label("Speaker A"),
                &speaker_audio,
                &mut embedder,
            )
            .unwrap();
        let json = library.to_json_string().unwrap();
        let restored = SpeakerLibrary::from_json_str(&json).unwrap();

        assert_eq!(restored.len(), 1);
        assert_eq!(
            restored.embedding_model().unwrap().family,
            SpeakerEmbeddingModelFamily::SpectralBaseline
        );
    }

    #[test]
    fn snapshot_loading_rejects_bad_versions_duplicates_models_and_values() {
        let mut library = SpeakerLibrary::new();
        library
            .add_profile(
                SpeakerProfile::new(id("alice"), label("Alice"))
                    .with_embedding(embedding([1.0, 0.0]))
                    .unwrap(),
            )
            .unwrap();
        let mut snapshot = library.to_snapshot().unwrap();
        snapshot.version = 999;
        assert!(SpeakerLibrary::from_snapshot(snapshot).is_err());

        let mut duplicate = library.to_snapshot().unwrap();
        duplicate.profiles.push(duplicate.profiles[0].clone());
        assert!(SpeakerLibrary::from_snapshot(duplicate).is_err());

        let bad_json = r#"{
            "version": 1,
            "embedding_model": {
                "family": "SpeechBrain",
                "name": "spkrec",
                "version": "1",
                "dimensions": 2
            },
            "profiles": [{
                "id": "alice",
                "label": "Alice",
                "embeddings": [{
                    "values": [2.0, 0.0],
                    "model": {
                        "family": "SpeechBrain",
                        "name": "spkrec",
                        "version": "1",
                        "dimensions": 2
                    },
                    "sample_rate": 16000
                }],
                "metadata": {}
            }]
        }"#;
        assert!(SpeakerLibrary::from_json_str(bad_json).is_err());

        let model_mismatch_json = r#"{
            "version": 1,
            "embedding_model": {
                "family": "SpeechBrain",
                "name": "spkrec",
                "version": "1",
                "dimensions": 2
            },
            "profiles": [{
                "id": "alice",
                "label": "Alice",
                "embeddings": [{
                    "values": [1.0, 0.0],
                    "model": {
                        "family": "SpeechBrain",
                        "name": "spkrec",
                        "version": "2",
                        "dimensions": 2
                    },
                    "sample_rate": 16000
                }],
                "metadata": {}
            }]
        }"#;
        assert!(SpeakerLibrary::from_json_str(model_mismatch_json).is_err());
    }

    #[test]
    fn onnx_embedder_exposes_model_metadata_and_fails_until_runtime_exists() {
        let mut embedder = OnnxSpeakerEmbedder::new("speaker.onnx", test_model()).unwrap();
        let samples = [0.1_f32; 160];
        let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

        assert_eq!(embedder.model_path(), Path::new("speaker.onnx"));
        assert_eq!(embedder.model_info(), test_model());
        assert!(embedder.embed_speaker(&audio).is_err());
    }

    #[test]
    fn onnx_config_validation_rejects_missing_bundle_path() {
        let err = OnnxSpeakerEmbedder::from_config(OnnxSpeakerEmbeddingConfig {
            embedding_dimension: 2,
            sample_rate: 16_000,
            ..OnnxSpeakerEmbeddingConfig::default()
        })
        .unwrap_err();

        assert!(err.to_string().contains("setup_error"));
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn onnx_runtime_unavailable_without_feature_returns_typed_error() {
        let (dir, model_path) = temp_onnx_model();
        let err = OnnxSpeakerEmbedder::from_config(OnnxSpeakerEmbeddingConfig {
            bundle_path: dir,
            embedding_dimension: 2,
            sample_rate: 16_000,
            ..OnnxSpeakerEmbeddingConfig::default()
        })
        .unwrap_err();
        let _ = std::fs::remove_file(model_path);

        assert!(err.to_string().contains("unsupported_runtime"));
    }

    #[cfg(feature = "onnx")]
    #[derive(Debug, Clone)]
    struct MockOnnxRunner {
        metadata: runtime_onnx::OnnxSessionMetadata,
        outputs: Vec<runtime_onnx::OnnxNamedTensor>,
    }

    #[cfg(feature = "onnx")]
    impl runtime_onnx::OnnxRunner for MockOnnxRunner {
        fn metadata(&self) -> runtime_onnx::Result<runtime_onnx::OnnxSessionMetadata> {
            Ok(self.metadata.clone())
        }

        fn run(
            &mut self,
            inputs: Vec<runtime_onnx::OnnxNamedTensor>,
        ) -> runtime_onnx::Result<Vec<runtime_onnx::OnnxNamedTensor>> {
            assert_eq!(inputs.len(), 1);
            Ok(self.outputs.clone())
        }
    }

    #[cfg(feature = "onnx")]
    fn mock_onnx_metadata(
        input_dimensions: Vec<runtime_onnx::OnnxDimension>,
        output_dimensions: Vec<runtime_onnx::OnnxDimension>,
    ) -> runtime_onnx::OnnxSessionMetadata {
        runtime_onnx::OnnxSessionMetadata {
            inputs: vec![runtime_onnx::OnnxIoInfo {
                name: "waveform".to_string(),
                element_type: Some(runtime_onnx::OnnxTensorElementType::F32),
                dimensions: input_dimensions,
            }],
            outputs: vec![runtime_onnx::OnnxIoInfo {
                name: "embedding".to_string(),
                element_type: Some(runtime_onnx::OnnxTensorElementType::F32),
                dimensions: output_dimensions,
            }],
        }
    }

    #[cfg(feature = "onnx")]
    fn mock_onnx_runner(output_shape: Vec<usize>) -> MockOnnxRunner {
        MockOnnxRunner {
            metadata: mock_onnx_metadata(
                vec![
                    runtime_onnx::OnnxDimension::Fixed(1),
                    runtime_onnx::OnnxDimension::Unknown,
                ],
                output_shape
                    .iter()
                    .copied()
                    .map(runtime_onnx::OnnxDimension::Fixed)
                    .collect(),
            ),
            outputs: vec![runtime_onnx::OnnxNamedTensor {
                name: "embedding".to_string(),
                tensor: runtime_onnx::OnnxTensorValue::F32(
                    runtime_onnx::OnnxF32Tensor::new(
                        output_shape.clone(),
                        if output_shape == vec![2] || output_shape == vec![1, 2] {
                            vec![3.0, 4.0]
                        } else {
                            vec![1.0; output_shape.iter().product()]
                        },
                    )
                    .unwrap(),
                ),
            }],
        }
    }

    #[cfg(feature = "onnx")]
    fn onnx_config_for_temp_model() -> (OnnxSpeakerEmbeddingConfig, PathBuf) {
        let (dir, _model_path) = temp_onnx_model();
        (
            OnnxSpeakerEmbeddingConfig {
                bundle_path: dir.clone(),
                input_name: Some("waveform".to_string()),
                output_name: Some("embedding".to_string()),
                embedding_dimension: 2,
                sample_rate: 16_000,
                ..OnnxSpeakerEmbeddingConfig::default()
            },
            dir,
        )
    }

    fn temp_onnx_model() -> (PathBuf, PathBuf) {
        let unique = format!(
            "audio-analysis-speakers-onnx-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("model.onnx");
        std::fs::write(&model_path, b"fixture").unwrap();
        (dir, model_path)
    }

    fn optional_smoke_env_value(value: Option<String>) -> Option<String> {
        value.filter(|value| !value.trim().is_empty())
    }

    #[test]
    fn onnx_smoke_env_values_accept_model_file_and_io_names() {
        assert_eq!(
            optional_smoke_env_value(Some("voxceleb_resnet34_LM.onnx".to_string())).as_deref(),
            Some("voxceleb_resnet34_LM.onnx")
        );
        assert_eq!(
            optional_smoke_env_value(Some("waveform".to_string())).as_deref(),
            Some("waveform")
        );
        assert_eq!(
            optional_smoke_env_value(Some("embedding".to_string())).as_deref(),
            Some("embedding")
        );
        assert_eq!(optional_smoke_env_value(Some("   ".to_string())), None);
        assert_eq!(optional_smoke_env_value(None), None);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn mock_onnx_rank1_output_produces_normalized_embedding() {
        let (config, dir) = onnx_config_for_temp_model();
        let mut embedder =
            OnnxSpeakerEmbedder::from_runner(config, mock_onnx_runner(vec![2])).unwrap();
        let samples = [0.1_f32; 160];
        let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

        let response = embedder.embed(SpeakerEmbeddingRequest { audio }).unwrap();

        assert_eq!(response.runtime, AudioRuntime::Onnx);
        assert_eq!(response.embedding.values(), &[0.6, 0.8]);
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "speakerEmbeddingProvider=onnx"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "speakerEmbeddingSelectedInputName=waveform"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "speakerEmbeddingSelectedInputRank=2"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "speakerEmbeddingSelectedOutputName=embedding"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn mock_onnx_rank2_output_produces_normalized_embedding() {
        let (config, dir) = onnx_config_for_temp_model();
        let mut embedder =
            OnnxSpeakerEmbedder::from_runner(config, mock_onnx_runner(vec![1, 2])).unwrap();
        let samples = [0.1_f32; 160];
        let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

        let embedding = embedder.embed_speaker(&audio).unwrap();

        assert_eq!(embedding.values(), &[0.6, 0.8]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn mock_onnx_feature_tensor_input_produces_embedding() {
        let (mut config, dir) = onnx_config_for_temp_model();
        config.input_name = Some("feats".to_string());
        config.output_name = Some("embs".to_string());
        config.embedding_dimension = 256;
        let runner = MockOnnxRunner {
            metadata: runtime_onnx::OnnxSessionMetadata {
                inputs: vec![runtime_onnx::OnnxIoInfo {
                    name: "feats".to_string(),
                    element_type: Some(runtime_onnx::OnnxTensorElementType::F32),
                    dimensions: vec![
                        runtime_onnx::OnnxDimension::Symbolic("B".to_string()),
                        runtime_onnx::OnnxDimension::Symbolic("T".to_string()),
                        runtime_onnx::OnnxDimension::Fixed(80),
                    ],
                }],
                outputs: vec![runtime_onnx::OnnxIoInfo {
                    name: "embs".to_string(),
                    element_type: Some(runtime_onnx::OnnxTensorElementType::F32),
                    dimensions: vec![
                        runtime_onnx::OnnxDimension::Symbolic("B".to_string()),
                        runtime_onnx::OnnxDimension::Fixed(256),
                    ],
                }],
            },
            outputs: vec![runtime_onnx::OnnxNamedTensor {
                name: "embs".to_string(),
                tensor: runtime_onnx::OnnxTensorValue::F32(
                    runtime_onnx::OnnxF32Tensor::new(vec![1, 256], vec![1.0; 256]).unwrap(),
                ),
            }],
        };
        let mut embedder = OnnxSpeakerEmbedder::from_runner(config, runner).unwrap();
        let samples = [0.1_f32; 160];
        let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

        let response = embedder.embed(SpeakerEmbeddingRequest { audio }).unwrap();

        assert_eq!(response.embedding.dimensions(), 256);
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "speakerEmbeddingInputKind=feature"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "speakerFeatureExtractor=fbank-logmel-cpu"));
        assert!(response
            .diagnostics
            .iter()
            .any(|item| item == "speakerFeatureBins=80"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_speaker_input_selection_detects_waveform_and_feature_shapes() {
        let waveform_rank2 = SelectedOnnxIo {
            name: "waveform".to_string(),
            index: 0,
            dimensions: vec![
                runtime_onnx::OnnxDimension::Fixed(1),
                runtime_onnx::OnnxDimension::Unknown,
            ],
        };
        let waveform_rank3 = SelectedOnnxIo {
            name: "waveform".to_string(),
            index: 0,
            dimensions: vec![
                runtime_onnx::OnnxDimension::Fixed(1),
                runtime_onnx::OnnxDimension::Fixed(1),
                runtime_onnx::OnnxDimension::Unknown,
            ],
        };
        let feature_rank3 = SelectedOnnxIo {
            name: "feats".to_string(),
            index: 0,
            dimensions: vec![
                runtime_onnx::OnnxDimension::Symbolic("B".to_string()),
                runtime_onnx::OnnxDimension::Symbolic("T".to_string()),
                runtime_onnx::OnnxDimension::Fixed(80),
            ],
        };

        assert_eq!(
            onnx_speaker_input_kind(&waveform_rank2).unwrap(),
            OnnxSpeakerInputKind::Waveform
        );
        assert_eq!(
            onnx_speaker_input_kind(&waveform_rank3).unwrap(),
            OnnxSpeakerInputKind::Waveform
        );
        assert_eq!(
            onnx_speaker_input_kind(&feature_rank3).unwrap(),
            OnnxSpeakerInputKind::Feature {
                mel_bins: 80,
                fixed_frames: None
            }
        );
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn feature_preprocessing_returns_finite_tensor_with_padded_short_audio() {
        let samples = [0.1_f32; 160];

        let (frame_count, values) = speaker_logmel_features(&samples, 16_000, 80, None).unwrap();

        assert_eq!(frame_count, 1);
        assert_eq!(values.len(), 80);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn fixed_feature_time_dimension_pads_and_truncates_deterministically() {
        let short = [0.1_f32; 160];
        let long = vec![0.1_f32; 400 + 160 * 5];

        let (padded_frames, padded_values) =
            speaker_logmel_features(&short, 16_000, 80, Some(3)).unwrap();
        let (truncated_frames, truncated_values) =
            speaker_logmel_features(&long, 16_000, 80, Some(2)).unwrap();

        assert_eq!(padded_frames, 3);
        assert_eq!(padded_values.len(), 3 * 80);
        assert!(padded_values[80..].iter().all(|value| *value == 0.0));
        assert_eq!(truncated_frames, 2);
        assert_eq!(truncated_values.len(), 2 * 80);
        assert!(truncated_values.iter().all(|value| value.is_finite()));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn mock_onnx_bad_output_shape_returns_model_output_mismatch() {
        let (config, dir) = onnx_config_for_temp_model();
        let mut embedder =
            OnnxSpeakerEmbedder::from_runner(config, mock_onnx_runner(vec![1, 1, 2])).unwrap();
        let samples = [0.1_f32; 160];
        let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

        let err = embedder.embed_speaker(&audio).unwrap_err();

        assert!(err.to_string().contains("model_output_mismatch"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn onnx_embedder_rejects_non_matching_sample_rate() {
        let (config, dir) = onnx_config_for_temp_model();
        let mut embedder =
            OnnxSpeakerEmbedder::from_runner(config, mock_onnx_runner(vec![2])).unwrap();
        let samples = [0.1_f32; 160];
        let audio = SpeakerAudio::mono(&samples, 8_000).unwrap();

        let err = embedder.embed_speaker(&audio).unwrap_err();

        assert!(err.to_string().contains("invalid_request"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn energy_vad_detects_and_merges_speech_spans() {
        let sample_rate = 1_000;
        let mut samples = vec![0.0; 100];
        samples.extend(vec![0.2; 120]);
        samples.extend(vec![0.0; 20]);
        samples.extend(vec![0.2; 120]);
        samples.extend(vec![0.0; 100]);
        let audio = SpeakerAudio::mono(&samples, sample_rate).unwrap();
        let mut vad = EnergyVoiceActivityDetector::new(EnergyVadConfig {
            rms_threshold: 0.05,
            frame_seconds: 0.04,
            hop_seconds: 0.02,
            min_speech_seconds: 0.05,
            merge_gap_seconds: 0.05,
        })
        .unwrap();

        let spans = vad.detect_speech(&audio).unwrap();

        assert_eq!(spans.len(), 1);
        assert!(spans[0].duration_seconds() >= 0.25);
    }

    #[test]
    fn energy_vad_rejects_invalid_config_and_filters_short_or_quiet_audio() {
        assert!(EnergyVoiceActivityDetector::new(EnergyVadConfig {
            rms_threshold: -1.0,
            ..EnergyVadConfig::default()
        })
        .is_err());
        assert!(EnergyVoiceActivityDetector::new(EnergyVadConfig {
            frame_seconds: 0.0,
            ..EnergyVadConfig::default()
        })
        .is_err());

        let quiet = [0.0_f32; 200];
        let audio = SpeakerAudio::mono(&quiet, 1_000).unwrap();
        let mut vad = EnergyVoiceActivityDetector::default();
        assert!(vad.detect_speech(&audio).unwrap().is_empty());

        let short = [0.2_f32; 20];
        let audio = SpeakerAudio::mono(&short, 1_000).unwrap();
        let mut vad = EnergyVoiceActivityDetector::new(EnergyVadConfig {
            rms_threshold: 0.05,
            frame_seconds: 0.01,
            hop_seconds: 0.01,
            min_speech_seconds: 0.05,
            merge_gap_seconds: 0.0,
        })
        .unwrap();
        assert!(vad.detect_speech(&audio).unwrap().is_empty());
    }

    #[test]
    fn diarizer_returns_unknown_segments_for_unenrolled_clusters() {
        let sample_rate = 8_000;
        let samples = sine(220.0, sample_rate, 0.25);
        let audio = SpeakerAudio::mono(&samples, sample_rate).unwrap();
        let embedder =
            SpectralSpeakerEmbedder::new(SpectralEmbeddingConfig::new(512, 256, 8).unwrap())
                .unwrap();
        let vad = EnergyVoiceActivityDetector::new(EnergyVadConfig {
            rms_threshold: 0.01,
            frame_seconds: 0.05,
            hop_seconds: 0.05,
            min_speech_seconds: 0.05,
            merge_gap_seconds: 0.02,
        })
        .unwrap();
        let mut diarizer = WindowedSpeakerDiarizer::new(embedder, vad);

        let result = diarizer.diarize(&audio).unwrap();

        assert!(!result.segments.is_empty());
        assert!(matches!(
            result.segments[0].speaker,
            DiarizedSpeaker::Unknown(_)
        ));
    }

    #[test]
    fn diarizer_maps_known_speakers_and_clusters_unknown_segments() {
        let mut samples = vec![0.4_f32; 100];
        samples.extend(vec![-0.4_f32; 100]);
        let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
        let mut library = SpeakerLibrary::new();
        library
            .add_profile(
                SpeakerProfile::new(id("known"), label("Known"))
                    .with_embedding(embedding([1.0, 0.0]))
                    .unwrap(),
            )
            .unwrap();
        let vad = FixedVad {
            spans: vec![
                SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
                SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
            ],
        };
        let mut options = SpeakerIdentificationOptions::new(0.8).unwrap();
        options.min_margin = None;
        let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
            .library(library)
            .cluster_threshold(0.8)
            .unwrap();
        diarizer.identification_options = options;

        let result = diarizer.diarize(&audio).unwrap();

        assert_eq!(result.segments.len(), 2);
        assert_eq!(
            result.segments[0].speaker,
            DiarizedSpeaker::Known(id("known"))
        );
        assert_eq!(
            result.segments[1].speaker,
            DiarizedSpeaker::Unknown("speaker_0".to_string())
        );
    }

    #[test]
    fn bounded_diarizer_produces_requested_two_unknown_speakers() {
        let mut samples = vec![0.4_f32; 100];
        samples.extend(vec![-0.4_f32; 100]);
        samples.extend(vec![0.4_f32; 100]);
        let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
        let vad = FixedVad {
            spans: vec![
                SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
                SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
                SpeechSpan::new(0.2, 0.3, 0.9).unwrap(),
            ],
        };
        let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
            .cluster_threshold(0.95)
            .unwrap()
            .speaker_bounds(Some(2), Some(2))
            .unwrap();

        let result = diarizer.diarize(&audio).unwrap();
        let speakers = result
            .segments
            .iter()
            .filter_map(|segment| match &segment.speaker {
                DiarizedSpeaker::Unknown(label) => Some(label.clone()),
                DiarizedSpeaker::Known(_) => None,
            })
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(speakers.len(), 2);
        assert!(speakers.contains("speaker_0"));
        assert!(speakers.contains("speaker_1"));
    }

    #[test]
    fn bounded_diarizer_max_one_collapses_distinct_unknown_profiles() {
        let mut samples = vec![0.4_f32; 100];
        samples.extend(vec![-0.4_f32; 100]);
        let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
        let vad = FixedVad {
            spans: vec![
                SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
                SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
            ],
        };
        let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
            .cluster_threshold(0.95)
            .unwrap()
            .speaker_bounds(None, Some(1))
            .unwrap();

        let result = diarizer.diarize(&audio).unwrap();
        assert!(result
            .segments
            .iter()
            .all(|segment| segment.speaker == DiarizedSpeaker::Unknown("speaker_0".to_string())));
    }

    #[test]
    fn bounded_diarizer_min_above_window_count_is_saturated_by_windows() {
        let mut samples = vec![0.4_f32; 100];
        samples.extend(vec![-0.4_f32; 100]);
        let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
        let vad = FixedVad {
            spans: vec![
                SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
                SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
            ],
        };
        let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
            .cluster_threshold(0.95)
            .unwrap()
            .speaker_bounds(Some(3), None)
            .unwrap();

        let result = diarizer.diarize(&audio).unwrap();
        let speakers = result
            .segments
            .iter()
            .filter_map(|segment| match &segment.speaker {
                DiarizedSpeaker::Unknown(label) => Some(label.clone()),
                DiarizedSpeaker::Known(_) => None,
            })
            .collect::<std::collections::BTreeSet<_>>();

        assert!(speakers.len() <= 2);
    }

    #[test]
    fn bounded_diarizer_invalid_bounds_return_invalid_argument() {
        for error in [
            WindowedSpeakerDiarizer::new(
                MeanSignSpeakerEmbedder::new(),
                FixedVad { spans: vec![] },
            )
            .speaker_bounds(Some(0), None)
            .unwrap_err(),
            WindowedSpeakerDiarizer::new(
                MeanSignSpeakerEmbedder::new(),
                FixedVad { spans: vec![] },
            )
            .speaker_bounds(None, Some(0))
            .unwrap_err(),
            WindowedSpeakerDiarizer::new(
                MeanSignSpeakerEmbedder::new(),
                FixedVad { spans: vec![] },
            )
            .speaker_bounds(Some(3), Some(2))
            .unwrap_err(),
        ] {
            assert!(matches!(error, DetectError::InvalidArgument(_)));
        }
    }

    #[test]
    fn diarizer_without_bounds_keeps_threshold_clustering_behavior() {
        let mut samples = vec![0.4_f32; 100];
        samples.extend(vec![-0.4_f32; 100]);
        let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
        let vad = FixedVad {
            spans: vec![
                SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
                SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
            ],
        };
        let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
            .cluster_threshold(0.95)
            .unwrap();

        let result = diarizer.diarize(&audio).unwrap();
        assert_eq!(
            result
                .segments
                .iter()
                .map(|segment| segment.speaker.clone())
                .collect::<Vec<_>>(),
            vec![
                DiarizedSpeaker::Unknown("speaker_0".to_string()),
                DiarizedSpeaker::Unknown("speaker_1".to_string()),
            ]
        );
    }

    #[test]
    fn native_diarization_baseline_returns_stable_speaker_segments() {
        let sample_rate = 8_000;
        let samples = sine(220.0, sample_rate, 0.30);
        let response = diarize_speaker_audio_baseline(&samples, sample_rate).unwrap();

        assert!(response.accepted);
        assert_eq!(response.runtime, AudioRuntime::Heuristic);
        assert!(!response.segments.is_empty());
        assert_eq!(response.segments[0].speaker, "speaker_0");
        assert!(response.segments[0].end_seconds > response.segments[0].start_seconds);
    }

    #[test]
    fn native_diarization_baseline_splits_distinct_spectral_profiles() {
        let sample_rate = 8_000;
        let samples = two_profile_speech_fixture(sample_rate);
        let response = diarize_speaker_audio_baseline(&samples, sample_rate).unwrap();

        assert!(response.accepted);
        assert!(response.segments.len() >= 2, "{:?}", response.segments);
        assert_eq!(response.segments[0].speaker, "speaker_0");
        assert_eq!(response.segments[1].speaker, "speaker_1");
        assert!(response.segments.windows(2).all(|pair| {
            pair[0].start_seconds.is_finite()
                && pair[0].end_seconds.is_finite()
                && pair[0].end_seconds <= pair[1].start_seconds
        }));
    }

    #[test]
    fn speaker_prediction_merge_combines_adjacent_same_speaker_only() {
        let merged = merge_speaker_predictions(
            vec![
                speaker_segment("speaker_0", 0.0, 0.4),
                speaker_segment("speaker_0", 0.43, 0.7),
                speaker_segment("speaker_1", 0.72, 1.0),
            ],
            0.05,
        )
        .unwrap();

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].speaker, "speaker_0");
        assert_eq!(merged[0].start_seconds, 0.0);
        assert_eq!(merged[0].end_seconds, 0.7);
        assert_eq!(merged[1].speaker, "speaker_1");
    }

    #[test]
    fn speaker_prediction_merge_preserves_same_speaker_after_large_gap() {
        let merged = merge_speaker_predictions(
            vec![
                speaker_segment("speaker_0", 0.0, 0.4),
                speaker_segment("speaker_0", 0.6, 0.9),
            ],
            0.05,
        )
        .unwrap();

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].speaker, "speaker_0");
        assert_eq!(merged[1].speaker, "speaker_0");
    }

    #[test]
    #[ignore]
    fn native_diarization_baseline_smoke_when_requested() {
        if std::env::var("RUN_NATIVE_DIARIZATION_TESTS").as_deref() != Ok("1") {
            eprintln!("skipping native diarization smoke; set RUN_NATIVE_DIARIZATION_TESTS=1");
            return;
        }
        let audio_path = std::env::var_os("DIARIZATION_AUDIO_PATH")
            .map(std::path::PathBuf::from)
            .expect("DIARIZATION_AUDIO_PATH is required");
        let mut reader = hound::WavReader::open(&audio_path).unwrap_or_else(|error| {
            panic!("failed to open WAV `{}`: {error}", audio_path.display())
        });
        let spec = reader.spec();
        assert!(spec.sample_rate > 0, "WAV sample rate must be non-zero");
        assert!(spec.channels > 0, "WAV channel count must be non-zero");
        let interleaved = match spec.sample_format {
            hound::SampleFormat::Float => {
                assert_eq!(spec.bits_per_sample, 32, "float WAV must be 32-bit");
                reader
                    .samples::<f32>()
                    .map(|sample| sample.expect("failed to read float WAV sample"))
                    .collect::<Vec<_>>()
            }
            hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
                .samples::<i16>()
                .map(|sample| sample.expect("failed to read int WAV sample") as f32 / 32_768.0)
                .collect::<Vec<_>>(),
            hound::SampleFormat::Int => reader
                .samples::<i32>()
                .map(|sample| {
                    let scale = 2_f32.powi(spec.bits_per_sample as i32 - 1);
                    sample.expect("failed to read int WAV sample") as f32 / scale
                })
                .collect::<Vec<_>>(),
        };
        let channels = spec.channels as usize;
        let samples = if channels == 1 {
            interleaved
        } else {
            interleaved
                .chunks_exact(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
                .collect::<Vec<_>>()
        };
        let response = diarize_speaker_audio_baseline(&samples, spec.sample_rate).unwrap();
        assert!(response.accepted);
        assert!(!response.segments.is_empty());
        assert!(response
            .segments
            .iter()
            .all(|segment| segment.end_seconds > segment.start_seconds));
    }

    #[cfg(feature = "onnx")]
    #[test]
    #[ignore]
    fn onnx_speaker_embedding_smoke_when_requested() {
        if std::env::var("RUN_NATIVE_SPEAKER_MODEL_TESTS").as_deref() != Ok("1") {
            eprintln!(
                "skipping ONNX speaker embedding smoke; set RUN_NATIVE_SPEAKER_MODEL_TESTS=1"
            );
            return;
        }
        let bundle_path = std::env::var_os("SPEAKER_EMBEDDING_MODEL_BUNDLE")
            .map(std::path::PathBuf::from)
            .expect("SPEAKER_EMBEDDING_MODEL_BUNDLE is required");
        let audio_path = std::env::var_os("DIARIZATION_AUDIO_PATH")
            .map(std::path::PathBuf::from)
            .expect("DIARIZATION_AUDIO_PATH is required");
        let embedding_dimension = std::env::var("SPEAKER_EMBEDDING_DIMENSION")
            .ok()
            .map(|value| value.parse::<usize>().expect("valid embedding dimension"))
            .unwrap_or(192);
        let model_file =
            optional_smoke_env_value(std::env::var("SPEAKER_EMBEDDING_MODEL_FILE").ok());
        let input_name =
            optional_smoke_env_value(std::env::var("SPEAKER_EMBEDDING_INPUT_NAME").ok());
        let output_name =
            optional_smoke_env_value(std::env::var("SPEAKER_EMBEDDING_OUTPUT_NAME").ok());
        let mut reader = hound::WavReader::open(&audio_path).unwrap_or_else(|error| {
            panic!("failed to open WAV `{}`: {error}", audio_path.display())
        });
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000, "smoke WAV must be 16 kHz");
        assert!(spec.channels > 0, "WAV channel count must be non-zero");
        let interleaved = match spec.sample_format {
            hound::SampleFormat::Float => {
                assert_eq!(spec.bits_per_sample, 32, "float WAV must be 32-bit");
                reader
                    .samples::<f32>()
                    .map(|sample| sample.expect("failed to read float WAV sample"))
                    .collect::<Vec<_>>()
            }
            hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
                .samples::<i16>()
                .map(|sample| sample.expect("failed to read int WAV sample") as f32 / 32_768.0)
                .collect::<Vec<_>>(),
            hound::SampleFormat::Int => reader
                .samples::<i32>()
                .map(|sample| {
                    let scale = 2_f32.powi(spec.bits_per_sample as i32 - 1);
                    sample.expect("failed to read int WAV sample") as f32 / scale
                })
                .collect::<Vec<_>>(),
        };
        let mono = interleaved_to_mono(
            &AudioBuffer::F32(interleaved),
            spec.channels,
            ChannelMix::Average,
        )
        .expect("valid mono mix");
        let audio = SpeakerAudio::mono(&mono, spec.sample_rate).expect("valid speaker audio");
        let config = OnnxSpeakerEmbeddingConfig {
            bundle_path,
            model_file,
            input_name,
            output_name,
            embedding_dimension,
            sample_rate: 16_000,
        };
        let model_path =
            resolve_onnx_speaker_model_path(&config).expect("resolved ONNX speaker model path");
        eprintln!("speakerEmbeddingResolvedModelPath={}", model_path.display());
        eprintln!("speakerEmbeddingExpectedDimension={embedding_dimension}");
        eprintln!(
            "speakerEmbeddingConfiguredInputName={}",
            config.input_name.as_deref().unwrap_or("<auto>")
        );
        eprintln!(
            "speakerEmbeddingConfiguredOutputName={}",
            config.output_name.as_deref().unwrap_or("<auto>")
        );
        let static_metadata =
            runtime_onnx::inspect_model_metadata(&model_path).expect("static ONNX metadata");
        eprintln!("onnxStaticMetadata=ok");
        for diagnostic in runtime_onnx::inspect_model_graph_diagnostics(&model_path)
            .expect("static ONNX graph diagnostics")
        {
            eprintln!("{diagnostic}");
        }
        eprintln!(
            "onnxLoadMode={}",
            std::env::var("ONNX_RUNTIME_LOAD_MODE").unwrap_or_else(|_| "file".to_string())
        );
        eprintln!(
            "onnxRuntimeDylib={}",
            if std::env::var_os("ORT_DYLIB_PATH").is_some() {
                "set"
            } else {
                "unset"
            }
        );
        let static_input =
            select_onnx_speaker_input(&static_metadata, config.input_name.as_deref())
                .expect("static ONNX speaker input metadata");
        let static_output =
            select_onnx_speaker_output(&static_metadata, config.output_name.as_deref())
                .expect("static ONNX speaker output metadata");
        eprintln!("speakerEmbeddingStaticInputName={}", static_input.name);
        eprintln!(
            "speakerEmbeddingStaticInputDimensions={}",
            format_onnx_dimensions(&static_input.dimensions)
        );
        eprintln!("speakerEmbeddingStaticOutputName={}", static_output.name);
        eprintln!(
            "speakerEmbeddingStaticOutputDimensions={}",
            format_onnx_dimensions(&static_output.dimensions)
        );
        eprintln!(
            "onnxSessionOptions=cpu-single-threaded,no-memory-pattern,graph-optimization-disabled"
        );
        let mut provider = match OnnxSpeakerEmbedder::from_config(config) {
            Ok(provider) => {
                eprintln!("onnxSessionLoad=ok");
                provider
            }
            Err(error) => {
                eprintln!("onnxSessionLoad={error}");
                panic!("ONNX speaker embedding provider: {error}");
            }
        };
        eprintln!(
            "{}",
            provider
                .onnx_runtime_metadata_diagnostics()
                .expect("ONNX speaker embedding metadata")
                .join("\n")
        );

        let response = provider
            .embed(SpeakerEmbeddingRequest { audio })
            .expect("ONNX speaker embedding");

        assert_eq!(response.runtime, AudioRuntime::Onnx);
        assert_eq!(response.embedding.dimensions(), embedding_dimension);
        assert!(response
            .embedding
            .values()
            .iter()
            .all(|value| value.is_finite()));
        eprintln!("{:#?}", response.diagnostics);
    }
}
