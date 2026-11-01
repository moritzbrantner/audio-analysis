//! Speaker audio, embeddings, and the spectral and ONNX embedding extractors.

use std::path::{Path, PathBuf};

use audio_analysis_core::{interleaved_to_mono, ChannelMix};
use audio_analysis_recognition::{
    AudioEmbeddingExtractor, AudioRuntime, SpectralAudioEmbedder, SpectralEmbeddingConfig,
};
use audio_contracts::{AudioBuffer, AudioFrame, DetectError, Result};

#[cfg(feature = "onnx")]
use crate::errors::model_output_mismatch;
use crate::errors::{invalid_request, setup_error, unsupported_runtime};
use crate::library::validate_model_compatible;
use crate::SpeechSpan;

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

    pub(crate) fn validate(&self) -> Result<()> {
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
    pub(crate) values: Vec<f32>,
    model: SpeakerEmbeddingModel,
    pub(crate) sample_rate: u32,
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

    pub(crate) fn validate_stored(&self) -> Result<()> {
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
    pub(crate) fn onnx_runtime_metadata_diagnostics(&mut self) -> Result<Vec<String>> {
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

pub(crate) fn resolve_onnx_speaker_model_path(
    config: &OnnxSpeakerEmbeddingConfig,
) -> Result<PathBuf> {
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
pub(crate) struct SelectedOnnxIo {
    pub(crate) name: String,
    pub(crate) index: usize,
    pub(crate) dimensions: Vec<runtime_onnx::OnnxDimension>,
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OnnxSpeakerInputKind {
    Waveform,
    Feature {
        mel_bins: usize,
        fixed_frames: Option<usize>,
    },
}

#[cfg(feature = "onnx")]
pub(crate) fn select_onnx_speaker_input(
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
pub(crate) fn select_onnx_speaker_output(
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
pub(crate) fn format_onnx_dimensions(dimensions: &[runtime_onnx::OnnxDimension]) -> String {
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
pub(crate) fn onnx_speaker_input_kind(input: &SelectedOnnxIo) -> Result<OnnxSpeakerInputKind> {
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
pub(crate) fn speaker_logmel_features(
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
pub(crate) fn map_onnx_session_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
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
pub(crate) fn map_onnx_runtime_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::Unavailable => {
            unsupported_runtime("ONNX speaker embedding runtime is unavailable")
        }
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message) => invalid_request(message),
        runtime_onnx::OnnxRuntimeError::InvalidTensorShape(message)
        | runtime_onnx::OnnxRuntimeError::UnsupportedTensorType(message)
        | runtime_onnx::OnnxRuntimeError::Source(message) => model_output_mismatch(message),
        runtime_onnx::OnnxRuntimeError::Io(error) => setup_error(error.to_string()),
        error @ (runtime_onnx::OnnxRuntimeError::RuntimeUnavailable { .. }
        | runtime_onnx::OnnxRuntimeError::RuntimeTooOld { .. }) => {
            unsupported_runtime(error.to_string())
        }
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
