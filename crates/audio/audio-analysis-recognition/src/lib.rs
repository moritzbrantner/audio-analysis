#![doc = include_str!("../README.md")]

pub mod surface;
pub mod transcription;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use audio_analysis_core::{
    interleaved_to_mono, peak, rms, zero_pad_to, ChannelMix, FrameSpec, StreamingFrameBuffer,
    StreamingFrameConfig, WindowFunction,
};
use audio_analysis_fourier::{FourierTransform, Spectrum};
use audio_contracts::{AnalysisEvent, AudioAnalyzer, AudioFrame, DetectError, Result, Timestamp};
use model_runtime::ModelSpec;
use serde::Deserialize;
pub use text_transcripts::{TranscriptSegmentContract, TranscriptionContract};

#[allow(deprecated)]
pub use transcription::{
    transcribe, transcription_plan, AudioTranscriptionProvider, ImportedTranscriptionProvider,
    TranscriptionBackendPlan, TranscriptionInput, TranscriptionProviderKind, TranscriptionRequest,
    TranscriptionResponse, TranscriptionRuntimeSelection, WhisperCppTranscriptionPlan,
};

/// Runtime families for audio execution and postprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioRuntime {
    /// ONNX Runtime-backed native inference.
    Onnx,
    /// Candle-backed native inference.
    Candle,
    /// Whisper.cpp-backed ASR.
    WhisperCpp,
    /// Demucs-backed source separation.
    Demucs,
    /// External command-backed execution.
    External,
    /// Deterministic spectral fallback.
    Spectral,
    /// Deterministic heuristic fallback.
    Heuristic,
    /// Caller-supplied predictions or outputs.
    Imported,
}

/// Fallback behavior when the selected native runtime cannot run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FallbackPolicy {
    /// Return a typed error.
    #[default]
    Error,
    /// Use a fast deterministic fallback.
    FastFallback,
    /// Use a spectral or heuristic fallback.
    HeuristicFallback,
}

/// Runtime selection supplied by API, CLI, or UI callers.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct AudioRuntimeSelection {
    /// Optional preset or compatibility model identifier.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Optional generic model spec.
    #[serde(default)]
    pub model: Option<ModelSpec>,
    /// Optional preferred runtime.
    #[serde(default, rename = "runtime")]
    pub backend: Option<AudioRuntime>,
    /// Fallback policy for unsupported native paths.
    #[serde(default)]
    pub fallback_policy: FallbackPolicy,
}

/// Compatibility alias for callers migrating from model-shaped audio APIs.
#[deprecated(note = "use AudioRuntimeSelection")]
pub type AudioModelSelection = AudioRuntimeSelection;

/// Feature summary for deterministic fallback paths.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioFeatureSummary {
    /// Clip duration in seconds.
    #[serde(default)]
    pub duration_seconds: f32,
    /// Sample rate in Hz.
    #[serde(default)]
    pub sample_rate: u32,
    /// Root mean square energy.
    #[serde(default)]
    pub rms: f32,
    /// Peak absolute amplitude.
    #[serde(default)]
    pub peak: f32,
    /// Zero crossing rate.
    #[serde(default)]
    pub zero_crossing_rate: f32,
    /// Dominant frequency in Hz.
    #[serde(default)]
    pub dominant_frequency_hz: f32,
    /// Spectral centroid in Hz.
    #[serde(default)]
    pub spectral_centroid_hz: f32,
}

impl Default for AudioFeatureSummary {
    fn default() -> Self {
        Self {
            duration_seconds: 0.0,
            sample_rate: 0,
            rms: 0.0,
            peak: 0.0,
            zero_crossing_rate: 0.0,
            dominant_frequency_hz: 0.0,
            spectral_centroid_hz: 0.0,
        }
    }
}

/// Window-level feature frame for event fallback.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioFeatureFrame {
    /// Start time in seconds.
    pub start_seconds: f32,
    /// End time in seconds.
    pub end_seconds: f32,
    /// Root mean square energy.
    pub rms: f32,
    /// Peak absolute amplitude.
    #[serde(default)]
    pub peak: f32,
}

/// Imported prediction used by server, CLI, and UI postprocessing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportedAudioPrediction {
    /// Optional raw prediction kind.
    #[serde(default)]
    pub kind: Option<String>,
    /// Label emitted by the model.
    pub label: String,
    /// Confidence score.
    pub score: f32,
    /// Optional start time in seconds.
    #[serde(default)]
    pub start_seconds: Option<f32>,
    /// Optional end time in seconds.
    #[serde(default)]
    pub end_seconds: Option<f32>,
    /// Optional text payload.
    #[serde(default)]
    pub text: Option<String>,
    /// Arbitrary model attributes.
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

/// One label prediction.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioClassPrediction {
    /// Label name.
    pub label: String,
    /// Confidence score.
    pub score: f32,
}

/// Request for audio classification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioClassificationRequest {
    /// Optional source identifier.
    #[serde(default)]
    pub source: Option<String>,
    /// Optional fixed labels for fallback classification.
    #[serde(default)]
    pub labels: Vec<String>,
    /// Maximum prediction count.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Audio feature summary for deterministic fallback.
    #[serde(default)]
    pub features: Option<AudioFeatureSummary>,
    /// Runtime selection.
    #[serde(default)]
    pub model: AudioRuntimeSelection,
    /// Caller-supplied model predictions.
    #[serde(default)]
    pub imported_predictions: Vec<ImportedAudioPrediction>,
}

/// Response for audio classification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioClassificationResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Selected model id.
    pub model_id: String,
    /// Runtime used.
    pub runtime: AudioRuntime,
    /// Ranked label predictions.
    pub predictions: Vec<AudioClassPrediction>,
}

/// Request for acoustic event detection.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEventDetectionRequest {
    /// Optional source identifier.
    #[serde(default)]
    pub source: Option<String>,
    /// Frame summaries for deterministic fallback.
    #[serde(default)]
    pub frames: Vec<AudioFeatureFrame>,
    /// RMS threshold for fallback event detection.
    #[serde(default = "default_event_threshold")]
    pub threshold: f32,
    /// Runtime selection.
    #[serde(default)]
    pub model: AudioRuntimeSelection,
    /// Caller-supplied model predictions.
    #[serde(default)]
    pub imported_predictions: Vec<ImportedAudioPrediction>,
}

/// One timestamped audio event.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEventPrediction {
    /// Event label.
    pub label: String,
    /// Confidence score.
    pub score: f32,
    /// Start time in seconds.
    pub start_seconds: f32,
    /// End time in seconds.
    pub end_seconds: f32,
}

/// Response for acoustic event detection.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEventDetectionResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Selected model id.
    pub model_id: String,
    /// Runtime used.
    pub runtime: AudioRuntime,
    /// Timestamped events.
    pub events: Vec<AudioEventPrediction>,
}

/// Request for audio embeddings.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEmbeddingRequest {
    /// Feature summaries to embed.
    #[serde(default)]
    pub features: Vec<AudioFeatureSummary>,
    /// Fallback vector dimensions.
    #[serde(default = "default_embedding_dimensions")]
    pub dimensions: usize,
    /// Whether to L2-normalize fallback embeddings.
    #[serde(default = "default_true")]
    pub normalize: bool,
    /// Runtime selection.
    #[serde(default)]
    pub model: AudioRuntimeSelection,
    /// Caller-supplied embeddings.
    #[serde(default)]
    pub imported_embeddings: Vec<Vec<f32>>,
}

/// Response for audio embeddings.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEmbeddingResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Selected model id.
    pub model_id: String,
    /// Runtime used.
    pub runtime: AudioRuntime,
    /// Embedding dimensions.
    pub dimensions: usize,
    /// Dense embeddings.
    pub embeddings: Vec<Vec<f32>>,
}

/// Request for speech recognition.
#[deprecated(
    note = "use audio-analysis-transcription for real ASR or text_transcripts::normalize_imported_segments for imported transcript normalization"
)]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechRecognitionRequest {
    /// Optional source identifier.
    #[serde(default)]
    pub source: Option<String>,
    /// Optional language hint.
    #[serde(default)]
    pub language: Option<String>,
    /// Runtime selection.
    #[serde(default)]
    pub model: AudioRuntimeSelection,
    /// Caller-supplied transcript segments.
    #[serde(default)]
    pub imported_segments: Vec<TranscriptSegmentContract>,
}

/// Response for speech recognition.
#[deprecated(note = "use audio-analysis-transcription for real ASR")]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechRecognitionResponse {
    /// Accepted flag for generated package surfaces.
    pub accepted: bool,
    /// Operation name.
    pub operation: String,
    /// Selected model id.
    pub model_id: String,
    /// Runtime used.
    pub runtime: AudioRuntime,
    /// Transcript contract returned by ASR.
    pub transcript: TranscriptionContract,
}

#[allow(deprecated)]
impl SpeechRecognitionResponse {
    /// Returns the full transcript text, synthesizing it from segments when the
    /// transcript does not carry a separate aggregate text field.
    pub fn text(&self) -> String {
        self.transcript.text_or_joined()
    }

    /// Returns transcript segments for compatibility with older callers.
    pub fn segments(&self) -> &[TranscriptSegmentContract] {
        &self.transcript.segments
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Data type for audio embedding.
pub struct AudioEmbedding {
    values: Vec<f32>,
}

impl AudioEmbedding {
    /// Creates a new value.
    pub fn new(values: impl Into<Vec<f32>>) -> Result<Self> {
        let mut values = values.into();
        normalize_values(&mut values)?;
        Ok(Self { values })
    }

    /// Returns values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns dimensions.
    pub fn dimensions(&self) -> usize {
        self.values.len()
    }

    /// Returns cosine similarity.
    pub fn cosine_similarity(&self, other: &Self) -> Result<f32> {
        if self.dimensions() != other.dimensions() {
            return Err(DetectError::InvalidArgument(format!(
                "audio embedding dimensions differ: {} != {}",
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
}

/// Trait for audio embedding extractor implementations.
pub trait AudioEmbeddingExtractor {
    /// Returns embed samples.
    fn embed_samples(&self, samples: &[f32], sample_rate: u32) -> Result<AudioEmbedding>;

    /// Returns embed frame.
    fn embed_frame(&self, frame: &AudioFrame<'_>) -> Result<AudioEmbedding> {
        let samples = interleaved_to_mono(frame.data, frame.channels, ChannelMix::Average)?;
        self.embed_samples(&samples, frame.sample_rate)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for spectral embedding config.
pub struct SpectralEmbeddingConfig {
    /// The FFT size value.
    pub fft_size: usize,
    /// The hop size value.
    pub hop_size: usize,
    /// The bands value.
    pub bands: usize,
    /// The window value.
    pub window: WindowFunction,
}

impl SpectralEmbeddingConfig {
    /// Creates a new value.
    pub fn new(fft_size: usize, hop_size: usize, bands: usize) -> Result<Self> {
        FourierTransform::new(fft_size)?;
        FrameSpec::new(fft_size, hop_size)?;
        if bands == 0 {
            return Err(DetectError::InvalidArgument(
                "audio embedding bands must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            fft_size,
            hop_size,
            bands,
            window: WindowFunction::Hann,
        })
    }

    /// Returns window.
    pub fn window(mut self, window: WindowFunction) -> Self {
        self.window = window;
        self
    }
}

impl Default for SpectralEmbeddingConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop_size: 512,
            bands: 12,
            window: WindowFunction::Hann,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Data type for spectral audio embedder.
pub struct SpectralAudioEmbedder {
    /// The config value.
    pub config: SpectralEmbeddingConfig,
}

impl SpectralAudioEmbedder {
    /// Creates a new value.
    pub fn new(config: SpectralEmbeddingConfig) -> Result<Self> {
        SpectralEmbeddingConfig::new(config.fft_size, config.hop_size, config.bands)?;
        Ok(Self { config })
    }

    /// Returns streaming config.
    pub fn streaming_config(&self, channel_mix: ChannelMix) -> Result<StreamingFrameConfig> {
        Ok(
            StreamingFrameConfig::new(self.config.fft_size, self.config.hop_size)?
                .channel_mix(channel_mix)
                .max_buffered_samples(self.config.fft_size + self.config.hop_size),
        )
    }
}

impl AudioEmbeddingExtractor for SpectralAudioEmbedder {
    fn embed_samples(&self, samples: &[f32], sample_rate: u32) -> Result<AudioEmbedding> {
        if sample_rate == 0 {
            return Err(DetectError::InvalidAudioFormat {
                sample_rate,
                channels: 1,
            });
        }
        if samples.is_empty() {
            return Err(DetectError::InvalidArgument(
                "audio samples must not be empty".to_string(),
            ));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(DetectError::InvalidArgument(
                "audio samples must be finite".to_string(),
            ));
        }

        let transform = FourierTransform::with_window(self.config.fft_size, self.config.window)?;
        let frame_spec = FrameSpec::new(self.config.fft_size, self.config.hop_size)?;
        let frames = if samples.len() < self.config.fft_size {
            vec![zero_pad_to(samples.to_vec(), self.config.fft_size)]
        } else {
            frame_spec
                .frames(samples)
                .map(|(_, frame)| frame.to_vec())
                .collect::<Vec<_>>()
        };

        let mut accumulator = EmbeddingAccumulator::new(self.config.bands);
        for frame in frames {
            let spectrum = transform.analyze_samples(&frame, sample_rate)?;
            accumulator.add_frame(&frame, sample_rate, &spectrum);
        }
        AudioEmbedding::new(accumulator.finish())
    }
}

#[derive(Debug, Clone)]
struct EmbeddingAccumulator {
    bands: usize,
    count: usize,
    sum_rms: f32,
    sum_peak: f32,
    sum_zcr: f32,
    sum_centroid: f32,
    sum_bandwidth: f32,
    sum_rolloff: f32,
    sum_flatness: f32,
    sum_dominant: f32,
    band_energy: Vec<f32>,
}

impl EmbeddingAccumulator {
    fn new(bands: usize) -> Self {
        Self {
            bands,
            count: 0,
            sum_rms: 0.0,
            sum_peak: 0.0,
            sum_zcr: 0.0,
            sum_centroid: 0.0,
            sum_bandwidth: 0.0,
            sum_rolloff: 0.0,
            sum_flatness: 0.0,
            sum_dominant: 0.0,
            band_energy: vec![0.0; bands],
        }
    }

    fn add_frame(&mut self, samples: &[f32], sample_rate: u32, spectrum: &Spectrum) {
        let nyquist = sample_rate as f32 * 0.5;
        let features = spectrum.features();
        self.count += 1;
        self.sum_rms += rms(samples);
        self.sum_peak += peak(samples);
        self.sum_zcr += zero_crossing_rate(samples);
        self.sum_centroid += normalize_frequency(features.centroid_hz, nyquist);
        self.sum_bandwidth += normalize_frequency(features.bandwidth_hz, nyquist);
        self.sum_rolloff += normalize_frequency(features.rolloff_hz, nyquist);
        self.sum_flatness += features.flatness.clamp(0.0, 1.0);
        self.sum_dominant += features
            .dominant_frequency_hz
            .map(|frequency| normalize_frequency(frequency, nyquist))
            .unwrap_or(0.0);
        for (band, energy) in band_energies(spectrum, self.bands).into_iter().enumerate() {
            self.band_energy[band] += energy;
        }
    }

    fn finish(mut self) -> Vec<f32> {
        let count = self.count.max(1) as f32;
        for energy in &mut self.band_energy {
            *energy = (*energy / count).ln_1p();
        }
        let mut values = vec![
            self.sum_rms / count,
            self.sum_peak / count,
            self.sum_zcr / count,
            self.sum_centroid / count,
            self.sum_bandwidth / count,
            self.sum_rolloff / count,
            self.sum_flatness / count,
            self.sum_dominant / count,
        ];
        values.extend(self.band_energy);
        values
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Data type for audio reference.
pub struct AudioReference {
    id: String,
    label: String,
    embeddings: Vec<AudioEmbedding>,
    attributes: BTreeMap<String, String>,
}

impl AudioReference {
    /// Creates a new value.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            embeddings: Vec::new(),
            attributes: BTreeMap::new(),
        }
    }

    /// Returns this value with embedding.
    pub fn with_embedding(mut self, embedding: AudioEmbedding) -> Result<Self> {
        self.add_embedding(embedding)?;
        Ok(self)
    }

    /// Returns this value with samples.
    pub fn with_samples<E: AudioEmbeddingExtractor>(
        mut self,
        samples: &[f32],
        sample_rate: u32,
        extractor: &E,
    ) -> Result<Self> {
        self.add_samples(samples, sample_rate, extractor)?;
        Ok(self)
    }

    /// Returns attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Adds add embedding to this value.
    pub fn add_embedding(&mut self, embedding: AudioEmbedding) -> Result<()> {
        if let Some(dimensions) = self.dimensions() {
            if dimensions != embedding.dimensions() {
                return Err(DetectError::InvalidArgument(format!(
                    "audio reference `{}` has mixed embedding dimensions: {} != {}",
                    self.id,
                    dimensions,
                    embedding.dimensions()
                )));
            }
        }
        self.embeddings.push(embedding);
        Ok(())
    }

    /// Adds add samples to this value.
    pub fn add_samples<E: AudioEmbeddingExtractor>(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        extractor: &E,
    ) -> Result<()> {
        self.add_embedding(extractor.embed_samples(samples, sample_rate)?)
    }

    /// Returns identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns embeddings.
    pub fn embeddings(&self) -> &[AudioEmbedding] {
        &self.embeddings
    }

    /// Returns attributes.
    pub fn attributes(&self) -> &BTreeMap<String, String> {
        &self.attributes
    }

    /// Returns dimensions.
    pub fn dimensions(&self) -> Option<usize> {
        self.embeddings.first().map(AudioEmbedding::dimensions)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Data type for audio reference library.
pub struct AudioReferenceLibrary {
    references: BTreeMap<String, AudioReference>,
    dimensions: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Data type for audio reference library snapshot.
pub struct AudioReferenceLibrarySnapshot {
    /// The version value.
    pub version: u32,
    /// The dimensions value.
    pub dimensions: Option<usize>,
    /// The references value.
    pub references: Vec<AudioReference>,
}

impl AudioReferenceLibrary {
    /// Creates a new value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds add reference to this value.
    pub fn add_reference(&mut self, reference: AudioReference) -> Result<()> {
        if reference.id().is_empty() {
            return Err(DetectError::InvalidArgument(
                "audio reference id must not be empty".to_string(),
            ));
        }
        if reference.label().is_empty() {
            return Err(DetectError::InvalidArgument(
                "audio reference label must not be empty".to_string(),
            ));
        }
        let dimensions = reference.dimensions().ok_or_else(|| {
            DetectError::InvalidArgument(format!(
                "audio reference `{}` must contain at least one embedding",
                reference.id()
            ))
        })?;
        self.validate_dimensions(dimensions)?;
        if self.references.contains_key(reference.id()) {
            return Err(DetectError::InvalidArgument(
                "duplicate audio reference id".to_string(),
            ));
        }
        self.dimensions = Some(dimensions);
        self.references
            .insert(reference.id().to_string(), reference);
        Ok(())
    }

    /// Adds add reference embedding to this value.
    pub fn add_reference_embedding(
        &mut self,
        id: impl Into<String>,
        label: impl Into<String>,
        embedding: AudioEmbedding,
    ) -> Result<()> {
        self.add_reference(AudioReference::new(id, label).with_embedding(embedding)?)
    }

    /// Adds add reference samples to this value.
    pub fn add_reference_samples<E: AudioEmbeddingExtractor>(
        &mut self,
        id: impl Into<String>,
        label: impl Into<String>,
        samples: &[f32],
        sample_rate: u32,
        extractor: &E,
    ) -> Result<()> {
        self.add_reference(AudioReference::new(id, label).with_samples(
            samples,
            sample_rate,
            extractor,
        )?)
    }

    /// Adds add sample to this value.
    pub fn add_sample(&mut self, id: &str, embedding: AudioEmbedding) -> Result<()> {
        self.validate_dimensions(embedding.dimensions())?;
        let reference = self.references.get_mut(id).ok_or_else(|| {
            DetectError::InvalidArgument(format!("unknown audio reference id `{id}`"))
        })?;
        reference.add_embedding(embedding)
    }

    /// Adds add sample from samples to this value.
    pub fn add_sample_from_samples<E: AudioEmbeddingExtractor>(
        &mut self,
        id: &str,
        samples: &[f32],
        sample_rate: u32,
        extractor: &E,
    ) -> Result<()> {
        self.add_sample(id, extractor.embed_samples(samples, sample_rate)?)
    }

    /// Returns reference.
    pub fn reference(&self, id: &str) -> Option<&AudioReference> {
        self.references.get(id)
    }

    /// Returns references.
    pub fn references(&self) -> impl Iterator<Item = &AudioReference> {
        self.references.values()
    }

    /// Returns len.
    pub fn len(&self) -> usize {
        self.references.len()
    }

    /// Returns whether is empty.
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    /// Returns dimensions.
    pub fn dimensions(&self) -> Option<usize> {
        self.dimensions
    }

    /// Converts this value to snapshot.
    pub fn to_snapshot(&self) -> AudioReferenceLibrarySnapshot {
        AudioReferenceLibrarySnapshot {
            version: 1,
            dimensions: self.dimensions,
            references: self.references.values().cloned().collect(),
        }
    }

    /// Builds this value from snapshot.
    pub fn from_snapshot(snapshot: AudioReferenceLibrarySnapshot) -> Result<Self> {
        if snapshot.version != 1 {
            return Err(DetectError::InvalidArgument(format!(
                "unsupported audio reference library snapshot version `{}`",
                snapshot.version
            )));
        }
        let mut library = Self::new();
        for reference in snapshot.references {
            library.add_reference(reference)?;
        }
        if snapshot.dimensions.is_some() && snapshot.dimensions != library.dimensions {
            return Err(DetectError::InvalidArgument(
                "audio reference library snapshot dimensions did not match references".to_string(),
            ));
        }
        Ok(library)
    }

    /// Converts this value to JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.to_snapshot())
            .map_err(|err| DetectError::Source(err.to_string()))
    }

    /// Builds this value from JSON str.
    pub fn from_json_str(json: &str) -> Result<Self> {
        let snapshot = serde_json::from_str::<AudioReferenceLibrarySnapshot>(json)
            .map_err(|err| DetectError::Source(err.to_string()))?;
        Self::from_snapshot(snapshot)
    }

    /// Returns search.
    pub fn search(
        &self,
        embedding: &AudioEmbedding,
        options: &AudioMatchOptions,
    ) -> Result<Vec<AudioRecognitionMatch>> {
        options.validate()?;
        self.validate_dimensions(embedding.dimensions())?;
        let mut matches = Vec::new();
        for reference in self.references.values() {
            let Some(score) = best_reference_score(reference, embedding)? else {
                continue;
            };
            if score < options.min_score {
                continue;
            }
            matches.push(AudioRecognitionMatch {
                reference_id: reference.id().to_string(),
                label: reference.label().to_string(),
                score,
                attributes: reference.attributes().clone(),
            });
        }
        matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.reference_id.cmp(&right.reference_id))
        });
        matches.truncate(options.max_results);
        Ok(matches)
    }

    fn validate_dimensions(&self, dimensions: usize) -> Result<()> {
        if let Some(expected) = self.dimensions {
            if expected != dimensions {
                return Err(DetectError::InvalidArgument(format!(
                    "audio embedding dimensions differ from reference library: {dimensions} != {expected}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for audio match options.
pub struct AudioMatchOptions {
    /// The min score value.
    pub min_score: f32,
    /// The max results value.
    pub max_results: usize,
}

impl AudioMatchOptions {
    /// Creates a new value.
    pub fn new(min_score: f32) -> Result<Self> {
        let options = Self {
            min_score,
            ..Self::default()
        };
        options.validate()?;
        Ok(options)
    }

    /// Returns max results.
    pub fn max_results(mut self, value: usize) -> Self {
        self.max_results = value;
        self
    }

    fn validate(&self) -> Result<()> {
        if !self.min_score.is_finite() || !(-1.0..=1.0).contains(&self.min_score) {
            return Err(DetectError::InvalidArgument(
                "minimum audio recognition score must be finite and between -1.0 and 1.0"
                    .to_string(),
            ));
        }
        if self.max_results == 0 {
            return Err(DetectError::InvalidArgument(
                "max audio recognition results must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for AudioMatchOptions {
    fn default() -> Self {
        Self {
            min_score: 0.85,
            max_results: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for audio recognition match.
pub struct AudioRecognitionMatch {
    /// The reference identifier value.
    pub reference_id: String,
    /// Label assigned to this value.
    pub label: String,
    /// Score assigned to this value.
    pub score: f32,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for audio similarity.
pub struct AudioSimilarity {
    /// Score assigned to this value.
    pub score: f32,
}

/// Returns compare audio samples.
pub fn compare_audio_samples<E: AudioEmbeddingExtractor>(
    left: &[f32],
    right: &[f32],
    sample_rate: u32,
    extractor: &E,
) -> Result<AudioSimilarity> {
    let left = extractor.embed_samples(left, sample_rate)?;
    let right = extractor.embed_samples(right, sample_rate)?;
    Ok(AudioSimilarity {
        score: left.cosine_similarity(&right)?,
    })
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for fingerprint record.
pub struct FingerprintRecord {
    /// Timestamp associated with this value.
    pub timestamp: Option<f64>,
    /// The duration value.
    pub duration: f64,
    /// The compressed value.
    pub compressed: Option<String>,
    /// The raw value.
    pub raw: Vec<i64>,
    /// The raw JSON value.
    pub raw_json: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Data type for fingerprint matched segment.
pub struct FingerprintMatchedSegment {
    /// The query start seconds value.
    pub query_start_seconds: Option<f64>,
    /// The catalog start seconds value.
    pub catalog_start_seconds: Option<f64>,
    /// Duration in seconds.
    pub duration_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for fingerprint candidate.
pub struct FingerprintCandidate<T> {
    /// The item value.
    pub item: T,
    /// Score assigned to this value.
    pub score: f32,
    /// The matched segment value.
    pub matched_segment: Option<FingerprintMatchedSegment>,
}

#[derive(Debug, Deserialize)]
struct FpcalcOutput {
    timestamp: Option<f64>,
    duration: f64,
    fingerprint: serde_json::Value,
}

/// Runs fpcalc.
pub fn run_fpcalc(
    command: impl AsRef<Path>,
    media_path: impl AsRef<Path>,
    chunk_seconds: Option<u64>,
) -> Result<Vec<FingerprintRecord>> {
    let mut process = Command::new(command.as_ref());
    process.arg("-json").arg("-raw");
    if let Some(chunk_seconds) = chunk_seconds {
        process
            .arg("-chunk")
            .arg(chunk_seconds.to_string())
            .arg("-overlap");
    }
    let output = process
        .arg(media_path.as_ref())
        .stdin(Stdio::null())
        .output()?;
    if !output.status.success() {
        return Err(DetectError::Source(format!(
            "fpcalc failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    parse_fpcalc_json(&String::from_utf8_lossy(&output.stdout))
}

/// Runs default fpcalc.
pub fn run_default_fpcalc(
    media_path: impl AsRef<Path>,
    chunk_seconds: Option<u64>,
) -> Result<Vec<FingerprintRecord>> {
    run_fpcalc(PathBuf::from("fpcalc"), media_path, chunk_seconds)
}

/// Parses parse fpcalc JSON.
pub fn parse_fpcalc_json(output: &str) -> Result<Vec<FingerprintRecord>> {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if let Ok(parsed) = parse_fpcalc_record(trimmed) {
        return Ok(vec![parsed]);
    }
    trimmed
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(parse_fpcalc_record)
        .collect()
}

/// Parses parse fpcalc record.
pub fn parse_fpcalc_record(line: &str) -> Result<FingerprintRecord> {
    let parsed: FpcalcOutput = serde_json::from_str(line)
        .map_err(|err| DetectError::InvalidArgument(format!("invalid fpcalc JSON: {err}")))?;
    let raw = match &parsed.fingerprint {
        serde_json::Value::Array(values) => values
            .iter()
            .map(|value| {
                value.as_i64().ok_or_else(|| {
                    DetectError::InvalidArgument(
                        "fpcalc raw fingerprint contains a non-integer".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?,
        serde_json::Value::String(_) => Vec::new(),
        _ => {
            return Err(DetectError::InvalidArgument(
                "fpcalc fingerprint has an unsupported shape".to_string(),
            ))
        }
    };
    let compressed = parsed.fingerprint.as_str().map(|value| value.to_string());
    Ok(FingerprintRecord {
        timestamp: parsed.timestamp,
        duration: parsed.duration,
        compressed,
        raw,
        raw_json: line.trim().to_string(),
    })
}

/// Returns duration prefilter.
pub fn duration_prefilter(query_duration: f64, catalog_duration: f64) -> bool {
    let tolerance = query_duration.max(catalog_duration) * 0.10;
    (query_duration - catalog_duration).abs() <= tolerance.max(10.0)
}

/// Returns shifted similarity.
pub fn shifted_similarity(left: &[i64], right: &[i64]) -> f32 {
    (-12..=12)
        .map(|offset| fingerprint_similarity_with_offset(left, right, offset))
        .fold(0.0, f32::max)
}

/// Returns fingerprint similarity with offset.
pub fn fingerprint_similarity_with_offset(left: &[i64], right: &[i64], offset: isize) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let (left_start, right_start) = if offset >= 0 {
        (offset as usize, 0_usize)
    } else {
        (0_usize, (-offset) as usize)
    };
    if left_start >= left.len() || right_start >= right.len() {
        return 0.0;
    }
    let len = (left.len() - left_start).min(right.len() - right_start);
    if len == 0 {
        return 0.0;
    }
    let mut distance = 0_u32;
    for index in 0..len {
        distance += (left[left_start + index] ^ right[right_start + index]).count_ones();
    }
    let max_distance = len as u32 * 64;
    1.0 - (distance as f32 / max_distance as f32)
}

/// Returns rank fingerprint candidates.
pub fn rank_fingerprint_candidates<T, I>(
    query_full: &FingerprintRecord,
    query_chunks: &[FingerprintRecord],
    candidates: I,
    max_candidates: usize,
) -> Vec<FingerprintCandidate<T>>
where
    I: IntoIterator<Item = (T, FingerprintRecord, Vec<FingerprintRecord>)>,
{
    let mut ranked = Vec::new();
    for (item, catalog_full, catalog_chunks) in candidates {
        let mut score = 0.0_f32;
        let mut matched_segment = None;
        if duration_prefilter(query_full.duration, catalog_full.duration) {
            let fingerprint_score = shifted_similarity(&query_full.raw, &catalog_full.raw);
            let duration_factor = (1.0
                - ((query_full.duration - catalog_full.duration).abs()
                    / query_full.duration.max(catalog_full.duration)) as f32)
                .clamp(0.0, 1.0);
            score = fingerprint_score * duration_factor;
            matched_segment = Some(FingerprintMatchedSegment {
                query_start_seconds: Some(0.0),
                catalog_start_seconds: catalog_full.timestamp.or(Some(0.0)),
                duration_seconds: query_full.duration.min(catalog_full.duration),
            });
        }

        for catalog_chunk in &catalog_chunks {
            let queries: Box<dyn Iterator<Item = &FingerprintRecord>> = if query_chunks.is_empty() {
                Box::new(std::iter::once(query_full))
            } else {
                Box::new(query_chunks.iter())
            };
            for query_chunk in queries {
                let chunk_score = shifted_similarity(&query_chunk.raw, &catalog_chunk.raw);
                if chunk_score > score {
                    score = chunk_score;
                    matched_segment = Some(FingerprintMatchedSegment {
                        query_start_seconds: query_chunk.timestamp,
                        catalog_start_seconds: catalog_chunk.timestamp,
                        duration_seconds: query_chunk.duration.min(catalog_chunk.duration),
                    });
                }
            }
        }

        if score > 0.0 {
            ranked.push(FingerprintCandidate {
                item,
                score,
                matched_segment,
            });
        }
    }
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    ranked.truncate(max_candidates);
    ranked
}

/// Data type for audio recognition analyzer.
pub struct AudioRecognitionAnalyzer<E> {
    name: String,
    extractor: E,
    library: AudioReferenceLibrary,
    match_options: AudioMatchOptions,
    windows: StreamingFrameBuffer,
}

impl<E: AudioEmbeddingExtractor> AudioRecognitionAnalyzer<E> {
    /// Creates a new value.
    pub fn new(
        name: impl Into<String>,
        extractor: E,
        library: AudioReferenceLibrary,
        streaming_config: StreamingFrameConfig,
    ) -> Result<Self> {
        Ok(Self {
            name: name.into(),
            extractor,
            library,
            match_options: AudioMatchOptions::default(),
            windows: StreamingFrameBuffer::new(streaming_config)?,
        })
    }

    /// Returns match options.
    pub fn match_options(mut self, options: AudioMatchOptions) -> Self {
        self.match_options = options;
        self
    }

    /// Returns library.
    pub fn library(&self) -> &AudioReferenceLibrary {
        &self.library
    }

    /// Returns extractor.
    pub fn extractor(&self) -> &E {
        &self.extractor
    }

    /// Returns reset.
    pub fn reset(&mut self) {
        self.windows.reset();
    }
}

impl AudioRecognitionAnalyzer<SpectralAudioEmbedder> {
    /// Returns spectral.
    pub fn spectral(name: impl Into<String>, library: AudioReferenceLibrary) -> Result<Self> {
        let extractor = SpectralAudioEmbedder::default();
        let streaming_config = extractor.streaming_config(ChannelMix::Average)?;
        Self::new(name, extractor, library, streaming_config)
    }
}

impl<E: AudioEmbeddingExtractor> AudioAnalyzer for AudioRecognitionAnalyzer<E> {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &AudioFrame<'_>) -> Result<Vec<AnalysisEvent>> {
        let windows = self.windows.push_frame(frame)?;
        let mut events = Vec::new();
        for window in windows {
            let embedding = self
                .extractor
                .embed_samples(&window.samples, window.sample_rate)?;
            for recognition_match in self.library.search(&embedding, &self.match_options)? {
                events.push(event_for_match(
                    self.name(),
                    window.timestamp,
                    recognition_match,
                ));
            }
        }
        Ok(events)
    }
}

fn best_reference_score(
    reference: &AudioReference,
    embedding: &AudioEmbedding,
) -> Result<Option<f32>> {
    let mut best = None;
    for reference_embedding in reference.embeddings() {
        let score = reference_embedding.cosine_similarity(embedding)?;
        best = Some(best.map(|value: f32| value.max(score)).unwrap_or(score));
    }
    Ok(best)
}

fn event_for_match(
    analyzer: &str,
    timestamp: Timestamp,
    recognition_match: AudioRecognitionMatch,
) -> AnalysisEvent {
    AnalysisEvent::new(
        analyzer,
        format!(
            "audio:recognized:{}:{}",
            recognition_match.reference_id, recognition_match.label
        ),
    )
    .at_timestamp(timestamp)
    .score(recognition_match.score)
}

fn normalize_values(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Err(DetectError::InvalidArgument(
            "audio embedding must not be empty".to_string(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "audio embedding values must be finite".to_string(),
        ));
    }
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::EPSILON || !norm.is_finite() {
        return Err(DetectError::InvalidArgument(
            "audio embedding norm must be finite and non-zero".to_string(),
        ));
    }
    for value in values {
        *value /= norm;
    }
    Ok(())
}

fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let crossings = samples
        .windows(2)
        .filter(|pair| (pair[0] >= 0.0 && pair[1] < 0.0) || (pair[0] < 0.0 && pair[1] >= 0.0))
        .count();
    crossings as f32 / (samples.len() - 1) as f32
}

fn normalize_frequency(frequency_hz: f32, nyquist: f32) -> f32 {
    if nyquist <= f32::EPSILON {
        0.0
    } else {
        (frequency_hz / nyquist).clamp(0.0, 1.0)
    }
}

fn band_energies(spectrum: &Spectrum, bands: usize) -> Vec<f32> {
    let mut energies = vec![0.0_f32; bands];
    if spectrum.bins.len() <= 1 {
        return energies;
    }
    let max_index = spectrum.bins.len() - 1;
    for bin in spectrum.bins.iter().skip(1) {
        let band = ((bin.index - 1) * bands / max_index).min(bands - 1);
        energies[band] += bin.power;
    }
    energies
}

/// Runs audio classification from imported predictions or an explicit fallback.
pub fn classify_audio(request: AudioClassificationRequest) -> Result<AudioClassificationResponse> {
    let model_id = selected_model_id(&request.model, "ast-audioset");

    if !request.imported_predictions.is_empty() {
        return Ok(AudioClassificationResponse {
            accepted: true,
            operation: "classify".to_string(),
            model_id,
            runtime: AudioRuntime::Imported,
            predictions: normalize_imported_predictions(
                request.imported_predictions,
                request.top_k,
            ),
        });
    }

    match request.model.fallback_policy {
        FallbackPolicy::FastFallback | FallbackPolicy::HeuristicFallback => {
            let features = request.features.unwrap_or_default();
            Ok(AudioClassificationResponse {
                accepted: true,
                operation: "classify".to_string(),
                model_id,
                runtime: AudioRuntime::Spectral,
                predictions: spectral_label_scores(&features, &request.labels, request.top_k),
            })
        }
        FallbackPolicy::Error => unsupported_runtime(
            "native audio classification requires imported predictions or an explicit fallback policy",
        ),
    }
}

/// Runs event detection from imported predictions or energy threshold fallback.
pub fn detect_audio_events(
    request: AudioEventDetectionRequest,
) -> Result<AudioEventDetectionResponse> {
    let model_id = selected_model_id(&request.model, "audioset-event-detector");

    if !request.imported_predictions.is_empty() {
        let events = request
            .imported_predictions
            .into_iter()
            .filter_map(|prediction| {
                Some(AudioEventPrediction {
                    label: prediction.label,
                    score: prediction.score,
                    start_seconds: prediction.start_seconds?,
                    end_seconds: prediction.end_seconds?,
                })
            })
            .collect::<Vec<_>>();
        return Ok(AudioEventDetectionResponse {
            accepted: true,
            operation: "events".to_string(),
            model_id,
            runtime: AudioRuntime::Imported,
            events,
        });
    }

    match request.model.fallback_policy {
        FallbackPolicy::FastFallback | FallbackPolicy::HeuristicFallback => {
            let events = request
                .frames
                .into_iter()
                .filter(|frame| frame.rms >= request.threshold || frame.peak >= request.threshold)
                .map(|frame| AudioEventPrediction {
                    label: "high_energy".to_string(),
                    score: frame.rms.max(frame.peak).clamp(0.0, 1.0),
                    start_seconds: frame.start_seconds,
                    end_seconds: frame.end_seconds,
                })
                .collect::<Vec<_>>();
            Ok(AudioEventDetectionResponse {
                accepted: true,
                operation: "events".to_string(),
                model_id,
                runtime: AudioRuntime::Heuristic,
                events,
            })
        }
        FallbackPolicy::Error => unsupported_runtime(
            "native audio event detection requires imported predictions or an explicit fallback policy",
        ),
    }
}

/// Creates imported or deterministic fallback audio embeddings.
pub fn embed_audio(request: AudioEmbeddingRequest) -> Result<AudioEmbeddingResponse> {
    let model_id = selected_model_id(&request.model, "clap-htsat-unfused");

    if !request.imported_embeddings.is_empty() {
        let dimensions = request
            .imported_embeddings
            .first()
            .map(Vec::len)
            .unwrap_or_default();
        return Ok(AudioEmbeddingResponse {
            accepted: true,
            operation: "embed".to_string(),
            model_id,
            runtime: AudioRuntime::Imported,
            dimensions,
            embeddings: request.imported_embeddings,
        });
    }

    match request.model.fallback_policy {
        FallbackPolicy::FastFallback | FallbackPolicy::HeuristicFallback => {
            if request.features.is_empty() {
                return Err(DetectError::InvalidArgument(
                    "audio embedding request must include features or imported embeddings"
                        .to_string(),
                ));
            }
            let dimensions = request.dimensions.max(1);
            let embeddings = request
                .features
                .iter()
                .map(|features| spectral_summary_embedding(features, dimensions, request.normalize))
                .collect::<Vec<_>>();
            Ok(AudioEmbeddingResponse {
                accepted: true,
                operation: "embed".to_string(),
                model_id,
                runtime: AudioRuntime::Spectral,
                dimensions,
                embeddings,
            })
        }
        FallbackPolicy::Error => unsupported_runtime(
            "native audio embedding requires imported embeddings or an explicit fallback policy",
        ),
    }
}

/// Handles ASR from imported transcript segments.
#[deprecated(
    note = "use audio-analysis-transcription for real ASR or text_transcripts::normalize_imported_segments for imported transcript normalization"
)]
#[allow(deprecated)]
pub fn transcribe_audio(request: SpeechRecognitionRequest) -> Result<SpeechRecognitionResponse> {
    if request.imported_segments.is_empty() {
        return unsupported_runtime(
            "native speech recognition requires text-transcripts native whisper.cpp execution or imported segments",
        );
    }

    let response = transcribe(request.into())?;
    Ok(SpeechRecognitionResponse {
        accepted: response.accepted,
        operation: response.operation,
        model_id: response.model_id,
        runtime: transcription::model_backend_to_audio_runtime(&response.backend),
        transcript: response.transcript,
    })
}

/// Builds a speech recognition response from a full transcript contract.
#[allow(deprecated)]
pub fn speech_recognition_response_from_transcription(
    model: &AudioRuntimeSelection,
    transcript: TranscriptionContract,
) -> Result<SpeechRecognitionResponse> {
    let response = transcribe(TranscriptionRequest {
        source: None,
        language: None,
        input: TranscriptionInput::ImportedContract { transcript },
        runtime: model.clone().into(),
    })?;
    Ok(SpeechRecognitionResponse {
        accepted: response.accepted,
        operation: response.operation,
        model_id: response.model_id,
        runtime: transcription::model_backend_to_audio_runtime(&response.backend),
        transcript: response.transcript,
    })
}

fn selected_model_id(selection: &AudioRuntimeSelection, default: &str) -> String {
    selection
        .model_id
        .clone()
        .or_else(|| selection.model.as_ref().map(|model| model.name.clone()))
        .unwrap_or_else(|| default.to_string())
}

fn default_top_k() -> usize {
    3
}

fn default_event_threshold() -> f32 {
    0.2
}

fn default_embedding_dimensions() -> usize {
    128
}

fn default_true() -> bool {
    true
}

fn unsupported_runtime<T>(message: &str) -> Result<T> {
    Err(DetectError::InvalidArgument(format!(
        "unsupported_runtime: {message}"
    )))
}

fn normalize_imported_predictions(
    predictions: Vec<ImportedAudioPrediction>,
    top_k: usize,
) -> Vec<AudioClassPrediction> {
    let mut predictions = predictions
        .into_iter()
        .map(|prediction| AudioClassPrediction {
            label: normalize_prediction_label(&prediction.label),
            score: prediction.score.clamp(0.0, 1.0),
        })
        .collect::<Vec<_>>();
    predictions.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.label.cmp(&right.label))
    });
    predictions.truncate(top_k.max(1));
    predictions
}

fn spectral_label_scores(
    features: &AudioFeatureSummary,
    labels: &[String],
    top_k: usize,
) -> Vec<AudioClassPrediction> {
    let defaults = if features.rms < 0.02 && features.peak < 0.05 {
        vec![
            ("silence".to_string(), 0.92),
            ("background".to_string(), 0.35),
        ]
    } else if features.dominant_frequency_hz > 180.0 && features.spectral_centroid_hz > 1_200.0 {
        vec![
            ("speech".to_string(), 0.72),
            ("foreground".to_string(), 0.58),
        ]
    } else if features.rms > 0.25 || features.peak > 0.6 {
        vec![
            ("music".to_string(), 0.68),
            ("loud_event".to_string(), 0.62),
        ]
    } else {
        vec![
            ("ambient".to_string(), 0.57),
            ("background".to_string(), 0.42),
        ]
    };

    let mut predictions = if labels.is_empty() {
        defaults
            .into_iter()
            .map(|(label, score)| AudioClassPrediction { label, score })
            .collect::<Vec<_>>()
    } else {
        labels
            .iter()
            .map(|label| {
                let normalized = normalize_prediction_label(label);
                let score = match normalized.as_str() {
                    "silence" => 1.0 - features.rms.clamp(0.0, 1.0),
                    "speech" | "voice" => {
                        (features.spectral_centroid_hz / 3_000.0).clamp(0.0, 1.0) * 0.75
                    }
                    "music" => (features.peak + features.rms)
                        .mul_add(0.5, 0.1)
                        .clamp(0.0, 1.0),
                    "noise" | "background" => (features.zero_crossing_rate + 0.25).clamp(0.0, 1.0),
                    _ => 0.35,
                };
                AudioClassPrediction {
                    label: normalized,
                    score,
                }
            })
            .collect::<Vec<_>>()
    };
    predictions.sort_by(|left, right| right.score.total_cmp(&left.score));
    predictions.truncate(top_k.max(1));
    predictions
}

fn spectral_summary_embedding(
    features: &AudioFeatureSummary,
    dimensions: usize,
    normalize: bool,
) -> Vec<f32> {
    let base = [
        features.duration_seconds / 60.0,
        features.sample_rate as f32 / 48_000.0,
        features.rms,
        features.peak,
        features.zero_crossing_rate,
        features.dominant_frequency_hz / 8_000.0,
        features.spectral_centroid_hz / 8_000.0,
    ];
    let mut values = (0..dimensions)
        .map(|index| {
            let seed = base[index % base.len()];
            let phase = (index as f32 + 1.0) * 0.173;
            (seed + phase.sin() * 0.05).clamp(-1.0, 1.0)
        })
        .collect::<Vec<_>>();
    if normalize {
        let _ = normalize_values(&mut values);
    }
    values
}

fn normalize_prediction_label(label: &str) -> String {
    label.trim().to_ascii_lowercase().replace(' ', "_")
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use audio_contracts::{AudioBuffer, OwnedAudioFrame, Timebase};

    #[test]
    fn task_classification_uses_imported_predictions() {
        let response = classify_audio(AudioClassificationRequest {
            source: None,
            labels: Vec::new(),
            top_k: 1,
            features: None,
            model: AudioRuntimeSelection::default(),
            imported_predictions: vec![ImportedAudioPrediction {
                kind: None,
                label: "Door Knock".to_string(),
                score: 0.82,
                start_seconds: None,
                end_seconds: None,
                text: None,
                attributes: BTreeMap::new(),
            }],
        })
        .unwrap();

        assert_eq!(response.runtime, AudioRuntime::Imported);
        assert_eq!(response.predictions[0].label, "door_knock");
    }

    #[test]
    fn task_event_detection_uses_energy_fallback() {
        let response = detect_audio_events(AudioEventDetectionRequest {
            source: None,
            frames: vec![AudioFeatureFrame {
                start_seconds: 1.0,
                end_seconds: 2.0,
                rms: 0.5,
                peak: 0.6,
            }],
            threshold: 0.2,
            model: AudioRuntimeSelection {
                fallback_policy: FallbackPolicy::HeuristicFallback,
                ..AudioRuntimeSelection::default()
            },
            imported_predictions: Vec::new(),
        })
        .unwrap();

        assert_eq!(response.runtime, AudioRuntime::Heuristic);
        assert_eq!(response.events[0].label, "high_energy");
    }

    #[test]
    fn task_transcription_imports_segments() {
        let response = transcribe_audio(SpeechRecognitionRequest {
            source: Some("clip.wav".to_string()),
            language: Some("en".to_string()),
            model: AudioRuntimeSelection::default(),
            imported_segments: vec![TranscriptSegmentContract {
                index: 0,
                start_seconds: Some(0.0),
                end_seconds: Some(1.0),
                text: "hello".to_string(),
                language: Some("en".to_string()),
                speaker: None,
                confidence: Some(0.9),
                is_final: true,
                words: Vec::new(),
                chars: Vec::new(),
                attributes: BTreeMap::new(),
            }],
        })
        .unwrap();

        assert_eq!(response.runtime, AudioRuntime::Imported);
        assert_eq!(response.text(), "hello");
    }

    #[test]
    fn speech_recognition_request_delegates_to_generic_transcription() {
        let response = transcribe_audio(SpeechRecognitionRequest {
            source: Some("clip.wav".to_string()),
            language: Some("en".to_string()),
            model: AudioRuntimeSelection::default(),
            imported_segments: vec![TranscriptSegmentContract::new(0, "delegated")],
        })
        .unwrap();

        assert_eq!(response.runtime, AudioRuntime::Imported);
        assert_eq!(response.text(), "delegated");
        assert_eq!(response.transcript.source.as_deref(), Some("clip.wav"));
    }

    fn assert_approx_eq(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    fn sine(freq_hz: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let samples = (sample_rate as f32 * seconds) as usize;
        (0..samples)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    fn embedder() -> SpectralAudioEmbedder {
        SpectralAudioEmbedder::new(SpectralEmbeddingConfig::new(512, 256, 8).unwrap()).unwrap()
    }

    #[test]
    fn embedding_validation_rejects_empty_non_finite_and_mismatched_dimensions() {
        assert!(AudioEmbedding::new(Vec::<f32>::new()).is_err());
        assert!(AudioEmbedding::new(vec![f32::NAN]).is_err());
        assert!(AudioEmbedding::new(vec![0.0, 0.0]).is_err());
        let left = AudioEmbedding::new(vec![1.0, 0.0]).unwrap();
        let right = AudioEmbedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        assert!(left.cosine_similarity(&right).is_err());
    }

    #[test]
    fn spectral_embedding_config_validates_dimensions() {
        assert!(SpectralEmbeddingConfig::new(0, 256, 8).is_err());
        assert!(SpectralEmbeddingConfig::new(500, 256, 8).is_err());
        assert!(SpectralEmbeddingConfig::new(512, 0, 8).is_err());
        assert!(SpectralEmbeddingConfig::new(512, 256, 0).is_err());
    }

    #[test]
    fn embeds_samples_with_stable_dimensions() {
        let embedding = embedder()
            .embed_samples(&sine(440.0, 8_000, 0.5), 8_000)
            .unwrap();

        assert_eq!(embedding.dimensions(), 16);
        let norm = embedding
            .values()
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert_approx_eq(norm, 1.0, 1.0e-5);
        assert!(embedding.values().iter().all(|value| value.is_finite()));
        assert!(embedder().embed_samples(&[], 8_000).is_err());
        assert!(embedder().embed_samples(&[f32::INFINITY], 8_000).is_err());
        assert!(embedder().embed_samples(&[0.0; 512], 0).is_err());
    }

    #[test]
    fn compares_similar_audio_above_different_audio() {
        let extractor = embedder();
        let a = sine(440.0, 8_000, 0.5);
        let b = sine(445.0, 8_000, 0.5);
        let c = sine(1_600.0, 8_000, 0.5);

        let similar = compare_audio_samples(&a, &b, 8_000, &extractor).unwrap();
        let different = compare_audio_samples(&a, &c, 8_000, &extractor).unwrap();

        assert!(similar.score > different.score);
    }

    #[test]
    fn parses_fpcalc_json_records() {
        let parsed = parse_fpcalc_json(
            r#"{"timestamp": 0.0, "duration": 10.0, "fingerprint": [1]}
{"timestamp": 10.0, "duration": 10.0, "fingerprint": [2]}"#,
        )
        .unwrap();

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].timestamp, Some(10.0));
        assert_eq!(parsed[1].raw, vec![2]);
    }

    #[test]
    fn shifted_similarity_orders_identical_shifted_and_unrelated() {
        let left = vec![1_i64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let mut shifted = vec![0_i64, 0];
        shifted.extend(left.iter());
        let unrelated = vec![i64::MAX; left.len()];

        assert_eq!(shifted_similarity(&left, &left), 1.0);
        assert!(shifted_similarity(&left, &shifted) > 0.95);
        assert!(shifted_similarity(&left, &unrelated) < 0.5);
        assert!(duration_prefilter(100.0, 108.0));
        assert!(!duration_prefilter(100.0, 140.0));
    }

    #[test]
    fn reference_library_searches_nearest_sample() {
        let extractor = embedder();
        let mut library = AudioReferenceLibrary::new();
        library
            .add_reference_samples(
                "low",
                "Low tone",
                &sine(220.0, 8_000, 0.5),
                8_000,
                &extractor,
            )
            .unwrap();
        library
            .add_reference_samples(
                "high",
                "High tone",
                &sine(1_600.0, 8_000, 0.5),
                8_000,
                &extractor,
            )
            .unwrap();
        let query = extractor
            .embed_samples(&sine(225.0, 8_000, 0.5), 8_000)
            .unwrap();

        let matches = library
            .search(&query, &AudioMatchOptions::default())
            .unwrap();

        assert_eq!(matches[0].reference_id, "low");
    }

    #[test]
    fn reference_library_rejects_duplicate_and_mixed_dimensions() {
        let mut library = AudioReferenceLibrary::new();
        library
            .add_reference_embedding("a", "A", AudioEmbedding::new(vec![1.0, 0.0]).unwrap())
            .unwrap();
        assert!(library
            .add_reference_embedding(
                "a",
                "Duplicate",
                AudioEmbedding::new(vec![1.0, 0.0]).unwrap()
            )
            .is_err());
        assert!(library
            .add_reference_embedding("b", "B", AudioEmbedding::new(vec![1.0, 0.0, 0.0]).unwrap())
            .is_err());
        let mut reference = AudioReference::new("mixed", "Mixed")
            .with_embedding(AudioEmbedding::new(vec![1.0, 0.0]).unwrap())
            .unwrap();
        assert!(reference
            .add_embedding(AudioEmbedding::new(vec![1.0, 0.0, 0.0]).unwrap())
            .is_err());
    }

    #[test]
    fn match_options_validate_score_range_and_result_count() {
        assert!(AudioMatchOptions::new(-1.1).is_err());
        assert!(AudioMatchOptions::new(1.1).is_err());
        assert!(AudioMatchOptions::new(f32::NAN).is_err());

        let mut library = AudioReferenceLibrary::new();
        library
            .add_reference_embedding("a", "A", AudioEmbedding::new(vec![1.0, 0.0]).unwrap())
            .unwrap();
        let query = AudioEmbedding::new(vec![1.0, 0.0]).unwrap();
        assert!(library
            .search(
                &query,
                &AudioMatchOptions {
                    min_score: -1.0,
                    max_results: 0,
                },
            )
            .is_err());
    }

    #[test]
    fn search_order_is_deterministic_for_equal_scores() {
        let mut library = AudioReferenceLibrary::new();
        library
            .add_reference_embedding("b", "B", AudioEmbedding::new(vec![1.0, 0.0]).unwrap())
            .unwrap();
        library
            .add_reference_embedding("a", "A", AudioEmbedding::new(vec![1.0, 0.0]).unwrap())
            .unwrap();
        let matches = library
            .search(
                &AudioEmbedding::new(vec![1.0, 0.0]).unwrap(),
                &AudioMatchOptions {
                    min_score: -1.0,
                    max_results: 2,
                },
            )
            .unwrap();
        assert_eq!(matches[0].reference_id, "a");
        assert_eq!(matches[1].reference_id, "b");
    }

    #[test]
    fn adding_extra_sample_improves_match() {
        let extractor = embedder();
        let mut library = AudioReferenceLibrary::new();
        library
            .add_reference_samples("tone", "Tone", &sine(220.0, 8_000, 0.5), 8_000, &extractor)
            .unwrap();
        let query = extractor
            .embed_samples(&sine(440.0, 8_000, 0.5), 8_000)
            .unwrap();
        let before = library
            .search(
                &query,
                &AudioMatchOptions {
                    min_score: -1.0,
                    max_results: 1,
                },
            )
            .unwrap()[0]
            .score;

        library
            .add_sample_from_samples("tone", &sine(440.0, 8_000, 0.5), 8_000, &extractor)
            .unwrap();
        let after = library
            .search(
                &query,
                &AudioMatchOptions {
                    min_score: -1.0,
                    max_results: 1,
                },
            )
            .unwrap()[0]
            .score;

        assert!(after > before);
    }

    #[test]
    fn analyzer_emits_recognition_events_from_streaming_windows() {
        let extractor = embedder();
        let mut library = AudioReferenceLibrary::new();
        let reference = AudioReference::new("a4", "A4")
            .attribute("source", "synthetic")
            .with_samples(&sine(440.0, 8_000, 0.5), 8_000, &extractor)
            .unwrap();
        library.add_reference(reference).unwrap();
        let config = StreamingFrameConfig::new(512, 512).unwrap();
        let mut analyzer =
            AudioRecognitionAnalyzer::new("audio_identity", extractor, library, config).unwrap();
        let samples = AudioBuffer::F32(sine(440.0, 8_000, 0.5));
        let frame = OwnedAudioFrame::new(
            Timestamp::new(0, Timebase::new(1, 8_000)),
            8_000,
            1,
            samples,
        )
        .unwrap();

        let events = analyzer.process_frame(&frame.as_frame().unwrap()).unwrap();

        assert!(!events.is_empty());
        assert_eq!(events[0].analyzer, "audio_identity");
        assert_eq!(
            events[0].timestamp,
            Some(Timestamp::new(0, Timebase::new(1, 8_000)))
        );
        assert!(events[0].score.unwrap() >= 0.85);
        assert!(events[0].label.starts_with("audio:recognized:a4:A4"));
        assert_eq!(
            analyzer
                .library()
                .reference("a4")
                .unwrap()
                .attributes()
                .get("source"),
            Some(&"synthetic".to_string())
        );
    }

    #[test]
    fn analyzer_reset_clears_streaming_window_state() {
        let extractor = embedder();
        let mut library = AudioReferenceLibrary::new();
        library
            .add_reference_samples("a4", "A4", &sine(440.0, 8_000, 0.5), 8_000, &extractor)
            .unwrap();
        let config = StreamingFrameConfig::new(512, 512).unwrap();
        let mut analyzer =
            AudioRecognitionAnalyzer::new("audio_identity", extractor, library, config).unwrap();
        let frame = OwnedAudioFrame::new(
            Timestamp::new(0, Timebase::new(1, 8_000)),
            8_000,
            1,
            AudioBuffer::F32(sine(440.0, 8_000, 0.5)),
        )
        .unwrap();

        assert!(!analyzer
            .process_frame(&frame.as_frame().unwrap())
            .unwrap()
            .is_empty());
        analyzer.reset();
        assert!(!analyzer
            .process_frame(&frame.as_frame().unwrap())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn reference_library_round_trips_through_json_snapshot() {
        let extractor = embedder();
        let mut library = AudioReferenceLibrary::new();
        library
            .add_reference_samples("a4", "A4", &sine(440.0, 8_000, 0.25), 8_000, &extractor)
            .unwrap();
        let json = library.to_json_string().unwrap();
        let restored = AudioReferenceLibrary::from_json_str(&json).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.dimensions(), library.dimensions());
        assert_eq!(restored.reference("a4").unwrap().label(), "A4");
    }
}
