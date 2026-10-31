//! Provider traits, request/response DTOs and the small provider adapters.

use crate::*;

mod candle;
mod diarization;
mod whisper_cpp;
mod whisperx;

pub use candle::*;
#[allow(unused_imports)] // Public providers are feature-gated.
pub use diarization::*;
pub use whisper_cpp::*;
pub use whisperx::*;

/// Loaded audio for native providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    #[serde(default)]
    pub source: Option<String>,
}

impl LoadedAudio {
    pub fn mono_16khz_from_source(source: &TranscriptionSource) -> Result<Self> {
        native_audio::mono_16khz_from_source(source)
    }

    /// Predecodes a finite media path with an optional zero-based audio-stream ordinal.
    ///
    /// Stream validation is completed by the FFmpeg provider before this function
    /// returns, and typed available-stream context is preserved in [`AudioIoError`].
    #[cfg(feature = "audio-io")]
    pub fn mono_16khz_from_selected_media(
        path: impl AsRef<Path>,
        audio_stream_index: Option<usize>,
    ) -> audio_analysis_io::AudioIoResult<Self> {
        native_audio::mono_16khz_from_selected_media(path.as_ref(), audio_stream_index)
    }

    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / self.channels as f64 / self.sample_rate as f64
    }

    /// Borrows these samples as validated audio for voice activity detection.
    pub fn speaker_audio(&self) -> Result<SpeakerAudio<'_>> {
        SpeakerAudio::interleaved(&self.samples, self.sample_rate, self.channels)
    }
}

/// Error returned by selected-media transcription entrypoints.
#[cfg(feature = "audio-io")]
#[derive(Debug)]
pub enum SelectedMediaTranscriptionError {
    /// Media decode or typed audio-stream validation failed.
    Decode(audio_analysis_io::AudioIoError),
    /// The transcription pipeline failed after media validation and decode.
    Pipeline(DetectError),
}

#[cfg(feature = "audio-io")]
impl std::fmt::Display for SelectedMediaTranscriptionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(error) => error.fmt(formatter),
            Self::Pipeline(error) => error.fmt(formatter),
        }
    }
}

#[cfg(feature = "audio-io")]
impl std::error::Error for SelectedMediaTranscriptionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(error) => Some(error),
            Self::Pipeline(error) => Some(error),
        }
    }
}

#[cfg(feature = "audio-io")]
impl From<audio_analysis_io::AudioIoError> for SelectedMediaTranscriptionError {
    fn from(error: audio_analysis_io::AudioIoError) -> Self {
        Self::Decode(error)
    }
}

/// ASR provider request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrRequest {
    pub audio: LoadedAudio,
    pub chunks: Vec<SpeechSpan>,
    #[serde(default)]
    pub task: TranscriptionTask,
    pub language: Option<String>,
    pub model_id: String,
}

/// ASR provider response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrResponse {
    pub model_id: String,
    pub language: Option<String>,
    pub transcript: TranscriptionContract,
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

/// Forced-alignment request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlignmentRequest {
    pub audio: LoadedAudio,
    pub transcript: TranscriptionContract,
    pub language: Option<String>,
    pub model_id: String,
}

/// Forced-alignment response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlignmentResponse {
    pub model_id: String,
    pub words: Vec<AlignedWord>,
    #[serde(default)]
    pub chars: Vec<AlignedChar>,
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

/// One aligned word timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlignedWord {
    pub segment_index: u64,
    pub word_index: usize,
    pub text: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
    #[serde(default)]
    pub confidence: Option<f32>,
}

/// One aligned character timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlignedChar {
    pub segment_index: u64,
    pub char_index: usize,
    pub character: String,
    #[serde(default)]
    pub start_seconds: Option<f64>,
    #[serde(default)]
    pub end_seconds: Option<f64>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

/// Alignment summary included in pipeline responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlignmentSummary {
    pub provider: String,
    pub model_id: String,
    pub word_count: usize,
}

/// Trait for audio transcription providers.
pub trait AudioTranscriptionProvider {
    fn provider_id(&self) -> &str;
    fn transcribe(&mut self, request: AsrRequest) -> Result<AsrResponse>;

    fn transcribe_with_observer(
        &mut self,
        request: AsrRequest,
        _observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AsrResponse> {
        self.transcribe(request)
    }
}

/// Trait for forced alignment providers.
pub trait ForcedAlignmentProvider {
    fn provider_id(&self) -> &str;
    fn align(&mut self, request: AlignmentRequest) -> Result<AlignmentResponse>;

    fn align_with_observer(
        &mut self,
        request: AlignmentRequest,
        _observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AlignmentResponse> {
        self.align(request)
    }
}

/// Trait for transcript diarization and speaker assignment providers.
pub trait TranscriptDiarizationProvider {
    fn provider_id(&self) -> &str;
    fn diarize(
        &mut self,
        audio: LoadedAudio,
        transcript: &TranscriptionContract,
        options: &DiarizationOptions,
    ) -> Result<SpeakerDiarizationResponse>;
}

/// Feature-gated CTC forced aligner.
#[derive(Debug, Clone, Default)]
pub struct CtcForcedAligner {
    pub options: AlignmentOptions,
}

impl ForcedAlignmentProvider for CtcForcedAligner {
    fn provider_id(&self) -> &str {
        "ctc-forced-aligner"
    }

    fn align(&mut self, request: AlignmentRequest) -> Result<AlignmentResponse> {
        let mut observer = NoopTranscriptionPipelineObserver;
        self.align_with_observer(request, &mut observer)
    }

    fn align_with_observer(
        &mut self,
        request: AlignmentRequest,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AlignmentResponse> {
        let _ = &observer;
        #[cfg(feature = "alignment")]
        {
            ctc_alignment::align_with_observer(&self.options, request, observer)
        }
        #[cfg(not(feature = "alignment"))]
        {
            validate_alignment_setup(&self.options)?;
            Err(unsupported_runtime(format!(
            "CTC alignment execution for `{}` is planned behind the alignment provider; default tests use mock alignment providers",
            request.model_id
        )))
        }
    }
}

pub(crate) fn validate_asr_request(request: &AsrRequest) -> Result<()> {
    native_audio::validate_loaded_audio(&request.audio)?;
    if request.chunks.is_empty() {
        return Err(invalid_request(
            "ASR request must contain at least one speech chunk",
        ));
    }
    let duration = request.audio.duration_seconds();
    let tolerance = 1.0 / request.audio.sample_rate as f64;
    for chunk in &request.chunks {
        chunk.validate()?;
        if chunk.end_seconds > duration + tolerance {
            return Err(invalid_request(format!(
                "speech chunk end {:.6} exceeds audio duration {:.6}",
                chunk.end_seconds, duration
            )));
        }
    }
    Ok(())
}

pub(crate) fn normalize_samples_source(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    source: Option<String>,
) -> Result<LoadedAudio> {
    if sample_rate == 0 || channels == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate,
            channels,
        });
    }
    if samples.is_empty() {
        return Err(invalid_request("empty audio"));
    }
    if !samples.len().is_multiple_of(channels as usize) {
        return Err(invalid_request(
            "sample count must contain complete interleaved frames",
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid_request("audio samples must be finite"));
    }
    let mono = if channels == 1 {
        samples.to_vec()
    } else {
        samples
            .chunks_exact(channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect::<Vec<_>>()
    };
    let samples = if sample_rate == 16_000 {
        mono
    } else {
        resample_linear(&mono, sample_rate, 16_000)
    };
    Ok(LoadedAudio {
        samples,
        sample_rate: 16_000,
        channels: 1,
        source,
    })
}

pub(crate) fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if samples.is_empty() || from_rate == to_rate {
        return samples.to_vec();
    }
    let output_len = ((samples.len() as f64 * to_rate as f64) / from_rate as f64)
        .round()
        .max(1.0) as usize;
    (0..output_len)
        .map(|index| {
            let position = index as f64 * from_rate as f64 / to_rate as f64;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (position - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}

#[cfg(not(feature = "alignment"))]
pub(crate) fn validate_alignment_setup(options: &AlignmentOptions) -> Result<()> {
    if !cfg!(feature = "alignment") {
        return Err(unsupported_runtime(
            "CTC alignment requested but the binary lacks the `alignment` feature",
        ));
    }
    if let Some(bundle) = &options.model_bundle {
        validate_model_bundle_files(
            bundle,
            &[
                "config.json",
                "tokenizer.json",
                "preprocessor_config.json",
                "model.safetensors",
            ],
        )
    } else {
        Err(setup_error(
            "required CTC alignment model bundle is missing",
        ))
    }
}

pub(crate) fn validate_model_bundle_files(bundle: &Path, files: &[&str]) -> Result<()> {
    native_bundles::validate_required_bundle_files(bundle, files)
}
