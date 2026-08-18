#![doc = include_str!("../README.md")]

pub mod surface;

mod native_f5;
mod native_vocos;

pub use native_f5::{
    run_f5_mel_diagnostic, NativeF5BundleReport, NativeF5ConfigReport, NativeF5DeviceReport,
    NativeF5MelDiagnosticOutput, NativeF5MelDiagnosticRequest, NativeF5MelReport,
};
pub use native_vocos::{
    run_vocos_vocoder_diagnostic, NativeVocosAudioFrameReport, NativeVocosBundleReport,
    NativeVocosConfigReport, NativeVocosDeviceReport, NativeVocosMelInput, NativeVocosMelReport,
    NativeVocosVocoderDiagnosticOutput, NativeVocosVocoderDiagnosticRequest,
    NativeVocosWeightsReport,
};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// In-memory PCM audio used by TTS inputs and outputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PcmAudio {
    /// Samples per second.
    pub sample_rate_hz: u32,
    /// Channel count.
    pub channels: u16,
    /// Interleaved normalized PCM samples.
    pub samples: Vec<f32>,
}

impl PcmAudio {
    /// Validates this PCM buffer.
    pub fn validate(&self, field: &str) -> Result<(), String> {
        if self.sample_rate_hz == 0 {
            return Err(format!(
                "invalid request: `{field}.sampleRateHz` must be greater than zero"
            ));
        }
        if self.channels == 0 {
            return Err(format!(
                "invalid request: `{field}.channels` must be greater than zero"
            ));
        }
        if self.samples.is_empty() {
            return Err(format!(
                "invalid request: `{field}.samples` must not be empty"
            ));
        }
        if self.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(format!(
                "invalid request: `{field}.samples` must contain only finite values"
            ));
        }
        Ok(())
    }
}

/// Reference audio source used by a Reference Voice Prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", untagged)]
pub enum ReferenceVoicePromptAudio {
    /// In-memory PCM samples supplied by the caller.
    Samples(PcmAudio),
    /// Path to caller-managed reference audio.
    Path { path: String },
}

impl ReferenceVoicePromptAudio {
    /// Validates the reference audio source without opening files.
    pub fn validate(&self, field: &str) -> Result<(), String> {
        match self {
            Self::Samples(audio) => audio.validate(field),
            Self::Path { path } => {
                if path.trim().is_empty() {
                    return Err(format!(
                        "invalid request: `{field}.path` must not be empty when provided"
                    ));
                }
                Ok(())
            }
        }
    }

    /// Stable source-kind string for package-surface plans.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Samples(_) => "samples",
            Self::Path { .. } => "path",
        }
    }

    /// Converts this source into the transcription crate source contract.
    #[cfg(feature = "asr")]
    pub fn to_transcription_source(&self) -> audio_analysis_transcription::TranscriptionSource {
        match self {
            Self::Samples(audio) => audio_analysis_transcription::TranscriptionSource::Samples {
                samples: audio.samples.clone(),
                sample_rate: audio.sample_rate_hz,
                channels: audio.channels,
                source: None,
            },
            Self::Path { path } => audio_analysis_transcription::TranscriptionSource::Path {
                path: PathBuf::from(path),
            },
        }
    }
}

/// ASR fallback configuration for a transcript-missing Reference Voice Prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReferencePromptAsrFallback {
    /// Planned transcription provider id from the transcription crate surface.
    #[serde(default = "default_reference_prompt_asr_provider")]
    pub provider_id: String,
    /// Optional model id to pass to the planned transcription provider.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Optional language hint for ASR fallback.
    #[serde(default)]
    pub language: Option<String>,
}

impl Default for ReferencePromptAsrFallback {
    fn default() -> Self {
        Self {
            provider_id: default_reference_prompt_asr_provider(),
            model_id: None,
            language: None,
        }
    }
}

impl ReferencePromptAsrFallback {
    /// Validates ASR fallback planning fields.
    pub fn validate(&self) -> Result<(), String> {
        if self.provider_id.trim().is_empty() {
            return Err(
                "invalid request: `referenceVoicePrompt.asrFallback.providerId` must not be empty"
                    .to_string(),
            );
        }
        if self
            .model_id
            .as_deref()
            .is_some_and(|model_id| model_id.trim().is_empty())
        {
            return Err(
                "invalid request: `referenceVoicePrompt.asrFallback.modelId` must not be empty when provided"
                    .to_string(),
            );
        }
        if self
            .language
            .as_deref()
            .is_some_and(|language| language.trim().is_empty())
        {
            return Err(
                "invalid request: `referenceVoicePrompt.asrFallback.language` must not be empty when provided"
                    .to_string(),
            );
        }
        Ok(())
    }
}

fn default_reference_prompt_asr_provider() -> String {
    "candle-whisper".to_string()
}

/// Reference Voice Prompt supplied by a package consumer for speaker conditioning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReferenceVoicePrompt {
    /// Caller-supplied reference audio.
    pub audio: ReferenceVoicePromptAudio,
    /// Optional transcript for the supplied reference audio.
    #[serde(default)]
    pub transcript: Option<String>,
    /// Optional BCP-47-ish language hint for transcript or ASR fallback planning.
    #[serde(default)]
    pub language: Option<String>,
    /// Optional ASR fallback used when the reference transcript is missing.
    #[serde(default)]
    pub asr_fallback: Option<ReferencePromptAsrFallback>,
    /// Optional caller metadata such as source id or preparation notes.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl ReferenceVoicePrompt {
    /// Returns whether the prompt includes a non-empty transcript.
    pub fn has_transcript(&self) -> bool {
        self.transcript
            .as_deref()
            .is_some_and(|transcript| !transcript.trim().is_empty())
    }

    /// Validates source, transcript text, language hints, and fallback fields.
    pub fn validate_source_and_hints(&self) -> Result<(), String> {
        self.audio.validate("referenceVoicePrompt.audio")?;
        if self
            .transcript
            .as_deref()
            .is_some_and(|transcript| transcript.trim().is_empty())
        {
            return Err(
                "invalid request: `referenceVoicePrompt.transcript` must not be empty when provided"
                    .to_string(),
            );
        }
        if self
            .language
            .as_deref()
            .is_some_and(|language| language.trim().is_empty())
        {
            return Err(
                "invalid request: `referenceVoicePrompt.language` must not be empty when provided"
                    .to_string(),
            );
        }
        if let Some(fallback) = &self.asr_fallback {
            fallback.validate()?;
        }
        Ok(())
    }

    /// Validates this prompt for synthesis setup.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_source_and_hints()?;
        if !self.has_transcript() && self.asr_fallback.is_none() {
            return Err(
                "setup_error: `referenceVoicePrompt.transcript` is required unless `referenceVoicePrompt.asrFallback` is configured"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Native TTS execution device preference used for planning.
///
/// `Auto` is CUDA-preferred when CUDA support is built and a CUDA device is
/// available; otherwise providers should fall back to CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NativeTtsDevicePreference {
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl NativeTtsDevicePreference {
    /// Stable protocol string used by package surfaces.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

/// Model bundle behavior requested by a package consumer for native TTS planning.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TtsModelBundleSelection {
    /// Optional caller-provided local model bundle path.
    #[serde(default)]
    pub bundle_path: Option<String>,
    /// Whether a later native provider may materialize missing model files.
    #[serde(default)]
    pub auto_download: bool,
    /// Whether planning and execution must use already-cached files only.
    #[serde(default)]
    pub cache_only: bool,
}

impl TtsModelBundleSelection {
    /// Validates model-bundle planning options without touching the filesystem.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_field("provider.modelBundle")
    }

    fn validate_field(&self, field: &str) -> Result<(), String> {
        if self
            .bundle_path
            .as_deref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err(format!(
                "invalid request: `{field}.bundlePath` must not be empty when provided"
            ));
        }
        Ok(())
    }
}

/// Provider selection requested by the package consumer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisProviderSelection {
    /// Provider id, for example `generic`, `speaker-conditioned`, `f5`, or a downstream id.
    pub provider_id: String,
    /// Optional model id. Later slices add explicit model preset metadata.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Whether native provider execution is requested.
    #[serde(default)]
    pub native: bool,
    /// Native execution device preference for planning.
    #[serde(default)]
    pub device: NativeTtsDevicePreference,
    /// Model-bundle resolution and download policy for native planning.
    #[serde(default)]
    pub model_bundle: TtsModelBundleSelection,
    /// Optional vocoder selection used by native speaker-conditioned providers.
    #[serde(default)]
    pub vocoder: Option<SpeechSynthesisVocoderSelection>,
}

impl Default for SpeechSynthesisProviderSelection {
    fn default() -> Self {
        Self {
            provider_id: "generic".to_string(),
            model_id: None,
            native: false,
            device: NativeTtsDevicePreference::Auto,
            model_bundle: TtsModelBundleSelection::default(),
            vocoder: None,
        }
    }
}

impl SpeechSynthesisProviderSelection {
    /// Validates this provider selection.
    pub fn validate(&self) -> Result<(), String> {
        if self.provider_id.trim().is_empty() {
            return Err("invalid request: `provider.providerId` must not be empty".to_string());
        }
        if self
            .model_id
            .as_deref()
            .is_some_and(|model_id| model_id.trim().is_empty())
        {
            return Err(
                "invalid request: `provider.modelId` must not be empty when provided".to_string(),
            );
        }
        self.model_bundle.validate()?;
        if let Some(vocoder) = &self.vocoder {
            vocoder.validate()?;
        }
        Ok(())
    }
}

/// Vocoder requested by a native speech synthesis provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisVocoderSelection {
    /// Vocoder provider id. `vocos` is the first native vocoder path.
    #[serde(default = "default_vocoder_provider_id")]
    pub provider_id: String,
    /// Vocoder model preset id.
    #[serde(default = "default_vocoder_model_id")]
    pub model_id: String,
    /// Model-bundle resolution and download policy for the vocoder.
    #[serde(default)]
    pub model_bundle: TtsModelBundleSelection,
}

impl Default for SpeechSynthesisVocoderSelection {
    fn default() -> Self {
        Self {
            provider_id: default_vocoder_provider_id(),
            model_id: default_vocoder_model_id(),
            model_bundle: TtsModelBundleSelection::default(),
        }
    }
}

impl SpeechSynthesisVocoderSelection {
    /// Validates this vocoder selection.
    pub fn validate(&self) -> Result<(), String> {
        if self.provider_id.trim().is_empty() {
            return Err(
                "invalid request: `provider.vocoder.providerId` must not be empty".to_string(),
            );
        }
        if self.model_id.trim().is_empty() {
            return Err(
                "invalid request: `provider.vocoder.modelId` must not be empty".to_string(),
            );
        }
        self.model_bundle
            .validate_field("provider.vocoder.modelBundle")
    }
}

fn default_vocoder_provider_id() -> String {
    "vocos".to_string()
}

fn default_vocoder_model_id() -> String {
    "vocos-mel-24khz".to_string()
}

/// Inference controls that package consumers can pass through to TTS providers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisOptions {
    /// Desired output sample rate.
    #[serde(default = "default_sample_rate_hz")]
    pub sample_rate_hz: u32,
    /// Desired output channel count.
    #[serde(default = "default_channels")]
    pub channels: u16,
    /// Optional deterministic seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optional native sampling step count.
    #[serde(default)]
    pub steps: Option<u32>,
    /// Optional classifier-free guidance strength.
    #[serde(default)]
    pub cfg_strength: Option<f32>,
    /// Optional speech speed multiplier.
    #[serde(default)]
    pub speed: Option<f32>,
    /// Optional maximum generated duration.
    #[serde(default)]
    pub max_duration_seconds: Option<f32>,
    /// Whether downstream native providers should remove generated silence.
    #[serde(default)]
    pub remove_silence: bool,
}

impl Default for SpeechSynthesisOptions {
    fn default() -> Self {
        Self {
            sample_rate_hz: default_sample_rate_hz(),
            channels: default_channels(),
            seed: None,
            steps: None,
            cfg_strength: None,
            speed: None,
            max_duration_seconds: None,
            remove_silence: false,
        }
    }
}

impl SpeechSynthesisOptions {
    /// Validates inference options.
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_rate_hz == 0 {
            return Err(
                "invalid request: `options.sampleRateHz` must be greater than zero".to_string(),
            );
        }
        if self.channels == 0 {
            return Err(
                "invalid request: `options.channels` must be greater than zero".to_string(),
            );
        }
        if self.steps.is_some_and(|steps| steps == 0) {
            return Err(
                "invalid request: `options.steps` must be greater than zero when provided"
                    .to_string(),
            );
        }
        for (field, value) in [
            ("options.cfgStrength", self.cfg_strength),
            ("options.speed", self.speed),
            ("options.maxDurationSeconds", self.max_duration_seconds),
        ] {
            if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                return Err(format!(
                    "invalid request: `{field}` must be finite and greater than zero when provided"
                ));
            }
        }
        Ok(())
    }
}

fn default_sample_rate_hz() -> u32 {
    24_000
}

fn default_channels() -> u16 {
    1
}

/// Text-to-speech synthesis request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisRequest {
    /// Target text to synthesize.
    pub text: String,
    /// Optional Reference Voice Prompt for speaker-conditioned synthesis.
    #[serde(default)]
    pub reference_voice_prompt: Option<ReferenceVoicePrompt>,
    /// Provider selection.
    #[serde(default)]
    pub provider: SpeechSynthesisProviderSelection,
    /// Inference options.
    #[serde(default)]
    pub options: SpeechSynthesisOptions,
}

impl SpeechSynthesisRequest {
    /// Validates the request through the public contract.
    pub fn validate(&self) -> Result<(), String> {
        if self.text.trim().is_empty() {
            return Err("invalid request: `text` must not be empty".to_string());
        }
        self.provider.validate()?;
        self.options.validate()?;
        if let Some(prompt) = &self.reference_voice_prompt {
            prompt.validate()?;
        }
        Ok(())
    }

    /// Returns whether the request is speaker-conditioned.
    pub fn is_speaker_conditioned(&self) -> bool {
        self.reference_voice_prompt.is_some()
            || self.provider.provider_id == "speaker-conditioned"
            || self.provider.provider_id == "f5"
            || self.provider.provider_id == "e2"
    }
}

/// Status returned by synthesis operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum SpeechSynthesisStatus {
    Ready,
    SetupRequired,
    UnsupportedRuntime,
}

/// A single setup or runtime diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisDiagnostic {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub help: Option<String>,
}

/// Result of a synthesis request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisOutput {
    pub status: SpeechSynthesisStatus,
    pub provider: SpeechSynthesisProviderSelection,
    #[serde(default)]
    pub audio: Option<PcmAudio>,
    #[serde(default)]
    pub native_diagnostics: Option<SpeechSynthesisNativeDiagnostics>,
    pub diagnostics: Vec<SpeechSynthesisDiagnostic>,
}

/// Resolved native inference controls that were accepted into a provider path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisInferenceControlsReport {
    #[serde(default)]
    pub seed: Option<u64>,
    pub steps: u32,
    pub cfg_strength: f32,
    pub speed: f32,
    pub max_duration_seconds: f32,
    pub remove_silence: bool,
}

impl SpeechSynthesisInferenceControlsReport {
    pub fn from_options(
        options: &SpeechSynthesisOptions,
        default_steps: u32,
        default_cfg_strength: f32,
        default_speed: f32,
        default_max_duration_seconds: f32,
    ) -> Self {
        Self {
            seed: options.seed,
            steps: options.steps.unwrap_or(default_steps),
            cfg_strength: options.cfg_strength.unwrap_or(default_cfg_strength),
            speed: options.speed.unwrap_or(default_speed),
            max_duration_seconds: options
                .max_duration_seconds
                .unwrap_or(default_max_duration_seconds),
            remove_silence: options.remove_silence,
        }
    }
}

const DEFAULT_NATIVE_F5_STEPS: u32 = 32;
const DEFAULT_NATIVE_F5_CFG_STRENGTH: f32 = 2.0;
const DEFAULT_NATIVE_F5_SPEED: f32 = 1.0;
const DEFAULT_NATIVE_F5_MAX_DURATION_SECONDS: f32 = 0.25;

/// Native provider/vocoder diagnostics returned by end-to-end synthesis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisNativeDiagnostics {
    pub provider: String,
    pub model_id: String,
    pub vocoder: String,
    pub vocoder_model_id: String,
    pub inference: SpeechSynthesisInferenceControlsReport,
    #[serde(default)]
    pub text_chunking: SpeechSynthesisTextChunkingReport,
    pub runtime: String,
    pub device: String,
    pub bundle_source: String,
    pub provider_bundle_source: String,
    pub vocoder_bundle_source: String,
}

/// Native/provider-facing chunking report for long target text.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisTextChunkingReport {
    pub strategy: String,
    pub max_characters: usize,
    pub text_character_count: usize,
    pub chunk_count: usize,
    pub preserves_in_memory_output_contract: bool,
    pub chunks: Vec<SpeechSynthesisTextChunkReport>,
}

impl Default for SpeechSynthesisTextChunkingReport {
    fn default() -> Self {
        Self {
            strategy: TTS_TEXT_CHUNKING_STRATEGY.to_string(),
            max_characters: DEFAULT_TTS_TEXT_CHUNK_MAX_CHARS,
            text_character_count: 0,
            chunk_count: 0,
            preserves_in_memory_output_contract: true,
            chunks: Vec::new(),
        }
    }
}

/// A target text chunk span, expressed in character offsets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SpeechSynthesisTextChunkReport {
    pub index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub character_count: usize,
}

const DEFAULT_TTS_TEXT_CHUNK_MAX_CHARS: usize = 120;
const TTS_TEXT_CHUNKING_STRATEGY: &str = "sentence-boundary";

/// Reports deterministic long-text chunking while preserving single-buffer PCM output.
pub fn speech_synthesis_text_chunking_report(text: &str) -> SpeechSynthesisTextChunkingReport {
    let chunks = speech_synthesis_text_chunk_spans(text)
        .into_iter()
        .enumerate()
        .map(
            |(index, (start_char, end_char))| SpeechSynthesisTextChunkReport {
                index,
                start_char,
                end_char,
                character_count: end_char.saturating_sub(start_char),
            },
        )
        .collect::<Vec<_>>();
    SpeechSynthesisTextChunkingReport {
        strategy: TTS_TEXT_CHUNKING_STRATEGY.to_string(),
        max_characters: DEFAULT_TTS_TEXT_CHUNK_MAX_CHARS,
        text_character_count: text.chars().count(),
        chunk_count: chunks.len(),
        preserves_in_memory_output_contract: true,
        chunks,
    }
}

fn speech_synthesis_text_chunks(text: &str) -> Vec<String> {
    speech_synthesis_text_chunk_spans(text)
        .into_iter()
        .filter_map(|(start, end)| text_by_char_range(text, start, end))
        .map(|chunk| chunk.trim().to_string())
        .filter(|chunk| !chunk.is_empty())
        .collect()
}

fn speech_synthesis_text_chunk_spans(text: &str) -> Vec<(usize, usize)> {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let mut start = leading_non_whitespace(&chars, 0);
    while start < chars.len() {
        let hard_end = (start + DEFAULT_TTS_TEXT_CHUNK_MAX_CHARS).min(chars.len());
        if hard_end == chars.len() {
            spans.push((start, chars.len()));
            break;
        }
        let minimum_soft_end = start + (DEFAULT_TTS_TEXT_CHUNK_MAX_CHARS / 3);
        let end = (minimum_soft_end..=hard_end)
            .rev()
            .find(|candidate| is_chunk_boundary(chars[*candidate - 1]))
            .unwrap_or(hard_end);
        spans.push((start, end));
        start = leading_non_whitespace(&chars, end);
    }
    spans
}

fn leading_non_whitespace(chars: &[char], mut index: usize) -> usize {
    while index < chars.len() && chars[index].is_whitespace() {
        index += 1;
    }
    index
}

fn is_chunk_boundary(character: char) -> bool {
    matches!(character, '.' | '?' | '!' | ';' | ':' | '\n') || character.is_whitespace()
}

fn text_by_char_range(text: &str, start_char: usize, end_char: usize) -> Option<&str> {
    let start_byte = byte_index_for_char(text, start_char)?;
    let end_byte = byte_index_for_char(text, end_char)?;
    Some(&text[start_byte..end_byte])
}

fn byte_index_for_char(text: &str, char_index: usize) -> Option<usize> {
    if char_index == text.chars().count() {
        return Some(text.len());
    }
    text.char_indices()
        .nth(char_index)
        .map(|(byte_index, _)| byte_index)
}

/// Trait implemented by concrete TTS providers.
pub trait SpeechSynthesisProvider {
    /// Stable provider id.
    fn provider_id(&self) -> &'static str;

    /// Validates and synthesizes speech or returns explicit setup diagnostics.
    fn synthesize(&self, request: &SpeechSynthesisRequest)
        -> Result<SpeechSynthesisOutput, String>;
}

/// Side-effect-free provider used until native providers are implemented.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnsupportedSpeechSynthesisProvider;

impl SpeechSynthesisProvider for UnsupportedSpeechSynthesisProvider {
    fn provider_id(&self) -> &'static str {
        "unsupported-runtime"
    }

    fn synthesize(
        &self,
        request: &SpeechSynthesisRequest,
    ) -> Result<SpeechSynthesisOutput, String> {
        request.validate()?;
        let status = if request.provider.native || reference_prompt_asr_unavailable(request) {
            SpeechSynthesisStatus::SetupRequired
        } else {
            SpeechSynthesisStatus::UnsupportedRuntime
        };
        Ok(SpeechSynthesisOutput {
            status,
            provider: request.provider.clone(),
            audio: None,
            native_diagnostics: None,
            diagnostics: unsupported_provider_diagnostics(request),
        })
    }
}

/// Native F5 mel generation plus Vocos vocoding provider.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeF5VocosSpeechSynthesisProvider;

impl SpeechSynthesisProvider for NativeF5VocosSpeechSynthesisProvider {
    fn provider_id(&self) -> &'static str {
        "f5"
    }

    fn synthesize(
        &self,
        request: &SpeechSynthesisRequest,
    ) -> Result<SpeechSynthesisOutput, String> {
        request.validate()?;
        let Some(reference_prompt) = &request.reference_voice_prompt else {
            return Ok(SpeechSynthesisOutput {
                status: SpeechSynthesisStatus::SetupRequired,
                provider: request.provider.clone(),
                audio: None,
                native_diagnostics: Some(native_diagnostics_for_request(
                    request,
                    "notRun",
                    "unresolved",
                    "unresolved",
                )),
                diagnostics: vec![SpeechSynthesisDiagnostic {
                    code: "reference_voice_prompt_missing".to_string(),
                    message: "Native F5 synthesis requires a Reference Voice Prompt.".to_string(),
                    help: Some(
                        "Provide referenceVoicePrompt audio and transcript before running native synthesis."
                            .to_string(),
                    ),
                }],
            });
        };
        if !reference_prompt.has_transcript() {
            return Ok(SpeechSynthesisOutput {
                status: SpeechSynthesisStatus::SetupRequired,
                provider: request.provider.clone(),
                audio: None,
                native_diagnostics: Some(native_diagnostics_for_request(
                    request,
                    "notRun",
                    "unresolved",
                    "unresolved",
                )),
                diagnostics: vec![SpeechSynthesisDiagnostic {
                    code: "reference_prompt_transcript_required".to_string(),
                    message:
                        "Native F5 synthesis requires a transcript for the Reference Voice Prompt."
                            .to_string(),
                    help: Some(
                        "Run or provide ASR before calling audio.tts.synthesize; this operation does not transcribe the reference prompt."
                            .to_string(),
                    ),
                }],
            });
        }

        let vocoder = request.provider.vocoder.clone().unwrap_or_default();
        if vocoder.provider_id != "vocos" {
            return Ok(SpeechSynthesisOutput {
                status: SpeechSynthesisStatus::UnsupportedRuntime,
                provider: request.provider.clone(),
                audio: None,
                native_diagnostics: Some(native_diagnostics_for_request(
                    request,
                    "notRun",
                    provider_bundle_source(&request.provider.model_bundle),
                    provider_bundle_source(&vocoder.model_bundle),
                )),
                diagnostics: vec![SpeechSynthesisDiagnostic {
                    code: "tts_vocoder_unsupported".to_string(),
                    message: format!(
                        "unsupported TTS vocoder `{}`; use `vocos`",
                        vocoder.provider_id
                    ),
                    help: Some(
                        "F5 synthesis is currently wired to the Vocos native vocoder path."
                            .to_string(),
                    ),
                }],
            });
        }

        let f5_model_id = request
            .provider
            .model_id
            .clone()
            .unwrap_or_else(|| "f5-tts-v1-base".to_string());
        let f5_bundle_source = provider_bundle_source(&request.provider.model_bundle);
        let vocoder_bundle_source = provider_bundle_source(&vocoder.model_bundle);
        let mut selected_device = "notRun".to_string();
        let mut combined_audio: Option<PcmAudio> = None;
        let mut output_diagnostics = Vec::new();

        for chunk in speech_synthesis_text_chunks(&request.text) {
            let f5_request = NativeF5MelDiagnosticRequest {
                text: chunk,
                model_id: f5_model_id.clone(),
                bundle_path: request
                    .provider
                    .model_bundle
                    .bundle_path
                    .as_ref()
                    .map(PathBuf::from),
                device: request.provider.device,
                options: request.options.clone(),
            };
            let f5_output = run_f5_mel_diagnostic(&f5_request)?;
            selected_device = f5_output.device.selected.clone();
            let diagnostics = native_diagnostics_from_reports(
                request,
                &selected_device,
                f5_bundle_source,
                vocoder_bundle_source,
            );

            let Some(f5_mel) = f5_output.mel.clone() else {
                return Ok(SpeechSynthesisOutput {
                    status: f5_output.status,
                    provider: request.provider.clone(),
                    audio: None,
                    native_diagnostics: Some(diagnostics),
                    diagnostics: f5_output.diagnostics,
                });
            };
            if f5_output.status != SpeechSynthesisStatus::Ready {
                return Ok(SpeechSynthesisOutput {
                    status: f5_output.status,
                    provider: request.provider.clone(),
                    audio: None,
                    native_diagnostics: Some(diagnostics),
                    diagnostics: f5_output.diagnostics,
                });
            }
            output_diagnostics.extend(f5_output.diagnostics);

            let vocos_request = NativeVocosVocoderDiagnosticRequest {
                model_id: vocoder.model_id.clone(),
                bundle_path: vocoder.model_bundle.bundle_path.as_ref().map(PathBuf::from),
                device: request.provider.device,
                mel: Some(NativeVocosMelInput {
                    frames: f5_mel.frames,
                    channels: f5_mel.channels,
                    values: diagnostic_f5_mel_values(f5_mel.frames, f5_mel.channels),
                }),
                options: request.options.clone(),
            };
            let vocos_output = run_vocos_vocoder_diagnostic(&vocos_request)?;
            selected_device = vocos_output.device.selected.clone();
            if vocos_output.status != SpeechSynthesisStatus::Ready {
                let diagnostics = native_diagnostics_from_reports(
                    request,
                    &selected_device,
                    f5_bundle_source,
                    vocoder_bundle_source,
                );
                return Ok(SpeechSynthesisOutput {
                    status: vocos_output.status,
                    provider: request.provider.clone(),
                    audio: vocos_output.audio,
                    native_diagnostics: Some(diagnostics),
                    diagnostics: vocos_output.diagnostics,
                });
            }
            if let Some(audio) = vocos_output.audio {
                append_pcm_audio(&mut combined_audio, audio)?;
            }
            output_diagnostics.extend(vocos_output.diagnostics);
        }

        let diagnostics = native_diagnostics_from_reports(
            request,
            &selected_device,
            f5_bundle_source,
            vocoder_bundle_source,
        );
        Ok(SpeechSynthesisOutput {
            status: SpeechSynthesisStatus::Ready,
            provider: request.provider.clone(),
            audio: combined_audio,
            native_diagnostics: Some(diagnostics),
            diagnostics: output_diagnostics,
        })
    }
}

/// Validates a request and returns the current default synthesis response.
pub fn synthesize(request: &SpeechSynthesisRequest) -> Result<SpeechSynthesisOutput, String> {
    if request.provider.native && request.provider.provider_id == "f5" {
        NativeF5VocosSpeechSynthesisProvider.synthesize(request)
    } else {
        UnsupportedSpeechSynthesisProvider.synthesize(request)
    }
}

fn native_diagnostics_for_request(
    request: &SpeechSynthesisRequest,
    device: &str,
    provider_bundle_source: &str,
    vocoder_bundle_source: &str,
) -> SpeechSynthesisNativeDiagnostics {
    let vocoder = request.provider.vocoder.clone().unwrap_or_default();
    let model_id = request
        .provider
        .model_id
        .clone()
        .unwrap_or_else(|| "f5-tts-v1-base".to_string());
    SpeechSynthesisNativeDiagnostics {
        provider: request.provider.provider_id.clone(),
        model_id,
        vocoder: vocoder.provider_id,
        vocoder_model_id: vocoder.model_id,
        inference: SpeechSynthesisInferenceControlsReport::from_options(
            &request.options,
            DEFAULT_NATIVE_F5_STEPS,
            DEFAULT_NATIVE_F5_CFG_STRENGTH,
            DEFAULT_NATIVE_F5_SPEED,
            DEFAULT_NATIVE_F5_MAX_DURATION_SECONDS,
        ),
        text_chunking: speech_synthesis_text_chunking_report(&request.text),
        runtime: native_runtime(),
        device: device.to_string(),
        bundle_source: combined_bundle_source(provider_bundle_source, vocoder_bundle_source)
            .to_string(),
        provider_bundle_source: provider_bundle_source.to_string(),
        vocoder_bundle_source: vocoder_bundle_source.to_string(),
    }
}

fn native_diagnostics_from_reports(
    request: &SpeechSynthesisRequest,
    device: &str,
    provider_bundle_source: &str,
    vocoder_bundle_source: &str,
) -> SpeechSynthesisNativeDiagnostics {
    native_diagnostics_for_request(
        request,
        device,
        provider_bundle_source,
        vocoder_bundle_source,
    )
}

fn append_pcm_audio(target: &mut Option<PcmAudio>, next: PcmAudio) -> Result<(), String> {
    match target {
        Some(existing) => {
            if existing.sample_rate_hz != next.sample_rate_hz || existing.channels != next.channels
            {
                return Err(
                    "native synthesis chunk output mismatch: sample rate and channels must stay stable"
                        .to_string(),
                );
            }
            existing.samples.extend(next.samples);
        }
        None => {
            *target = Some(next);
        }
    }
    Ok(())
}

fn native_runtime() -> String {
    if cfg!(feature = "candle") {
        "candle".to_string()
    } else {
        "unavailable".to_string()
    }
}

fn diagnostic_f5_mel_values(frames: usize, channels: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(frames * channels);
    for channel in 0..channels {
        for frame in 0..frames {
            let phase = ((channel + 1) * (frame + 1)) as f32 * 0.017;
            values.push(phase.sin() * 0.5);
        }
    }
    values
}

fn provider_bundle_source(bundle: &TtsModelBundleSelection) -> &'static str {
    if bundle.bundle_path.is_some() {
        "explicitBundlePath"
    } else if bundle.auto_download && cfg!(feature = "model-bundles") && !bundle.cache_only {
        "autoResolveRequested"
    } else {
        "unresolved"
    }
}

fn combined_bundle_source(provider_source: &str, vocoder_source: &str) -> &'static str {
    if provider_source == "explicitBundlePath" && vocoder_source == "explicitBundlePath" {
        "explicitBundlePath"
    } else if provider_source == "autoResolveRequested" && vocoder_source == "autoResolveRequested"
    {
        "autoResolveRequested"
    } else if provider_source == "unresolved" && vocoder_source == "unresolved" {
        "unresolved"
    } else {
        "mixed"
    }
}

fn unsupported_provider_diagnostics(
    request: &SpeechSynthesisRequest,
) -> Vec<SpeechSynthesisDiagnostic> {
    let mut diagnostics = vec![SpeechSynthesisDiagnostic {
        code: "tts_provider_not_available".to_string(),
        message: "No native TTS provider is available in this build.".to_string(),
        help: Some(
            "Later slices add explicit model presets and native provider setup.".to_string(),
        ),
    }];
    if request.is_speaker_conditioned() && request.reference_voice_prompt.is_none() {
        diagnostics.push(SpeechSynthesisDiagnostic {
            code: "reference_voice_prompt_missing".to_string(),
            message: "Speaker-conditioned TTS requires a Reference Voice Prompt.".to_string(),
            help: Some(
                "Provide referenceVoicePrompt audio and, when available, a transcript.".to_string(),
            ),
        });
    }
    if reference_prompt_asr_unavailable(request) {
        diagnostics.push(SpeechSynthesisDiagnostic {
            code: "reference_prompt_asr_unavailable".to_string(),
            message: "Reference Voice Prompt ASR fallback is configured but unavailable in this build."
                .to_string(),
            help: Some(
                "Build audio-generation-tts with the `asr` feature to plan fallback through audio-analysis-transcription."
                    .to_string(),
            ),
        });
    }
    diagnostics
}

fn reference_prompt_asr_unavailable(request: &SpeechSynthesisRequest) -> bool {
    request
        .reference_voice_prompt
        .as_ref()
        .is_some_and(|prompt| {
            !prompt.has_transcript()
                && prompt.asr_fallback.is_some()
                && !reference_prompt_asr_available()
        })
}

fn reference_prompt_asr_available() -> bool {
    cfg!(feature = "asr")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesis_rejects_empty_text() {
        let request = SpeechSynthesisRequest {
            text: String::new(),
            reference_voice_prompt: None,
            provider: SpeechSynthesisProviderSelection::default(),
            options: SpeechSynthesisOptions::default(),
        };

        let error = synthesize(&request).expect_err("empty text");
        assert!(error.contains("text"));
    }

    #[test]
    fn synthesis_returns_explicit_unsupported_runtime_without_audio() {
        let request = SpeechSynthesisRequest {
            text: "Hello from TTS".to_string(),
            reference_voice_prompt: None,
            provider: SpeechSynthesisProviderSelection::default(),
            options: SpeechSynthesisOptions::default(),
        };

        let output = synthesize(&request).expect("unsupported provider response");
        assert_eq!(output.status, SpeechSynthesisStatus::UnsupportedRuntime);
        assert!(output.audio.is_none());
        assert!(output
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "tts_provider_not_available"));
    }

    #[test]
    fn speaker_conditioned_synthesis_requires_transcript_without_asr_fallback() {
        let request = SpeechSynthesisRequest {
            text: "Match this speaker.".to_string(),
            reference_voice_prompt: Some(ReferenceVoicePrompt {
                audio: ReferenceVoicePromptAudio::Samples(PcmAudio {
                    sample_rate_hz: 24_000,
                    channels: 1,
                    samples: vec![0.0, 0.01, -0.01, 0.0],
                }),
                transcript: None,
                language: Some("en".to_string()),
                asr_fallback: None,
                metadata: serde_json::json!({}),
            }),
            provider: SpeechSynthesisProviderSelection {
                provider_id: "speaker-conditioned".to_string(),
                ..SpeechSynthesisProviderSelection::default()
            },
            options: SpeechSynthesisOptions::default(),
        };

        let error = synthesize(&request).expect_err("missing transcript");
        assert!(error.contains("setup_error"));
        assert!(error.contains("referenceVoicePrompt.transcript"));
        assert!(error.contains("referenceVoicePrompt.asrFallback"));
    }

    #[test]
    fn speaker_conditioned_synthesis_reports_asr_unavailable_when_fallback_is_configured() {
        let request = SpeechSynthesisRequest {
            text: "Match this speaker.".to_string(),
            reference_voice_prompt: Some(ReferenceVoicePrompt {
                audio: ReferenceVoicePromptAudio::Samples(PcmAudio {
                    sample_rate_hz: 24_000,
                    channels: 1,
                    samples: vec![0.0, 0.01, -0.01, 0.0],
                }),
                transcript: None,
                language: Some("en".to_string()),
                asr_fallback: Some(ReferencePromptAsrFallback {
                    provider_id: "candle-whisper".to_string(),
                    model_id: Some("openai/whisper-large-v3-turbo".to_string()),
                    language: None,
                }),
                metadata: serde_json::json!({}),
            }),
            provider: SpeechSynthesisProviderSelection {
                provider_id: "speaker-conditioned".to_string(),
                ..SpeechSynthesisProviderSelection::default()
            },
            options: SpeechSynthesisOptions::default(),
        };

        let output = synthesize(&request).expect("asr fallback setup response");
        assert_eq!(output.status, SpeechSynthesisStatus::SetupRequired);
        assert!(output.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "reference_prompt_asr_unavailable"
                && diagnostic.message.contains("ASR fallback")
        }));
    }

    #[test]
    fn native_f5_synthesis_requires_reference_prompt_transcript_before_provider_execution() {
        let request = SpeechSynthesisRequest {
            text: "Match this speaker.".to_string(),
            reference_voice_prompt: Some(ReferenceVoicePrompt {
                audio: ReferenceVoicePromptAudio::Samples(PcmAudio {
                    sample_rate_hz: 24_000,
                    channels: 1,
                    samples: vec![0.0, 0.01, -0.01, 0.0],
                }),
                transcript: None,
                language: Some("en".to_string()),
                asr_fallback: Some(ReferencePromptAsrFallback {
                    provider_id: "candle-whisper".to_string(),
                    model_id: None,
                    language: None,
                }),
                metadata: serde_json::json!({}),
            }),
            provider: SpeechSynthesisProviderSelection {
                provider_id: "f5".to_string(),
                model_id: Some("f5-tts-v1-base".to_string()),
                native: true,
                device: NativeTtsDevicePreference::Cpu,
                model_bundle: TtsModelBundleSelection {
                    bundle_path: Some("unused-f5".to_string()),
                    auto_download: false,
                    cache_only: false,
                },
                vocoder: Some(SpeechSynthesisVocoderSelection {
                    provider_id: "vocos".to_string(),
                    model_id: "vocos-mel-24khz".to_string(),
                    model_bundle: TtsModelBundleSelection {
                        bundle_path: Some("unused-vocos".to_string()),
                        auto_download: false,
                        cache_only: false,
                    },
                }),
            },
            options: SpeechSynthesisOptions::default(),
        };

        let output = synthesize(&request).expect("native setup response");

        assert_eq!(output.status, SpeechSynthesisStatus::SetupRequired);
        assert!(output.audio.is_none());
        assert!(output.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "reference_prompt_transcript_required"
                && diagnostic.message.contains("transcript")
        }));
    }

    #[test]
    fn native_f5_synthesis_reports_inference_controls_in_diagnostics() {
        let request = SpeechSynthesisRequest {
            text: "Match this speaker.".to_string(),
            reference_voice_prompt: Some(ReferenceVoicePrompt {
                audio: ReferenceVoicePromptAudio::Samples(PcmAudio {
                    sample_rate_hz: 24_000,
                    channels: 1,
                    samples: vec![0.0, 0.01, -0.01, 0.0],
                }),
                transcript: Some("The reference speaker reads this sentence.".to_string()),
                language: Some("en".to_string()),
                asr_fallback: None,
                metadata: serde_json::json!({}),
            }),
            provider: SpeechSynthesisProviderSelection {
                provider_id: "f5".to_string(),
                model_id: Some("f5-tts-v1-base".to_string()),
                native: true,
                device: NativeTtsDevicePreference::Cpu,
                model_bundle: TtsModelBundleSelection {
                    bundle_path: Some("missing-f5".to_string()),
                    auto_download: false,
                    cache_only: false,
                },
                vocoder: Some(SpeechSynthesisVocoderSelection {
                    provider_id: "vocos".to_string(),
                    model_id: "vocos-mel-24khz".to_string(),
                    model_bundle: TtsModelBundleSelection {
                        bundle_path: Some("missing-vocos".to_string()),
                        auto_download: false,
                        cache_only: false,
                    },
                }),
            },
            options: SpeechSynthesisOptions {
                seed: Some(29),
                steps: Some(5),
                cfg_strength: Some(1.75),
                speed: Some(1.1),
                max_duration_seconds: Some(0.3),
                remove_silence: true,
                ..SpeechSynthesisOptions::default()
            },
        };

        let output = synthesize(&request).expect("native setup response");
        let diagnostics = output.native_diagnostics.expect("native diagnostics");

        assert_eq!(diagnostics.inference.seed, Some(29));
        assert_eq!(diagnostics.inference.steps, 5);
        assert_eq!(diagnostics.inference.cfg_strength, 1.75);
        assert_eq!(diagnostics.inference.speed, 1.1);
        assert_eq!(diagnostics.inference.max_duration_seconds, 0.3);
        assert!(diagnostics.inference.remove_silence);
    }

    #[test]
    fn provider_selection_deserializes_native_device_preferences() {
        for (device, expected) in [
            ("auto", NativeTtsDevicePreference::Auto),
            ("cpu", NativeTtsDevicePreference::Cpu),
            ("cuda", NativeTtsDevicePreference::Cuda),
        ] {
            let selection: SpeechSynthesisProviderSelection = serde_json::from_value(
                serde_json::json!({"providerId":"f5","native":true,"device": device}),
            )
            .expect("provider selection");

            assert_eq!(selection.device, expected);
        }
    }

    #[cfg(all(feature = "candle", feature = "cuda", feature = "external-tests"))]
    #[test]
    #[ignore = "requires CUDA-capable host, F5_TTS_BUNDLE, and VOCOS_BUNDLE; run with `cargo test -p moenarch-audio-generation-tts --features candle,cuda,external-tests native_f5_vocos_cuda_preferred_synthesis_smoke_when_requested -- --ignored`"]
    fn native_f5_vocos_cuda_preferred_synthesis_smoke_when_requested() {
        let f5_bundle = std::env::var_os("F5_TTS_BUNDLE")
            .map(PathBuf::from)
            .expect("set F5_TTS_BUNDLE to a local compatible F5 bundle");
        let vocos_bundle = std::env::var_os("VOCOS_BUNDLE")
            .map(PathBuf::from)
            .expect("set VOCOS_BUNDLE to a local compatible Vocos bundle");

        let request = SpeechSynthesisRequest {
            text: "CUDA preferred native synthesis smoke.".to_string(),
            reference_voice_prompt: Some(ReferenceVoicePrompt {
                audio: ReferenceVoicePromptAudio::Samples(PcmAudio {
                    sample_rate_hz: 24_000,
                    channels: 1,
                    samples: vec![0.0, 0.01, -0.01, 0.0],
                }),
                transcript: Some("Reference voice prompt text.".to_string()),
                language: Some("en".to_string()),
                asr_fallback: None,
                metadata: serde_json::json!({}),
            }),
            provider: SpeechSynthesisProviderSelection {
                provider_id: "f5".to_string(),
                model_id: Some(
                    std::env::var("F5_TTS_MODEL_ID")
                        .unwrap_or_else(|_| "f5-tts-v1-base".to_string()),
                ),
                native: true,
                device: NativeTtsDevicePreference::Auto,
                model_bundle: TtsModelBundleSelection {
                    bundle_path: Some(f5_bundle.display().to_string()),
                    auto_download: false,
                    cache_only: false,
                },
                vocoder: Some(SpeechSynthesisVocoderSelection {
                    provider_id: "vocos".to_string(),
                    model_id: std::env::var("VOCOS_MODEL_ID")
                        .unwrap_or_else(|_| "vocos-mel-24khz".to_string()),
                    model_bundle: TtsModelBundleSelection {
                        bundle_path: Some(vocos_bundle.display().to_string()),
                        auto_download: false,
                        cache_only: false,
                    },
                }),
            },
            options: SpeechSynthesisOptions {
                seed: Some(1),
                steps: Some(1),
                cfg_strength: Some(1.0),
                speed: Some(1.0),
                max_duration_seconds: Some(0.02),
                remove_silence: true,
                ..SpeechSynthesisOptions::default()
            },
        };

        let output = synthesize(&request).expect("native CUDA-preferred synthesis");

        assert_eq!(output.status, SpeechSynthesisStatus::Ready);
        assert!(output.audio.is_some());
        let diagnostics = output.native_diagnostics.expect("native diagnostics");
        assert!(diagnostics.device.starts_with("cuda"));
    }
}
