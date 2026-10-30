//! Candle Whisper provider, decode and runtime options.

use crate::*;

/// Options for the Candle Whisper native provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandleWhisperOptions {
    #[serde(default = "default_candle_whisper_model")]
    pub model_id: String,
    #[serde(default)]
    pub task: TranscriptionTask,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub device: NativeDevicePreference,
    #[serde(default)]
    pub compute_type: CandleWhisperComputeType,
    #[serde(default)]
    pub model_bundle: Option<PathBuf>,
    #[serde(default)]
    pub model_dir: Option<PathBuf>,
    #[serde(default)]
    pub model_cache_only: bool,
    #[serde(default)]
    pub batch_chunks: bool,
    #[serde(default)]
    pub max_batch_size: Option<usize>,
    #[serde(default)]
    pub decode_runtime: CandleWhisperDecodeRuntime,
}

impl Default for CandleWhisperOptions {
    fn default() -> Self {
        Self {
            model_id: default_candle_whisper_model(),
            task: TranscriptionTask::Transcribe,
            language: None,
            device: NativeDevicePreference::Auto,
            compute_type: CandleWhisperComputeType::Automatic,
            model_bundle: None,
            model_dir: None,
            model_cache_only: false,
            batch_chunks: true,
            max_batch_size: Some(4),
            decode_runtime: CandleWhisperDecodeRuntime::AutoregressiveKvCache,
        }
    }
}

/// Request-scoped execution controls for the native Candle Whisper provider.
///
/// These controls are separate from [`CandleWhisperOptions`] so existing
/// exhaustive option literals remain source compatible. Omitting them keeps
/// CUDA index `0` and the caller's default Rayon runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandleWhisperRuntimeControls {
    #[serde(default)]
    pub cuda_device_index: usize,
    #[serde(default)]
    pub decoder_threads: Option<usize>,
}

/// Request-scoped native Whisper segment-timing behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CandleWhisperTimingMode {
    /// Prefer Whisper timestamp tokens and retain the existing bounded
    /// chunk-window fallback when timestamp output is unavailable or unstable.
    #[default]
    Auto,
    /// Decode without timestamp tokens and use expanded VAD-window timing.
    NoTimestamps,
    /// Require stable Whisper timestamp-token segments instead of falling back.
    TimestampTokensRequired,
}

impl CandleWhisperTimingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::NoTimestamps => "noTimestamps",
            Self::TimestampTokensRequired => "timestampTokensRequired",
        }
    }
}

/// Request-scoped context and timing controls for native Whisper windows.
///
/// Defaults preserve the existing 250 ms leading and 40 ms trailing ASR
/// context around each VAD chunk.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandleWhisperWindowControls {
    #[serde(default)]
    pub timing_mode: CandleWhisperTimingMode,
    #[serde(default = "default_candle_whisper_leading_context_seconds")]
    pub leading_context_seconds: f64,
    #[serde(default = "default_candle_whisper_trailing_context_seconds")]
    pub trailing_context_seconds: f64,
}

impl Default for CandleWhisperWindowControls {
    fn default() -> Self {
        Self {
            timing_mode: CandleWhisperTimingMode::Auto,
            leading_context_seconds: default_candle_whisper_leading_context_seconds(),
            trailing_context_seconds: default_candle_whisper_trailing_context_seconds(),
        }
    }
}

impl CandleWhisperWindowControls {
    fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("leading_context_seconds", self.leading_context_seconds),
            ("trailing_context_seconds", self.trailing_context_seconds),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid_request(format!(
                    "Candle Whisper {field} must be finite and greater than or equal to zero"
                )));
            }
        }
        Ok(())
    }
}

/// Request-scoped token selection controls for native Candle Whisper decoding.
///
/// The default configuration preserves the existing deterministic greedy path.
/// Positive temperatures enable seeded sampling, while `beam_size > 1` enables
/// beam search for an all-zero temperature schedule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandleWhisperDecodeConfig {
    #[serde(default = "default_candle_whisper_temperature_schedule")]
    pub temperature_schedule: Vec<f64>,
    #[serde(default = "default_candle_whisper_search_width")]
    pub best_of: usize,
    #[serde(default = "default_candle_whisper_search_width")]
    pub beam_size: usize,
    #[serde(default = "default_candle_whisper_score_factor")]
    pub patience: f64,
    #[serde(default = "default_candle_whisper_score_factor")]
    pub length_penalty: f64,
    #[serde(default)]
    pub seed: u64,
}

impl Default for CandleWhisperDecodeConfig {
    fn default() -> Self {
        Self {
            temperature_schedule: default_candle_whisper_temperature_schedule(),
            best_of: default_candle_whisper_search_width(),
            beam_size: default_candle_whisper_search_width(),
            patience: default_candle_whisper_score_factor(),
            length_penalty: default_candle_whisper_score_factor(),
            seed: 0,
        }
    }
}

impl CandleWhisperDecodeConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.temperature_schedule.is_empty() {
            return Err(invalid_request(
                "Candle Whisper temperature_schedule must not be empty",
            ));
        }
        if self
            .temperature_schedule
            .iter()
            .any(|temperature| !temperature.is_finite() || *temperature < 0.0)
        {
            return Err(invalid_request(
                "Candle Whisper temperatures must be finite and greater than or equal to zero",
            ));
        }
        if self.best_of == 0 {
            return Err(invalid_request(
                "Candle Whisper best_of must be greater than zero",
            ));
        }
        if self.beam_size == 0 {
            return Err(invalid_request(
                "Candle Whisper beam_size must be greater than zero",
            ));
        }
        if !self.patience.is_finite() || self.patience <= 0.0 {
            return Err(invalid_request(
                "Candle Whisper patience must be finite and greater than zero",
            ));
        }
        if !self.length_penalty.is_finite() || self.length_penalty < 0.0 {
            return Err(invalid_request(
                "Candle Whisper length_penalty must be finite and greater than or equal to zero",
            ));
        }

        let samples = self
            .temperature_schedule
            .iter()
            .any(|temperature| *temperature > 0.0);
        if self.beam_size > 1 && samples {
            return Err(invalid_request(
                "Candle Whisper beam search requires an all-zero temperature_schedule",
            ));
        }
        if self.beam_size > 1 && self.best_of != 1 {
            return Err(invalid_request(
                "Candle Whisper best_of must be 1 when beam_size is greater than 1",
            ));
        }
        if self.beam_size == 1 && self.best_of > 1 && !samples {
            return Err(invalid_request(
                "Candle Whisper best_of greater than 1 requires a positive temperature",
            ));
        }
        if self.beam_size == 1 && self.patience != 1.0 {
            return Err(invalid_request(
                "Candle Whisper patience only applies when beam_size is greater than 1",
            ));
        }
        if self.beam_size == 1 && self.length_penalty != 1.0 {
            return Err(invalid_request(
                "Candle Whisper length_penalty only applies when beam_size is greater than 1",
            ));
        }
        Ok(())
    }

    #[cfg_attr(not(feature = "candle"), allow(dead_code))]
    pub(crate) fn uses_default_greedy_path(&self) -> bool {
        self.temperature_schedule == [0.0]
            && self.best_of == 1
            && self.beam_size == 1
            && self.patience == 1.0
            && self.length_penalty == 1.0
    }
}

/// Complete request-scoped native Whisper decode configuration.
///
/// This additive wrapper keeps [`CandleWhisperDecodeConfig`] exhaustively
/// constructible while adding prompt, suppression, conditioning, and fallback
/// controls. All state derived from these fields is owned by one transcription
/// request and is never retained in reusable model sessions.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandleWhisperDecodeRequestConfig {
    /// Token search settings used by each ordered fallback attempt.
    #[serde(default)]
    pub search: CandleWhisperDecodeConfig,
    /// Prompt text tokenized by the loaded Whisper tokenizer for this request.
    #[serde(default)]
    pub initial_prompt: Option<String>,
    /// Token IDs placed after `<|startofprev|>` and before Whisper's
    /// SOT/language/task/timestamp controls.
    ///
    /// The prompt and request-local previous-text history are bounded to the
    /// model context; the oldest tokens are truncated first.
    #[serde(default)]
    pub initial_prompt_tokens: Vec<u32>,
    /// Token IDs suppressed for every generation step in this request.
    #[serde(default)]
    pub suppressed_token_ids: Vec<u32>,
    /// Suppresses tokenizer entries containing numerals, excluding special controls.
    #[serde(default)]
    pub suppress_numerals: bool,
    /// Carries bounded generated text tokens into later windows in this request.
    #[serde(default)]
    pub condition_on_previous_text: bool,
    /// Retries at the next temperature when average log probability is lower.
    #[serde(default)]
    pub min_average_log_probability: Option<f64>,
    /// Rejects when SOT no-speech is higher and text confidence is low.
    #[serde(default)]
    pub max_no_speech_probability: Option<f64>,
    /// Retries when the decoded text's zlib-compatible Whisper compression
    /// ratio is higher.
    #[serde(default)]
    pub max_compression_ratio: Option<f64>,
}

impl From<CandleWhisperDecodeConfig> for CandleWhisperDecodeRequestConfig {
    fn from(search: CandleWhisperDecodeConfig) -> Self {
        Self {
            search,
            ..Self::default()
        }
    }
}

impl CandleWhisperDecodeRequestConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        self.search.validate()?;
        if self
            .min_average_log_probability
            .is_some_and(|value| !value.is_finite())
        {
            return Err(invalid_request(
                "Candle Whisper min_average_log_probability must be finite",
            ));
        }
        if self
            .max_no_speech_probability
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(invalid_request(
                "Candle Whisper max_no_speech_probability must be between zero and one",
            ));
        }
        if self
            .max_compression_ratio
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(invalid_request(
                "Candle Whisper max_compression_ratio must be finite and greater than zero",
            ));
        }
        Ok(())
    }

    #[cfg_attr(not(feature = "candle"), allow(dead_code))]
    pub(crate) fn preserves_legacy_greedy_path(&self) -> bool {
        self.search.uses_default_greedy_path()
            && self.initial_prompt.as_ref().is_none_or(String::is_empty)
            && self.initial_prompt_tokens.is_empty()
            && self.suppressed_token_ids.is_empty()
            && !self.suppress_numerals
            && !self.condition_on_previous_text
            && self.min_average_log_probability.is_none()
            && self.max_no_speech_probability.is_none()
            && self.max_compression_ratio.is_none()
    }

    pub(crate) fn validate_runtime(&self, options: &CandleWhisperOptions) -> Result<()> {
        if self.condition_on_previous_text
            && options.decode_runtime == CandleWhisperDecodeRuntime::ActiveRowTensorBatch
        {
            return Err(invalid_request(
                "Candle Whisper condition_on_previous_text requires autoregressiveKvCache decode runtime",
            ));
        }
        Ok(())
    }
}

/// Complete request-scoped configuration for native Candle Whisper execution.
///
/// The aggregate keeps runtime, decode, and window concerns behind one
/// canonical transcription interface while the narrower legacy methods remain
/// source compatible by delegating with defaults.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandleWhisperTranscriptionRequestConfig {
    #[serde(default)]
    pub runtime: CandleWhisperRuntimeControls,
    #[serde(default)]
    pub decode: CandleWhisperDecodeRequestConfig,
    #[serde(default)]
    pub window: CandleWhisperWindowControls,
}

impl CandleWhisperTranscriptionRequestConfig {
    pub(crate) fn validate(&self, options: &CandleWhisperOptions) -> Result<()> {
        self.decode.validate()?;
        self.decode.validate_runtime(options)?;
        self.window.validate()
    }
}

pub(crate) fn default_candle_whisper_temperature_schedule() -> Vec<f64> {
    vec![0.0]
}

pub(crate) const fn default_candle_whisper_leading_context_seconds() -> f64 {
    0.25
}

pub(crate) const fn default_candle_whisper_trailing_context_seconds() -> f64 {
    0.04
}

pub(crate) const fn default_candle_whisper_search_width() -> usize {
    1
}

pub(crate) const fn default_candle_whisper_score_factor() -> f64 {
    1.0
}

pub(crate) fn default_candle_whisper_model() -> String {
    "openai/whisper-large-v3-turbo".to_string()
}

/// Native Candle Whisper compute-type preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CandleWhisperComputeType {
    #[default]
    #[serde(alias = "auto")]
    Automatic,
    #[serde(alias = "float16")]
    Fp16,
    #[serde(alias = "float32")]
    Fp32,
    Int8,
}

impl CandleWhisperComputeType {
    const SAFETENSORS_BUNDLE_FILES: &'static [&'static str] = &[
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "model.safetensors",
    ];
    const Q8_BUNDLE_FILES: &'static [&'static str] = &[
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "model.q8_0.gguf",
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Automatic => "automatic",
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
            Self::Int8 => "int8",
        }
    }

    /// Returns the exact files required by this compute type's model bundle.
    pub fn required_bundle_files(self) -> &'static [&'static str] {
        match self {
            Self::Automatic | Self::Fp16 | Self::Fp32 => Self::SAFETENSORS_BUNDLE_FILES,
            Self::Int8 => Self::Q8_BUNDLE_FILES,
        }
    }

    pub(crate) fn resolve_for_device(self, cuda_active: bool) -> Result<Self> {
        match (self, cuda_active) {
            (Self::Automatic, true) => Ok(Self::Fp16),
            (Self::Automatic, false) => Ok(Self::Fp32),
            (Self::Fp16, true) => Ok(Self::Fp16),
            (Self::Fp16, false) => Err(setup_error(
                "native Candle Whisper compute type fp16 requires a CUDA device; use automatic or fp32 for CPU execution",
            )),
            (Self::Fp32, _) => Ok(Self::Fp32),
            (Self::Int8, false) => Ok(Self::Int8),
            (Self::Int8, true) => Err(setup_error(
                "native Candle Whisper compute type int8 is CPU-only; select device=cpu or use automatic, fp16, or fp32 for CUDA execution",
            )),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn setup_fallback_eligible(self) -> bool {
        matches!(self, Self::Automatic)
    }
}

/// Native Candle Whisper chunk decode runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CandleWhisperDecodeRuntime {
    /// Existing safe per-window autoregressive decode with KV-cache reuse inside each window.
    #[default]
    AutoregressiveKvCache,
    /// Future true tensor-batched active-row decode path.
    ActiveRowTensorBatch,
}

impl CandleWhisperDecodeRuntime {
    pub fn execution_id(self) -> &'static str {
        match self {
            Self::AutoregressiveKvCache => CANDLE_WHISPER_AUTOREGRESSIVE_KV_CACHE_EXECUTION,
            Self::ActiveRowTensorBatch => CANDLE_WHISPER_ACTIVE_ROW_TENSOR_BATCH_EXECUTION,
        }
    }

    pub fn is_supported(self) -> bool {
        matches!(
            self,
            Self::AutoregressiveKvCache | Self::ActiveRowTensorBatch
        )
    }
}

