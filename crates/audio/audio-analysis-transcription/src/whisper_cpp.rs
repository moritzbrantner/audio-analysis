//! Native whisper.cpp transcription support for the audio adapter.

#[cfg(feature = "native")]
#[path = "ffi.rs"]
mod ffi;

#[cfg(feature = "native")]
use std::ffi::CStr;
#[cfg(any(feature = "native", test))]
use std::ffi::CString;
use std::fmt::{Display, Formatter};
#[cfg(feature = "native")]
use std::fs::File;
#[cfg(any(feature = "native", test))]
use std::fs::{self, OpenOptions};
#[cfg(any(feature = "native", test))]
use std::io::Write;
#[cfg(feature = "native")]
use std::io::{BufWriter, Read};
use std::path::{Path, PathBuf};
#[cfg(any(feature = "native", test))]
use std::thread;
#[cfg(any(feature = "native", test))]
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
#[cfg(feature = "native")]
use sha2::{Digest, Sha256};

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
/// Variants describing whisper cpp model.
pub enum WhisperCppModel {
    #[serde(rename = "tiny.en")]
    /// The tiny en variant.
    TinyEn,
    #[serde(rename = "tiny")]
    /// The tiny variant.
    Tiny,
    #[serde(rename = "base.en")]
    #[default]
    /// The base en variant.
    BaseEn,
    #[serde(rename = "base")]
    /// The base variant.
    Base,
    #[serde(rename = "small.en")]
    /// The small en variant.
    SmallEn,
    #[serde(rename = "small")]
    /// The small variant.
    Small,
    #[serde(rename = "medium.en")]
    /// The medium en variant.
    MediumEn,
    #[serde(rename = "medium")]
    /// The medium variant.
    Medium,
    #[serde(rename = "large-v1")]
    /// The large v1 variant.
    LargeV1,
    #[serde(rename = "large-v2")]
    /// The large v2 variant.
    LargeV2,
    #[serde(rename = "large-v3")]
    /// The large v3 variant.
    LargeV3,
    #[serde(rename = "large-v3-turbo")]
    /// The large v3 turbo variant.
    LargeV3Turbo,
}

impl WhisperCppModel {
    /// Constant for all.
    pub const ALL: [Self; 12] = [
        Self::TinyEn,
        Self::Tiny,
        Self::BaseEn,
        Self::Base,
        Self::SmallEn,
        Self::Small,
        Self::MediumEn,
        Self::Medium,
        Self::LargeV1,
        Self::LargeV2,
        Self::LargeV3,
        Self::LargeV3Turbo,
    ];

    /// Returns identifier.
    pub fn id(self) -> &'static str {
        match self {
            Self::TinyEn => "tiny.en",
            Self::Tiny => "tiny",
            Self::BaseEn => "base.en",
            Self::Base => "base",
            Self::SmallEn => "small.en",
            Self::Small => "small",
            Self::MediumEn => "medium.en",
            Self::Medium => "medium",
            Self::LargeV1 => "large-v1",
            Self::LargeV2 => "large-v2",
            Self::LargeV3 => "large-v3",
            Self::LargeV3Turbo => "large-v3-turbo",
        }
    }

    /// Returns file name.
    pub fn file_name(self) -> String {
        format!("ggml-{}.bin", self.id())
    }

    /// Returns download URL.
    pub fn download_url(self) -> String {
        format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            self.file_name()
        )
    }

    /// Returns checksum sha256.
    pub fn checksum_sha256(self) -> &'static str {
        match self {
            Self::TinyEn => "0d686a2a6a22b02da2ef3101d4c86e68461363a623c58f27f81b1b2d36b42317",
            Self::Tiny => "518970a29bedb265f23ac48d486ddbc63bedffd90967b10140ae5ac61243acf3",
            Self::BaseEn => "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
            Self::Base => "2f62d18b50c3f3feafbf990eec23a93d319660b1efbdd3fff55e52b7cde2e374",
            Self::SmallEn => "0d57184d34ae7d736e5bb2db5bf83debe730bd53dcefa235a0979b9dcfd33fb3",
            Self::Small => "edd29d67e70b000132af65205b99bb774b77abc13d10103e14f80ce2242913e1",
            Self::MediumEn => "a163589aa264d5188df3b05ed4eac56bfd97e26910f207809d869f7e99886fd2",
            Self::Medium => "d3d5696e6a3e0ca2aa08eb31cad208ffa1e87b3cc341f59e628fbdcf8122de9b",
            Self::LargeV1 => "cbcb187d1e1abe979d33636cdc63381de20738eeda0885c39440b086e184248a",
            Self::LargeV2 => "c6d6d3dcebc5e0074175386e17eba305fc5cc7d3d5dff3ecfd11e8f2bd4222d7",
            Self::LargeV3 => "766d11cebbdf5a67c179c5774e2642b609e35e1a30240e7b559d5647c655b0a4",
            Self::LargeV3Turbo => {
                "5a4b65b05933d70ce9d5aa6265eb128fa5eba38f6fee40836fdedc4d2fde42ad"
            }
        }
    }

    /// Returns multilingual.
    pub fn multilingual(self) -> bool {
        !matches!(
            self,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn
        )
    }
}

impl Display for WhisperCppModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
/// Data type for whisper cpp config.
pub struct WhisperCppConfig {
    #[serde(default)]
    /// The model value.
    pub model: WhisperCppModel,
    /// Language tag for this value.
    pub language: Option<String>,
    #[serde(default)]
    /// The translate value.
    pub translate: bool,
    /// The threads value.
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Data type for whisper cpp segment.
pub struct WhisperCppSegment {
    /// The index value.
    pub index: u64,
    /// The start seconds value.
    pub start_seconds: Option<f64>,
    /// The end seconds value.
    pub end_seconds: Option<f64>,
    /// Text content for this value.
    pub text: String,
    /// Confidence score for this value.
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Data type for whisper cpp transcription.
pub struct WhisperCppTranscription {
    /// Text content for this value.
    pub text: Option<String>,
    /// Language tag for this value.
    pub language: Option<String>,
    /// The segments value.
    pub segments: Vec<WhisperCppSegment>,
    /// The source value.
    pub source: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Variants describing whisper cpp phase.
pub enum WhisperCppPhase {
    /// The preparing variant.
    Preparing,
    /// The downloading model variant.
    DownloadingModel,
    /// The loading model variant.
    LoadingModel,
    /// The transcribing variant.
    Transcribing,
}

impl WhisperCppPhase {
    /// Borrows this value as a str.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::DownloadingModel => "downloading_model",
            Self::LoadingModel => "loading_model",
            Self::Transcribing => "transcribing",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Data type for whisper cpp progress event.
pub struct WhisperCppProgressEvent {
    /// The phase value.
    pub phase: WhisperCppPhase,
    /// The message value.
    pub message: String,
    /// The progress value.
    pub progress: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Data type for whisper cpp model status.
pub struct WhisperCppModelStatus {
    /// The model value.
    pub model: WhisperCppModel,
    /// The cached value.
    pub cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Data type for whisper cpp catalog.
pub struct WhisperCppCatalog {
    /// The default model value.
    pub default_model: WhisperCppModel,
    /// The models value.
    pub models: Vec<WhisperCppModelStatus>,
}

#[derive(Debug, thiserror::Error)]
/// Variants describing whisper cpp error.
pub enum WhisperCppError {
    #[error("I/O error: {0}")]
    /// The I/O variant.
    Io(#[from] std::io::Error),
    #[error("wave input error: {0}")]
    /// The wav variant.
    Wav(#[from] hound::Error),
    #[error("network error: {0}")]
    /// The http variant.
    Http(String),
    #[error("invalid input: {0}")]
    /// The invalid input variant.
    InvalidInput(String),
    #[error("unsupported language `{0}`")]
    /// The unsupported language variant.
    UnsupportedLanguage(String),
    #[error("downloaded model `{model}` failed checksum verification")]
    /// The invalid checksum variant.
    InvalidChecksum {
        /// Model associated with this variant.
        model: WhisperCppModel,
    },
    #[error("failed to initialize whisper.cpp from `{0}`")]
    /// The initialization variant.
    Initialization(String),
    #[error("whisper.cpp inference failed for `{0}`")]
    /// The inference variant.
    Inference(String),
    #[error("invalid utf-8 returned by whisper.cpp")]
    /// The invalid utf8 variant.
    InvalidUtf8,
}

/// Type alias for result.
pub type Result<T> = std::result::Result<T, WhisperCppError>;

type OwnedProgressCallback = dyn FnMut(WhisperCppProgressEvent) + 'static;

#[derive(Clone)]
/// Data type for model store.
pub struct ModelStore {
    root: PathBuf,
}

impl Default for ModelStore {
    fn default() -> Self {
        Self {
            root: cache_root().join("whisper-cpp"),
        }
    }
}

impl ModelStore {
    /// Creates a new value.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Returns models dir.
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// Returns model path.
    pub fn model_path(&self, model: WhisperCppModel) -> PathBuf {
        self.models_dir().join(model.file_name())
    }

    /// Returns lock path.
    pub fn lock_path(&self, model: WhisperCppModel) -> PathBuf {
        self.models_dir()
            .join(format!("{}.lock", model.file_name()))
    }

    /// Returns catalog.
    pub fn catalog(&self) -> WhisperCppCatalog {
        WhisperCppCatalog {
            default_model: WhisperCppModel::default(),
            models: WhisperCppModel::ALL
                .into_iter()
                .map(|model| WhisperCppModelStatus {
                    model,
                    cached: self.model_path(model).is_file(),
                })
                .collect(),
        }
    }

    #[cfg(feature = "native")]
    fn ensure_model(
        &self,
        model: WhisperCppModel,
        progress: &mut ProgressSink<'_>,
    ) -> Result<PathBuf> {
        fs::create_dir_all(self.models_dir())?;
        let model_path = self.model_path(model);
        if model_path.is_file() {
            return Ok(model_path);
        }

        let _lock = FileLock::acquire(self.lock_path(model))?;
        if model_path.is_file() {
            return Ok(model_path);
        }

        progress.emit(
            WhisperCppPhase::DownloadingModel,
            format!("downloading whisper.cpp model `{model}`"),
            Some(0.0),
        );

        let temp_path = model_path.with_extension("bin.part");
        if temp_path.exists() {
            let _ = fs::remove_file(&temp_path);
        }

        let response = ureq::get(&model.download_url())
            .call()
            .map_err(|error| WhisperCppError::Http(error.to_string()))?;
        let total_bytes = response
            .header("Content-Length")
            .and_then(|value| value.parse::<u64>().ok());
        let mut reader = response.into_reader();
        let mut file = BufWriter::new(File::create(&temp_path)?);
        let mut hasher = Sha256::new();
        let mut downloaded = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];

        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|error| WhisperCppError::Http(error.to_string()))?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])?;
            hasher.update(&buffer[..read]);
            downloaded += read as u64;
            let fraction =
                total_bytes.map(|total| (downloaded as f32 / total as f32).clamp(0.0, 1.0));
            progress.emit(
                WhisperCppPhase::DownloadingModel,
                format!("downloading whisper.cpp model `{model}`"),
                fraction,
            );
        }
        file.flush()?;

        let checksum = format!("{:x}", hasher.finalize());
        if checksum != model.checksum_sha256() {
            let _ = fs::remove_file(&temp_path);
            return Err(WhisperCppError::InvalidChecksum { model });
        }

        fs::rename(temp_path, &model_path)?;
        Ok(model_path)
    }
}

/// Data type for whisper cpp transcriber.
pub struct WhisperCppTranscriber {
    config: WhisperCppConfig,
    store: ModelStore,
    progress: Option<Box<OwnedProgressCallback>>,
}

impl WhisperCppTranscriber {
    /// Creates a new value.
    pub fn new(config: WhisperCppConfig) -> Self {
        Self {
            config,
            store: ModelStore::default(),
            progress: None,
        }
    }

    /// Returns this value with model store.
    pub fn with_model_store(mut self, store: ModelStore) -> Self {
        self.store = store;
        self
    }

    /// Returns on progress.
    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: FnMut(WhisperCppProgressEvent) + 'static,
    {
        self.progress = Some(Box::new(callback));
        self
    }

    /// Returns transcribe file.
    pub fn transcribe_file(&mut self, input: &Path) -> Result<WhisperCppTranscription> {
        let store = self.store.clone();
        let config = self.config.clone();
        let mut progress = ProgressSink::new(self.progress_deref_mut());
        transcribe_impl(&store, &config, input, &mut progress)
    }

    /// Returns transcribe file with progress.
    pub fn transcribe_file_with_progress(
        &mut self,
        input: &Path,
        progress: &mut dyn FnMut(WhisperCppProgressEvent),
    ) -> Result<WhisperCppTranscription> {
        let mut progress = ProgressSink::new(Some(progress));
        transcribe_impl(&self.store, &self.config, input, &mut progress)
    }

    fn progress_deref_mut(&mut self) -> Option<&mut dyn FnMut(WhisperCppProgressEvent)> {
        self.progress
            .as_mut()
            .map(|callback| callback.as_mut() as &mut dyn FnMut(WhisperCppProgressEvent))
    }
}

/// Returns transcription catalog.
pub fn transcription_catalog() -> WhisperCppCatalog {
    ModelStore::default().catalog()
}

/// Returns whisper cpp system info.
pub fn whisper_cpp_system_info() -> Option<String> {
    #[cfg(not(feature = "native"))]
    {
        None
    }

    #[cfg(feature = "native")]
    {
        let value = unsafe { ffi::whisper_print_system_info() };
        if value.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(value) }
            .to_str()
            .ok()
            .map(|value| value.to_string())
    }
}

#[cfg(feature = "native")]
fn transcribe_impl(
    store: &ModelStore,
    config: &WhisperCppConfig,
    input: &Path,
    progress: &mut ProgressSink<'_>,
) -> Result<WhisperCppTranscription> {
    let model = config.model;
    progress.emit(
        WhisperCppPhase::Preparing,
        format!(
            "preparing native whisper.cpp transcription for {}",
            input.display()
        ),
        None,
    );

    let model_path = store.ensure_model(model, progress)?;
    progress.emit(
        WhisperCppPhase::LoadingModel,
        format!("loading whisper.cpp model `{model}`"),
        None,
    );

    let audio = read_wav_mono_f32(input)?;
    progress.emit(
        WhisperCppPhase::Transcribing,
        format!("transcribing audio with whisper.cpp model `{model}`"),
        None,
    );

    let context = WhisperContext::from_model(&model_path)?;
    let mut params = unsafe {
        ffi::whisper_full_default_params(ffi::whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY)
    };
    params.n_threads = resolve_threads(config.threads);
    params.translate = config.translate;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_special = false;
    params.print_timestamps = false;
    params.no_timestamps = false;

    let language = resolve_language(config)?;
    if let Some(language) = language.as_ref() {
        let lang_id = unsafe { ffi::whisper_lang_id(language.as_ptr()) };
        if lang_id < 0 {
            return Err(WhisperCppError::UnsupportedLanguage(
                language.to_string_lossy().into_owned(),
            ));
        }
        params.language = language.as_ptr();
    } else {
        params.language = std::ptr::null();
    }
    params.detect_language = false;

    let status = unsafe {
        ffi::whisper_full(
            context.raw,
            params,
            audio.samples.as_ptr(),
            audio.samples.len() as i32,
        )
    };
    if status != 0 {
        return Err(WhisperCppError::Inference(model_path.display().to_string()));
    }

    let segment_count = unsafe { ffi::whisper_full_n_segments(context.raw) };
    let mut segments = Vec::with_capacity(segment_count.max(0) as usize);
    for index in 0..segment_count {
        let text_ptr = unsafe { ffi::whisper_full_get_segment_text(context.raw, index) };
        let text = c_string(text_ptr)?.trim().to_string();
        let start = unsafe { ffi::whisper_full_get_segment_t0(context.raw, index) };
        let end = unsafe { ffi::whisper_full_get_segment_t1(context.raw, index) };
        let token_count = unsafe { ffi::whisper_full_n_tokens(context.raw, index) };
        let confidence = if token_count > 0 {
            let mut total = 0.0_f32;
            for token_index in 0..token_count {
                total += unsafe { ffi::whisper_full_get_token_p(context.raw, index, token_index) };
            }
            Some(total / token_count as f32)
        } else {
            None
        };
        segments.push(WhisperCppSegment {
            index: index as u64,
            start_seconds: Some(timestamp_to_seconds(start)),
            end_seconds: Some(timestamp_to_seconds(end)),
            text,
            confidence,
        });
    }

    let language = unsafe { ffi::whisper_full_lang_id(context.raw) };
    let language = if language >= 0 {
        Some(c_string(unsafe { ffi::whisper_lang_str(language) })?)
    } else {
        None
    };
    let text = join_segments(&segments);

    Ok(WhisperCppTranscription {
        text,
        language,
        segments,
        source: Some(model_path.to_string_lossy().into_owned()),
    })
}

#[cfg(not(feature = "native"))]
fn transcribe_impl(
    _store: &ModelStore,
    _config: &WhisperCppConfig,
    _input: &Path,
    _progress: &mut ProgressSink<'_>,
) -> Result<WhisperCppTranscription> {
    Err(WhisperCppError::Initialization(
        "audio-analysis-transcription was built without the `native` feature".to_string(),
    ))
}

#[cfg(any(feature = "native", test))]
fn resolve_language(config: &WhisperCppConfig) -> Result<Option<CString>> {
    match config.language.as_deref().map(str::trim) {
        Some("") => resolve_default_language(config.model),
        Some(value) if value.eq_ignore_ascii_case("auto") => resolve_default_language(config.model),
        Some(value) => CString::new(value)
            .map(Some)
            .map_err(|_| WhisperCppError::UnsupportedLanguage(value.to_string())),
        None => resolve_default_language(config.model),
    }
}

#[cfg(any(feature = "native", test))]
fn resolve_default_language(model: WhisperCppModel) -> Result<Option<CString>> {
    if model.multilingual() {
        Ok(None)
    } else {
        CString::new("en")
            .map(Some)
            .map_err(|_| WhisperCppError::UnsupportedLanguage("en".to_string()))
    }
}

#[cfg_attr(not(feature = "native"), allow(dead_code))]
struct ProgressSink<'a> {
    callback: Option<&'a mut dyn FnMut(WhisperCppProgressEvent)>,
}

impl<'a> ProgressSink<'a> {
    fn new(callback: Option<&'a mut dyn FnMut(WhisperCppProgressEvent)>) -> Self {
        Self { callback }
    }

    #[cfg(feature = "native")]
    fn emit(&mut self, phase: WhisperCppPhase, message: String, progress: Option<f32>) {
        if let Some(callback) = self.callback.as_mut() {
            callback(WhisperCppProgressEvent {
                phase,
                message,
                progress,
            });
        }
    }
}

#[cfg(feature = "native")]
fn read_wav_mono_f32(path: &Path) -> Result<AudioSamples> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err(WhisperCppError::InvalidInput(
            "wav file has no channels".to_string(),
        ));
    }
    if spec.sample_rate != 16_000 {
        return Err(WhisperCppError::InvalidInput(format!(
            "expected 16 kHz wav input, got {} Hz",
            spec.sample_rate
        )));
    }

    let interleaved = match spec.sample_format {
        hound::SampleFormat::Int => read_int_samples(&mut reader, spec.bits_per_sample)?,
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
    };

    let channels = spec.channels as usize;
    let samples = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
            .collect()
    };

    Ok(AudioSamples { samples })
}

#[cfg(feature = "native")]
fn read_int_samples(
    reader: &mut hound::WavReader<std::io::BufReader<File>>,
    bits_per_sample: u16,
) -> Result<Vec<f32>> {
    let scale = ((1_i64 << (bits_per_sample.saturating_sub(1) as u32)) - 1) as f32;
    if bits_per_sample <= 16 {
        Ok(reader
            .samples::<i16>()
            .map(|sample| sample.map(|sample| sample as f32 / scale))
            .collect::<std::result::Result<Vec<_>, _>>()?)
    } else {
        Ok(reader
            .samples::<i32>()
            .map(|sample| sample.map(|sample| sample as f32 / scale))
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }
}

#[cfg(feature = "native")]
fn resolve_threads(value: Option<usize>) -> i32 {
    value
        .or_else(|| thread::available_parallelism().ok().map(usize::from))
        .unwrap_or(4)
        .min(i32::MAX as usize) as i32
}

#[cfg(feature = "native")]
fn timestamp_to_seconds(value: i64) -> f64 {
    value as f64 / 100.0
}

#[cfg(feature = "native")]
fn join_segments(segments: &[WhisperCppSegment]) -> Option<String> {
    let text = segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    (!text.is_empty()).then_some(text)
}

#[cfg(feature = "native")]
fn c_string(value: *const std::ffi::c_char) -> Result<String> {
    if value.is_null() {
        return Ok(String::new());
    }
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(|value| value.to_string())
        .map_err(|_| WhisperCppError::InvalidUtf8)
}

fn cache_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("VIDEO_ANALYSIS_STUDIO_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("video-analysis-studio");
    }
    if cfg!(target_os = "windows") {
        if let Some(dir) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(dir).join("video-analysis-studio");
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("video-analysis-studio");
    }
    PathBuf::from(".cache/video-analysis-studio")
}

#[cfg(feature = "native")]
struct AudioSamples {
    samples: Vec<f32>,
}

#[cfg(feature = "native")]
struct WhisperContext {
    raw: *mut ffi::whisper_context,
}

#[cfg(feature = "native")]
impl WhisperContext {
    fn from_model(path: &Path) -> Result<Self> {
        let model_path = CString::new(path.to_string_lossy().into_owned())
            .map_err(|_| WhisperCppError::Initialization(path.display().to_string()))?;
        let mut params = unsafe { ffi::whisper_context_default_params() };
        params.use_gpu = cfg!(target_os = "macos");
        params.flash_attn = false;
        let raw = unsafe { ffi::whisper_init_from_file_with_params(model_path.as_ptr(), params) };
        if raw.is_null() {
            return Err(WhisperCppError::Initialization(path.display().to_string()));
        }
        Ok(Self { raw })
    }
}

#[cfg(feature = "native")]
impl Drop for WhisperContext {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { ffi::whisper_free(self.raw) };
        }
    }
}

#[cfg(any(feature = "native", test))]
struct FileLock {
    path: PathBuf,
}

#[cfg(any(feature = "native", test))]
impl FileLock {
    fn acquire(path: PathBuf) -> Result<Self> {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            match OpenOptions::new().create_new(true).write(true).open(&path) {
                Ok(mut file) => {
                    let _ = writeln!(file, "{}", std::process::id());
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if Instant::now() >= deadline {
                        return Err(WhisperCppError::Io(error));
                    }
                    thread::sleep(Duration::from_millis(250));
                }
                Err(error) => return Err(WhisperCppError::Io(error)),
            }
        }
    }
}

#[cfg(any(feature = "native", test))]
impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn model_metadata_matches_expected_file_names() {
        assert_eq!(WhisperCppModel::BaseEn.file_name(), "ggml-base.en.bin");
        assert_eq!(
            WhisperCppModel::LargeV3Turbo.download_url(),
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
        );
    }

    #[test]
    fn catalog_uses_base_en_by_default() {
        let catalog = ModelStore::new(PathBuf::from("/tmp/video-analysis-studio-test")).catalog();
        assert_eq!(catalog.default_model, WhisperCppModel::BaseEn);
        assert_eq!(catalog.models.len(), WhisperCppModel::ALL.len());
    }

    #[test]
    fn cache_paths_are_stable() {
        let store = ModelStore::new(PathBuf::from("/tmp/video-analysis-studio-test"));
        assert_eq!(
            store.model_path(WhisperCppModel::SmallEn),
            PathBuf::from("/tmp/video-analysis-studio-test/models/ggml-small.en.bin")
        );
        assert_eq!(
            store.lock_path(WhisperCppModel::SmallEn),
            PathBuf::from("/tmp/video-analysis-studio-test/models/ggml-small.en.bin.lock")
        );
    }

    #[test]
    fn file_lock_creates_and_releases_lock_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.lock");
        {
            let _lock = FileLock::acquire(path.clone()).unwrap();
            assert!(path.is_file());
        }
        assert!(!path.exists());
    }

    #[test]
    fn english_only_models_default_to_english() {
        let config = WhisperCppConfig {
            model: WhisperCppModel::BaseEn,
            language: None,
            translate: false,
            threads: None,
        };

        let language = resolve_language(&config).unwrap().unwrap();
        assert_eq!(language.to_str().unwrap(), "en");
    }

    #[test]
    fn multilingual_models_default_to_auto_detection_without_detect_only_mode() {
        let config = WhisperCppConfig {
            model: WhisperCppModel::Base,
            language: None,
            translate: false,
            threads: None,
        };

        assert_eq!(resolve_language(&config).unwrap(), None);
    }

    #[test]
    fn auto_language_uses_english_for_english_only_models() {
        let config = WhisperCppConfig {
            model: WhisperCppModel::SmallEn,
            language: Some("auto".to_string()),
            translate: false,
            threads: None,
        };

        let language = resolve_language(&config).unwrap().unwrap();
        assert_eq!(language.to_str().unwrap(), "en");
    }
}
