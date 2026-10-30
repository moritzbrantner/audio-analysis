//! External WhisperX command options.

use crate::*;

/// Options for the external WhisperX command provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WhisperXCommandOptions {
    pub command: PathBuf,
    pub model: String,
    #[serde(default)]
    pub task: TranscriptionTask,
    #[serde(default)]
    pub language: Option<String>,
    pub device: WhisperXDevice,
    #[serde(default)]
    pub compute_type: Option<String>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub align_model: Option<String>,
    #[serde(default)]
    pub model_dir: Option<PathBuf>,
    #[serde(default)]
    pub model_cache_only: bool,
    #[serde(default)]
    pub no_align: bool,
    #[serde(default)]
    pub interpolate_method: AlignmentInterpolationMethod,
    #[serde(default)]
    pub return_char_alignments: bool,
    #[serde(default)]
    pub diarize: bool,
    #[serde(default)]
    pub min_speakers: Option<usize>,
    #[serde(default)]
    pub max_speakers: Option<usize>,
    #[serde(default)]
    pub hf_token_env: Option<String>,
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    #[serde(default)]
    pub extra_args: Vec<String>,
}

/// Backward-compatible name for the external WhisperX provider options.
pub type WhisperXOptions = WhisperXCommandOptions;

impl Default for WhisperXCommandOptions {
    fn default() -> Self {
        Self {
            command: PathBuf::from("whisperx"),
            model: "large-v2".to_string(),
            task: TranscriptionTask::Transcribe,
            language: None,
            device: WhisperXDevice::Cpu,
            compute_type: None,
            batch_size: None,
            align_model: None,
            model_dir: None,
            model_cache_only: false,
            no_align: false,
            interpolate_method: AlignmentInterpolationMethod::Nearest,
            return_char_alignments: false,
            diarize: false,
            min_speakers: None,
            max_speakers: None,
            hf_token_env: None,
            output_dir: None,
            timeout_seconds: None,
            extra_args: Vec::new(),
        }
    }
}

/// WhisperX device selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WhisperXDevice {
    Cpu,
    Cuda,
}

impl WhisperXDevice {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}
