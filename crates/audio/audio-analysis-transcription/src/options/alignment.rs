//! Forced-alignment options.

use crate::*;

/// Timestamp interpolation behavior for missing alignment spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlignmentInterpolationMethod {
    #[default]
    Nearest,
    Linear,
    Ignore,
}

impl AlignmentInterpolationMethod {
    pub fn as_whisperx_arg(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Linear => "linear",
            Self::Ignore => "ignore",
        }
    }
}

/// Forced-alignment options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlignmentOptions {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_alignment_model")]
    pub model_id: String,
    #[serde(default = "default_alignment_device")]
    pub device: NativeDevicePreference,
    #[serde(default)]
    pub model_bundle: Option<PathBuf>,
    #[serde(default)]
    pub model_dir: Option<PathBuf>,
    #[serde(default)]
    pub model_cache_only: bool,
    #[serde(default)]
    pub interpolate_method: AlignmentInterpolationMethod,
    #[serde(default)]
    pub return_char_alignments: bool,
}

impl Default for AlignmentOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            model_id: default_alignment_model(),
            device: default_alignment_device(),
            model_bundle: None,
            model_dir: None,
            model_cache_only: false,
            interpolate_method: AlignmentInterpolationMethod::Nearest,
            return_char_alignments: false,
        }
    }
}

pub(crate) fn default_alignment_model() -> String {
    "facebook/wav2vec2-base-960h".to_string()
}

pub(crate) fn default_alignment_device() -> NativeDevicePreference {
    NativeDevicePreference::Cpu
}
