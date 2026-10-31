//! whisper.cpp provider options.

use crate::*;

impl From<WhisperCppConfig> for WhisperCppProviderOptions {
    fn from(value: WhisperCppConfig) -> Self {
        Self {
            model_id: value.model.id().to_string(),
            task: if value.translate {
                TranscriptionTask::Translate
            } else {
                TranscriptionTask::Transcribe
            },
            language: value.language,
            model_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WhisperCppProviderOptions {
    #[serde(default = "default_whisper_cpp_model")]
    pub model_id: String,
    #[serde(default)]
    pub task: TranscriptionTask,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
}

impl Default for WhisperCppProviderOptions {
    fn default() -> Self {
        Self {
            model_id: default_whisper_cpp_model(),
            task: TranscriptionTask::Transcribe,
            language: None,
            model_path: None,
        }
    }
}

pub(crate) fn default_whisper_cpp_model() -> String {
    "large-v3-turbo".to_string()
}

