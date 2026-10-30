//! Transcription output options.

use crate::*;

/// Output preferences for transcription.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionOutputOptions {
    #[serde(default = "default_output_formats")]
    pub formats: Vec<String>,
}

impl Default for TranscriptionOutputOptions {
    fn default() -> Self {
        Self {
            formats: default_output_formats(),
        }
    }
}

pub(crate) fn default_output_formats() -> Vec<String> {
    vec!["json".to_string()]
}
