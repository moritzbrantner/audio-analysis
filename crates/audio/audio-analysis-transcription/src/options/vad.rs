//! VAD options used before ASR chunking.

use crate::*;

/// VAD options used before ASR chunking.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionVadOptions {
    /// Run VAD; when disabled the full source is one ASR chunk.
    #[serde(default = "default_vad_enabled")]
    pub enabled: bool,
    /// Detection options passed to the voice activity detector.
    #[serde(flatten)]
    pub detection: VadOptions,
}

impl Default for TranscriptionVadOptions {
    fn default() -> Self {
        Self {
            enabled: default_vad_enabled(),
            detection: VadOptions::default(),
        }
    }
}

pub(crate) fn default_vad_enabled() -> bool {
    true
}
