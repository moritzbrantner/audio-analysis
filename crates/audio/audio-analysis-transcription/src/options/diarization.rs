//! Transcript diarization options.

use crate::*;

/// Diarization options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiarizationOptions {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, flatten)]
    pub speaker: SpeakerDiarizationOptions,
}

impl std::ops::Deref for DiarizationOptions {
    type Target = SpeakerDiarizationOptions;

    fn deref(&self) -> &Self::Target {
        &self.speaker
    }
}

impl std::ops::DerefMut for DiarizationOptions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.speaker
    }
}
