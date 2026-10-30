//! Provider, VAD, alignment, diarization and output options.

use crate::*;

mod alignment;
mod candle;
mod diarization;
mod output;
mod vad;
mod whisper_cpp;
mod whisperx;

pub use alignment::*;
pub use candle::*;
pub use diarization::*;
pub use output::*;
pub use vad::*;
pub use whisper_cpp::*;
pub use whisperx::*;

/// Native execution device preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NativeDevicePreference {
    #[default]
    Auto,
    Cpu,
    Cuda,
}
