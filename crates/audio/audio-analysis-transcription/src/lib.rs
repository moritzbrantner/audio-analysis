#![doc = include_str!("../README.md")]

mod bounded_window;
mod ctc;
pub use ctc::{align_ctc_emissions, CtcEmissionAlignmentRequest, CtcVocabulary};
mod resources;
pub mod surface;
pub use resources::{
    inspect_alignment_resources, inspect_candle_whisper_resources, CandleWhisperResourceInspection,
    NativeModelResourceInspection, NativeResourceDevice,
};

#[cfg(feature = "alignment")]
mod ctc_alignment;
mod native_audio;
mod native_bundles;
mod native_device;
#[cfg(feature = "alignment")]
mod native_wav2vec2;
#[cfg(feature = "alignment")]
mod native_wav2vec2_model;
#[cfg(feature = "candle")]
mod native_whisper;
#[cfg(feature = "candle")]
mod native_whisper_decode;
#[cfg(feature = "candle")]
mod native_whisper_q8_cpu;
#[cfg(feature = "candle")]
mod native_whisper_quantized;
#[cfg(any(feature = "silero-vad", feature = "pyannote-vad", test))]
mod silero_vad;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use media_core::{DetectError, Result};
use media_core::{
    TranscriptCharContract, TranscriptSegmentContract, TranscriptWordContract,
    TranscriptionContract,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

mod whisper_cpp;
pub use bounded_window::{
    BoundedPcmWindow, BoundedPcmWindowConfig, BoundedPcmWindowSession, BoundedPcmWindowStats,
};
pub use whisper_cpp::{
    transcription_catalog as whisper_cpp_catalog, whisper_cpp_system_info,
    ModelStore as WhisperCppModelStore, WhisperCppCatalog, WhisperCppConfig, WhisperCppError,
    WhisperCppModel, WhisperCppModelStatus, WhisperCppPhase, WhisperCppProgressEvent,
    WhisperCppSegment, WhisperCppTranscriber as NativeWhisperCppTranscriber,
    WhisperCppTranscription,
};

mod errors;
mod options;
mod pipeline;
mod plans;
mod providers;
mod runner;
mod transcript;
mod whisperx_import;

pub(crate) use errors::*;
pub use options::*;
pub use pipeline::*;
pub use plans::*;
pub use providers::*;
pub use runner::*;
pub(crate) use transcript::*;
pub use whisperx_import::*;

#[cfg(feature = "audio-io")]
pub use audio_analysis_io::{AudioIoError, SelectedMediaSource};
pub use audio_analysis_speakers::{
    AudioRuntime, EnergyVoiceActivityDetector, SpeakerAudio, SpeakerDiarizationOptions,
    SpeakerDiarizationResponse, SpeakerSegmentPrediction, SpeakerTranscriptAssignmentPolicy,
    SpeechSpan, VadOptions, VadResponse, VoiceActivityDetector,
};
#[cfg(feature = "candle")]
pub use native_whisper_quantized::{
    CandleQ8WhisperBatchDiagnostics, CandleQ8WhisperBatchOutput, CandleQ8WhisperDecoder,
    CandleQ8WhisperDecoderDiagnostics, CandleQ8WhisperDecoderOutput,
};
#[cfg(feature = "pyannote-vad")]
pub use silero_vad::{
    inspect_pyannote_vad_bundle, PyannoteVadBundleReport, PyannoteVadOptions,
    PyannoteVadTranscriptionProvider,
};
#[cfg(feature = "silero-vad")]
pub use silero_vad::{SileroVadOptions, SileroVadTranscriptionProvider};

/// Backward-compatible name for Transcript Speaker Assignment policy.
pub type SpeakerAssignmentPolicy = SpeakerTranscriptAssignmentPolicy;

#[cfg(test)]
mod tests;
