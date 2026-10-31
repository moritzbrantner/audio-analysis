#![doc = include_str!("../README.md")]

mod assignment;
mod diarizer;
mod embedding;
mod errors;
mod library;
mod library_transfer;
#[cfg(feature = "pyannote-core")]
pub mod pyannote;
mod request;
pub mod surface;
#[cfg(test)]
mod tests;
mod vad;

pub use assignment::{
    assign_speakers_to_transcript, assign_speakers_to_transcript_with_policy,
    SpeakerTranscriptAssignmentPolicy, TranscriptSpeakerOverlapPolicy,
};
pub use audio_analysis_recognition::AudioRuntime;
pub use diarizer::{
    DiarizationResult, DiarizationSegment, DiarizedSpeaker, SpeakerDiarizer,
    WindowedSpeakerDiarizer, DEFAULT_CLUSTER_THRESHOLD, DEFAULT_EMBEDDING_WINDOW_SECONDS,
    DEFAULT_MIN_CLUSTER_FRACTION, DEFAULT_MIN_CLUSTER_SECONDS,
    DEFAULT_MIN_SPEAKER_EVIDENCE_SECONDS,
};
pub use embedding::{
    OnnxSpeakerEmbedder, OnnxSpeakerEmbeddingConfig, SpeakerAudio, SpeakerEmbedding,
    SpeakerEmbeddingExtractor, SpeakerEmbeddingModel, SpeakerEmbeddingModelFamily,
    SpeakerEmbeddingProvider, SpeakerEmbeddingRequest, SpeakerEmbeddingResponse,
    SpectralSpeakerEmbedder,
};
pub use library::{
    SpeakerConfidence, SpeakerId, SpeakerIdentificationOptions, SpeakerIdentificationResult,
    SpeakerLabel, SpeakerLibrary, SpeakerLibrarySnapshot, SpeakerMatch, SpeakerProfile,
    SpeakerProfileUpdate, UnknownSpeakerPolicy,
};
pub use library_transfer::{
    SpeakerLibraryImportMode, SpeakerLibraryImportPlan, SpeakerLibraryImportReceipt,
    SpeakerLibraryImportSummary,
};
#[cfg(feature = "pyannote-diarization")]
pub use pyannote::{
    inspect_pyannote_community_diarization_bundle, PyannoteCommunityDiarizationBundleReport,
    PyannoteCommunityDiarizationConfig, PyannoteCommunityDiarizationResult,
    PyannoteCommunityDiarizer,
};
pub use request::{
    default_speaker_diarization_model, diarize_speaker_audio_baseline, diarize_speakers,
    is_pyannote_diarization_model, SpeakerDiarizationOptions, SpeakerDiarizationRequest,
    SpeakerDiarizationResponse, SpeakerSegmentPrediction,
};
pub use vad::{
    EnergyVoiceActivityDetector, SpeechSpan, VadOptions, VadResponse, VoiceActivityDetector,
};
