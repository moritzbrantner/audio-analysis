//! Private migration shim for the timed-text ownership cutover.
//!
//! Audio transcription used to import transcript DTOs from `nlp-stack` through
//! the `text_transcripts` crate name. The canonical DTOs now live in
//! `moenarch-media-core`. This unpublished crate preserves the old Rust import
//! spelling while keeping the concrete type identity owned by foundation.
//!
//! Do not add parsing, formatting, NLP enrichment, or other behavior here.

pub use media_core::{
    TimedTextCharContract, TimedTextContract, TimedTextSegmentContract, TimedTextWordContract,
    TranscriptCharContract, TranscriptSegmentContract, TranscriptWordContract,
    TranscriptionContract,
};
