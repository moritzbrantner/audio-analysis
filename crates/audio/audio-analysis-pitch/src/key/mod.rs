//! Polyphonic chroma, dominant-key, and change-aware whole-track key analysis.
//!
//! The established harmonic-key implementation remains isolated below while this
//! module also exposes a reusable confidence-bearing timeline contract. Transport
//! adapters should delegate key semantics here rather than reimplementing them.

mod implementation;
mod timeline;

pub use implementation::*;
pub use timeline::{
    analyze_key_track, analyze_key_track_with_boundaries, KeySegment, KeyTimelineConfig,
    KeyTimelineWindow, TrackKeyAnalysis,
};
