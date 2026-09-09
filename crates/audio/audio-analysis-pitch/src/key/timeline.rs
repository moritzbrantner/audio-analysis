//! Confidence-bearing whole-track musical-key timeline analysis.
//!
//! This module keeps key-change semantics in the reusable pitch library. Local
//! harmonic observations are decoded as a 24-state major/minor sequence with a
//! musical transition prior. When bar/downbeat boundaries are supplied, emitted
//! key windows and stable segments are aligned to those boundaries while analysis
//! uses surrounding bars for harmonic context.

use audio_contracts::{DetectError, Result};

use super::{
    estimate_from_chroma, estimate_musical_key, harmonic_chroma, score_key_candidates,
    HarmonicChromaAnalysis, HarmonicKeyConfig, KeyCandidate, MusicalKeyEstimate, MusicalScale,
};
use crate::NoteName;

const KEY_STATE_COUNT: usize = 24;
const MIN_STABLE_WINDOWS: usize = 2;

/// Configuration for change-aware whole-track key analysis.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyTimelineConfig {
    /// Duration of each overlapping local key window in seconds when no musical boundaries exist.
    pub window_seconds: f64,
    /// Distance between successive local key windows when no musical boundaries exist.
    pub hop_seconds: f64,
    /// Minimum key confidence required before a local window is treated as known.
    pub min_confidence: f32,
    /// Defensive upper bound on emitted windows.
    pub max_windows: usize,
}

impl Default for KeyTimelineConfig {
    fn default() -> Self {
        Self {
            window_seconds: 24.0,
            hop_seconds: 8.0,
            min_confidence: 0.10,
            max_windows: 256,
        }
    }
}

impl KeyTimelineConfig {
    /// Validates the timeline configuration.
    pub fn validate(&self) -> Result<()> {
        if !self.window_seconds.is_finite() || self.window_seconds <= 0.0 {
            return Err(DetectError::InvalidArgument(
                "key timeline window_seconds must be finite and positive".to_string(),
            ));
        }
        if !self.hop_seconds.is_finite() || self.hop_seconds <= 0.0 {
            return Err(DetectError::InvalidArgument(
                "key timeline hop_seconds must be finite and positive".to_string(),
            ));
        }
        if !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(DetectError::InvalidArgument(
                "key timeline min_confidence must be between zero and one".to_string(),
            ));
        }
        if self.max_windows == 0 {
            return Err(DetectError::InvalidArgument(
                "key timeline max_windows must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// One confidence-bearing local key window.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyTimelineWindow {
    /// Window start in seconds from the beginning of the analyzed PCM.
    pub start_seconds: f64,
    /// Window end in seconds from the beginning of the analyzed PCM.
    pub end_seconds: f64,
    /// Window center in seconds.
    pub center_seconds: f64,
    /// Key estimate when confidence and temporal persistence clear the thresholds.
    pub key: Option<MusicalKeyEstimate>,
}

/// One stable decoded key segment.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeySegment {
    /// Segment start in seconds.
    pub start_seconds: f64,
    /// Segment end in seconds.
    pub end_seconds: f64,
    /// Stable decoded key for this segment.
    pub key: MusicalKeyEstimate,
    /// Mean confidence of the contributing local windows.
    pub confidence: f32,
    /// Number of local windows supporting the segment.
    pub window_count: usize,
}

/// Reusable whole-track key result, separate from browser or WASM transport.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackKeyAnalysis {
    /// Versioned machine-readable contract identifier.
    pub schema_version: String,
    /// PCM sample rate used by the analysis.
    pub sample_rate: u32,
    /// Number of mono PCM samples analyzed.
    pub sample_count: usize,
    /// Analyzed duration in seconds.
    pub duration_seconds: f64,
    /// Dominant full-track key, if the complete track contains sufficient tonal evidence.
    pub dominant: Option<MusicalKeyEstimate>,
    /// Fallback timeline window duration in seconds.
    pub timeline_window_seconds: f64,
    /// Fallback timeline hop duration in seconds.
    pub timeline_hop_seconds: f64,
    /// Confidence threshold below which local windows remain explicitly unknown.
    pub timeline_min_confidence: f32,
    /// Whether supplied bar/downbeat boundaries were used for timeline alignment.
    pub boundary_aligned: bool,
    /// Whether the timeline reached the end of the requested observation set.
    pub timeline_complete: bool,
    /// Local decoded key windows. Uncertain or non-persistent windows retain `key: null`.
    pub timeline: Vec<KeyTimelineWindow>,
    /// Stable decoded key segments assembled from consecutive known windows.
    pub segments: Vec<KeySegment>,
}

#[derive(Debug, Clone, Copy)]
struct ObservationRange {
    display_start_seconds: f64,
    display_end_seconds: f64,
    analysis_start_seconds: f64,
    analysis_end_seconds: f64,
}

#[derive(Debug, Clone)]
struct HarmonicObservation {
    range: ObservationRange,
    analysis: HarmonicChromaAnalysis,
    candidates: Vec<KeyCandidate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KeyState {
    tonic: NoteName,
    scale: MusicalScale,
}

/// Analyzes a complete mono track with the fixed-window fallback timeline.
pub fn analyze_key_track(
    samples: &[f32],
    sample_rate: u32,
    harmonic_config: HarmonicKeyConfig,
    timeline_config: KeyTimelineConfig,
) -> Result<TrackKeyAnalysis> {
    analyze_key_track_with_boundaries(
        samples,
        sample_rate,
        harmonic_config,
        timeline_config,
        &[],
    )
}

/// Analyzes a complete mono track with optional musical boundaries.
///
/// `boundaries_seconds` is intended for downbeats or bar starts supplied by a
/// rhythm analyzer. With at least two valid boundaries, one output observation
/// is aligned to each adjacent boundary pair and uses neighbouring bars as
/// harmonic context. The pitch crate does not depend on rhythm implementation
/// details; callers own the boundary transport.
pub fn analyze_key_track_with_boundaries(
    samples: &[f32],
    sample_rate: u32,
    harmonic_config: HarmonicKeyConfig,
    timeline_config: KeyTimelineConfig,
    boundaries_seconds: &[f64],
) -> Result<TrackKeyAnalysis> {
    harmonic_config.validate()?;
    timeline_config.validate()?;
    validate_track(samples, sample_rate)?;

    let duration_seconds = samples.len() as f64 / sample_rate as f64;
    let dominant = estimate_musical_key(samples, sample_rate, harmonic_config)?;
    let normalized_boundaries = validate_boundaries(boundaries_seconds, duration_seconds)?;
    let boundary_aligned = normalized_boundaries.len() >= 2;
    let (ranges, timeline_complete) = if boundary_aligned {
        boundary_ranges(
            &normalized_boundaries,
            duration_seconds,
            timeline_config.max_windows,
        )
    } else {
        fixed_ranges(duration_seconds, timeline_config)
    };

    let mut observations = Vec::with_capacity(ranges.len());
    for range in ranges {
        let start = seconds_to_sample(range.analysis_start_seconds, sample_rate, samples.len());
        let end = seconds_to_sample(range.analysis_end_seconds, sample_rate, samples.len());
        if end <= start {
            continue;
        }
        let analysis = harmonic_chroma(&samples[start..end], sample_rate, harmonic_config)?;
        let candidates = score_key_candidates(&analysis.chroma.bins, harmonic_config.profile);
        observations.push(HarmonicObservation {
            range,
            analysis,
            candidates,
        });
    }

    let decoded_states = decode_key_states(&observations);
    let stable = stable_run_mask(&decoded_states, MIN_STABLE_WINDOWS);
    let timeline = observations
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            let key = decoded_states
                .get(index)
                .copied()
                .filter(|_| stable.get(index).copied().unwrap_or(false))
                .and_then(|state| {
                    estimate_from_chroma(
                        &observation.analysis,
                        harmonic_config.profile,
                        Some((state.tonic, state.scale)),
                    )
                })
                .filter(|estimate| estimate.confidence >= timeline_config.min_confidence);
            KeyTimelineWindow {
                start_seconds: observation.range.display_start_seconds,
                end_seconds: observation.range.display_end_seconds,
                center_seconds: (observation.range.display_start_seconds
                    + observation.range.display_end_seconds)
                    * 0.5,
                key,
            }
        })
        .collect::<Vec<_>>();
    let segments = build_segments(&timeline);

    Ok(TrackKeyAnalysis {
        schema_version: "audio-analysis-key-track/v2".to_string(),
        sample_rate,
        sample_count: samples.len(),
        duration_seconds,
        dominant,
        timeline_window_seconds: timeline_config.window_seconds,
        timeline_hop_seconds: timeline_config.hop_seconds,
        timeline_min_confidence: timeline_config.min_confidence,
        boundary_aligned,
        timeline_complete,
        timeline,
        segments,
    })
}

fn validate_track(samples: &[f32], sample_rate: u32) -> Result<()> {
    if sample_rate == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate,
            channels: 1,
        });
    }
    if samples.is_empty() {
        return Err(DetectError::InvalidArgument(
            "track key samples must not be empty".to_string(),
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "track key samples must contain only finite values".to_string(),
        ));
    }
    Ok(())
}

fn validate_boundaries(boundaries: &[f64], duration_seconds: f64) -> Result<Vec<f64>> {
    if boundaries.is_empty() {
        return Ok(Vec::new());
    }
    let mut result = Vec::with_capacity(boundaries.len());
    let mut previous = None;
    for boundary in boundaries {
        if !boundary.is_finite() || *boundary < 0.0 || *boundary > duration_seconds {
            return Err(DetectError::InvalidArgument(
                "key timeline boundaries must be finite and inside the analyzed duration"
                    .to_string(),
            ));
        }
        if previous.is_some_and(|value| *boundary <= value) {
            return Err(DetectError::InvalidArgument(
                "key timeline boundaries must be strictly increasing".to_string(),
            ));
        }
        previous = Some(*boundary);
        result.push(*boundary);
    }
    Ok(result)
}

fn boundary_ranges(
    boundaries: &[f64],
    duration_seconds: f64,
    max_windows: usize,
) -> (Vec<ObservationRange>, bool) {
    let requested = boundaries.len().saturating_sub(1);
    let count = requested.min(max_windows);
    let mut ranges = Vec::with_capacity(count);
    for index in 0..count {
        let display_start = boundaries[index];
        let display_end = boundaries[index + 1];
        let analysis_start = if index > 0 {
            boundaries[index - 1]
        } else {
            display_start
        };
        let analysis_end = if index + 2 < boundaries.len() {
            boundaries[index + 2]
        } else {
            display_end
        };
        ranges.push(ObservationRange {
            display_start_seconds: display_start,
            display_end_seconds: display_end,
            analysis_start_seconds: analysis_start.max(0.0),
            analysis_end_seconds: analysis_end.min(duration_seconds),
        });
    }
    (ranges, count == requested)
}

fn fixed_ranges(
    duration_seconds: f64,
    config: KeyTimelineConfig,
) -> (Vec<ObservationRange>, bool) {
    let window_seconds = config.window_seconds.min(duration_seconds).max(f64::EPSILON);
    let mut ranges = Vec::new();
    let mut start = 0.0_f64;
    let mut complete = true;
    while start < duration_seconds {
        if ranges.len() >= config.max_windows {
            complete = false;
            break;
        }
        let end = (start + window_seconds).min(duration_seconds);
        ranges.push(ObservationRange {
            display_start_seconds: start,
            display_end_seconds: end,
            analysis_start_seconds: start,
            analysis_end_seconds: end,
        });
        if end >= duration_seconds {
            break;
        }
        start += config.hop_seconds;
    }
    (ranges, complete)
}

fn seconds_to_sample(seconds: f64, sample_rate: u32, sample_count: usize) -> usize {
    (seconds * sample_rate as f64)
        .round()
        .clamp(0.0, sample_count as f64) as usize
}

fn decode_key_states(observations: &[HarmonicObservation]) -> Vec<KeyState> {
    if observations.is_empty() {
        return Vec::new();
    }
    let states = all_key_states();
    let mut scores: Vec<Vec<f32>> = Vec::with_capacity(observations.len());
    let mut back: Vec<Vec<usize>> = Vec::with_capacity(observations.len());

    for (observation_index, observation) in observations.iter().enumerate() {
        let mut current_scores = vec![f32::NEG_INFINITY; KEY_STATE_COUNT];
        let mut current_back = vec![0_usize; KEY_STATE_COUNT];
        for (state_index, state) in states.iter().enumerate() {
            let emission = emission_score(observation, *state);
            if observation_index == 0 {
                current_scores[state_index] = emission;
                continue;
            }
            let mut best_score = f32::NEG_INFINITY;
            let mut best_previous = 0_usize;
            for (previous_index, previous) in states.iter().enumerate() {
                let score = scores[observation_index - 1][previous_index]
                    + transition_score(*previous, *state)
                    + emission;
                if score > best_score {
                    best_score = score;
                    best_previous = previous_index;
                }
            }
            current_scores[state_index] = best_score;
            current_back[state_index] = best_previous;
        }
        scores.push(current_scores);
        back.push(current_back);
    }

    let mut selected = vec![0_usize; observations.len()];
    let mut state_index = scores
        .last()
        .and_then(|last| {
            last.iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
        })
        .unwrap_or(0);
    for observation_index in (0..observations.len()).rev() {
        selected[observation_index] = state_index;
        if observation_index > 0 {
            state_index = back[observation_index][state_index];
        }
    }
    selected.into_iter().map(|index| states[index]).collect()
}

fn all_key_states() -> [KeyState; KEY_STATE_COUNT] {
    std::array::from_fn(|index| {
        let tonic = NoteName::from_index(index / 2).expect("key-state tonic index is bounded");
        let scale = if index % 2 == 0 {
            MusicalScale::Major
        } else {
            MusicalScale::Minor
        };
        KeyState { tonic, scale }
    })
}

fn emission_score(observation: &HarmonicObservation, state: KeyState) -> f32 {
    if observation.analysis.peak_count == 0 {
        return 0.0;
    }
    observation
        .candidates
        .iter()
        .find(|candidate| candidate.tonic == state.tonic && candidate.scale == state.scale)
        .map_or(0.0, |candidate| candidate.correlation * 1.5)
}

fn transition_score(previous: KeyState, current: KeyState) -> f32 {
    if previous == current {
        return 0.0;
    }
    let previous_tonic = note_index(previous.tonic);
    let current_tonic = note_index(current.tonic);
    let interval = (current_tonic + 12 - previous_tonic) % 12;
    if previous.tonic == current.tonic {
        return -0.20;
    }
    let relative = matches!(
        (previous.scale, current.scale, interval),
        (MusicalScale::Major, MusicalScale::Minor, 9)
            | (MusicalScale::Minor, MusicalScale::Major, 3)
    );
    if relative {
        return -0.14;
    }
    if interval == 5 || interval == 7 {
        return -0.24;
    }
    -0.65
}

fn note_index(note: NoteName) -> usize {
    (0..12)
        .find(|index| NoteName::from_index(*index) == Some(note))
        .expect("all NoteName variants map to a pitch-class index")
}

fn stable_run_mask(states: &[KeyState], min_windows: usize) -> Vec<bool> {
    if states.is_empty() {
        return Vec::new();
    }
    if min_windows <= 1 {
        return vec![true; states.len()];
    }
    if states.len() < min_windows {
        return vec![false; states.len()];
    }
    let mut stable = vec![false; states.len()];
    let mut start = 0_usize;
    while start < states.len() {
        let mut end = start + 1;
        while end < states.len() && states[end] == states[start] {
            end += 1;
        }
        if end - start >= min_windows {
            stable[start..end].fill(true);
        }
        start = end;
    }
    stable
}

fn build_segments(timeline: &[KeyTimelineWindow]) -> Vec<KeySegment> {
    let mut segments: Vec<KeySegment> = Vec::new();
    for window in timeline {
        let Some(key) = &window.key else {
            continue;
        };
        if let Some(last) = segments.last_mut() {
            if last.key.tonic == key.tonic
                && last.key.scale == key.scale
                && window.start_seconds <= last.end_seconds + f64::EPSILON
            {
                let total = last.window_count as f32 + 1.0;
                last.confidence =
                    (last.confidence * last.window_count as f32 + key.confidence) / total;
                last.window_count += 1;
                last.end_seconds = last.end_seconds.max(window.end_seconds);
                if key.confidence > last.key.confidence {
                    last.key = key.clone();
                }
                continue;
            }
        }
        segments.push(KeySegment {
            start_seconds: window.start_seconds,
            end_seconds: window.end_seconds,
            key: key.clone(),
            confidence: key.confidence,
            window_count: 1,
        });
    }
    for index in 0..segments.len().saturating_sub(1) {
        if segments[index].end_seconds > segments[index + 1].start_seconds {
            let boundary =
                (segments[index].end_seconds + segments[index + 1].start_seconds) * 0.5;
            segments[index].end_seconds = boundary;
            segments[index + 1].start_seconds = boundary;
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChromaVector;

    fn test_key(tonic: NoteName, scale: MusicalScale, confidence: f32) -> MusicalKeyEstimate {
        MusicalKeyEstimate {
            tonic,
            scale,
            strength: 1.0,
            confidence,
            runner_up: KeyCandidate {
                tonic: NoteName::B,
                scale: MusicalScale::Minor,
                correlation: 0.0,
            },
            tuning_cents: 0.0,
            chroma: ChromaVector { bins: [0.0; 12] },
            peak_count: 1,
        }
    }

    #[test]
    fn timeline_config_rejects_invalid_values() {
        assert!(KeyTimelineConfig {
            window_seconds: 0.0,
            ..KeyTimelineConfig::default()
        }
        .validate()
        .is_err());
        assert!(KeyTimelineConfig {
            hop_seconds: f64::NAN,
            ..KeyTimelineConfig::default()
        }
        .validate()
        .is_err());
        assert!(KeyTimelineConfig {
            min_confidence: 1.1,
            ..KeyTimelineConfig::default()
        }
        .validate()
        .is_err());
        assert!(KeyTimelineConfig {
            max_windows: 0,
            ..KeyTimelineConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn boundary_ranges_align_outputs_and_expand_analysis_context() {
        let (ranges, complete) = boundary_ranges(&[1.0, 3.0, 5.0, 7.0], 8.0, 10);
        assert!(complete);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[1].display_start_seconds, 3.0);
        assert_eq!(ranges[1].display_end_seconds, 5.0);
        assert_eq!(ranges[1].analysis_start_seconds, 1.0);
        assert_eq!(ranges[1].analysis_end_seconds, 7.0);
    }

    #[test]
    fn related_keys_cost_less_than_remote_modulations() {
        let c_major = KeyState {
            tonic: NoteName::C,
            scale: MusicalScale::Major,
        };
        let a_minor = KeyState {
            tonic: NoteName::A,
            scale: MusicalScale::Minor,
        };
        let fs_major = KeyState {
            tonic: NoteName::FSharp,
            scale: MusicalScale::Major,
        };
        assert!(transition_score(c_major, a_minor) > transition_score(c_major, fs_major));
    }

    #[test]
    fn one_window_is_not_stable_when_two_are_required() {
        let c_major = KeyState {
            tonic: NoteName::C,
            scale: MusicalScale::Major,
        };
        assert_eq!(stable_run_mask(&[c_major], 2), vec![false]);
    }

    #[test]
    fn single_window_key_change_is_not_stable() {
        let c_major = KeyState {
            tonic: NoteName::C,
            scale: MusicalScale::Major,
        };
        let g_major = KeyState {
            tonic: NoteName::G,
            scale: MusicalScale::Major,
        };
        let states = [c_major, c_major, g_major, c_major, c_major];
        assert_eq!(
            stable_run_mask(&states, 2),
            vec![true, true, false, true, true]
        );
    }

    #[test]
    fn overlapping_fixed_windows_do_not_create_overlapping_key_segments() {
        let c_major = test_key(NoteName::C, MusicalScale::Major, 0.8);
        let g_major = test_key(NoteName::G, MusicalScale::Major, 0.9);
        let timeline = vec![
            KeyTimelineWindow {
                start_seconds: 0.0,
                end_seconds: 24.0,
                center_seconds: 12.0,
                key: Some(c_major.clone()),
            },
            KeyTimelineWindow {
                start_seconds: 8.0,
                end_seconds: 32.0,
                center_seconds: 20.0,
                key: Some(c_major.clone()),
            },
            KeyTimelineWindow {
                start_seconds: 16.0,
                end_seconds: 40.0,
                center_seconds: 28.0,
                key: Some(c_major),
            },
            KeyTimelineWindow {
                start_seconds: 24.0,
                end_seconds: 48.0,
                center_seconds: 36.0,
                key: Some(g_major.clone()),
            },
            KeyTimelineWindow {
                start_seconds: 32.0,
                end_seconds: 56.0,
                center_seconds: 44.0,
                key: Some(g_major),
            },
        ];
        let segments = build_segments(&timeline);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].end_seconds, 32.0);
        assert_eq!(segments[1].start_seconds, 32.0);
    }

    #[test]
    fn silence_keeps_uncertain_windows_explicit() {
        let analysis = analyze_key_track(
            &vec![0.0; 16_000 * 3],
            16_000,
            HarmonicKeyConfig {
                fft_size: 512,
                hop_size: 256,
                ..HarmonicKeyConfig::default()
            },
            KeyTimelineConfig {
                window_seconds: 1.0,
                hop_seconds: 0.5,
                min_confidence: 0.1,
                ..KeyTimelineConfig::default()
            },
        )
        .expect("track key timeline");

        assert!(analysis.dominant.is_none());
        assert!(!analysis.timeline.is_empty());
        assert!(analysis.timeline.iter().all(|window| window.key.is_none()));
        assert!(analysis.segments.is_empty());
        assert!(analysis.timeline_complete);
        assert!(!analysis.boundary_aligned);
    }

    #[test]
    fn supplied_boundaries_are_reported_as_authoritative_alignment() {
        let analysis = analyze_key_track_with_boundaries(
            &vec![0.0; 8_000 * 4],
            8_000,
            HarmonicKeyConfig {
                fft_size: 256,
                hop_size: 128,
                max_frequency_hz: 3_000.0,
                ..HarmonicKeyConfig::default()
            },
            KeyTimelineConfig::default(),
            &[0.0, 1.0, 2.0, 3.0, 4.0],
        )
        .expect("boundary-aligned timeline");
        assert!(analysis.boundary_aligned);
        assert_eq!(analysis.timeline.len(), 4);
        assert_eq!(analysis.timeline[2].start_seconds, 2.0);
        assert_eq!(analysis.timeline[2].end_seconds, 3.0);
    }

    #[test]
    fn max_windows_reports_incomplete_timeline_instead_of_hiding_truncation() {
        let analysis = analyze_key_track(
            &vec![0.0; 8_000 * 5],
            8_000,
            HarmonicKeyConfig {
                fft_size: 256,
                hop_size: 128,
                max_frequency_hz: 3_000.0,
                ..HarmonicKeyConfig::default()
            },
            KeyTimelineConfig {
                window_seconds: 1.0,
                hop_seconds: 0.5,
                min_confidence: 0.1,
                max_windows: 2,
            },
        )
        .expect("bounded timeline");

        assert_eq!(analysis.timeline.len(), 2);
        assert!(!analysis.timeline_complete);
    }
}
