#![doc = include_str!("../README.md")]

use std::fmt::Write as _;

use audio_analysis_pitch::SungNoteSegment;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const ULTRASTAR_V1_BEATS_PER_QUARTER: f64 = 4.0;
const MIDDLE_C_MIDI: i16 = 60;

/// Karaoke chart construction or export failure.
#[derive(Debug, Error, PartialEq)]
pub enum KaraokeError {
    /// An input or option violated a deterministic chart invariant.
    #[error("invalid karaoke input: {0}")]
    InvalidArgument(String),
}

/// Result type used by this capability.
pub type Result<T> = std::result::Result<T, KaraokeError>;

/// One already-aligned lyric fragment.
///
/// The text is a presentation fragment and its whitespace is preserved. For
/// best karaoke quality this should be a syllable. Word-level fragments are
/// also valid and deliberately remain word-level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AlignedLyricFragment {
    /// Text rendered for this fragment.
    pub text: String,
    /// Fragment start relative to the beginning of the audio.
    pub start_seconds: f64,
    /// Fragment end relative to the beginning of the audio.
    pub end_seconds: f64,
    /// Optional alignment/transcription confidence in `0.0..=1.0`.
    pub confidence: Option<f32>,
}

impl AlignedLyricFragment {
    /// Validates the fragment.
    pub fn validate(&self) -> Result<()> {
        if self.text.trim().is_empty() {
            return Err(invalid("lyric text must contain a non-whitespace character"));
        }
        if self.text.contains(['\r', '\n']) {
            return Err(invalid("lyric text must not contain line breaks"));
        }
        validate_time_range(self.start_seconds, self.end_seconds, "lyric")?;
        validate_optional_confidence(self.confidence, "lyric confidence")
    }
}

/// Evidence retained for one generated karaoke note.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KaraokeNoteEvidence {
    /// Confidence reported by the selected pitch segment.
    pub pitch_confidence: f32,
    /// Confidence reported by lyric alignment, when available.
    pub lyric_confidence: Option<f32>,
    /// Fraction of the lyric fragment covered by the selected pitch segment.
    pub overlap_ratio: f32,
}

/// One neutral pitched lyric note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KaraokeNote {
    /// Lyric fragment rendered for this note.
    pub text: String,
    /// Note start relative to the beginning of the audio.
    pub start_seconds: f64,
    /// Note end relative to the beginning of the audio.
    pub end_seconds: f64,
    /// Nearest MIDI note number.
    pub midi_note: u8,
    /// Evidence used to construct the note.
    pub evidence: KaraokeNoteEvidence,
}

impl KaraokeNote {
    /// Validates this note.
    pub fn validate(&self) -> Result<()> {
        if self.text.trim().is_empty() {
            return Err(invalid("karaoke note text must not be empty"));
        }
        if self.text.contains(['\r', '\n']) {
            return Err(invalid("karaoke note text must not contain line breaks"));
        }
        validate_time_range(self.start_seconds, self.end_seconds, "karaoke note")?;
        validate_confidence(self.evidence.pitch_confidence, "pitch confidence")?;
        validate_optional_confidence(self.evidence.lyric_confidence, "lyric confidence")?;
        validate_confidence(self.evidence.overlap_ratio, "overlap ratio")
    }
}

/// One lyric phrase/line in a neutral karaoke chart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KaraokePhrase {
    /// Notes in chronological order.
    pub notes: Vec<KaraokeNote>,
}

impl KaraokePhrase {
    /// Validates this phrase.
    pub fn validate(&self) -> Result<()> {
        if self.notes.is_empty() {
            return Err(invalid("karaoke phrase must contain at least one note"));
        }
        for note in &self.notes {
            note.validate()?;
        }
        validate_non_overlapping_notes(&self.notes, "phrase")
    }
}

/// Neutral karaoke chart independent of any file format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KaraokeChart {
    /// Musical tempo in quarter notes per minute.
    pub tempo_bpm: f64,
    /// Audio time corresponding to musical beat zero.
    pub beat_zero_seconds: f64,
    /// Chronological lyric phrases.
    pub phrases: Vec<KaraokePhrase>,
}

impl KaraokeChart {
    /// Validates this chart.
    pub fn validate(&self) -> Result<()> {
        validate_tempo(self.tempo_bpm)?;
        if !self.beat_zero_seconds.is_finite() || self.beat_zero_seconds < 0.0 {
            return Err(invalid(
                "beat_zero_seconds must be finite and non-negative",
            ));
        }
        if self.phrases.is_empty() {
            return Err(invalid("karaoke chart must contain at least one phrase"));
        }

        let mut previous_note: Option<&KaraokeNote> = None;
        for phrase in &self.phrases {
            phrase.validate()?;
            for note in &phrase.notes {
                if let Some(previous) = previous_note {
                    if note.start_seconds < previous.end_seconds {
                        return Err(invalid(
                            "karaoke chart notes must be chronological and non-overlapping",
                        ));
                    }
                }
                previous_note = Some(note);
            }
        }
        Ok(())
    }
}

/// Options for deterministic lyric/pitch fusion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KaraokeChartBuildOptions {
    /// Musical tempo in quarter notes per minute.
    pub tempo_bpm: f64,
    /// Audio time corresponding to musical beat zero.
    pub beat_zero_seconds: f64,
    /// Minimum fraction of a lyric fragment that must overlap a pitch segment.
    pub min_overlap_ratio: f64,
    /// Minimum duration of a fused note.
    pub min_note_duration_seconds: f64,
    /// Gap after which a new lyric phrase is started.
    pub phrase_gap_seconds: f64,
}

impl Default for KaraokeChartBuildOptions {
    fn default() -> Self {
        Self {
            tempo_bpm: 120.0,
            beat_zero_seconds: 0.0,
            min_overlap_ratio: 0.35,
            min_note_duration_seconds: 0.05,
            phrase_gap_seconds: 1.0,
        }
    }
}

impl KaraokeChartBuildOptions {
    /// Validates these options.
    pub fn validate(self) -> Result<()> {
        validate_tempo(self.tempo_bpm)?;
        if !self.beat_zero_seconds.is_finite() || self.beat_zero_seconds < 0.0 {
            return Err(invalid(
                "beat_zero_seconds must be finite and non-negative",
            ));
        }
        if !self.min_overlap_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.min_overlap_ratio)
        {
            return Err(invalid(
                "min_overlap_ratio must be finite and between 0.0 and 1.0",
            ));
        }
        if !self.min_note_duration_seconds.is_finite()
            || self.min_note_duration_seconds <= 0.0
        {
            return Err(invalid(
                "min_note_duration_seconds must be finite and greater than zero",
            ));
        }
        if !self.phrase_gap_seconds.is_finite() || self.phrase_gap_seconds < 0.0 {
            return Err(invalid(
                "phrase_gap_seconds must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Result of building a chart, including evidence about skipped fragments.
#[derive(Debug, Clone, PartialEq)]
pub struct KaraokeChartBuildResult {
    /// Constructed neutral chart.
    pub chart: KaraokeChart,
    /// Stable diagnostics for fragments that could not become notes.
    pub diagnostics: Vec<String>,
}

/// Fuses aligned lyric fragments with existing pitch segments.
///
/// Each lyric fragment selects the pitch segment with the strongest
/// overlap-duration × pitch-confidence score. This first slice intentionally
/// produces at most one pitched note per lyric fragment; melisma splitting is
/// a later refinement rather than hidden heuristic behavior.
pub fn build_karaoke_chart(
    lyrics: &[AlignedLyricFragment],
    pitch_segments: &[SungNoteSegment],
    options: KaraokeChartBuildOptions,
) -> Result<KaraokeChartBuildResult> {
    options.validate()?;
    validate_lyrics(lyrics)?;
    validate_pitch_segments(pitch_segments)?;

    let mut diagnostics = Vec::new();
    let mut notes = Vec::new();

    for (lyric_index, lyric) in lyrics.iter().enumerate() {
        let lyric_duration = lyric.end_seconds - lyric.start_seconds;
        let mut best: Option<(&SungNoteSegment, f64, f64)> = None;

        for segment in pitch_segments {
            let overlap_start = lyric.start_seconds.max(segment.start_seconds);
            let overlap_end = lyric.end_seconds.min(segment.end_seconds);
            let overlap = (overlap_end - overlap_start).max(0.0);
            if overlap <= 0.0 {
                continue;
            }
            let overlap_ratio = overlap / lyric_duration;
            if overlap_ratio < options.min_overlap_ratio {
                continue;
            }
            let score = overlap * f64::from(segment.confidence);
            let replace = best
                .as_ref()
                .map(|(current, current_overlap, current_score)| {
                    score > *current_score
                        || (score == *current_score && overlap > *current_overlap)
                        || (score == *current_score
                            && overlap == *current_overlap
                            && segment.start_seconds < current.start_seconds)
                })
                .unwrap_or(true);
            if replace {
                best = Some((segment, overlap, score));
            }
        }

        let Some((segment, overlap, _)) = best else {
            diagnostics.push(format!(
                "lyric[{lyric_index}] had no pitch segment meeting the overlap threshold"
            ));
            continue;
        };

        let start_seconds = lyric.start_seconds.max(segment.start_seconds);
        let end_seconds = lyric.end_seconds.min(segment.end_seconds);
        if end_seconds - start_seconds < options.min_note_duration_seconds {
            diagnostics.push(format!(
                "lyric[{lyric_index}] matched pitch but the fused duration was below the minimum"
            ));
            continue;
        }

        let rounded_midi = segment.midi_note.round();
        if !rounded_midi.is_finite() || !(0.0..=127.0).contains(&rounded_midi) {
            return Err(invalid(format!(
                "pitch segment for lyric[{lyric_index}] resolved outside the MIDI range"
            )));
        }

        notes.push(KaraokeNote {
            text: lyric.text.clone(),
            start_seconds,
            end_seconds,
            midi_note: rounded_midi as u8,
            evidence: KaraokeNoteEvidence {
                pitch_confidence: segment.confidence,
                lyric_confidence: lyric.confidence,
                overlap_ratio: (overlap / lyric_duration) as f32,
            },
        });
    }

    if notes.is_empty() {
        return Err(invalid(
            "no lyric fragments could be fused with pitched vocal segments",
        ));
    }
    validate_non_overlapping_notes(&notes, "fused")?;

    let mut phrases: Vec<KaraokePhrase> = Vec::new();
    for note in notes {
        let starts_new_phrase = phrases
            .last()
            .and_then(|phrase| phrase.notes.last())
            .map(|previous| note.start_seconds - previous.end_seconds > options.phrase_gap_seconds)
            .unwrap_or(true);
        if starts_new_phrase {
            phrases.push(KaraokePhrase { notes: vec![note] });
        } else if let Some(phrase) = phrases.last_mut() {
            phrase.notes.push(note);
        }
    }

    let chart = KaraokeChart {
        tempo_bpm: options.tempo_bpm,
        beat_zero_seconds: options.beat_zero_seconds,
        phrases,
    };
    chart.validate()?;

    Ok(KaraokeChartBuildResult { chart, diagnostics })
}

/// Required metadata for an UltraStar v1 text file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UltraStarV1Metadata {
    /// Song title.
    pub title: String,
    /// Song artist.
    pub artist: String,
    /// Audio file reference written to `#MP3`.
    pub audio_file: String,
}

impl UltraStarV1Metadata {
    /// Validates header values against the v1 text grammar used here.
    pub fn validate(&self) -> Result<()> {
        validate_header_value(&self.title, "title")?;
        validate_header_value(&self.artist, "artist")?;
        validate_header_value(&self.audio_file, "audio_file")
    }
}

/// Exports a neutral chart as an UltraStar v1-compatible UTF-8 text file.
///
/// UltraStar v1 implicitly quadruples `#BPM` for its integer note grid. This
/// exporter therefore writes the chart's real quarter-note BPM and converts
/// seconds to four UltraStar beats per quarter note only at this boundary.
pub fn export_ultrastar_v1(
    chart: &KaraokeChart,
    metadata: &UltraStarV1Metadata,
) -> Result<String> {
    chart.validate()?;
    metadata.validate()?;

    let gap_millis = (chart.beat_zero_seconds * 1_000.0).round();
    if !(0.0..=(u64::MAX as f64)).contains(&gap_millis) {
        return Err(invalid("beat_zero_seconds cannot be represented as #GAP"));
    }

    let mut output = String::new();
    writeln!(output, "#TITLE:{}", metadata.title).expect("writing to String cannot fail");
    writeln!(output, "#ARTIST:{}", metadata.artist).expect("writing to String cannot fail");
    writeln!(output, "#MP3:{}", metadata.audio_file).expect("writing to String cannot fail");
    writeln!(output, "#BPM:{}", decimal(chart.tempo_bpm)).expect("writing to String cannot fail");
    writeln!(output, "#GAP:{gap_millis:.0}").expect("writing to String cannot fail");

    for (phrase_index, phrase) in chart.phrases.iter().enumerate() {
        for note in &phrase.notes {
            let start_beat = ultrastar_beat(chart, note.start_seconds)?;
            let end_beat = ultrastar_beat(chart, note.end_seconds)?;
            let duration = end_beat.saturating_sub(start_beat).max(1);
            let pitch = i16::from(note.midi_note) - MIDDLE_C_MIDI;
            writeln!(
                output,
                ": {start_beat} {duration} {pitch} {}",
                note.text
            )
            .expect("writing to String cannot fail");
        }

        if let Some(next_phrase) = chart.phrases.get(phrase_index + 1) {
            let current_end = phrase
                .notes
                .last()
                .expect("validated phrase contains a note")
                .end_seconds;
            let next_start = next_phrase
                .notes
                .first()
                .expect("validated phrase contains a note")
                .start_seconds;
            let marker_beat = ultrastar_beat(chart, current_end)?;
            let next_start_beat = ultrastar_beat(chart, next_start)?;
            if marker_beat >= next_start_beat {
                return Err(invalid(
                    "phrase gap disappears on the UltraStar integer beat grid",
                ));
            }
            writeln!(output, "- {marker_beat}").expect("writing to String cannot fail");
        }
    }
    writeln!(output, "E").expect("writing to String cannot fail");
    Ok(output)
}

fn ultrastar_beat(chart: &KaraokeChart, seconds: f64) -> Result<u64> {
    let relative_seconds = seconds - chart.beat_zero_seconds;
    if relative_seconds < 0.0 {
        return Err(invalid(
            "a karaoke note starts before beat_zero_seconds and cannot be encoded in UltraStar v1",
        ));
    }
    let beats_per_second = chart.tempo_bpm * ULTRASTAR_V1_BEATS_PER_QUARTER / 60.0;
    let beat = (relative_seconds * beats_per_second).round();
    if !(0.0..=(u64::MAX as f64)).contains(&beat) {
        return Err(invalid("UltraStar beat position is out of range"));
    }
    Ok(beat as u64)
}

fn validate_lyrics(lyrics: &[AlignedLyricFragment]) -> Result<()> {
    if lyrics.is_empty() {
        return Err(invalid("at least one aligned lyric fragment is required"));
    }
    for lyric in lyrics {
        lyric.validate()?;
    }
    for pair in lyrics.windows(2) {
        if pair[1].start_seconds < pair[0].end_seconds {
            return Err(invalid(
                "aligned lyric fragments must be chronological and non-overlapping",
            ));
        }
    }
    Ok(())
}

fn validate_pitch_segments(segments: &[SungNoteSegment]) -> Result<()> {
    if segments.is_empty() {
        return Err(invalid("at least one sung pitch segment is required"));
    }
    for segment in segments {
        validate_time_range(segment.start_seconds, segment.end_seconds, "pitch segment")?;
        if !segment.frequency_hz.is_finite() || segment.frequency_hz <= 0.0 {
            return Err(invalid(
                "pitch segment frequency_hz must be finite and greater than zero",
            ));
        }
        if !segment.midi_note.is_finite() || !(0.0..=127.0).contains(&segment.midi_note) {
            return Err(invalid(
                "pitch segment midi_note must be finite and in the MIDI range",
            ));
        }
        validate_confidence(segment.confidence, "pitch segment confidence")?;
        if segment.frames == 0 {
            return Err(invalid("pitch segment must contain at least one frame"));
        }
    }
    for pair in segments.windows(2) {
        if pair[1].start_seconds < pair[0].end_seconds {
            return Err(invalid(
                "pitch segments must be chronological and non-overlapping for a single lead voice",
            ));
        }
    }
    Ok(())
}

fn validate_non_overlapping_notes(notes: &[KaraokeNote], context: &str) -> Result<()> {
    for pair in notes.windows(2) {
        if pair[1].start_seconds < pair[0].end_seconds {
            return Err(invalid(format!(
                "{context} karaoke notes must be chronological and non-overlapping"
            )));
        }
    }
    Ok(())
}

fn validate_time_range(start_seconds: f64, end_seconds: f64, name: &str) -> Result<()> {
    if !start_seconds.is_finite()
        || !end_seconds.is_finite()
        || start_seconds < 0.0
        || end_seconds <= start_seconds
    {
        return Err(invalid(format!(
            "{name} start/end seconds must be finite, non-negative, and ordered"
        )));
    }
    Ok(())
}

fn validate_tempo(tempo_bpm: f64) -> Result<()> {
    if !tempo_bpm.is_finite() || tempo_bpm <= 0.0 {
        return Err(invalid("tempo_bpm must be finite and greater than zero"));
    }
    Ok(())
}

fn validate_confidence(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(invalid(format!(
            "{name} must be finite and between 0.0 and 1.0"
        )));
    }
    Ok(())
}

fn validate_optional_confidence(value: Option<f32>, name: &str) -> Result<()> {
    if let Some(value) = value {
        validate_confidence(value, name)?;
    }
    Ok(())
}

fn validate_header_value(value: &str, name: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(invalid(format!("UltraStar {name} must not be empty")));
    }
    if value.contains(['\r', '\n', ':']) {
        return Err(invalid(format!(
            "UltraStar {name} must not contain a colon or line break"
        )));
    }
    Ok(())
}

fn decimal(value: f64) -> String {
    let mut rendered = format!("{value:.6}");
    while rendered.contains('.') && rendered.ends_with('0') {
        rendered.pop();
    }
    if rendered.ends_with('.') {
        rendered.pop();
    }
    rendered
}

fn invalid(message: impl Into<String>) -> KaraokeError {
    KaraokeError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pitch_segment(
        start_seconds: f64,
        end_seconds: f64,
        midi_note: f32,
        confidence: f32,
    ) -> SungNoteSegment {
        SungNoteSegment {
            start_seconds,
            end_seconds,
            frequency_hz: 440.0,
            midi_note,
            note_name: "A4".to_string(),
            confidence,
            frames: 1,
        }
    }

    fn lyric(
        text: &str,
        start_seconds: f64,
        end_seconds: f64,
        confidence: Option<f32>,
    ) -> AlignedLyricFragment {
        AlignedLyricFragment {
            text: text.to_string(),
            start_seconds,
            end_seconds,
            confidence,
        }
    }

    #[test]
    fn builds_notes_from_strongest_overlap_and_groups_phrases() {
        let lyrics = vec![
            lyric("Hel", 0.0, 0.25, Some(0.95)),
            lyric("lo", 0.25, 0.50, Some(0.90)),
            lyric(" world", 1.25, 1.55, Some(0.88)),
        ];
        let pitch = vec![
            pitch_segment(0.0, 0.10, 59.8, 0.60),
            pitch_segment(0.10, 0.25, 60.1, 0.95),
            pitch_segment(0.25, 0.50, 64.2, 0.90),
            pitch_segment(1.25, 1.55, 67.0, 0.85),
        ];
        let result = build_karaoke_chart(
            &lyrics,
            &pitch,
            KaraokeChartBuildOptions {
                tempo_bpm: 120.0,
                min_overlap_ratio: 0.35,
                min_note_duration_seconds: 0.05,
                phrase_gap_seconds: 0.5,
                ..KaraokeChartBuildOptions::default()
            },
        )
        .unwrap();

        assert!(result.diagnostics.is_empty());
        assert_eq!(result.chart.phrases.len(), 2);
        assert_eq!(result.chart.phrases[0].notes.len(), 2);
        assert_eq!(result.chart.phrases[0].notes[0].midi_note, 60);
        assert_eq!(result.chart.phrases[0].notes[1].midi_note, 64);
        assert_eq!(result.chart.phrases[1].notes[0].midi_note, 67);
        assert_eq!(
            result.chart.phrases[0].notes[0].evidence,
            KaraokeNoteEvidence {
                pitch_confidence: 0.95,
                lyric_confidence: Some(0.95),
                overlap_ratio: 0.6,
            }
        );
    }

    #[test]
    fn records_unmatched_lyrics_without_inventing_pitch() {
        let lyrics = vec![
            lyric("one", 0.0, 0.25, None),
            lyric(" two", 0.50, 0.75, None),
        ];
        let pitch = vec![pitch_segment(0.0, 0.25, 60.0, 0.9)];

        let result = build_karaoke_chart(
            &lyrics,
            &pitch,
            KaraokeChartBuildOptions::default(),
        )
        .unwrap();

        assert_eq!(result.chart.phrases[0].notes.len(), 1);
        assert_eq!(result.diagnostics.len(), 1);
        assert!(result.diagnostics[0].contains("lyric[1]"));
    }

    #[test]
    fn exports_ultrastar_v1_with_format_specific_quantization() {
        let chart = KaraokeChart {
            tempo_bpm: 120.0,
            beat_zero_seconds: 0.5,
            phrases: vec![
                KaraokePhrase {
                    notes: vec![
                        KaraokeNote {
                            text: "Hel".to_string(),
                            start_seconds: 0.5,
                            end_seconds: 0.75,
                            midi_note: 60,
                            evidence: KaraokeNoteEvidence {
                                pitch_confidence: 0.95,
                                lyric_confidence: Some(0.9),
                                overlap_ratio: 1.0,
                            },
                        },
                        KaraokeNote {
                            text: "lo".to_string(),
                            start_seconds: 0.75,
                            end_seconds: 1.0,
                            midi_note: 64,
                            evidence: KaraokeNoteEvidence {
                                pitch_confidence: 0.9,
                                lyric_confidence: Some(0.9),
                                overlap_ratio: 1.0,
                            },
                        },
                    ],
                },
                KaraokePhrase {
                    notes: vec![KaraokeNote {
                        text: " world".to_string(),
                        start_seconds: 1.5,
                        end_seconds: 1.75,
                        midi_note: 67,
                        evidence: KaraokeNoteEvidence {
                            pitch_confidence: 0.85,
                            lyric_confidence: Some(0.88),
                            overlap_ratio: 1.0,
                        },
                    }],
                },
            ],
        };
        let metadata = UltraStarV1Metadata {
            title: "Example".to_string(),
            artist: "Singer".to_string(),
            audio_file: "song.ogg".to_string(),
        };

        let output = export_ultrastar_v1(&chart, &metadata).unwrap();
        assert_eq!(
            output,
            "#TITLE:Example\n#ARTIST:Singer\n#MP3:song.ogg\n#BPM:120\n#GAP:500\n: 0 2 0 Hel\n: 2 2 4 lo\n- 4\n: 8 2 7  world\nE\n"
        );
    }

    #[test]
    fn exporter_rejects_notes_before_beat_zero() {
        let chart = KaraokeChart {
            tempo_bpm: 120.0,
            beat_zero_seconds: 1.0,
            phrases: vec![KaraokePhrase {
                notes: vec![KaraokeNote {
                    text: "early".to_string(),
                    start_seconds: 0.5,
                    end_seconds: 0.75,
                    midi_note: 60,
                    evidence: KaraokeNoteEvidence {
                        pitch_confidence: 1.0,
                        lyric_confidence: None,
                        overlap_ratio: 1.0,
                    },
                }],
            }],
        };
        let metadata = UltraStarV1Metadata {
            title: "Example".to_string(),
            artist: "Singer".to_string(),
            audio_file: "song.ogg".to_string(),
        };

        let error = export_ultrastar_v1(&chart, &metadata).unwrap_err();
        assert!(error.to_string().contains("before beat_zero_seconds"));
    }

    #[test]
    fn rejects_overlapping_single_voice_pitch_segments() {
        let lyrics = vec![lyric("word", 0.0, 0.4, None)];
        let pitch = vec![
            pitch_segment(0.0, 0.3, 60.0, 0.9),
            pitch_segment(0.2, 0.4, 64.0, 0.9),
        ];

        let error = build_karaoke_chart(
            &lyrics,
            &pitch,
            KaraokeChartBuildOptions::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("single lead voice"));
    }
}
