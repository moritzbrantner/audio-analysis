#![doc = include_str!("../README.md")]

use audio_contracts::{DetectError, Result};
use audio_generation_midi::karaoke::{KaraokeChart, KaraokeNote};

const ULTRASTAR_VERSION: &str = "1.0.0";
const GRID_UNITS_PER_QUARTER_NOTE: f64 = 4.0;
const MIDDLE_C_MIDI_NOTE: i16 = 60;

/// Required metadata for a single-voice UltraStar v1 song file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UltraStarV1Metadata {
    /// Song title written to `#TITLE`.
    pub title: String,
    /// Artist written to `#ARTIST`.
    pub artist: String,
    /// Relative audio file reference written to `#MP3`.
    pub audio_file: String,
}

impl UltraStarV1Metadata {
    /// Creates metadata for an UltraStar v1 export.
    pub fn new(
        title: impl Into<String>,
        artist: impl Into<String>,
        audio_file: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            artist: artist.into(),
            audio_file: audio_file.into(),
        }
    }

    /// Validates metadata without touching the filesystem.
    pub fn validate(&self) -> Result<()> {
        validate_header_text(&self.title, "title")?;
        validate_header_text(&self.artist, "artist")?;
        validate_relative_file_reference(&self.audio_file)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuantizedNote {
    start_beat: u64,
    end_beat: u64,
    pitch: i16,
    text: String,
}

impl QuantizedNote {
    fn duration(&self) -> u64 {
        self.end_beat - self.start_beat
    }
}

/// Serializes a neutral karaoke chart as a single-voice UltraStar v1 text file.
///
/// The neutral chart stays authoritative. UltraStar-specific integer timing is
/// introduced only here, and export fails when rounding would collapse a note,
/// overlap notes, place a note before the gap, or leave no legal phrase-break
/// beat. The exporter never changes neutral note boundaries to force a valid
/// file.
pub fn export_ultrastar_v1(
    chart: &KaraokeChart,
    metadata: &UltraStarV1Metadata,
) -> Result<String> {
    chart.validate()?;
    metadata.validate()?;

    let phrases = quantize_chart(chart)?;
    let mut output = String::new();
    output.push_str("#VERSION:");
    output.push_str(ULTRASTAR_VERSION);
    output.push('\n');
    push_header(&mut output, "MP3", &metadata.audio_file);
    push_header(&mut output, "TITLE", &metadata.title);
    push_header(&mut output, "ARTIST", &metadata.artist);
    push_header(&mut output, "BPM", &format_decimal(f64::from(chart.tempo_bpm), 6));
    push_header(
        &mut output,
        "GAP",
        &format_decimal(f64::from(chart.beat_zero_seconds) * 1_000.0, 3),
    );

    for (phrase_index, phrase) in phrases.iter().enumerate() {
        for note in phrase {
            output.push_str(": ");
            output.push_str(&note.start_beat.to_string());
            output.push(' ');
            output.push_str(&note.duration().to_string());
            output.push(' ');
            output.push_str(&note.pitch.to_string());
            output.push(' ');
            output.push_str(&note.text);
            output.push('\n');
        }

        if let Some(next_phrase) = phrases.get(phrase_index + 1) {
            let marker = phrase
                .last()
                .ok_or_else(|| invalid_argument("quantized phrase unexpectedly contains no notes"))?
                .end_beat;
            let next_start = next_phrase
                .first()
                .ok_or_else(|| invalid_argument("quantized phrase unexpectedly contains no notes"))?
                .start_beat;
            if marker >= next_start {
                return Err(invalid_argument(format!(
                    "UltraStar quantization leaves no safe phrase marker between phrases {} and {}",
                    phrase_index,
                    phrase_index + 1
                )));
            }
            output.push_str("- ");
            output.push_str(&marker.to_string());
            output.push('\n');
        }
    }

    output.push_str("E\n");
    Ok(output)
}

fn quantize_chart(chart: &KaraokeChart) -> Result<Vec<Vec<QuantizedNote>>> {
    let units_per_second = f64::from(chart.tempo_bpm) * GRID_UNITS_PER_QUARTER_NOTE / 60.0;
    let gap_seconds = f64::from(chart.beat_zero_seconds);
    let mut previous_end = None;
    let mut phrases = Vec::with_capacity(chart.phrases.len());

    for (phrase_index, phrase) in chart.phrases.iter().enumerate() {
        let mut quantized_phrase = Vec::with_capacity(phrase.notes.len());
        for (note_index, note) in phrase.notes.iter().enumerate() {
            let quantized = quantize_note(
                note,
                gap_seconds,
                units_per_second,
                phrase_index,
                note_index,
            )?;

            if let Some(previous_end) = previous_end {
                if quantized.start_beat < previous_end {
                    return Err(invalid_argument(format!(
                        "UltraStar quantization makes phrase {phrase_index} note {note_index} overlap the previous note"
                    )));
                }
            }
            previous_end = Some(quantized.end_beat);
            quantized_phrase.push(quantized);
        }
        phrases.push(quantized_phrase);
    }

    Ok(phrases)
}

fn quantize_note(
    note: &KaraokeNote,
    gap_seconds: f64,
    units_per_second: f64,
    phrase_index: usize,
    note_index: usize,
) -> Result<QuantizedNote> {
    let start_seconds = f64::from(note.start_seconds);
    let end_seconds = f64::from(note.end_seconds);
    if start_seconds < gap_seconds {
        return Err(invalid_argument(format!(
            "phrase {phrase_index} note {note_index} starts before the UltraStar #GAP origin"
        )));
    }

    let start_beat = quantize_non_negative_time(
        start_seconds,
        gap_seconds,
        units_per_second,
        "note start",
    )?;
    let end_beat = quantize_non_negative_time(
        end_seconds,
        gap_seconds,
        units_per_second,
        "note end",
    )?;
    if end_beat <= start_beat {
        return Err(invalid_argument(format!(
            "UltraStar quantization collapses phrase {phrase_index} note {note_index} to zero duration"
        )));
    }

    Ok(QuantizedNote {
        start_beat,
        end_beat,
        pitch: i16::from(note.midi_note) - MIDDLE_C_MIDI_NOTE,
        text: note.text.clone(),
    })
}

fn quantize_non_negative_time(
    seconds: f64,
    gap_seconds: f64,
    units_per_second: f64,
    label: &str,
) -> Result<u64> {
    let units = (seconds - gap_seconds) * units_per_second;
    if !units.is_finite() || units < 0.0 {
        return Err(invalid_argument(format!(
            "UltraStar {label} is outside the representable non-negative beat grid"
        )));
    }

    let rounded = units.round();
    if rounded > u64::MAX as f64 {
        return Err(invalid_argument(format!(
            "UltraStar {label} exceeds the representable beat grid"
        )));
    }
    Ok(rounded as u64)
}

fn validate_header_text(value: &str, label: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(invalid_argument(format!(
            "UltraStar {label} must not be empty"
        )));
    }
    if value.contains(['\r', '\n', '\0']) {
        return Err(invalid_argument(format!(
            "UltraStar {label} must not contain line breaks or NUL characters"
        )));
    }
    Ok(())
}

fn validate_relative_file_reference(value: &str) -> Result<()> {
    validate_header_text(value, "audio file")?;

    if value.starts_with('/')
        || value.starts_with('\\')
        || has_windows_drive_prefix(value)
        || value.contains("://")
        || value.split(['/', '\\']).any(|segment| segment == "..")
    {
        return Err(invalid_argument(
            "UltraStar audio file must be a relative file reference without parent traversal",
        ));
    }
    Ok(())
}

fn has_windows_drive_prefix(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

fn push_header(output: &mut String, name: &str, value: &str) {
    output.push('#');
    output.push_str(name);
    output.push(':');
    output.push_str(value);
    output.push('\n');
}

fn format_decimal(value: f64, fractional_digits: usize) -> String {
    let rendered = format!("{value:.fractional_digits$}");
    let trimmed = rendered.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_generation_midi::karaoke::{
        KaraokeNoteEvidence, KaraokePhrase,
    };

    fn note(text: &str, start_seconds: f32, end_seconds: f32, midi_note: u8) -> KaraokeNote {
        KaraokeNote {
            text: text.to_string(),
            start_seconds,
            end_seconds,
            midi_note,
            evidence: KaraokeNoteEvidence {
                pitch_confidence: 0.9,
                lyric_confidence: Some(0.8),
                overlap_ratio: 1.0,
            },
        }
    }

    fn metadata() -> UltraStarV1Metadata {
        UltraStarV1Metadata::new("Example", "Artist", "audio/song.ogg")
    }

    #[test]
    fn exports_ultrastar_v1_with_safe_integer_grid() {
        let chart = KaraokeChart {
            tempo_bpm: 120.0,
            beat_zero_seconds: 0.5,
            phrases: vec![
                KaraokePhrase {
                    notes: vec![
                        note("Hel", 0.5, 0.75, 60),
                        note("lo", 0.75, 1.0, 64),
                    ],
                },
                KaraokePhrase {
                    notes: vec![note(" world", 1.5, 1.75, 58)],
                },
            ],
        };

        let exported = export_ultrastar_v1(&chart, &metadata()).expect("valid chart exports");
        assert_eq!(
            exported,
            "#VERSION:1.0.0\n#MP3:audio/song.ogg\n#TITLE:Example\n#ARTIST:Artist\n#BPM:120\n#GAP:500\n: 0 2 0 Hel\n: 2 2 4 lo\n- 4\n: 8 2 -2  world\nE\n"
        );
    }

    #[test]
    fn rejects_note_collapsed_by_quantization_instead_of_stretching_it() {
        let chart = KaraokeChart {
            tempo_bpm: 120.0,
            beat_zero_seconds: 0.0,
            phrases: vec![KaraokePhrase {
                notes: vec![note("short", 0.001, 0.02, 60)],
            }],
        };

        let error = export_ultrastar_v1(&chart, &metadata()).expect_err("note must fail closed");
        assert!(error.to_string().contains("zero duration"));
    }

    #[test]
    fn rejects_note_before_gap() {
        let chart = KaraokeChart {
            tempo_bpm: 120.0,
            beat_zero_seconds: 0.5,
            phrases: vec![KaraokePhrase {
                notes: vec![note("early", 0.25, 0.75, 60)],
            }],
        };

        let error = export_ultrastar_v1(&chart, &metadata()).expect_err("pre-gap note must fail");
        assert!(error.to_string().contains("before the UltraStar #GAP"));
    }

    #[test]
    fn rejects_phrase_boundary_without_safe_marker_beat() {
        let chart = KaraokeChart {
            tempo_bpm: 120.0,
            beat_zero_seconds: 0.0,
            phrases: vec![
                KaraokePhrase {
                    notes: vec![note("one", 0.0, 0.25, 60)],
                },
                KaraokePhrase {
                    notes: vec![note("two", 0.25, 0.5, 62)],
                },
            ],
        };

        let error = export_ultrastar_v1(&chart, &metadata())
            .expect_err("contiguous phrase boundary must fail closed");
        assert!(error.to_string().contains("no safe phrase marker"));
    }

    #[test]
    fn rejects_unsafe_audio_file_references() {
        for audio_file in ["../song.ogg", "/tmp/song.ogg", "C:\\song.ogg", "https://example.test/song.ogg"] {
            let metadata = UltraStarV1Metadata::new("Example", "Artist", audio_file);
            assert!(metadata.validate().is_err(), "{audio_file} must be rejected");
        }
    }

    #[test]
    fn preserves_note_text_whitespace() {
        let chart = KaraokeChart {
            tempo_bpm: 120.0,
            beat_zero_seconds: 0.0,
            phrases: vec![KaraokePhrase {
                notes: vec![note(" leading and trailing ", 0.0, 0.25, 60)],
            }],
        };

        let exported = export_ultrastar_v1(&chart, &metadata()).expect("valid chart exports");
        assert!(exported.contains(": 0 2 0  leading and trailing \n"));
    }
}
