#![doc = include_str!("../README.md")]

pub mod surface;
use audio_analysis_synthesis::{
    synthesize_timeline, AudioSynthesisConfig, ToneSegment, ToneSpec, Waveform,
};
use audio_contracts::{DetectError, OwnedAudioFrame, Result};
use data_inversion_core::{Generated, InversionMethod};

const DEFAULT_TICKS_PER_QUARTER: u16 = 480;
const DEFAULT_VELOCITY: u8 = 100;
const MAX_VARIABLE_LENGTH_QUANTITY: u32 = 0x0fff_ffff;

/// Rendering waveform used when a MIDI-like note sequence is synthesized to audio.
pub type RenderWaveform = Waveform;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Chromatic note names.
pub enum NoteName {
    /// C.
    C,
    /// C sharp / D flat.
    Cs,
    /// D.
    D,
    /// D sharp / E flat.
    Ds,
    /// E.
    E,
    /// F.
    F,
    /// F sharp / G flat.
    Fs,
    /// G.
    G,
    /// G sharp / A flat.
    Gs,
    /// A.
    A,
    /// A sharp / B flat.
    As,
    /// B.
    B,
}

impl NoteName {
    /// Returns this note's semitone offset from C.
    pub fn semitone(self) -> u8 {
        match self {
            NoteName::C => 0,
            NoteName::Cs => 1,
            NoteName::D => 2,
            NoteName::Ds => 3,
            NoteName::E => 4,
            NoteName::F => 5,
            NoteName::Fs => 6,
            NoteName::G => 7,
            NoteName::Gs => 8,
            NoteName::A => 9,
            NoteName::As => 10,
            NoteName::B => 11,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// MIDI note number in the inclusive range `0..=127`.
pub struct MidiNote(u8);

impl MidiNote {
    /// Creates a MIDI note from a raw note number.
    pub fn new(value: u8) -> Result<Self> {
        if value > 127 {
            return Err(invalid_argument("MIDI note must be in the range 0..=127"));
        }
        Ok(Self(value))
    }

    /// Creates a MIDI note from a chromatic note name and scientific pitch octave.
    ///
    /// `C4` is MIDI note 60 and `A4` is MIDI note 69.
    pub fn from_name(name: NoteName, octave: i8) -> Result<Self> {
        let value = (i16::from(octave) + 1) * 12 + i16::from(name.semitone());
        if !(0..=127).contains(&value) {
            return Err(invalid_argument(
                "named MIDI note must resolve to the range 0..=127",
            ));
        }
        Ok(Self(value as u8))
    }

    /// Returns the raw MIDI note number.
    pub fn value(self) -> u8 {
        self.0
    }

    /// Returns the equal-tempered frequency in hertz using A4 = 440 Hz.
    pub fn frequency_hz(self) -> f32 {
        440.0 * 2.0_f32.powf((self.0 as f32 - 69.0) / 12.0)
    }

    /// Creates the nearest MIDI note for an equal-tempered frequency in hertz.
    pub fn from_frequency_hz(frequency_hz: f32) -> Result<Self> {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(invalid_argument(
                "frequency_hz must be finite and greater than zero",
            ));
        }
        let value = (69.0 + 12.0 * (frequency_hz / 440.0).log2()).round();
        if !(0.0..=127.0).contains(&value) {
            return Err(invalid_argument(
                "frequency_hz resolves outside the MIDI note range",
            ));
        }
        Self::new(value as u8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// One pitch-track frame that can be quantized into MIDI-like note events.
pub struct PitchTrackFrame {
    /// Start time in seconds.
    pub start_seconds: f32,
    /// End time in seconds.
    pub end_seconds: f32,
    /// Estimated frequency in hertz.
    pub frequency_hz: f32,
    /// Confidence in the inclusive range `0.0..=1.0`.
    pub confidence: f32,
}

impl PitchTrackFrame {
    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if !self.start_seconds.is_finite()
            || !self.end_seconds.is_finite()
            || self.start_seconds < 0.0
            || self.end_seconds <= self.start_seconds
        {
            return Err(invalid_argument(
                "pitch frame start/end seconds must be finite, non-negative, and ordered",
            ));
        }
        MidiNote::from_frequency_hz(self.frequency_hz)?;
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(invalid_argument(
                "pitch frame confidence must be finite and between 0.0 and 1.0",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Options for converting pitch-track frames into MIDI-like notes.
pub struct PitchTrackMidiOptions {
    /// Tempo used to convert seconds into quarter-note beats.
    pub tempo_bpm: f32,
    /// Optional quantization grid in quarter-note beats.
    pub quantization_beats: Option<f32>,
    /// Minimum note duration in seconds.
    pub min_note_duration_seconds: f32,
    /// Fixed MIDI velocity.
    pub velocity: u8,
}

impl Default for PitchTrackMidiOptions {
    fn default() -> Self {
        Self {
            tempo_bpm: 120.0,
            quantization_beats: None,
            min_note_duration_seconds: 0.05,
            velocity: DEFAULT_VELOCITY,
        }
    }
}

impl PitchTrackMidiOptions {
    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if !self.tempo_bpm.is_finite() || self.tempo_bpm <= 0.0 {
            return Err(invalid_argument(
                "tempo_bpm must be finite and greater than zero",
            ));
        }
        if let Some(grid) = self.quantization_beats {
            if !grid.is_finite() || grid <= 0.0 {
                return Err(invalid_argument(
                    "quantization_beats must be finite and greater than zero",
                ));
            }
        }
        if !self.min_note_duration_seconds.is_finite() || self.min_note_duration_seconds < 0.0 {
            return Err(invalid_argument(
                "min_note_duration_seconds must be finite and non-negative",
            ));
        }
        if self.velocity > 127 {
            return Err(invalid_argument(
                "MIDI velocity must be in the range 0..=127",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Result of converting pitch-track frames into MIDI-like notes.
pub struct PitchTrackMidiResult {
    /// Merged note events.
    pub notes: Vec<MidiNoteEvent>,
    /// Diagnostics for dropped or merged frames.
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// A MIDI-like note event with beat-based timing.
pub struct MidiNoteEvent {
    /// Note pitch.
    pub note: MidiNote,
    /// Start position in quarter-note beats.
    pub start_beats: f32,
    /// Duration in quarter-note beats.
    pub duration_beats: f32,
    /// MIDI velocity in the inclusive range `0..=127`.
    pub velocity: u8,
    /// MIDI channel in the inclusive range `0..=15`.
    pub channel: u8,
}

impl MidiNoteEvent {
    /// Creates a note event.
    pub fn new(note: MidiNote, start_beats: f32, duration_beats: f32) -> Result<Self> {
        let event = Self {
            note,
            start_beats,
            duration_beats,
            velocity: DEFAULT_VELOCITY,
            channel: 0,
        };
        event.validate()?;
        Ok(event)
    }

    /// Sets MIDI velocity.
    pub fn velocity(mut self, velocity: u8) -> Result<Self> {
        self.velocity = velocity;
        self.validate()?;
        Ok(self)
    }

    /// Sets MIDI channel.
    pub fn channel(mut self, channel: u8) -> Result<Self> {
        self.channel = channel;
        self.validate()?;
        Ok(self)
    }

    /// Validates this value.
    pub fn validate(self) -> Result<()> {
        if !self.start_beats.is_finite() || self.start_beats < 0.0 {
            return Err(invalid_argument(
                "note start_beats must be finite and non-negative",
            ));
        }
        if !self.duration_beats.is_finite() || self.duration_beats <= 0.0 {
            return Err(invalid_argument(
                "note duration_beats must be finite and greater than zero",
            ));
        }
        if self.velocity > 127 {
            return Err(invalid_argument(
                "MIDI velocity must be in the range 0..=127",
            ));
        }
        if self.channel > 15 {
            return Err(invalid_argument("MIDI channel must be in the range 0..=15"));
        }
        Ok(())
    }

    fn start_tick(self, ticks_per_quarter: u16) -> Result<u32> {
        beats_to_ticks(self.start_beats, ticks_per_quarter)
    }

    fn end_tick(self, ticks_per_quarter: u16) -> Result<u32> {
        beats_to_ticks(self.start_beats + self.duration_beats, ticks_per_quarter)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// MIDI-like track with note events and rendering metadata.
pub struct MidiTrack {
    /// Track name.
    pub name: String,
    /// General MIDI program in the inclusive range `0..=127`.
    pub program: u8,
    /// Rendering waveform used by [`MidiSong::render`].
    pub waveform: RenderWaveform,
    notes: Vec<MidiNoteEvent>,
}

impl MidiTrack {
    /// Creates an empty track.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            program: 0,
            waveform: RenderWaveform::Sine,
            notes: Vec::new(),
        }
    }

    /// Sets the General MIDI program number.
    pub fn program(mut self, program: u8) -> Result<Self> {
        self.program = program;
        self.validate()?;
        Ok(self)
    }

    /// Sets the audio rendering waveform.
    pub fn waveform(mut self, waveform: RenderWaveform) -> Self {
        self.waveform = waveform;
        self
    }

    /// Adds a note event.
    pub fn push(&mut self, note: MidiNoteEvent) -> Result<()> {
        note.validate()?;
        self.notes.push(note);
        Ok(())
    }

    /// Returns track notes.
    pub fn notes(&self) -> &[MidiNoteEvent] {
        &self.notes
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if self.program > 127 {
            return Err(invalid_argument(
                "MIDI program must be in the range 0..=127",
            ));
        }
        for note in &self.notes {
            note.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// A MIDI-like song with beat-based timing.
pub struct MidiSong {
    /// Tempo in beats per minute.
    pub tempo_bpm: f32,
    /// Standard MIDI ticks per quarter note.
    pub ticks_per_quarter: u16,
    tracks: Vec<MidiTrack>,
}

impl MidiSong {
    /// Creates an empty song.
    pub fn new(tempo_bpm: f32) -> Result<Self> {
        let song = Self {
            tempo_bpm,
            ticks_per_quarter: DEFAULT_TICKS_PER_QUARTER,
            tracks: Vec::new(),
        };
        song.validate()?;
        Ok(song)
    }

    /// Sets ticks per quarter note.
    pub fn ticks_per_quarter(mut self, ticks_per_quarter: u16) -> Result<Self> {
        self.ticks_per_quarter = ticks_per_quarter;
        self.validate()?;
        Ok(self)
    }

    /// Adds a track and returns the song.
    pub fn with_track(mut self, track: MidiTrack) -> Result<Self> {
        self.push_track(track)?;
        Ok(self)
    }

    /// Adds a track.
    pub fn push_track(&mut self, track: MidiTrack) -> Result<()> {
        track.validate()?;
        self.tracks.push(track);
        Ok(())
    }

    /// Returns song tracks.
    pub fn tracks(&self) -> &[MidiTrack] {
        &self.tracks
    }

    /// Converts this song into a Standard MIDI File byte stream.
    pub fn to_midi_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let track_count = self
            .tracks
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid_argument("too many MIDI tracks"))?;
        if track_count > u16::MAX as usize {
            return Err(invalid_argument("too many MIDI tracks"));
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        write_u32_be(&mut bytes, 6);
        write_u16_be(&mut bytes, 1);
        write_u16_be(&mut bytes, track_count as u16);
        write_u16_be(&mut bytes, self.ticks_per_quarter);
        write_track_chunk(&mut bytes, &tempo_track_bytes(self.tempo_bpm)?);
        for track in &self.tracks {
            write_track_chunk(
                &mut bytes,
                &note_track_bytes(track, self.ticks_per_quarter)?,
            );
        }
        Ok(bytes)
    }

    /// Renders all note events into an owned audio frame.
    pub fn render(&self, config: AudioSynthesisConfig) -> Result<Generated<OwnedAudioFrame>> {
        self.validate()?;
        let mut segments = Vec::new();
        for track in &self.tracks {
            for note in track.notes() {
                segments.push(ToneSegment {
                    start_seconds: self.beats_to_seconds(note.start_beats),
                    tone: ToneSpec {
                        frequency_hz: note.note.frequency_hz(),
                        duration_seconds: self.beats_to_seconds(note.duration_beats),
                        amplitude: f32::from(note.velocity) / 127.0,
                        waveform: track.waveform,
                    },
                });
            }
        }
        if segments.is_empty() {
            return Err(invalid_argument(
                "song must contain at least one note to render",
            ));
        }
        let mut generated = synthesize_timeline(&segments, config)?;
        generated.trace.source_type = "midi_like_song".to_string();
        generated.trace = generated.trace.note(
            "notes",
            InversionMethod::Template,
            "MIDI-like note events are rendered as analytic waveform segments",
        );
        Ok(generated)
    }

    /// Converts beats to seconds using this song tempo.
    pub fn beats_to_seconds(&self, beats: f32) -> f32 {
        beats * 60.0 / self.tempo_bpm
    }

    /// Validates this value.
    pub fn validate(&self) -> Result<()> {
        if !self.tempo_bpm.is_finite() || self.tempo_bpm <= 0.0 {
            return Err(invalid_argument(
                "tempo_bpm must be finite and greater than zero",
            ));
        }
        if self.ticks_per_quarter == 0 {
            return Err(invalid_argument(
                "ticks_per_quarter must be greater than zero",
            ));
        }
        for track in &self.tracks {
            track.validate()?;
        }
        Ok(())
    }
}

/// Converts pitch-track frames into merged MIDI-like note events.
pub fn pitch_track_to_midi_notes(
    frames: &[PitchTrackFrame],
    options: PitchTrackMidiOptions,
) -> Result<PitchTrackMidiResult> {
    options.validate()?;
    let mut diagnostics = Vec::new();
    let mut notes = Vec::new();
    let mut current: Option<(MidiNote, f32, f32)> = None;
    let mut previous_start = None;

    for frame in frames {
        frame.validate()?;
        if let Some(previous) = previous_start {
            if frame.start_seconds < previous {
                return Err(invalid_argument(
                    "pitch frames must be ordered by start_seconds",
                ));
            }
        }
        previous_start = Some(frame.start_seconds);
        let note = MidiNote::from_frequency_hz(frame.frequency_hz)?;
        match &mut current {
            Some((current_note, _start, end))
                if *current_note == note && frame.start_seconds <= *end + 1.0e-4 =>
            {
                *end = (*end).max(frame.end_seconds);
                diagnostics.push(format!(
                    "merged pitch frame {:.3}-{:.3}s into MIDI note {}",
                    frame.start_seconds,
                    frame.end_seconds,
                    note.value()
                ));
            }
            Some((current_note, start, end)) => {
                push_pitch_note(
                    &mut notes,
                    &mut diagnostics,
                    *current_note,
                    *start,
                    *end,
                    options,
                )?;
                current = Some((note, frame.start_seconds, frame.end_seconds));
            }
            None => current = Some((note, frame.start_seconds, frame.end_seconds)),
        }
    }

    if let Some((note, start, end)) = current {
        push_pitch_note(&mut notes, &mut diagnostics, note, start, end, options)?;
    }
    Ok(PitchTrackMidiResult { notes, diagnostics })
}

fn push_pitch_note(
    notes: &mut Vec<MidiNoteEvent>,
    diagnostics: &mut Vec<String>,
    note: MidiNote,
    start_seconds: f32,
    end_seconds: f32,
    options: PitchTrackMidiOptions,
) -> Result<()> {
    let duration_seconds = end_seconds - start_seconds;
    if duration_seconds < options.min_note_duration_seconds {
        diagnostics.push(format!(
            "dropped MIDI note {} shorter than {:.3}s",
            note.value(),
            options.min_note_duration_seconds
        ));
        return Ok(());
    }
    let mut start_beats = start_seconds * options.tempo_bpm / 60.0;
    let mut duration_beats = duration_seconds * options.tempo_bpm / 60.0;
    if let Some(grid) = options.quantization_beats {
        start_beats = (start_beats / grid).round() * grid;
        duration_beats = ((duration_beats / grid).round() * grid).max(grid);
    }
    notes.push(MidiNoteEvent::new(note, start_beats, duration_beats)?.velocity(options.velocity)?);
    Ok(())
}

fn tempo_track_bytes(tempo_bpm: f32) -> Result<Vec<u8>> {
    let micros_per_quarter = (60_000_000.0 / tempo_bpm).round();
    if !micros_per_quarter.is_finite() || !(1.0..=16_777_215.0).contains(&micros_per_quarter) {
        return Err(invalid_argument(
            "tempo_bpm cannot be represented as a MIDI tempo event",
        ));
    }
    let micros_per_quarter = micros_per_quarter as u32;
    let mut bytes = Vec::new();
    bytes.push(0);
    bytes.extend_from_slice(&[0xff, 0x51, 0x03]);
    bytes.push(((micros_per_quarter >> 16) & 0xff) as u8);
    bytes.push(((micros_per_quarter >> 8) & 0xff) as u8);
    bytes.push((micros_per_quarter & 0xff) as u8);
    bytes.push(0);
    bytes.extend_from_slice(&[0xff, 0x2f, 0x00]);
    Ok(bytes)
}

fn note_track_bytes(track: &MidiTrack, ticks_per_quarter: u16) -> Result<Vec<u8>> {
    let mut events = Vec::<MidiEventBytes>::new();
    events.push(MidiEventBytes {
        tick: 0,
        order: 0,
        bytes: {
            let name_len = u32::try_from(track.name.len())
                .map_err(|_| invalid_argument("MIDI track name is too large"))?;
            if name_len > MAX_VARIABLE_LENGTH_QUANTITY {
                return Err(invalid_argument("MIDI track name is too large"));
            }
            let mut bytes = vec![0xff, 0x03];
            push_variable_length_quantity(&mut bytes, name_len);
            bytes.extend_from_slice(track.name.as_bytes());
            bytes
        },
    });
    events.push(MidiEventBytes {
        tick: 0,
        order: 1,
        bytes: vec![0xc0, track.program],
    });

    for note in track.notes() {
        let start_tick = note.start_tick(ticks_per_quarter)?;
        let end_tick = note.end_tick(ticks_per_quarter)?;
        events.push(MidiEventBytes {
            tick: start_tick,
            order: 2,
            bytes: vec![0x90 | note.channel, note.note.value(), note.velocity],
        });
        events.push(MidiEventBytes {
            tick: end_tick,
            order: 1,
            bytes: vec![0x80 | note.channel, note.note.value(), 0],
        });
    }
    events.sort_by_key(|event| (event.tick, event.order));

    let mut bytes = Vec::new();
    let mut last_tick = 0_u32;
    for event in events {
        let delta = event
            .tick
            .checked_sub(last_tick)
            .ok_or_else(|| invalid_argument("MIDI events are not sorted"))?;
        push_variable_length_quantity(&mut bytes, delta);
        bytes.extend_from_slice(&event.bytes);
        last_tick = event.tick;
    }
    bytes.push(0);
    bytes.extend_from_slice(&[0xff, 0x2f, 0x00]);
    Ok(bytes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MidiEventBytes {
    tick: u32,
    order: u8,
    bytes: Vec<u8>,
}

fn beats_to_ticks(beats: f32, ticks_per_quarter: u16) -> Result<u32> {
    if !beats.is_finite() || beats < 0.0 {
        return Err(invalid_argument("beats must be finite and non-negative"));
    }
    let ticks = beats * f32::from(ticks_per_quarter);
    if !ticks.is_finite() || ticks > MAX_VARIABLE_LENGTH_QUANTITY as f32 {
        return Err(invalid_argument(
            "beat position is too large for MIDI ticks",
        ));
    }
    Ok(ticks.round() as u32)
}

fn write_track_chunk(bytes: &mut Vec<u8>, track: &[u8]) {
    bytes.extend_from_slice(b"MTrk");
    write_u32_be(bytes, track.len() as u32);
    bytes.extend_from_slice(track);
}

fn write_u16_be(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn write_u32_be(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn push_variable_length_quantity(bytes: &mut Vec<u8>, value: u32) {
    let mut buffer = [0_u8; 5];
    let mut index = 4;
    buffer[index] = (value & 0x7f) as u8;
    let mut value = value >> 7;
    while value > 0 {
        index -= 1;
        buffer[index] = ((value & 0x7f) as u8) | 0x80;
        value >>= 7;
    }
    bytes.extend_from_slice(&buffer[index..=4]);
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_contracts::AudioBuffer;

    #[test]
    fn converts_named_notes_to_midi_and_frequency() {
        let a4 = MidiNote::from_name(NoteName::A, 4).unwrap();
        assert_eq!(a4.value(), 69);
        assert!((a4.frequency_hz() - 440.0).abs() < f32::EPSILON);

        let c4 = MidiNote::from_name(NoteName::C, 4).unwrap();
        assert_eq!(c4.value(), 60);
    }

    #[test]
    fn exports_standard_midi_bytes() {
        let mut track = MidiTrack::new("lead");
        track
            .push(
                MidiNoteEvent::new(MidiNote::from_name(NoteName::C, 4).unwrap(), 0.0, 1.0).unwrap(),
            )
            .unwrap();
        let song = MidiSong::new(120.0).unwrap().with_track(track).unwrap();
        let bytes = song.to_midi_bytes().unwrap();
        assert_eq!(&bytes[0..4], b"MThd");
        assert!(bytes.windows(4).any(|window| window == b"MTrk"));
        assert!(bytes.windows(3).any(|window| window == [0xff, 0x51, 0x03]));
    }

    #[test]
    fn renders_note_sequence_to_audio_frame() {
        let mut track = MidiTrack::new("bass").waveform(RenderWaveform::Square);
        track
            .push(
                MidiNoteEvent::new(MidiNote::from_name(NoteName::C, 3).unwrap(), 0.0, 0.5)
                    .unwrap()
                    .velocity(80)
                    .unwrap(),
            )
            .unwrap();
        let song = MidiSong::new(120.0).unwrap().with_track(track).unwrap();
        let generated = song
            .render(AudioSynthesisConfig::new(1_000, 1).unwrap())
            .unwrap();
        assert_eq!(generated.value.samples_per_channel(), 250);
        let AudioBuffer::F32(samples) = generated.value.data else {
            panic!("expected f32 samples");
        };
        assert!(samples.iter().any(|sample| *sample != 0.0));
    }

    #[test]
    fn converts_pitch_track_to_single_a4_note() {
        let result = pitch_track_to_midi_notes(
            &[
                PitchTrackFrame {
                    start_seconds: 0.0,
                    end_seconds: 0.5,
                    frequency_hz: 440.0,
                    confidence: 0.9,
                },
                PitchTrackFrame {
                    start_seconds: 0.5,
                    end_seconds: 1.0,
                    frequency_hz: 440.0,
                    confidence: 0.9,
                },
            ],
            PitchTrackMidiOptions::default(),
        )
        .unwrap();

        assert_eq!(result.notes.len(), 1);
        assert_eq!(result.notes[0].note.value(), 69);
        assert_eq!(result.notes[0].start_beats, 0.0);
        assert_eq!(result.notes[0].duration_beats, 2.0);
    }

    #[test]
    fn pitch_track_drops_notes_below_minimum_duration() {
        let result = pitch_track_to_midi_notes(
            &[PitchTrackFrame {
                start_seconds: 0.0,
                end_seconds: 0.01,
                frequency_hz: 440.0,
                confidence: 0.9,
            }],
            PitchTrackMidiOptions {
                min_note_duration_seconds: 0.05,
                ..PitchTrackMidiOptions::default()
            },
        )
        .unwrap();

        assert!(result.notes.is_empty());
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.contains("dropped")));
    }

    #[test]
    fn rejects_invalid_timing() {
        assert!(MidiNoteEvent::new(MidiNote::new(60).unwrap(), 0.0, 0.0).is_err());
        assert!(MidiSong::new(0.0).is_err());
    }
}
