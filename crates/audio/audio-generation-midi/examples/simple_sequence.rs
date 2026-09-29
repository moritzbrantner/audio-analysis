use audio_generation_midi::{
    MidiNote, MidiNoteEvent, MidiSong, MidiTrack, NoteName, RenderWaveform,
};

fn main() -> audio_contracts::Result<()> {
    let mut lead = MidiTrack::new("lead").waveform(RenderWaveform::Triangle);
    for (index, note) in [
        NoteName::C,
        NoteName::E,
        NoteName::G,
        NoteName::C,
        NoteName::G,
        NoteName::E,
    ]
    .into_iter()
    .enumerate()
    {
        lead.push(
            MidiNoteEvent::new(MidiNote::from_name(note, 4)?, index as f32 * 0.5, 0.45)?
                .velocity(96)?,
        )?;
    }

    let song = MidiSong::new(112.0)?.with_track(lead)?;
    let midi_bytes = song.to_midi_bytes()?;
    let audio = song.render(Default::default())?;

    println!(
        "generated {} MIDI bytes and {} samples per channel",
        midi_bytes.len(),
        audio.value.samples_per_channel()
    );
    Ok(())
}
