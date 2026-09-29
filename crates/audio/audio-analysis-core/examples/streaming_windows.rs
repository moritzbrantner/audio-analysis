use audio_analysis_core::{StreamingFrameBuffer, StreamingFrameConfig};
use audio_contracts::{AudioBuffer, AudioFrame, Timebase, Timestamp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut windows = StreamingFrameBuffer::new(StreamingFrameConfig::new(8, 4)?)?;
    let first = AudioBuffer::F32((0..6).map(|value| value as f32).collect());
    let second = AudioBuffer::F32((6..12).map(|value| value as f32).collect());

    let first_frame = AudioFrame::new(
        Timestamp::new(0, Timebase::new(1, 48_000)),
        48_000,
        1,
        &first,
    )?;
    let second_frame = AudioFrame::new(
        Timestamp::new(6, Timebase::new(1, 48_000)),
        48_000,
        1,
        &second,
    )?;

    let _ = windows.push_frame(&first_frame)?;
    let produced = windows.push_frame(&second_frame)?;

    println!("produced {} windows", produced.len());
    Ok(())
}
