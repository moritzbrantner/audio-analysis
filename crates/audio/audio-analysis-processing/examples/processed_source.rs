use audio_analysis_processing::{AudioProcessor, ProcessedAudioSource};
use audio_contracts::{AudioBuffer, OwnedAudioFrame, Timebase, Timestamp};
use video_analysis_ingest::{AudioFrameSource, AudioStreamInfo, MediaSourceInfo, SourceMode};

#[derive(Clone)]
struct SingleFrameSource {
    frame: Option<OwnedAudioFrame>,
    info: MediaSourceInfo,
}

impl SingleFrameSource {
    fn new(frame: OwnedAudioFrame) -> Self {
        Self {
            info: MediaSourceInfo {
                input: "synthetic".to_string(),
                mode: SourceMode::Recorded,
                video: None,
                audio: vec![AudioStreamInfo {
                    sample_rate: frame.sample_rate,
                    channels: frame.channels,
                    sample_format: frame.sample_format(),
                }],
                text: Vec::new(),
            },
            frame: Some(frame),
        }
    }
}

impl AudioFrameSource for SingleFrameSource {
    fn source_info(&self) -> &MediaSourceInfo {
        &self.info
    }

    fn next_audio_frame(&mut self) -> audio_contracts::Result<Option<OwnedAudioFrame>> {
        Ok(self.frame.take())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frame = OwnedAudioFrame::new(
        Timestamp::new(0, Timebase::new(1, 8_000)),
        8_000,
        1,
        AudioBuffer::F32(vec![0.25; 16]),
    )?;
    let source = SingleFrameSource::new(frame);
    let processor = AudioProcessor::new().gain(0.5).hard_clip(-0.1, 0.1);
    let mut processed = ProcessedAudioSource::new(source, processor);

    let frame = processed.next_audio_frame()?.expect("processed frame");
    println!("samples_per_channel={}", frame.samples_per_channel());
    Ok(())
}
