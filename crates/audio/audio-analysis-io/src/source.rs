use audio_contracts::{AudioSampleFormat, OwnedAudioFrame, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Whether a source is finite/recorded or live.
pub enum SourceMode {
    /// A finite recorded source.
    Recorded,
    /// A live source.
    Live,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Audio stream metadata independent of any visual capability.
pub struct AudioStreamInfo {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Sample representation.
    pub sample_format: AudioSampleFormat,
}

#[derive(Debug, Clone, PartialEq)]
/// Generic source metadata retained for compatibility with extracted callers.
pub struct MediaSourceInfo {
    /// User-facing input identifier.
    pub input: String,
    /// Source mode.
    pub mode: SourceMode,
    /// Reserved non-audio stream slot; audio repositories do not interpret it.
    pub video: Option<()>,
    /// Audio streams in source order.
    pub audio: Vec<AudioStreamInfo>,
    /// Reserved non-audio stream slots; audio repositories do not interpret them.
    pub text: Vec<()>,
}

impl MediaSourceInfo {
    /// Creates finite source metadata.
    pub fn recorded(input: impl Into<String>) -> Self {
        Self { input: input.into(), mode: SourceMode::Recorded, video: None, audio: Vec::new(), text: Vec::new() }
    }

    /// Creates live source metadata.
    pub fn live(input: impl Into<String>) -> Self {
        Self { input: input.into(), mode: SourceMode::Live, video: None, audio: Vec::new(), text: Vec::new() }
    }

    /// Appends an audio stream.
    pub fn with_audio(mut self, audio: AudioStreamInfo) -> Self {
        self.audio.push(audio);
        self
    }
}

/// Pull-based audio frame source independent of visual ingest.
pub trait AudioFrameSource {
    /// Returns source metadata.
    fn source_info(&self) -> &MediaSourceInfo;
    /// Returns the next frame, or `None` at end of stream.
    fn next_audio_frame(&mut self) -> Result<Option<OwnedAudioFrame>>;

    /// Returns whether this source is live.
    fn is_live(&self) -> bool {
        self.source_info().mode == SourceMode::Live
    }
}

/// Feeds all frames from a source into an audio pipeline.
pub fn analyze_audio_source<S, F>(
    source: &mut S,
    pipeline: &mut audio_contracts::AudioPipeline,
    mut on_frame: F,
) -> Result<audio_contracts::AudioAnalysisResult>
where
    S: AudioFrameSource,
    F: FnMut(&audio_contracts::AudioAnalysis) -> Result<()>,
{
    pipeline.reset();
    while let Some(frame) = source.next_audio_frame()? {
        let analysis = pipeline.process_frame(frame)?;
        on_frame(&analysis)?;
    }
    pipeline.finish_analysis()
}
