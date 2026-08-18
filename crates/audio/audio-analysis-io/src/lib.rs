#![doc = include_str!("../README.md")]

mod editing;
pub mod surface;
pub use editing::*;
use std::path::{Path, PathBuf};

pub use audio_analysis_core::ChannelMix;
use audio_analysis_core::{interleaved_to_mono, AudioClip, OwnedAudioWaveformBatch};
use audio_contracts::{OwnedAudioFrame, Result, Timebase, Timestamp};
/// Re-exports the video analysis FFmpeg API.
pub use video_analysis_ffmpeg::{
    probe_audio as probe_audio_file, probe_audio_input, AudioMetadata, FfmpegAudioSource,
    FfmpegAudioSourceOptions, FfmpegError,
};
/// Re-exports the video analysis ingest audio frame source API.
pub use video_analysis_ingest::{AudioFrameSource, AudioStreamInfo, SourceMode};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Variants describing audio input.
pub enum AudioInput {
    /// The file variant.
    File(PathBuf),
    /// The input variant.
    Input(String),
    /// The live variant.
    Live(String),
}

/// A finite media path with an optional zero-based audio-stream selection.
///
/// This additive source keeps [`AudioInput::File`] unchanged for existing
/// callers while allowing container-aware callers to select one audio stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedMediaSource {
    path: PathBuf,
    audio_stream_index: Option<usize>,
}

impl SelectedMediaSource {
    /// Creates a media source that preserves FFmpeg's existing default stream behavior.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            audio_stream_index: None,
        }
    }

    /// Selects a zero-based audio-stream ordinal.
    pub fn audio_stream_index(mut self, index: usize) -> Self {
        self.audio_stream_index = Some(index);
        self
    }

    /// Returns the media path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the selected zero-based audio-stream ordinal, when present.
    pub fn selected_audio_stream_index(&self) -> Option<usize> {
        self.audio_stream_index
    }
}

/// Error returned by checked selected-media decode operations.
#[derive(Debug)]
pub enum AudioIoError {
    /// FFprobe, FFmpeg startup, or typed stream-selection failure.
    Ffmpeg(FfmpegError),
    /// Failure while consuming decoded audio frames.
    Decode(audio_contracts::DetectError),
}

impl std::fmt::Display for AudioIoError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ffmpeg(error) => error.fmt(formatter),
            Self::Decode(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AudioIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Ffmpeg(error) => Some(error),
            Self::Decode(error) => Some(error),
        }
    }
}

impl From<FfmpegError> for AudioIoError {
    fn from(error: FfmpegError) -> Self {
        Self::Ffmpeg(error)
    }
}

impl From<audio_contracts::DetectError> for AudioIoError {
    fn from(error: audio_contracts::DetectError) -> Self {
        Self::Decode(error)
    }
}

/// Result returned by checked selected-media decode operations.
pub type AudioIoResult<T> = std::result::Result<T, AudioIoError>;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Data type for audio input options.
pub struct AudioInputOptions {
    /// The samples per chunk value.
    pub samples_per_chunk: usize,
    /// The realtime value.
    pub realtime: bool,
    /// The extra input args value.
    pub extra_input_args: Vec<String>,
}

impl AudioInputOptions {
    /// Returns recorded.
    pub fn recorded() -> Self {
        Self {
            samples_per_chunk: 16_384,
            realtime: false,
            extra_input_args: Vec::new(),
        }
    }

    /// Returns live.
    pub fn live() -> Self {
        Self {
            samples_per_chunk: 1024,
            realtime: true,
            extra_input_args: Vec::new(),
        }
    }

    /// Returns samples per chunk.
    pub fn samples_per_chunk(mut self, samples: usize) -> Self {
        self.samples_per_chunk = samples.max(1);
        self
    }

    /// Returns realtime.
    pub fn realtime(mut self, realtime: bool) -> Self {
        self.realtime = realtime;
        self
    }

    /// Returns extra input arg.
    pub fn extra_input_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_input_args.push(arg.into());
        self
    }

    /// Consumes this value into a FFmpeg options.
    pub fn into_ffmpeg_options(self, mode: SourceMode) -> FfmpegAudioSourceOptions {
        let mut options = if mode == SourceMode::Live || self.realtime {
            FfmpegAudioSourceOptions::live()
        } else {
            FfmpegAudioSourceOptions::recorded()
        }
        .samples_per_chunk(self.samples_per_chunk);
        for arg in self.extra_input_args {
            options = options.extra_input_arg(arg);
        }
        options
    }
}

impl Default for AudioInputOptions {
    fn default() -> Self {
        Self::recorded()
    }
}

/// Returns open audio input.
pub fn open_audio_input(
    input: AudioInput,
    options: AudioInputOptions,
) -> Result<FfmpegAudioSource> {
    match input {
        AudioInput::File(path) => {
            let options = options.into_ffmpeg_options(SourceMode::Recorded);
            FfmpegAudioSource::open_path_with_options(path, options)
        }
        AudioInput::Input(input) => {
            let options = options.into_ffmpeg_options(SourceMode::Recorded);
            FfmpegAudioSource::open_input_with_options(input, options)
        }
        AudioInput::Live(input) => {
            let options = options.into_ffmpeg_options(SourceMode::Live);
            FfmpegAudioSource::open_input_with_options(input, options)
        }
    }
}

/// Opens a finite selected-media source while preserving typed FFmpeg errors.
pub fn open_selected_media_source(
    source: SelectedMediaSource,
    options: AudioInputOptions,
) -> std::result::Result<FfmpegAudioSource, FfmpegError> {
    let mut options = options.into_ffmpeg_options(SourceMode::Recorded);
    if let Some(index) = source.audio_stream_index {
        options = options.audio_stream_index(index);
    }
    FfmpegAudioSource::open_path_with_options_checked(source.path, options)
}

/// Decodes a finite selected-media source to interleaved f32 frames.
pub fn decode_selected_media_to_f32(
    source: SelectedMediaSource,
    options: AudioInputOptions,
) -> AudioIoResult<(AudioMetadata, Vec<OwnedAudioFrame>)> {
    let mut source = open_selected_media_source(source, options)?;
    let metadata = source.metadata().clone();
    let mut frames = Vec::new();
    while let Some(frame) = source.next_audio_frame()? {
        frames.push(frame);
    }
    Ok((metadata, frames))
}

/// Decodes a finite selected-media source to mono f32 samples.
pub fn decode_selected_media_to_mono_f32(
    source: SelectedMediaSource,
    options: AudioInputOptions,
    mix: ChannelMix,
) -> AudioIoResult<(AudioMetadata, Vec<f32>)> {
    let (metadata, frames) = decode_selected_media_to_f32(source, options)?;
    let mut mono = Vec::new();
    for frame in frames {
        mono.extend(interleaved_to_mono(&frame.data, frame.channels, mix)?);
    }
    Ok((metadata, mono))
}

/// Returns probe audio input metadata.
pub fn probe_audio_input_metadata(input: &AudioInput) -> Result<AudioMetadata> {
    match input {
        AudioInput::File(path) => probe_audio_file(path).map_err(|err| {
            audio_contracts::DetectError::Source(format!(
                "failed to probe audio file `{}`: {err}",
                path.display()
            ))
        }),
        AudioInput::Input(input) | AudioInput::Live(input) => {
            probe_audio_input(input).map_err(|err| {
                audio_contracts::DetectError::Source(format!(
                    "failed to probe audio input `{input}`: {err}"
                ))
            })
        }
    }
}

/// Returns open audio input with metadata.
pub fn open_audio_input_with_metadata(
    input: AudioInput,
    options: AudioInputOptions,
) -> Result<(AudioMetadata, FfmpegAudioSource)> {
    let metadata = probe_audio_input_metadata(&input)?;
    let source = open_audio_input(input, options)?;
    Ok((metadata, source))
}

/// Returns decode audio to f32.
pub fn decode_audio_to_f32(
    input: AudioInput,
    options: AudioInputOptions,
) -> Result<(AudioMetadata, Vec<OwnedAudioFrame>)> {
    let (metadata, mut source) = open_audio_input_with_metadata(input, options)?;
    let mut frames = Vec::new();
    while let Some(frame) = source.next_audio_frame()? {
        frames.push(frame);
    }
    Ok((metadata, frames))
}

/// Returns decode audio to waveform batch.
pub fn decode_audio_to_waveform_batch(
    input: AudioInput,
    options: AudioInputOptions,
) -> Result<(AudioMetadata, OwnedAudioWaveformBatch)> {
    let (metadata, frames) = decode_audio_to_f32(input, options)?;
    let batch = OwnedAudioWaveformBatch::from_audio_frames(&frames)?;
    Ok((metadata, batch))
}

/// Returns decode audio to mono f32.
pub fn decode_audio_to_mono_f32(
    input: AudioInput,
    options: AudioInputOptions,
    mix: ChannelMix,
) -> Result<(AudioMetadata, Vec<f32>)> {
    let (metadata, frames) = decode_audio_to_f32(input, options)?;
    let mut mono = Vec::new();
    for frame in frames {
        let samples = interleaved_to_mono(&frame.data, frame.channels, mix)?;
        mono.extend(samples);
    }
    Ok((metadata, mono))
}

/// Returns decode audio to a whole-buffer clip.
pub fn decode_audio_to_clip(
    input: AudioInput,
    options: AudioInputOptions,
) -> Result<(AudioMetadata, AudioClip)> {
    let (metadata, frames) = decode_audio_to_f32(input, options)?;
    let clip = AudioClip::from_frames(&frames)?;
    Ok((metadata, clip))
}

/// Reads a WAV file into a whole-buffer interleaved f32 clip without invoking FFmpeg.
pub fn read_wav_as_clip(path: impl AsRef<Path>) -> Result<AudioClip> {
    let path = path.as_ref();
    let mut reader = hound::WavReader::open(path).map_err(|err| {
        audio_contracts::DetectError::Source(format!(
            "failed to open WAV `{}`: {err}",
            path.display()
        ))
    })?;
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        return Err(audio_contracts::DetectError::InvalidArgument(
            "WAV sample rate must be positive".to_string(),
        ));
    }
    if spec.channels == 0 {
        return Err(audio_contracts::DetectError::InvalidArgument(
            "WAV channel count must be positive".to_string(),
        ));
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                return Err(audio_contracts::DetectError::InvalidArgument(format!(
                    "unsupported float WAV bit depth {}; expected 32",
                    spec.bits_per_sample
                )));
            }
            reader
                .samples::<f32>()
                .map(|sample| {
                    sample.map_err(|err| {
                        audio_contracts::DetectError::Source(format!(
                            "failed to read WAV sample `{}`: {err}",
                            path.display()
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        }
        hound::SampleFormat::Int => read_pcm_wav_samples(&mut reader, spec.bits_per_sample, path)?,
    };
    AudioClip::new(spec.sample_rate, spec.channels, samples)
}

/// Reads a WAV file into a single-item waveform batch without invoking FFmpeg.
pub fn read_wav_as_waveform_batch(path: impl AsRef<Path>) -> Result<OwnedAudioWaveformBatch> {
    let clip = read_wav_as_clip(path)?;
    let frame = clip.to_frame(Timestamp::new(0, Timebase::new(1, clip.sample_rate as i32)))?;
    OwnedAudioWaveformBatch::from_audio_frames(&[frame])
}

fn read_pcm_wav_samples(
    reader: &mut hound::WavReader<std::io::BufReader<std::fs::File>>,
    bits_per_sample: u16,
    path: &Path,
) -> Result<Vec<f32>> {
    match bits_per_sample {
        1..=8 => reader
            .samples::<i8>()
            .map(|sample| {
                sample.map(|sample| sample as f32 / 128.0).map_err(|err| {
                    audio_contracts::DetectError::Source(format!(
                        "failed to read WAV sample `{}`: {err}",
                        path.display()
                    ))
                })
            })
            .collect(),
        9..=16 => reader
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|sample| sample as f32 / 32_768.0)
                    .map_err(|err| {
                        audio_contracts::DetectError::Source(format!(
                            "failed to read WAV sample `{}`: {err}",
                            path.display()
                        ))
                    })
            })
            .collect(),
        17..=32 => {
            let scale = 2_f32.powi(bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample.map(|sample| sample as f32 / scale).map_err(|err| {
                        audio_contracts::DetectError::Source(format!(
                            "failed to read WAV sample `{}`: {err}",
                            path.display()
                        ))
                    })
                })
                .collect()
        }
        _ => Err(audio_contracts::DetectError::InvalidArgument(format!(
            "unsupported PCM WAV bit depth {bits_per_sample}; expected 1 through 32"
        ))),
    }
}

/// Writes waveform batch as wav.
pub fn write_waveform_batch_as_wav(
    path: impl AsRef<Path>,
    batch: &OwnedAudioWaveformBatch,
) -> Result<()> {
    let view = batch.as_view()?;
    if view.batch_size() != 1 {
        return Err(audio_contracts::DetectError::InvalidArgument(
            "waveform WAV export requires a batch size of 1".to_string(),
        ));
    }
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: view.channel_count() as u16,
        sample_rate: view.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|err| {
        audio_contracts::DetectError::Source(format!(
            "failed to create WAV `{}`: {err}",
            path.display()
        ))
    })?;
    for time_index in 0..view.time_steps() {
        for channel_index in 0..view.channel_count() {
            let sample = view.waveform(0, channel_index)?[time_index];
            writer.write_sample(sample).map_err(|err| {
                audio_contracts::DetectError::Source(format!(
                    "failed to write WAV sample `{}`: {err}",
                    path.display()
                ))
            })?;
        }
    }
    writer.finalize().map_err(|err| {
        audio_contracts::DetectError::Source(format!(
            "failed to finalize WAV `{}`: {err}",
            path.display()
        ))
    })?;
    Ok(())
}

/// Writes an audio clip as a 32-bit float WAV file.
pub fn write_clip_as_wav(path: impl AsRef<Path>, clip: &AudioClip) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: clip.channels,
        sample_rate: clip.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|err| {
        audio_contracts::DetectError::Source(format!(
            "failed to create WAV `{}`: {err}",
            path.display()
        ))
    })?;
    for sample in &clip.samples {
        writer.write_sample(*sample).map_err(|err| {
            audio_contracts::DetectError::Source(format!(
                "failed to write WAV sample `{}`: {err}",
                path.display()
            ))
        })?;
    }
    writer.finalize().map_err(|err| {
        audio_contracts::DetectError::Source(format!(
            "failed to finalize WAV `{}`: {err}",
            path.display()
        ))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_contracts::{AudioBuffer, Timebase, Timestamp};
    use tempfile::tempdir;

    #[test]
    fn recorded_options_map_to_ffmpeg_recorded_mode() {
        let options = AudioInputOptions::recorded()
            .samples_per_chunk(4096)
            .extra_input_arg("-safe")
            .extra_input_arg("0")
            .into_ffmpeg_options(SourceMode::Recorded);

        assert_eq!(options.mode, SourceMode::Recorded);
        assert!(!options.realtime);
        assert_eq!(options.samples_per_chunk, 4096);
        assert_eq!(options.extra_input_args, vec!["-safe", "0"]);
    }

    #[test]
    fn live_options_map_to_ffmpeg_live_mode() {
        let options = AudioInputOptions::live().into_ffmpeg_options(SourceMode::Live);

        assert_eq!(options.mode, SourceMode::Live);
        assert!(options.realtime);
        assert_eq!(options.samples_per_chunk, 1024);
    }

    #[test]
    fn chunk_size_defaults_differ_for_live_and_recorded() {
        assert_eq!(AudioInputOptions::recorded().samples_per_chunk, 16_384);
        assert_eq!(AudioInputOptions::live().samples_per_chunk, 1024);
    }

    #[test]
    fn realtime_recorded_input_uses_live_ffmpeg_options() {
        let options = AudioInputOptions::recorded()
            .realtime(true)
            .into_ffmpeg_options(SourceMode::Recorded);

        assert_eq!(options.mode, SourceMode::Live);
        assert!(options.realtime);
    }

    #[test]
    fn samples_per_chunk_is_clamped_to_at_least_one() {
        assert_eq!(
            AudioInputOptions::recorded()
                .samples_per_chunk(0)
                .samples_per_chunk,
            1
        );
    }

    #[test]
    fn input_variants_choose_recorded_or_live_modes() {
        let cases = [
            (AudioInput::File("audio.wav".into()), SourceMode::Recorded),
            (
                AudioInput::Input("pipe:0".to_string()),
                SourceMode::Recorded,
            ),
            (
                AudioInput::Live("rtsp://example.test/live".to_string()),
                SourceMode::Live,
            ),
        ];
        for (input, expected_mode) in cases {
            let requested_mode = match input {
                AudioInput::File(_) | AudioInput::Input(_) => SourceMode::Recorded,
                AudioInput::Live(_) => SourceMode::Live,
            };
            let options = AudioInputOptions::default().into_ffmpeg_options(requested_mode);
            assert_eq!(options.mode, expected_mode);
        }
    }

    #[test]
    fn metadata_probe_errors_include_input_context() {
        let error = probe_audio_input_metadata(&AudioInput::File("missing.wav".into()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing.wav"));
    }

    #[test]
    fn writes_single_item_waveform_batches_to_wav() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("waveform.wav");
        let frame = OwnedAudioFrame::new(
            Timestamp::new(0, Timebase::new(1, 16_000)),
            16_000,
            2,
            AudioBuffer::F32(vec![0.0, 0.5, -0.25, 0.25]),
        )
        .unwrap();
        let batch = OwnedAudioWaveformBatch::from_audio_frames(&[frame]).unwrap();

        write_waveform_batch_as_wav(&path, &batch).unwrap();

        let mut reader = hound::WavReader::open(&path).unwrap();
        let samples = reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(reader.spec().channels, 2);
        assert_eq!(samples, vec![0.0, 0.5, -0.25, 0.25]);
    }

    #[test]
    fn reads_float_wav_as_clip_and_waveform_batch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("float.wav");
        let clip = AudioClip::new(8_000, 2, vec![0.0, 0.5, -0.25, 0.25]).unwrap();

        write_clip_as_wav(&path, &clip).unwrap();

        let decoded = read_wav_as_clip(&path).unwrap();
        assert_eq!(decoded.sample_rate, 8_000);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.samples, clip.samples);

        let batch = read_wav_as_waveform_batch(&path).unwrap();
        let view = batch.as_view().unwrap();
        assert_eq!(view.batch_size(), 1);
        assert_eq!(view.channel_count(), 2);
        assert_eq!(view.time_steps(), 2);
        assert_eq!(view.waveform(0, 0).unwrap(), &[0.0, -0.25]);
        assert_eq!(view.waveform(0, 1).unwrap(), &[0.5, 0.25]);
    }

    #[test]
    fn reads_pcm_wav_as_normalized_clip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pcm.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 4,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        writer.write_sample::<i16>(0).unwrap();
        writer.write_sample::<i16>(16_384).unwrap();
        writer.write_sample::<i16>(-16_384).unwrap();
        writer.finalize().unwrap();

        let decoded = read_wav_as_clip(&path).unwrap();

        assert_eq!(decoded.sample_rate, 4);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples, vec![0.0, 0.5, -0.5]);
    }

    #[test]
    fn rejects_multi_item_waveform_batches_for_wav_export() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("waveform.wav");
        let first = OwnedAudioFrame::new(
            Timestamp::new(0, Timebase::new(1, 16_000)),
            16_000,
            1,
            AudioBuffer::F32(vec![0.0, 0.25]),
        )
        .unwrap();
        let second = OwnedAudioFrame::new(
            Timestamp::new(2, Timebase::new(1, 16_000)),
            16_000,
            1,
            AudioBuffer::F32(vec![0.5, 0.75]),
        )
        .unwrap();
        let batch = OwnedAudioWaveformBatch::from_audio_frames(&[first, second]).unwrap();
        assert!(write_waveform_batch_as_wav(&path, &batch).is_err());
    }
}
