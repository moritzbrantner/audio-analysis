use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use audio_contracts::{AudioBuffer, AudioSampleFormat, DetectError, OwnedAudioFrame, Result, Timebase, Timestamp};
use thiserror::Error;

use crate::{AudioFrameSource, AudioStreamInfo, MediaSourceInfo, SourceMode};

#[derive(Debug, Error)]
/// Typed FFmpeg command-adapter failures.
pub enum FfmpegError {
    /// FFprobe failed.
    #[error("ffprobe failed for `{input}`: {message}")]
    ProbeFailed { input: String, message: String },
    /// FFmpeg failed to start.
    #[error("ffmpeg failed to start for `{input}`: {message}")]
    StartFailed { input: String, message: String },
    /// Probe metadata was missing or invalid.
    #[error("missing or invalid audio metadata: {0}")]
    InvalidMetadata(String),
    /// A requested audio stream was unavailable.
    #[error("invalid audio stream selection {selection:?}: {reason:?}")]
    InvalidAudioStreamSelection {
        selection: AudioStreamSelection,
        reason: AudioStreamSelectionErrorReason,
        available_streams: MediaStreamInventory,
    },
    /// The selected backend cannot decode this input.
    #[error("unsupported FFmpeg runtime for decode: {message}")]
    UnsupportedRuntime { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Typed audio stream selector.
pub enum AudioStreamSelection {
    /// Zero-based audio-stream ordinal.
    AudioOrdinal(usize),
    /// Global container stream index.
    GlobalStreamIndex(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Why an audio stream selection failed.
pub enum AudioStreamSelectionErrorReason { NoAudioStreams, OutOfRange, NotAudio }

#[derive(Debug, Clone, PartialEq, Eq)]
/// Media type reported by FFprobe. Non-audio variants are inventory-only.
pub enum MediaType { Video, Audio, Subtitle, Data, Attachment, Unknown(String) }

#[derive(Debug, Clone, PartialEq, Eq)]
/// One FFprobe stream record.
pub struct MediaStream {
    pub index: u32,
    pub media_type: MediaType,
    pub audio_stream_ordinal: Option<usize>,
    pub codec: Option<String>,
    pub channels: Option<u16>,
    pub sample_rate: Option<u32>,
    pub language: Option<String>,
    pub default_disposition: Option<bool>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// Typed FFprobe stream inventory.
pub struct MediaStreamInventory { pub streams: Vec<MediaStream> }

/// Validates an audio selection against an inventory.
pub fn validate_audio_stream_selection(
    inventory: &MediaStreamInventory,
    selection: AudioStreamSelection,
) -> std::result::Result<&MediaStream, FfmpegError> {
    let selected = match selection {
        AudioStreamSelection::AudioOrdinal(ordinal) => {
            if !inventory.streams.iter().any(|stream| stream.media_type == MediaType::Audio) {
                return Err(selection_error(inventory, selection, AudioStreamSelectionErrorReason::NoAudioStreams));
            }
            inventory.streams.iter().find(|stream| stream.audio_stream_ordinal == Some(ordinal))
        }
        AudioStreamSelection::GlobalStreamIndex(index) => {
            let stream = inventory.streams.iter().find(|stream| stream.index == index);
            if stream.is_some_and(|stream| stream.media_type != MediaType::Audio) {
                return Err(selection_error(inventory, selection, AudioStreamSelectionErrorReason::NotAudio));
            }
            stream
        }
    };
    selected.ok_or_else(|| selection_error(inventory, selection, AudioStreamSelectionErrorReason::OutOfRange))
}

fn selection_error(inventory: &MediaStreamInventory, selection: AudioStreamSelection, reason: AudioStreamSelectionErrorReason) -> FfmpegError {
    FfmpegError::InvalidAudioStreamSelection { selection, reason, available_streams: inventory.clone() }
}

#[derive(Debug, Clone)]
/// Probed audio metadata.
pub struct AudioMetadata {
    pub input: String,
    pub path: Option<PathBuf>,
    pub mode: SourceMode,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_seconds: Option<f64>,
}

#[derive(Debug, Clone, Default)]
/// Runtime backend for FFmpeg integration.
pub enum FfmpegRuntimeBackend { Native, #[default] Command }

#[derive(Debug, Clone, Default)]
/// Runtime selection options.
pub struct FfmpegRuntimeOptions { pub backend: FfmpegRuntimeBackend }

impl FfmpegRuntimeOptions {
    pub fn command() -> Self { Self { backend: FfmpegRuntimeBackend::Command } }
    pub fn native() -> Self { Self { backend: FfmpegRuntimeBackend::Native } }
}

#[derive(Debug, Clone)]
/// Options for the audio-only FFmpeg source.
pub struct FfmpegAudioSourceOptions {
    pub mode: SourceMode,
    pub realtime: bool,
    pub samples_per_chunk: usize,
    pub extra_input_args: Vec<String>,
    pub audio_stream_index: Option<usize>,
    pub runtime: FfmpegRuntimeOptions,
}

impl FfmpegAudioSourceOptions {
    pub fn recorded() -> Self {
        Self { mode: SourceMode::Recorded, realtime: false, samples_per_chunk: 1024, extra_input_args: Vec::new(), audio_stream_index: None, runtime: FfmpegRuntimeOptions::command() }
    }
    pub fn live() -> Self {
        Self { mode: SourceMode::Live, realtime: true, samples_per_chunk: 1024, extra_input_args: Vec::new(), audio_stream_index: None, runtime: FfmpegRuntimeOptions::command() }
    }
    pub fn samples_per_chunk(mut self, samples: usize) -> Self { self.samples_per_chunk = samples.max(1); self }
    pub fn extra_input_arg(mut self, arg: impl Into<String>) -> Self { self.extra_input_args.push(arg.into()); self }
    pub fn audio_stream_index(mut self, index: usize) -> Self { self.audio_stream_index = Some(index); self }
    pub fn runtime(mut self, runtime: FfmpegRuntimeOptions) -> Self { self.runtime = runtime; self }
}

/// Pull-based f32 audio decoder backed by the external FFmpeg command.
pub struct FfmpegAudioSource {
    metadata: AudioMetadata,
    source_info: MediaSourceInfo,
    child: Child,
    stdout: ChildStdout,
    next_sample_index: u64,
    samples_per_chunk: usize,
}

impl FfmpegAudioSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> { Self::open_path_with_options(path, FfmpegAudioSourceOptions::recorded()) }
    pub fn open_path_with_options(path: impl AsRef<Path>, options: FfmpegAudioSourceOptions) -> Result<Self> {
        Self::open_path_with_options_checked(path, options).map_err(into_detect_error)
    }
    pub fn open_path_with_options_checked(path: impl AsRef<Path>, options: FfmpegAudioSourceOptions) -> std::result::Result<Self, FfmpegError> {
        let path = path.as_ref().to_path_buf();
        let input = path.to_string_lossy().into_owned();
        let metadata = probe_selected_audio_input(&input, Some(path), options.mode, &options.runtime, options.audio_stream_index)?;
        Self::spawn(input, metadata, options)
    }
    pub fn open_input(input: impl Into<String>) -> Result<Self> { Self::open_input_with_options(input, FfmpegAudioSourceOptions::recorded()) }
    pub fn open_live(input: impl Into<String>) -> Result<Self> { Self::open_input_with_options(input, FfmpegAudioSourceOptions::live()) }
    pub fn open_input_with_options(input: impl Into<String>, options: FfmpegAudioSourceOptions) -> Result<Self> {
        Self::open_input_with_options_checked(input, options).map_err(into_detect_error)
    }
    pub fn open_input_with_options_checked(input: impl Into<String>, options: FfmpegAudioSourceOptions) -> std::result::Result<Self, FfmpegError> {
        let input = input.into();
        let metadata = probe_selected_audio_input(&input, None, options.mode, &options.runtime, options.audio_stream_index)?;
        Self::spawn(input, metadata, options)
    }
    fn spawn(input: String, metadata: AudioMetadata, options: FfmpegAudioSourceOptions) -> std::result::Result<Self, FfmpegError> {
        if matches!(options.runtime.backend, FfmpegRuntimeBackend::Native) {
            return Err(FfmpegError::UnsupportedRuntime { message: "native probing does not provide decode in this extraction".into() });
        }
        let source_info = MediaSourceInfo::recorded(metadata.input.clone()).with_audio(AudioStreamInfo {
            sample_rate: metadata.sample_rate,
            channels: metadata.channels,
            sample_format: AudioSampleFormat::F32,
        });
        let selected = options.audio_stream_index.unwrap_or(0);
        let mut command = Command::new("ffmpeg");
        command.args(build_audio_ffmpeg_args(&input, &options, selected)).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|error| FfmpegError::StartFailed { input: input.clone(), message: error.to_string() })?;
        let stdout = child.stdout.take().ok_or_else(|| FfmpegError::StartFailed { input, message: "ffmpeg stdout pipe unavailable".into() })?;
        Ok(Self { metadata, source_info, child, stdout, next_sample_index: 0, samples_per_chunk: options.samples_per_chunk.max(1) })
    }
    pub fn metadata(&self) -> &AudioMetadata { &self.metadata }
    pub fn source_info(&self) -> &MediaSourceInfo { &self.source_info }
    fn read_next_audio_frame(&mut self) -> Result<Option<OwnedAudioFrame>> {
        let frame_bytes = self.metadata.channels as usize * std::mem::size_of::<f32>();
        let target = self.samples_per_chunk * frame_bytes;
        let mut bytes = vec![0; target];
        let mut offset = 0;
        while offset < target {
            match self.stdout.read(&mut bytes[offset..]) {
                Ok(0) if offset == 0 => return Ok(None),
                Ok(0) => break,
                Ok(count) => offset += count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(DetectError::Io(error)),
            }
        }
        bytes.truncate(offset - (offset % frame_bytes));
        if bytes.is_empty() { return Ok(None); }
        let samples = bytes.chunks_exact(4).map(|part| f32::from_le_bytes([part[0], part[1], part[2], part[3]])).collect::<Vec<_>>();
        let timestamp = Timestamp::new(self.next_sample_index as i64, Timebase::new(1, self.metadata.sample_rate as i32));
        self.next_sample_index += (samples.len() / self.metadata.channels as usize) as u64;
        OwnedAudioFrame::new(timestamp, self.metadata.sample_rate, self.metadata.channels, AudioBuffer::F32(samples)).map(Some)
    }
}

impl Drop for FfmpegAudioSource {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}

impl AudioFrameSource for FfmpegAudioSource {
    fn source_info(&self) -> &MediaSourceInfo { &self.source_info }
    fn next_audio_frame(&mut self) -> Result<Option<OwnedAudioFrame>> { self.read_next_audio_frame() }
}

fn build_audio_ffmpeg_args(input: &str, options: &FfmpegAudioSourceOptions, stream: usize) -> Vec<String> {
    let mut args = vec!["-v".into(), "error".into()];
    if options.realtime { args.extend(["-fflags", "nobuffer", "-flags", "low_delay"].into_iter().map(str::to_owned)); }
    args.extend(options.extra_input_args.iter().cloned());
    args.extend(["-i".into(), input.into(), "-map".into(), format!("0:a:{stream}"), "-vn".into(), "-f".into(), "f32le".into(), "-acodec".into(), "pcm_f32le".into(), "pipe:1".into()]);
    args
}

/// Probes the default audio stream in a path.
pub fn probe_audio(path: impl AsRef<Path>) -> std::result::Result<AudioMetadata, FfmpegError> {
    let path = path.as_ref().to_path_buf();
    let input = path.to_string_lossy().into_owned();
    probe_audio_input_with_ordinal(&input, Some(path), SourceMode::Recorded, 0)
}

/// Probes the default audio stream in an arbitrary input.
pub fn probe_audio_input(input: impl AsRef<str>) -> std::result::Result<AudioMetadata, FfmpegError> {
    probe_audio_input_with_ordinal(input.as_ref(), None, SourceMode::Recorded, 0)
}

/// Returns a typed inventory of every stream in a path.
pub fn probe_streams(path: impl AsRef<Path>) -> std::result::Result<MediaStreamInventory, FfmpegError> {
    probe_streams_input(path.as_ref().to_string_lossy())
}

/// Returns a typed inventory of every stream in an input.
pub fn probe_streams_input(input: impl AsRef<str>) -> std::result::Result<MediaStreamInventory, FfmpegError> {
    let input = input.as_ref();
    let output = Command::new("ffprobe").args(["-v", "error", "-show_entries", "stream=index,codec_type,codec_name,channels,sample_rate:stream_tags=language:stream_disposition=default", "-of", "json", input]).output().map_err(|error| FfmpegError::ProbeFailed { input: input.into(), message: error.to_string() })?;
    if !output.status.success() { return Err(FfmpegError::ProbeFailed { input: input.into(), message: String::from_utf8_lossy(&output.stderr).trim().into() }); }
    parse_stream_inventory(&String::from_utf8_lossy(&output.stdout))
}

fn probe_selected_audio_input(input: &str, path: Option<PathBuf>, mode: SourceMode, runtime: &FfmpegRuntimeOptions, ordinal: Option<usize>) -> std::result::Result<AudioMetadata, FfmpegError> {
    if matches!(runtime.backend, FfmpegRuntimeBackend::Native) { return Err(FfmpegError::UnsupportedRuntime { message: "native probing is not enabled".into() }); }
    if let Some(ordinal) = ordinal {
        validate_audio_stream_selection(&probe_streams_input(input)?, AudioStreamSelection::AudioOrdinal(ordinal))?;
    }
    let ordinal = ordinal.unwrap_or(0);
    probe_audio_input_with_ordinal(input, path, mode, ordinal)
}

fn probe_audio_input_with_ordinal(input: &str, path: Option<PathBuf>, mode: SourceMode, ordinal: usize) -> std::result::Result<AudioMetadata, FfmpegError> {
    let output = Command::new("ffprobe").args(["-v", "error", "-select_streams", &format!("a:{ordinal}"), "-show_entries", "stream=sample_rate,channels,duration", "-of", "default=noprint_wrappers=1:nokey=1", input]).output().map_err(|error| FfmpegError::ProbeFailed { input: input.into(), message: error.to_string() })?;
    if !output.status.success() { return Err(FfmpegError::ProbeFailed { input: input.into(), message: String::from_utf8_lossy(&output.stderr).trim().into() }); }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut lines = text.lines();
    let sample_rate = parse_u32(lines.next(), "sample_rate")?;
    let channels = parse_u16(lines.next(), "channels")?;
    let duration_seconds = lines.next().and_then(|value| value.parse().ok());
    Ok(AudioMetadata { input: input.into(), path, mode, sample_rate, channels, duration_seconds })
}

fn parse_u32(value: Option<&str>, name: &str) -> std::result::Result<u32, FfmpegError> {
    value.ok_or_else(|| FfmpegError::InvalidMetadata(format!("missing {name}")))?.parse().map_err(|error| FfmpegError::InvalidMetadata(format!("invalid {name}: {error}")))
}
fn parse_u16(value: Option<&str>, name: &str) -> std::result::Result<u16, FfmpegError> {
    value.ok_or_else(|| FfmpegError::InvalidMetadata(format!("missing {name}")))?.parse().map_err(|error| FfmpegError::InvalidMetadata(format!("invalid {name}: {error}")))
}

fn parse_stream_inventory(json: &str) -> std::result::Result<MediaStreamInventory, FfmpegError> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|error| FfmpegError::InvalidMetadata(format!("invalid ffprobe JSON: {error}")))?;
    let streams = value.get("streams").and_then(serde_json::Value::as_array).ok_or_else(|| FfmpegError::InvalidMetadata("missing streams array".into()))?;
    let mut ordinal = 0;
    let mut parsed = Vec::with_capacity(streams.len());
    for stream in streams {
        let index = stream.get("index").and_then(serde_json::Value::as_u64).and_then(|value| u32::try_from(value).ok()).ok_or_else(|| FfmpegError::InvalidMetadata("invalid stream index".into()))?;
        let media_type = match stream.get("codec_type").and_then(serde_json::Value::as_str).unwrap_or("unknown") {
            "video" => MediaType::Video, "audio" => MediaType::Audio, "subtitle" => MediaType::Subtitle, "data" => MediaType::Data, "attachment" => MediaType::Attachment, other => MediaType::Unknown(other.into()),
        };
        let audio_stream_ordinal = (media_type == MediaType::Audio).then(|| { let value = ordinal; ordinal += 1; value });
        parsed.push(MediaStream {
            index, media_type, audio_stream_ordinal,
            codec: stream.get("codec_name").and_then(serde_json::Value::as_str).map(str::to_owned),
            channels: stream.get("channels").and_then(serde_json::Value::as_u64).and_then(|value| u16::try_from(value).ok()),
            sample_rate: stream.get("sample_rate").and_then(serde_json::Value::as_str).and_then(|value| value.parse().ok()),
            language: stream.get("tags").and_then(|tags| tags.get("language")).and_then(serde_json::Value::as_str).map(str::to_owned),
            default_disposition: stream.get("disposition").and_then(|value| value.get("default")).and_then(serde_json::Value::as_u64).map(|value| value != 0),
        });
    }
    Ok(MediaStreamInventory { streams: parsed })
}

fn into_detect_error(error: FfmpegError) -> DetectError { DetectError::Source(error.to_string()) }

/// Returns whether the FFmpeg command is available.
pub fn is_ffmpeg_available() -> bool { command_available("ffmpeg") }
/// Returns whether the FFprobe command is available.
pub fn is_ffprobe_available() -> bool { command_available("ffprobe") }
fn command_available(command: &str) -> bool { Command::new(command).arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|status| status.success()) }

/// Writes a small container with two distinguishable audio streams.
pub fn write_two_audio_stream_test_media(path: impl AsRef<Path>) -> Result<()> {
    let status = Command::new("ffmpeg").args(["-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=16x16:d=0.2:r=10", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2:sample_rate=48000", "-f", "lavfi", "-i", "sine=frequency=880:duration=0.2:sample_rate=24000", "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-c:v", "ffv1", "-c:a", "pcm_s16le"]).arg(path.as_ref()).status()?;
    if status.success() { Ok(()) } else { Err(DetectError::Source("ffmpeg failed to generate multi-audio fixture".into())) }
}
