use std::path::{Path, PathBuf};
use std::process::Command;

use audio_contracts::{DetectError, Result};
use video_analysis_ffmpeg::is_ffmpeg_available;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Supported audio output file formats.
pub enum AudioFileFormat {
    /// WAV.
    Wav,
    /// MP3.
    Mp3,
    /// FLAC.
    Flac,
    /// AAC in an M4A container.
    M4a,
    /// Ogg Vorbis/Opus container.
    Ogg,
}

impl AudioFileFormat {
    /// Returns file extension without a leading dot.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::M4a => "m4a",
            Self::Ogg => "ogg",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Segment range for file splitting.
pub struct AudioSegmentSpec {
    /// Start time in seconds.
    pub start_seconds: f64,
    /// End time in seconds.
    pub end_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
/// Request for FFmpeg-backed file splitting.
pub struct SplitAudioFileRequest {
    /// Input path.
    pub input: PathBuf,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Segments to extract.
    pub segments: Vec<AudioSegmentSpec>,
    /// Output format.
    pub output_format: AudioFileFormat,
}

#[derive(Debug, Clone, PartialEq)]
/// Response from file splitting.
pub struct SplitAudioFileResponse {
    /// Written output files.
    pub outputs: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
/// Request for FFmpeg-backed file joining.
pub struct JoinAudioFilesRequest {
    /// Input paths.
    pub inputs: Vec<PathBuf>,
    /// Output path.
    pub output: PathBuf,
    /// Optional crossfade length.
    pub crossfade_seconds: Option<f64>,
    /// Output format.
    pub output_format: AudioFileFormat,
}

#[derive(Debug, Clone, PartialEq)]
/// Response from file joining.
pub struct JoinAudioFilesResponse {
    /// Output path.
    pub output: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
/// Request for FFmpeg-backed file processing.
pub struct ProcessAudioFileRequest {
    /// Input path.
    pub input: PathBuf,
    /// Output path.
    pub output: PathBuf,
    /// Edit spec.
    pub edit: FfmpegAudioEditSpec,
}

#[derive(Debug, Clone, PartialEq)]
/// Response from file processing.
pub struct ProcessAudioFileResponse {
    /// Output path.
    pub output: PathBuf,
    /// Audio filter chain used.
    pub filter_chain: String,
}

#[derive(Debug, Clone, PartialEq)]
/// FFmpeg audio edit specification.
pub struct FfmpegAudioEditSpec {
    /// Speed factor preserving pitch through atempo.
    pub speed_factor: Option<f32>,
    /// Pitch/key shift in semitones preserving duration.
    pub pitch_shift_semitones: Option<f32>,
    /// Ordered effects.
    pub effects: Vec<FfmpegAudioEffect>,
    /// Optional output sample rate.
    pub output_sample_rate: Option<u32>,
    /// Optional output channel count.
    pub output_channels: Option<u16>,
}

#[derive(Debug, Clone, PartialEq)]
/// FFmpeg-mapped audio effects.
pub enum FfmpegAudioEffect {
    /// Reverse audio.
    Reverse,
    /// Trim a range.
    Trim {
        /// Start seconds.
        start_seconds: f64,
        /// End seconds.
        end_seconds: f64,
    },
    /// Fade in/out.
    Fade {
        /// Fade-in seconds.
        fade_in_seconds: f64,
        /// Fade-out seconds.
        fade_out_seconds: f64,
        /// Optional total duration for fade-out placement.
        duration_seconds: Option<f64>,
    },
    /// Echo/delay through `aecho`.
    Echo {
        /// Input gain.
        in_gain: f32,
        /// Output gain.
        out_gain: f32,
        /// Delay in seconds.
        delay_seconds: f64,
        /// Decay amount.
        decay: f32,
    },
    /// Reverb fallback through `aecho`.
    Reverb {
        /// Room size hint.
        room_size: f32,
        /// Wet amount.
        wet: f32,
    },
    /// Compressor.
    Compressor {
        /// Threshold dB.
        threshold_db: f32,
        /// Ratio.
        ratio: f32,
        /// Attack ms.
        attack_ms: f32,
        /// Release ms.
        release_ms: f32,
    },
    /// Limiter.
    Limiter {
        /// Ceiling dB.
        ceiling_db: f32,
    },
    /// Parametric EQ.
    Eq {
        /// Frequency.
        frequency_hz: f32,
        /// Width/Q.
        width_q: f32,
        /// Gain dB.
        gain_db: f32,
    },
    /// Low pass.
    LowPass {
        /// Frequency.
        frequency_hz: f32,
    },
    /// High pass.
    HighPass {
        /// Frequency.
        frequency_hz: f32,
    },
    /// Chorus.
    Chorus,
    /// Flanger.
    Flanger,
    /// Tremolo.
    Tremolo {
        /// Frequency.
        frequency_hz: f32,
        /// Depth.
        depth: f32,
    },
    /// Loudness normalization.
    Normalize,
}

/// Builds an FFmpeg audio filter chain from an edit spec.
pub fn build_ffmpeg_audio_filter_chain(spec: &FfmpegAudioEditSpec) -> Result<String> {
    let mut filters = Vec::new();
    if let Some(semitones) = spec.pitch_shift_semitones {
        if !semitones.is_finite() {
            return Err(invalid("pitch_shift_semitones must be finite"));
        }
        let factor = 2.0_f32.powf(semitones / 12.0);
        let rate = spec.output_sample_rate.unwrap_or(48_000);
        filters.push(format!("asetrate={rate}*{factor:.8}"));
        filters.push(format!("aresample={rate}"));
        filters.extend(atempo_filters(1.0 / factor)?);
    }
    if let Some(speed) = spec.speed_factor {
        filters.extend(atempo_filters(speed)?);
    }
    for effect in &spec.effects {
        filters.extend(effect_filters(effect)?);
    }
    if let Some(rate) = spec.output_sample_rate {
        if rate == 0 {
            return Err(invalid("output_sample_rate must be positive"));
        }
        filters.push(format!("aresample={rate}"));
    }
    if let Some(channels) = spec.output_channels {
        if channels == 0 {
            return Err(invalid("output_channels must be positive"));
        }
        filters.push(format!(
            "aformat=channel_layouts={}",
            channel_layout(channels)
        ));
    }
    Ok(filters.join(","))
}

/// Processes an audio file through FFmpeg.
pub fn process_audio_file(request: ProcessAudioFileRequest) -> Result<ProcessAudioFileResponse> {
    ensure_ffmpeg_available()?;
    let filter_chain = build_ffmpeg_audio_filter_chain(&request.edit)?;
    ensure_parent(&request.output)?;
    let mut command = Command::new("ffmpeg");
    command.arg("-y").arg("-i").arg(&request.input);
    if !filter_chain.is_empty() {
        command.arg("-af").arg(&filter_chain);
    }
    command.arg(&request.output);
    run_ffmpeg(command, "process audio file")?;
    Ok(ProcessAudioFileResponse {
        output: request.output,
        filter_chain,
    })
}

/// Splits an audio file through FFmpeg.
pub fn split_audio_file(request: SplitAudioFileRequest) -> Result<SplitAudioFileResponse> {
    ensure_ffmpeg_available()?;
    if request.segments.is_empty() {
        return Err(invalid("split requires at least one segment"));
    }
    std::fs::create_dir_all(&request.output_dir)?;
    let mut outputs = Vec::with_capacity(request.segments.len());
    for (index, segment) in request.segments.iter().enumerate() {
        validate_segment(*segment)?;
        let output = request.output_dir.join(format!(
            "segment_{index:03}.{}",
            request.output_format.extension()
        ));
        let filter = format!(
            "atrim=start={:.6}:end={:.6},asetpts=PTS-STARTPTS",
            segment.start_seconds, segment.end_seconds
        );
        let mut command = Command::new("ffmpeg");
        command
            .arg("-y")
            .arg("-i")
            .arg(&request.input)
            .arg("-af")
            .arg(filter)
            .arg(&output);
        run_ffmpeg(command, "split audio file")?;
        outputs.push(output);
    }
    Ok(SplitAudioFileResponse { outputs })
}

/// Joins audio files through FFmpeg.
pub fn join_audio_files(request: JoinAudioFilesRequest) -> Result<JoinAudioFilesResponse> {
    ensure_ffmpeg_available()?;
    if request.inputs.is_empty() {
        return Err(invalid("join requires at least one input"));
    }
    ensure_parent(&request.output)?;
    let mut command = Command::new("ffmpeg");
    command.arg("-y");
    for input in &request.inputs {
        command.arg("-i").arg(input);
    }
    if let Some(crossfade) = request.crossfade_seconds {
        if !crossfade.is_finite() || crossfade < 0.0 {
            return Err(invalid("crossfade_seconds must be finite and non-negative"));
        }
        if request.inputs.len() == 1 || crossfade == 0.0 {
            command
                .arg("-filter_complex")
                .arg("[0:a]anull[out]")
                .arg("-map")
                .arg("[out]");
        } else {
            command
                .arg("-filter_complex")
                .arg(crossfade_filter(request.inputs.len(), crossfade))
                .arg("-map")
                .arg("[out]");
        }
    } else {
        let inputs = (0..request.inputs.len())
            .map(|index| format!("[{index}:a]"))
            .collect::<String>();
        command
            .arg("-filter_complex")
            .arg(format!(
                "{inputs}concat=n={}:v=0:a=1[out]",
                request.inputs.len()
            ))
            .arg("-map")
            .arg("[out]");
    }
    command.arg(&request.output);
    run_ffmpeg(command, "join audio files")?;
    Ok(JoinAudioFilesResponse {
        output: request.output,
    })
}

fn effect_filters(effect: &FfmpegAudioEffect) -> Result<Vec<String>> {
    Ok(match effect {
        FfmpegAudioEffect::Reverse => vec!["areverse".to_string()],
        FfmpegAudioEffect::Trim {
            start_seconds,
            end_seconds,
        } => {
            validate_segment(AudioSegmentSpec {
                start_seconds: *start_seconds,
                end_seconds: *end_seconds,
            })?;
            vec![format!(
                "atrim=start={start_seconds:.6}:end={end_seconds:.6}",
            ), "asetpts=PTS-STARTPTS".to_string()]
        }
        FfmpegAudioEffect::Fade {
            fade_in_seconds,
            fade_out_seconds,
            duration_seconds,
        } => {
            let mut filters = Vec::new();
            if *fade_in_seconds > 0.0 {
                filters.push(format!("afade=t=in:st=0:d={fade_in_seconds:.6}"));
            }
            if *fade_out_seconds > 0.0 {
                let start = duration_seconds
                    .map(|duration| (duration - fade_out_seconds).max(0.0))
                    .unwrap_or(0.0);
                filters.push(format!("afade=t=out:st={start:.6}:d={fade_out_seconds:.6}"));
            }
            filters
        }
        FfmpegAudioEffect::Echo {
            in_gain,
            out_gain,
            delay_seconds,
            decay,
        } => vec![format!(
            "aecho={in_gain:.6}:{out_gain:.6}:{:.3}:{decay:.6}",
            delay_seconds * 1_000.0
        )],
        FfmpegAudioEffect::Reverb { room_size, wet } => {
            let delay_ms = 40.0 + room_size.clamp(0.0, 1.0) as f64 * 90.0;
            vec![format!("aecho=0.8:{wet:.6}:{delay_ms:.3}:0.35")]
        }
        FfmpegAudioEffect::Compressor {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
        } => vec![format!(
            "acompressor=threshold={threshold_db:.6}dB:ratio={ratio:.6}:attack={attack_ms:.6}:release={release_ms:.6}"
        )],
        FfmpegAudioEffect::Limiter { ceiling_db } => {
            vec![format!("alimiter=limit={:.8}", 10.0_f32.powf(ceiling_db / 20.0))]
        }
        FfmpegAudioEffect::Eq {
            frequency_hz,
            width_q,
            gain_db,
        } => vec![format!(
            "equalizer=f={frequency_hz:.6}:width_type=q:width={width_q:.6}:g={gain_db:.6}"
        )],
        FfmpegAudioEffect::LowPass { frequency_hz } => {
            vec![format!("lowpass=f={frequency_hz:.6}")]
        }
        FfmpegAudioEffect::HighPass { frequency_hz } => {
            vec![format!("highpass=f={frequency_hz:.6}")]
        }
        FfmpegAudioEffect::Chorus => vec!["chorus=0.7:0.9:55:0.4:0.25:2".to_string()],
        FfmpegAudioEffect::Flanger => vec!["flanger".to_string()],
        FfmpegAudioEffect::Tremolo {
            frequency_hz,
            depth,
        } => vec![format!("tremolo=f={frequency_hz:.6}:d={depth:.6}")],
        FfmpegAudioEffect::Normalize => vec!["loudnorm".to_string()],
    })
}

fn atempo_filters(mut factor: f32) -> Result<Vec<String>> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(invalid("tempo factor must be finite and greater than zero"));
    }
    let mut filters = Vec::new();
    while factor > 2.0 {
        filters.push("atempo=2.000000".to_string());
        factor /= 2.0;
    }
    while factor < 0.5 {
        filters.push("atempo=0.500000".to_string());
        factor /= 0.5;
    }
    filters.push(format!("atempo={factor:.6}"));
    Ok(filters)
}

fn crossfade_filter(inputs: usize, seconds: f64) -> String {
    let mut filter = String::new();
    filter.push_str(&format!("[0:a][1:a]acrossfade=d={seconds:.6}[xf1];"));
    for index in 2..inputs {
        let previous = format!("xf{}", index - 1);
        let next = if index == inputs - 1 {
            "out".to_string()
        } else {
            format!("xf{index}")
        };
        filter.push_str(&format!(
            "[{previous}][{index}:a]acrossfade=d={seconds:.6}[{next}];"
        ));
    }
    filter.trim_end_matches(';').to_string()
}

fn channel_layout(channels: u16) -> &'static str {
    match channels {
        1 => "mono",
        2 => "stereo",
        _ => "0",
    }
}

fn validate_segment(segment: AudioSegmentSpec) -> Result<()> {
    if !segment.start_seconds.is_finite()
        || !segment.end_seconds.is_finite()
        || segment.start_seconds < 0.0
        || segment.end_seconds <= segment.start_seconds
    {
        return Err(invalid(
            "segment start/end must be finite, non-negative, and ordered",
        ));
    }
    Ok(())
}

fn ensure_ffmpeg_available() -> Result<()> {
    if is_ffmpeg_available() {
        Ok(())
    } else {
        Err(DetectError::Source(
            "ffmpeg is required for file-level audio editing but was not found on PATH".to_string(),
        ))
    }
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn run_ffmpeg(mut command: Command, action: &str) -> Result<()> {
    let output = command.output().map_err(|err| {
        DetectError::Source(format!("failed to start ffmpeg for {action}: {err}"))
    })?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(DetectError::Source(format!(
            "ffmpeg failed to {action}: {stderr}"
        )))
    }
}

fn invalid(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpeg_filter_chain_maps_audio_effects() {
        let filter = build_ffmpeg_audio_filter_chain(&FfmpegAudioEditSpec {
            speed_factor: Some(4.0),
            pitch_shift_semitones: Some(12.0),
            effects: vec![
                FfmpegAudioEffect::Reverse,
                FfmpegAudioEffect::Compressor {
                    threshold_db: -18.0,
                    ratio: 3.0,
                    attack_ms: 10.0,
                    release_ms: 100.0,
                },
                FfmpegAudioEffect::Limiter { ceiling_db: -1.0 },
                FfmpegAudioEffect::Eq {
                    frequency_hz: 1_000.0,
                    width_q: 1.0,
                    gain_db: 3.0,
                },
                FfmpegAudioEffect::Echo {
                    in_gain: 0.8,
                    out_gain: 0.9,
                    delay_seconds: 0.25,
                    decay: 0.35,
                },
                FfmpegAudioEffect::Chorus,
                FfmpegAudioEffect::Flanger,
                FfmpegAudioEffect::Tremolo {
                    frequency_hz: 5.0,
                    depth: 0.5,
                },
                FfmpegAudioEffect::Normalize,
            ],
            output_sample_rate: Some(48_000),
            output_channels: Some(2),
        })
        .unwrap();

        for expected in [
            "asetrate",
            "aresample",
            "atempo",
            "areverse",
            "acompressor",
            "alimiter",
            "equalizer",
            "aecho",
            "chorus",
            "flanger",
            "tremolo",
            "loudnorm",
        ] {
            assert!(filter.contains(expected), "missing {expected} in {filter}");
        }
    }
}
