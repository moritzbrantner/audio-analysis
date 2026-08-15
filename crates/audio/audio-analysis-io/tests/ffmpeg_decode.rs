use audio_analysis_core::ChannelMix;
use audio_analysis_io::{
    decode_audio_to_mono_f32, decode_selected_media_to_mono_f32, open_audio_input,
    AudioFrameSource, AudioInput, AudioInputOptions, AudioIoError, SelectedMediaSource,
};
use audio_contracts::{AudioBuffer, AudioSampleFormat};
use video_analysis_ffmpeg::{
    is_ffmpeg_available, is_ffprobe_available, write_two_audio_stream_test_media,
    AudioStreamSelection, AudioStreamSelectionErrorReason, FfmpegError,
};

fn sine(freq_hz: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
    let samples = (sample_rate as f32 * seconds) as usize;
    (0..samples)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

fn write_pcm16_wav(
    path: impl AsRef<std::path::Path>,
    sample_rate: u32,
    channels: u16,
    samples: &[f32],
) -> std::io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    let data_len = samples.len() as u32 * 2;
    let byte_rate = sample_rate * channels as u32 * 2;
    let block_align = channels * 2;

    use std::io::Write;

    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_len).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16_u32.to_le_bytes())?;
    file.write_all(&1_u16.to_le_bytes())?;
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&16_u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn zero_crossings(samples: &[f32]) -> usize {
    samples
        .windows(2)
        .filter(|window| window[0] <= 0.0 && window[1] > 0.0)
        .count()
}

#[test]
fn decodes_generated_pcm16_wav_when_ffmpeg_is_available() {
    let required = std::env::var("FFMPEG_EXTERNAL_TESTS").ok().as_deref() == Some("1");
    if !required {
        eprintln!("skipping FFmpeg decode test; set FFMPEG_EXTERNAL_TESTS=1");
        return;
    }
    if !(is_ffmpeg_available() && is_ffprobe_available()) {
        panic!("FFMPEG_EXTERNAL_TESTS=1 but ffmpeg/ffprobe is unavailable");
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tone.wav");
    write_pcm16_wav(&path, 8_000, 1, &sine(440.0, 8_000, 0.1)).unwrap();

    let mut source = open_audio_input(
        AudioInput::File(path.clone()),
        AudioInputOptions::recorded().samples_per_chunk(128),
    )
    .unwrap();

    assert_eq!(source.metadata().sample_rate, 8_000);
    assert_eq!(source.metadata().channels, 1);
    assert_eq!(
        source.source_info().audio[0].sample_format,
        AudioSampleFormat::F32
    );

    let mut total_samples = 0;
    let mut saw_signal = false;
    while let Some(frame) = source.next_audio_frame().unwrap() {
        assert_eq!(frame.sample_rate, 8_000);
        assert_eq!(frame.channels, 1);
        let AudioBuffer::F32(samples) = frame.data else {
            panic!("expected f32 decoded audio");
        };
        total_samples += samples.len();
        saw_signal |= samples.iter().any(|sample| sample.abs() > 0.01);
    }

    assert_eq!(total_samples, 800);
    assert!(saw_signal);
}

#[test]
fn decodes_compressed_audio_when_ffmpeg_is_available() {
    let required = std::env::var("FFMPEG_EXTERNAL_TESTS").ok().as_deref() == Some("1");
    if !required {
        eprintln!("skipping compressed FFmpeg decode test; set FFMPEG_EXTERNAL_TESTS=1");
        return;
    }
    if !(is_ffmpeg_available() && is_ffprobe_available()) {
        panic!("FFMPEG_EXTERNAL_TESTS=1 but ffmpeg/ffprobe is unavailable");
    }

    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("tone.wav");
    let mp3 = dir.path().join("tone.mp3");
    write_pcm16_wav(&wav, 8_000, 1, &sine(330.0, 8_000, 0.1)).unwrap();
    let status = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(&wav)
        .arg(&mp3)
        .status()
        .unwrap();
    assert!(status.success());

    let (metadata, mono) = decode_audio_to_mono_f32(
        AudioInput::File(mp3),
        AudioInputOptions::recorded().samples_per_chunk(128),
        ChannelMix::Average,
    )
    .unwrap();

    assert_eq!(metadata.channels, 1);
    assert!(mono.iter().any(|sample| sample.abs() > 0.01));
}

#[test]
fn selected_media_decodes_distinguishable_tracks_and_preserves_default() {
    let required = std::env::var("FFMPEG_EXTERNAL_TESTS").ok().as_deref() == Some("1");
    if !required {
        eprintln!("skipping selected-media test; set FFMPEG_EXTERNAL_TESTS=1");
        return;
    }
    if !(is_ffmpeg_available() && is_ffprobe_available()) {
        panic!("FFMPEG_EXTERNAL_TESTS=1 but ffmpeg/ffprobe is unavailable");
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two-audio-streams.mkv");
    write_two_audio_stream_test_media(&path).unwrap();

    let (_, default) = decode_selected_media_to_mono_f32(
        SelectedMediaSource::new(&path),
        AudioInputOptions::recorded(),
        ChannelMix::Average,
    )
    .unwrap();
    let (_, first) = decode_selected_media_to_mono_f32(
        SelectedMediaSource::new(&path).audio_stream_index(0),
        AudioInputOptions::recorded(),
        ChannelMix::Average,
    )
    .unwrap();
    let (_, second) = decode_selected_media_to_mono_f32(
        SelectedMediaSource::new(&path).audio_stream_index(1),
        AudioInputOptions::recorded(),
        ChannelMix::Average,
    )
    .unwrap();

    assert_eq!(zero_crossings(&default), zero_crossings(&first));
    assert!(zero_crossings(&second) > zero_crossings(&first) * 3 / 2);
}

#[test]
fn invalid_selected_media_retains_typed_available_streams() {
    let required = std::env::var("FFMPEG_EXTERNAL_TESTS").ok().as_deref() == Some("1");
    if !required {
        eprintln!("skipping selected-media test; set FFMPEG_EXTERNAL_TESTS=1");
        return;
    }
    if !(is_ffmpeg_available() && is_ffprobe_available()) {
        panic!("FFMPEG_EXTERNAL_TESTS=1 but ffmpeg/ffprobe is unavailable");
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two-audio-streams.mkv");
    write_two_audio_stream_test_media(&path).unwrap();
    let error = decode_selected_media_to_mono_f32(
        SelectedMediaSource::new(path).audio_stream_index(2),
        AudioInputOptions::recorded(),
        ChannelMix::Average,
    )
    .unwrap_err();

    let AudioIoError::Ffmpeg(FfmpegError::InvalidAudioStreamSelection {
        selection,
        reason,
        available_streams,
    }) = error
    else {
        panic!("expected typed selected-stream error");
    };
    assert_eq!(selection, AudioStreamSelection::AudioOrdinal(2));
    assert_eq!(reason, AudioStreamSelectionErrorReason::OutOfRange);
    assert_eq!(available_streams.streams.len(), 3);
}
