#![cfg(feature = "onnx")]

//! Real-model check for issue #154: a single LibriSpeech speaker must not be
//! split by a short tail span.
//!
//! Inputs (free, public):
//! - `AUDIO_SPEAKERS_WESPEAKER_ONNX`: `Wespeaker/wespeaker-voxceleb-resnet34-LM`
//!   revision `f0c48c298fd835726c27956a5d617bad7115627e`, `voxceleb_resnet34_LM.onnx`
//!   (SHA-256 `7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068`).
//! - `AUDIO_SPEAKERS_SINGLE_SPEAKER_WAV`: LibriSpeech `1272-128104-0000` decoded to
//!   16 kHz mono.
//! - `ORT_DYLIB_PATH`: ONNX Runtime 1.24 shared library (`load-dynamic`).

use std::collections::BTreeSet;
use std::path::PathBuf;

use audio_analysis_speakers::{
    DiarizedSpeaker, EnergyVoiceActivityDetector, OnnxSpeakerEmbedder, OnnxSpeakerEmbeddingConfig,
    SpeakerAudio, SpeakerDiarizer, WindowedSpeakerDiarizer, DEFAULT_MIN_SPEAKER_EVIDENCE_SECONDS,
};

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn read_mono_16k(path: &PathBuf) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "WAV must be 16 kHz");
    assert_eq!(spec.channels, 1, "WAV must be mono");
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .expect("float samples"),
        hound::SampleFormat::Int => {
            let scale = (1_i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .expect("int samples")
        }
    }
}

#[test]
#[ignore = "requires AUDIO_SPEAKERS_WESPEAKER_ONNX and AUDIO_SPEAKERS_SINGLE_SPEAKER_WAV"]
fn single_librispeech_speaker_is_not_split_by_short_tail_span() {
    let (Some(model), Some(wav)) = (
        env_path("AUDIO_SPEAKERS_WESPEAKER_ONNX"),
        env_path("AUDIO_SPEAKERS_SINGLE_SPEAKER_WAV"),
    ) else {
        return;
    };
    let samples = read_mono_16k(&wav);
    let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();
    let embedder = OnnxSpeakerEmbedder::from_config(OnnxSpeakerEmbeddingConfig {
        bundle_path: model,
        embedding_dimension: 256,
        ..OnnxSpeakerEmbeddingConfig::default()
    })
    .unwrap();
    let mut diarizer = WindowedSpeakerDiarizer::new(embedder, EnergyVoiceActivityDetector);

    let result = diarizer.diarize(&audio).unwrap();
    for segment in &result.segments {
        eprintln!(
            "{:.2}..{:.2} {:?}",
            segment.start_seconds, segment.end_seconds, segment.speaker
        );
    }

    let speakers = result
        .segments
        .iter()
        .filter_map(|segment| match &segment.speaker {
            DiarizedSpeaker::Unknown(label) => Some(label.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(speakers.len(), 1, "single-speaker reference: {speakers:?}");
    // The 180 ms tail stays an explicit unresolved observation.
    for segment in &result.segments {
        let short =
            segment.end_seconds - segment.start_seconds < DEFAULT_MIN_SPEAKER_EVIDENCE_SECONDS;
        assert_eq!(short, segment.speaker == DiarizedSpeaker::Unresolved);
    }
    assert!(result
        .segments
        .iter()
        .any(|segment| segment.speaker == DiarizedSpeaker::Unresolved));
}
