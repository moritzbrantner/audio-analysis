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
    SpeakerAudio, SpeakerDiarizer, SpeechSpan, VadOptions, VadResponse, VoiceActivityDetector,
    WindowedSpeakerDiarizer, DEFAULT_MIN_SPEAKER_EVIDENCE_SECONDS,
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

/// Reference speaker turn from an RTTM file.
struct ReferenceTurn {
    start: f64,
    end: f64,
    speaker: String,
}

fn read_rttm(path: &PathBuf) -> Vec<ReferenceTurn> {
    std::fs::read_to_string(path)
        .expect("read RTTM")
        .lines()
        .filter_map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            (fields.first() == Some(&"SPEAKER")).then(|| {
                let start = fields[3].parse::<f64>().expect("RTTM onset");
                let duration = fields[4].parse::<f64>().expect("RTTM duration");
                ReferenceTurn {
                    start,
                    end: start + duration,
                    speaker: fields[7].to_string(),
                }
            })
        })
        .collect()
}

/// Frame-level (10 ms, no collar, overlap scored) diarization error against an
/// RTTM reference, with the optimal one-to-one speaker mapping.
#[derive(Debug)]
#[allow(dead_code)] // fields are read through the Debug report
struct DiarizationError {
    reference_speakers: usize,
    predicted_speakers: usize,
    /// Missed + false alarm + confusion, over reference speech.
    der: f64,
    /// Reference speech with no resolved prediction. Word-timed spans leave
    /// pauses and overlaps uncovered, so this is mostly a span-coverage cost.
    missed: f64,
    false_alarm: f64,
    /// Speech attributed to the wrong speaker: the who-spoke-when error.
    confusion: f64,
}

fn diarization_error(
    reference: &[ReferenceTurn],
    segments: &[audio_analysis_speakers::DiarizationSegment],
    audio_seconds: f64,
) -> DiarizationError {
    // Excerpt RTTMs keep turns that run past the cut, so clip to the audio.
    let frames = (audio_seconds * 100.0).round() as usize;
    let frame = |seconds: f64| ((seconds * 100.0).round() as usize).min(frames);
    let reference_labels = reference
        .iter()
        .map(|turn| turn.speaker.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let predicted_labels = segments
        .iter()
        .filter_map(|segment| match &segment.speaker {
            DiarizedSpeaker::Unknown(label) => Some(label.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut reference_frames = vec![0_u64; frames];
    for turn in reference {
        let bit = reference_labels
            .iter()
            .position(|label| *label == turn.speaker)
            .unwrap();
        for value in &mut reference_frames[frame(turn.start)..frame(turn.end)] {
            *value |= 1 << bit;
        }
    }
    // Transcript spans never overlap, so each frame has at most one prediction.
    let mut predicted_frames = vec![None::<usize>; frames];
    for segment in segments {
        let DiarizedSpeaker::Unknown(label) = &segment.speaker else {
            continue;
        };
        let index = predicted_labels.iter().position(|value| value == label);
        for value in &mut predicted_frames[frame(segment.start_seconds)..frame(segment.end_seconds)]
        {
            *value = index;
        }
    }
    // co[p][r]: frames where predicted speaker p and reference speaker r overlap.
    let mut co = vec![vec![0_usize; reference_labels.len()]; predicted_labels.len()];
    let (mut total, mut missed, mut false_alarm, mut matched) = (0, 0, 0, 0);
    for (reference_bits, predicted) in reference_frames.iter().zip(&predicted_frames) {
        let r = reference_bits.count_ones() as usize;
        let p = usize::from(predicted.is_some());
        total += r;
        missed += r.saturating_sub(p);
        false_alarm += p.saturating_sub(r);
        matched += r.min(p);
        if let Some(pi) = predicted {
            for (ri, count) in co[*pi].iter_mut().enumerate() {
                if reference_bits & (1 << ri) != 0 {
                    *count += 1;
                }
            }
        }
    }
    // Optimal one-to-one mapping: DP over subsets of reference speakers.
    let mut best = vec![None::<usize>; 1 << reference_labels.len()];
    best[0] = Some(0);
    for row in &co {
        let mut next = best.clone();
        for (mask, value) in best.iter().enumerate() {
            let Some(value) = value else { continue };
            for (ri, count) in row.iter().enumerate() {
                if mask & (1 << ri) == 0 {
                    let slot = &mut next[mask | (1 << ri)];
                    *slot = Some(slot.unwrap_or(0).max(value + count));
                }
            }
        }
        best = next;
    }
    let correct = best.into_iter().flatten().max().unwrap_or(0);
    let total = total as f64;
    let confusion = (matched - correct) as f64;
    DiarizationError {
        reference_speakers: reference_labels.len(),
        predicted_speakers: predicted_labels.len(),
        der: (missed as f64 + false_alarm as f64 + confusion) / total,
        missed: missed as f64 / total,
        false_alarm: false_alarm as f64 / total,
        confusion: confusion / total,
    }
}

/// Word spans from a WhisperX JSON transcript, merged like the transcript
/// diarization provider (`merge_transcript_speech_spans`, default VAD gap).
fn read_whisperx_word_spans(path: &PathBuf) -> Vec<SpeechSpan> {
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read transcript"))
            .expect("transcript JSON");
    let mut words = json["segments"]
        .as_array()
        .expect("segments")
        .iter()
        .flat_map(|segment| segment["words"].as_array().cloned().unwrap_or_default())
        .filter_map(|word| Some((word["start"].as_f64()?, word["end"].as_f64()?)))
        .filter(|(start, end)| end > start)
        .collect::<Vec<_>>();
    words.sort_by(|left, right| left.0.total_cmp(&right.0));
    let merge_gap = VadOptions::default().merge_gap_seconds;
    let mut spans: Vec<SpeechSpan> = Vec::new();
    for (start, end) in words {
        match spans.last_mut() {
            Some(last) if start - last.end_seconds <= merge_gap => {
                last.end_seconds = last.end_seconds.max(end);
            }
            _ => spans.push(SpeechSpan::new(start, end, 1.0).unwrap()),
        }
    }
    spans
}

struct FixedSpans(Vec<SpeechSpan>);

impl VoiceActivityDetector for FixedSpans {
    fn provider_id(&self) -> &str {
        "transcript-word-spans"
    }

    fn detect_speech(
        &mut self,
        _audio: &SpeakerAudio<'_>,
        _options: &VadOptions,
    ) -> audio_contracts::Result<VadResponse> {
        Ok(VadResponse {
            segments: self.0.clone(),
            diagnostics: Vec::new(),
        })
    }
}

fn overlap(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.1.min(b.1) - a.0.max(b.0)).max(0.0)
}

/// Short-turn survival under one evidence threshold.
#[derive(Debug)]
#[allow(dead_code)] // fields are read through the Debug report
struct ShortTurnReport {
    threshold: f64,
    predicted_speakers: usize,
    spans: usize,
    unresolved_spans: usize,
    unresolved_seconds: f64,
    /// Reference turns shorter than one second that the spans touch at all.
    short_turns_covered: usize,
    /// Of those, turns with less than half their duration in resolved spans.
    short_turns_unresolved: usize,
}

fn short_turn_report(
    threshold: f64,
    reference: &[ReferenceTurn],
    segments: &[audio_analysis_speakers::DiarizationSegment],
) -> ShortTurnReport {
    let span = |segment: &audio_analysis_speakers::DiarizationSegment| {
        (segment.start_seconds, segment.end_seconds)
    };
    let unresolved = segments
        .iter()
        .filter(|segment| segment.speaker == DiarizedSpeaker::Unresolved)
        .collect::<Vec<_>>();
    let mut short_turns_covered = 0;
    let mut short_turns_unresolved = 0;
    for turn in reference.iter().filter(|turn| turn.end - turn.start < 1.0) {
        let turn_span = (turn.start, turn.end);
        let covered = segments
            .iter()
            .map(|segment| overlap(turn_span, span(segment)))
            .sum::<f64>();
        if covered <= 0.0 {
            continue;
        }
        short_turns_covered += 1;
        let resolved = segments
            .iter()
            .filter(|segment| segment.speaker != DiarizedSpeaker::Unresolved)
            .map(|segment| overlap(turn_span, span(segment)))
            .sum::<f64>();
        if resolved < 0.5 * (turn.end - turn.start) {
            short_turns_unresolved += 1;
        }
    }
    ShortTurnReport {
        threshold,
        predicted_speakers: segments
            .iter()
            .filter_map(|segment| match &segment.speaker {
                DiarizedSpeaker::Unknown(label) => Some(label.as_str()),
                _ => None,
            })
            .collect::<BTreeSet<_>>()
            .len(),
        spans: segments.len(),
        unresolved_spans: unresolved.len(),
        unresolved_seconds: unresolved
            .iter()
            .map(|segment| segment.end_seconds - segment.start_seconds)
            .sum(),
        short_turns_covered,
        short_turns_unresolved,
    }
}

/// Real multi-speaker check on the transcript-span provider path (merged word
/// spans and default clustering, as in `audio-analysis-transcription`):
/// - issue #152: the default evidence threshold must not swallow genuine short
///   turns;
/// - issue #204: the speaker count must match the reference and few frames may
///   go to the wrong speaker.
///
/// Inputs (free, public):
/// - `AUDIO_SPEAKERS_WESPEAKER_ONNX`: as above.
/// - `AUDIO_SPEAKERS_MULTI_SPEAKER_WAV`: 16 kHz mono conversation, e.g. the
///   pyannote.audio 4.0.4 `sample.wav` (MIT) or AMI `EN2002c` Mix-Headset
///   0–300 s (CC BY 4.0).
/// - `AUDIO_SPEAKERS_MULTI_SPEAKER_RTTM`: matching human reference turns
///   (`sample.rttm`, or pyannote AMI-diarization-setup `only_words` RTTM).
/// - `AUDIO_SPEAKERS_MULTI_SPEAKER_TRANSCRIPT`: WhisperX JSON with word timings.
#[test]
#[ignore = "requires AUDIO_SPEAKERS_WESPEAKER_ONNX and AUDIO_SPEAKERS_MULTI_SPEAKER_* inputs"]
fn transcript_spans_keep_short_turns_and_reference_speakers() {
    let (Some(model), Some(wav), Some(rttm), Some(transcript)) = (
        env_path("AUDIO_SPEAKERS_WESPEAKER_ONNX"),
        env_path("AUDIO_SPEAKERS_MULTI_SPEAKER_WAV"),
        env_path("AUDIO_SPEAKERS_MULTI_SPEAKER_RTTM"),
        env_path("AUDIO_SPEAKERS_MULTI_SPEAKER_TRANSCRIPT"),
    ) else {
        return;
    };
    let samples = read_mono_16k(&wav);
    let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();
    let reference = read_rttm(&rttm);
    let spans = read_whisperx_word_spans(&transcript);

    // The former 0.5 s default is reported for comparison.
    let mut reports = Vec::new();
    for threshold in [DEFAULT_MIN_SPEAKER_EVIDENCE_SECONDS, 0.5] {
        let embedder = OnnxSpeakerEmbedder::from_config(OnnxSpeakerEmbeddingConfig {
            bundle_path: model.clone(),
            embedding_dimension: 256,
            ..OnnxSpeakerEmbeddingConfig::default()
        })
        .unwrap();
        let mut diarizer = WindowedSpeakerDiarizer::new(embedder, FixedSpans(spans.clone()))
            .min_speaker_evidence_seconds(threshold)
            .unwrap();
        let result = diarizer.diarize(&audio).unwrap();
        let report = short_turn_report(threshold, &reference, &result.segments);
        let error = diarization_error(
            &reference,
            &result.segments,
            samples.len() as f64 / 16_000.0,
        );
        eprintln!("{report:?}\n{error:?}");
        reports.push((report, error));
    }

    let (default, error) = &reports[0];
    assert!(default.short_turns_covered > 0, "{default:?}");
    assert!(
        default.short_turns_unresolved * 2 < default.short_turns_covered,
        "default threshold leaves most real short turns unresolved: {default:?}"
    );
    assert_eq!(
        error.predicted_speakers, error.reference_speakers,
        "{error:?}"
    );
    assert!(error.confusion < 0.1, "{error:?}");
}
