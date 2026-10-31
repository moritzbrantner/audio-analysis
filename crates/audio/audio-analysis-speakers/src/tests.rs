use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "onnx")]
use audio_analysis_core::{interleaved_to_mono, ChannelMix};
use audio_analysis_recognition::{
    AudioRuntimeSelection, FallbackPolicy, SpectralEmbeddingConfig, TranscriptSegmentContract,
    TranscriptionContract,
};
#[cfg(feature = "onnx")]
use audio_contracts::AudioBuffer;
use audio_contracts::{DetectError, Result};

use super::*;
#[cfg(feature = "onnx")]
use crate::embedding::{
    format_onnx_dimensions, onnx_speaker_input_kind, resolve_onnx_speaker_model_path,
    select_onnx_speaker_input, select_onnx_speaker_output, speaker_logmel_features,
    OnnxSpeakerInputKind, SelectedOnnxIo,
};
use crate::request::merge_speaker_predictions;

#[test]
fn task_diarization_returns_imported_segments() {
    let response = diarize_speakers(SpeakerDiarizationRequest {
        source: None,
        duration_seconds: None,
        model: AudioRuntimeSelection::default(),
        imported_segments: vec![SpeakerSegmentPrediction {
            speaker: "speaker_a".to_string(),
            start_seconds: 0.0,
            end_seconds: 1.5,
            score: Some(0.8),
        }],
    })
    .unwrap();

    assert_eq!(response.runtime, AudioRuntime::Imported);
    assert_eq!(response.segments[0].speaker, "speaker_a");
}

#[test]
fn task_diarization_heuristic_returns_single_segment() {
    let response = diarize_speakers(SpeakerDiarizationRequest {
        source: None,
        duration_seconds: Some(2.5),
        model: AudioRuntimeSelection {
            fallback_policy: FallbackPolicy::HeuristicFallback,
            ..AudioRuntimeSelection::default()
        },
        imported_segments: Vec::new(),
    })
    .unwrap();

    assert_eq!(response.runtime, AudioRuntime::Heuristic);
    assert_eq!(response.segments[0].speaker, "speaker_0");
    assert_eq!(response.segments[0].end_seconds, 2.5);
}

#[test]
fn task_diarization_rejects_invalid_imported_segments() {
    let empty_speaker = diarize_speakers(SpeakerDiarizationRequest {
        source: None,
        duration_seconds: None,
        model: AudioRuntimeSelection::default(),
        imported_segments: vec![SpeakerSegmentPrediction {
            speaker: "   ".to_string(),
            start_seconds: 0.0,
            end_seconds: 1.0,
            score: Some(0.8),
        }],
    });
    assert!(empty_speaker.is_err());

    let reversed = diarize_speakers(SpeakerDiarizationRequest {
        source: None,
        duration_seconds: None,
        model: AudioRuntimeSelection::default(),
        imported_segments: vec![SpeakerSegmentPrediction {
            speaker: "speaker_a".to_string(),
            start_seconds: 2.0,
            end_seconds: 1.0,
            score: Some(0.8),
        }],
    });
    assert!(reversed.is_err());
}

#[test]
fn task_diarization_clamps_imported_scores() {
    let response = diarize_speakers(SpeakerDiarizationRequest {
        source: None,
        duration_seconds: None,
        model: AudioRuntimeSelection::default(),
        imported_segments: vec![SpeakerSegmentPrediction {
            speaker: " speaker_a ".to_string(),
            start_seconds: 0.0,
            end_seconds: 1.0,
            score: Some(2.0),
        }],
    })
    .unwrap();

    assert_eq!(response.segments[0].speaker, "speaker_a");
    assert_eq!(response.segments[0].score, Some(1.0));
}

#[test]
fn assigns_speakers_to_transcript_by_overlap() {
    let first = TranscriptSegmentContract::new(0, "hello")
        .with_time_range(Some(0.0), Some(1.0))
        .unwrap();
    let second = TranscriptSegmentContract::new(1, "world")
        .with_time_range(Some(1.0), Some(2.0))
        .unwrap();
    let transcript = TranscriptionContract::new(vec![first, second]);
    let diarization = SpeakerDiarizationResponse {
        accepted: true,
        operation: "diarize".to_string(),
        model_id: "fixture".to_string(),
        runtime: AudioRuntime::Imported,
        segments: vec![
            SpeakerSegmentPrediction {
                speaker: "speaker_b".to_string(),
                start_seconds: 0.5,
                end_seconds: 2.0,
                score: Some(0.7),
            },
            SpeakerSegmentPrediction {
                speaker: "speaker_a".to_string(),
                start_seconds: 0.0,
                end_seconds: 1.1,
                score: Some(0.9),
            },
        ],
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    };

    let assigned = assign_speakers_to_transcript(&transcript, &diarization).unwrap();

    assert_eq!(assigned.segments[0].speaker.as_deref(), Some("speaker_a"));
    assert_eq!(assigned.segments[1].speaker.as_deref(), Some("speaker_b"));
    assert_eq!(
        assigned.segments[0]
            .attributes
            .get("speakerScore")
            .map(String::as_str),
        Some("0.9")
    );
}

#[test]
fn speaker_assignment_preserves_existing_speaker_without_match() {
    let mut segment = TranscriptSegmentContract::new(0, "hello")
        .with_time_range(Some(0.0), Some(1.0))
        .unwrap();
    segment.speaker = Some("original".to_string());
    let transcript = TranscriptionContract::new(vec![segment]);
    let diarization = SpeakerDiarizationResponse {
        accepted: true,
        operation: "diarize".to_string(),
        model_id: "fixture".to_string(),
        runtime: AudioRuntime::Imported,
        segments: vec![SpeakerSegmentPrediction {
            speaker: "other".to_string(),
            start_seconds: 3.0,
            end_seconds: 4.0,
            score: Some(1.0),
        }],
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    };

    let assigned = assign_speakers_to_transcript(&transcript, &diarization).unwrap();

    assert_eq!(assigned.segments[0].speaker.as_deref(), Some("original"));
    assert!(!assigned.segments[0].attributes.contains_key("speakerScore"));
}

#[test]
fn speaker_assignment_supports_nearest_start_policy() {
    let segment = TranscriptSegmentContract::new(0, "hello")
        .with_time_range(Some(5.0), Some(6.0))
        .unwrap();
    let transcript = TranscriptionContract::new(vec![segment]);
    let diarization = SpeakerDiarizationResponse {
        accepted: true,
        operation: "diarize".to_string(),
        model_id: "fixture".to_string(),
        runtime: AudioRuntime::Imported,
        segments: vec![
            SpeakerSegmentPrediction {
                speaker: "speaker_far".to_string(),
                start_seconds: 0.0,
                end_seconds: 1.0,
                score: Some(1.0),
            },
            SpeakerSegmentPrediction {
                speaker: "speaker_near".to_string(),
                start_seconds: 4.9,
                end_seconds: 5.1,
                score: Some(0.1),
            },
        ],
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    };

    let assigned = assign_speakers_to_transcript_with_policy(
        &transcript,
        &diarization,
        SpeakerTranscriptAssignmentPolicy::NearestStart,
    )
    .unwrap();

    assert_eq!(
        assigned.segments[0].speaker.as_deref(),
        Some("speaker_near")
    );
}

#[test]
fn speaker_assignment_strict_contained_uses_unknown_without_match() {
    let segment = TranscriptSegmentContract::new(0, "hello")
        .with_time_range(Some(0.0), Some(2.0))
        .unwrap();
    let transcript = TranscriptionContract::new(vec![segment]);
    let diarization = SpeakerDiarizationResponse {
        accepted: true,
        operation: "diarize".to_string(),
        model_id: "fixture".to_string(),
        runtime: AudioRuntime::Imported,
        segments: vec![SpeakerSegmentPrediction {
            speaker: "partial".to_string(),
            start_seconds: 0.5,
            end_seconds: 1.5,
            score: Some(1.0),
        }],
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    };

    let assigned = assign_speakers_to_transcript_with_policy(
        &transcript,
        &diarization,
        SpeakerTranscriptAssignmentPolicy::StrictContained,
    )
    .unwrap();

    assert_eq!(assigned.segments[0].speaker.as_deref(), Some("unknown"));
}

#[test]
fn speaker_diarization_options_accept_flat_camel_case_json() {
    let options: SpeakerDiarizationOptions = serde_json::from_value(serde_json::json!({
        "modelId": "pyannote-community-1",
        "pyannoteModelBundle": "/models/pyannote",
        "pyannoteManifestFile": "manifest.json",
        "returnSpeakerEmbeddings": true,
        "minSpeakers": 1,
        "maxSpeakers": 3,
        "assignmentPolicy": "nearestStart"
    }))
    .unwrap();

    assert_eq!(options.model_id, "pyannote-community-1");
    assert!(options.is_pyannote_model());
    assert_eq!(
        options.pyannote_model_bundle.as_deref(),
        Some(Path::new("/models/pyannote"))
    );
    assert_eq!(
        options.pyannote_manifest_file.as_deref(),
        Some("manifest.json")
    );
    assert!(options.return_speaker_embeddings);
    assert_eq!(options.min_speakers, Some(1));
    assert_eq!(options.max_speakers, Some(3));
    assert_eq!(
        options.assignment_policy,
        SpeakerTranscriptAssignmentPolicy::NearestStart
    );
    options.validate().unwrap();
}

#[test]
fn speaker_diarization_options_validate_owned_bounds_and_model_options() {
    let invalid_bounds = SpeakerDiarizationOptions {
        min_speakers: Some(3),
        max_speakers: Some(2),
        ..SpeakerDiarizationOptions::default()
    };
    assert!(invalid_bounds
        .validate()
        .unwrap_err()
        .to_string()
        .contains("invalid_request"));

    let invalid_embedding_dimension = SpeakerDiarizationOptions {
        speaker_embedding_dimension: Some(0),
        ..SpeakerDiarizationOptions::default()
    };
    assert!(invalid_embedding_dimension
        .validate()
        .unwrap_err()
        .to_string()
        .contains("speaker_embedding_dimension"));

    let pyannote_with_onnx_bundle = SpeakerDiarizationOptions {
        model_id: "pyannote/speaker-diarization-community-1".to_string(),
        speaker_embedding_model_bundle: Some(PathBuf::from("speaker-model")),
        ..SpeakerDiarizationOptions::default()
    };
    assert!(pyannote_with_onnx_bundle
        .validate()
        .unwrap_err()
        .to_string()
        .contains("pyannote_model_bundle"));
}

#[test]
fn speaker_assignment_assigns_words_and_majority_segment_speaker() {
    let mut segment = TranscriptSegmentContract::new(0, "one two three")
        .with_time_range(Some(0.0), Some(0.9))
        .unwrap();
    for (text, start, end) in [("one", 0.0, 0.3), ("two", 0.3, 0.6), ("three", 0.6, 0.9)] {
        segment
            .push_word(
                serde_json::from_value(serde_json::json!({
                    "text": text,
                    "startSeconds": start,
                    "endSeconds": end
                }))
                .unwrap(),
            )
            .unwrap();
    }
    let transcript = TranscriptionContract::new(vec![segment]);
    let diarization = SpeakerDiarizationResponse {
        accepted: true,
        operation: "diarize".to_string(),
        model_id: "fixture".to_string(),
        runtime: AudioRuntime::Imported,
        segments: vec![
            SpeakerSegmentPrediction {
                speaker: "speaker_0".to_string(),
                start_seconds: 0.0,
                end_seconds: 0.65,
                score: Some(0.9),
            },
            SpeakerSegmentPrediction {
                speaker: "speaker_1".to_string(),
                start_seconds: 0.65,
                end_seconds: 1.0,
                score: Some(0.9),
            },
        ],
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    };

    let assigned = assign_speakers_to_transcript_with_policy(
        &transcript,
        &diarization,
        SpeakerTranscriptAssignmentPolicy::Majority,
    )
    .unwrap();

    assert_eq!(
        assigned.segments[0].words()[0].speaker.as_deref(),
        Some("speaker_0")
    );
    assert_eq!(
        assigned.segments[0].words()[1].speaker.as_deref(),
        Some("speaker_0")
    );
    assert_eq!(
        assigned.segments[0].words()[2].speaker.as_deref(),
        Some("speaker_1")
    );
    assert_eq!(assigned.segments[0].speaker.as_deref(), Some("speaker_0"));
}

#[test]
fn speaker_assignment_word_policy_variants_match_transcription_contract() {
    let mut segment = TranscriptSegmentContract::new(0, "hello")
        .with_time_range(Some(0.2), Some(0.8))
        .unwrap();
    segment
        .push_word(
            serde_json::from_value(serde_json::json!({
                "text": "hello",
                "startSeconds": 0.2,
                "endSeconds": 0.8
            }))
            .unwrap(),
        )
        .unwrap();
    let transcript = TranscriptionContract::new(vec![segment]);
    let diarization = SpeakerDiarizationResponse {
        accepted: true,
        operation: "diarize".to_string(),
        model_id: "fixture".to_string(),
        runtime: AudioRuntime::Imported,
        segments: vec![
            SpeakerSegmentPrediction {
                speaker: "speaker_far".to_string(),
                start_seconds: 0.0,
                end_seconds: 0.1,
                score: Some(0.9),
            },
            SpeakerSegmentPrediction {
                speaker: "speaker_near".to_string(),
                start_seconds: 0.18,
                end_seconds: 0.3,
                score: Some(0.9),
            },
        ],
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    };

    let nearest = assign_speakers_to_transcript_with_policy(
        &transcript,
        &diarization,
        SpeakerTranscriptAssignmentPolicy::NearestStart,
    )
    .unwrap();
    assert_eq!(
        nearest.segments[0].words()[0].speaker.as_deref(),
        Some("speaker_near")
    );

    let strict = assign_speakers_to_transcript_with_policy(
        &transcript,
        &diarization,
        SpeakerTranscriptAssignmentPolicy::StrictContained,
    )
    .unwrap();
    assert!(strict.segments[0].words()[0].speaker.is_none());
    assert!(strict.segments[0].speaker.is_none());
}

#[derive(Debug, Clone)]
struct MeanSignSpeakerEmbedder {
    model: SpeakerEmbeddingModel,
}

impl MeanSignSpeakerEmbedder {
    fn new() -> Self {
        Self {
            model: test_model(),
        }
    }
}

impl SpeakerEmbeddingExtractor for MeanSignSpeakerEmbedder {
    fn model_info(&self) -> SpeakerEmbeddingModel {
        self.model.clone()
    }

    fn embed_speaker(&mut self, audio: &SpeakerAudio<'_>) -> Result<SpeakerEmbedding> {
        let mean = audio.samples().iter().sum::<f32>() / audio.samples().len() as f32;
        if mean >= 0.0 {
            SpeakerEmbedding::new(vec![1.0, 0.0], self.model_info(), audio.sample_rate())
        } else {
            SpeakerEmbedding::new(vec![0.0, 1.0], self.model_info(), audio.sample_rate())
        }
    }
}

#[derive(Debug, Clone)]
struct FixedVad {
    spans: Vec<SpeechSpan>,
}

impl VoiceActivityDetector for FixedVad {
    fn provider_id(&self) -> &str {
        "fixed"
    }

    fn detect_speech(
        &mut self,
        _audio: &SpeakerAudio<'_>,
        _options: &VadOptions,
    ) -> Result<VadResponse> {
        Ok(VadResponse {
            segments: self.spans.clone(),
            diagnostics: Vec::new(),
        })
    }
}

fn sine(freq_hz: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
    let samples = (sample_rate as f32 * seconds) as usize;
    (0..samples)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5
        })
        .collect()
}

fn two_profile_speech_fixture(sample_rate: u32) -> Vec<f32> {
    let mut samples = sine(220.0, sample_rate, 0.35);
    samples.extend(vec![0.0; (sample_rate as f32 * 0.12) as usize]);
    samples.extend(sine(1_200.0, sample_rate, 0.35));
    samples
}

fn speaker_segment(
    speaker: &str,
    start_seconds: f32,
    end_seconds: f32,
) -> SpeakerSegmentPrediction {
    SpeakerSegmentPrediction {
        speaker: speaker.to_string(),
        start_seconds,
        end_seconds,
        score: Some(0.8),
    }
}

fn test_model() -> SpeakerEmbeddingModel {
    SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::SpeechBrain, "spkrec", "1", 2).unwrap()
}

fn embedding(values: impl Into<Vec<f32>>) -> SpeakerEmbedding {
    SpeakerEmbedding::new(values, test_model(), 16_000).unwrap()
}

fn id(value: &str) -> SpeakerId {
    SpeakerId::new(value).unwrap()
}

fn label(value: &str) -> SpeakerLabel {
    SpeakerLabel::new(value).unwrap()
}

#[test]
fn speaker_audio_validates_samples_rate_channels_and_duration() {
    assert!(SpeakerAudio::mono(&[], 16_000).is_err());
    assert!(SpeakerAudio::mono(&[0.0], 0).is_err());
    assert!(SpeakerAudio::mono(&[f32::NAN], 16_000).is_err());
    assert!(SpeakerAudio::interleaved(&[0.0, 1.0, 0.0], 16_000, 2).is_err());
    assert!(SpeakerAudio::mono(&[0.0; 160], 16_000)
        .unwrap()
        .duration_bounds(Some(0.02), None)
        .is_err());
}

#[test]
fn speaker_audio_accepts_valid_speech_spans_and_frame_mono_conversion() {
    let samples = [0.1, -0.1, 0.2, -0.2];
    let audio = SpeakerAudio::interleaved(&samples, 16_000, 2)
        .unwrap()
        .with_speech_spans(vec![SpeechSpan::new(0.0, 0.0001, 0.9).unwrap()])
        .unwrap();

    assert_eq!(audio.channels(), 2);
    assert_eq!(audio.speech_spans().len(), 1);
    assert_eq!(audio.to_mono().unwrap(), vec![0.0, 0.0]);
    assert!(SpeakerAudio::mono(&[0.0; 100], 1_000)
        .unwrap()
        .with_speech_spans(vec![SpeechSpan::new(0.0, 1.0, 0.5).unwrap()])
        .is_err());
}

#[test]
fn ids_labels_models_spans_and_options_validate_inputs() {
    assert!(SpeakerId::new(" ").is_err());
    assert!(SpeakerLabel::new("").is_err());
    assert!(SpeakerEmbeddingModel::new(
        SpeakerEmbeddingModelFamily::Custom(" ".to_string()),
        "custom",
        "1",
        2,
    )
    .is_err());
    assert!(
        SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::EcapaTdnn, "", "1", 2).is_err()
    );
    assert!(SpeechSpan::new(0.2, 0.1, 0.5).is_err());
    assert!(SpeechSpan::new(0.0, 0.1, f32::NAN).is_err());
    assert!(SpeakerIdentificationOptions::new(f32::NAN).is_err());
    assert!(SpeakerIdentificationOptions::new(0.8)
        .unwrap()
        .min_margin(f32::NAN)
        .is_err());
    assert!(SpeakerIdentificationOptions {
        max_results: 0,
        ..SpeakerIdentificationOptions::default()
    }
    .validate()
    .is_err());
}

#[test]
fn speaker_embedding_carries_model_identity() {
    let model = SpeakerEmbeddingModel::new(
        SpeakerEmbeddingModelFamily::EcapaTdnn,
        "ecapa",
        "2026-01",
        2,
    )
    .unwrap();
    let embedding = SpeakerEmbedding::new(vec![2.0, 0.0], model.clone(), 16_000).unwrap();

    assert_eq!(embedding.model(), &model);
    assert_eq!(embedding.values(), &[1.0, 0.0]);
    assert!(SpeakerEmbedding::new(vec![1.0, 0.0, 0.0], model, 16_000).is_err());
}

#[derive(Debug, Clone)]
struct MockSpeakerEmbeddingProvider {
    model: SpeakerEmbeddingModel,
}

impl SpeakerEmbeddingProvider for MockSpeakerEmbeddingProvider {
    fn provider_id(&self) -> &str {
        "mock-speaker-embedding"
    }

    fn embed(&mut self, request: SpeakerEmbeddingRequest<'_>) -> Result<SpeakerEmbeddingResponse> {
        Ok(SpeakerEmbeddingResponse {
            model_id: self.model.name.clone(),
            runtime: AudioRuntime::Onnx,
            embedding: SpeakerEmbedding::new(
                vec![3.0, 4.0],
                self.model.clone(),
                request.audio.sample_rate(),
            )?,
            diagnostics: vec!["speakerEmbeddingProvider=mock".to_string()],
        })
    }
}

#[test]
fn speaker_embedding_provider_trait_returns_runtime_response() {
    let samples = [0.1_f32; 160];
    let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();
    let mut provider = MockSpeakerEmbeddingProvider {
        model: test_model(),
    };

    let response = provider
        .embed(SpeakerEmbeddingRequest { audio })
        .expect("mock embedding");

    assert_eq!(provider.provider_id(), "mock-speaker-embedding");
    assert_eq!(response.model_id, "spkrec");
    assert_eq!(response.runtime, AudioRuntime::Onnx);
    assert_eq!(response.diagnostics, ["speakerEmbeddingProvider=mock"]);
    assert_eq!(response.embedding.values(), &[0.6, 0.8]);
}

#[test]
fn speaker_embedding_rejects_model_mismatch_and_invalid_stored_values() {
    let left = embedding([1.0, 0.0]);
    let other_model =
        SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::XVector, "xvector", "1", 2)
            .unwrap();
    let right = SpeakerEmbedding::new(vec![1.0, 0.0], other_model, 16_000).unwrap();

    assert!(left.cosine_similarity(&right).is_err());

    let mut stored = left.clone();
    stored.values = vec![2.0, 0.0];
    assert!(stored.validate_stored().is_err());
    stored.values = vec![f32::NAN, 0.0];
    assert!(stored.validate_stored().is_err());
    stored.values = vec![1.0, 0.0];
    stored.sample_rate = 0;
    assert!(stored.validate_stored().is_err());
}

#[test]
fn library_identifies_with_score_and_margin_thresholds() {
    let model = test_model();
    let alice = SpeakerEmbedding::new(vec![1.0, 0.0], model.clone(), 16_000).unwrap();
    let bob = SpeakerEmbedding::new(vec![0.0, 1.0], model.clone(), 16_000).unwrap();
    let query = SpeakerEmbedding::new(vec![0.99, 0.01], model, 16_000).unwrap();
    let mut library = SpeakerLibrary::new();
    library
        .add_profile(
            SpeakerProfile::new(id("alice"), label("Alice"))
                .with_embedding(alice)
                .unwrap(),
        )
        .unwrap();
    library
        .add_profile(
            SpeakerProfile::new(id("bob"), label("Bob"))
                .with_embedding(bob)
                .unwrap(),
        )
        .unwrap();

    let result = library
        .identify(&query, &SpeakerIdentificationOptions::default())
        .unwrap();

    assert_eq!(result.best_match.unwrap().speaker_id.as_str(), "alice");
    assert!(!result.unknown);
    assert!(result.margin.unwrap() > 0.9);
}

#[test]
fn library_orders_matches_applies_max_results_and_confidence() {
    let mut library = SpeakerLibrary::new();
    for (speaker_id, values) in [
        ("c", vec![0.0, 1.0]),
        ("a", vec![1.0, 0.0]),
        ("b", vec![1.0, 0.0]),
    ] {
        library
            .add_profile(
                SpeakerProfile::new(id(speaker_id), label(speaker_id))
                    .with_embedding(embedding(values))
                    .unwrap(),
            )
            .unwrap();
    }
    let options = SpeakerIdentificationOptions {
        min_score: -1.0,
        min_margin: None,
        max_results: 2,
        unknown_policy: UnknownSpeakerPolicy::ReturnUnknown,
    };

    let result = library.identify(&embedding([1.0, 0.0]), &options).unwrap();

    assert_eq!(result.ranked_matches.len(), 2);
    assert_eq!(result.ranked_matches[0].speaker_id.as_str(), "a");
    assert_eq!(result.ranked_matches[1].speaker_id.as_str(), "b");
    assert_eq!(result.ranked_matches[0].confidence, SpeakerConfidence::Low);
    assert_eq!(result.margin, Some(0.0));
}

#[test]
fn library_marks_unknown_when_margin_fails() {
    let model = SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::XVector, "xvector", "1", 2)
        .unwrap();
    let mut library = SpeakerLibrary::new();
    library
        .add_profile(
            SpeakerProfile::new(id("a"), label("A"))
                .with_embedding(
                    SpeakerEmbedding::new(vec![1.0, 0.0], model.clone(), 16_000).unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
    library
        .add_profile(
            SpeakerProfile::new(id("b"), label("B"))
                .with_embedding(
                    SpeakerEmbedding::new(vec![0.99, 0.01], model.clone(), 16_000).unwrap(),
                )
                .unwrap(),
        )
        .unwrap();
    let query = SpeakerEmbedding::new(vec![1.0, 0.0], model, 16_000).unwrap();

    let result = library
        .identify(&query, &SpeakerIdentificationOptions::default())
        .unwrap();

    assert!(result.best_match.is_none());
    assert!(result.unknown);
}

#[test]
fn library_no_match_policy_suppresses_unknown_flag() {
    let mut library = SpeakerLibrary::new();
    library
        .add_profile(
            SpeakerProfile::new(id("alice"), label("Alice"))
                .with_embedding(embedding([1.0, 0.0]))
                .unwrap(),
        )
        .unwrap();
    let options = SpeakerIdentificationOptions {
        min_score: 1.0,
        min_margin: Some(0.1),
        max_results: 1,
        unknown_policy: UnknownSpeakerPolicy::NoMatch,
    };

    let result = library.identify(&embedding([1.0, 0.0]), &options).unwrap();

    assert!(result.best_match.is_none());
    assert!(!result.unknown);
}

#[test]
fn library_rejects_duplicate_ids_unknown_ids_empty_profiles_and_mixed_models() {
    let mut library = SpeakerLibrary::new();
    library
        .add_profile(
            SpeakerProfile::new(id("alice"), label("Alice"))
                .with_embedding(embedding([1.0, 0.0]))
                .unwrap(),
        )
        .unwrap();
    assert!(library
        .add_profile(
            SpeakerProfile::new(id("alice"), label("Duplicate"))
                .with_embedding(embedding([0.0, 1.0]))
                .unwrap()
        )
        .is_err());
    assert!(library
        .add_profile(SpeakerProfile::new(id("empty"), label("Empty")))
        .is_err());
    assert!(library
        .add_embedding(&id("missing"), embedding([1.0, 0.0]))
        .is_err());

    let other_model =
        SpeakerEmbeddingModel::new(SpeakerEmbeddingModelFamily::Pyannote, "pyannote", "1", 2)
            .unwrap();
    assert!(library
        .add_embedding(
            &id("alice"),
            SpeakerEmbedding::new(vec![1.0, 0.0], other_model, 16_000).unwrap(),
        )
        .is_err());
}

#[test]
fn spectral_speaker_embedder_enrolls_and_round_trips_snapshot() {
    let sample_rate = 8_000;
    let audio = sine(220.0, sample_rate, 0.4);
    let speaker_audio = SpeakerAudio::mono(&audio, sample_rate).unwrap();
    let mut embedder =
        SpectralSpeakerEmbedder::new(SpectralEmbeddingConfig::new(512, 256, 8).unwrap()).unwrap();
    let mut library = SpeakerLibrary::new();
    library
        .enroll(
            id("speaker-a"),
            label("Speaker A"),
            &speaker_audio,
            &mut embedder,
        )
        .unwrap();
    let json = library.to_json_string().unwrap();
    let restored = SpeakerLibrary::from_json_str(&json).unwrap();

    assert_eq!(restored.len(), 1);
    assert_eq!(
        restored.embedding_model().unwrap().family,
        SpeakerEmbeddingModelFamily::SpectralBaseline
    );
}

#[test]
fn snapshot_loading_rejects_bad_versions_duplicates_models_and_values() {
    let mut library = SpeakerLibrary::new();
    library
        .add_profile(
            SpeakerProfile::new(id("alice"), label("Alice"))
                .with_embedding(embedding([1.0, 0.0]))
                .unwrap(),
        )
        .unwrap();
    let mut snapshot = library.to_snapshot().unwrap();
    snapshot.version = 999;
    assert!(SpeakerLibrary::from_snapshot(snapshot).is_err());

    let mut duplicate = library.to_snapshot().unwrap();
    duplicate.profiles.push(duplicate.profiles[0].clone());
    assert!(SpeakerLibrary::from_snapshot(duplicate).is_err());

    let bad_json = r#"{
        "version": 1,
        "embedding_model": {
            "family": "SpeechBrain",
            "name": "spkrec",
            "version": "1",
            "dimensions": 2
        },
        "profiles": [{
            "id": "alice",
            "label": "Alice",
            "embeddings": [{
                "values": [2.0, 0.0],
                "model": {
                    "family": "SpeechBrain",
                    "name": "spkrec",
                    "version": "1",
                    "dimensions": 2
                },
                "sample_rate": 16000
            }],
            "metadata": {}
        }]
    }"#;
    assert!(SpeakerLibrary::from_json_str(bad_json).is_err());

    let model_mismatch_json = r#"{
        "version": 1,
        "embedding_model": {
            "family": "SpeechBrain",
            "name": "spkrec",
            "version": "1",
            "dimensions": 2
        },
        "profiles": [{
            "id": "alice",
            "label": "Alice",
            "embeddings": [{
                "values": [1.0, 0.0],
                "model": {
                    "family": "SpeechBrain",
                    "name": "spkrec",
                    "version": "2",
                    "dimensions": 2
                },
                "sample_rate": 16000
            }],
            "metadata": {}
        }]
    }"#;
    assert!(SpeakerLibrary::from_json_str(model_mismatch_json).is_err());
}

#[test]
fn speaker_library_persistence_updates_profiles_and_removes_an_empty_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("library.json");
    let mut library = SpeakerLibrary::new();
    library
        .add_profile(
            SpeakerProfile::new(id("alice"), label("Alice"))
                .with_embedding(embedding([1.0, 0.0]))
                .unwrap(),
        )
        .unwrap();

    library.save(&path).unwrap();
    let mut loaded = SpeakerLibrary::load(&path).unwrap();
    loaded
        .update_profile(
            &id("alice"),
            SpeakerProfileUpdate {
                label: Some("Alice Updated".to_string()),
                metadata: Some(BTreeMap::from([(
                    "status".to_string(),
                    "draft".to_string(),
                )])),
            },
        )
        .unwrap();
    let profile = loaded.profile(&id("alice")).unwrap();
    assert_eq!(profile.label().as_str(), "Alice Updated");
    assert_eq!(profile.embeddings().len(), 1);
    assert_eq!(
        profile.metadata_map().get("status"),
        Some(&"draft".to_string())
    );

    loaded.delete_profile(&id("alice")).unwrap();
    loaded.save(&path).unwrap();
    assert_eq!(SpeakerLibrary::load_if_present(&path).unwrap(), None);
}

#[test]
fn speaker_library_rejects_unknown_snapshot_fields() {
    let mut library = SpeakerLibrary::new();
    library
        .add_profile(
            SpeakerProfile::new(id("alice"), label("Alice"))
                .with_embedding(embedding([1.0, 0.0]))
                .unwrap(),
        )
        .unwrap();
    let mut snapshot: serde_json::Value =
        serde_json::from_str(&library.to_json_string().unwrap()).unwrap();
    snapshot["speaker_trace"] = serde_json::json!({"files": []});

    let error = SpeakerLibrary::from_json_str(&snapshot.to_string())
        .expect_err("unknown product-owned fields must not enter the canonical snapshot");

    assert!(error.to_string().contains("unknown field `speaker_trace`"));
}

#[test]
fn onnx_embedder_exposes_model_metadata_and_fails_until_runtime_exists() {
    let mut embedder = OnnxSpeakerEmbedder::new("speaker.onnx", test_model()).unwrap();
    let samples = [0.1_f32; 160];
    let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

    assert_eq!(embedder.model_path(), Path::new("speaker.onnx"));
    assert_eq!(embedder.model_info(), test_model());
    assert!(embedder.embed_speaker(&audio).is_err());
}

#[test]
fn onnx_config_validation_rejects_missing_bundle_path() {
    let err = OnnxSpeakerEmbedder::from_config(OnnxSpeakerEmbeddingConfig {
        embedding_dimension: 2,
        sample_rate: 16_000,
        ..OnnxSpeakerEmbeddingConfig::default()
    })
    .unwrap_err();

    assert!(err.to_string().contains("setup_error"));
}

#[cfg(not(feature = "onnx"))]
#[test]
fn onnx_runtime_unavailable_without_feature_returns_typed_error() {
    let (dir, model_path) = temp_onnx_model();
    let err = OnnxSpeakerEmbedder::from_config(OnnxSpeakerEmbeddingConfig {
        bundle_path: dir,
        embedding_dimension: 2,
        sample_rate: 16_000,
        ..OnnxSpeakerEmbeddingConfig::default()
    })
    .unwrap_err();
    let _ = std::fs::remove_file(model_path);

    assert!(err.to_string().contains("unsupported_runtime"));
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone)]
struct MockOnnxRunner {
    metadata: runtime_onnx::OnnxSessionMetadata,
    outputs: Vec<runtime_onnx::OnnxNamedTensor>,
}

#[cfg(feature = "onnx")]
impl runtime_onnx::OnnxRunner for MockOnnxRunner {
    fn metadata(&self) -> runtime_onnx::Result<runtime_onnx::OnnxSessionMetadata> {
        Ok(self.metadata.clone())
    }

    fn run(
        &mut self,
        inputs: Vec<runtime_onnx::OnnxNamedTensor>,
    ) -> runtime_onnx::Result<Vec<runtime_onnx::OnnxNamedTensor>> {
        assert_eq!(inputs.len(), 1);
        Ok(self.outputs.clone())
    }
}

#[cfg(feature = "onnx")]
fn mock_onnx_metadata(
    input_dimensions: Vec<runtime_onnx::OnnxDimension>,
    output_dimensions: Vec<runtime_onnx::OnnxDimension>,
) -> runtime_onnx::OnnxSessionMetadata {
    runtime_onnx::OnnxSessionMetadata {
        inputs: vec![runtime_onnx::OnnxIoInfo {
            name: "waveform".to_string(),
            element_type: Some(runtime_onnx::OnnxTensorElementType::F32),
            dimensions: input_dimensions,
        }],
        outputs: vec![runtime_onnx::OnnxIoInfo {
            name: "embedding".to_string(),
            element_type: Some(runtime_onnx::OnnxTensorElementType::F32),
            dimensions: output_dimensions,
        }],
    }
}

#[cfg(feature = "onnx")]
fn mock_onnx_runner(output_shape: Vec<usize>) -> MockOnnxRunner {
    MockOnnxRunner {
        metadata: mock_onnx_metadata(
            vec![
                runtime_onnx::OnnxDimension::Fixed(1),
                runtime_onnx::OnnxDimension::Unknown,
            ],
            output_shape
                .iter()
                .copied()
                .map(runtime_onnx::OnnxDimension::Fixed)
                .collect(),
        ),
        outputs: vec![runtime_onnx::OnnxNamedTensor {
            name: "embedding".to_string(),
            tensor: runtime_onnx::OnnxTensorValue::F32(
                runtime_onnx::OnnxF32Tensor::new(
                    output_shape.clone(),
                    if output_shape == vec![2] || output_shape == vec![1, 2] {
                        vec![3.0, 4.0]
                    } else {
                        vec![1.0; output_shape.iter().product()]
                    },
                )
                .unwrap(),
            ),
        }],
    }
}

#[cfg(feature = "onnx")]
fn onnx_config_for_temp_model() -> (OnnxSpeakerEmbeddingConfig, PathBuf) {
    let (dir, _model_path) = temp_onnx_model();
    (
        OnnxSpeakerEmbeddingConfig {
            bundle_path: dir.clone(),
            input_name: Some("waveform".to_string()),
            output_name: Some("embedding".to_string()),
            embedding_dimension: 2,
            sample_rate: 16_000,
            ..OnnxSpeakerEmbeddingConfig::default()
        },
        dir,
    )
}

fn temp_onnx_model() -> (PathBuf, PathBuf) {
    let unique = format!(
        "audio-analysis-speakers-onnx-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.onnx");
    std::fs::write(&model_path, b"fixture").unwrap();
    (dir, model_path)
}

fn optional_smoke_env_value(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
}

#[test]
fn onnx_smoke_env_values_accept_model_file_and_io_names() {
    assert_eq!(
        optional_smoke_env_value(Some("voxceleb_resnet34_LM.onnx".to_string())).as_deref(),
        Some("voxceleb_resnet34_LM.onnx")
    );
    assert_eq!(
        optional_smoke_env_value(Some("waveform".to_string())).as_deref(),
        Some("waveform")
    );
    assert_eq!(
        optional_smoke_env_value(Some("embedding".to_string())).as_deref(),
        Some("embedding")
    );
    assert_eq!(optional_smoke_env_value(Some("   ".to_string())), None);
    assert_eq!(optional_smoke_env_value(None), None);
}

#[cfg(feature = "onnx")]
#[test]
fn mock_onnx_rank1_output_produces_normalized_embedding() {
    let (config, dir) = onnx_config_for_temp_model();
    let mut embedder = OnnxSpeakerEmbedder::from_runner(config, mock_onnx_runner(vec![2])).unwrap();
    let samples = [0.1_f32; 160];
    let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

    let response = embedder.embed(SpeakerEmbeddingRequest { audio }).unwrap();

    assert_eq!(response.runtime, AudioRuntime::Onnx);
    assert_eq!(response.embedding.values(), &[0.6, 0.8]);
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerEmbeddingProvider=onnx"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerEmbeddingSelectedInputName=waveform"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerEmbeddingSelectedInputRank=2"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerEmbeddingSelectedOutputName=embedding"));
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(feature = "onnx")]
#[test]
fn mock_onnx_rank2_output_produces_normalized_embedding() {
    let (config, dir) = onnx_config_for_temp_model();
    let mut embedder =
        OnnxSpeakerEmbedder::from_runner(config, mock_onnx_runner(vec![1, 2])).unwrap();
    let samples = [0.1_f32; 160];
    let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

    let embedding = embedder.embed_speaker(&audio).unwrap();

    assert_eq!(embedding.values(), &[0.6, 0.8]);
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(feature = "onnx")]
#[test]
fn mock_onnx_feature_tensor_input_produces_embedding() {
    let (mut config, dir) = onnx_config_for_temp_model();
    config.input_name = Some("feats".to_string());
    config.output_name = Some("embs".to_string());
    config.embedding_dimension = 256;
    let runner = MockOnnxRunner {
        metadata: runtime_onnx::OnnxSessionMetadata {
            inputs: vec![runtime_onnx::OnnxIoInfo {
                name: "feats".to_string(),
                element_type: Some(runtime_onnx::OnnxTensorElementType::F32),
                dimensions: vec![
                    runtime_onnx::OnnxDimension::Symbolic("B".to_string()),
                    runtime_onnx::OnnxDimension::Symbolic("T".to_string()),
                    runtime_onnx::OnnxDimension::Fixed(80),
                ],
            }],
            outputs: vec![runtime_onnx::OnnxIoInfo {
                name: "embs".to_string(),
                element_type: Some(runtime_onnx::OnnxTensorElementType::F32),
                dimensions: vec![
                    runtime_onnx::OnnxDimension::Symbolic("B".to_string()),
                    runtime_onnx::OnnxDimension::Fixed(256),
                ],
            }],
        },
        outputs: vec![runtime_onnx::OnnxNamedTensor {
            name: "embs".to_string(),
            tensor: runtime_onnx::OnnxTensorValue::F32(
                runtime_onnx::OnnxF32Tensor::new(vec![1, 256], vec![1.0; 256]).unwrap(),
            ),
        }],
    };
    let mut embedder = OnnxSpeakerEmbedder::from_runner(config, runner).unwrap();
    let samples = [0.1_f32; 160];
    let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

    let response = embedder.embed(SpeakerEmbeddingRequest { audio }).unwrap();

    assert_eq!(response.embedding.dimensions(), 256);
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerEmbeddingInputKind=feature"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerFeatureExtractor=fbank-logmel-cpu"));
    assert!(response
        .diagnostics
        .iter()
        .any(|item| item == "speakerFeatureBins=80"));
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(feature = "onnx")]
#[test]
fn onnx_speaker_input_selection_detects_waveform_and_feature_shapes() {
    let waveform_rank2 = SelectedOnnxIo {
        name: "waveform".to_string(),
        index: 0,
        dimensions: vec![
            runtime_onnx::OnnxDimension::Fixed(1),
            runtime_onnx::OnnxDimension::Unknown,
        ],
    };
    let waveform_rank3 = SelectedOnnxIo {
        name: "waveform".to_string(),
        index: 0,
        dimensions: vec![
            runtime_onnx::OnnxDimension::Fixed(1),
            runtime_onnx::OnnxDimension::Fixed(1),
            runtime_onnx::OnnxDimension::Unknown,
        ],
    };
    let feature_rank3 = SelectedOnnxIo {
        name: "feats".to_string(),
        index: 0,
        dimensions: vec![
            runtime_onnx::OnnxDimension::Symbolic("B".to_string()),
            runtime_onnx::OnnxDimension::Symbolic("T".to_string()),
            runtime_onnx::OnnxDimension::Fixed(80),
        ],
    };

    assert_eq!(
        onnx_speaker_input_kind(&waveform_rank2).unwrap(),
        OnnxSpeakerInputKind::Waveform
    );
    assert_eq!(
        onnx_speaker_input_kind(&waveform_rank3).unwrap(),
        OnnxSpeakerInputKind::Waveform
    );
    assert_eq!(
        onnx_speaker_input_kind(&feature_rank3).unwrap(),
        OnnxSpeakerInputKind::Feature {
            mel_bins: 80,
            fixed_frames: None
        }
    );
}

#[cfg(feature = "onnx")]
#[test]
fn feature_preprocessing_returns_finite_tensor_with_padded_short_audio() {
    let samples = [0.1_f32; 160];

    let (frame_count, values) = speaker_logmel_features(&samples, 16_000, 80, None).unwrap();

    assert_eq!(frame_count, 1);
    assert_eq!(values.len(), 80);
    assert!(values.iter().all(|value| value.is_finite()));
}

#[cfg(feature = "onnx")]
#[test]
fn fixed_feature_time_dimension_pads_and_truncates_deterministically() {
    let short = [0.1_f32; 160];
    let long = vec![0.1_f32; 400 + 160 * 5];

    let (padded_frames, padded_values) =
        speaker_logmel_features(&short, 16_000, 80, Some(3)).unwrap();
    let (truncated_frames, truncated_values) =
        speaker_logmel_features(&long, 16_000, 80, Some(2)).unwrap();

    assert_eq!(padded_frames, 3);
    assert_eq!(padded_values.len(), 3 * 80);
    assert!(padded_values[80..].iter().all(|value| *value == 0.0));
    assert_eq!(truncated_frames, 2);
    assert_eq!(truncated_values.len(), 2 * 80);
    assert!(truncated_values.iter().all(|value| value.is_finite()));
}

#[cfg(feature = "onnx")]
#[test]
fn mock_onnx_bad_output_shape_returns_model_output_mismatch() {
    let (config, dir) = onnx_config_for_temp_model();
    let mut embedder =
        OnnxSpeakerEmbedder::from_runner(config, mock_onnx_runner(vec![1, 1, 2])).unwrap();
    let samples = [0.1_f32; 160];
    let audio = SpeakerAudio::mono(&samples, 16_000).unwrap();

    let err = embedder.embed_speaker(&audio).unwrap_err();

    assert!(err.to_string().contains("model_output_mismatch"));
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(feature = "onnx")]
#[test]
fn onnx_embedder_rejects_non_matching_sample_rate() {
    let (config, dir) = onnx_config_for_temp_model();
    let mut embedder = OnnxSpeakerEmbedder::from_runner(config, mock_onnx_runner(vec![2])).unwrap();
    let samples = [0.1_f32; 160];
    let audio = SpeakerAudio::mono(&samples, 8_000).unwrap();

    let err = embedder.embed_speaker(&audio).unwrap_err();

    assert!(err.to_string().contains("invalid_request"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn diarizer_returns_unknown_segments_for_unenrolled_clusters() {
    let sample_rate = 8_000;
    let samples = sine(220.0, sample_rate, 0.25);
    let audio = SpeakerAudio::mono(&samples, sample_rate).unwrap();
    let embedder =
        SpectralSpeakerEmbedder::new(SpectralEmbeddingConfig::new(512, 256, 8).unwrap()).unwrap();
    let mut diarizer = WindowedSpeakerDiarizer::new(embedder, EnergyVoiceActivityDetector)
        .min_speaker_evidence_seconds(0.0)
        .unwrap()
        .vad_options(VadOptions {
            rms_threshold: 0.01,
            frame_seconds: 0.05,
            hop_seconds: 0.05,
            min_speech_seconds: 0.05,
            merge_gap_seconds: 0.02,
            ..VadOptions::unpadded()
        })
        .unwrap();

    let result = diarizer.diarize(&audio).unwrap();

    assert!(!result.segments.is_empty());
    assert!(matches!(
        result.segments[0].speaker,
        DiarizedSpeaker::Unknown(_)
    ));
}

#[test]
fn diarizer_maps_known_speakers_and_clusters_unknown_segments() {
    let mut samples = vec![0.4_f32; 100];
    samples.extend(vec![-0.4_f32; 100]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
    let mut library = SpeakerLibrary::new();
    library
        .add_profile(
            SpeakerProfile::new(id("known"), label("Known"))
                .with_embedding(embedding([1.0, 0.0]))
                .unwrap(),
        )
        .unwrap();
    let vad = FixedVad {
        spans: vec![
            SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
            SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
        ],
    };
    let mut options = SpeakerIdentificationOptions::new(0.8).unwrap();
    options.min_margin = None;
    let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
        .min_speaker_evidence_seconds(0.0)
        .unwrap()
        .library(library)
        .cluster_threshold(0.8)
        .unwrap();
    diarizer.identification_options = options;

    let result = diarizer.diarize(&audio).unwrap();

    assert_eq!(result.segments.len(), 2);
    assert_eq!(
        result.segments[0].speaker,
        DiarizedSpeaker::Known(id("known"))
    );
    assert_eq!(
        result.segments[1].speaker,
        DiarizedSpeaker::Unknown("speaker_0".to_string())
    );
}

#[test]
fn bounded_diarizer_produces_requested_two_unknown_speakers() {
    let mut samples = vec![0.4_f32; 100];
    samples.extend(vec![-0.4_f32; 100]);
    samples.extend(vec![0.4_f32; 100]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
    let vad = FixedVad {
        spans: vec![
            SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
            SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
            SpeechSpan::new(0.2, 0.3, 0.9).unwrap(),
        ],
    };
    let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
        .min_speaker_evidence_seconds(0.0)
        .unwrap()
        .cluster_threshold(0.95)
        .unwrap()
        .speaker_bounds(Some(2), Some(2))
        .unwrap();

    let result = diarizer.diarize(&audio).unwrap();
    let speakers = result
        .segments
        .iter()
        .filter_map(|segment| match &segment.speaker {
            DiarizedSpeaker::Unknown(label) => Some(label.clone()),
            DiarizedSpeaker::Known(_) | DiarizedSpeaker::Unresolved => None,
        })
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(speakers.len(), 2);
    assert!(speakers.contains("speaker_0"));
    assert!(speakers.contains("speaker_1"));
}

#[test]
fn bounded_diarizer_max_one_collapses_distinct_unknown_profiles() {
    let mut samples = vec![0.4_f32; 100];
    samples.extend(vec![-0.4_f32; 100]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
    let vad = FixedVad {
        spans: vec![
            SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
            SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
        ],
    };
    let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
        .min_speaker_evidence_seconds(0.0)
        .unwrap()
        .cluster_threshold(0.95)
        .unwrap()
        .speaker_bounds(None, Some(1))
        .unwrap();

    let result = diarizer.diarize(&audio).unwrap();
    assert!(result
        .segments
        .iter()
        .all(|segment| segment.speaker == DiarizedSpeaker::Unknown("speaker_0".to_string())));
}

#[test]
fn bounded_diarizer_min_above_window_count_is_saturated_by_windows() {
    let mut samples = vec![0.4_f32; 100];
    samples.extend(vec![-0.4_f32; 100]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
    let vad = FixedVad {
        spans: vec![
            SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
            SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
        ],
    };
    let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
        .min_speaker_evidence_seconds(0.0)
        .unwrap()
        .cluster_threshold(0.95)
        .unwrap()
        .speaker_bounds(Some(3), None)
        .unwrap();

    let result = diarizer.diarize(&audio).unwrap();
    let speakers = result
        .segments
        .iter()
        .filter_map(|segment| match &segment.speaker {
            DiarizedSpeaker::Unknown(label) => Some(label.clone()),
            DiarizedSpeaker::Known(_) | DiarizedSpeaker::Unresolved => None,
        })
        .collect::<std::collections::BTreeSet<_>>();

    assert!(speakers.len() <= 2);
}

#[test]
fn bounded_diarizer_invalid_bounds_return_invalid_argument() {
    for error in [
        WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), FixedVad { spans: vec![] })
            .speaker_bounds(Some(0), None)
            .unwrap_err(),
        WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), FixedVad { spans: vec![] })
            .speaker_bounds(None, Some(0))
            .unwrap_err(),
        WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), FixedVad { spans: vec![] })
            .speaker_bounds(Some(3), Some(2))
            .unwrap_err(),
    ] {
        assert!(matches!(error, DetectError::InvalidArgument(_)));
    }
}

#[test]
fn diarizer_without_bounds_keeps_threshold_clustering_behavior() {
    let mut samples = vec![0.4_f32; 100];
    samples.extend(vec![-0.4_f32; 100]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
    let vad = FixedVad {
        spans: vec![
            SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
            SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
        ],
    };
    let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad)
        .min_speaker_evidence_seconds(0.0)
        .unwrap()
        .cluster_threshold(0.95)
        .unwrap();

    let result = diarizer.diarize(&audio).unwrap();
    assert_eq!(
        result
            .segments
            .iter()
            .map(|segment| segment.speaker.clone())
            .collect::<Vec<_>>(),
        vec![
            DiarizedSpeaker::Unknown("speaker_0".to_string()),
            DiarizedSpeaker::Unknown("speaker_1".to_string()),
        ]
    );
}

#[test]
fn default_diarizer_leaves_short_spans_unresolved_instead_of_new_speaker() {
    // A long positive span followed by a short negative tail: with full
    // attribution the tail would become `speaker_1`.
    let mut samples = vec![0.4_f32; 1_000];
    samples.extend(vec![-0.4_f32; 180]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
    let vad = FixedVad {
        spans: vec![
            SpeechSpan::new(0.0, 1.0, 0.9).unwrap(),
            SpeechSpan::new(1.0, 1.18, 0.7).unwrap(),
        ],
    };
    let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad);
    assert_eq!(
        diarizer.min_speaker_evidence_seconds,
        DEFAULT_MIN_SPEAKER_EVIDENCE_SECONDS
    );

    let result = diarizer.diarize(&audio).unwrap();

    assert_eq!(
        result
            .segments
            .iter()
            .map(|segment| (
                segment.start_seconds,
                segment.end_seconds,
                segment.speaker.clone()
            ))
            .collect::<Vec<_>>(),
        vec![
            (0.0, 1.0, DiarizedSpeaker::Unknown("speaker_0".to_string())),
            (1.0, 1.18, DiarizedSpeaker::Unresolved),
        ]
    );
    assert_eq!(result.segments[1].score, 0.7);
}

#[test]
fn long_span_is_split_into_windows_so_a_turn_change_stays_separable() {
    let mut samples = vec![0.4_f32; 4_000];
    samples.extend(vec![-0.4_f32; 4_000]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
    let vad = FixedVad {
        spans: vec![SpeechSpan::new(0.0, 8.0, 0.9).unwrap()],
    };
    let mut diarizer = WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), vad);

    let result = diarizer.diarize(&audio).unwrap();

    // Three 2.67 s windows; the last two share a speaker and are rejoined.
    let segments = result
        .segments
        .iter()
        .map(|segment| (segment.end_seconds, segment.speaker.clone()))
        .collect::<Vec<_>>();
    assert_eq!(segments.len(), 2, "{segments:?}");
    assert!((segments[0].0 - 8.0 / 3.0).abs() < 1e-9);
    assert_eq!(
        segments[0].1,
        DiarizedSpeaker::Unknown("speaker_0".to_string())
    );
    assert_eq!(
        segments[1],
        (8.0, DiarizedSpeaker::Unknown("speaker_1".to_string()))
    );
}

#[test]
fn thin_cluster_is_folded_into_nearest_supported_speaker() {
    let mut samples = vec![0.4_f32; 2_500];
    samples.extend(vec![-0.4_f32; 500]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
    let spans = vec![
        SpeechSpan::new(0.0, 2.5, 0.9).unwrap(),
        SpeechSpan::new(2.5, 3.0, 0.9).unwrap(),
    ];
    let speakers = |diarizer: &mut WindowedSpeakerDiarizer<MeanSignSpeakerEmbedder, FixedVad>| {
        diarizer
            .diarize(&audio)
            .unwrap()
            .segments
            .into_iter()
            .map(|segment| segment.speaker)
            .collect::<Vec<_>>()
    };

    let mut folding = WindowedSpeakerDiarizer::new(
        MeanSignSpeakerEmbedder::new(),
        FixedVad {
            spans: spans.clone(),
        },
    );
    assert_eq!(
        speakers(&mut folding),
        vec![DiarizedSpeaker::Unknown("speaker_0".to_string()); 2]
    );

    let mut keeping =
        WindowedSpeakerDiarizer::new(MeanSignSpeakerEmbedder::new(), FixedVad { spans })
            .min_cluster_speech(0.0, 0.0)
            .unwrap();
    assert_eq!(
        speakers(&mut keeping),
        vec![
            DiarizedSpeaker::Unknown("speaker_0".to_string()),
            DiarizedSpeaker::Unknown("speaker_1".to_string()),
        ]
    );
}

#[test]
fn min_speaker_evidence_seconds_rejects_negative_and_non_finite() {
    for seconds in [-0.1, f64::NAN, f64::INFINITY] {
        let error = WindowedSpeakerDiarizer::new(
            MeanSignSpeakerEmbedder::new(),
            FixedVad { spans: vec![] },
        )
        .min_speaker_evidence_seconds(seconds)
        .unwrap_err();
        assert!(matches!(error, DetectError::InvalidArgument(_)));
    }
}

#[test]
fn native_diarization_baseline_returns_stable_speaker_segments() {
    let sample_rate = 8_000;
    let samples = sine(220.0, sample_rate, 0.30);
    let response = diarize_speaker_audio_baseline(&samples, sample_rate).unwrap();

    assert!(response.accepted);
    assert_eq!(response.runtime, AudioRuntime::Heuristic);
    assert!(!response.segments.is_empty());
    assert_eq!(response.segments[0].speaker, "speaker_0");
    assert!(response.segments[0].end_seconds > response.segments[0].start_seconds);
}

#[test]
fn native_diarization_baseline_splits_distinct_spectral_profiles() {
    let sample_rate = 8_000;
    let samples = two_profile_speech_fixture(sample_rate);
    let response = diarize_speaker_audio_baseline(&samples, sample_rate).unwrap();

    assert!(response.accepted);
    assert!(response.segments.len() >= 2, "{:?}", response.segments);
    assert_eq!(response.segments[0].speaker, "speaker_0");
    assert_eq!(response.segments[1].speaker, "speaker_1");
    assert!(response.segments.windows(2).all(|pair| {
        pair[0].start_seconds.is_finite()
            && pair[0].end_seconds.is_finite()
            && pair[0].end_seconds <= pair[1].start_seconds
    }));
}

#[test]
fn speaker_prediction_merge_combines_adjacent_same_speaker_only() {
    let merged = merge_speaker_predictions(
        vec![
            speaker_segment("speaker_0", 0.0, 0.4),
            speaker_segment("speaker_0", 0.43, 0.7),
            speaker_segment("speaker_1", 0.72, 1.0),
        ],
        0.05,
    )
    .unwrap();

    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].speaker, "speaker_0");
    assert_eq!(merged[0].start_seconds, 0.0);
    assert_eq!(merged[0].end_seconds, 0.7);
    assert_eq!(merged[1].speaker, "speaker_1");
}

#[test]
fn speaker_prediction_merge_preserves_same_speaker_after_large_gap() {
    let merged = merge_speaker_predictions(
        vec![
            speaker_segment("speaker_0", 0.0, 0.4),
            speaker_segment("speaker_0", 0.6, 0.9),
        ],
        0.05,
    )
    .unwrap();

    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].speaker, "speaker_0");
    assert_eq!(merged[1].speaker, "speaker_0");
}

#[test]
#[ignore]
fn native_diarization_baseline_smoke_when_requested() {
    if std::env::var("RUN_NATIVE_DIARIZATION_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping native diarization smoke; set RUN_NATIVE_DIARIZATION_TESTS=1");
        return;
    }
    let audio_path = std::env::var_os("DIARIZATION_AUDIO_PATH")
        .map(std::path::PathBuf::from)
        .expect("DIARIZATION_AUDIO_PATH is required");
    let mut reader = hound::WavReader::open(&audio_path)
        .unwrap_or_else(|error| panic!("failed to open WAV `{}`: {error}", audio_path.display()));
    let spec = reader.spec();
    assert!(spec.sample_rate > 0, "WAV sample rate must be non-zero");
    assert!(spec.channels > 0, "WAV channel count must be non-zero");
    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => {
            assert_eq!(spec.bits_per_sample, 32, "float WAV must be 32-bit");
            reader
                .samples::<f32>()
                .map(|sample| sample.expect("failed to read float WAV sample"))
                .collect::<Vec<_>>()
        }
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.expect("failed to read int WAV sample") as f32 / 32_768.0)
            .collect::<Vec<_>>(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|sample| {
                let scale = 2_f32.powi(spec.bits_per_sample as i32 - 1);
                sample.expect("failed to read int WAV sample") as f32 / scale
            })
            .collect::<Vec<_>>(),
    };
    let channels = spec.channels as usize;
    let samples = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect::<Vec<_>>()
    };
    let response = diarize_speaker_audio_baseline(&samples, spec.sample_rate).unwrap();
    assert!(response.accepted);
    assert!(!response.segments.is_empty());
    assert!(response
        .segments
        .iter()
        .all(|segment| segment.end_seconds > segment.start_seconds));
}

#[cfg(feature = "onnx")]
#[test]
#[ignore]
fn onnx_speaker_embedding_smoke_when_requested() {
    if std::env::var("RUN_NATIVE_SPEAKER_MODEL_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping ONNX speaker embedding smoke; set RUN_NATIVE_SPEAKER_MODEL_TESTS=1");
        return;
    }
    let bundle_path = std::env::var_os("SPEAKER_EMBEDDING_MODEL_BUNDLE")
        .map(std::path::PathBuf::from)
        .expect("SPEAKER_EMBEDDING_MODEL_BUNDLE is required");
    let audio_path = std::env::var_os("DIARIZATION_AUDIO_PATH")
        .map(std::path::PathBuf::from)
        .expect("DIARIZATION_AUDIO_PATH is required");
    let embedding_dimension = std::env::var("SPEAKER_EMBEDDING_DIMENSION")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid embedding dimension"))
        .unwrap_or(192);
    let model_file = optional_smoke_env_value(std::env::var("SPEAKER_EMBEDDING_MODEL_FILE").ok());
    let input_name = optional_smoke_env_value(std::env::var("SPEAKER_EMBEDDING_INPUT_NAME").ok());
    let output_name = optional_smoke_env_value(std::env::var("SPEAKER_EMBEDDING_OUTPUT_NAME").ok());
    let mut reader = hound::WavReader::open(&audio_path)
        .unwrap_or_else(|error| panic!("failed to open WAV `{}`: {error}", audio_path.display()));
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "smoke WAV must be 16 kHz");
    assert!(spec.channels > 0, "WAV channel count must be non-zero");
    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => {
            assert_eq!(spec.bits_per_sample, 32, "float WAV must be 32-bit");
            reader
                .samples::<f32>()
                .map(|sample| sample.expect("failed to read float WAV sample"))
                .collect::<Vec<_>>()
        }
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.expect("failed to read int WAV sample") as f32 / 32_768.0)
            .collect::<Vec<_>>(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|sample| {
                let scale = 2_f32.powi(spec.bits_per_sample as i32 - 1);
                sample.expect("failed to read int WAV sample") as f32 / scale
            })
            .collect::<Vec<_>>(),
    };
    let mono = interleaved_to_mono(
        &AudioBuffer::F32(interleaved),
        spec.channels,
        ChannelMix::Average,
    )
    .expect("valid mono mix");
    let audio = SpeakerAudio::mono(&mono, spec.sample_rate).expect("valid speaker audio");
    let config = OnnxSpeakerEmbeddingConfig {
        bundle_path,
        model_file,
        input_name,
        output_name,
        embedding_dimension,
        sample_rate: 16_000,
    };
    let model_path =
        resolve_onnx_speaker_model_path(&config).expect("resolved ONNX speaker model path");
    eprintln!("speakerEmbeddingResolvedModelPath={}", model_path.display());
    eprintln!("speakerEmbeddingExpectedDimension={embedding_dimension}");
    eprintln!(
        "speakerEmbeddingConfiguredInputName={}",
        config.input_name.as_deref().unwrap_or("<auto>")
    );
    eprintln!(
        "speakerEmbeddingConfiguredOutputName={}",
        config.output_name.as_deref().unwrap_or("<auto>")
    );
    let static_metadata =
        runtime_onnx::inspect_model_metadata(&model_path).expect("static ONNX metadata");
    eprintln!("onnxStaticMetadata=ok");
    for diagnostic in runtime_onnx::inspect_model_graph_diagnostics(&model_path)
        .expect("static ONNX graph diagnostics")
    {
        eprintln!("{diagnostic}");
    }
    eprintln!(
        "onnxLoadMode={}",
        std::env::var("ONNX_RUNTIME_LOAD_MODE").unwrap_or_else(|_| "file".to_string())
    );
    eprintln!(
        "onnxRuntimeDylib={}",
        if std::env::var_os("ORT_DYLIB_PATH").is_some() {
            "set"
        } else {
            "unset"
        }
    );
    let static_input = select_onnx_speaker_input(&static_metadata, config.input_name.as_deref())
        .expect("static ONNX speaker input metadata");
    let static_output = select_onnx_speaker_output(&static_metadata, config.output_name.as_deref())
        .expect("static ONNX speaker output metadata");
    eprintln!("speakerEmbeddingStaticInputName={}", static_input.name);
    eprintln!(
        "speakerEmbeddingStaticInputDimensions={}",
        format_onnx_dimensions(&static_input.dimensions)
    );
    eprintln!("speakerEmbeddingStaticOutputName={}", static_output.name);
    eprintln!(
        "speakerEmbeddingStaticOutputDimensions={}",
        format_onnx_dimensions(&static_output.dimensions)
    );
    eprintln!(
        "onnxSessionOptions=cpu-single-threaded,no-memory-pattern,graph-optimization-disabled"
    );
    let mut provider = match OnnxSpeakerEmbedder::from_config(config) {
        Ok(provider) => {
            eprintln!("onnxSessionLoad=ok");
            provider
        }
        Err(error) => {
            eprintln!("onnxSessionLoad={error}");
            panic!("ONNX speaker embedding provider: {error}");
        }
    };
    eprintln!(
        "{}",
        provider
            .onnx_runtime_metadata_diagnostics()
            .expect("ONNX speaker embedding metadata")
            .join("\n")
    );

    let response = provider
        .embed(SpeakerEmbeddingRequest { audio })
        .expect("ONNX speaker embedding");

    assert_eq!(response.runtime, AudioRuntime::Onnx);
    assert_eq!(response.embedding.dimensions(), embedding_dimension);
    assert!(response
        .embedding
        .values()
        .iter()
        .all(|value| value.is_finite()));
    eprintln!("{:#?}", response.diagnostics);
}
