//! Transcript diarization providers and speaker-assignment helpers.

use crate::*;

/// Native deterministic speaker diarization adapter.
#[cfg(feature = "diarization")]
#[derive(Debug, Clone, Default)]
pub struct NativeSpeakerDiarizationProvider;

#[cfg(feature = "diarization")]
impl TranscriptDiarizationProvider for NativeSpeakerDiarizationProvider {
    fn provider_id(&self) -> &str {
        "native-speaker-diarization"
    }

    fn diarize(
        &mut self,
        audio: LoadedAudio,
        transcript: &TranscriptionContract,
        options: &DiarizationOptions,
    ) -> Result<SpeakerDiarizationResponse> {
        native_audio::validate_loaded_audio(&audio)?;
        if options.is_pyannote_model() {
            #[cfg(feature = "pyannote-diarization")]
            {
                let mut provider = PyannoteCommunityTranscriptDiarizationProvider;
                return provider.diarize(audio, transcript, options);
            }
            #[cfg(not(feature = "pyannote-diarization"))]
            {
                return Err(setup_error(
                    "native pyannote diarization requires the pyannote-diarization feature",
                ));
            }
        }
        if options.speaker_embedding_model_bundle.is_some() {
            return diarize_with_onnx_speaker_embeddings(audio, transcript, options);
        }
        let spans = speech_spans_from_transcript(transcript, audio.duration_seconds())?;
        if !spans.is_empty() {
            let speaker_audio =
                audio_analysis_speakers::SpeakerAudio::mono(&audio.samples, audio.sample_rate)?;
            let embedder = audio_analysis_speakers::SpectralSpeakerEmbedder::default();
            let vad = TranscriptSpeechSpanVad { spans };
            let mut diarizer = audio_analysis_speakers::WindowedSpeakerDiarizer::new(embedder, vad)
                .cluster_threshold(0.95)?
                .speaker_bounds(options.min_speakers, options.max_speakers)?
                .min_speaker_evidence_seconds(0.0)?;
            let result =
                audio_analysis_speakers::SpeakerDiarizer::diarize(&mut diarizer, &speaker_audio)?;
            return Ok(SpeakerDiarizationResponse {
                accepted: true,
                operation: "audio.speakers.diarize".to_string(),
                model_id: options.model_id.clone(),
                runtime: audio_analysis_speakers::AudioRuntime::Heuristic,
                segments: stable_speaker_predictions_from_diarization(result.segments)?,
                speaker_embeddings: None,
                diagnostics: Vec::new(),
            });
        }

        let speaker_audio =
            audio_analysis_speakers::SpeakerAudio::mono(&audio.samples, audio.sample_rate)?;
        let embedder = audio_analysis_speakers::SpectralSpeakerEmbedder::default();
        let vad = EnergyVoiceActivityDetector;
        let mut diarizer = audio_analysis_speakers::WindowedSpeakerDiarizer::new(embedder, vad)
            .cluster_threshold(0.95)?
            .speaker_bounds(options.min_speakers, options.max_speakers)?
            .min_speaker_evidence_seconds(0.0)?;
        let result =
            audio_analysis_speakers::SpeakerDiarizer::diarize(&mut diarizer, &speaker_audio)?;
        Ok(SpeakerDiarizationResponse {
            accepted: true,
            operation: "audio.speakers.diarize".to_string(),
            model_id: options.model_id.clone(),
            runtime: audio_analysis_speakers::AudioRuntime::Heuristic,
            segments: stable_speaker_predictions_from_diarization(result.segments)?,
            speaker_embeddings: None,
            diagnostics: Vec::new(),
        })
    }
}

/// Native pyannote community diarization adapter.
#[cfg(all(feature = "diarization", feature = "pyannote-diarization"))]
#[derive(Debug, Clone, Default)]
pub struct PyannoteCommunityTranscriptDiarizationProvider;

#[cfg(all(feature = "diarization", feature = "pyannote-diarization"))]
impl TranscriptDiarizationProvider for PyannoteCommunityTranscriptDiarizationProvider {
    fn provider_id(&self) -> &str {
        "pyannote-community-diarization"
    }

    fn diarize(
        &mut self,
        audio: LoadedAudio,
        _transcript: &TranscriptionContract,
        options: &DiarizationOptions,
    ) -> Result<SpeakerDiarizationResponse> {
        native_audio::validate_loaded_audio(&audio)?;
        if !options.is_pyannote_model() {
            return Err(invalid_request(format!(
                "pyannote community diarization provider does not support model `{}`",
                options.model_id
            )));
        }
        let bundle_path = options.pyannote_model_bundle.clone().ok_or_else(|| {
            setup_error(
                "native pyannote diarization requires --diarization-model-bundle or DiarizationOptions.pyannote_model_bundle",
            )
        })?;
        let speaker_audio =
            audio_analysis_speakers::SpeakerAudio::mono(&audio.samples, audio.sample_rate)?;
        let mut diarizer = audio_analysis_speakers::PyannoteCommunityDiarizer::from_config(
            audio_analysis_speakers::PyannoteCommunityDiarizationConfig {
                bundle_path,
                manifest_file: options.pyannote_manifest_file.clone(),
                segmentation_model_file: options.pyannote_segmentation_model_file.clone(),
                embedding_model_file: options.pyannote_embedding_model_file.clone(),
                plda_transform_file: options.pyannote_plda_transform_file.clone(),
                plda_model_file: options.pyannote_plda_model_file.clone(),
                clustering_config_file: options.pyannote_clustering_config_file.clone(),
                min_speakers: options.min_speakers,
                max_speakers: options.max_speakers,
                return_speaker_embeddings: options.return_speaker_embeddings,
            },
        )?;
        let mut result = diarizer.diarize(&speaker_audio)?.response;
        result.model_id = options.model_id.clone();
        Ok(result)
    }
}

#[cfg(feature = "diarization")]
pub(crate) fn diarize_with_onnx_speaker_embeddings(
    audio: LoadedAudio,
    transcript: &TranscriptionContract,
    options: &DiarizationOptions,
) -> Result<SpeakerDiarizationResponse> {
    let config = options.onnx_speaker_embedding_config()?;
    let speaker_audio =
        audio_analysis_speakers::SpeakerAudio::mono(&audio.samples, audio.sample_rate)?;
    let embedder = audio_analysis_speakers::OnnxSpeakerEmbedder::from_config(config)?;
    let spans = speech_spans_from_transcript(transcript, audio.duration_seconds())?;
    let result = if spans.is_empty() {
        let vad = EnergyVoiceActivityDetector;
        let mut diarizer = audio_analysis_speakers::WindowedSpeakerDiarizer::new(embedder, vad)
            .speaker_bounds(options.min_speakers, options.max_speakers)?;
        audio_analysis_speakers::SpeakerDiarizer::diarize(&mut diarizer, &speaker_audio)?
    } else {
        let vad = TranscriptSpeechSpanVad { spans };
        let mut diarizer = audio_analysis_speakers::WindowedSpeakerDiarizer::new(embedder, vad)
            .speaker_bounds(options.min_speakers, options.max_speakers)?;
        audio_analysis_speakers::SpeakerDiarizer::diarize(&mut diarizer, &speaker_audio)?
    };
    Ok(SpeakerDiarizationResponse {
        accepted: true,
        operation: "audio.speakers.diarize".to_string(),
        model_id: options.model_id.clone(),
        runtime: audio_analysis_speakers::AudioRuntime::Onnx,
        segments: stable_speaker_predictions_from_diarization(result.segments)?,
        speaker_embeddings: None,
        diagnostics: Vec::new(),
    })
}

pub(crate) fn validate_diarization_options(options: &DiarizationOptions) -> Result<()> {
    options.speaker.validate()
}

#[cfg(feature = "diarization")]
pub(crate) fn speech_spans_from_transcript(
    transcript: &TranscriptionContract,
    audio_duration_seconds: f64,
) -> Result<Vec<SpeechSpan>> {
    const AUDIO_DURATION_EPSILON: f64 = 1e-6;

    let has_timed_words = transcript.segments.iter().any(|segment| {
        segment.words().iter().any(|word| {
            !word.text.trim().is_empty()
                && word.start_seconds().is_some()
                && word.end_seconds().is_some()
        })
    });

    let mut spans = Vec::new();
    if has_timed_words {
        for word in transcript
            .segments
            .iter()
            .flat_map(|segment| segment.words())
        {
            if word.text.trim().is_empty() {
                continue;
            }
            let Some((start, end)) = word.start_seconds().zip(word.end_seconds()) else {
                continue;
            };
            spans.push(transcript_timing_span(
                start,
                end,
                audio_duration_seconds,
                AUDIO_DURATION_EPSILON,
            )?);
        }
    } else {
        for segment in &transcript.segments {
            if segment.text.trim().is_empty() {
                continue;
            }
            let Some((start, end)) = segment.start_seconds().zip(segment.end_seconds()) else {
                continue;
            };
            spans.push(transcript_timing_span(
                start,
                end,
                audio_duration_seconds,
                AUDIO_DURATION_EPSILON,
            )?);
        }
    }

    merge_transcript_speech_spans(spans, VadOptions::default().merge_gap_seconds)
}

#[cfg(feature = "diarization")]
pub(crate) fn transcript_timing_span(
    start_seconds: f64,
    end_seconds: f64,
    audio_duration_seconds: f64,
    audio_duration_epsilon: f64,
) -> Result<SpeechSpan> {
    if !start_seconds.is_finite() || !end_seconds.is_finite() || !audio_duration_seconds.is_finite()
    {
        return Err(invalid_request(
            "transcript diarization timing values must be finite",
        ));
    }
    if start_seconds < 0.0 || end_seconds <= start_seconds {
        return Err(invalid_request(
            "transcript diarization timing must be non-negative with positive duration",
        ));
    }
    if start_seconds > audio_duration_seconds + audio_duration_epsilon
        || end_seconds > audio_duration_seconds + audio_duration_epsilon
    {
        return Err(invalid_request(format!(
            "transcript diarization timing end {:.6} exceeds audio duration {:.6}",
            end_seconds, audio_duration_seconds
        )));
    }
    let end_seconds = if end_seconds > audio_duration_seconds {
        audio_duration_seconds
    } else {
        end_seconds
    };
    SpeechSpan::new(start_seconds, end_seconds, 1.0)
}

#[cfg(feature = "diarization")]
pub(crate) fn merge_transcript_speech_spans(
    mut spans: Vec<SpeechSpan>,
    merge_gap_seconds: f64,
) -> Result<Vec<SpeechSpan>> {
    spans.sort_by(|left, right| left.start_seconds.total_cmp(&right.start_seconds));
    let mut merged: Vec<SpeechSpan> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut() {
            if span.start_seconds - last.end_seconds <= merge_gap_seconds {
                let last_duration = last.duration_seconds();
                let span_duration = span.duration_seconds();
                let total_duration = last_duration + span_duration;
                last.end_seconds = last.end_seconds.max(span.end_seconds);
                last.score = if total_duration > f64::EPSILON {
                    (((last.score as f64 * last_duration) + (span.score as f64 * span_duration))
                        / total_duration) as f32
                } else {
                    last.score.max(span.score)
                };
                continue;
            }
        }
        merged.push(span);
    }
    Ok(merged)
}

#[cfg(feature = "diarization")]
#[derive(Debug, Clone)]
pub(crate) struct TranscriptSpeechSpanVad {
    spans: Vec<SpeechSpan>,
}

#[cfg(feature = "diarization")]
impl VoiceActivityDetector for TranscriptSpeechSpanVad {
    fn provider_id(&self) -> &str {
        "transcript-speech-spans"
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

#[cfg(feature = "diarization")]
pub(crate) fn stable_speaker_predictions_from_diarization(
    segments: Vec<audio_analysis_speakers::DiarizationSegment>,
) -> Result<Vec<SpeakerSegmentPrediction>> {
    let mut unknown_labels: Vec<(String, String)> = Vec::new();
    let mut predictions = Vec::new();
    for segment in segments {
        let speaker = match segment.speaker {
            // Too-short spans assert no speaker; transcript words there stay unassigned.
            audio_analysis_speakers::DiarizedSpeaker::Unresolved => continue,
            audio_analysis_speakers::DiarizedSpeaker::Known(id) => id.as_str().to_string(),
            audio_analysis_speakers::DiarizedSpeaker::Unknown(label) => {
                if let Some((_, stable)) = unknown_labels
                    .iter()
                    .find(|(existing, _)| existing == &label)
                {
                    stable.clone()
                } else {
                    let stable = format!("speaker_{}", unknown_labels.len());
                    unknown_labels.push((label, stable.clone()));
                    stable
                }
            }
        };
        predictions.push(normalize_speaker_prediction(SpeakerSegmentPrediction {
            speaker,
            start_seconds: segment.start_seconds as f32,
            end_seconds: segment.end_seconds as f32,
            score: Some(segment.score),
        })?);
    }
    merge_speaker_predictions(predictions, VadOptions::default().merge_gap_seconds as f32)
}

#[cfg(feature = "diarization")]
pub(crate) fn normalize_speaker_prediction(
    mut segment: SpeakerSegmentPrediction,
) -> Result<SpeakerSegmentPrediction> {
    segment.speaker = segment.speaker.trim().to_string();
    if segment.speaker.is_empty() {
        return Err(invalid_request("speaker label must not be empty"));
    }
    if !segment.start_seconds.is_finite() || !segment.end_seconds.is_finite() {
        return Err(invalid_request("speaker segment timestamps must be finite"));
    }
    if segment.end_seconds < segment.start_seconds {
        return Err(invalid_request(
            "speaker segment end_seconds must be greater than or equal to start_seconds",
        ));
    }
    segment.score = segment
        .score
        .and_then(|score| score.is_finite().then(|| score.clamp(0.0, 1.0)));
    Ok(segment)
}

#[cfg(feature = "diarization")]
pub(crate) fn merge_speaker_predictions(
    segments: Vec<SpeakerSegmentPrediction>,
    merge_gap_seconds: f32,
) -> Result<Vec<SpeakerSegmentPrediction>> {
    let mut merged: Vec<SpeakerSegmentPrediction> = Vec::new();
    for segment in segments {
        if let Some(last) = merged.last_mut() {
            if last.speaker == segment.speaker
                && segment.start_seconds - last.end_seconds <= merge_gap_seconds
            {
                let last_duration = (last.end_seconds - last.start_seconds).max(0.0);
                let segment_duration = (segment.end_seconds - segment.start_seconds).max(0.0);
                let total_duration = last_duration + segment_duration;
                last.end_seconds = segment.end_seconds;
                last.score = match (last.score, segment.score) {
                    (Some(left), Some(right)) if total_duration > f32::EPSILON => {
                        Some(((left * last_duration) + (right * segment_duration)) / total_duration)
                    }
                    (Some(left), Some(right)) => Some(left.max(right)),
                    (Some(left), None) => Some(left),
                    (None, Some(right)) => Some(right),
                    (None, None) => None,
                };
                continue;
            }
        }
        merged.push(segment);
    }
    Ok(merged)
}

pub(crate) fn diarization_speaker_count(response: &SpeakerDiarizationResponse) -> usize {
    response
        .segments
        .iter()
        .map(|segment| segment.speaker.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

pub(crate) fn diagnostic_usize(diagnostics: &[String], key: &str) -> Option<usize> {
    let prefix = format!("{key}=");
    diagnostics
        .iter()
        .find_map(|diagnostic| diagnostic.strip_prefix(&prefix))
        .and_then(|value| value.parse().ok())
}

pub(crate) fn diarization_progress_provider(
    provider_id: &str,
    options: &DiarizationOptions,
) -> String {
    if options
        .model_id
        .trim()
        .to_ascii_lowercase()
        .starts_with("pyannote/")
    {
        "pyannote".to_string()
    } else {
        provider_id.to_string()
    }
}

pub(crate) fn diarization_diagnostics(
    provider_id: &str,
    response: &SpeakerDiarizationResponse,
    options: &DiarizationOptions,
) -> Vec<String> {
    let speaker_count = diarization_speaker_count(response);
    let mut diagnostics = vec![
        format!("diarizationProvider={provider_id}"),
        format!("diarizationRuntime={}", diarization_runtime_value(response)),
        format!("diarizationModelId={}", response.model_id),
        format!("diarizationSegmentCount={}", response.segments.len()),
        format!("diarizationSpeakerCount={speaker_count}"),
        format!(
            "diarizationAssignmentPolicy={}",
            speaker_assignment_policy_value(options.assignment_policy)
        ),
    ];
    if let Some(min) = options.min_speakers {
        diagnostics.push(format!("diarizationMinSpeakers={min}"));
        if speaker_count < min {
            diagnostics.push(format!(
                "diarizationSpeakerCountBelowRequestedMin={speaker_count}/{min}"
            ));
            if response.segments.len() < min {
                diagnostics.push("diarizationSpeakerBoundsSaturated=true".to_string());
            }
        }
    }
    if let Some(max) = options.max_speakers {
        diagnostics.push(format!("diarizationMaxSpeakers={max}"));
        if speaker_count > max {
            diagnostics.push(format!(
                "diarizationSpeakerCountAboveRequestedMax={speaker_count}/{max}"
            ));
        }
    }
    if options.min_speakers.is_some() || options.max_speakers.is_some() {
        diagnostics.push("diarizationSpeakerBoundsApplied=true".to_string());
    }
    if diarization_runtime_is_heuristic(response) {
        diagnostics.push("diarizationBaseline=heuristic-native".to_string());
    } else if diarization_runtime_value(response) == "onnx" {
        diagnostics.push("speakerEmbeddingProvider=onnx".to_string());
        if let Some(dimension) = options.speaker_embedding_dimension {
            diagnostics.push(format!("speakerEmbeddingDimension={dimension}"));
        }
        diagnostics.push("diarizationBaseline=false".to_string());
    }
    diagnostics
}

pub(crate) fn diarization_runtime_value(response: &SpeakerDiarizationResponse) -> &'static str {
    match response.runtime {
        audio_analysis_speakers::AudioRuntime::Onnx => "onnx",
        audio_analysis_speakers::AudioRuntime::Candle => "candle",
        audio_analysis_speakers::AudioRuntime::WhisperCpp => "whisper_cpp",
        audio_analysis_speakers::AudioRuntime::Demucs => "demucs",
        audio_analysis_speakers::AudioRuntime::External => "external",
        audio_analysis_speakers::AudioRuntime::Spectral => "spectral",
        audio_analysis_speakers::AudioRuntime::Heuristic => "heuristic",
        audio_analysis_speakers::AudioRuntime::Imported => "imported",
    }
}

pub(crate) fn diarization_runtime_is_heuristic(response: &SpeakerDiarizationResponse) -> bool {
    response.runtime == audio_analysis_speakers::AudioRuntime::Heuristic
}

pub(crate) fn speaker_assignment_policy_value(policy: SpeakerAssignmentPolicy) -> &'static str {
    match policy {
        SpeakerAssignmentPolicy::Majority => "majority",
        SpeakerAssignmentPolicy::NearestStart => "nearestStart",
        SpeakerAssignmentPolicy::StrictContained => "strictContained",
    }
}
