//! Pipeline request, events, observer, response and the provider-agnostic run functions.

use crate::*;

/// Request for an audio/video-to-text transcription pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionPipelineRequest {
    pub source: TranscriptionSource,
    pub provider: TranscriptionProviderSelection,
    #[serde(default)]
    pub vad: TranscriptionVadOptions,
    #[serde(default)]
    pub alignment: AlignmentOptions,
    #[serde(default)]
    pub diarization: DiarizationOptions,
    #[serde(default)]
    pub output: TranscriptionOutputOptions,
}

/// Phase-level observer event emitted by the native transcription pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionPipelineEvent {
    ValidationStart,
    DecodeStart,
    DecodeEnd {
        duration_seconds: f64,
        samples: usize,
    },
    VadStart {
        provider: String,
    },
    VadEnd {
        segments: usize,
        windows: Option<usize>,
    },
    AsrStart {
        model_id: String,
    },
    AsrEnd {
        segments: usize,
    },
    ModelLoadStart {
        stage: String,
        provider: String,
        model_id: String,
    },
    ModelLoadEnd {
        stage: String,
        provider: String,
        model_id: String,
        duration_seconds: f64,
    },
    ModelReuse {
        stage: String,
        provider: String,
        model_id: String,
    },
    AlignmentStart {
        model_id: String,
    },
    AlignmentEnd {
        words: usize,
    },
    DiarizationStart {
        provider: String,
    },
    DiarizationEnd {
        speakers: usize,
        segments: usize,
    },
}

/// Observer for phase-level native transcription progress.
pub trait TranscriptionPipelineObserver {
    fn observe(&mut self, event: TranscriptionPipelineEvent);

    fn model_resolution_start(&mut self, _stage: &str, _provider: &str, _model_id: &str) {}

    fn model_resolution_end(
        &mut self,
        _stage: &str,
        _provider: &str,
        _model_id: &str,
        _source: &str,
    ) {
    }

    fn model_download_start(&mut self, _stage: &str, _provider: &str, _model_id: &str) {}

    fn model_download_end(
        &mut self,
        _stage: &str,
        _provider: &str,
        _model_id: &str,
        _duration_seconds: f64,
    ) {
    }

    /// Returns whether the current pipeline should stop at its next safe phase boundary.
    fn cancellation_requested(&self) -> bool {
        false
    }
}

/// Observer implementation that discards all events.
#[derive(Debug, Default)]
pub struct NoopTranscriptionPipelineObserver;

impl TranscriptionPipelineObserver for NoopTranscriptionPipelineObserver {
    fn observe(&mut self, _event: TranscriptionPipelineEvent) {}
}

/// Source accepted by transcription providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", untagged)]
pub enum TranscriptionSource {
    /// A finite media path with an explicitly selected zero-based audio stream.
    ///
    /// [`Self::Path`] remains the compatibility/default-selection source. This
    /// variant makes an explicit container-audio selection part of the
    /// reusable transcription request instead of requiring callers to decode
    /// samples before invoking the pipeline.
    Media {
        path: PathBuf,
        #[serde(rename = "audioStream", alias = "audio_stream")]
        audio_stream: usize,
    },
    Path {
        path: PathBuf,
    },
    Samples {
        samples: Vec<f32>,
        #[serde(rename = "sampleRate", alias = "sample_rate")]
        sample_rate: u32,
        channels: u16,
        #[serde(default)]
        source: Option<String>,
    },
}

impl TranscriptionSource {
    pub(crate) fn path(&self) -> Result<&Path> {
        match self {
            Self::Path { path } => Ok(path),
            Self::Media { .. } => Err(invalid_request(
                "external command transcription does not support explicit media stream selection",
            )),
            Self::Samples { .. } => Err(invalid_request(
                "external command transcription requires a path source",
            )),
        }
    }
}

/// Provider selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum TranscriptionProviderSelection {
    #[serde(rename = "candleWhisper", alias = "candle-whisper")]
    CandleWhisper(CandleWhisperOptions),
    #[serde(rename = "whisperCpp", alias = "whisper-cpp")]
    WhisperCpp(WhisperCppProviderOptions),
    #[serde(rename = "externalWhisperX", alias = "whisperx")]
    ExternalWhisperX(WhisperXCommandOptions),
}

impl TranscriptionProviderSelection {
    pub fn provider_id(&self) -> &'static str {
        match self {
            Self::CandleWhisper(_) => "candle-whisper",
            Self::WhisperCpp(_) => "whisper-cpp",
            Self::ExternalWhisperX(_) => "whisperx-command",
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            Self::CandleWhisper(options) => &options.model_id,
            Self::WhisperCpp(options) => &options.model_id,
            Self::ExternalWhisperX(options) => &options.model,
        }
    }

    pub fn task(&self) -> TranscriptionTask {
        match self {
            Self::CandleWhisper(options) => options.task,
            Self::WhisperCpp(options) => options.task,
            Self::ExternalWhisperX(options) => options.task,
        }
    }
}

/// Whisper speech task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TranscriptionTask {
    #[default]
    Transcribe,
    Translate,
}

impl TranscriptionTask {
    pub fn as_whisper_task(self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
        }
    }

    pub fn output_language_hint(self) -> Option<&'static str> {
        match self {
            Self::Transcribe => None,
            Self::Translate => Some("en"),
        }
    }
}

/// Artifact produced or discovered by a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionArtifact {
    pub kind: String,
    pub path: PathBuf,
}

/// Response from a transcription pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionPipelineResponse {
    pub accepted: bool,
    pub operation: String,
    pub provider: String,
    pub model_id: String,
    pub transcript: TranscriptionContract,
    pub vad_segments: Vec<SpeechSpan>,
    pub alignment: Option<AlignmentSummary>,
    pub diarization: Option<SpeakerDiarizationResponse>,
    pub artifacts: Vec<TranscriptionArtifact>,
    pub diagnostics: Vec<String>,
}

/// Metadata-only provider plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionProviderPlan {
    pub provider_id: String,
    pub external_runtime: bool,
    pub wasm_supported: bool,
    pub primary: bool,
    pub setup: Vec<String>,
    pub diagnostics: Vec<String>,
}

pub(crate) fn run_native_transcription_pipeline(
    request: TranscriptionPipelineRequest,
    vad: &mut dyn VoiceActivityDetector,
    asr: &mut dyn AudioTranscriptionProvider,
    diarization_provider: Option<&mut dyn TranscriptDiarizationProvider>,
) -> Result<TranscriptionPipelineResponse> {
    if !request.alignment.enabled {
        return run_transcription_pipeline(request, vad, asr, None, diarization_provider);
    }

    let mut aligner = CtcForcedAligner {
        options: request.alignment.clone(),
    };
    run_transcription_pipeline(
        request,
        vad,
        asr,
        Some(&mut aligner as &mut dyn ForcedAlignmentProvider),
        diarization_provider,
    )
}

/// Runs a transcription request with the selected primary provider.
pub fn transcribe(request: TranscriptionPipelineRequest) -> Result<TranscriptionPipelineResponse> {
    match &request.provider {
        TranscriptionProviderSelection::ExternalWhisperX(_) => {
            let mut provider = WhisperXCommandTranscriber;
            provider.transcribe_pipeline(request)
        }
        TranscriptionProviderSelection::CandleWhisper(options) => {
            let mut vad = EnergyVoiceActivityDetector;
            let mut asr = CandleWhisperTranscriber::new(options.clone());
            #[cfg(feature = "diarization")]
            {
                let mut diarizer = NativeSpeakerDiarizationProvider;
                let diarization_provider = request
                    .diarization
                    .enabled
                    .then_some(&mut diarizer as &mut dyn TranscriptDiarizationProvider);
                run_native_transcription_pipeline(request, &mut vad, &mut asr, diarization_provider)
            }
            #[cfg(not(feature = "diarization"))]
            {
                run_native_transcription_pipeline(request, &mut vad, &mut asr, None)
            }
        }
        TranscriptionProviderSelection::WhisperCpp(options) => {
            let mut vad = EnergyVoiceActivityDetector;
            let mut asr = WhisperCppTranscriber {
                options: options.clone(),
            };
            #[cfg(feature = "diarization")]
            {
                let mut diarizer = NativeSpeakerDiarizationProvider;
                let diarization_provider = request
                    .diarization
                    .enabled
                    .then_some(&mut diarizer as &mut dyn TranscriptDiarizationProvider);
                run_native_transcription_pipeline(request, &mut vad, &mut asr, diarization_provider)
            }
            #[cfg(not(feature = "diarization"))]
            {
                run_native_transcription_pipeline(request, &mut vad, &mut asr, None)
            }
        }
    }
}

/// Predecodes an optional zero-based audio stream, then transcribes it.
///
/// The request must retain its existing [`TranscriptionSource::Path`] shape.
/// Omitting `audio_stream_index` preserves first/default-stream behavior. The
/// media is validated and decoded before native providers are constructed, so
/// invalid selections cannot start model resolution, cache work, or progress.
#[cfg(feature = "audio-io")]
pub fn transcribe_selected_media(
    request: TranscriptionPipelineRequest,
    audio_stream_index: Option<usize>,
) -> std::result::Result<TranscriptionPipelineResponse, SelectedMediaTranscriptionError> {
    let request = predecode_selected_media_request(request, audio_stream_index)?;
    transcribe(request).map_err(SelectedMediaTranscriptionError::Pipeline)
}

/// Predecodes selected media before running the provider-agnostic pipeline.
#[cfg(feature = "audio-io")]
pub fn run_transcription_pipeline_from_selected_media_with_observer(
    request: TranscriptionPipelineRequest,
    audio_stream_index: Option<usize>,
    vad_provider: &mut dyn VoiceActivityDetector,
    asr_provider: &mut dyn AudioTranscriptionProvider,
    alignment_provider: Option<&mut dyn ForcedAlignmentProvider>,
    diarization_provider: Option<&mut dyn TranscriptDiarizationProvider>,
    observer: &mut dyn TranscriptionPipelineObserver,
) -> std::result::Result<TranscriptionPipelineResponse, SelectedMediaTranscriptionError> {
    let request = predecode_selected_media_request(request, audio_stream_index)?;
    run_transcription_pipeline_with_observer(
        request,
        vad_provider,
        asr_provider,
        alignment_provider,
        diarization_provider,
        observer,
    )
    .map_err(SelectedMediaTranscriptionError::Pipeline)
}

#[cfg(feature = "audio-io")]
pub(crate) fn predecode_selected_media_request(
    mut request: TranscriptionPipelineRequest,
    audio_stream_index: Option<usize>,
) -> std::result::Result<TranscriptionPipelineRequest, SelectedMediaTranscriptionError> {
    let path = match &request.source {
        TranscriptionSource::Path { path } => path.clone(),
        TranscriptionSource::Media { .. } => {
            return Err(SelectedMediaTranscriptionError::Pipeline(invalid_request(
                "selected-media transcription source is already decoded by the pipeline",
            )))
        }
        TranscriptionSource::Samples { .. } => {
            return Err(SelectedMediaTranscriptionError::Pipeline(invalid_request(
                "selected-media transcription requires a path source",
            )))
        }
    };
    let audio = LoadedAudio::mono_16khz_from_selected_media(&path, audio_stream_index)?;
    request.source = TranscriptionSource::Samples {
        samples: audio.samples,
        sample_rate: audio.sample_rate,
        channels: audio.channels,
        source: audio.source,
    };
    Ok(request)
}

/// Runs the provider-agnostic native transcription pipeline.
pub fn run_transcription_pipeline(
    request: TranscriptionPipelineRequest,
    vad_provider: &mut dyn VoiceActivityDetector,
    asr_provider: &mut dyn AudioTranscriptionProvider,
    alignment_provider: Option<&mut dyn ForcedAlignmentProvider>,
    diarization_provider: Option<&mut dyn TranscriptDiarizationProvider>,
) -> Result<TranscriptionPipelineResponse> {
    let mut observer = NoopTranscriptionPipelineObserver;
    run_transcription_pipeline_with_observer(
        request,
        vad_provider,
        asr_provider,
        alignment_provider,
        diarization_provider,
        &mut observer,
    )
}

/// Runs the provider-agnostic native transcription pipeline and emits phase events.
pub fn run_transcription_pipeline_with_observer(
    request: TranscriptionPipelineRequest,
    vad_provider: &mut dyn VoiceActivityDetector,
    asr_provider: &mut dyn AudioTranscriptionProvider,
    alignment_provider: Option<&mut dyn ForcedAlignmentProvider>,
    diarization_provider: Option<&mut dyn TranscriptDiarizationProvider>,
    observer: &mut dyn TranscriptionPipelineObserver,
) -> Result<TranscriptionPipelineResponse> {
    observer.observe(TranscriptionPipelineEvent::ValidationStart);
    ensure_pipeline_active(observer)?;
    validate_batch_options_for_provider(&request.provider)?;
    validate_task_options_for_request(&request)?;
    let provider = request.provider.provider_id().to_string();
    let model_id = request.provider.model_id().to_string();
    let task = request.provider.task();
    observer.observe(TranscriptionPipelineEvent::DecodeStart);
    ensure_pipeline_active(observer)?;
    let decode_started = Instant::now();
    let audio = LoadedAudio::mono_16khz_from_source(&request.source)?;
    observer.observe(TranscriptionPipelineEvent::DecodeEnd {
        duration_seconds: decode_started.elapsed().as_secs_f64(),
        samples: audio.samples.len(),
    });
    ensure_pipeline_active(observer)?;
    observer.observe(TranscriptionPipelineEvent::VadStart {
        provider: vad_provider.provider_id().to_string(),
    });
    ensure_pipeline_active(observer)?;
    let full_source_seconds = audio.duration_seconds().max(1.0 / audio.sample_rate as f64);
    let vad_response = if request.vad.enabled {
        let mut response =
            vad_provider.detect_speech(&audio.speaker_audio()?, &request.vad.detection)?;
        response.segments = asr_chunks_from_speech(
            response.segments,
            &audio,
            full_source_seconds,
            request.vad.detection.max_chunk_seconds,
        )?;
        if response.segments.is_empty() {
            response
                .segments
                .push(SpeechSpan::new(0.0, full_source_seconds, 0.0)?);
            response
                .diagnostics
                .push("VAD found no speech; using full source as one ASR chunk".to_string());
        }
        push_cpu_stage_device(&mut response.diagnostics, "vadDevice");
        response
    } else {
        VadResponse {
            segments: vec![SpeechSpan::new(0.0, full_source_seconds, 1.0)?],
            diagnostics: vec!["VAD disabled; using full source as one ASR chunk".to_string()],
        }
    };
    observer.observe(TranscriptionPipelineEvent::VadEnd {
        segments: vad_response.segments.len(),
        windows: diagnostic_usize(&vad_response.diagnostics, "pyannoteVadWindows")
            .or_else(|| diagnostic_usize(&vad_response.diagnostics, "sileroVadWindows")),
    });
    ensure_pipeline_active(observer)?;

    let language = provider_language(&request.provider);
    observer.observe(TranscriptionPipelineEvent::AsrStart {
        model_id: model_id.clone(),
    });
    ensure_pipeline_active(observer)?;
    let mut asr_response = asr_provider.transcribe_with_observer(
        AsrRequest {
            audio: audio.clone(),
            chunks: vad_response.segments.clone(),
            task,
            language: language.clone(),
            model_id: model_id.clone(),
        },
        observer,
    )?;
    observer.observe(TranscriptionPipelineEvent::AsrEnd {
        segments: asr_response.transcript.segments.len(),
    });
    ensure_pipeline_active(observer)?;
    if !asr_response
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.starts_with("batchChunks="))
    {
        if let TranscriptionProviderSelection::CandleWhisper(options) = &request.provider {
            extend_missing_candle_batch_diagnostics(
                &mut asr_response.diagnostics,
                options,
                vad_response.segments.len(),
            );
        }
    }
    let mut transcript = normalize_transcription_contract(asr_response.transcript)
        .map_err(|error| model_output_mismatch(error.to_string()))?;
    offset_chunk_local_segments(&mut transcript, &vad_response.segments)?;

    let mut diagnostics = vec![format!("asrTask={}", task.as_whisper_task())];
    if task == TranscriptionTask::Translate {
        diagnostics.push("translationRuntime=whisper-task".to_string());
        if let Some(language) = task.output_language_hint() {
            diagnostics.push(format!("translationTargetLanguage={language}"));
        }
    }
    diagnostics.extend(vad_response.diagnostics);
    diagnostics.extend(asr_response.diagnostics);
    let mut alignment_summary = None;
    if request.alignment.enabled {
        let provider = alignment_provider.ok_or_else(|| {
            setup_error("alignment requested but no alignment provider is available")
        })?;
        observer.observe(TranscriptionPipelineEvent::AlignmentStart {
            model_id: request.alignment.model_id.clone(),
        });
        ensure_pipeline_active(observer)?;
        let alignment_response = provider.align_with_observer(
            AlignmentRequest {
                audio: audio.clone(),
                transcript: transcript.clone(),
                language: language.clone(),
                model_id: request.alignment.model_id.clone(),
            },
            observer,
        )?;
        observer.observe(TranscriptionPipelineEvent::AlignmentEnd {
            words: alignment_response.words.len(),
        });
        ensure_pipeline_active(observer)?;
        apply_alignment_words(&mut transcript, &alignment_response.words)?;
        apply_alignment_chars(&mut transcript, &alignment_response.chars)?;
        alignment_summary = Some(AlignmentSummary {
            provider: provider.provider_id().to_string(),
            model_id: alignment_response.model_id,
            word_count: alignment_response.words.len(),
        });
        diagnostics.extend(alignment_response.diagnostics);
    }
    order_segments_chronologically(&mut transcript);

    let mut diarization = None;
    if request.diarization.enabled {
        validate_diarization_options(&request.diarization)?;
        let provider = diarization_provider.ok_or_else(|| {
            setup_error("diarization requested but no diarization provider is available")
        })?;
        observer.observe(TranscriptionPipelineEvent::DiarizationStart {
            provider: diarization_progress_provider(provider.provider_id(), &request.diarization),
        });
        ensure_pipeline_active(observer)?;
        let response = provider.diarize(audio, &transcript, &request.diarization)?;
        observer.observe(TranscriptionPipelineEvent::DiarizationEnd {
            speakers: diarization_speaker_count(&response),
            segments: response.segments.len(),
        });
        ensure_pipeline_active(observer)?;
        diagnostics.extend(diarization_diagnostics(
            provider.provider_id(),
            &response,
            &request.diarization,
        ));
        let mut provider_diagnostics = response.diagnostics.clone();
        push_cpu_stage_device(&mut provider_diagnostics, "diarizationDevice");
        diagnostics.extend(provider_diagnostics);
        transcript = audio_analysis_speakers::assign_speakers_to_transcript_with_policy(
            &transcript,
            &response,
            request.diarization.assignment_policy,
        )?;
        diarization = Some(response);
    }

    transcript = normalize_transcription_contract(transcript)
        .map_err(|error| model_output_mismatch(error.to_string()))?;
    transcript
        .validate_strict()
        .map_err(|error| model_output_mismatch(error.to_string()))?;

    Ok(TranscriptionPipelineResponse {
        accepted: true,
        operation: "audio.transcription.transcribe".to_string(),
        provider,
        model_id,
        transcript,
        vad_segments: vad_response.segments,
        alignment: alignment_summary,
        diarization,
        artifacts: Vec::new(),
        diagnostics,
    })
}

/// Records `key=cpu` unless the provider reported its own device. Every VAD
/// (energy, Silero, pyannote) and diarization provider here runs on the CPU:
/// the ONNX sessions use the CPU execution provider only.
fn push_cpu_stage_device(diagnostics: &mut Vec<String>, key: &str) {
    let prefix = format!("{key}=");
    if !diagnostics.iter().any(|item| item.starts_with(&prefix)) {
        diagnostics.push(format!("{prefix}cpu"));
    }
}

/// Turns detector speech spans into ASR chunks: clamps them to the audio and,
/// with a chunk size, merges adjacent spans up to it (WhisperX `merge_chunks`)
/// and repairs the sub-second stubs a split at the chunk-size limit leaves.
pub(crate) fn asr_chunks_from_speech(
    mut spans: Vec<SpeechSpan>,
    audio: &LoadedAudio,
    duration_seconds: f64,
    chunk_seconds: Option<f64>,
) -> Result<Vec<SpeechSpan>> {
    spans.sort_by(|left, right| left.start_seconds.total_cmp(&right.start_seconds));
    let clamped = spans
        .into_iter()
        .filter_map(|mut span| {
            span.start_seconds = span.start_seconds.max(0.0);
            span.end_seconds = span.end_seconds.min(duration_seconds);
            (span.end_seconds > span.start_seconds).then_some(span)
        })
        .collect();
    match chunk_seconds {
        Some(chunk_seconds) => {
            let merged = merge_whisperx_vad_chunks(clamped, chunk_seconds)?;
            Ok(repair_split_stub_chunks(merged, audio, chunk_seconds))
        }
        None => Ok(clamped),
    }
}

/// Shortest ASR chunk a split at the chunk-size limit may leave behind (capped
/// at a quarter of the chunk size). Whisper turns shorter remainders into
/// "Thank you." hallucinations, and wav2vec2 cannot align them.
pub const MIN_SPLIT_CHUNK_SECONDS: f64 = 1.0;
/// Chunks this close together are one VAD region cut at the chunk-size limit.
const SPLIT_CONTIGUITY_SECONDS: f64 = 1.0e-3;
/// How far from the forced boundary the low-energy search may move it.
const SPLIT_SEARCH_SECONDS: f64 = 2.0;
/// RMS window used to find a quiet point for a moved boundary.
const SPLIT_ENERGY_WINDOW_SECONDS: f64 = 0.02;

/// A VAD region split at the chunk-size limit can leave a remainder of a few
/// frames as its own chunk; `merge_chunks` cannot absorb it because the joined
/// chunk would exceed `chunk_size`. Moves the boundary it shares with its
/// contiguous neighbour to the quietest point that leaves both sides at least
/// the minimum and within `chunk_size`. Short speech apart from its neighbours
/// is real speech and is kept.
pub(crate) fn repair_split_stub_chunks(
    mut chunks: Vec<SpeechSpan>,
    audio: &LoadedAudio,
    chunk_size: f64,
) -> Vec<SpeechSpan> {
    let min_seconds = MIN_SPLIT_CHUNK_SECONDS.min(chunk_size / 4.0);
    for index in 0..chunks.len() {
        if chunks[index].duration_seconds() >= min_seconds {
            continue;
        }
        let joins_previous = index > 0
            && chunks[index].start_seconds - chunks[index - 1].end_seconds
                <= SPLIT_CONTIGUITY_SECONDS;
        let joins_next = index + 1 < chunks.len()
            && chunks[index + 1].start_seconds - chunks[index].end_seconds
                <= SPLIT_CONTIGUITY_SECONDS;
        let (left, right) = match (joins_previous, joins_next) {
            (true, _) => (index - 1, index),
            (false, true) => (index, index + 1),
            (false, false) => continue,
        };
        let start = chunks[left].start_seconds;
        let end = chunks[right].end_seconds;
        let lowest_valid = (start + min_seconds).max(end - chunk_size);
        let highest_valid = (end - min_seconds).min(start + chunk_size);
        let stub_is_right = right == index;
        let (lowest, highest) = if stub_is_right {
            (
                (highest_valid - SPLIT_SEARCH_SECONDS).max(lowest_valid),
                highest_valid,
            )
        } else {
            (
                lowest_valid,
                (lowest_valid + SPLIT_SEARCH_SECONDS).min(highest_valid),
            )
        };
        if lowest <= highest {
            let boundary = quietest_point(audio, lowest, highest, stub_is_right);
            chunks[left].end_seconds = boundary;
            chunks[right].start_seconds = boundary;
        }
    }
    chunks
}

/// Centre of the lowest-energy window in `[lowest, highest]`. Ties go to the end
/// nearest the original boundary: `highest` when `prefer_late`, else `lowest`.
fn quietest_point(audio: &LoadedAudio, lowest: f64, highest: f64, prefer_late: bool) -> f64 {
    let fallback = if prefer_late { highest } else { lowest };
    let rate = f64::from(audio.sample_rate);
    let channels = usize::from(audio.channels.max(1));
    let frames = audio.samples.len() / channels;
    if rate <= 0.0 || frames == 0 {
        return fallback;
    }
    let half = ((SPLIT_ENERGY_WINDOW_SECONDS * rate) as usize / 2).max(1);
    let first = (lowest * rate).ceil() as usize;
    let last = ((highest * rate).floor() as usize).min(frames - 1);
    if first > last {
        return fallback;
    }
    let energy = |centre: usize| {
        let from = centre.saturating_sub(half);
        let to = (centre + half).min(frames);
        audio.samples[from * channels..to * channels]
            .iter()
            .map(|sample| f64::from(*sample).powi(2))
            .sum::<f64>()
            / (to - from).max(1) as f64
    };
    let mut best = if prefer_late { last } else { first };
    let mut best_energy = energy(best);
    let mut centre = first;
    while centre <= last {
        let value = energy(centre);
        if value < best_energy || (prefer_late && value == best_energy) {
            best = centre;
            best_energy = value;
        }
        centre += half;
    }
    (best as f64 / rate).clamp(lowest, highest)
}

/// WhisperX `merge_chunks`: greedily joins consecutive speech spans while the
/// joined span stays within `chunk_size` seconds. A single longer span is kept.
pub fn merge_whisperx_vad_chunks(
    segments: Vec<SpeechSpan>,
    chunk_size: f64,
) -> Result<Vec<SpeechSpan>> {
    let Some(first) = segments.first() else {
        return Ok(Vec::new());
    };
    let mut merged = Vec::new();
    let mut curr_start = first.start_seconds;
    let mut curr_end = 0.0;
    let mut curr_score = first.score;

    for segment in segments {
        if segment.end_seconds - curr_start > chunk_size && curr_end - curr_start > 0.0 {
            merged.push(SpeechSpan::new(curr_start, curr_end, curr_score)?);
            curr_start = segment.start_seconds;
            curr_score = segment.score;
        } else {
            curr_score = curr_score.max(segment.score);
        }
        curr_end = segment.end_seconds;
    }

    merged.push(SpeechSpan::new(curr_start, curr_end, curr_score)?);
    Ok(merged)
}

pub(crate) fn ensure_pipeline_active(observer: &dyn TranscriptionPipelineObserver) -> Result<()> {
    if observer.cancellation_requested() {
        return Err(media_core::DetectError::InvalidArgument(
            "transcription cancelled at a safe workflow boundary".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn provider_language(provider: &TranscriptionProviderSelection) -> Option<String> {
    match provider {
        TranscriptionProviderSelection::CandleWhisper(options) => options.language.clone(),
        TranscriptionProviderSelection::WhisperCpp(options) => options.language.clone(),
        TranscriptionProviderSelection::ExternalWhisperX(options) => options.language.clone(),
    }
}

pub(crate) fn validate_batch_options_for_provider(
    provider: &TranscriptionProviderSelection,
) -> Result<()> {
    if let TranscriptionProviderSelection::CandleWhisper(options) = provider {
        validate_candle_batch_options(options)?;
    }
    Ok(())
}

pub(crate) fn validate_task_options_for_request(
    request: &TranscriptionPipelineRequest,
) -> Result<()> {
    let task = request.provider.task();
    if task == TranscriptionTask::Translate && request.alignment.enabled {
        return Err(invalid_request(
            "native Whisper translation output cannot be wav2vec2/CTC-aligned against source-language audio in this implementation",
        ));
    }
    if matches!(
        request.provider,
        TranscriptionProviderSelection::WhisperCpp(_)
    ) && task == TranscriptionTask::Translate
    {
        return Err(invalid_request(
            "Whisper translate is not supported by the whisper.cpp provider in this crate; use candleWhisper or externalWhisperX",
        ));
    }
    Ok(())
}
