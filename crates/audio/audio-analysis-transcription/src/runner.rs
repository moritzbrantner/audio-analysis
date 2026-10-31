//! Reusable native transcription runner.

use crate::*;

/// Native runner configuration for repeated finite transcription requests.
///
/// The options describe the provider stack the runner owns. Requests passed to
/// [`NativeTranscriptionRunner::run`] must use the same provider, VAD,
/// alignment, and diarization options so the runner can safely reuse provider
/// state across different sources without changing pipeline semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeTranscriptionRunnerOptions {
    pub provider: TranscriptionProviderSelection,
    #[serde(default)]
    pub vad: TranscriptionVadOptions,
    #[serde(default)]
    pub alignment: AlignmentOptions,
    #[serde(default)]
    pub diarization: DiarizationOptions,
}

impl NativeTranscriptionRunnerOptions {
    /// Builds runner options from the reusable parts of a pipeline request.
    pub fn from_request(request: &TranscriptionPipelineRequest) -> Self {
        Self {
            provider: request.provider.clone(),
            vad: request.vad,
            alignment: request.alignment.clone(),
            diarization: request.diarization.clone(),
        }
    }
}

impl Default for NativeTranscriptionRunnerOptions {
    fn default() -> Self {
        Self {
            provider: TranscriptionProviderSelection::CandleWhisper(CandleWhisperOptions::default()),
            vad: TranscriptionVadOptions::default(),
            alignment: AlignmentOptions::default(),
            diarization: DiarizationOptions::default(),
        }
    }
}

/// Owning native transcription runner for repeated finite requests.
///
/// Use this when callers want the crate to own native provider/session
/// lifecycle across multiple compatible requests. Advanced callers can still
/// call [`run_transcription_pipeline_with_observer`] directly with their own
/// provider trait implementations.
pub struct NativeTranscriptionRunner {
    pub(crate) options: NativeTranscriptionRunnerOptions,
    mode: NativeTranscriptionRunnerMode,
    vad_provider: Box<dyn VoiceActivityDetector>,
    pub(crate) asr_provider: Box<dyn AudioTranscriptionProvider>,
    alignment_provider: Option<Box<dyn ForcedAlignmentProvider>>,
    diarization_provider: Option<Box<dyn TranscriptDiarizationProvider>>,
}

/// Backend-agnostic reusable session for sequential finite transcription requests.
///
/// This is the preferred public name for the reusable pipeline lifecycle. The
/// legacy native name remains available for source compatibility.
pub type ReusableTranscriptionSession = NativeTranscriptionRunner;

/// Configuration for a [`ReusableTranscriptionSession`].
pub type ReusableTranscriptionSessionOptions = NativeTranscriptionRunnerOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeTranscriptionRunnerMode {
    DefaultStack,
    CallerProviders,
}

impl NativeTranscriptionRunner {
    /// Builds the default native provider stack for the configured provider.
    ///
    /// Candle Whisper requests use [`ReusableCandleWhisperTranscriber`] so
    /// compatible repeated requests can report reuse through
    /// [`TranscriptionPipelineEvent::ModelReuse`] and response diagnostics such
    /// as `asrModelSession=reused`.
    pub fn new(options: NativeTranscriptionRunnerOptions) -> Result<Self> {
        Self::default_stack_for_options(options)
    }

    fn default_stack_for_options(options: NativeTranscriptionRunnerOptions) -> Result<Self> {
        let asr_provider: Box<dyn AudioTranscriptionProvider> = match &options.provider {
            TranscriptionProviderSelection::CandleWhisper(provider_options) => Box::new(
                ReusableCandleWhisperTranscriber::new(provider_options.clone()),
            ),
            TranscriptionProviderSelection::WhisperCpp(provider_options) => {
                Box::new(WhisperCppTranscriber {
                    options: provider_options.clone(),
                })
            }
            TranscriptionProviderSelection::ExternalWhisperX(_) => {
                return Err(invalid_request(
                    "native transcription runner does not execute external WhisperX command providers",
                ));
            }
        };

        let alignment_provider = options.alignment.enabled.then(|| {
            Box::new(CtcForcedAligner {
                options: options.alignment.clone(),
            }) as Box<dyn ForcedAlignmentProvider>
        });

        #[cfg(feature = "diarization")]
        let diarization_provider = options.diarization.enabled.then(|| {
            Box::new(NativeSpeakerDiarizationProvider) as Box<dyn TranscriptDiarizationProvider>
        });
        #[cfg(not(feature = "diarization"))]
        let diarization_provider = None;

        Ok(Self {
            options,
            mode: NativeTranscriptionRunnerMode::DefaultStack,
            vad_provider: Box::new(EnergyVoiceActivityDetector),
            asr_provider,
            alignment_provider,
            diarization_provider,
        })
    }

    fn rebuild_default_stack(&mut self, options: NativeTranscriptionRunnerOptions) -> Result<()> {
        *self = Self::default_stack_for_options(options)?;
        Ok(())
    }

    /// Builds a runner from caller-provided provider adapters.
    ///
    /// This keeps the customization seam at the existing provider traits rather
    /// than introducing a separate test-only abstraction.
    pub fn from_providers(
        options: NativeTranscriptionRunnerOptions,
        vad_provider: Box<dyn VoiceActivityDetector>,
        asr_provider: Box<dyn AudioTranscriptionProvider>,
        alignment_provider: Option<Box<dyn ForcedAlignmentProvider>>,
        diarization_provider: Option<Box<dyn TranscriptDiarizationProvider>>,
    ) -> Self {
        Self {
            options,
            mode: NativeTranscriptionRunnerMode::CallerProviders,
            vad_provider,
            asr_provider,
            alignment_provider,
            diarization_provider,
        }
    }

    /// Runs a request through the owned provider stack.
    ///
    /// Default runners rebuild their crate-owned providers when request
    /// options change. Runners built from caller-provided adapters still
    /// require exact option compatibility because the runner cannot rebuild
    /// external provider state.
    pub fn run(
        &mut self,
        request: TranscriptionPipelineRequest,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<TranscriptionPipelineResponse> {
        self.prepare_for_request(&request)?;
        let alignment_provider = self
            .alignment_provider
            .as_mut()
            .map(|provider| provider.as_mut() as &mut dyn ForcedAlignmentProvider);
        let diarization_provider = self
            .diarization_provider
            .as_mut()
            .map(|provider| provider.as_mut() as &mut dyn TranscriptDiarizationProvider);
        run_transcription_pipeline_with_observer(
            request,
            self.vad_provider.as_mut(),
            self.asr_provider.as_mut(),
            alignment_provider,
            diarization_provider,
            observer,
        )
    }

    /// Predecodes a selected audio stream, then runs it through the owned provider stack.
    ///
    /// Invalid stream selections return before request compatibility checks can
    /// rebuild crate-owned providers and before the observer receives progress.
    #[cfg(feature = "audio-io")]
    pub fn run_selected_media(
        &mut self,
        request: TranscriptionPipelineRequest,
        audio_stream_index: Option<usize>,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> std::result::Result<TranscriptionPipelineResponse, SelectedMediaTranscriptionError> {
        let request = predecode_selected_media_request(request, audio_stream_index)?;
        self.run(request, observer)
            .map_err(SelectedMediaTranscriptionError::Pipeline)
    }

    pub(crate) fn prepare_for_request(
        &mut self,
        request: &TranscriptionPipelineRequest,
    ) -> Result<()> {
        match self.mode {
            NativeTranscriptionRunnerMode::CallerProviders => {
                self.validate_compatible_request(request)
            }
            NativeTranscriptionRunnerMode::DefaultStack => {
                let request_options = NativeTranscriptionRunnerOptions::from_request(request);
                if request_options != self.options {
                    self.rebuild_default_stack(request_options)?;
                }
                Ok(())
            }
        }
    }

    fn validate_compatible_request(&self, request: &TranscriptionPipelineRequest) -> Result<()> {
        if request.provider != self.options.provider {
            return Err(invalid_request(
                "native transcription runner request provider does not match runner provider",
            ));
        }
        if request.vad != self.options.vad {
            return Err(invalid_request(
                "native transcription runner request VAD options do not match runner options",
            ));
        }
        if request.alignment != self.options.alignment {
            return Err(invalid_request(
                "native transcription runner request alignment options do not match runner options",
            ));
        }
        if request.diarization != self.options.diarization {
            return Err(invalid_request(
                "native transcription runner request diarization options do not match runner options",
            ));
        }
        Ok(())
    }
}
