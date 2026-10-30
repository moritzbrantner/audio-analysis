//! Candle Whisper transcription providers and their setup validation.

use crate::*;

pub(crate) const CANDLE_WHISPER_AUTOREGRESSIVE_KV_CACHE_EXECUTION: &str =
    "candle-whisper-autoregressive-kv-cache";
pub(crate) const CANDLE_WHISPER_ACTIVE_ROW_TENSOR_BATCH_EXECUTION: &str =
    "candle-whisper-active-row-tensor-batch";

/// Feature-gated Candle Whisper provider.
#[derive(Debug, Clone, Default)]
pub struct CandleWhisperTranscriber {
    pub options: CandleWhisperOptions,
}

impl CandleWhisperTranscriber {
    pub fn new(options: CandleWhisperOptions) -> Self {
        Self { options }
    }

    /// Transcribes one request with complete request-scoped native controls.
    pub fn transcribe_with_request_config(
        &mut self,
        request: AsrRequest,
        config: CandleWhisperTranscriptionRequestConfig,
    ) -> Result<AsrResponse> {
        let mut observer = NoopTranscriptionPipelineObserver;
        self.transcribe_with_request_config_and_observer(request, config, &mut observer)
    }

    /// Transcribes one request with complete request-scoped native controls
    /// while preserving pipeline progress and cancellation observations.
    pub fn transcribe_with_request_config_and_observer(
        &mut self,
        request: AsrRequest,
        config: CandleWhisperTranscriptionRequestConfig,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AsrResponse> {
        validate_asr_request(&request)?;
        config.validate(&self.options)?;
        validate_candle_setup(&self.options, &config.runtime)?;
        let _ = &observer;
        #[cfg(feature = "candle")]
        {
            let chunk_count = request.chunks.len();
            let model_id = request.model_id.clone();
            let mut response = native_whisper::transcribe_with_load_observer(
                &self.options,
                &config,
                request,
                |event| {
                    match event {
                        native_whisper::WhisperModelResolutionEvent::ResolutionStart => {
                            observer.model_resolution_start("asr", "candle-whisper", &model_id);
                        }
                        native_whisper::WhisperModelResolutionEvent::ResolutionEnd { source } => {
                            observer.model_resolution_end(
                                "asr",
                                "candle-whisper",
                                &model_id,
                                source,
                            );
                        }
                        native_whisper::WhisperModelResolutionEvent::DownloadStart => {
                            observer.model_download_start("asr", "hugging-face", &model_id);
                        }
                        native_whisper::WhisperModelResolutionEvent::DownloadEnd {
                            duration_seconds,
                        } => observer.model_download_end(
                            "asr",
                            "hugging-face",
                            &model_id,
                            duration_seconds,
                        ),
                        native_whisper::WhisperModelResolutionEvent::LoadStart => {
                            observer.observe(TranscriptionPipelineEvent::ModelLoadStart {
                                stage: "asr".to_string(),
                                provider: "candle-whisper".to_string(),
                                model_id: model_id.clone(),
                            });
                        }
                        native_whisper::WhisperModelResolutionEvent::LoadEnd {
                            duration_seconds,
                        } => observer.observe(TranscriptionPipelineEvent::ModelLoadEnd {
                            stage: "asr".to_string(),
                            provider: "candle-whisper".to_string(),
                            model_id: model_id.clone(),
                            duration_seconds,
                        }),
                    }
                    ensure_pipeline_active(observer)
                },
            )?;
            extend_missing_candle_batch_diagnostics(
                &mut response.diagnostics,
                &self.options,
                chunk_count,
            );
            Ok(response)
        }
        #[cfg(not(feature = "candle"))]
        {
            Err(unsupported_runtime(format!(
                "Candle Whisper requested for `{}` but the binary lacks the `candle` feature; {}; build with `candle` for native execution and `model-bundles` for Hugging Face cache resolution",
                request.model_id,
                candle_whisper_setup_context(&self.options)
            )))
        }
    }

    /// Transcribes one request with request-scoped native runtime controls.
    pub fn transcribe_with_runtime_controls(
        &mut self,
        request: AsrRequest,
        controls: CandleWhisperRuntimeControls,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                runtime: controls,
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with request-scoped token selection controls.
    pub fn transcribe_with_decode_config(
        &mut self,
        request: AsrRequest,
        decode: CandleWhisperDecodeConfig,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                decode: decode.into(),
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with request-scoped runtime and token selection controls.
    pub fn transcribe_with_runtime_controls_and_decode_config(
        &mut self,
        request: AsrRequest,
        controls: CandleWhisperRuntimeControls,
        decode: CandleWhisperDecodeConfig,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                runtime: controls,
                decode: decode.into(),
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with complete request-scoped decode controls.
    pub fn transcribe_with_decode_request_config(
        &mut self,
        request: AsrRequest,
        decode: CandleWhisperDecodeRequestConfig,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                decode,
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with runtime and complete decode controls.
    pub fn transcribe_with_runtime_controls_and_decode_request_config(
        &mut self,
        request: AsrRequest,
        controls: CandleWhisperRuntimeControls,
        decode: CandleWhisperDecodeRequestConfig,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                runtime: controls,
                decode,
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with complete request-scoped controls while
    /// preserving pipeline progress and cancellation observations.
    pub fn transcribe_with_runtime_controls_and_decode_request_config_and_observer(
        &mut self,
        request: AsrRequest,
        controls: CandleWhisperRuntimeControls,
        decode: CandleWhisperDecodeRequestConfig,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config_and_observer(
            request,
            CandleWhisperTranscriptionRequestConfig {
                runtime: controls,
                decode,
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
            observer,
        )
    }
}

impl AudioTranscriptionProvider for CandleWhisperTranscriber {
    fn provider_id(&self) -> &str {
        "candle-whisper"
    }

    fn transcribe(&mut self, request: AsrRequest) -> Result<AsrResponse> {
        let mut observer = NoopTranscriptionPipelineObserver;
        self.transcribe_with_observer(request, &mut observer)
    }

    fn transcribe_with_observer(
        &mut self,
        request: AsrRequest,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config_and_observer(
            request,
            CandleWhisperTranscriptionRequestConfig::default(),
            observer,
        )
    }
}

/// Candle Whisper provider that keeps a compatible native model session loaded
/// across transcription requests.
///
/// Downstream callers can use this as the public provider-reuse primitive with
/// `run_transcription_pipeline_with_observer`. Compatible repeated requests
/// emit `TranscriptionPipelineEvent::ModelReuse` and include response
/// diagnostics such as `asrModelSession=reused` when a loaded session is reused.
#[derive(Default)]
pub struct ReusableCandleWhisperTranscriber {
    pub options: CandleWhisperOptions,
    #[cfg(feature = "candle")]
    session: Option<native_whisper::ReusableCandleWhisperSession>,
}

impl ReusableCandleWhisperTranscriber {
    pub fn new(options: CandleWhisperOptions) -> Self {
        Self {
            options,
            #[cfg(feature = "candle")]
            session: None,
        }
    }

    /// Transcribes one request with complete request-scoped native controls.
    pub fn transcribe_with_request_config(
        &mut self,
        request: AsrRequest,
        config: CandleWhisperTranscriptionRequestConfig,
    ) -> Result<AsrResponse> {
        let mut observer = NoopTranscriptionPipelineObserver;
        self.transcribe_with_request_config_and_observer(request, config, &mut observer)
    }

    /// Transcribes one request with complete request-scoped native controls
    /// while preserving pipeline progress, reuse, and cancellation observations.
    pub fn transcribe_with_request_config_and_observer(
        &mut self,
        request: AsrRequest,
        config: CandleWhisperTranscriptionRequestConfig,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AsrResponse> {
        validate_asr_request(&request)?;
        config.validate(&self.options)?;
        validate_candle_setup(&self.options, &config.runtime)?;
        let _ = &observer;
        #[cfg(feature = "candle")]
        {
            let chunk_count = request.chunks.len();
            let model_id = request.model_id.clone();
            let mut response = native_whisper::ReusableCandleWhisperSession::transcribe(
                &mut self.session,
                &self.options,
                &config,
                request,
                |event| {
                    match event {
                        native_whisper::ReusableCandleWhisperSessionEvent::ResolutionStart => {
                            observer.model_resolution_start("asr", "candle-whisper", &model_id);
                        }
                        native_whisper::ReusableCandleWhisperSessionEvent::ResolutionEnd {
                            source,
                        } => {
                            observer.model_resolution_end(
                                "asr",
                                "candle-whisper",
                                &model_id,
                                source,
                            );
                        }
                        native_whisper::ReusableCandleWhisperSessionEvent::DownloadStart => {
                            observer.model_download_start("asr", "hugging-face", &model_id);
                        }
                        native_whisper::ReusableCandleWhisperSessionEvent::DownloadEnd {
                            duration_seconds,
                        } => {
                            observer.model_download_end(
                                "asr",
                                "hugging-face",
                                &model_id,
                                duration_seconds,
                            );
                        }
                        native_whisper::ReusableCandleWhisperSessionEvent::LoadStart => {
                            observer.observe(TranscriptionPipelineEvent::ModelLoadStart {
                                stage: "asr".to_string(),
                                provider: "candle-whisper".to_string(),
                                model_id: model_id.clone(),
                            });
                        }
                        native_whisper::ReusableCandleWhisperSessionEvent::LoadEnd {
                            duration_seconds,
                        } => {
                            observer.observe(TranscriptionPipelineEvent::ModelLoadEnd {
                                stage: "asr".to_string(),
                                provider: "candle-whisper".to_string(),
                                model_id: model_id.clone(),
                                duration_seconds,
                            });
                        }
                        native_whisper::ReusableCandleWhisperSessionEvent::Reuse => {
                            observer.observe(TranscriptionPipelineEvent::ModelReuse {
                                stage: "asr".to_string(),
                                provider: "candle-whisper".to_string(),
                                model_id: model_id.clone(),
                            });
                        }
                    }
                    ensure_pipeline_active(observer)
                },
            )?;
            extend_missing_candle_batch_diagnostics(
                &mut response.diagnostics,
                &self.options,
                chunk_count,
            );
            Ok(response)
        }
        #[cfg(not(feature = "candle"))]
        {
            Err(unsupported_runtime(format!(
                "Candle Whisper requested for `{}` but the binary lacks the `candle` feature; {}; build with `candle` for native execution and `model-bundles` for Hugging Face cache resolution",
                request.model_id,
                candle_whisper_setup_context(&self.options)
            )))
        }
    }

    /// Transcribes one request with request-scoped native runtime controls.
    pub fn transcribe_with_runtime_controls(
        &mut self,
        request: AsrRequest,
        controls: CandleWhisperRuntimeControls,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                runtime: controls,
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with request-scoped token selection controls.
    pub fn transcribe_with_decode_config(
        &mut self,
        request: AsrRequest,
        decode: CandleWhisperDecodeConfig,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                decode: decode.into(),
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with request-scoped runtime and token selection controls.
    pub fn transcribe_with_runtime_controls_and_decode_config(
        &mut self,
        request: AsrRequest,
        controls: CandleWhisperRuntimeControls,
        decode: CandleWhisperDecodeConfig,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                runtime: controls,
                decode: decode.into(),
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with complete request-scoped decode controls.
    pub fn transcribe_with_decode_request_config(
        &mut self,
        request: AsrRequest,
        decode: CandleWhisperDecodeRequestConfig,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                decode,
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with runtime and complete decode controls.
    pub fn transcribe_with_runtime_controls_and_decode_request_config(
        &mut self,
        request: AsrRequest,
        controls: CandleWhisperRuntimeControls,
        decode: CandleWhisperDecodeRequestConfig,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config(
            request,
            CandleWhisperTranscriptionRequestConfig {
                runtime: controls,
                decode,
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
        )
    }

    /// Transcribes one request with complete request-scoped controls while
    /// preserving pipeline progress, reuse, and cancellation observations.
    pub fn transcribe_with_runtime_controls_and_decode_request_config_and_observer(
        &mut self,
        request: AsrRequest,
        controls: CandleWhisperRuntimeControls,
        decode: CandleWhisperDecodeRequestConfig,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config_and_observer(
            request,
            CandleWhisperTranscriptionRequestConfig {
                runtime: controls,
                decode,
                ..CandleWhisperTranscriptionRequestConfig::default()
            },
            observer,
        )
    }
}

impl std::fmt::Debug for ReusableCandleWhisperTranscriber {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReusableCandleWhisperTranscriber")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl AudioTranscriptionProvider for ReusableCandleWhisperTranscriber {
    fn provider_id(&self) -> &str {
        "candle-whisper"
    }

    fn transcribe(&mut self, request: AsrRequest) -> Result<AsrResponse> {
        let mut observer = NoopTranscriptionPipelineObserver;
        self.transcribe_with_observer(request, &mut observer)
    }

    fn transcribe_with_observer(
        &mut self,
        request: AsrRequest,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AsrResponse> {
        self.transcribe_with_request_config_and_observer(
            request,
            CandleWhisperTranscriptionRequestConfig::default(),
            observer,
        )
    }
}

/// Candle Whisper provider with complete request-scoped controls.
///
/// The adapter owns either a one-shot or reusable Candle provider and applies
/// the current request configuration through the ordinary
/// [`AudioTranscriptionProvider`] seam. This keeps runtime, decode, and window
/// control execution in the transcription capability crate while callers
/// compose the provider into their workflow.
pub struct RequestConfiguredCandleWhisperTranscriber {
    inner: RequestConfiguredCandleWhisperInner,
    request_config: CandleWhisperTranscriptionRequestConfig,
}

pub(crate) enum RequestConfiguredCandleWhisperInner {
    OneShot(CandleWhisperTranscriber),
    Reusable(ReusableCandleWhisperTranscriber),
}

impl RequestConfiguredCandleWhisperTranscriber {
    /// Builds a provider for one request lifecycle.
    pub fn new(
        options: CandleWhisperOptions,
        request_config: CandleWhisperTranscriptionRequestConfig,
    ) -> Self {
        Self {
            inner: RequestConfiguredCandleWhisperInner::OneShot(CandleWhisperTranscriber::new(
                options,
            )),
            request_config,
        }
    }

    /// Builds a provider that can reuse a compatible loaded model session.
    pub fn reusable(
        options: CandleWhisperOptions,
        request_config: CandleWhisperTranscriptionRequestConfig,
    ) -> Self {
        Self {
            inner: RequestConfiguredCandleWhisperInner::Reusable(
                ReusableCandleWhisperTranscriber::new(options),
            ),
            request_config,
        }
    }

    /// Returns the provider setup shared across request configurations.
    pub fn options(&self) -> &CandleWhisperOptions {
        match &self.inner {
            RequestConfiguredCandleWhisperInner::OneShot(provider) => &provider.options,
            RequestConfiguredCandleWhisperInner::Reusable(provider) => &provider.options,
        }
    }

    /// Replaces the controls applied to subsequent requests.
    pub fn set_request_config(&mut self, request_config: CandleWhisperTranscriptionRequestConfig) {
        self.request_config = request_config;
    }
}

impl std::fmt::Debug for RequestConfiguredCandleWhisperTranscriber {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestConfiguredCandleWhisperTranscriber")
            .field("options", self.options())
            .field("request_config", &self.request_config)
            .finish_non_exhaustive()
    }
}

impl AudioTranscriptionProvider for RequestConfiguredCandleWhisperTranscriber {
    fn provider_id(&self) -> &str {
        "candle-whisper"
    }

    fn transcribe(&mut self, request: AsrRequest) -> Result<AsrResponse> {
        let mut observer = NoopTranscriptionPipelineObserver;
        self.transcribe_with_observer(request, &mut observer)
    }

    fn transcribe_with_observer(
        &mut self,
        request: AsrRequest,
        observer: &mut dyn TranscriptionPipelineObserver,
    ) -> Result<AsrResponse> {
        match &mut self.inner {
            RequestConfiguredCandleWhisperInner::OneShot(provider) => provider
                .transcribe_with_request_config_and_observer(
                    request,
                    self.request_config.clone(),
                    observer,
                ),
            RequestConfiguredCandleWhisperInner::Reusable(provider) => provider
                .transcribe_with_request_config_and_observer(
                    request,
                    self.request_config.clone(),
                    observer,
                ),
        }
    }
}

pub(crate) fn candle_whisper_setup_context(options: &CandleWhisperOptions) -> String {
    let model_location = options
        .model_bundle
        .as_ref()
        .map(|path| format!("--whisper-bundle={}", path.display()))
        .or_else(|| {
            options
                .model_dir
                .as_ref()
                .map(|path| format!("--model-dir={}", path.display()))
        })
        .unwrap_or_else(|| "--model-dir=<default huggingface cache>".to_string());
    format!("{model_location}; cache-only={}", options.model_cache_only)
}

pub(crate) fn validate_candle_batch_options(options: &CandleWhisperOptions) -> Result<()> {
    if options.max_batch_size == Some(0) {
        return Err(invalid_request(
            "Candle Whisper max_batch_size must be greater than zero",
        ));
    }
    if matches!(
        options.decode_runtime,
        CandleWhisperDecodeRuntime::ActiveRowTensorBatch
    ) {
        if !options.batch_chunks {
            return Err(invalid_request(
                "Candle Whisper activeRowTensorBatch decodeRuntime requires batch_chunks=true",
            ));
        }
        if options.max_batch_size == Some(1) {
            return Err(invalid_request(
                "Candle Whisper activeRowTensorBatch decodeRuntime requires max_batch_size greater than one or unbounded batching",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_candle_runtime_controls(
    options: &CandleWhisperOptions,
    controls: &CandleWhisperRuntimeControls,
) -> Result<()> {
    if controls.decoder_threads == Some(0) {
        return Err(invalid_request(
            "Candle Whisper decoder_threads must be greater than zero",
        ));
    }
    if options.device == NativeDevicePreference::Cpu && controls.cuda_device_index != 0 {
        return Err(invalid_request(format!(
            "Candle Whisper cuda_device_index {} cannot be used with device=cpu; use cuda_device_index=0 for CPU execution or select device=cuda",
            controls.cuda_device_index
        )));
    }
    Ok(())
}

pub(crate) fn candle_batch_count(options: &CandleWhisperOptions, chunk_count: usize) -> usize {
    if chunk_count == 0 {
        return 0;
    }
    if !options.batch_chunks {
        return chunk_count;
    }
    match options.max_batch_size {
        Some(max_batch_size) => chunk_count.div_ceil(max_batch_size),
        None => 1,
    }
}

pub(crate) fn candle_batch_diagnostics(
    options: &CandleWhisperOptions,
    chunk_count: usize,
) -> Vec<String> {
    vec![
        format!("chunkCount={chunk_count}"),
        format!("batchChunks={}", options.batch_chunks),
        format!(
            "maxBatchSize={}",
            options
                .max_batch_size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unbounded".to_string())
        ),
        format!("batchCount={}", candle_batch_count(options, chunk_count)),
        format!("batchExecution={CANDLE_WHISPER_AUTOREGRESSIVE_KV_CACHE_EXECUTION}"),
    ]
}

pub(crate) fn extend_missing_candle_batch_diagnostics(
    diagnostics: &mut Vec<String>,
    options: &CandleWhisperOptions,
    chunk_count: usize,
) {
    for diagnostic in candle_batch_diagnostics(options, chunk_count) {
        let Some((key, _)) = diagnostic.split_once('=') else {
            diagnostics.push(diagnostic);
            continue;
        };
        let prefix = format!("{key}=");
        if diagnostics.iter().any(|item| item.starts_with(&prefix)) {
            continue;
        }
        diagnostics.push(diagnostic);
    }
}

pub(crate) fn validate_candle_setup(
    options: &CandleWhisperOptions,
    controls: &CandleWhisperRuntimeControls,
) -> Result<()> {
    validate_candle_batch_options(options)?;
    validate_candle_runtime_controls(options, controls)?;
    let resolved_device =
        native_device::resolve_native_device(options.device, controls.cuda_device_index)?;
    options
        .compute_type
        .resolve_for_device(resolved_device.cuda_active())?;
    if !cfg!(feature = "candle") {
        return Err(unsupported_runtime(format!(
            "Candle Whisper requested but the binary lacks the `candle` feature; {}; build with `candle` for native execution and `model-bundles` for Hugging Face cache resolution",
            candle_whisper_setup_context(options)
        )));
    }
    if let Some(bundle) = &options.model_bundle {
        validate_model_bundle_files(bundle, options.compute_type.required_bundle_files())
    } else {
        Ok(())
    }
}
