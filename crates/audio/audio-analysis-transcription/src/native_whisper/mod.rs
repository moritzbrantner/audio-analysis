mod decode_state;
mod features;
mod model;
mod resolution;
mod session;
#[cfg(test)]
mod tests;
mod timestamps;
mod token_decode;

use crate::native_device::{select_native_device, ResolvedNativeDevice};
use crate::{
    invalid_request, setup_error, validate_asr_request, AsrRequest, AsrResponse,
    CandleWhisperComputeType, CandleWhisperOptions, CandleWhisperRuntimeControls,
    CandleWhisperTranscriptionRequestConfig, TranscriptionTask,
};
use candle_core::{DType, Device, Tensor};
use media_core::Result;
use serde::Deserialize;
use std::path::Path;

use decode_state::*;
use features::*;
use model::*;
#[cfg(test)]
pub(crate) use resolution::resolve_whisper_bundle_paths;
use resolution::*;
pub(crate) use resolution::{resolve_whisper_resources, ResourceInspection, WhisperBundlePaths};
use session::*;
use timestamps::*;
use token_decode::*;

#[derive(Debug, Clone, PartialEq, Eq)]
struct WhisperRunSetup {
    model_id: String,
    task: TranscriptionTask,
    language: Option<String>,
    bundle: WhisperBundlePaths,
    model_source: &'static str,
    resolved_device: ResolvedNativeDevice,
    /// `device=`, `deviceRequested=`, `deviceName=` and
    /// `deviceFallbackReason=` facts for the report.
    device_diagnostics: Vec<String>,
    requested_compute_type: CandleWhisperComputeType,
    resolved_compute_type: CandleWhisperComputeType,
    model_weight_dtype: DType,
    model_format: WhisperModelFormat,
}

#[allow(dead_code)]
pub(crate) fn transcribe(
    options: &CandleWhisperOptions,
    request: AsrRequest,
) -> Result<AsrResponse> {
    transcribe_with_load_observer(
        options,
        &CandleWhisperTranscriptionRequestConfig::default(),
        request,
        |_| Ok(()),
    )
}

pub(crate) enum WhisperModelResolutionEvent {
    ResolutionStart,
    #[allow(dead_code)] // emitted only by the optional model-bundles resolver
    DownloadStart,
    #[allow(dead_code)] // emitted only by the optional model-bundles resolver
    DownloadEnd {
        duration_seconds: f64,
    },
    ResolutionEnd {
        source: &'static str,
    },
    LoadStart,
    LoadEnd {
        duration_seconds: f64,
    },
}

pub(crate) fn transcribe_with_load_observer(
    options: &CandleWhisperOptions,
    config: &CandleWhisperTranscriptionRequestConfig,
    request: AsrRequest,
    mut on_resolution: impl FnMut(WhisperModelResolutionEvent) -> Result<()>,
) -> Result<AsrResponse> {
    let setup = WhisperRunSetup::from_options_and_request_with_observer(
        options,
        &config.runtime,
        &request,
        &mut on_resolution,
    )?;
    on_resolution(WhisperModelResolutionEvent::LoadStart)?;
    let load_started = std::time::Instant::now();
    let mut session = CandleWhisperSession::load(setup)?;
    on_resolution(WhisperModelResolutionEvent::LoadEnd {
        duration_seconds: load_started.elapsed().as_secs_f64(),
    })?;
    let resolved_device = session.setup.resolved_device.clone();
    with_decoder_threads(config.runtime.decoder_threads, &resolved_device, || {
        session.transcribe_chunks(options, config, request)
    })
}

fn with_decoder_threads<T: Send>(
    decoder_threads: Option<usize>,
    resolved_device: &ResolvedNativeDevice,
    run: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    let Some(decoder_threads) = decoder_threads else {
        return run();
    };
    if decoder_threads == 0 {
        return Err(invalid_request(
            "Candle Whisper decoder_threads must be greater than zero",
        ));
    }
    if resolved_device.cuda_active() {
        return run();
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(decoder_threads)
        .build()
        .map_err(|error| {
            setup_error(format!(
                "failed to create a request-scoped Candle Whisper decoder pool with {decoder_threads} threads: {error}"
            ))
        })?;
    pool.install(run)
}

pub(crate) enum ReusableCandleWhisperSessionEvent {
    ResolutionStart,
    DownloadStart,
    DownloadEnd { duration_seconds: f64 },
    ResolutionEnd { source: &'static str },
    LoadStart,
    LoadEnd { duration_seconds: f64 },
    Reuse,
}

pub(crate) struct ReusableCandleWhisperSession {
    session: CandleWhisperSession,
}

impl ReusableCandleWhisperSession {
    pub(crate) fn transcribe(
        current: &mut Option<Self>,
        options: &CandleWhisperOptions,
        config: &CandleWhisperTranscriptionRequestConfig,
        request: AsrRequest,
        mut observe: impl FnMut(ReusableCandleWhisperSessionEvent) -> Result<()>,
    ) -> Result<AsrResponse> {
        let setup = WhisperRunSetup::from_options_and_request_with_observer(
            options,
            &config.runtime,
            &request,
            &mut |event| {
                observe(match event {
                    WhisperModelResolutionEvent::ResolutionStart => {
                        ReusableCandleWhisperSessionEvent::ResolutionStart
                    }
                    WhisperModelResolutionEvent::DownloadStart => {
                        ReusableCandleWhisperSessionEvent::DownloadStart
                    }
                    WhisperModelResolutionEvent::DownloadEnd { duration_seconds } => {
                        ReusableCandleWhisperSessionEvent::DownloadEnd { duration_seconds }
                    }
                    WhisperModelResolutionEvent::ResolutionEnd { source } => {
                        ReusableCandleWhisperSessionEvent::ResolutionEnd { source }
                    }
                    WhisperModelResolutionEvent::LoadEnd { .. } => {
                        unreachable!("setup resolution does not emit load completion")
                    }
                    WhisperModelResolutionEvent::LoadStart => {
                        unreachable!("setup resolution does not emit load start")
                    }
                })
            },
        )?;
        let session_reused = match current.as_ref() {
            Some(existing) if existing.session.setup == setup => true,
            Some(_) | None => {
                observe(ReusableCandleWhisperSessionEvent::LoadStart)?;
                let load_started = std::time::Instant::now();
                *current = Some(Self {
                    session: CandleWhisperSession::load(setup)?,
                });
                observe(ReusableCandleWhisperSessionEvent::LoadEnd {
                    duration_seconds: load_started.elapsed().as_secs_f64(),
                })?;
                false
            }
        };
        if session_reused {
            observe(ReusableCandleWhisperSessionEvent::Reuse)?;
        }
        let session = current
            .as_mut()
            .expect("reusable Candle Whisper session is loaded");
        let resolved_device = session.session.setup.resolved_device.clone();
        let mut response =
            with_decoder_threads(config.runtime.decoder_threads, &resolved_device, || {
                session.session.transcribe_chunks(options, config, request)
            })?;
        response.diagnostics.push(if session_reused {
            "asrModelSession=reused".to_string()
        } else {
            "asrModelSession=loaded".to_string()
        });
        Ok(response)
    }
}

impl WhisperRunSetup {
    #[allow(dead_code)]
    fn from_options_and_request(
        options: &CandleWhisperOptions,
        request: &AsrRequest,
    ) -> Result<Self> {
        Self::from_options_and_request_with_observer(
            options,
            &CandleWhisperRuntimeControls::default(),
            request,
            &mut |_| Ok(()),
        )
    }

    fn from_options_and_request_with_observer(
        options: &CandleWhisperOptions,
        controls: &CandleWhisperRuntimeControls,
        request: &AsrRequest,
        observe: &mut dyn FnMut(WhisperModelResolutionEvent) -> Result<()>,
    ) -> Result<Self> {
        validate_asr_request(request)?;
        if options.compute_type == CandleWhisperComputeType::Int8
            && options.device == crate::NativeDevicePreference::Cuda
        {
            return Err(setup_error(
                "native Candle Whisper compute type int8 is CPU-only; select device=cpu",
            ));
        }
        let device = select_native_device(options.device, controls.cuda_device_index)?;
        let device_diagnostics = device.diagnostics(None);
        let resolved_device = device.resolved;
        let resolved_compute_type = options
            .compute_type
            .resolve_for_device(resolved_device.cuda_active())?;
        let model = resolve_whisper_model_with_observer(
            options,
            &request.model_id,
            resolved_compute_type,
            observe,
        )?;
        let model_weight_dtype = candle_whisper_model_weight_dtype(resolved_compute_type);
        let model_format = if resolved_compute_type == CandleWhisperComputeType::Int8 {
            WhisperModelFormat::GgufQ8_0
        } else {
            WhisperModelFormat::Safetensors
        };
        Ok(Self {
            model_id: model.model_id,
            task: request.task,
            language: request
                .language
                .clone()
                .or_else(|| options.language.clone()),
            bundle: model.bundle,
            model_source: model.source,
            resolved_device,
            device_diagnostics,
            requested_compute_type: options.compute_type,
            resolved_compute_type,
            model_weight_dtype,
            model_format,
        })
    }
}

fn candle_whisper_model_weight_dtype(compute_type: CandleWhisperComputeType) -> DType {
    match compute_type {
        CandleWhisperComputeType::Automatic => unreachable!("compute type must be resolved first"),
        CandleWhisperComputeType::Fp16 => DType::F16,
        CandleWhisperComputeType::Fp32 => DType::F32,
        CandleWhisperComputeType::Int8 => DType::F32,
    }
}

fn candle_dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::F32 => "f32",
        _ => "other",
    }
}

fn prepare_whisper_encoder_input(
    input: Tensor,
    model_weight_dtype: DType,
) -> candle_core::Result<Tensor> {
    input.to_dtype(model_weight_dtype)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, label: &str) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|error| {
        setup_error(format!(
            "failed to read {label} `{}`: {error}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        invalid_request(format!(
            "failed to parse {label} `{}`: {error}",
            path.display()
        ))
    })
}

fn candle_device(resolved: &ResolvedNativeDevice) -> Result<Device> {
    resolved.candle_device()
}

fn device_is_cuda(resolved: &ResolvedNativeDevice) -> bool {
    resolved.cuda_active()
}

fn should_microbatch_encoder(resolved: &ResolvedNativeDevice, window_count: usize) -> bool {
    window_count > 1 && device_is_cuda(resolved)
}
