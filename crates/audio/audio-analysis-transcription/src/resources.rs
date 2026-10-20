//! Read-only preflight through the same resource and device rules as execution.
use std::path::PathBuf;

use media_core::Result;
use serde::{Deserialize, Serialize};

use crate::{
    AlignmentOptions, CandleWhisperComputeType, CandleWhisperOptions, CandleWhisperRuntimeControls,
};

/// A resolved device. Automatic selection has already been evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NativeResourceDevice {
    Cpu,
    Cuda { index: usize },
}

/// Local resource availability, not a guarantee that a model can execute.
///
/// Paths are canonical. Inspection does not parse model configuration or open,
/// map, or validate tensor data. Files may still be corrupt or incompatible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeModelResourceInspection {
    pub model_id: String,
    pub bundle_path: PathBuf,
    pub required_files: Vec<PathBuf>,
    pub device: NativeResourceDevice,
}

/// Whisper resource availability with the resolved compute type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandleWhisperResourceInspection {
    pub model: NativeModelResourceInspection,
    pub compute_type: CandleWhisperComputeType,
}

/// Resolves cached or explicit Whisper resources without downloading or loading
/// a model, regardless of `options.model_cache_only`.
///
/// Explicit unavailable devices fail; `Auto` follows normal execution device
/// selection. Device probing may initialize the CUDA driver, but no model or
/// inference session is constructed. No audio input is required or decoded.
pub fn inspect_candle_whisper_resources(
    options: &CandleWhisperOptions,
    controls: CandleWhisperRuntimeControls,
) -> Result<CandleWhisperResourceInspection> {
    #[cfg(feature = "candle")]
    {
        let device = crate::native_device::resolve_native_device(
            options.device,
            controls.cuda_device_index,
        )?;
        let compute_type = options
            .compute_type
            .resolve_for_device(device.cuda_active())?;
        let mut cached = options.clone();
        cached.model_cache_only = true;
        let resolved = crate::native_whisper::resolve_whisper_resources(
            &cached,
            &cached.model_id,
            compute_type,
            crate::native_whisper::ResourceInspection::FilesOnly,
            &mut |_| Ok(()),
        )?;
        let bundle = resolved.bundle;
        let weights = bundle.model_q8_0_gguf.unwrap_or(bundle.model_safetensors);
        Ok(CandleWhisperResourceInspection {
            model: inspect_paths(
                resolved.model_id,
                bundle.root,
                vec![
                    bundle.config_json,
                    bundle.generation_config_json,
                    bundle.tokenizer_json,
                    bundle.preprocessor_config_json,
                    weights,
                ],
                device,
            )?,
            compute_type,
        })
    }
    #[cfg(not(feature = "candle"))]
    {
        let _ = (options, controls);
        Err(crate::setup_error(
            "Whisper resource inspection requires the `candle` feature",
        ))
    }
}

/// Resolves enabled alignment resources without downloading, parsing weight
/// headers, or loading a model. Disabled alignment returns `None` even when the
/// binary lacks alignment support.
pub fn inspect_alignment_resources(
    options: &AlignmentOptions,
) -> Result<Option<NativeModelResourceInspection>> {
    if !options.enabled {
        return Ok(None);
    }
    #[cfg(feature = "alignment")]
    {
        let device = crate::native_device::resolve_native_device(options.device, 0)?;
        let mut cached = options.clone();
        cached.model_cache_only = true;
        let resolved = crate::ctc_alignment::resolve_alignment_model(&cached, &cached.model_id)?;
        let bundle = crate::native_wav2vec2::resolve_wav2vec2_bundle_paths(&resolved.bundle)?;
        inspect_paths(
            resolved.model_id,
            resolved.bundle,
            vec![
                bundle.config_json,
                bundle.tokenizer_vocab,
                bundle.preprocessor_config_json,
                bundle.model_safetensors,
            ],
            device,
        )
        .map(Some)
    }
    #[cfg(not(feature = "alignment"))]
    {
        Err(crate::setup_error(
            "Alignment resource inspection requires the `alignment` feature",
        ))
    }
}

#[cfg(feature = "candle")]
fn inspect_paths(
    model_id: String,
    bundle_path: PathBuf,
    files: Vec<PathBuf>,
    device: crate::native_device::ResolvedNativeDevice,
) -> Result<NativeModelResourceInspection> {
    let canonical = |path: PathBuf| {
        path.canonicalize().map_err(|error| {
            crate::setup_error(format!(
                "cannot resolve local model resource `{}`: {error}",
                path.display()
            ))
        })
    };
    let required_files = files
        .into_iter()
        .map(|path| {
            if !path.is_file() {
                return Err(crate::setup_error(format!(
                    "model resource `{}` is not a regular file",
                    path.display()
                )));
            }
            canonical(path)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(NativeModelResourceInspection {
        model_id,
        bundle_path: canonical(bundle_path)?,
        required_files,
        device: match device {
            crate::native_device::ResolvedNativeDevice::Cpu => NativeResourceDevice::Cpu,
            crate::native_device::ResolvedNativeDevice::Cuda(index) => {
                NativeResourceDevice::Cuda { index }
            }
        },
    })
}
