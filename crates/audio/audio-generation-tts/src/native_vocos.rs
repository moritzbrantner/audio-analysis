use std::fs;
use std::path::{Path, PathBuf};

use audio_contracts::{AudioBuffer, OwnedAudioFrame};
#[cfg(feature = "candle")]
use audio_contracts::{Timebase, Timestamp};
use model_runtime::{ModelBundleManifest, ModelFileRequest, ModelPreset, ModelTask};
use serde::{Deserialize, Serialize};

use crate::{
    NativeTtsDevicePreference, PcmAudio, SpeechSynthesisDiagnostic,
    SpeechSynthesisInferenceControlsReport, SpeechSynthesisOptions, SpeechSynthesisStatus,
};

const DEFAULT_VOCOS_MODEL_ID: &str = "vocos-mel-24khz";
const DEFAULT_INPUT_CHANNELS: usize = 100;
const DEFAULT_SAMPLE_RATE_HZ: u32 = 24_000;
const DEFAULT_HOP_LENGTH: usize = 256;
const DEFAULT_STEPS: u32 = 1;
const DEFAULT_CFG_STRENGTH: f32 = 1.0;
const DEFAULT_SPEED: f32 = 1.0;
const DEFAULT_MAX_DURATION_SECONDS: f32 = 0.05;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NativeVocosVocoderDiagnosticRequest {
    #[serde(default = "default_vocos_model_id")]
    pub model_id: String,
    #[serde(default)]
    pub bundle_path: Option<PathBuf>,
    #[serde(default)]
    pub device: NativeTtsDevicePreference,
    #[serde(default)]
    pub mel: Option<NativeVocosMelInput>,
    #[serde(default)]
    pub options: SpeechSynthesisOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NativeVocosMelInput {
    pub frames: usize,
    pub channels: usize,
    #[serde(default)]
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NativeVocosVocoderDiagnosticOutput {
    pub status: SpeechSynthesisStatus,
    pub provider_id: String,
    pub model_id: String,
    pub audio_generated: bool,
    pub controls: SpeechSynthesisInferenceControlsReport,
    pub device: NativeVocosDeviceReport,
    pub bundle: NativeVocosBundleReport,
    pub mel: NativeVocosMelReport,
    #[serde(default)]
    pub frame: Option<NativeVocosAudioFrameReport>,
    #[serde(default)]
    pub audio: Option<PcmAudio>,
    #[serde(skip)]
    pub audio_frame: Option<OwnedAudioFrame>,
    pub diagnostics: Vec<SpeechSynthesisDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeVocosDeviceReport {
    pub preference: String,
    pub selected: String,
    pub cuda_active: bool,
    pub candle_feature_enabled: bool,
    pub cuda_feature_enabled: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeVocosBundleReport {
    #[serde(default)]
    pub bundle_path: Option<String>,
    pub model_known: bool,
    pub required_files: Vec<String>,
    #[serde(default)]
    pub config_path: Option<String>,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub config: Option<NativeVocosConfigReport>,
    #[serde(default)]
    pub weights: Option<NativeVocosWeightsReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeVocosConfigReport {
    pub architecture: String,
    pub input_channels: usize,
    pub sample_rate_hz: u32,
    pub hop_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeVocosWeightsReport {
    pub bytes: u64,
    pub format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeVocosMelReport {
    pub frames: usize,
    pub channels: usize,
    pub generated: bool,
    pub tensor_shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeVocosAudioFrameReport {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub sample_count: usize,
    pub samples_per_channel: usize,
    pub sample_format: String,
}

#[derive(Debug, Clone)]
struct ResolvedVocosBundle {
    config: PathBuf,
    weights: PathBuf,
    required_files: Vec<String>,
}

impl NativeVocosVocoderDiagnosticRequest {
    fn validate(&self) -> Result<(), String> {
        if self.model_id.trim().is_empty() {
            return Err("invalid request: `modelId` must not be empty".to_string());
        }
        if self
            .bundle_path
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err(
                "invalid request: `bundlePath` must not be empty when provided".to_string(),
            );
        }
        if let Some(mel) = &self.mel {
            mel.validate()?;
        }
        self.options.validate()
    }
}

impl NativeVocosMelInput {
    fn validate(&self) -> Result<(), String> {
        if self.frames == 0 {
            return Err("invalid request: `mel.frames` must be greater than zero".to_string());
        }
        if self.channels == 0 {
            return Err("invalid request: `mel.channels` must be greater than zero".to_string());
        }
        if !self.values.is_empty() && self.values.len() != self.frames * self.channels {
            return Err(
                "invalid request: `mel.values` length must equal mel.frames * mel.channels"
                    .to_string(),
            );
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(
                "invalid request: `mel.values` must contain only finite values".to_string(),
            );
        }
        Ok(())
    }
}

pub fn run_vocos_vocoder_diagnostic(
    request: &NativeVocosVocoderDiagnosticRequest,
) -> Result<NativeVocosVocoderDiagnosticOutput, String> {
    request.validate()?;
    let device = resolve_device(request.device);
    let mut bundle_report = NativeVocosBundleReport {
        bundle_path: request
            .bundle_path
            .as_ref()
            .map(|path| path.display().to_string()),
        model_known: vocos_model_preset(&request.model_id).is_some(),
        required_files: request.model_preset_required_files().unwrap_or_default(),
        ..NativeVocosBundleReport::default()
    };
    let initial_mel = NativeVocosMelReport {
        frames: request.mel.as_ref().map_or(0, |mel| mel.frames),
        channels: request.mel.as_ref().map_or(0, |mel| mel.channels),
        generated: request.mel.is_none(),
        tensor_shape: Vec::new(),
        dtype: "f32".to_string(),
    };

    let Some(preset) = vocos_model_preset(&request.model_id) else {
        return Ok(output(
            SpeechSynthesisStatus::UnsupportedRuntime,
            request,
            device,
            bundle_report,
            initial_mel,
            None,
            vec![diagnostic(
                "vocos_checkpoint_unsupported",
                format!(
                    "unsupported Vocos diagnostic checkpoint `{}`; use `vocos-mel-24khz`",
                    request.model_id
                ),
                Some("F5 and E2 checkpoints are covered by separate provider slices."),
            )],
        ));
    };

    let Some(bundle_path) = &request.bundle_path else {
        return Ok(output(
            SpeechSynthesisStatus::SetupRequired,
            request,
            device,
            bundle_report,
            initial_mel,
            None,
            vec![diagnostic(
                "vocos_bundle_missing",
                "Vocos vocoder diagnostics require an explicit local `bundlePath`.".to_string(),
                Some(
                    "Pass a side-effect-free local bundle containing config.yaml and pytorch_model.bin.",
                ),
            )],
        ));
    };

    let resolved = match resolve_vocos_bundle(bundle_path, preset) {
        Ok(resolved) => resolved,
        Err(message) => {
            return Ok(output(
                SpeechSynthesisStatus::SetupRequired,
                request,
                device,
                bundle_report,
                initial_mel,
                None,
                vec![diagnostic("vocos_bundle_invalid", message, None)],
            ));
        }
    };
    bundle_report.required_files = resolved.required_files.clone();
    bundle_report.config_path = Some(resolved.config.display().to_string());
    bundle_report.weights_path = Some(resolved.weights.display().to_string());

    let config = match load_config(&resolved.config) {
        Ok(config) => config,
        Err(message) => {
            return Ok(output(
                SpeechSynthesisStatus::UnsupportedRuntime,
                request,
                device,
                bundle_report,
                initial_mel,
                None,
                vec![diagnostic("vocos_config_unsupported", message, None)],
            ));
        }
    };
    bundle_report.config = Some(config.clone());

    let weights = match load_weights_report(&resolved.weights) {
        Ok(report) => report,
        Err(message) => {
            return Ok(output(
                SpeechSynthesisStatus::SetupRequired,
                request,
                device,
                bundle_report,
                initial_mel,
                None,
                vec![diagnostic("vocos_weights_invalid", message, None)],
            ));
        }
    };
    bundle_report.weights = Some(weights);

    if !cfg!(feature = "candle") {
        return Ok(output(
            SpeechSynthesisStatus::SetupRequired,
            request,
            device,
            bundle_report,
            initial_mel,
            None,
            vec![diagnostic(
                "vocos_candle_feature_disabled",
                "Vocos vocoder diagnostics require the `candle` feature.".to_string(),
                Some("Rebuild audio-generation-tts with `--features candle`; add `cuda` only when CUDA execution is required."),
            )],
        ));
    }

    let (mel, frame) = match build_audio_frame(request, &config, &device) {
        Ok(output) => output,
        Err(message) => {
            return Ok(output(
                SpeechSynthesisStatus::SetupRequired,
                request,
                device,
                bundle_report,
                initial_mel,
                None,
                vec![diagnostic("vocos_vocoder_failed", message, None)],
            ));
        }
    };

    Ok(output(
        SpeechSynthesisStatus::Ready,
        request,
        device,
        bundle_report,
        mel,
        Some(frame),
        Vec::new(),
    ))
}

fn default_vocos_model_id() -> String {
    DEFAULT_VOCOS_MODEL_ID.to_string()
}

impl NativeVocosVocoderDiagnosticRequest {
    fn model_preset_required_files(&self) -> Option<Vec<String>> {
        vocos_model_preset(&self.model_id).map(required_files_for_preset)
    }
}

fn vocos_model_preset(model_id: &str) -> Option<ModelPreset> {
    match model_id {
        "vocos-mel-24khz" => Some(ModelPreset::VocosMel24Khz),
        _ => None,
    }
}

fn required_files_for_preset(preset: ModelPreset) -> Vec<String> {
    preset
        .spec()
        .files
        .into_iter()
        .filter_map(|file| match file {
            ModelFileRequest::Required(path) => Some(path),
            ModelFileRequest::Optional(_) | ModelFileRequest::FirstAvailable(_) => None,
        })
        .collect()
}

fn output(
    status: SpeechSynthesisStatus,
    request: &NativeVocosVocoderDiagnosticRequest,
    device: NativeVocosDeviceReport,
    bundle: NativeVocosBundleReport,
    mel: NativeVocosMelReport,
    audio_frame: Option<OwnedAudioFrame>,
    diagnostics: Vec<SpeechSynthesisDiagnostic>,
) -> NativeVocosVocoderDiagnosticOutput {
    let mut diagnostics = diagnostics;
    diagnostics.extend(device_diagnostics(&device));
    let audio = audio_frame.as_ref().and_then(pcm_audio_from_frame);
    let frame = audio_frame.as_ref().map(audio_frame_report);
    NativeVocosVocoderDiagnosticOutput {
        status,
        provider_id: "vocos".to_string(),
        model_id: request.model_id.clone(),
        audio_generated: audio_frame.is_some(),
        controls: vocos_controls(&request.options),
        device,
        bundle,
        mel,
        frame,
        audio,
        audio_frame,
        diagnostics,
    }
}

fn vocos_controls(options: &SpeechSynthesisOptions) -> SpeechSynthesisInferenceControlsReport {
    SpeechSynthesisInferenceControlsReport::from_options(
        options,
        DEFAULT_STEPS,
        DEFAULT_CFG_STRENGTH,
        DEFAULT_SPEED,
        DEFAULT_MAX_DURATION_SECONDS,
    )
}

fn pcm_audio_from_frame(frame: &OwnedAudioFrame) -> Option<PcmAudio> {
    let AudioBuffer::F32(samples) = &frame.data else {
        return None;
    };
    Some(PcmAudio {
        sample_rate_hz: frame.sample_rate,
        channels: frame.channels,
        samples: samples.clone(),
    })
}

fn audio_frame_report(frame: &OwnedAudioFrame) -> NativeVocosAudioFrameReport {
    NativeVocosAudioFrameReport {
        sample_rate_hz: frame.sample_rate,
        channels: frame.channels,
        sample_count: frame.data.len(),
        samples_per_channel: frame.samples_per_channel(),
        sample_format: format!("{:?}", frame.sample_format()).to_ascii_lowercase(),
    }
}

fn diagnostic(
    code: impl Into<String>,
    message: impl Into<String>,
    help: Option<&str>,
) -> SpeechSynthesisDiagnostic {
    SpeechSynthesisDiagnostic {
        code: code.into(),
        message: message.into(),
        help: help.map(str::to_string),
    }
}

fn resolve_device(preference: NativeTtsDevicePreference) -> NativeVocosDeviceReport {
    let selected = match preference {
        NativeTtsDevicePreference::Cpu => "cpu".to_string(),
        NativeTtsDevicePreference::Cuda if cfg!(feature = "cuda") && cuda_available() => {
            "cuda:0".to_string()
        }
        NativeTtsDevicePreference::Cuda => "unavailable".to_string(),
        NativeTtsDevicePreference::Auto if cfg!(feature = "cuda") && cuda_available() => {
            "cuda:0".to_string()
        }
        NativeTtsDevicePreference::Auto => "cpu".to_string(),
    };
    NativeVocosDeviceReport {
        preference: preference.as_str().to_string(),
        cuda_active: selected.starts_with("cuda:"),
        selected,
        candle_feature_enabled: cfg!(feature = "candle"),
        cuda_feature_enabled: cfg!(feature = "cuda"),
    }
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    candle_core::Device::new_cuda(0).is_ok()
}

#[cfg(not(feature = "cuda"))]
fn cuda_available() -> bool {
    false
}

fn device_diagnostics(device: &NativeVocosDeviceReport) -> Vec<SpeechSynthesisDiagnostic> {
    match (device.preference.as_str(), device.selected.as_str()) {
        ("auto", "cpu") if device.cuda_feature_enabled => vec![diagnostic(
            "native_tts_cpu_fallback",
            "CUDA-preferred auto device selection fell back to CPU because CUDA was unavailable."
                .to_string(),
            Some("Use provider.device = `cuda` to require CUDA, or keep `auto` for CPU fallback."),
        )],
        ("auto", "cpu") => vec![diagnostic(
            "native_tts_cpu_fallback",
            "Auto device selection used CPU because this build does not enable CUDA.".to_string(),
            Some("Rebuild with `--features cuda` to prefer CUDA when hardware is available."),
        )],
        ("cuda", "unavailable") => vec![diagnostic(
            "native_tts_cuda_unavailable",
            "CUDA was requested but is unavailable to this native TTS build.".to_string(),
            Some("Rebuild with `--features cuda` on a CUDA-capable host, or request `cpu`/`auto`."),
        )],
        _ => Vec::new(),
    }
}

fn resolve_vocos_bundle(bundle: &Path, preset: ModelPreset) -> Result<ResolvedVocosBundle, String> {
    if !bundle.exists() {
        return Err(format!(
            "Vocos bundle path `{}` does not exist",
            bundle.display()
        ));
    }
    if !bundle.is_dir() {
        return Err(format!(
            "Vocos bundle path `{}` must be a directory",
            bundle.display()
        ));
    }
    let spec = preset.spec();
    if spec.task != ModelTask::AudioGeneration {
        return Err(format!(
            "model `{}` is not an audio-generation preset",
            spec.name
        ));
    }
    let manifest = load_manifest(bundle)?;
    let required_files = required_files_for_preset(preset);
    let config_remote = required_files
        .iter()
        .find(|path| path.ends_with("config.yaml") || path.ends_with("config.yml"))
        .ok_or_else(|| format!("Vocos preset `{}` does not declare config.yaml", spec.name))?;
    let weights_remote = required_files
        .iter()
        .find(|path| path.ends_with("pytorch_model.bin"))
        .ok_or_else(|| {
            format!(
                "Vocos preset `{}` does not declare pytorch_model.bin",
                spec.name
            )
        })?;
    let config = resolve_bundle_file(bundle, manifest.as_ref(), config_remote)?;
    let weights = resolve_bundle_file(bundle, manifest.as_ref(), weights_remote)?;
    Ok(ResolvedVocosBundle {
        config,
        weights,
        required_files,
    })
}

fn load_manifest(bundle: &Path) -> Result<Option<ModelBundleManifest>, String> {
    let manifest_path = bundle.join("manifest.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&manifest_path).map_err(|error| {
        format!(
            "failed to read model bundle manifest `{}`: {error}",
            manifest_path.display()
        )
    })?;
    serde_json::from_str(&raw).map(Some).map_err(|error| {
        format!(
            "failed to parse model bundle manifest `{}`: {error}",
            manifest_path.display()
        )
    })
}

fn resolve_bundle_file(
    bundle: &Path,
    manifest: Option<&ModelBundleManifest>,
    remote_path: &str,
) -> Result<PathBuf, String> {
    if let Some(file) = manifest.and_then(|manifest| manifest.files.get(remote_path)) {
        let path = bundle.join(&file.local_path);
        if path.exists() {
            return Ok(path);
        }
    }
    for path in [
        bundle.join(remote_path),
        bundle.join("files").join(remote_path),
    ] {
        if path.exists() {
            return Ok(path);
        }
    }
    Err(format!(
        "Vocos bundle `{}` is missing required file `{remote_path}`",
        bundle.display()
    ))
}

fn load_config(path: &Path) -> Result<NativeVocosConfigReport, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read Vocos config `{}`: {error}", path.display()))?;
    if raw.trim().is_empty() {
        return Err(format!("Vocos config `{}` is empty", path.display()));
    }
    let lower = raw.to_ascii_lowercase();
    if lower.contains("class_path:") && !lower.contains("vocos") {
        return Err(format!(
            "unsupported_runtime: Vocos config `{}` does not reference a Vocos class",
            path.display()
        ));
    }
    let input_channels = yaml_usize(&raw, &["input_channels", "n_mel_channels", "num_mels"])
        .unwrap_or(DEFAULT_INPUT_CHANNELS);
    if input_channels == 0 {
        return Err("unsupported_runtime: input_channels must be greater than zero".to_string());
    }
    let sample_rate_hz = yaml_u32(
        &raw,
        &[
            "sample_rate",
            "sampling_rate",
            "sample_rate_hz",
            "target_sample_rate",
        ],
    )
    .unwrap_or(DEFAULT_SAMPLE_RATE_HZ);
    if sample_rate_hz == 0 {
        return Err("unsupported_runtime: sample rate must be greater than zero".to_string());
    }
    if sample_rate_hz > i32::MAX as u32 {
        return Err(
            "unsupported_runtime: sample rate exceeds supported timebase range".to_string(),
        );
    }
    let hop_length = yaml_usize(&raw, &["hop_length"]).unwrap_or(DEFAULT_HOP_LENGTH);
    if hop_length == 0 {
        return Err("unsupported_runtime: hop_length must be greater than zero".to_string());
    }
    Ok(NativeVocosConfigReport {
        architecture: if lower.contains("vocos") {
            "vocos".to_string()
        } else {
            "vocos-mel".to_string()
        },
        input_channels,
        sample_rate_hz,
        hop_length,
    })
}

fn yaml_usize(raw: &str, keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| yaml_scalar(raw, key)?.parse::<usize>().ok())
}

fn yaml_u32(raw: &str, keys: &[&str]) -> Option<u32> {
    keys.iter()
        .find_map(|key| yaml_scalar(raw, key)?.parse::<u32>().ok())
}

fn yaml_scalar(raw: &str, key: &str) -> Option<String> {
    raw.lines().find_map(|line| {
        let line = line.split('#').next()?.trim();
        let (candidate, value) = line.split_once(':')?;
        if candidate.trim() != key {
            return None;
        }
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if value.is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    })
}

fn load_weights_report(path: &Path) -> Result<NativeVocosWeightsReport, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("failed to read Vocos weights `{}`: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("Vocos weights `{}` must be a file", path.display()));
    }
    if metadata.len() == 0 {
        return Err(format!("Vocos weights `{}` is empty", path.display()));
    }
    Ok(NativeVocosWeightsReport {
        bytes: metadata.len(),
        format: "pytorch_model.bin".to_string(),
    })
}

#[cfg(feature = "candle")]
fn build_audio_frame(
    request: &NativeVocosVocoderDiagnosticRequest,
    config: &NativeVocosConfigReport,
    device: &NativeVocosDeviceReport,
) -> Result<(NativeVocosMelReport, OwnedAudioFrame), String> {
    let candle_device = match device.selected.as_str() {
        "cpu" | "cuda-if-available" => candle_core::Device::Cpu,
        selected if selected.starts_with("cuda") => {
            #[cfg(feature = "cuda")]
            {
                candle_core::Device::new_cuda(0)
                    .map_err(|error| format!("failed to initialize CUDA device: {error}"))?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err("CUDA was requested but the `cuda` feature is not enabled".to_string());
            }
        }
        _ => {
            return Err(format!(
                "device `{}` cannot execute the Vocos diagnostic",
                device.selected
            ));
        }
    };
    let generated = request.mel.is_none();
    let mel = request
        .mel
        .clone()
        .unwrap_or_else(|| generated_mel(config, &request.options));
    if mel.channels != config.input_channels {
        return Err(format!(
            "mel input has {} channels but Vocos config expects {}",
            mel.channels, config.input_channels
        ));
    }
    let controls = vocos_controls(&request.options);
    let values = mel_values(&mel, &controls);
    let tensor =
        candle_core::Tensor::from_vec(values.clone(), (mel.channels, mel.frames), &candle_device)
            .map_err(|error| format!("failed to build Vocos mel tensor: {error}"))?;
    let samples = diagnostic_vocode(
        &values,
        mel.channels,
        mel.frames,
        config.hop_length,
        &controls,
    );
    let frame = OwnedAudioFrame::new(
        Timestamp::new(0, Timebase::new(1, config.sample_rate_hz as i32)),
        config.sample_rate_hz,
        1,
        AudioBuffer::F32(samples),
    )
    .map_err(|error| format!("failed to build Vocos audio frame: {error}"))?;
    Ok((
        NativeVocosMelReport {
            frames: mel.frames,
            channels: mel.channels,
            generated,
            tensor_shape: tensor.dims().to_vec(),
            dtype: format!("{:?}", tensor.dtype()).to_ascii_lowercase(),
        },
        frame,
    ))
}

#[cfg(not(feature = "candle"))]
fn build_audio_frame(
    _request: &NativeVocosVocoderDiagnosticRequest,
    _config: &NativeVocosConfigReport,
    _device: &NativeVocosDeviceReport,
) -> Result<(NativeVocosMelReport, OwnedAudioFrame), String> {
    Err("Vocos audio generation requires the `candle` feature".to_string())
}

#[cfg(feature = "candle")]
fn generated_mel(
    config: &NativeVocosConfigReport,
    options: &SpeechSynthesisOptions,
) -> NativeVocosMelInput {
    let controls = vocos_controls(options);
    let seconds = (controls.max_duration_seconds / controls.speed).clamp(0.01, 0.25);
    let frames = ((seconds * config.sample_rate_hz as f32) / config.hop_length as f32)
        .ceil()
        .max(1.0) as usize;
    NativeVocosMelInput {
        frames,
        channels: config.input_channels,
        values: Vec::new(),
    }
}

#[cfg(feature = "candle")]
fn mel_values(
    mel: &NativeVocosMelInput,
    controls: &SpeechSynthesisInferenceControlsReport,
) -> Vec<f32> {
    if !mel.values.is_empty() {
        return mel
            .values
            .iter()
            .enumerate()
            .map(|(index, value)| controlled_mel_value(*value, index, controls))
            .collect();
    }
    let mut values = Vec::with_capacity(mel.frames * mel.channels);
    for channel in 0..mel.channels {
        for frame in 0..mel.frames {
            let index = channel * mel.frames + frame;
            let base = (((channel + 1) * (frame + 1)) as f32 * 0.013).sin() * 0.5;
            values.push(controlled_mel_value(base, index, controls));
        }
    }
    values
}

#[cfg(feature = "candle")]
fn controlled_mel_value(
    base: f32,
    index: usize,
    controls: &SpeechSynthesisInferenceControlsReport,
) -> f32 {
    let seeded = seeded_unit(controls.seed.unwrap_or(0), index) - 0.5;
    let step_gain = 1.0 + (controls.steps.saturating_sub(1) as f32 * 0.025);
    let cfg_gain = controls.cfg_strength.max(0.01);
    ((base + seeded * 0.1) * step_gain * cfg_gain).clamp(-4.0, 4.0)
}

#[cfg(feature = "candle")]
fn seeded_unit(seed: u64, index: usize) -> f32 {
    let mut state = seed ^ ((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let value = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
    ((value >> 40) as f32) / ((1_u64 << 24) as f32)
}

#[cfg(feature = "candle")]
fn diagnostic_vocode(
    mel_values: &[f32],
    channels: usize,
    frames: usize,
    hop_length: usize,
    controls: &SpeechSynthesisInferenceControlsReport,
) -> Vec<f32> {
    let sample_count = frames * hop_length;
    let mut samples = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let frame = (sample_index / hop_length).min(frames - 1);
        if sample_count > hop_length * 2
            && (sample_index < hop_length || sample_index >= sample_count - hop_length)
        {
            samples.push(0.0);
            continue;
        }
        let mut energy = 0.0_f32;
        for channel in 0..channels {
            energy += mel_values[channel * frames + frame];
        }
        energy /= channels as f32;
        let phase = sample_index as f32 * 0.03;
        samples.push((energy.tanh() * phase.sin()).clamp(-1.0, 1.0));
    }
    if controls.remove_silence {
        trim_silence(samples)
    } else {
        samples
    }
}

#[cfg(feature = "candle")]
fn trim_silence(samples: Vec<f32>) -> Vec<f32> {
    let Some(start) = samples.iter().position(|sample| sample.abs() > 1.0e-6) else {
        return vec![0.0];
    };
    let end = samples
        .iter()
        .rposition(|sample| sample.abs() > 1.0e-6)
        .expect("start guarantees one non-silent sample");
    samples[start..=end].to_vec()
}

#[cfg(all(test, feature = "candle"))]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn write_test_vocos_bundle(root: &Path, config: &str, weights: &[u8]) {
        std::fs::create_dir_all(root).expect("vocos dir");
        std::fs::write(root.join("config.yaml"), config).expect("config");
        std::fs::write(root.join("pytorch_model.bin"), weights).expect("weights");
    }

    pub(crate) fn test_config(
        input_channels: usize,
        sample_rate_hz: u32,
        hop_length: usize,
    ) -> String {
        format!(
            "model:\n  class_path: vocos.models.Vocos\n  init_args:\n    input_channels: {input_channels}\nfeature_extractor:\n  init_args:\n    sample_rate: {sample_rate_hz}\n    hop_length: {hop_length}\n"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocos_diagnostic_reports_missing_bundle_as_setup_error() {
        let request = NativeVocosVocoderDiagnosticRequest {
            model_id: "vocos-mel-24khz".to_string(),
            bundle_path: None,
            device: NativeTtsDevicePreference::Cpu,
            mel: None,
            options: SpeechSynthesisOptions::default(),
        };

        let output = run_vocos_vocoder_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.status, SpeechSynthesisStatus::SetupRequired);
        assert!(output.audio_frame.is_none());
        assert_eq!(output.device.selected, "cpu");
        assert!(output
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "vocos_bundle_missing"));
    }

    #[test]
    fn vocos_diagnostic_validates_config_and_weights_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join("config.yaml"),
            "model:\n  class_path: vocos.models.Vocos\n",
        )
        .expect("config");
        let request = NativeVocosVocoderDiagnosticRequest {
            model_id: "vocos-mel-24khz".to_string(),
            bundle_path: Some(temp.path().to_path_buf()),
            device: NativeTtsDevicePreference::Cpu,
            mel: None,
            options: SpeechSynthesisOptions::default(),
        };

        let output = run_vocos_vocoder_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.status, SpeechSynthesisStatus::SetupRequired);
        assert!(output
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "vocos_bundle_invalid"));
    }

    #[test]
    fn vocos_diagnostic_validates_missing_config_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("pytorch_model.bin"), [1, 2, 3, 4]).expect("weights");
        let request = NativeVocosVocoderDiagnosticRequest {
            model_id: "vocos-mel-24khz".to_string(),
            bundle_path: Some(temp.path().to_path_buf()),
            device: NativeTtsDevicePreference::Cpu,
            mel: None,
            options: SpeechSynthesisOptions::default(),
        };

        let output = run_vocos_vocoder_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.status, SpeechSynthesisStatus::SetupRequired);
        assert!(output
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "vocos_bundle_invalid"));
    }

    #[test]
    fn vocos_diagnostic_reports_native_inference_controls() {
        let request = NativeVocosVocoderDiagnosticRequest {
            model_id: "vocos-mel-24khz".to_string(),
            bundle_path: None,
            device: NativeTtsDevicePreference::Cpu,
            mel: None,
            options: SpeechSynthesisOptions {
                seed: Some(17),
                steps: Some(4),
                cfg_strength: Some(1.25),
                speed: Some(1.5),
                max_duration_seconds: Some(0.2),
                remove_silence: true,
                ..SpeechSynthesisOptions::default()
            },
        };

        let output = run_vocos_vocoder_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.controls.seed, Some(17));
        assert_eq!(output.controls.steps, 4);
        assert_eq!(output.controls.cfg_strength, 1.25);
        assert_eq!(output.controls.speed, 1.5);
        assert_eq!(output.controls.max_duration_seconds, 0.2);
        assert!(output.controls.remove_silence);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn vocos_diagnostic_converts_generated_mel_to_audio_frame() {
        let temp = tempfile::tempdir().expect("tempdir");
        test_support::write_test_vocos_bundle(
            temp.path(),
            &test_support::test_config(4, 22_050, 128),
            &[1, 2, 3, 4],
        );
        let request = NativeVocosVocoderDiagnosticRequest {
            model_id: "vocos-mel-24khz".to_string(),
            bundle_path: Some(temp.path().to_path_buf()),
            device: NativeTtsDevicePreference::Cpu,
            mel: None,
            options: SpeechSynthesisOptions {
                max_duration_seconds: Some(0.02),
                ..SpeechSynthesisOptions::default()
            },
        };

        let output = run_vocos_vocoder_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.status, SpeechSynthesisStatus::Ready);
        assert!(output.diagnostics.is_empty());
        assert!(output.mel.generated);
        assert_eq!(output.mel.channels, 4);
        let frame = output.audio_frame.as_ref().expect("audio frame");
        assert_eq!(frame.sample_rate, 22_050);
        assert_eq!(frame.channels, 1);
        assert_eq!(frame.samples_per_channel(), 512);
        assert_eq!(output.frame.as_ref().expect("frame report").channels, 1);
        assert_eq!(
            output.audio.as_ref().expect("pcm audio").sample_rate_hz,
            22_050
        );
    }

    #[cfg(feature = "candle")]
    #[test]
    fn vocos_diagnostic_applies_native_inference_controls_to_generated_audio() {
        let temp = tempfile::tempdir().expect("tempdir");
        test_support::write_test_vocos_bundle(
            temp.path(),
            &test_support::test_config(4, 24_000, 64),
            &[1, 2, 3, 4],
        );
        let request = |options| NativeVocosVocoderDiagnosticRequest {
            model_id: "vocos-mel-24khz".to_string(),
            bundle_path: Some(temp.path().to_path_buf()),
            device: NativeTtsDevicePreference::Cpu,
            mel: None,
            options,
        };

        let slow = run_vocos_vocoder_diagnostic(&request(SpeechSynthesisOptions {
            seed: Some(1),
            steps: Some(1),
            cfg_strength: Some(0.5),
            speed: Some(1.0),
            max_duration_seconds: Some(0.04),
            remove_silence: false,
            ..SpeechSynthesisOptions::default()
        }))
        .expect("slow diagnostic");
        let fast = run_vocos_vocoder_diagnostic(&request(SpeechSynthesisOptions {
            seed: Some(1),
            steps: Some(1),
            cfg_strength: Some(0.5),
            speed: Some(2.0),
            max_duration_seconds: Some(0.04),
            remove_silence: false,
            ..SpeechSynthesisOptions::default()
        }))
        .expect("fast diagnostic");
        assert!(fast.mel.frames < slow.mel.frames);

        let reseeded = run_vocos_vocoder_diagnostic(&request(SpeechSynthesisOptions {
            seed: Some(99),
            steps: Some(3),
            cfg_strength: Some(1.5),
            speed: Some(1.0),
            max_duration_seconds: Some(0.04),
            remove_silence: false,
            ..SpeechSynthesisOptions::default()
        }))
        .expect("reseeded diagnostic");
        assert_ne!(
            slow.audio.as_ref().expect("slow audio").samples,
            reseeded.audio.as_ref().expect("reseeded audio").samples
        );

        let trimmed = run_vocos_vocoder_diagnostic(&request(SpeechSynthesisOptions {
            seed: Some(1),
            steps: Some(1),
            cfg_strength: Some(0.5),
            speed: Some(1.0),
            max_duration_seconds: Some(0.04),
            remove_silence: true,
            ..SpeechSynthesisOptions::default()
        }))
        .expect("trimmed diagnostic");
        assert!(
            trimmed.frame.as_ref().expect("trimmed frame").sample_count
                < slow.frame.as_ref().expect("slow frame").sample_count
        );
    }

    #[cfg(all(feature = "candle", feature = "external-tests"))]
    #[test]
    #[ignore = "requires VOCOS_BUNDLE pointing at a local compatible Vocos bundle"]
    fn vocos_native_smoke_when_requested() {
        let bundle = std::env::var_os("VOCOS_BUNDLE")
            .map(PathBuf::from)
            .expect("set VOCOS_BUNDLE to a local compatible Vocos bundle");
        let request = NativeVocosVocoderDiagnosticRequest {
            model_id: std::env::var("VOCOS_MODEL_ID")
                .unwrap_or_else(|_| "vocos-mel-24khz".to_string()),
            bundle_path: Some(bundle),
            device: NativeTtsDevicePreference::Cpu,
            mel: None,
            options: SpeechSynthesisOptions {
                max_duration_seconds: Some(0.02),
                ..SpeechSynthesisOptions::default()
            },
        };

        let output = run_vocos_vocoder_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.status, SpeechSynthesisStatus::Ready);
        assert!(output.audio_generated);
        assert!(output.audio_frame.is_some());
        assert_eq!(output.frame.as_ref().expect("frame").channels, 1);
    }
}
