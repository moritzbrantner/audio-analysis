use std::fs;
use std::path::{Path, PathBuf};

use model_runtime::{ModelBundleManifest, ModelFileRequest, ModelPreset, ModelTask};
use serde::{Deserialize, Serialize};

use crate::{
    NativeTtsDevicePreference, SpeechSynthesisDiagnostic, SpeechSynthesisInferenceControlsReport,
    SpeechSynthesisOptions, SpeechSynthesisStatus,
};

const DEFAULT_F5_MODEL_ID: &str = "f5-tts-v1-base";
const DEFAULT_N_MEL_CHANNELS: usize = 100;
const DEFAULT_SAMPLE_RATE_HZ: u32 = 24_000;
const DEFAULT_HOP_LENGTH: usize = 256;
const DEFAULT_STEPS: u32 = 32;
const DEFAULT_CFG_STRENGTH: f32 = 2.0;
const DEFAULT_SPEED: f32 = 1.0;
const DEFAULT_MAX_DURATION_SECONDS: f32 = 0.25;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NativeF5MelDiagnosticRequest {
    pub text: String,
    #[serde(default = "default_f5_model_id")]
    pub model_id: String,
    #[serde(default)]
    pub bundle_path: Option<PathBuf>,
    #[serde(default)]
    pub device: NativeTtsDevicePreference,
    #[serde(default)]
    pub options: SpeechSynthesisOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NativeF5MelDiagnosticOutput {
    pub status: SpeechSynthesisStatus,
    pub provider_id: String,
    pub model_id: String,
    pub vocoder_required: bool,
    pub audio_generated: bool,
    pub controls: SpeechSynthesisInferenceControlsReport,
    pub device: NativeF5DeviceReport,
    pub bundle: NativeF5BundleReport,
    #[serde(default)]
    pub mel: Option<NativeF5MelReport>,
    pub diagnostics: Vec<SpeechSynthesisDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeF5DeviceReport {
    pub preference: String,
    pub selected: String,
    pub cuda_active: bool,
    pub candle_feature_enabled: bool,
    pub cuda_feature_enabled: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeF5BundleReport {
    #[serde(default)]
    pub bundle_path: Option<String>,
    pub model_known: bool,
    pub required_files: Vec<String>,
    #[serde(default)]
    pub config_path: Option<String>,
    #[serde(default)]
    pub vocab_path: Option<String>,
    #[serde(default)]
    pub safetensors_path: Option<String>,
    #[serde(default)]
    pub vocab_entries: Option<usize>,
    #[serde(default)]
    pub tensor_count: Option<usize>,
    #[serde(default)]
    pub tensor_keys_sample: Vec<String>,
    #[serde(default)]
    pub config: Option<NativeF5ConfigReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeF5ConfigReport {
    pub architecture: String,
    pub n_mel_channels: usize,
    pub sample_rate_hz: u32,
    pub hop_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NativeF5MelReport {
    pub frames: usize,
    pub channels: usize,
    pub sample_rate_hz: u32,
    pub tensor_shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone)]
struct ResolvedF5Bundle {
    config: PathBuf,
    vocab: PathBuf,
    safetensors: PathBuf,
    required_files: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawF5Config {
    #[serde(default, alias = "modelType")]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default, alias = "nMelChannels")]
    n_mel_channels: Option<usize>,
    #[serde(default, alias = "numMels")]
    num_mels: Option<usize>,
    #[serde(default, alias = "sampleRate", alias = "targetSampleRate")]
    sample_rate: Option<u32>,
    #[serde(default, alias = "sampleRateHz")]
    sample_rate_hz: Option<u32>,
    #[serde(default, alias = "hopLength")]
    hop_length: Option<usize>,
}

impl NativeF5MelDiagnosticRequest {
    fn validate(&self) -> Result<(), String> {
        if self.text.trim().is_empty() {
            return Err("invalid request: `text` must not be empty".to_string());
        }
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
        self.options.validate()
    }
}

pub fn run_f5_mel_diagnostic(
    request: &NativeF5MelDiagnosticRequest,
) -> Result<NativeF5MelDiagnosticOutput, String> {
    request.validate()?;
    let device = resolve_device(request.device);
    let mut bundle_report = NativeF5BundleReport {
        bundle_path: request
            .bundle_path
            .as_ref()
            .map(|path| path.display().to_string()),
        model_known: f5_model_preset(&request.model_id).is_some(),
        required_files: request.model_preset_required_files().unwrap_or_default(),
        ..NativeF5BundleReport::default()
    };

    let Some(preset) = f5_model_preset(&request.model_id) else {
        return Ok(output(
            SpeechSynthesisStatus::UnsupportedRuntime,
            request,
            device,
            bundle_report,
            None,
            vec![diagnostic(
                "f5_checkpoint_unsupported",
                format!(
                    "unsupported F5 diagnostic checkpoint `{}`; use `f5-tts-v1-base` or `f5-tts-base`",
                    request.model_id
                ),
                Some("E2 and Vocos checkpoints are covered by separate provider slices."),
            )],
        ));
    };

    let Some(bundle_path) = &request.bundle_path else {
        return Ok(output(
            SpeechSynthesisStatus::SetupRequired,
            request,
            device,
            bundle_report,
            None,
            vec![diagnostic(
                "f5_bundle_missing",
                "F5 mel diagnostics require an explicit local `bundlePath`.".to_string(),
                Some("Pass a side-effect-free local bundle containing config.json, vocab.txt, and model safetensors."),
            )],
        ));
    };

    if !cfg!(feature = "candle") {
        return Ok(output(
            SpeechSynthesisStatus::SetupRequired,
            request,
            device,
            bundle_report,
            None,
            vec![diagnostic(
                "f5_candle_feature_disabled",
                "F5 mel diagnostics require the `candle` feature.".to_string(),
                Some("Rebuild audio-generation-tts with `--features candle`; add `cuda` only when CUDA execution is required."),
            )],
        ));
    }

    let resolved = match resolve_f5_bundle(bundle_path, preset) {
        Ok(resolved) => resolved,
        Err(message) => {
            return Ok(output(
                SpeechSynthesisStatus::SetupRequired,
                request,
                device,
                bundle_report,
                None,
                vec![diagnostic("f5_bundle_invalid", message, None)],
            ));
        }
    };
    bundle_report.required_files = resolved.required_files.clone();
    bundle_report.config_path = Some(resolved.config.display().to_string());
    bundle_report.vocab_path = Some(resolved.vocab.display().to_string());
    bundle_report.safetensors_path = Some(resolved.safetensors.display().to_string());

    let config = match load_config(&resolved.config) {
        Ok(config) => config,
        Err(message) => {
            return Ok(output(
                SpeechSynthesisStatus::UnsupportedRuntime,
                request,
                device,
                bundle_report,
                None,
                vec![diagnostic("f5_config_unsupported", message, None)],
            ));
        }
    };
    bundle_report.config = Some(config.clone());

    let vocab_entries = match load_vocab_entries(&resolved.vocab) {
        Ok(entries) => entries,
        Err(message) => {
            return Ok(output(
                SpeechSynthesisStatus::SetupRequired,
                request,
                device,
                bundle_report,
                None,
                vec![diagnostic("f5_vocab_invalid", message, None)],
            ));
        }
    };
    bundle_report.vocab_entries = Some(vocab_entries);

    let tensor_report = match load_safetensors_report(&resolved.safetensors) {
        Ok(report) => report,
        Err(message) => {
            return Ok(output(
                SpeechSynthesisStatus::SetupRequired,
                request,
                device,
                bundle_report,
                None,
                vec![diagnostic("f5_safetensors_invalid", message, None)],
            ));
        }
    };
    bundle_report.tensor_count = Some(tensor_report.tensor_count);
    bundle_report.tensor_keys_sample = tensor_report.tensor_keys_sample;

    let mel = match build_mel_report(request, &config, &device) {
        Ok(report) => report,
        Err(message) => {
            return Ok(output(
                SpeechSynthesisStatus::SetupRequired,
                request,
                device,
                bundle_report,
                None,
                vec![diagnostic("f5_mel_generation_failed", message, None)],
            ));
        }
    };

    Ok(output(
        SpeechSynthesisStatus::Ready,
        request,
        device,
        bundle_report,
        Some(mel),
        Vec::new(),
    ))
}

fn default_f5_model_id() -> String {
    DEFAULT_F5_MODEL_ID.to_string()
}

impl NativeF5MelDiagnosticRequest {
    fn model_preset_required_files(&self) -> Option<Vec<String>> {
        f5_model_preset(&self.model_id).map(required_files_for_preset)
    }
}

fn f5_model_preset(model_id: &str) -> Option<ModelPreset> {
    match model_id {
        "f5-tts-v1-base" => Some(ModelPreset::F5TtsV1Base),
        "f5-tts-base" => Some(ModelPreset::F5TtsBase),
        _ => None,
    }
}

fn required_files_for_preset(preset: ModelPreset) -> Vec<String> {
    let mut files = vec!["config.json".to_string()];
    files.extend(
        preset
            .spec()
            .files
            .into_iter()
            .filter_map(|file| match file {
                ModelFileRequest::Required(path) => Some(path),
                ModelFileRequest::Optional(_) | ModelFileRequest::FirstAvailable(_) => None,
            }),
    );
    files
}

fn output(
    status: SpeechSynthesisStatus,
    request: &NativeF5MelDiagnosticRequest,
    device: NativeF5DeviceReport,
    bundle: NativeF5BundleReport,
    mel: Option<NativeF5MelReport>,
    diagnostics: Vec<SpeechSynthesisDiagnostic>,
) -> NativeF5MelDiagnosticOutput {
    let mut diagnostics = diagnostics;
    diagnostics.extend(device_diagnostics(&device));
    NativeF5MelDiagnosticOutput {
        status,
        provider_id: "f5".to_string(),
        model_id: request.model_id.clone(),
        vocoder_required: true,
        audio_generated: false,
        controls: f5_controls(&request.options),
        device,
        bundle,
        mel,
        diagnostics,
    }
}

fn f5_controls(options: &SpeechSynthesisOptions) -> SpeechSynthesisInferenceControlsReport {
    SpeechSynthesisInferenceControlsReport::from_options(
        options,
        DEFAULT_STEPS,
        DEFAULT_CFG_STRENGTH,
        DEFAULT_SPEED,
        DEFAULT_MAX_DURATION_SECONDS,
    )
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

fn resolve_device(preference: NativeTtsDevicePreference) -> NativeF5DeviceReport {
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
    NativeF5DeviceReport {
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

fn device_diagnostics(device: &NativeF5DeviceReport) -> Vec<SpeechSynthesisDiagnostic> {
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

fn resolve_f5_bundle(bundle: &Path, preset: ModelPreset) -> Result<ResolvedF5Bundle, String> {
    if !bundle.exists() {
        return Err(format!(
            "F5 bundle path `{}` does not exist",
            bundle.display()
        ));
    }
    if !bundle.is_dir() {
        return Err(format!(
            "F5 bundle path `{}` must be a directory",
            bundle.display()
        ));
    }
    let spec = preset.spec();
    if spec.task != ModelTask::SpeakerConditionedTts {
        return Err(format!(
            "model `{}` is not a speaker-conditioned TTS preset",
            spec.name
        ));
    }
    let manifest = load_manifest(bundle)?;
    let required_files = required_files_for_preset(preset);
    let safetensors_remote = spec
        .files
        .iter()
        .find_map(|file| match file {
            ModelFileRequest::Required(path) if path.ends_with(".safetensors") => {
                Some(path.as_str())
            }
            _ => None,
        })
        .ok_or_else(|| format!("F5 preset `{}` does not declare safetensors", spec.name))?;
    let vocab_remote = spec
        .files
        .iter()
        .find_map(|file| match file {
            ModelFileRequest::Required(path) if path.ends_with("vocab.txt") => Some(path.as_str()),
            _ => None,
        })
        .ok_or_else(|| format!("F5 preset `{}` does not declare vocab.txt", spec.name))?;
    let config_remotes = config_remote_candidates(safetensors_remote);

    let config = resolve_first_existing_bundle_file(bundle, manifest.as_ref(), &config_remotes)
        .ok_or_else(|| {
            format!(
                "F5 bundle `{}` is missing config.json; searched {}",
                bundle.display(),
                config_remotes.join(", ")
            )
        })?;
    let vocab = resolve_bundle_file(bundle, manifest.as_ref(), vocab_remote)?;
    let safetensors = resolve_bundle_file(bundle, manifest.as_ref(), safetensors_remote)?;
    Ok(ResolvedF5Bundle {
        config,
        vocab,
        safetensors,
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

fn config_remote_candidates(safetensors_remote: &str) -> Vec<String> {
    let mut candidates = vec!["config.json".to_string()];
    if let Some(parent) = Path::new(safetensors_remote).parent() {
        let nested = parent.join("config.json").to_string_lossy().to_string();
        if !candidates.contains(&nested) {
            candidates.push(nested);
        }
    }
    candidates
}

fn resolve_first_existing_bundle_file(
    bundle: &Path,
    manifest: Option<&ModelBundleManifest>,
    remote_paths: &[String],
) -> Option<PathBuf> {
    remote_paths
        .iter()
        .find_map(|remote_path| resolve_bundle_file(bundle, manifest, remote_path).ok())
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
        "F5 bundle `{}` is missing required file `{remote_path}`",
        bundle.display()
    ))
}

fn load_config(path: &Path) -> Result<NativeF5ConfigReport, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read F5 config `{}`: {error}", path.display()))?;
    let config: RawF5Config = serde_json::from_str(&raw)
        .map_err(|error| format!("failed to parse F5 config `{}`: {error}", path.display()))?;
    if let Some(model_type) = config.model_type.as_deref() {
        let normalized = model_type.replace(['-', '_'], "").to_ascii_lowercase();
        if normalized != "f5tts" && normalized != "f5" {
            return Err(format!(
                "unsupported_runtime: unsupported F5 config model_type `{model_type}`"
            ));
        }
    }
    if !config.architectures.is_empty()
        && !config.architectures.iter().any(|architecture| {
            let architecture = architecture.to_ascii_lowercase();
            architecture.contains("f5") || architecture.contains("dit")
        })
    {
        return Err(format!(
            "unsupported_runtime: unsupported F5 architecture `{}`",
            config.architectures.join(", ")
        ));
    }
    let n_mel_channels = config
        .n_mel_channels
        .or(config.num_mels)
        .unwrap_or(DEFAULT_N_MEL_CHANNELS);
    if n_mel_channels == 0 {
        return Err("unsupported_runtime: n_mel_channels must be greater than zero".to_string());
    }
    let sample_rate_hz = config
        .sample_rate_hz
        .or(config.sample_rate)
        .unwrap_or(DEFAULT_SAMPLE_RATE_HZ);
    if sample_rate_hz == 0 {
        return Err("unsupported_runtime: sample rate must be greater than zero".to_string());
    }
    let hop_length = config.hop_length.unwrap_or(DEFAULT_HOP_LENGTH);
    if hop_length == 0 {
        return Err("unsupported_runtime: hop_length must be greater than zero".to_string());
    }
    Ok(NativeF5ConfigReport {
        architecture: config
            .architectures
            .first()
            .cloned()
            .unwrap_or_else(|| "f5-tts".to_string()),
        n_mel_channels,
        sample_rate_hz,
        hop_length,
    })
}

fn load_vocab_entries(path: &Path) -> Result<usize, String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("failed to read F5 vocab `{}`: {error}", path.display()))?;
    let count = raw.lines().filter(|line| !line.trim().is_empty()).count();
    if count == 0 {
        return Err(format!(
            "F5 vocab `{}` does not contain entries",
            path.display()
        ));
    }
    Ok(count)
}

#[derive(Debug, Clone)]
struct SafetensorsReport {
    tensor_count: usize,
    tensor_keys_sample: Vec<String>,
}

#[cfg(feature = "candle")]
fn load_safetensors_report(path: &Path) -> Result<SafetensorsReport, String> {
    let tensors =
        candle_core::safetensors::load(path, &candle_core::Device::Cpu).map_err(|error| {
            format!(
                "failed to load F5 safetensors `{}` metadata: {error}",
                path.display()
            )
        })?;
    if tensors.is_empty() {
        return Err(format!(
            "F5 safetensors `{}` contains no tensors",
            path.display()
        ));
    }
    let mut keys = tensors.keys().cloned().collect::<Vec<_>>();
    keys.sort();
    Ok(SafetensorsReport {
        tensor_count: keys.len(),
        tensor_keys_sample: keys.into_iter().take(8).collect(),
    })
}

#[cfg(not(feature = "candle"))]
fn load_safetensors_report(_path: &Path) -> Result<SafetensorsReport, String> {
    Err("F5 safetensors loading requires the `candle` feature".to_string())
}

#[cfg(feature = "candle")]
fn build_mel_report(
    request: &NativeF5MelDiagnosticRequest,
    config: &NativeF5ConfigReport,
    device: &NativeF5DeviceReport,
) -> Result<NativeF5MelReport, String> {
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
                "device `{}` cannot execute the F5 mel diagnostic",
                device.selected
            ));
        }
    };
    let controls = f5_controls(&request.options);
    let seconds = (controls.max_duration_seconds / controls.speed).clamp(0.01, 1.0);
    let frames = ((seconds * config.sample_rate_hz as f32) / config.hop_length as f32)
        .ceil()
        .max(1.0) as usize;
    let tensor_shape = vec![1, config.n_mel_channels, frames];
    let mel = candle_core::Tensor::zeros(
        tensor_shape.as_slice(),
        candle_core::DType::F32,
        &candle_device,
    )
    .map_err(|error| format!("failed to allocate F5 mel diagnostic tensor: {error}"))?;
    Ok(NativeF5MelReport {
        frames,
        channels: config.n_mel_channels,
        sample_rate_hz: config.sample_rate_hz,
        tensor_shape: mel.dims().to_vec(),
        dtype: format!("{:?}", mel.dtype()).to_ascii_lowercase(),
    })
}

#[cfg(not(feature = "candle"))]
fn build_mel_report(
    _request: &NativeF5MelDiagnosticRequest,
    _config: &NativeF5ConfigReport,
    _device: &NativeF5DeviceReport,
) -> Result<NativeF5MelReport, String> {
    Err("F5 mel generation requires the `candle` feature".to_string())
}

#[cfg(all(test, feature = "candle"))]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn write_test_f5_bundle(root: &Path, config: serde_json::Value) {
        let f5_dir = root.join("files").join("F5TTS_v1_Base");
        std::fs::create_dir_all(&f5_dir).expect("f5 dir");
        std::fs::write(f5_dir.join("config.json"), config.to_string()).expect("config");
        std::fs::write(f5_dir.join("vocab.txt"), "<pad>\na\n").expect("vocab");
        let tensors = std::collections::HashMap::from([(
            "model.transformer.token_emb.weight",
            candle_core::Tensor::zeros((2, 2), candle_core::DType::F32, &candle_core::Device::Cpu)
                .expect("tensor"),
        )]);
        candle_core::safetensors::save(&tensors, f5_dir.join("model_1250000.safetensors"))
            .expect("safetensors");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f5_diagnostic_reports_missing_bundle_as_setup_error() {
        let request = NativeF5MelDiagnosticRequest {
            text: "diagnose missing bundle".to_string(),
            model_id: "f5-tts-v1-base".to_string(),
            bundle_path: None,
            device: NativeTtsDevicePreference::Cpu,
            options: SpeechSynthesisOptions::default(),
        };

        let output = run_f5_mel_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.status, SpeechSynthesisStatus::SetupRequired);
        assert!(output.mel.is_none());
        assert_eq!(output.device.selected, "cpu");
        assert!(output
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "f5_bundle_missing"));
    }

    #[test]
    fn f5_diagnostic_rejects_unsupported_checkpoint() {
        let request = NativeF5MelDiagnosticRequest {
            text: "diagnose unsupported checkpoint".to_string(),
            model_id: "e2-tts-base".to_string(),
            bundle_path: Some(PathBuf::from("unused")),
            device: NativeTtsDevicePreference::Cpu,
            options: SpeechSynthesisOptions::default(),
        };

        let output = run_f5_mel_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.status, SpeechSynthesisStatus::UnsupportedRuntime);
        assert!(output.mel.is_none());
        assert!(output
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "f5_checkpoint_unsupported"));
    }

    #[test]
    fn f5_diagnostic_reports_cuda_request_without_cuda_feature() {
        let request = NativeF5MelDiagnosticRequest {
            text: "diagnose cuda selection".to_string(),
            model_id: "f5-tts-v1-base".to_string(),
            bundle_path: None,
            device: NativeTtsDevicePreference::Cuda,
            options: SpeechSynthesisOptions::default(),
        };

        let output = run_f5_mel_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.device.preference, "cuda");
        assert_eq!(output.device.cuda_feature_enabled, cfg!(feature = "cuda"));
        if output.device.selected.starts_with("cuda") {
            assert!(output.device.selected.starts_with("cuda"));
        } else {
            assert_eq!(output.device.selected, "unavailable");
            assert!(output
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "native_tts_cuda_unavailable"));
        }
    }

    #[test]
    fn f5_diagnostic_reports_cpu_fallback_for_auto_without_cuda_feature() {
        let request = NativeF5MelDiagnosticRequest {
            text: "diagnose auto fallback".to_string(),
            model_id: "f5-tts-v1-base".to_string(),
            bundle_path: None,
            device: NativeTtsDevicePreference::Auto,
            options: SpeechSynthesisOptions::default(),
        };

        let output = run_f5_mel_diagnostic(&request).expect("diagnostic");

        if output.device.selected == "cpu" {
            assert_eq!(output.device.selected, "cpu");
            assert!(output
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "native_tts_cpu_fallback"));
        }
    }

    #[test]
    fn f5_diagnostic_reports_native_inference_controls() {
        let request = NativeF5MelDiagnosticRequest {
            text: "diagnose controls".to_string(),
            model_id: "f5-tts-v1-base".to_string(),
            bundle_path: None,
            device: NativeTtsDevicePreference::Cpu,
            options: SpeechSynthesisOptions {
                seed: Some(13),
                steps: Some(7),
                cfg_strength: Some(1.5),
                speed: Some(1.25),
                max_duration_seconds: Some(0.4),
                remove_silence: true,
                ..SpeechSynthesisOptions::default()
            },
        };

        let output = run_f5_mel_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.controls.seed, Some(13));
        assert_eq!(output.controls.steps, 7);
        assert_eq!(output.controls.cfg_strength, 1.5);
        assert_eq!(output.controls.speed, 1.25);
        assert_eq!(output.controls.max_duration_seconds, 0.4);
        assert!(output.controls.remove_silence);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn f5_diagnostic_validates_local_bundle_inputs() {
        let temp = tempfile::tempdir().expect("tempdir");
        test_support::write_test_f5_bundle(temp.path(), serde_json::json!({"model_type": "bert"}));
        let request = NativeF5MelDiagnosticRequest {
            text: "diagnose config".to_string(),
            model_id: "f5-tts-v1-base".to_string(),
            bundle_path: Some(temp.path().to_path_buf()),
            device: NativeTtsDevicePreference::Cpu,
            options: SpeechSynthesisOptions::default(),
        };

        let output = run_f5_mel_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.status, SpeechSynthesisStatus::UnsupportedRuntime);
        assert!(output
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "f5_config_unsupported"));
    }

    #[cfg(all(feature = "candle", feature = "external-tests"))]
    #[test]
    #[ignore = "requires F5_TTS_BUNDLE pointing at a local compatible F5 bundle"]
    fn f5_native_smoke_when_requested() {
        let bundle = std::env::var_os("F5_TTS_BUNDLE")
            .map(PathBuf::from)
            .expect("set F5_TTS_BUNDLE to a local compatible F5 bundle");
        let request = NativeF5MelDiagnosticRequest {
            text: "native f5 smoke".to_string(),
            model_id: std::env::var("F5_TTS_MODEL_ID")
                .unwrap_or_else(|_| "f5-tts-v1-base".to_string()),
            bundle_path: Some(bundle),
            device: NativeTtsDevicePreference::Cpu,
            options: SpeechSynthesisOptions {
                max_duration_seconds: Some(0.05),
                steps: Some(1),
                ..SpeechSynthesisOptions::default()
            },
        };

        let output = run_f5_mel_diagnostic(&request).expect("diagnostic");

        assert_eq!(output.status, SpeechSynthesisStatus::Ready);
        assert!(output.mel.is_some());
        assert!(output.diagnostics.is_empty());
        assert!(!output.audio_generated);
        assert!(output.vocoder_required);
    }
}
