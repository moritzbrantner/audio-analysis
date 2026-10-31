//! Model and bundle resolution, Q8 bundle validation and setup diagnostics.

use crate::{setup_error, CandleWhisperComputeType, CandleWhisperOptions};
use candle_core::quantized::{gguf_file, GgmlDType};
use candle_transformers::models::whisper;
use media_core::Result;
use std::fs::File;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhisperBundlePaths {
    pub root: PathBuf,
    pub config_json: PathBuf,
    pub generation_config_json: PathBuf,
    pub tokenizer_json: PathBuf,
    pub preprocessor_config_json: PathBuf,
    pub model_safetensors: PathBuf,
    pub model_q8_0_gguf: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WhisperModelFormat {
    Safetensors,
    GgufQ8_0,
}

impl WhisperModelFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::GgufQ8_0 => "gguf-q8_0",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedWhisperModel {
    pub model_id: String,
    pub bundle: WhisperBundlePaths,
    pub(super) source: &'static str,
}

#[allow(dead_code)]
pub(super) fn resolve_whisper_model(
    options: &CandleWhisperOptions,
    requested_model_id: &str,
) -> Result<ResolvedWhisperModel> {
    let resolved_compute_type = options.compute_type.resolve_for_device(false)?;
    resolve_whisper_model_with_observer(
        options,
        requested_model_id,
        resolved_compute_type,
        &mut |_| Ok(()),
    )
}

pub(super) fn resolve_whisper_model_with_observer(
    options: &CandleWhisperOptions,
    requested_model_id: &str,
    resolved_compute_type: CandleWhisperComputeType,
    observe: &mut dyn FnMut(WhisperModelResolutionEvent) -> Result<()>,
) -> Result<ResolvedWhisperModel> {
    resolve_whisper_resources(
        options,
        requested_model_id,
        resolved_compute_type,
        ResourceInspection::ValidateQ8,
        observe,
    )
}

#[derive(Clone, Copy)]
pub(crate) enum ResourceInspection {
    FilesOnly,
    ValidateQ8,
}

pub(crate) fn resolve_whisper_resources(
    options: &CandleWhisperOptions,
    requested_model_id: &str,
    resolved_compute_type: CandleWhisperComputeType,
    inspection: ResourceInspection,
    observe: &mut dyn FnMut(WhisperModelResolutionEvent) -> Result<()>,
) -> Result<ResolvedWhisperModel> {
    observe(WhisperModelResolutionEvent::ResolutionStart)?;
    let model_id = canonical_whisper_model_id(requested_model_id)?;
    if let Some(bundle) = &options.model_bundle {
        let bundle = if resolved_compute_type == CandleWhisperComputeType::Int8 {
            let paths = resolve_q8_whisper_bundle_file_paths(bundle)?;
            if matches!(inspection, ResourceInspection::ValidateQ8) {
                validate_q8_whisper_bundle(&paths)?;
            }
            paths
        } else {
            resolve_whisper_bundle_paths(bundle)?
        };
        observe(WhisperModelResolutionEvent::ResolutionEnd {
            source: "explicit-bundle",
        })?;
        return Ok(ResolvedWhisperModel {
            model_id,
            bundle,
            source: "explicit-bundle",
        });
    }
    if resolved_compute_type == CandleWhisperComputeType::Int8 {
        return Err(setup_error(format!(
            "native Candle Whisper compute type int8 requires an explicit local Q8_0 bundle containing {}; automatic downloads and safetensors fallback are disabled",
            resolved_compute_type.required_bundle_files().join(", ")
        )));
    }

    #[cfg(feature = "model-bundles")]
    {
        if options.model_cache_only {
            let bundle = resolve_cached_whisper_model(&model_id, options.model_dir.as_deref())
                .ok_or_else(|| missing_whisper_model_error(&model_id, options))?;
            observe(WhisperModelResolutionEvent::ResolutionEnd {
                source: "hugging-face-cache",
            })?;
            return Ok(ResolvedWhisperModel {
                model_id,
                bundle,
                source: "hugging-face-cache",
            });
        }

        let mut downloader = model_runtime::HuggingFaceDownloader::new().progress(false);
        if let Some(model_dir) = &options.model_dir {
            downloader = downloader.cache_dir(model_dir.clone());
        }
        observe(WhisperModelResolutionEvent::DownloadStart)?;
        let download_started = std::time::Instant::now();
        let downloaded = downloader
            .download(&whisper_model_spec(&model_id))
            .map_err(|error| missing_whisper_model_error_with_source(&model_id, options, error))?;
        observe(WhisperModelResolutionEvent::DownloadEnd {
            duration_seconds: download_started.elapsed().as_secs_f64(),
        })?;
        let bundle = downloaded
            .model_dir()
            .ok_or_else(|| {
                setup_error(format!(
                    "native Candle Whisper model `{model_id}` resolved without a local model directory"
                ))
            })
            .and_then(resolve_whisper_bundle_paths)?;
        observe(WhisperModelResolutionEvent::ResolutionEnd {
            source: "hugging-face-cache",
        })?;
        Ok(ResolvedWhisperModel {
            model_id,
            bundle,
            source: "hugging-face-cache",
        })
    }

    #[cfg(not(feature = "model-bundles"))]
    {
        Err(setup_error(format!(
            "native Candle Whisper model `{model_id}` requires --whisper-bundle or the model-bundles feature for Hugging Face resolution"
        )))
    }
}

pub(super) fn canonical_whisper_model_id(value: &str) -> Result<String> {
    match value {
        "tiny" => Ok("openai/whisper-tiny".to_string()),
        "tiny.en" => Ok("openai/whisper-tiny.en".to_string()),
        "base" => Ok("openai/whisper-base".to_string()),
        "base.en" => Ok("openai/whisper-base.en".to_string()),
        "small" => Ok("openai/whisper-small".to_string()),
        "small.en" => Ok("openai/whisper-small.en".to_string()),
        "medium" => Ok("openai/whisper-medium".to_string()),
        "medium.en" => Ok("openai/whisper-medium.en".to_string()),
        "large" => Ok("openai/whisper-large-v3".to_string()),
        "large-v1" => Ok("openai/whisper-large-v1".to_string()),
        "large-v2" => Ok("openai/whisper-large-v2".to_string()),
        "large-v3" => Ok("openai/whisper-large-v3".to_string()),
        "large-v3-turbo" => Ok("openai/whisper-large-v3-turbo".to_string()),
        other if looks_like_hf_repo_id(other) => Ok(other.to_string()),
        other => Err(setup_error(format!(
            "unsupported native Candle Whisper model alias `{other}`; native Candle Whisper requires a supported Whisper alias, a Hugging Face repo ID with Candle-compatible files, or --whisper-bundle"
        ))),
    }
}

fn looks_like_hf_repo_id(value: &str) -> bool {
    let mut parts = value.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(owner), Some(repo), None) if !owner.is_empty() && !repo.is_empty())
}

#[cfg(feature = "model-bundles")]
fn resolve_cached_whisper_model(
    model_id: &str,
    model_dir: Option<&Path>,
) -> Option<WhisperBundlePaths> {
    let mut roots = Vec::new();
    if let Some(model_dir) = model_dir {
        roots.push(model_dir.to_path_buf());
    } else if let Some(home) = std::env::var_os("HF_HOME") {
        roots.push(PathBuf::from(home).join("hub"));
    } else if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
    }
    for root in roots {
        for candidate in whisper_cache_candidates(&root, model_id) {
            if let Ok(paths) = resolve_whisper_bundle_paths(&candidate) {
                return Some(paths);
            }
        }
    }
    None
}

#[cfg(feature = "model-bundles")]
fn whisper_cache_candidates(root: &Path, model_id: &str) -> Vec<PathBuf> {
    let mut candidates = vec![root.to_path_buf(), root.join(model_id.replace('/', "--"))];
    let hf_repo_dir = root.join(format!("models--{}", model_id.replace('/', "--")));
    if let Ok(snapshot) = std::fs::read_to_string(hf_repo_dir.join("refs/main")) {
        candidates.push(hf_repo_dir.join("snapshots").join(snapshot.trim()));
    }
    if let Ok(entries) = std::fs::read_dir(hf_repo_dir.join("snapshots")) {
        for entry in entries.flatten() {
            candidates.push(entry.path());
        }
    }
    candidates
}

#[cfg(feature = "model-bundles")]
pub(super) fn whisper_model_spec(model_id: &str) -> model_runtime::HuggingFaceModelSpec {
    let mut spec = model_runtime::HuggingFaceModelSpec::new(
        model_id.to_string(),
        model_runtime::ModelTask::SpeechRecognition,
    );
    spec.files = CandleWhisperComputeType::Automatic
        .required_bundle_files()
        .iter()
        .copied()
        .map(model_runtime::ModelFileRequest::required)
        .collect();
    spec
}

#[cfg(feature = "model-bundles")]
fn missing_whisper_model_error(
    model_id: &str,
    options: &CandleWhisperOptions,
) -> media_core::DetectError {
    setup_error(format!(
        "failed to resolve native Candle Whisper model `{model_id}`; required files: {}; --model-dir={}; cache-only={}",
        CandleWhisperComputeType::Automatic
            .required_bundle_files()
            .join(", "),
        options
            .model_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<default huggingface cache>".to_string()),
        options.model_cache_only
    ))
}

#[cfg(feature = "model-bundles")]
fn missing_whisper_model_error_with_source(
    model_id: &str,
    options: &CandleWhisperOptions,
    source: impl std::fmt::Display,
) -> media_core::DetectError {
    setup_error(format!(
        "failed to resolve native Candle Whisper model `{model_id}`; required files: {}; --model-dir={}; cache-only={}: {source}",
        CandleWhisperComputeType::Automatic
            .required_bundle_files()
            .join(", "),
        options
            .model_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<default huggingface cache>".to_string()),
        options.model_cache_only
    ))
}

pub(super) fn whisper_setup_diagnostics(setup: &WhisperRunSetup) -> Vec<String> {
    vec![
        format!("asrModelResolved={}", setup.bundle.root.display()),
        format!("asrModelSource={}", setup.model_source),
        format!("asrModelId={}", setup.model_id),
        format!(
            "requestedComputeType={}",
            setup.requested_compute_type.as_str()
        ),
        format!(
            "resolvedComputeType={}",
            setup.resolved_compute_type.as_str()
        ),
        format!(
            "modelWeightDtype={}",
            candle_dtype_name(setup.model_weight_dtype)
        ),
        format!("computeType={}", setup.resolved_compute_type.as_str()),
        format!("modelFormat={}", setup.model_format.as_str()),
    ]
}

pub(super) fn q8_execution_diagnostics(execution_route: &str, kernel_route: &str) -> [String; 2] {
    [
        format!("q8ExecutionRoute={execution_route}"),
        format!("q8LinearKernel={kernel_route}"),
    ]
}

pub(crate) fn resolve_whisper_bundle_paths(bundle: &Path) -> Result<WhisperBundlePaths> {
    if !bundle.exists() {
        return Err(setup_error(format!(
            "required Candle Whisper model bundle `{}` is missing",
            bundle.display()
        )));
    }
    Ok(WhisperBundlePaths {
        root: bundle.to_path_buf(),
        config_json: crate::native_bundles::resolve_required_bundle_file(bundle, "config.json")?,
        generation_config_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "generation_config.json",
        )?,
        tokenizer_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "tokenizer.json",
        )?,
        preprocessor_config_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "preprocessor_config.json",
        )?,
        model_safetensors: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "model.safetensors",
        )?,
        model_q8_0_gguf: None,
    })
}

#[cfg(test)]
pub(super) fn resolve_q8_whisper_bundle_paths(bundle: &Path) -> Result<WhisperBundlePaths> {
    let paths = resolve_q8_whisper_bundle_file_paths(bundle)?;
    validate_q8_whisper_bundle(&paths)?;
    Ok(paths)
}

fn resolve_q8_whisper_bundle_file_paths(bundle: &Path) -> Result<WhisperBundlePaths> {
    if !bundle.exists() {
        return Err(setup_error(format!(
            "required Candle Whisper Q8_0 model bundle `{}` is missing; required files: {}",
            bundle.display(),
            CandleWhisperComputeType::Int8
                .required_bundle_files()
                .join(", ")
        )));
    }
    crate::native_bundles::validate_required_bundle_files(
        bundle,
        CandleWhisperComputeType::Int8.required_bundle_files(),
    )?;
    let paths = WhisperBundlePaths {
        root: bundle.to_path_buf(),
        config_json: crate::native_bundles::resolve_required_bundle_file(bundle, "config.json")?,
        generation_config_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "generation_config.json",
        )?,
        tokenizer_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "tokenizer.json",
        )?,
        preprocessor_config_json: crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "preprocessor_config.json",
        )?,
        // Int8 deliberately leaves this path unresolved so model.safetensors can
        // never satisfy or mask the Q8 bundle contract.
        model_safetensors: bundle.join("model.safetensors"),
        model_q8_0_gguf: Some(crate::native_bundles::resolve_required_bundle_file(
            bundle,
            "model.q8_0.gguf",
        )?),
    };
    Ok(paths)
}

fn validate_q8_whisper_bundle(paths: &WhisperBundlePaths) -> Result<()> {
    let config: whisper::Config = read_json(&paths.config_json, "config.json")?;
    let generation: GenerationConfig =
        read_json(&paths.generation_config_json, "generation_config.json")?;
    let _: serde_json::Value =
        read_json(&paths.preprocessor_config_json, "preprocessor_config.json")?;
    let tokenizer = Tokenizer::from_file(&paths.tokenizer_json).map_err(|error| {
        setup_error(format!(
            "failed to load Q8 Whisper tokenizer `{}`: {error}",
            paths.tokenizer_json.display()
        ))
    })?;
    validate_whisper_companion_metadata(&config, &generation, &tokenizer)?;

    let gguf_path = paths
        .model_q8_0_gguf
        .as_deref()
        .expect("Q8 bundle always has GGUF weights");
    let mut file = File::open(gguf_path).map_err(|error| {
        setup_error(format!(
            "failed to open Q8 Whisper GGUF `{}`: {error}",
            gguf_path.display()
        ))
    })?;
    let content = gguf_file::Content::read(&mut file).map_err(|error| {
        setup_error(format!(
            "invalid Q8 Whisper GGUF `{}`: {error}",
            gguf_path.display()
        ))
    })?;
    let architecture = content
        .metadata
        .get("general.architecture")
        .and_then(|value| match value {
            gguf_file::Value::String(value) => Some(value.as_str()),
            _ => None,
        })
        .ok_or_else(|| {
            setup_error("Q8 Whisper GGUF is missing string metadata `general.architecture`")
        })?;
    if architecture != "whisper" {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF metadata `general.architecture` must be `whisper`, got `{architecture}`"
        )));
    }
    let file_type = content
        .metadata
        .get("general.file_type")
        .and_then(|value| value.to_u64().ok())
        .ok_or_else(|| {
            setup_error("Q8 Whisper GGUF is missing integer metadata `general.file_type`")
        })?;
    if file_type != 7 {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF metadata `general.file_type` must identify Q8_0 (7), got {file_type}"
        )));
    }
    validate_q8_tensor(
        &content,
        "model.decoder.embed_tokens.weight",
        &[config.vocab_size, config.d_model],
    )?;
    validate_tensor_shape(
        &content,
        "model.encoder.conv1.weight",
        &[config.d_model, config.num_mel_bins, 3],
    )?;
    validate_tensor_shape(
        &content,
        "model.encoder.conv2.weight",
        &[config.d_model, config.d_model, 3],
    )?;
    for (name, info) in &content.tensor_infos {
        if q8_required_tensor_name(name) && info.ggml_dtype != GgmlDType::Q8_0 {
            return Err(setup_error(format!(
                "Q8 Whisper GGUF tensor `{name}` must use Q8_0, got {:?}",
                info.ggml_dtype
            )));
        }
    }
    Ok(())
}

fn validate_whisper_companion_metadata(
    config: &whisper::Config,
    generation: &GenerationConfig,
    tokenizer: &Tokenizer,
) -> Result<()> {
    if config.d_model == 0
        || config.encoder_layers == 0
        || config.decoder_layers == 0
        || config.encoder_attention_heads == 0
        || config.decoder_attention_heads == 0
        || !config
            .d_model
            .is_multiple_of(config.encoder_attention_heads)
        || !config
            .d_model
            .is_multiple_of(config.decoder_attention_heads)
    {
        return Err(setup_error(
            "Q8 Whisper config contains incompatible model or attention dimensions",
        ));
    }
    let tokenizer_vocab = tokenizer.get_vocab_size(true);
    if tokenizer_vocab != config.vocab_size {
        return Err(setup_error(format!(
            "Q8 Whisper tokenizer vocabulary size {tokenizer_vocab} does not match config vocab_size {}",
            config.vocab_size
        )));
    }
    for (name, token) in [
        ("decoder_start_token_id", generation.decoder_start_token_id),
        ("eos_token_id", generation.eos_token_id),
        ("no_timestamps_token_id", generation.no_timestamps_token_id),
    ] {
        if token.is_some_and(|token| token as usize >= config.vocab_size) {
            return Err(setup_error(format!(
                "Q8 Whisper generation metadata `{name}` is outside the tokenizer vocabulary"
            )));
        }
    }
    Ok(())
}

fn validate_q8_tensor(
    content: &gguf_file::Content,
    name: &str,
    expected_dims: &[usize],
) -> Result<()> {
    let info = content.tensor_infos.get(name).ok_or_else(|| {
        setup_error(format!(
            "Q8 Whisper GGUF is missing required tensor `{name}`"
        ))
    })?;
    if info.shape.dims() != expected_dims {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF tensor `{name}` has shape {:?}, expected {expected_dims:?}",
            info.shape.dims()
        )));
    }
    if info.ggml_dtype != GgmlDType::Q8_0 {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF tensor `{name}` must use Q8_0, got {:?}",
            info.ggml_dtype
        )));
    }
    Ok(())
}

fn validate_tensor_shape(
    content: &gguf_file::Content,
    name: &str,
    expected_dims: &[usize],
) -> Result<()> {
    let info = content.tensor_infos.get(name).ok_or_else(|| {
        setup_error(format!(
            "Q8 Whisper GGUF is missing required tensor `{name}`"
        ))
    })?;
    if info.shape.dims() != expected_dims {
        return Err(setup_error(format!(
            "Q8 Whisper GGUF tensor `{name}` has shape {:?}, expected {expected_dims:?}",
            info.shape.dims()
        )));
    }
    Ok(())
}

fn q8_required_tensor_name(name: &str) -> bool {
    name.ends_with(".weight")
        && !name.contains(".conv")
        && !name.ends_with("embed_positions.weight")
        && !name.contains("layer_norm.weight")
}
