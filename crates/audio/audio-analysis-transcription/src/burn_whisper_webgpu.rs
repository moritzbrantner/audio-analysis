use crate::{invalid_request, setup_error, LoadedAudio, TranscriptionTask};
use media_core::Result;
use serde_json::Value;

pub const BURN_WHISPER_WEBGPU_PROVIDER_ID: &str = "burn-whisper-webgpu";
pub const BURN_WHISPER_WEBGPU_SAMPLE_RATE: u32 = 16_000;

const WHISPER_TINY_ID: &str = "openai/whisper-tiny";
const WHISPER_TINY_EN_ID: &str = "openai/whisper-tiny.en";

/// Whisper checkpoints intentionally admitted by the first browser WebGPU provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnWhisperWebGpuModel {
    Tiny,
    TinyEn,
}

impl BurnWhisperWebGpuModel {
    pub fn from_model_id(model_id: &str) -> Result<Self> {
        match model_id.trim() {
            "tiny" | WHISPER_TINY_ID => Ok(Self::Tiny),
            "tiny.en" | WHISPER_TINY_EN_ID => Ok(Self::TinyEn),
            other => Err(invalid_request(format!(
                "Burn WebGPU Whisper currently supports only {WHISPER_TINY_ID} and {WHISPER_TINY_EN_ID}; got {other:?}"
            ))),
        }
    }

    pub const fn model_id(self) -> &'static str {
        match self {
            Self::Tiny => WHISPER_TINY_ID,
            Self::TinyEn => WHISPER_TINY_EN_ID,
        }
    }

    pub const fn english_only(self) -> bool {
        matches!(self, Self::TinyEn)
    }
}

/// Caller-owned Hugging Face Whisper assets used by the browser provider.
///
/// No filesystem or network lookup is performed by this contract. Keeping the
/// bytes caller-owned makes the same provider usable from a browser worker,
/// an application cache, or an explicitly managed model store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnWhisperWebGpuAssets {
    pub config_json: Vec<u8>,
    pub generation_config_json: Vec<u8>,
    pub tokenizer_json: Vec<u8>,
    pub preprocessor_config_json: Vec<u8>,
    pub model_safetensors: Vec<u8>,
}

impl BurnWhisperWebGpuAssets {
    pub fn validate(&self) -> Result<()> {
        let config = parse_json_asset("config.json", &self.config_json)?;
        let generation = parse_json_asset("generation_config.json", &self.generation_config_json)?;
        let tokenizer = parse_json_asset("tokenizer.json", &self.tokenizer_json)?;
        let preprocessor = parse_json_asset(
            "preprocessor_config.json",
            &self.preprocessor_config_json,
        )?;

        validate_tiny_whisper_config(&config)?;
        validate_preprocessor_config(&preprocessor)?;
        require_json_object("generation_config.json", &generation)?;
        require_json_object("tokenizer.json", &tokenizer)?;
        validate_safetensors_container(&self.model_safetensors)?;
        Ok(())
    }
}

/// Validated first-slice browser provider configuration.
///
/// This type deliberately does not claim inference support yet. It establishes
/// the model and input boundary that the Burn model/decoder implementation must
/// satisfy before the public browser capability can leave
/// `pending-webgpu-provider`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurnWhisperWebGpuProvider {
    model: BurnWhisperWebGpuModel,
    assets: BurnWhisperWebGpuAssets,
}

impl BurnWhisperWebGpuProvider {
    pub fn new(model_id: &str, assets: BurnWhisperWebGpuAssets) -> Result<Self> {
        let model = BurnWhisperWebGpuModel::from_model_id(model_id)?;
        assets.validate()?;
        Ok(Self { model, assets })
    }

    pub const fn provider_id(&self) -> &'static str {
        BURN_WHISPER_WEBGPU_PROVIDER_ID
    }

    pub const fn model(&self) -> BurnWhisperWebGpuModel {
        self.model
    }

    pub fn assets(&self) -> &BurnWhisperWebGpuAssets {
        &self.assets
    }

    pub fn validate_audio_request(
        &self,
        audio: &LoadedAudio,
        task: TranscriptionTask,
    ) -> Result<()> {
        if task != TranscriptionTask::Transcribe {
            return Err(invalid_request(
                "Burn WebGPU Whisper browser MVP supports transcription only; translation remains unavailable",
            ));
        }
        if audio.sample_rate != BURN_WHISPER_WEBGPU_SAMPLE_RATE {
            return Err(invalid_request(format!(
                "Burn WebGPU Whisper requires {BURN_WHISPER_WEBGPU_SAMPLE_RATE} Hz PCM; got {} Hz",
                audio.sample_rate
            )));
        }
        if audio.channels != 1 {
            return Err(invalid_request(format!(
                "Burn WebGPU Whisper requires mono PCM; got {} channels",
                audio.channels
            )));
        }
        if audio.samples.is_empty() {
            return Err(invalid_request(
                "Burn WebGPU Whisper requires at least one PCM sample",
            ));
        }
        if audio.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(invalid_request(
                "Burn WebGPU Whisper PCM samples must all be finite",
            ));
        }
        Ok(())
    }
}

/// Non-fused Burn backend used by the browser implementation.
///
/// Burn documents that fusion may need to be disabled on WASM. The dependency
/// is therefore configured without default features and this alias names the
/// underlying WebGPU Cube backend directly instead of the fusion wrapper.
#[cfg(target_arch = "wasm32")]
pub type BurnWhisperWebGpuBackend =
    burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

/// Compile-time backend anchor used by the wasm32 gate.
#[cfg(target_arch = "wasm32")]
pub fn assert_burn_whisper_webgpu_backend() {
    fn assert_backend<B: burn::tensor::backend::Backend>() {}
    assert_backend::<BurnWhisperWebGpuBackend>();
}

fn parse_json_asset(name: &str, bytes: &[u8]) -> Result<Value> {
    if bytes.is_empty() {
        return Err(setup_error(format!(
            "Burn WebGPU Whisper requires non-empty {name} bytes"
        )));
    }
    serde_json::from_slice(bytes).map_err(|error| {
        setup_error(format!(
            "Burn WebGPU Whisper could not parse {name}: {error}"
        ))
    })
}

fn require_json_object(name: &str, value: &Value) -> Result<()> {
    if value.is_object() {
        Ok(())
    } else {
        Err(setup_error(format!(
            "Burn WebGPU Whisper requires {name} to contain a JSON object"
        )))
    }
}

fn validate_tiny_whisper_config(config: &Value) -> Result<()> {
    require_json_object("config.json", config)?;
    require_string(config, "model_type", "whisper")?;
    require_u64(config, "d_model", 384)?;
    require_u64(config, "encoder_layers", 4)?;
    require_u64(config, "decoder_layers", 4)?;
    require_u64(config, "encoder_attention_heads", 6)?;
    require_u64(config, "decoder_attention_heads", 6)?;
    require_u64(config, "num_mel_bins", 80)?;
    require_u64(config, "max_source_positions", 1_500)?;
    require_u64(config, "max_target_positions", 448)?;
    Ok(())
}

fn validate_preprocessor_config(config: &Value) -> Result<()> {
    require_json_object("preprocessor_config.json", config)?;
    require_u64(config, "feature_size", 80)?;
    require_u64(
        config,
        "sampling_rate",
        u64::from(BURN_WHISPER_WEBGPU_SAMPLE_RATE),
    )?;
    Ok(())
}

fn require_string(config: &Value, key: &str, expected: &str) -> Result<()> {
    let actual = config.get(key).and_then(Value::as_str).ok_or_else(|| {
        setup_error(format!(
            "Burn WebGPU Whisper config is missing string field {key:?}"
        ))
    })?;
    if actual == expected {
        Ok(())
    } else {
        Err(setup_error(format!(
            "Burn WebGPU Whisper config field {key:?} must be {expected:?}; got {actual:?}"
        )))
    }
}

fn require_u64(config: &Value, key: &str, expected: u64) -> Result<()> {
    let actual = config.get(key).and_then(Value::as_u64).ok_or_else(|| {
        setup_error(format!(
            "Burn WebGPU Whisper config is missing integer field {key:?}"
        ))
    })?;
    if actual == expected {
        Ok(())
    } else {
        Err(setup_error(format!(
            "Burn WebGPU Whisper config field {key:?} must be {expected}; got {actual}"
        )))
    }
}

fn validate_safetensors_container(bytes: &[u8]) -> Result<()> {
    if bytes.len() < 8 {
        return Err(setup_error(
            "Burn WebGPU Whisper model.safetensors is shorter than its header prefix",
        ));
    }
    let header_len = u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("slice length checked before safetensors header decode"),
    );
    let header_len = usize::try_from(header_len).map_err(|_| {
        setup_error("Burn WebGPU Whisper model.safetensors header is too large for this target")
    })?;
    let header_end = 8usize.checked_add(header_len).ok_or_else(|| {
        setup_error("Burn WebGPU Whisper model.safetensors header length overflowed")
    })?;
    if header_end > bytes.len() {
        return Err(setup_error(format!(
            "Burn WebGPU Whisper model.safetensors header declares {header_len} bytes but the asset contains only {} bytes after the prefix",
            bytes.len().saturating_sub(8)
        )));
    }
    let header: Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|error| {
        setup_error(format!(
            "Burn WebGPU Whisper model.safetensors header is not valid JSON: {error}"
        ))
    })?;
    require_json_object("model.safetensors header", &header)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_assets() -> BurnWhisperWebGpuAssets {
        let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut model_safetensors = (header.len() as u64).to_le_bytes().to_vec();
        model_safetensors.extend_from_slice(header);
        model_safetensors.extend_from_slice(&0.0f32.to_le_bytes());

        BurnWhisperWebGpuAssets {
            config_json: br#"{
                "model_type":"whisper",
                "d_model":384,
                "encoder_layers":4,
                "decoder_layers":4,
                "encoder_attention_heads":6,
                "decoder_attention_heads":6,
                "num_mel_bins":80,
                "max_source_positions":1500,
                "max_target_positions":448
            }"#
            .to_vec(),
            generation_config_json: br#"{}"#.to_vec(),
            tokenizer_json: br#"{}"#.to_vec(),
            preprocessor_config_json:
                br#"{"feature_size":80,"sampling_rate":16000}"#.to_vec(),
            model_safetensors,
        }
    }

    fn mono_audio() -> LoadedAudio {
        LoadedAudio {
            samples: vec![0.0, 0.25, -0.25],
            sample_rate: BURN_WHISPER_WEBGPU_SAMPLE_RATE,
            channels: 1,
            source: Some("browser-test".to_string()),
        }
    }

    #[test]
    fn accepts_tiny_and_tiny_en_caller_owned_assets() {
        for model_id in ["tiny", WHISPER_TINY_ID, "tiny.en", WHISPER_TINY_EN_ID] {
            let provider = BurnWhisperWebGpuProvider::new(model_id, tiny_assets()).unwrap();
            provider
                .validate_audio_request(&mono_audio(), TranscriptionTask::Transcribe)
                .unwrap();
        }
    }

    #[test]
    fn rejects_models_outside_the_first_browser_slice() {
        let error = BurnWhisperWebGpuProvider::new("openai/whisper-base", tiny_assets())
            .unwrap_err()
            .to_string();
        assert!(error.contains("whisper-tiny"));
        assert!(error.contains("whisper-tiny.en"));
    }

    #[test]
    fn rejects_non_tiny_model_dimensions_before_webgpu_work() {
        let mut assets = tiny_assets();
        assets.config_json = br#"{
            "model_type":"whisper",
            "d_model":512,
            "encoder_layers":4,
            "decoder_layers":4,
            "encoder_attention_heads":6,
            "decoder_attention_heads":6,
            "num_mel_bins":80,
            "max_source_positions":1500,
            "max_target_positions":448
        }"#
        .to_vec();
        let error = BurnWhisperWebGpuProvider::new("tiny", assets)
            .unwrap_err()
            .to_string();
        assert!(error.contains("d_model"));
        assert!(error.contains("384"));
    }

    #[test]
    fn rejects_translation_and_invalid_browser_pcm() {
        let provider = BurnWhisperWebGpuProvider::new("tiny.en", tiny_assets()).unwrap();
        assert!(provider
            .validate_audio_request(&mono_audio(), TranscriptionTask::Translate)
            .is_err());

        let mut wrong_rate = mono_audio();
        wrong_rate.sample_rate = 48_000;
        assert!(provider
            .validate_audio_request(&wrong_rate, TranscriptionTask::Transcribe)
            .is_err());

        let mut stereo = mono_audio();
        stereo.channels = 2;
        assert!(provider
            .validate_audio_request(&stereo, TranscriptionTask::Transcribe)
            .is_err());

        let mut non_finite = mono_audio();
        non_finite.samples[0] = f32::NAN;
        assert!(provider
            .validate_audio_request(&non_finite, TranscriptionTask::Transcribe)
            .is_err());
    }
}
