use media_core::{DetectError, Result};
use serde_json::Value;

pub const PROVIDER_ID: &str = "burn-whisper-webgpu";
pub const SAMPLE_RATE: u32 = 16_000;
const WHISPER_TINY_ID: &str = "openai/whisper-tiny";
const WHISPER_TINY_EN_ID: &str = "openai/whisper-tiny.en";

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
                "Burn WebGPU Whisper supports only {WHISPER_TINY_ID} and {WHISPER_TINY_EN_ID}; got {other:?}"
            ))),
        }
    }

    pub const fn model_id(self) -> &'static str {
        match self {
            Self::Tiny => WHISPER_TINY_ID,
            Self::TinyEn => WHISPER_TINY_EN_ID,
        }
    }
}

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
        let preprocessor = parse_json_asset("preprocessor_config.json", &self.preprocessor_config_json)?;

        require_json_object("config.json", &config)?;
        require_string(&config, "model_type", "whisper")?;
        require_u64(&config, "d_model", 384)?;
        require_u64(&config, "encoder_layers", 4)?;
        require_u64(&config, "decoder_layers", 4)?;
        require_u64(&config, "encoder_attention_heads", 6)?;
        require_u64(&config, "decoder_attention_heads", 6)?;
        require_u64(&config, "num_mel_bins", 80)?;
        require_u64(&config, "max_source_positions", 1_500)?;
        require_u64(&config, "max_target_positions", 448)?;
        require_json_object("generation_config.json", &generation)?;
        require_json_object("tokenizer.json", &tokenizer)?;
        require_json_object("preprocessor_config.json", &preprocessor)?;
        require_u64(&preprocessor, "feature_size", 80)?;
        require_u64(&preprocessor, "sampling_rate", u64::from(SAMPLE_RATE))?;
        validate_safetensors_container(&self.model_safetensors)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BurnWhisperWebGpuAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BurnWhisperWebGpuTask {
    Transcribe,
    Translate,
}

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
        PROVIDER_ID
    }

    pub const fn model(&self) -> BurnWhisperWebGpuModel {
        self.model
    }

    pub fn assets(&self) -> &BurnWhisperWebGpuAssets {
        &self.assets
    }

    pub fn validate_audio_request(
        &self,
        audio: &BurnWhisperWebGpuAudio,
        task: BurnWhisperWebGpuTask,
    ) -> Result<()> {
        if task != BurnWhisperWebGpuTask::Transcribe {
            return Err(invalid_request(
                "browser MVP is transcription-only; translation remains unavailable",
            ));
        }
        if audio.sample_rate != SAMPLE_RATE {
            return Err(invalid_request(format!(
                "Burn WebGPU Whisper requires {SAMPLE_RATE} Hz PCM; got {} Hz",
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
            return Err(invalid_request("Burn WebGPU Whisper requires PCM samples"));
        }
        if audio.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(invalid_request("Burn WebGPU Whisper PCM must be finite"));
        }
        Ok(())
    }
}

// Burn documents that fusion may need to be disabled on wasm. The binding uses
// the underlying Cube backend directly and enables only Burn's WebGPU/WGSL path.
#[cfg(target_arch = "wasm32")]
pub type BurnWhisperWebGpuBackend =
    burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

#[cfg(target_arch = "wasm32")]
pub fn assert_backend_contract() {
    fn assert_backend<B: burn_core::tensor::backend::Backend>() {}
    assert_backend::<BurnWhisperWebGpuBackend>();
}

fn parse_json_asset(name: &str, bytes: &[u8]) -> Result<Value> {
    if bytes.is_empty() {
        return Err(setup_error(format!("{name} must not be empty")));
    }
    serde_json::from_slice(bytes)
        .map_err(|error| setup_error(format!("could not parse {name}: {error}")))
}

fn require_json_object(name: &str, value: &Value) -> Result<()> {
    value
        .is_object()
        .then_some(())
        .ok_or_else(|| setup_error(format!("{name} must contain a JSON object")))
}

fn require_string(config: &Value, key: &str, expected: &str) -> Result<()> {
    let actual = config
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| setup_error(format!("missing string field {key:?}")))?;
    (actual == expected)
        .then_some(())
        .ok_or_else(|| setup_error(format!("field {key:?} must be {expected:?}; got {actual:?}")))
}

fn require_u64(config: &Value, key: &str, expected: u64) -> Result<()> {
    let actual = config
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| setup_error(format!("missing integer field {key:?}")))?;
    (actual == expected)
        .then_some(())
        .ok_or_else(|| setup_error(format!("field {key:?} must be {expected}; got {actual}")))
}

fn validate_safetensors_container(bytes: &[u8]) -> Result<()> {
    if bytes.len() < 8 {
        return Err(setup_error("model.safetensors is shorter than its header prefix"));
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
        .map_err(|_| setup_error("model.safetensors header is too large"))?;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| setup_error("model.safetensors header length overflowed"))?;
    if header_end > bytes.len() {
        return Err(setup_error("model.safetensors header exceeds asset length"));
    }
    let header: Value = serde_json::from_slice(&bytes[8..header_end])
        .map_err(|error| setup_error(format!("invalid safetensors header JSON: {error}")))?;
    require_json_object("model.safetensors header", &header)
}

fn invalid_request(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("invalid_request: {}", message.into()))
}

fn setup_error(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("setup_error: {}", message.into()))
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
            config_json: br#"{"model_type":"whisper","d_model":384,"encoder_layers":4,"decoder_layers":4,"encoder_attention_heads":6,"decoder_attention_heads":6,"num_mel_bins":80,"max_source_positions":1500,"max_target_positions":448}"#.to_vec(),
            generation_config_json: br#"{}"#.to_vec(),
            tokenizer_json: br#"{}"#.to_vec(),
            preprocessor_config_json: br#"{"feature_size":80,"sampling_rate":16000}"#.to_vec(),
            model_safetensors,
        }
    }

    fn mono_audio() -> BurnWhisperWebGpuAudio {
        BurnWhisperWebGpuAudio {
            samples: vec![0.0, 0.25, -0.25],
            sample_rate: SAMPLE_RATE,
            channels: 1,
        }
    }

    #[test]
    fn validates_first_slice_assets_and_audio() {
        for model in ["tiny", WHISPER_TINY_ID, "tiny.en", WHISPER_TINY_EN_ID] {
            BurnWhisperWebGpuProvider::new(model, tiny_assets())
                .unwrap()
                .validate_audio_request(&mono_audio(), BurnWhisperWebGpuTask::Transcribe)
                .unwrap();
        }
    }

    #[test]
    fn fails_closed_for_unsupported_model_task_or_pcm() {
        assert!(BurnWhisperWebGpuProvider::new("base", tiny_assets()).is_err());
        let provider = BurnWhisperWebGpuProvider::new("tiny", tiny_assets()).unwrap();
        assert!(provider
            .validate_audio_request(&mono_audio(), BurnWhisperWebGpuTask::Translate)
            .is_err());
        let mut wrong_rate = mono_audio();
        wrong_rate.sample_rate = 48_000;
        assert!(provider
            .validate_audio_request(&wrong_rate, BurnWhisperWebGpuTask::Transcribe)
            .is_err());
    }
}
