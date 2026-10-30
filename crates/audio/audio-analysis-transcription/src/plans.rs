//! Static transcription provider plans.

use crate::*;

/// Returns provider plans.
pub fn transcription_provider_plans() -> Vec<TranscriptionProviderPlan> {
    vec![
        candle_whisper_provider_plan(),
        whisper_cpp_provider_plan(),
        whisperx_provider_plan(),
    ]
}

/// Returns the primary Candle Whisper provider plan.
pub fn candle_whisper_provider_plan() -> TranscriptionProviderPlan {
    TranscriptionProviderPlan {
        provider_id: "candle-whisper".to_string(),
        external_runtime: false,
        wasm_supported: false,
        primary: true,
        setup: vec![
            "Provide an offline model bundle with config.json, generation_config.json, tokenizer.json, preprocessor_config.json, and model.safetensors.".to_string(),
            "Build with feature `candle`; add `cuda` for CUDA device execution.".to_string(),
        ],
        diagnostics: vec![
            "Candle Whisper is the primary planned Rust-native ASR and translate-to-English provider.".to_string(),
            "Set task=translate for native Whisper translation; wav2vec2/CTC alignment is not supported for translated output.".to_string(),
            "Default tests do not download models or require CUDA.".to_string(),
            "Default decodeRuntime=autoregressiveKvCache preserves the safe per-window KV-cache path.".to_string(),
            "decodeRuntime=activeRowTensorBatch enables true tensor-batched active-row decode for eligible multi-window native Candle Whisper input.".to_string(),
        ],
    }
}

/// Returns the whisper.cpp compatibility provider plan.
pub fn whisper_cpp_provider_plan() -> TranscriptionProviderPlan {
    TranscriptionProviderPlan {
        provider_id: "whisper-cpp".to_string(),
        external_runtime: false,
        wasm_supported: false,
        primary: false,
        setup: vec!["Provide a local whisper.cpp model path explicitly.".to_string()],
        diagnostics: vec![
            "whisper.cpp is retained as a native compatibility provider, not the primary ASR path."
                .to_string(),
            "Whisper translate is not supported through this provider in this crate.".to_string(),
        ],
    }
}

/// Returns the external WhisperX provider plan.
pub fn whisperx_provider_plan() -> TranscriptionProviderPlan {
    TranscriptionProviderPlan {
        provider_id: "whisperx-command".to_string(),
        external_runtime: true,
        wasm_supported: false,
        primary: false,
        setup: vec![
            "Install whisperx in the active Python environment.".to_string(),
            "Ensure ffmpeg is available on PATH.".to_string(),
            "Set HF_TOKEN before diarization requests.".to_string(),
        ],
        diagnostics: vec![
            "WhisperX execution is opt-in and never required by default tests.".to_string(),
            "The compatibility command path forwards task=transcribe or task=translate to Python WhisperX.".to_string(),
            "Transcript normalization and WhisperX JSON import are owned by the provider adapter."
                .to_string(),
        ],
    }
}

