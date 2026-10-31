//! whisper.cpp compatibility provider.

use crate::*;

/// Runs the compatibility whisper.cpp file adapter and returns the text-owned
/// transcript document contract.
pub fn transcribe_whisper_cpp_file_with_progress(
    config: WhisperCppConfig,
    input: &Path,
    progress: &mut dyn FnMut(WhisperCppProgressEvent),
) -> Result<TranscriptionContract> {
    let mut transcriber = NativeWhisperCppTranscriber::new(config);
    let transcript = transcriber
        .transcribe_file_with_progress(input, progress)
        .map_err(|error| DetectError::Source(error.to_string()))?;
    let language = transcript.language.clone();
    let segments = transcript
        .segments
        .into_iter()
        .map(|segment| {
            let mut contract = TranscriptSegmentContract::new(segment.index, segment.text)
                .with_time_range(segment.start_seconds, segment.end_seconds)
                .map_err(|error| DetectError::Source(error.to_string()))?
                .with_confidence(sanitize_confidence(segment.confidence))
                .map_err(|error| DetectError::Source(error.to_string()))?;
            contract.language = language.clone();
            Ok(contract)
        })
        .collect::<Result<Vec<_>>>()?;
    normalize_transcription_contract(TranscriptionContract {
        text: transcript.text,
        language,
        segments,
        source: transcript.source,
        attributes: BTreeMap::new(),
    })
    .map_err(|error| DetectError::Source(error.to_string()))
}

/// Native whisper.cpp compatibility provider.
#[derive(Debug, Clone, Default)]
pub struct WhisperCppTranscriber {
    pub options: WhisperCppProviderOptions,
}

impl AudioTranscriptionProvider for WhisperCppTranscriber {
    fn provider_id(&self) -> &str {
        "whisper-cpp"
    }

    fn transcribe(&mut self, _request: AsrRequest) -> Result<AsrResponse> {
        let Some(model_path) = &self.options.model_path else {
            return Err(setup_error("required whisper.cpp model path is missing"));
        };
        if !model_path.exists() {
            return Err(setup_error(format!(
                "required whisper.cpp model `{}` is missing",
                model_path.display()
            )));
        }
        Err(unsupported_runtime(
            "whisper.cpp compatibility provider is not the primary transcription path",
        ))
    }
}

