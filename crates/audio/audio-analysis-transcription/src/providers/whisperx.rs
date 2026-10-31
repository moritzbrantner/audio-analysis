//! External WhisperX command provider.

use crate::*;

/// External command provider for Python WhisperX.
#[derive(Debug, Clone, Default)]
pub struct WhisperXCommandTranscriber;

impl WhisperXCommandTranscriber {
    pub fn transcribe_pipeline(
        &mut self,
        request: TranscriptionPipelineRequest,
    ) -> Result<TranscriptionPipelineResponse> {
        match request.provider {
            TranscriptionProviderSelection::ExternalWhisperX(options) => {
                run_whisperx_command(request.source.path()?, options)
            }
            other => Err(invalid_request(format!(
                "whisperx-command cannot run provider `{}`",
                other.provider_id()
            ))),
        }
    }
}

pub(crate) fn run_whisperx_command(
    source_path: &Path,
    options: WhisperXCommandOptions,
) -> Result<TranscriptionPipelineResponse> {
    let task = options.task;
    let output_dir = options
        .output_dir
        .clone()
        .unwrap_or_else(default_whisperx_output_dir);
    fs::create_dir_all(&output_dir)?;

    let hf_token = if options.diarize {
        let env_name = options
            .hf_token_env
            .clone()
            .unwrap_or_else(|| "HF_TOKEN".to_string());
        Some(std::env::var(&env_name).map_err(|_| {
            setup_error(format!(
                "diarization requires `{env_name}` to be set before running WhisperX"
            ))
        })?)
    } else {
        None
    };

    let mut command = whisperx_command(source_path, &output_dir, &options, hf_token.as_deref());
    let child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                setup_error(format!(
                    "WhisperX command `{}` was not found; install whisperx or pass provider.command",
                    options.command.display()
                ))
            } else {
                DetectError::Io(error)
            }
        })?;
    let output = wait_with_optional_timeout(child, &options.command, options.timeout_seconds)?;
    if !output.status.success() {
        return Err(setup_error(format!(
            "WhisperX command `{}` failed: {}",
            options.command.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    let (transcript_path, transcript_bytes) =
        whisperx_json_bytes(source_path, &output_dir, &output.stdout).ok_or_else(|| {
            model_output_mismatch(format!(
                "WhisperX completed but no JSON transcript was found in `{}`",
                output_dir.display()
            ))
        })?;
    let mut transcript = import_whisperx_json(&transcript_bytes)?;
    if transcript.source.is_none() {
        transcript.source = Some(source_path.to_string_lossy().into_owned());
    }
    let vad_segments = whisperx_stdout_vad_segments(&output.stdout);
    let artifacts = transcript_path
        .map(|path| TranscriptionArtifact {
            kind: "whisperx-json".to_string(),
            path,
        })
        .into_iter()
        .collect();
    let mut diagnostics = vec![
        format!("asrTask={}", task.as_whisper_task()),
        format!("ran WhisperX output in `{}`", output_dir.display()),
        "parsed WhisperX JSON in the transcription provider adapter".to_string(),
    ];
    if task == TranscriptionTask::Translate {
        diagnostics.push("translationRuntime=whisperx-command".to_string());
        if let Some(language) = task.output_language_hint() {
            diagnostics.push(format!("translationTargetLanguage={language}"));
        }
    }
    if !vad_segments.is_empty() {
        diagnostics.push(format!(
            "whisperxVadSegmentsFromStdout={}",
            vad_segments.len()
        ));
    }
    Ok(TranscriptionPipelineResponse {
        accepted: true,
        operation: "audio.transcription.transcribe".to_string(),
        provider: "whisperx-command".to_string(),
        model_id: options.model,
        transcript,
        vad_segments,
        alignment: None,
        diarization: None,
        artifacts,
        diagnostics,
    })
}

pub(crate) fn whisperx_command(
    source_path: &Path,
    output_dir: &Path,
    options: &WhisperXCommandOptions,
    hf_token: Option<&str>,
) -> Command {
    let mut command = Command::new(&options.command);
    command.args(whisperx_args(source_path, output_dir, options));
    if let Some(hf_token) = hf_token {
        command.env("HF_TOKEN", hf_token);
    }
    command
}

pub(crate) fn whisperx_args(
    source_path: &Path,
    output_dir: &Path,
    options: &WhisperXCommandOptions,
) -> Vec<String> {
    let mut args = vec![
        source_path.to_string_lossy().into_owned(),
        "--model".to_string(),
        options.model.clone(),
        "--task".to_string(),
        options.task.as_whisper_task().to_string(),
        "--device".to_string(),
        options.device.as_str().to_string(),
        "--output_format".to_string(),
        "json".to_string(),
        "--output_dir".to_string(),
        output_dir.to_string_lossy().into_owned(),
    ];
    if let Some(language) = &options.language {
        args.extend(["--language".to_string(), language.clone()]);
    }
    if let Some(compute_type) = &options.compute_type {
        args.extend(["--compute_type".to_string(), compute_type.clone()]);
    }
    if let Some(batch_size) = options.batch_size {
        args.extend(["--batch_size".to_string(), batch_size.to_string()]);
    }
    if options.no_align {
        args.push("--no_align".to_string());
    }
    if let Some(align_model) = &options.align_model {
        args.extend(["--align_model".to_string(), align_model.clone()]);
    }
    if let Some(model_dir) = &options.model_dir {
        args.extend([
            "--model_dir".to_string(),
            model_dir.to_string_lossy().into_owned(),
        ]);
    }
    if options.model_cache_only {
        args.push("--model_cache_only".to_string());
    }
    args.extend([
        "--interpolate_method".to_string(),
        options.interpolate_method.as_whisperx_arg().to_string(),
    ]);
    if options.return_char_alignments {
        args.push("--return_char_alignments".to_string());
    }
    if options.diarize {
        args.push("--diarize".to_string());
        if let Some(min_speakers) = options.min_speakers {
            args.extend(["--min_speakers".to_string(), min_speakers.to_string()]);
        }
        if let Some(max_speakers) = options.max_speakers {
            args.extend(["--max_speakers".to_string(), max_speakers.to_string()]);
        }
    }
    args.extend(options.extra_args.clone());
    args
}

pub(crate) fn whisperx_json_bytes(
    source_path: &Path,
    output_dir: &Path,
    stdout: &[u8],
) -> Option<(Option<PathBuf>, Vec<u8>)> {
    let path = find_json_artifact_for_source(output_dir, source_path);
    if let Some(path) = path {
        return fs::read(&path).ok().map(|bytes| (Some(path), bytes));
    }
    serde_json::from_slice::<serde_json::Value>(stdout)
        .ok()
        .map(|_| (None, stdout.to_vec()))
}

pub(crate) fn whisperx_stdout_vad_segments(stdout: &[u8]) -> Vec<SpeechSpan> {
    let stdout = String::from_utf8_lossy(stdout);
    stdout
        .lines()
        .filter_map(|line| {
            let start_marker = "Transcript: [";
            let start = line.find(start_marker)? + start_marker.len();
            let range = line[start..].split_once(']')?.0;
            let (start_seconds, end_seconds) = range.split_once("-->")?;
            let start_seconds = start_seconds.trim().parse::<f64>().ok()?;
            let end_seconds = end_seconds.trim().parse::<f64>().ok()?;
            SpeechSpan::new(start_seconds, end_seconds, 1.0).ok()
        })
        .collect()
}

pub(crate) fn find_json_artifact_for_source(
    output_dir: &Path,
    source_path: &Path,
) -> Option<PathBuf> {
    let expected = source_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .map(|stem| output_dir.join(format!("{stem}.json")));
    if let Some(expected) = expected.filter(|path| path.is_file()) {
        return Some(expected);
    }

    let mut candidates = fs::read_dir(output_dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().next()
}

pub(crate) fn default_whisperx_output_dir() -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    std::env::temp_dir().join(format!("video-analysis-whisperx-{millis}"))
}

pub(crate) fn wait_with_optional_timeout(
    mut child: Child,
    command: &Path,
    timeout_seconds: Option<u64>,
) -> Result<Output> {
    let Some(seconds) = timeout_seconds else {
        return child.wait_with_output().map_err(DetectError::Io);
    };
    let started = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            return child.wait_with_output().map_err(DetectError::Io);
        }
        if started.elapsed() >= Duration::from_secs(seconds) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(timeout_error(format!(
                "WhisperX command `{}` timed out after {seconds} seconds",
                command.display()
            )));
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}
