//! Library-owned runtime surface for `audio-analysis-separation`.

use std::path::{Path, PathBuf};

use runtime_core::{
    structured_surface_response, OperationId, PackageSurface, RuntimeCapabilities,
    SurfaceOperation, SurfaceRequest, SurfaceResponse,
};

use crate::{
    is_demucs_command_available, DemucsModel, HtdemucsOptions, HtdemucsSeparator,
    SeparationOutputFormat, Stem, StemLayout,
};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Describe package",
                "Deterministic source-separation command planning for video-analysis.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "audio.separation.models",
                "Inspect separation models",
                "Inspects known Demucs model layouts without executing Demucs.",
                serde_json::json!({}),
            ),
            operation(
                "audio.separation.plan",
                "Preview separation command",
                "Previews a Demucs command without running an external process.",
                serde_json::json!({"input": "song.wav", "outputDir": "stems", "model": "htdemucs", "format": "wav"}),
            ),
            operation(
                "audio.separation.expectedStems",
                "Expected stems",
                "Returns expected separated stem filenames for a model/layout.",
                serde_json::json!({"model": "htdemucs", "format": "wav"}),
            ),
            operation_with_support(
                "audio.separation.runDemucs",
                "Run Demucs",
                "Opt-in native/server Demucs execution with setup checks before spawning.",
                serde_json::json!({"input": "song.wav", "outputDir": "stems", "model": "htdemucs", "format": "wav", "execute": false}),
                false,
                true,
            ),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    operation_with_support(id, name, description, example_request, true, true)
}

fn operation_with_support(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
    wasm_supported: bool,
    server_supported: bool,
) -> SurfaceOperation {
    SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true, "xOperationCategory": runtime_core::operation_category(id)}),
        output_schema: serde_json::json!({"type": "object", "xOperationCategory": runtime_core::operation_category(id)}),
        example_request,
        wasm_supported,
        server_supported,
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "audio.separation.models" => models_value(),
        "audio.separation.plan" => plan_value(request.input)?,
        "audio.separation.expectedStems" => expected_stems_value(request.input)?,
        "audio.separation.runDemucs" => run_demucs_value(request.input)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ));
        }
    };
    Ok(response(operation, value))
}

fn response(operation: OperationId, value: serde_json::Value) -> SurfaceResponse {
    let (title, message, summary) = match operation.as_str() {
        "describe" => (
            "Separation package metadata",
            "Inspected the deterministic source-separation planning operations exposed by this package.",
            serde_json::json!({
                "operationCount": value.get("operationCount").cloned().unwrap_or(serde_json::Value::Null)
            }),
        ),
        "audio.separation.models" => (
            "Separation model inventory",
            "Inspected known Demucs model layouts without executing Demucs.",
            serde_json::json!({
                "backend": value.get("backend").cloned().unwrap_or(serde_json::Value::Null),
                "modelCount": value.get("models").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.separation.plan" => (
            "Separation command preview",
            "Built a Demucs command preview only; this operation does not run Demucs, read audio, or write stems.",
            serde_json::json!({
                "program": value.get("program").cloned().unwrap_or(serde_json::Value::Null),
                "model": value.get("model").cloned().unwrap_or(serde_json::Value::Null),
                "outputDir": value.get("outputDir").cloned().unwrap_or(serde_json::Value::Null),
                "executed": value.get("executed").cloned().unwrap_or(serde_json::Value::Null),
                "expectedStemCount": value.get("expectedStemPaths").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.separation.expectedStems" => (
            "Expected separation stems",
            "Computed deterministic separated stem filenames for the requested model and output format.",
            serde_json::json!({
                "model": value.get("model").cloned().unwrap_or(serde_json::Value::Null),
                "layout": value.get("layout").cloned().unwrap_or(serde_json::Value::Null),
                "stemCount": value.get("stems").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        "audio.separation.runDemucs" => (
            "Demucs execution result",
            "Handled an opt-in Demucs execution request with setup checks before spawning.",
            serde_json::json!({
                "model": value.get("model").cloned().unwrap_or(serde_json::Value::Null),
                "executed": value.get("executed").cloned().unwrap_or(serde_json::Value::Null),
                "requiresExternalTool": value.get("requiresExternalTool").cloned().unwrap_or(serde_json::Value::Null),
                "stemCount": value.get("stems").and_then(serde_json::Value::as_array).map_or(0, Vec::len)
            }),
        ),
        _ => (
            "Separation operation result",
            "Completed the separation package surface operation.",
            serde_json::json!({}),
        ),
    };
    structured_surface_response(operation, title, message, summary, value)
}

fn describe_value(input: serde_json::Value) -> serde_json::Value {
    let surface = package_surface();
    serde_json::json!({
        "library": surface.library,
        "version": surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface.operations.iter().map(|operation| operation.id.as_str()).collect::<Vec<_>>(),
        "input": input
    })
}

fn models_value() -> serde_json::Value {
    let models = [
        DemucsModel::Htdemucs,
        DemucsModel::HtdemucsFt,
        DemucsModel::Htdemucs6s,
        DemucsModel::MdX,
        DemucsModel::MdXExtra,
        DemucsModel::MdXQ,
    ];
    serde_json::json!({
        "backend": "demucs",
        "defaultModel": DemucsModel::Htdemucs.as_str(),
        "models": models.iter().map(|model| serde_json::json!({
            "id": model.as_str(),
            "layout": layout_name(&model.default_layout()),
            "stems": model.default_layout().stems().iter().map(Stem::as_str).collect::<Vec<_>>()
        })).collect::<Vec<_>>()
    })
}

fn plan_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let input_path = string_field(&input, "input", "input.wav")?;
    let separator = separator_from_input(&input)?;
    let output_dir = separator.options.output_dir.clone();
    let output_dir_string = output_dir.to_string_lossy().to_string();
    let format = output_format(&input)?;
    let command = separator
        .build_command(PathBuf::from(&input_path))
        .map_err(|error| error.to_string())?;
    let args = command
        .args
        .iter()
        .map(|arg| arg.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let expected_stem_paths = separator
        .expected_stems()
        .iter()
        .map(|stem| format!("{}/{}", output_dir_string, stem.file_name(&format)))
        .collect::<Vec<_>>();
    let command_preview = command_preview(&command.program, &args);
    Ok(serde_json::json!({
        "input": input_path,
        "outputDir": output_dir_string,
        "model": separator.options.model.as_str(),
        "outputLayout": layout_value(&separator.expected_layout()),
        "program": command.program.clone(),
        "args": args,
        "commandPreview": command_preview,
        "requiresExternalTool": true,
        "executed": false,
        "doesNot": ["read audio", "write stems", "run Demucs"],
        "setup": {
            "pythonPackage": "demucs",
            "exampleCommand": "python -m pip install demucs",
            "expectedExecutable": command.program
        },
        "setupCommands": ["python -m pip install demucs"],
        "missingToolBehavior": "A native execution path should report a missing Demucs executable before spawning a command; this surface operation only previews the command.",
        "expectedStems": separator.expected_stems().iter().map(Stem::as_str).collect::<Vec<_>>(),
        "expectedStemPaths": expected_stem_paths,
        "diagnostics": ["Plan preview only; no audio is read and Demucs is not executed."]
    }))
}

fn expected_stems_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let separator = separator_from_input(&input)?;
    let format = output_format(&input)?;
    Ok(serde_json::json!({
        "model": separator.options.model.as_str(),
        "layout": layout_name(&separator.expected_layout()),
        "outputLayout": layout_value(&separator.expected_layout()),
        "format": format.extension(),
        "requiresExternalTool": false,
        "setupCommands": ["python -m pip install demucs"],
        "diagnostics": ["Expected stems are computed from model/layout metadata only."],
        "stems": separator.expected_stems().iter().map(|stem| serde_json::json!({
            "stem": stem.as_str(),
            "fileName": stem.file_name(&format)
        })).collect::<Vec<_>>()
    }))
}

fn run_demucs_value(input: serde_json::Value) -> Result<serde_json::Value, String> {
    let execute = input
        .get("execute")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let input_path = string_field(&input, "input", "")?;
    let separator = separator_from_input(&input)?;
    let format = output_format(&input)?;
    let command = separator
        .build_command(PathBuf::from(&input_path))
        .map_err(|error| error.to_string())?;
    let args = command
        .args
        .iter()
        .map(|arg| arg.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    if !execute {
        return Ok(serde_json::json!({
            "input": input_path,
            "model": separator.options.model.as_str(),
            "outputLayout": layout_value(&separator.expected_layout()),
            "format": format.extension(),
            "program": command.program.clone(),
            "args": args,
            "commandPreview": command_preview(&command.program, &args),
            "requiresExternalTool": true,
            "serverSupported": true,
            "wasmSupported": false,
            "executed": false,
            "setupCommands": ["python -m pip install demucs"],
            "diagnostics": ["Set execute=true in a native/server runtime to run Demucs."]
        }));
    }
    if !is_demucs_command_available(&command.program) {
        return Err(format!(
            "Demucs executable `{}` is not available; run `python -m pip install demucs` or pass a valid command",
            command.program.display()
        ));
    }
    let input_path_buf = PathBuf::from(&input_path);
    if !input_path_buf.is_file() {
        return Err(format!(
            "demucs input `{}` does not exist or is not a file",
            input_path_buf.display()
        ));
    }
    let result = separator
        .separate(&input_path_buf)
        .map_err(|error| error.to_string())?;
    Ok(serde_json::json!({
        "input": input_path,
        "model": result.model.as_str(),
        "outputDir": result.output_dir,
        "outputLayout": layout_value(&result.layout),
        "format": format.extension(),
        "requiresExternalTool": true,
        "executed": true,
        "allOutputsPresent": result.all_outputs_present,
        "missingStems": result.missing_stems.iter().map(Stem::as_str).collect::<Vec<_>>(),
        "stems": result.stems.iter().map(|stem| serde_json::json!({
            "stem": stem.stem.as_str(),
            "path": stem.path,
            "exists": stem.exists,
            "bytes": stem.bytes
        })).collect::<Vec<_>>(),
        "diagnostics": ["Demucs completed and expected outputs were inspected."]
    }))
}

fn separator_from_input(input: &serde_json::Value) -> Result<HtdemucsSeparator, String> {
    let mut options = HtdemucsOptions::new(string_field(input, "outputDir", "stems")?)
        .command(string_field(input, "command", "demucs")?)
        .model(string_field(input, "model", "htdemucs")?)
        .output_format(output_format(input)?);
    if let Some(primary) = input.get("twoStems").and_then(serde_json::Value::as_str) {
        options = options.two_stems(primary.parse::<Stem>().map_err(|error| error.to_string())?);
    }
    if input
        .get("layout")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|layout| layout == "sixStem" || layout == "six-stem")
    {
        options = options.layout(StemLayout::SixStem);
    }
    HtdemucsSeparator::new(options).map_err(|error| error.to_string())
}

fn output_format(input: &serde_json::Value) -> Result<SeparationOutputFormat, String> {
    match input.get("format").and_then(serde_json::Value::as_str) {
        Some("mp3") => Ok(SeparationOutputFormat::Mp3),
        Some("flac") => Ok(SeparationOutputFormat::Flac),
        Some("wav") | None => Ok(SeparationOutputFormat::Wav),
        Some(other) => Err(format!("unsupported separation output format `{other}`")),
    }
}

fn command_preview(program: &Path, args: &[String]) -> String {
    std::iter::once(program.display().to_string())
        .chain(args.iter().map(|arg| shell_preview_arg(arg)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_preview_arg(arg: &str) -> String {
    if arg
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_./:{}".contains(ch))
    {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}

fn string_field(
    input: &serde_json::Value,
    field: &str,
    default_value: &str,
) -> Result<String, String> {
    let value = input
        .get(field)
        .and_then(serde_json::Value::as_str)
        .unwrap_or(default_value)
        .to_string();
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    Ok(value)
}

fn layout_name(layout: &StemLayout) -> &'static str {
    match layout {
        StemLayout::FourStem => "fourStem",
        StemLayout::SixStem => "sixStem",
        StemLayout::TwoStem { .. } => "twoStem",
        StemLayout::Custom(_) => "custom",
    }
}

fn layout_value(layout: &StemLayout) -> serde_json::Value {
    serde_json::json!({
        "name": layout_name(layout),
        "stems": layout.stems().iter().map(Stem::as_str).collect::<Vec<_>>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_surface_lists_separation_operations() {
        let surface = package_surface();
        let ids = surface
            .operations
            .iter()
            .map(|operation| operation.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"audio.separation.models"));
        assert!(ids.contains(&"audio.separation.plan"));
        assert!(ids.contains(&"audio.separation.runDemucs"));
    }

    #[test]
    fn plan_operation_does_not_execute() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.separation.plan"),
            input: serde_json::json!({"input": "song.wav", "outputDir": "stems"}),
        })
        .expect("plan");
        assert_eq!(response.value["operation"], "audio.separation.plan");
        assert!(response.value["title"].is_string());
        assert!(response.value["summary"].is_object());
        assert!(response.value["result"].is_object());
        assert_eq!(response.value["executed"], false);
        assert_eq!(response.value["model"], "htdemucs");
        assert!(
            response.value["expectedStemPaths"]
                .as_array()
                .unwrap()
                .len()
                >= 4
        );
        assert!(response.value["missingToolBehavior"]
            .as_str()
            .unwrap()
            .contains("missing Demucs"));
        assert!(response.value["args"]
            .as_array()
            .unwrap()
            .iter()
            .any(|arg| arg == "song.wav"));
        assert_eq!(response.value["requiresExternalTool"], true);
        assert!(response.value["commandPreview"]
            .as_str()
            .unwrap()
            .contains("demucs"));
        assert!(response.value["outputLayout"].is_object());
    }

    #[test]
    fn run_demucs_default_is_non_executing_plan() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.separation.runDemucs"),
            input: serde_json::json!({"input": "song.wav", "execute": false}),
        })
        .expect("run plan");
        assert_eq!(response.value["operation"], "audio.separation.runDemucs");
        assert_eq!(response.value["executed"], false);
        assert_eq!(response.value["wasmSupported"], false);
        assert_eq!(response.value["requiresExternalTool"], true);
    }

    #[test]
    fn invalid_output_format_returns_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.separation.expectedStems"),
            input: serde_json::json!({"format": "aac"}),
        })
        .unwrap_err();
        assert!(error.contains("format"));
    }

    #[test]
    fn example_requests_run_with_structured_outputs() {
        for operation in package_surface().operations {
            let response = run_surface_operation(SurfaceRequest {
                operation: operation.id.clone(),
                input: operation.example_request.clone(),
            })
            .unwrap_or_else(|error| panic!("{} example failed: {error}", operation.id.as_str()));
            assert_eq!(response.value["operation"], operation.id.as_str());
            assert!(response.value["title"].is_string());
            assert!(response.value["summary"].is_object());
            assert!(response.value["result"].is_object());
        }
    }

    #[test]
    fn invalid_input_returns_error() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("audio.separation.plan"),
            input: serde_json::json!({"input": ""}),
        })
        .unwrap_err();
        assert!(error.contains("input"));
    }
}
