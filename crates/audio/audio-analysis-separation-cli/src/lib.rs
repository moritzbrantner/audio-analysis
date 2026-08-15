use runtime_core::{
    cli::{self, CliAdapterMetadata},
    PackageSurface, SurfaceResponse,
};

/// Wrapped library crate name.
pub const LIBRARY_CRATE: &str = "audio-analysis-separation";
/// Adapter surface kind.
pub const SURFACE_KIND: &str = "cli";
/// Rust import path for the wrapped crate.
pub const LIBRARY_IMPORT: &str = "use audio_analysis_separation";
/// Companion server package name.
pub const SERVER_PACKAGE: &str = "audio-analysis-separation-server";
/// Companion React app package name.
pub const APP_PACKAGE: &str = "audio-analysis-separation-app";
/// Companion WASM package name.
pub const WASM_PACKAGE: &str = "audio-analysis-separation-wasm";

const METADATA: CliAdapterMetadata = CliAdapterMetadata {
    library_crate: LIBRARY_CRATE,
    surface_kind: SURFACE_KIND,
    library_import: LIBRARY_IMPORT,
    server_package: SERVER_PACKAGE,
    app_package: APP_PACKAGE,
    wasm_package: WASM_PACKAGE,
};

pub fn package_surface() -> PackageSurface {
    audio_analysis_separation::surface::package_surface()
}

pub fn package_metadata_json() -> String {
    cli::package_metadata_json(METADATA, package_surface())
}

pub fn command_schema_json() -> String {
    cli::command_schema_json()
}

pub fn run_operation(operation: &str, input: serde_json::Value) -> Result<SurfaceResponse, String> {
    cli::run_wrapped_operation(
        operation,
        input,
        audio_analysis_separation::surface::run_surface_operation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_mentions_wrapped_library() {
        let metadata = package_metadata_json();
        assert!(metadata.contains(LIBRARY_CRATE));
        assert!(metadata.contains(SURFACE_KIND));
    }
}
