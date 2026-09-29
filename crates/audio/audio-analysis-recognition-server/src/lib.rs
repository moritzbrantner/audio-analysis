use std::io;

use runtime_core::{
    server::{self, ServerAdapterMetadata},
    PackageSurface,
};

/// Wrapped library crate name.
pub const LIBRARY_CRATE: &str = "audio-analysis-recognition";
/// Adapter surface kind.
pub const SURFACE_KIND: &str = "api";
/// Rust import path for the wrapped crate.
pub const LIBRARY_IMPORT: &str = "use audio_analysis_recognition";
/// Companion CLI package name.
pub const CLI_PACKAGE: &str = "audio-analysis-recognition-cli";
/// Companion React app package name.
pub const APP_PACKAGE: &str = "audio-analysis-recognition-app";
/// Companion WASM package name.
pub const WASM_PACKAGE: &str = "audio-analysis-recognition-wasm";

pub type HttpResponse = server::HttpResponse;

const METADATA: ServerAdapterMetadata = ServerAdapterMetadata {
    library_crate: LIBRARY_CRATE,
    surface_kind: SURFACE_KIND,
    library_import: LIBRARY_IMPORT,
    cli_package: CLI_PACKAGE,
    app_package: APP_PACKAGE,
    wasm_package: WASM_PACKAGE,
};

pub fn package_surface() -> PackageSurface {
    audio_analysis_recognition::surface::package_surface()
}

pub fn serve(addr: &str) -> io::Result<()> {
    server::serve(
        addr,
        METADATA,
        package_surface,
        audio_analysis_recognition::surface::run_surface_operation,
    )
}

pub fn response_for(method: &str, path: &str, body: &str) -> HttpResponse {
    server::response_for(
        method,
        path,
        body,
        METADATA,
        package_surface,
        audio_analysis_recognition::surface::run_surface_operation,
    )
}

pub fn package_metadata_json() -> String {
    server::package_metadata_json(METADATA, package_surface())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_endpoint_reports_package() {
        let response = response_for("GET", "/health", "");
        assert_eq!(response.status_code, 200);
        assert!(response.body.contains(LIBRARY_CRATE));
    }
}
