//! Build script support for generated or native crate artifacts.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if env::var_os("CARGO_FEATURE_NATIVE").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let vendor_dir = env::var_os("WHISPER_CPP_SOURCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            manifest_dir
                .ancestors()
                .nth(3)
                .expect("workspace root")
                .join("vendor/whisper.cpp")
        });
    if !vendor_dir.join("CMakeLists.txt").is_file() {
        panic!(
            "audio-analysis-transcription native builds require a whisper.cpp source checkout; \
             set WHISPER_CPP_SOURCE_DIR or build from this repository with vendor/whisper.cpp present"
        );
    }

    println!("cargo:rerun-if-changed={}", vendor_dir.display());

    let mut config = cmake::Config::new(&vendor_dir);
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_SERVER", "OFF")
        .define("GGML_BACKEND_DL", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define(
            "GGML_METAL",
            if cfg!(target_os = "macos") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define(
            "GGML_METAL_EMBED_LIBRARY",
            if cfg!(target_os = "macos") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define("WHISPER_COREML", "OFF")
        .define("WHISPER_OPENVINO", "OFF");

    let install_dir = config.build();
    let lib_dir = install_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    for lib in static_library_names(&lib_dir) {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn static_library_names(lib_dir: &Path) -> Vec<String> {
    let mut names = fs::read_dir(lib_dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|file_name| {
            let name = file_name.strip_prefix("lib")?.strip_suffix(".a")?;
            Some(name.to_string())
        })
        .collect::<Vec<_>>();
    names.sort();
    names
}
