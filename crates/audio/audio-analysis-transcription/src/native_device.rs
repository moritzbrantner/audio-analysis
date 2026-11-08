use media_core::Result;

use crate::{invalid_request, setup_error, NativeDevicePreference};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResolvedNativeDevice {
    Cpu,
    Cuda(usize),
    Metal(usize),
}

impl ResolvedNativeDevice {
    pub(crate) fn diagnostic_name(&self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Cuda(index) => format!("cuda:{index}"),
            Self::Metal(index) => format!("metal:{index}"),
        }
    }

    /// True only for CUDA; feeds the `cuda=` and `alignmentCuda=` diagnostics.
    #[allow(dead_code)]
    pub(crate) fn cuda_active(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// True for any GPU backend (CUDA or Metal).
    #[allow(dead_code)]
    pub(crate) fn gpu_active(&self) -> bool {
        self.gpu_backend().is_some()
    }

    /// Backend name of a GPU device, `None` on the CPU.
    #[allow(dead_code)]
    pub(crate) fn gpu_backend(&self) -> Option<&'static str> {
        match self {
            Self::Cpu => None,
            Self::Cuda(_) => Some("cuda"),
            Self::Metal(_) => Some("metal"),
        }
    }

    #[cfg(feature = "candle")]
    pub(crate) fn candle_device(&self) -> Result<candle_core::Device> {
        match self {
            Self::Cpu => Ok(candle_core::Device::Cpu),
            Self::Cuda(index) => {
                #[cfg(feature = "cuda")]
                {
                    candle_core::Device::new_cuda(*index).map_err(|error| {
                        setup_error(format!(
                            "resolved CUDA device cuda:{index} became unavailable: {error}"
                        ))
                    })
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(setup_error(format!(
                        "resolved CUDA device cuda:{index} cannot be created because the binary lacks the `cuda` feature"
                    )))
                }
            }
            Self::Metal(index) => {
                if !metal_compiled() {
                    return Err(setup_error(format!(
                        "resolved Metal device metal:{index} cannot be created because the binary lacks the `metal` feature"
                    )));
                }
                candle_core::Device::new_metal(*index).map_err(|error| {
                    setup_error(format!(
                        "resolved Metal device metal:{index} became unavailable: {error}"
                    ))
                })
            }
        }
    }
}

/// The device a stage runs on, plus what a report needs to know about it: the
/// requested preference, the GPU device name, and why `auto` fell back to the
/// CPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeDeviceSelection {
    pub(crate) requested: NativeDevicePreference,
    pub(crate) resolved: ResolvedNativeDevice,
    pub(crate) name: Option<String>,
    pub(crate) fallback_reason: Option<String>,
}

impl NativeDeviceSelection {
    /// Flat `key=value` diagnostics. Without a stage the keys are `device`,
    /// `deviceRequested`, `deviceName` and `deviceFallbackReason` (ASR); a
    /// stage such as `alignment` prefixes them (`alignmentDevice`, ...).
    pub(crate) fn diagnostics(&self, stage: Option<&str>) -> Vec<String> {
        let key = |suffix: &str| match stage {
            Some(stage) => format!("{stage}Device{suffix}"),
            None => format!("device{suffix}"),
        };
        let mut diagnostics = vec![
            format!("{}={}", key(""), self.resolved.diagnostic_name()),
            format!(
                "{}={}",
                key("Requested"),
                device_preference_name(self.requested)
            ),
        ];
        if let Some(name) = &self.name {
            diagnostics.push(format!("{}={name}", key("Name")));
        }
        if let Some(reason) = &self.fallback_reason {
            diagnostics.push(format!("{}={reason}", key("FallbackReason")));
        }
        diagnostics
    }
}

fn device_preference_name(preference: NativeDevicePreference) -> &'static str {
    match preference {
        NativeDevicePreference::Auto => "auto",
        NativeDevicePreference::Cpu => "cpu",
        NativeDevicePreference::Cuda => "cuda",
        NativeDevicePreference::Metal => "metal",
    }
}

pub(crate) fn resolve_native_device(
    preference: NativeDevicePreference,
    cuda_device_index: usize,
) -> Result<ResolvedNativeDevice> {
    select_native_device(preference, cuda_device_index).map(|selection| selection.resolved)
}

pub(crate) fn select_native_device(
    preference: NativeDevicePreference,
    cuda_device_index: usize,
) -> Result<NativeDeviceSelection> {
    select_native_device_with(
        preference,
        cuda_device_index,
        metal_compiled(),
        cuda_device_name,
        metal_device_name,
    )
}

/// `auto` probes CUDA first and, only when the binary was built with the
/// `metal` feature, Metal next. It falls back to the CPU only for GPU index
/// `0`. Builds without `metal` keep the CUDA-only `auto` behavior unchanged.
fn select_native_device_with(
    preference: NativeDevicePreference,
    cuda_device_index: usize,
    metal_compiled: bool,
    mut cuda_device_name: impl FnMut(usize) -> Result<Option<String>>,
    mut metal_device_name: impl FnMut(usize) -> Result<Option<String>>,
) -> Result<NativeDeviceSelection> {
    let selection = |resolved, name, fallback_reason| NativeDeviceSelection {
        requested: preference,
        resolved,
        name,
        fallback_reason,
    };
    match preference {
        NativeDevicePreference::Cpu if cuda_device_index != 0 => Err(invalid_request(format!(
            "Candle Whisper cuda_device_index {cuda_device_index} cannot be used with device=cpu; use cuda_device_index=0 for CPU execution or select device=cuda"
        ))),
        NativeDevicePreference::Cpu => Ok(selection(ResolvedNativeDevice::Cpu, None, None)),
        NativeDevicePreference::Cuda => cuda_device_name(cuda_device_index)
            .map(|name| selection(ResolvedNativeDevice::Cuda(cuda_device_index), name, None))
            .map_err(|error| {
                setup_error(format!(
                    "CUDA device cuda:{cuda_device_index} was requested but is unavailable: {error}"
                ))
            }),
        NativeDevicePreference::Metal => metal_device_name(cuda_device_index)
            .map(|name| selection(ResolvedNativeDevice::Metal(cuda_device_index), name, None))
            .map_err(|error| {
                setup_error(format!(
                    "Metal device metal:{cuda_device_index} was requested but is unavailable: {error}"
                ))
            }),
        NativeDevicePreference::Auto => {
            let cuda_error = match cuda_device_name(cuda_device_index) {
                Ok(name) => {
                    return Ok(selection(
                        ResolvedNativeDevice::Cuda(cuda_device_index),
                        name,
                        None,
                    ))
                }
                Err(error) => error.to_string(),
            };
            let unavailable = if metal_compiled {
                match metal_device_name(cuda_device_index) {
                    Ok(name) => {
                        return Ok(selection(
                            ResolvedNativeDevice::Metal(cuda_device_index),
                            name,
                            None,
                        ))
                    }
                    Err(error) => format!("{cuda_error}; {error}"),
                }
            } else {
                cuda_error
            };
            if cuda_device_index == 0 {
                Ok(selection(
                    ResolvedNativeDevice::Cpu,
                    None,
                    Some(unavailable),
                ))
            } else if metal_compiled {
                Err(setup_error(format!(
                    "GPU device {cuda_device_index} selected by cuda_device_index is unavailable: {unavailable}; use cuda_device_index=0 to preserve automatic CPU fallback or select an available GPU device"
                )))
            } else {
                Err(setup_error(format!(
                    "CUDA device cuda:{cuda_device_index} selected by cuda_device_index is unavailable: {unavailable}; use cuda_device_index=0 to preserve automatic CPU fallback or select an available CUDA device"
                )))
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn cuda_device_name(index: usize) -> Result<Option<String>> {
    Err(setup_error(format!(
        "CUDA device cuda:{index} cannot be used because the binary lacks the `cuda` feature"
    )))
}

/// Probes the CUDA device and returns its driver-reported name.
#[cfg(feature = "cuda")]
fn cuda_device_name(index: usize) -> Result<Option<String>> {
    let device =
        candle_core::Device::new_cuda(index).map_err(|error| setup_error(error.to_string()))?;
    Ok(device
        .as_cuda_device()
        .ok()
        .and_then(|cuda| cuda.cuda_stream().context().name().ok()))
}

/// Whether Candle was built with Metal. This follows Candle's own `metal`
/// feature rather than this crate's, so a consumer that enables
/// `candle-core/metal` directly in the same build also gets Metal here; this
/// crate's `metal` feature only forwards to the Candle features.
#[cfg(feature = "candle")]
fn metal_compiled() -> bool {
    candle_core::utils::metal_is_available()
}

#[cfg(not(feature = "candle"))]
fn metal_compiled() -> bool {
    false
}

/// Probes the Metal device. A successful probe reports no device name.
fn metal_device_name(index: usize) -> Result<Option<String>> {
    if !metal_compiled() {
        return Err(setup_error(format!(
            "Metal device metal:{index} cannot be used because the binary lacks the `metal` feature"
        )));
    }
    #[cfg(feature = "candle")]
    candle_core::Device::new_metal(index).map_err(|error| setup_error(error.to_string()))?;
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_cuda(_: usize) -> Result<Option<String>> {
        Err(setup_error("test CUDA unavailable"))
    }

    fn no_metal(_: usize) -> Result<Option<String>> {
        Err(setup_error("test Metal unavailable"))
    }

    fn metal_ok(_: usize) -> Result<Option<String>> {
        Ok(None)
    }

    #[test]
    fn cuda_resolution_uses_the_requested_device_index() {
        let mut observed_index = None;

        let selection = select_native_device_with(
            NativeDevicePreference::Cuda,
            2,
            true,
            |index| {
                observed_index = Some(index);
                Ok(Some("Test GPU".to_string()))
            },
            |_| panic!("explicit CUDA must not probe Metal"),
        )
        .unwrap();

        assert_eq!(observed_index, Some(2));
        assert_eq!(selection.resolved, ResolvedNativeDevice::Cuda(2));
        assert_eq!(
            selection.diagnostics(None),
            [
                "device=cuda:2",
                "deviceRequested=cuda",
                "deviceName=Test GPU"
            ]
        );
        assert_eq!(
            selection.diagnostics(Some("alignment")),
            [
                "alignmentDevice=cuda:2",
                "alignmentDeviceRequested=cuda",
                "alignmentDeviceName=Test GPU"
            ]
        );
    }

    #[test]
    fn default_auto_device_keeps_cpu_fallback_for_cuda_index_zero() {
        let selection =
            select_native_device_with(NativeDevicePreference::Auto, 0, false, no_cuda, no_metal)
                .unwrap();

        assert_eq!(selection.resolved, ResolvedNativeDevice::Cpu);
        let diagnostics = selection.diagnostics(None);
        assert_eq!(diagnostics[..2], ["device=cpu", "deviceRequested=auto"]);
        assert_eq!(diagnostics.len(), 3);
        assert!(diagnostics[2].starts_with("deviceFallbackReason="));
        assert!(diagnostics[2].contains("test CUDA unavailable"));
    }

    #[test]
    fn explicit_cpu_reports_no_name_or_fallback() {
        let selection = select_native_device_with(
            NativeDevicePreference::Cpu,
            0,
            true,
            |_| panic!("explicit CPU must not probe CUDA"),
            |_| panic!("explicit CPU must not probe Metal"),
        )
        .unwrap();

        assert_eq!(
            selection.diagnostics(None),
            ["device=cpu", "deviceRequested=cpu"]
        );
    }

    #[test]
    fn explicit_metal_resolves_the_metal_device() {
        let selection = select_native_device_with(
            NativeDevicePreference::Metal,
            0,
            true,
            |_| panic!("explicit Metal must not probe CUDA"),
            metal_ok,
        )
        .unwrap();

        assert_eq!(selection.resolved, ResolvedNativeDevice::Metal(0));
        assert!(selection.resolved.gpu_active());
        assert!(!selection.resolved.cuda_active());
        assert_eq!(
            selection.diagnostics(None),
            ["device=metal:0", "deviceRequested=metal"]
        );
    }

    #[test]
    fn explicit_metal_fails_instead_of_falling_back() {
        let error = select_native_device_with(
            NativeDevicePreference::Metal,
            0,
            true,
            |_| panic!("explicit Metal must not probe CUDA"),
            no_metal,
        )
        .unwrap_err();

        assert!(error.to_string().contains("metal:0 was requested"));
        assert!(error.to_string().contains("test Metal unavailable"));
    }

    #[test]
    fn auto_picks_metal_when_built_with_metal_and_cuda_is_unavailable() {
        let selection =
            select_native_device_with(NativeDevicePreference::Auto, 0, true, no_cuda, metal_ok)
                .unwrap();

        assert_eq!(selection.resolved, ResolvedNativeDevice::Metal(0));
        assert_eq!(selection.fallback_reason, None);
        assert_eq!(
            selection.diagnostics(None),
            ["device=metal:0", "deviceRequested=auto"]
        );
    }

    #[test]
    fn auto_prefers_cuda_over_metal() {
        let selection = select_native_device_with(
            NativeDevicePreference::Auto,
            0,
            true,
            |_| Ok(Some("Test GPU".to_string())),
            |_| panic!("auto must not probe Metal after CUDA succeeded"),
        )
        .unwrap();

        assert_eq!(selection.resolved, ResolvedNativeDevice::Cuda(0));
    }

    #[test]
    fn auto_without_metal_feature_never_probes_metal() {
        let selection =
            select_native_device_with(NativeDevicePreference::Auto, 0, false, no_cuda, |_| {
                panic!("auto must not probe Metal in a build without it")
            })
            .unwrap();

        assert_eq!(selection.resolved, ResolvedNativeDevice::Cpu);
    }

    #[test]
    fn auto_falls_back_to_cpu_with_both_gpu_reasons() {
        let selection =
            select_native_device_with(NativeDevicePreference::Auto, 0, true, no_cuda, no_metal)
                .unwrap();

        assert_eq!(selection.resolved, ResolvedNativeDevice::Cpu);
        let reason = selection.fallback_reason.unwrap();
        assert!(reason.contains("test CUDA unavailable"));
        assert!(reason.contains("test Metal unavailable"));
    }

    #[test]
    fn metal_request_without_metal_feature_names_the_feature() {
        if metal_compiled() {
            return;
        }
        let error = select_native_device(NativeDevicePreference::Metal, 0).unwrap_err();

        assert!(error.to_string().contains("`metal` feature"));
    }

    /// Needs a CUDA GPU: `cargo test --features cuda -- --ignored cuda_probe`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn cuda_probe_reports_the_device_name() {
        let selection = select_native_device(NativeDevicePreference::Auto, 0).unwrap();

        assert_eq!(selection.resolved, ResolvedNativeDevice::Cuda(0));
        assert!(selection.name.is_some_and(|name| !name.is_empty()));
    }

    /// Needs Apple Silicon: `cargo test --features metal -- --ignored metal_probe`.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore]
    fn metal_probe_resolves_metal() {
        let selection = select_native_device(NativeDevicePreference::Metal, 0).unwrap();

        assert_eq!(selection.resolved, ResolvedNativeDevice::Metal(0));
    }

    #[test]
    fn auto_device_does_not_silently_ignore_an_unavailable_nonzero_index() {
        let error =
            select_native_device_with(NativeDevicePreference::Auto, 3, false, no_cuda, no_metal)
                .unwrap_err();

        assert!(error.to_string().contains("cuda:3"));
        assert!(error.to_string().contains("cuda_device_index=0"));
    }
}
