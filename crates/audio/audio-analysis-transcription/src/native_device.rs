use media_core::Result;

use crate::{invalid_request, setup_error, NativeDevicePreference};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResolvedNativeDevice {
    Cpu,
    Cuda(usize),
}

impl ResolvedNativeDevice {
    pub(crate) fn diagnostic_name(&self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Cuda(index) => format!("cuda:{index}"),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn cuda_active(&self) -> bool {
        match self {
            Self::Cpu => false,
            Self::Cuda(_) => true,
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
        }
    }
}

/// The device a stage runs on, plus what a report needs to know about it: the
/// requested preference, the CUDA device name, and why `auto` fell back to the
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
    select_native_device_with(preference, cuda_device_index, cuda_device_name)
}

fn select_native_device_with(
    preference: NativeDevicePreference,
    cuda_device_index: usize,
    mut cuda_device_name: impl FnMut(usize) -> Result<Option<String>>,
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
        NativeDevicePreference::Auto => match cuda_device_name(cuda_device_index) {
            Ok(name) => Ok(selection(
                ResolvedNativeDevice::Cuda(cuda_device_index),
                name,
                None,
            )),
            Err(error) if cuda_device_index == 0 => Ok(selection(
                ResolvedNativeDevice::Cpu,
                None,
                Some(error.to_string()),
            )),
            Err(error) => Err(setup_error(format!(
                "CUDA device cuda:{cuda_device_index} selected by cuda_device_index is unavailable: {error}; use cuda_device_index=0 to preserve automatic CPU fallback or select an available CUDA device"
            ))),
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_resolution_uses_the_requested_device_index() {
        let mut observed_index = None;

        let selection = select_native_device_with(NativeDevicePreference::Cuda, 2, |index| {
            observed_index = Some(index);
            Ok(Some("Test GPU".to_string()))
        })
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
        let selection = select_native_device_with(NativeDevicePreference::Auto, 0, |_| {
            Err(setup_error("test CUDA unavailable"))
        })
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
        let selection = select_native_device_with(NativeDevicePreference::Cpu, 0, |_| {
            panic!("explicit CPU must not probe CUDA")
        })
        .unwrap();

        assert_eq!(
            selection.diagnostics(None),
            ["device=cpu", "deviceRequested=cpu"]
        );
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

    #[test]
    fn auto_device_does_not_silently_ignore_an_unavailable_nonzero_index() {
        let error = select_native_device_with(NativeDevicePreference::Auto, 3, |_| {
            Err(setup_error("test CUDA unavailable"))
        })
        .unwrap_err();

        assert!(error.to_string().contains("cuda:3"));
        assert!(error.to_string().contains("cuda_device_index=0"));
    }
}
