use media_core::Result;

use crate::{invalid_request, setup_error, NativeDevicePreference};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResolvedNativeDevice {
    Cpu,
    Cuda(usize),
}

impl ResolvedNativeDevice {
    #[allow(dead_code)]
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

pub(crate) fn resolve_native_device(
    preference: NativeDevicePreference,
    cuda_device_index: usize,
) -> Result<ResolvedNativeDevice> {
    resolve_native_device_with(preference, cuda_device_index, cuda_device_available)
}

fn resolve_native_device_with(
    preference: NativeDevicePreference,
    cuda_device_index: usize,
    mut cuda_device_available: impl FnMut(usize) -> Result<()>,
) -> Result<ResolvedNativeDevice> {
    match preference {
        NativeDevicePreference::Cpu if cuda_device_index != 0 => Err(invalid_request(format!(
            "Candle Whisper cuda_device_index {cuda_device_index} cannot be used with device=cpu; use cuda_device_index=0 for CPU execution or select device=cuda"
        ))),
        NativeDevicePreference::Cpu => Ok(ResolvedNativeDevice::Cpu),
        NativeDevicePreference::Cuda => cuda_device_available(cuda_device_index)
            .map(|_| ResolvedNativeDevice::Cuda(cuda_device_index))
            .map_err(|error| {
                setup_error(format!(
                    "CUDA device cuda:{cuda_device_index} was requested but is unavailable: {error}"
                ))
            }),
        NativeDevicePreference::Auto => match cuda_device_available(cuda_device_index) {
            Ok(()) => Ok(ResolvedNativeDevice::Cuda(cuda_device_index)),
            Err(_) if cuda_device_index == 0 => Ok(ResolvedNativeDevice::Cpu),
            Err(error) => Err(setup_error(format!(
                "CUDA device cuda:{cuda_device_index} selected by cuda_device_index is unavailable: {error}; use cuda_device_index=0 to preserve automatic CPU fallback or select an available CUDA device"
            ))),
        },
    }
}

#[cfg(not(feature = "cuda"))]
fn cuda_device_available(index: usize) -> Result<()> {
    Err(setup_error(format!(
        "CUDA device cuda:{index} cannot be used because the binary lacks the `cuda` feature"
    )))
}

#[cfg(feature = "cuda")]
fn cuda_device_available(index: usize) -> Result<()> {
    candle_core::Device::new_cuda(index)
        .map(|_| ())
        .map_err(|error| setup_error(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_resolution_uses_the_requested_device_index() {
        let mut observed_index = None;

        let resolved = resolve_native_device_with(NativeDevicePreference::Cuda, 2, |index| {
            observed_index = Some(index);
            Ok(())
        })
        .unwrap();

        assert_eq!(observed_index, Some(2));
        assert_eq!(resolved, ResolvedNativeDevice::Cuda(2));
    }

    #[test]
    fn default_auto_device_keeps_cpu_fallback_for_cuda_index_zero() {
        let resolved = resolve_native_device_with(NativeDevicePreference::Auto, 0, |_| {
            Err(setup_error("test CUDA unavailable"))
        })
        .unwrap();

        assert_eq!(resolved, ResolvedNativeDevice::Cpu);
    }

    #[test]
    fn auto_device_does_not_silently_ignore_an_unavailable_nonzero_index() {
        let error = resolve_native_device_with(NativeDevicePreference::Auto, 3, |_| {
            Err(setup_error("test CUDA unavailable"))
        })
        .unwrap_err();

        assert!(error.to_string().contains("cuda:3"));
        assert!(error.to_string().contains("cuda_device_index=0"));
    }
}
