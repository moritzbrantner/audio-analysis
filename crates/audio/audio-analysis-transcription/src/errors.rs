//! Crate error constructors.

use crate::*;

pub(crate) fn setup_error(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("setup_error: {}", message.into()))
}

pub(crate) fn invalid_request(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("invalid_request: {}", message.into()))
}

pub(crate) fn model_output_mismatch(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("model_output_mismatch: {}", message.into()))
}

pub(crate) fn timeout_error(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("timeout: {}", message.into()))
}

pub(crate) fn unsupported_runtime(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("unsupported_runtime: {}", message.into()))
}
