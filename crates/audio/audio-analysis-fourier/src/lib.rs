#![doc = include_str!("../README.md")]

pub mod surface;

pub use audio_analysis_core::spectral::{
    complex_spectral_difference, spectral_feature_frames, spectral_features, spectral_flux,
    spectrogram, zero_crossing_rate, FourierTransform, SpectralAnalyzer, SpectralFeatureFrame,
    SpectralFeatureOptions, SpectralFeatures, SpectrogramFrame, Spectrum, SpectrumBin, StftConfig,
};
