use audio_analysis_core::spectral as core;
use audio_analysis_fourier as compatibility;

#[test]
fn fourier_package_reexports_core_spectral_contract() {
    let core_transform = core::FourierTransform::new(512).expect("core transform");
    let compatibility_transform =
        compatibility::FourierTransform::new(512).expect("compatibility transform");
    assert_eq!(compatibility_transform, core_transform);

    let config = compatibility::StftConfig::new(512, 256).expect("STFT config");
    let frames = compatibility::spectrogram(&vec![0.0; 512], 48_000, &config).expect("spectrogram");
    assert_eq!(frames.len(), 1);

    let novelty =
        compatibility::surface::complex_spectral_difference(&vec![0.0; 512], 48_000, 512, 256)
            .expect("legacy surface novelty");
    assert_eq!(novelty.len(), 2);
}

#[test]
fn legacy_paths_preserve_spectral_values_and_validation() {
    let samples = (0..1_300)
        .map(|index| (std::f32::consts::TAU * 440.0 * index as f32 / 8_192.0).sin())
        .collect::<Vec<_>>();
    let config = compatibility::StftConfig::new(256, 128)
        .expect("STFT config")
        .pad_final_frame(true);
    let legacy = compatibility::spectrogram(&samples, 8_192, &config).expect("legacy frames");
    let current = core::spectrogram(&samples, 8_192, &config).expect("core frames");
    assert_eq!(legacy, current);

    // A fresh single-frame FFT is the simple reference for the reused STFT state,
    // including zero padding after a sequence of non-silent frames.
    let reference =
        core::FourierTransform::with_window(config.fft_size, config.window).expect("reference FFT");
    for frame in &legacy {
        let end = (frame.start_sample + config.fft_size).min(samples.len());
        let expected = reference
            .analyze_samples(&samples[frame.start_sample..end], 8_192)
            .expect("reference spectrum");
        for (actual, expected) in frame.spectrum.bins.iter().zip(&expected.bins) {
            assert!((actual.magnitude - expected.magnitude).abs() < 1.0e-6);
            assert!((actual.power - expected.power).abs() < 1.0e-6);
        }
    }

    let legacy_novelty =
        compatibility::surface::complex_spectral_difference(&samples, 8_192, 256, 128)
            .expect("legacy novelty");
    assert_eq!(
        legacy_novelty,
        core::complex_spectral_difference(&samples, 8_192, 256, 128).expect("core novelty")
    );
    assert!(legacy_novelty.iter().skip(1).any(|value| *value > 0.0));
    for (sample_rate, fft_size, hop_size) in [(0, 256, 128), (8_192, 255, 128), (8_192, 256, 0)] {
        let legacy_error = compatibility::surface::complex_spectral_difference(
            &samples,
            sample_rate,
            fft_size,
            hop_size,
        )
        .expect_err("legacy validation");
        let current_error =
            core::complex_spectral_difference(&samples, sample_rate, fft_size, hop_size)
                .expect_err("core validation");
        assert_eq!(legacy_error.to_string(), current_error.to_string());
    }
}
