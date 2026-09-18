use audio_analysis_core::spectral as core;
use audio_analysis_fourier as compatibility;

#[test]
fn fourier_package_reexports_core_spectral_contract() {
    let core_transform = core::FourierTransform::new(512).expect("core transform");
    let compatibility_transform =
        compatibility::FourierTransform::new(512).expect("compatibility transform");
    assert_eq!(compatibility_transform, core_transform);

    let config = compatibility::StftConfig::new(512, 256).expect("STFT config");
    let frames =
        compatibility::spectrogram(&vec![0.0; 512], 48_000, &config).expect("spectrogram");
    assert_eq!(frames.len(), 1);

    let novelty = compatibility::surface::complex_spectral_difference(
        &vec![0.0; 512],
        48_000,
        512,
        256,
    )
    .expect("legacy surface novelty");
    assert_eq!(novelty.len(), 2);
}
