use std::ffi::OsString;
use std::path::PathBuf;

#[cfg(feature = "external-tests")]
use audio_analysis_separation::is_demucs_available;
use audio_analysis_separation::{
    DemucsModel, HtdemucsOptions, HtdemucsSeparator, SeparationOutputFormat, Stem, StemLayout,
};
use tempfile::tempdir;

fn args_as_strings(args: Vec<OsString>) -> Vec<String> {
    args.into_iter()
        .map(|arg| arg.to_string_lossy().to_string())
        .collect()
}

#[test]
fn stem_names_and_file_names_are_stable() {
    let stems = [
        (Stem::Vocals, "vocals", "vocals.wav"),
        (Stem::Drums, "drums", "drums.wav"),
        (Stem::Bass, "bass", "bass.wav"),
        (Stem::Other, "other", "other.wav"),
        (Stem::Guitar, "guitar", "guitar.wav"),
        (Stem::Piano, "piano", "piano.wav"),
        (Stem::NoVocals, "no_vocals", "no_vocals.wav"),
        (
            Stem::Custom("karaoke".to_string()),
            "karaoke",
            "karaoke.wav",
        ),
    ];
    for (stem, name, file_name) in stems {
        assert_eq!(stem.as_str(), name);
        assert_eq!(stem.file_name(&SeparationOutputFormat::Wav), file_name);
    }
}

#[test]
fn demucs_models_derive_expected_default_layouts() {
    assert_eq!(DemucsModel::Htdemucs.default_layout(), StemLayout::FourStem);
    assert_eq!(DemucsModel::MdXExtra.default_layout(), StemLayout::FourStem);
    assert_eq!(
        DemucsModel::Htdemucs6s.default_layout(),
        StemLayout::SixStem
    );
}

#[test]
fn options_validation_rejects_invalid_values() {
    assert!(HtdemucsOptions::new("out").command("").validate().is_err());
    assert!(HtdemucsOptions::new("out").model(" ").validate().is_err());
    assert!(HtdemucsOptions::new("out")
        .overlap(-0.1)
        .validate()
        .is_err());
    assert!(HtdemucsOptions::new("out").overlap(1.0).validate().is_err());
    assert!(HtdemucsOptions::new("out").jobs(0).validate().is_err());
    assert!(HtdemucsOptions::new("out").segment(0).validate().is_err());
    assert!(HtdemucsOptions::new("out")
        .sample_rate(0)
        .validate()
        .is_err());
    assert!(HtdemucsOptions::new("out")
        .layout(StemLayout::Custom(Vec::new()))
        .validate()
        .is_err());
}

#[test]
fn builds_default_four_stem_command() {
    let separator = HtdemucsSeparator::new(HtdemucsOptions::new("out")).unwrap();
    let command = separator.build_command("song.wav").unwrap();
    assert_eq!(command.program, PathBuf::from("demucs"));
    assert_eq!(
        args_as_strings(command.args),
        vec!["-n", "htdemucs", "-o", "out", "song.wav"]
    );
}

#[test]
fn builds_htdemucs_command_arguments_with_options() {
    let separator = HtdemucsSeparator::new(
        HtdemucsOptions::new("out")
            .command_arg("python")
            .command_arg("-m")
            .model(DemucsModel::Htdemucs6s)
            .two_stems(Stem::Vocals)
            .device("cpu")
            .shifts(2)
            .overlap(0.25)
            .jobs(3)
            .segment(12)
            .sample_rate(44_100)
            .output_format(SeparationOutputFormat::Flac)
            .filename("{track}/{stem}.{ext}")
            .extra_arg("--jobs")
            .extra_arg("1"),
    )
    .unwrap();
    let args = args_as_strings(separator.build_args("song.wav").unwrap());
    assert_eq!(
        args,
        vec![
            "python",
            "-m",
            "-n",
            "htdemucs_6s",
            "-o",
            "out",
            "--flac",
            "--filename",
            "{track}/{stem}.{ext}",
            "--two-stems",
            "vocals",
            "--device",
            "cpu",
            "--shifts",
            "2",
            "--overlap",
            "0.25",
            "-j",
            "3",
            "--segment",
            "12",
            "--samplerate",
            "44100",
            "--jobs",
            "1",
            "song.wav"
        ]
    );
}

#[test]
fn predicts_standard_htdemucs_stem_paths() {
    let separator = HtdemucsSeparator::new(HtdemucsOptions::new("out")).unwrap();
    let result = separator.discover_result("/tmp/song.mp3").unwrap();
    assert_eq!(result.output_dir, PathBuf::from("out/htdemucs/song"));
    assert_eq!(result.layout, StemLayout::FourStem);
    assert_eq!(
        result
            .stems
            .iter()
            .map(|stem| stem.stem.as_str())
            .collect::<Vec<_>>(),
        vec!["vocals", "drums", "bass", "other"]
    );
    assert_eq!(
        result.stems[0].path,
        PathBuf::from("out/htdemucs/song/vocals.wav")
    );
    assert!(!result.all_outputs_present);
}

#[test]
fn predicts_six_stem_and_two_stem_outputs() {
    let six =
        HtdemucsSeparator::new(HtdemucsOptions::new("out").model(DemucsModel::Htdemucs6s)).unwrap();
    assert_eq!(
        six.expected_stems()
            .iter()
            .map(Stem::as_str)
            .collect::<Vec<_>>(),
        vec!["vocals", "drums", "bass", "other", "guitar", "piano"]
    );

    let two = HtdemucsSeparator::new(HtdemucsOptions::new("out").two_stems(Stem::Drums)).unwrap();
    assert_eq!(
        two.expected_layout(),
        StemLayout::TwoStem {
            primary: Stem::Drums,
            residual: Stem::Custom("no_drums".to_string()),
        }
    );
}

#[test]
fn discovers_present_and_missing_outputs_with_bytes() {
    let dir = tempdir().unwrap();
    let output_dir = dir.path().join("out");
    let model_dir = output_dir.join("htdemucs");
    let stem_dir = model_dir.join("song");
    std::fs::create_dir_all(&stem_dir).unwrap();
    std::fs::write(stem_dir.join("vocals.wav"), [1_u8, 2, 3]).unwrap();
    std::fs::write(stem_dir.join("drums.wav"), []).unwrap();

    let separator = HtdemucsSeparator::new(HtdemucsOptions::new(&output_dir)).unwrap();
    let result = separator
        .discover_result(dir.path().join("song.wav"))
        .unwrap();
    assert_eq!(
        result.missing_stems,
        vec![Stem::Drums, Stem::Bass, Stem::Other]
    );
    assert!(!result.all_outputs_present);
    let vocals = result
        .stems
        .iter()
        .find(|stem| stem.stem == Stem::Vocals)
        .unwrap();
    assert_eq!(vocals.bytes, Some(3));
    let drums = result
        .stems
        .iter()
        .find(|stem| stem.stem == Stem::Drums)
        .unwrap();
    assert!(!drums.exists);
    assert_eq!(drums.bytes, None);
}

#[test]
fn custom_filename_template_changes_discovery_paths() {
    let separator = HtdemucsSeparator::new(
        HtdemucsOptions::new("out")
            .output_format(SeparationOutputFormat::Mp3)
            .filename("{track}/{stem}.{ext}"),
    )
    .unwrap();
    let result = separator.discover_result("/tmp/song.wav").unwrap();
    assert_eq!(result.output_dir, PathBuf::from("out/htdemucs"));
    assert_eq!(
        result.stems[0].path,
        PathBuf::from("out/htdemucs/song/vocals.mp3")
    );
}

#[cfg(feature = "external-tests")]
#[test]
#[ignore]
fn real_demucs_smoke_test_when_requested() {
    if std::env::var_os("RUN_REAL_DEMUCS_TESTS").is_none() {
        return;
    }
    if !is_demucs_available() {
        panic!("RUN_REAL_DEMUCS_TESTS=1 but demucs is unavailable");
    }

    let dir = tempdir().unwrap();
    let input = dir.path().join("input.wav");
    std::fs::write(&input, [0_u8; 44]).unwrap();
    let separator = HtdemucsSeparator::new(HtdemucsOptions::new(dir.path().join("out"))).unwrap();
    let _ = separator.dry_run(&input).unwrap();
}
