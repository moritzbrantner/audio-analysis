#![cfg(feature = "pyannote-diarization")]

use std::path::PathBuf;

use audio_analysis_speakers::{
    PyannoteCommunityDiarizationConfig, PyannoteCommunityDiarizer, SpeakerAudio,
    SpeakerSegmentPrediction,
};

const UPSTREAM_TURNS: &[(f64, f64, &str)] = &[
    (0.030_969, 2.123_469, "SPEAKER_01"),
    (2.123_469, 2.140_344, "SPEAKER_00"),
    (2.663_469, 3.405_969, "SPEAKER_01"),
    (4.840_344, 8.400_969, "SPEAKER_01"),
    (8.789_094, 10.004_094, "SPEAKER_01"),
    (10.847_844, 16.399_719, "SPEAKER_00"),
    (11.387_844, 12.164_094, "SPEAKER_01"),
    (17.175_969, 18.036_594, "SPEAKER_00"),
    (18.070_344, 18.745_344, "SPEAKER_00"),
    (20.280_969, 20.567_844, "SPEAKER_01"),
    (20.939_094, 23.132_844, "SPEAKER_01"),
    (23.639_094, 24.381_594, "SPEAKER_01"),
    (25.832_844, 29.376_594, "SPEAKER_01"),
    (29.781_594, 30.996_594, "SPEAKER_01"),
    (31.840_344, 33.359_094, "SPEAKER_00"),
    (32.397_219, 33.156_594, "SPEAKER_01"),
    (33.359_094, 33.865_344, "SPEAKER_01"),
    (33.865_344, 37.442_844, "SPEAKER_00"),
    (38.185_344, 39.737_844, "SPEAKER_00"),
    (41.239_719, 41.492_844, "SPEAKER_00"),
];

fn active_reference(time: f64) -> [bool; 2] {
    let mut active = [false; 2];
    for (start, end, speaker) in UPSTREAM_TURNS {
        if time >= *start && time < *end {
            active[usize::from(*speaker == "SPEAKER_01")] = true;
        }
    }
    active
}

fn active_native(turns: &[SpeakerSegmentPrediction], time: f64, swapped: bool) -> [bool; 2] {
    let mut active = [false; 2];
    for turn in turns {
        if time >= f64::from(turn.start_seconds) && time < f64::from(turn.end_seconds) {
            let mut index = usize::from(turn.speaker == "SPEAKER_01");
            if swapped {
                index = 1 - index;
            }
            active[index] = true;
        }
    }
    active
}

fn permutation_invariant_disagreement(turns: &[SpeakerSegmentPrediction]) -> f64 {
    const STEP_SECONDS: f64 = 0.005;
    let mut disagreement = [0.0_f64; 2];
    let mut time = 0.0;
    while time < 41.5 {
        let expected = active_reference(time);
        for (swapped, total) in disagreement.iter_mut().enumerate() {
            if active_native(turns, time, swapped == 1) != expected {
                *total += STEP_SECONDS;
            }
        }
        time += STEP_SECONDS;
    }
    disagreement.into_iter().fold(f64::INFINITY, f64::min)
}

fn read_mono_wav(path: &PathBuf) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("open retained two-speaker WAV");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "retained fixture must be mono");
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| sample.expect("read float WAV sample"))
            .collect(),
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| f32::from(sample.expect("read PCM WAV sample")) / 32_768.0)
            .collect(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|sample| {
                sample.expect("read PCM WAV sample") as f32
                    / 2_f32.powi(i32::from(spec.bits_per_sample) - 1)
            })
            .collect(),
    };
    (samples, spec.sample_rate)
}

fn config(bundle_path: PathBuf) -> PyannoteCommunityDiarizationConfig {
    PyannoteCommunityDiarizationConfig {
        bundle_path,
        manifest_file: None,
        segmentation_model_file: None,
        embedding_model_file: None,
        plda_transform_file: None,
        plda_model_file: None,
        clustering_config_file: None,
        min_speakers: Some(2),
        max_speakers: Some(2),
        return_speaker_embeddings: true,
    }
}

fn custom_filename_config(bundle_path: PathBuf) -> PyannoteCommunityDiarizationConfig {
    PyannoteCommunityDiarizationConfig {
        bundle_path,
        manifest_file: Some("community-manifest.json".to_string()),
        segmentation_model_file: Some("community-segmentation.onnx".to_string()),
        embedding_model_file: Some("community-embedding.onnx".to_string()),
        plda_transform_file: Some("community-plda-transform.json".to_string()),
        plda_model_file: Some("community-plda-model.json".to_string()),
        clustering_config_file: Some("community-clustering.json".to_string()),
        min_speakers: Some(2),
        max_speakers: Some(2),
        return_speaker_embeddings: true,
    }
}

#[test]
#[ignore = "requires the caller-owned approved bundle and retained two-speaker WAV"]
fn approved_bundle_matches_retained_upstream_two_speaker_evidence() {
    let bundle_path = std::env::var_os("PYANNOTE_COMMUNITY_BUNDLE")
        .map(PathBuf::from)
        .expect("PYANNOTE_COMMUNITY_BUNDLE must name the approved local snapshot");
    let audio_path = std::env::var_os("PYANNOTE_TWO_SPEAKER_WAV")
        .map(PathBuf::from)
        .expect("PYANNOTE_TWO_SPEAKER_WAV must name the retained fixture");
    let (samples, sample_rate) = read_mono_wav(&audio_path);
    let audio = SpeakerAudio::mono(&samples, sample_rate).expect("valid retained fixture");
    let mut diarizer = PyannoteCommunityDiarizer::from_config(config(bundle_path))
        .expect("approved bundle must pass offline and ONNX contract validation");
    let result = diarizer
        .diarize(&audio)
        .expect("native pyannote community diarization");
    let speakers = result
        .response
        .segments
        .iter()
        .map(|segment| segment.speaker.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(speakers.len(), 2);
    assert!(!result.response.segments.is_empty());
    assert!(result.response.segments.iter().all(|segment| {
        segment.end_seconds > segment.start_seconds
            && segment
                .score
                .is_some_and(|score| score.is_finite() && (0.0..=1.0).contains(&score))
    }));
    let disagreement = permutation_invariant_disagreement(&result.response.segments);
    assert!(
        disagreement <= 0.35,
        "native/upstream frame disagreement {disagreement:.3}s exceeds retained 0.35s tolerance"
    );
    eprintln!("retainedFrameDisagreementSeconds={disagreement:.3}");
    let embeddings = result
        .response
        .speaker_embeddings
        .as_ref()
        .expect("requested speaker embeddings");
    assert_eq!(embeddings.len(), 2);
    assert!(result.diagnostics.iter().any(|item| {
        item == "pyannoteArtifactSetSha256=0a12189874dace9b590af9b09ef4637552006130716db57c35d72f984b36c577"
    }));
    assert!(result
        .diagnostics
        .iter()
        .all(|item| !item.contains(audio_path.to_string_lossy().as_ref())));
}

#[cfg(unix)]
#[test]
#[ignore = "requires the caller-owned approved bundle; creates only a temporary symlink staging directory"]
fn partial_and_mutated_plda_vbx_artifacts_fail_before_inference() {
    use std::{
        fs,
        os::unix::fs::symlink,
        time::{SystemTime, UNIX_EPOCH},
    };

    const FILES: [&str; 8] = [
        "pyannote_diarization_manifest.json",
        "segmentation.onnx",
        "embedding.onnx",
        "plda_transform.json",
        "plda_model.json",
        "clustering.json",
        "MODEL_PROVENANCE.md",
        "LICENSE.md",
    ];
    let bundle = std::env::var_os("PYANNOTE_COMMUNITY_BUNDLE")
        .map(PathBuf::from)
        .expect("PYANNOTE_COMMUNITY_BUNDLE must name the approved local snapshot");
    let base = std::env::var_os("PYANNOTE_TEST_TMPDIR")
        .or_else(|| std::env::var_os("CARGO_TARGET_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let staging = base.join(format!("pyannote-controlled-fault-{unique}"));
    fs::create_dir_all(&staging).expect("create controlled-fault staging directory");
    for file in FILES {
        symlink(bundle.join(file), staging.join(file)).expect("stage approved artifact");
    }

    let custom_staging = staging.join("custom-filenames-only");
    fs::create_dir(&custom_staging).expect("create custom-filename staging directory");
    let custom_names = [
        (
            "pyannote_diarization_manifest.json",
            "community-manifest.json",
        ),
        ("segmentation.onnx", "community-segmentation.onnx"),
        ("embedding.onnx", "community-embedding.onnx"),
        ("plda_transform.json", "community-plda-transform.json"),
        ("plda_model.json", "community-plda-model.json"),
        ("clustering.json", "community-clustering.json"),
        ("MODEL_PROVENANCE.md", "MODEL_PROVENANCE.md"),
        ("LICENSE.md", "LICENSE.md"),
    ];
    for (source, custom) in custom_names {
        symlink(bundle.join(source), custom_staging.join(custom))
            .expect("stage custom-named artifact");
    }
    PyannoteCommunityDiarizer::from_config(custom_filename_config(custom_staging))
        .expect("custom artifact filenames must preserve offline integrity validation");

    for file in ["plda_transform.json", "plda_model.json", "clustering.json"] {
        fs::remove_file(staging.join(file)).expect("remove staged symlink");
        fs::copy(bundle.join(file), staging.join(file)).expect("copy controlled artifact");
        fs::OpenOptions::new()
            .append(true)
            .open(staging.join(file))
            .and_then(|mut output| std::io::Write::write_all(&mut output, b"\n "))
            .expect("mutate controlled artifact");
        let error = PyannoteCommunityDiarizer::from_config(config(staging.clone()))
            .expect_err("mutated checksummed artifact must fail offline");
        assert!(error.to_string().contains("checksum mismatch"), "{error}");
        fs::remove_file(staging.join(file)).expect("remove mutated artifact");
        symlink(bundle.join(file), staging.join(file)).expect("restore approved artifact");
    }

    fs::remove_file(staging.join("clustering.json")).expect("remove required artifact");
    let error = PyannoteCommunityDiarizer::from_config(config(staging.clone()))
        .expect_err("partial bundle must fail offline");
    assert!(error.to_string().contains("does not exist"), "{error}");

    let hidden_path = staging.join("private-caller-path");
    let error = PyannoteCommunityDiarizer::from_config(config(hidden_path.clone()))
        .expect_err("missing caller bundle must fail before runtime");
    assert!(
        !error
            .to_string()
            .contains(hidden_path.to_string_lossy().as_ref()),
        "setup error leaked caller bundle path: {error}"
    );
    fs::remove_dir_all(staging).expect("remove controlled-fault staging directory");
}
