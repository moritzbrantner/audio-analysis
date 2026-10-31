//! Consumer-reported analysis failures (`tests/fixtures/consumer-failures.v1.json`).
//!
//! `open` cases must still reproduce their failure, `fixed` cases must pass.
//! When a fix makes an open case pass, this test fails until its status is
//! flipped to `fixed`, so the case then guards against regressions.

use std::path::Path;

use audio_analysis_rhythm::track::{analyze_rhythm_track, TrackRhythmConfig};
use serde_json::Value;
use sha2::{Digest, Sha256};

const DEFAULT_BPM_TOLERANCE: f64 = 0.02;
const DOWNBEAT_TOLERANCE_SECONDS: f64 = 0.07;

fn fixtures() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

fn load_excerpt(audio: &Value) -> (Vec<f32>, u32) {
    let path = fixtures().join(audio["path"].as_str().expect("audio.path"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest,
        audio["sha256"].as_str().expect("audio.sha256"),
        "{}",
        path.display()
    );
    let reader = hound::WavReader::new(bytes.as_slice()).expect("wav excerpt");
    let spec = reader.spec();
    assert_eq!(
        (spec.channels, spec.bits_per_sample),
        (1, 16),
        "mono 16-bit excerpt"
    );
    let samples = reader
        .into_samples::<i16>()
        .map(|s| s.expect("wav sample") as f32 / i16::MAX as f32)
        .collect();
    (samples, spec.sample_rate)
}

/// Returns the unmet expectations; empty means the analysis is correct.
fn misses(case: &Value) -> Vec<String> {
    let (samples, sample_rate) = load_excerpt(&case["audio"]);
    let analysis = analyze_rhythm_track(&samples, sample_rate, TrackRhythmConfig::default())
        .expect("rhythm analysis");
    let expected = &case["expected"];
    let mut misses = Vec::new();
    // `"bpm": null` expects the analysis to report no reliable tempo.
    if expected.get("bpm").is_some_and(Value::is_null) && analysis.bpm.is_some() {
        misses.push(format!("bpm {:?}, expected no tempo", analysis.bpm));
    }
    if let Some(bpm) = expected["bpm"].as_f64() {
        let tolerance = expected["bpmTolerance"]
            .as_f64()
            .unwrap_or(DEFAULT_BPM_TOLERANCE);
        match analysis.bpm {
            Some(actual) if (actual as f64 - bpm).abs() / bpm <= tolerance => {}
            actual => misses.push(format!("bpm {actual:?}, expected {bpm} ±{tolerance}")),
        }
    }
    if let Some(downbeat) = expected["firstDownbeatSeconds"].as_f64() {
        match analysis.downbeats.first() {
            Some(actual) if (actual - downbeat).abs() <= DOWNBEAT_TOLERANCE_SECONDS => {}
            actual => misses.push(format!("first downbeat {actual:?}, expected {downbeat}")),
        }
    }
    misses
}

#[test]
fn consumer_failures_match_their_status() {
    let manifest: Value =
        serde_json::from_str(include_str!("fixtures/consumer-failures.v1.json")).expect("manifest");
    assert_eq!(
        manifest["schemaVersion"],
        "audio-analysis-consumer-failures/v1"
    );
    let mut ids = std::collections::HashSet::new();
    for case in manifest["cases"].as_array().expect("cases") {
        let id = case["id"].as_str().expect("case id");
        assert!(ids.insert(id), "duplicate case {id}");
        for field in ["consumer", "scenario"] {
            assert!(case[field].is_string(), "{id}: {field}");
        }
        for field in ["consumerRevision", "audioAnalysisRevision", "entryPoint"] {
            assert!(
                case["provenance"][field].is_string(),
                "{id}: provenance.{field}"
            );
        }
        assert!(
            case["expected"].is_object() && case["observed"].is_object(),
            "{id}"
        );
        let status = case["status"].as_str().expect("status");
        match case["audio"]["kind"].as_str() {
            // Copyrighted or unshared audio: a record to reduce into a runnable case.
            Some("private") => assert_eq!(status, "open", "{id}: private cases cannot gate"),
            Some("excerpt") => {
                assert!(case["audio"]["license"].is_string(), "{id}: audio.license");
                let misses = misses(case);
                match status {
                    "open" => assert!(
                        !misses.is_empty(),
                        "{id} now passes; set its status to \"fixed\""
                    ),
                    "fixed" => assert!(misses.is_empty(), "{id} regressed: {misses:?}"),
                    other => panic!("{id}: unknown status {other}"),
                }
                println!("{id} [{status}]: {misses:?}");
            }
            other => panic!("{id}: unknown audio kind {other:?}"),
        }
    }
}
