//! Runs the shared capture-metrics fixtures that every runtime must reproduce.

use audio_analysis_core::{capture_metrics, CaptureMetricsConfig};

const FIXTURES: &str = include_str!("fixtures/capture-metrics.json");

#[test]
fn shared_capture_metrics_fixtures() {
    let document: serde_json::Value = serde_json::from_str(FIXTURES).expect("fixture JSON");
    let tolerance = document["tolerance"].as_f64().expect("tolerance");
    let cases = document["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let samples = case["samples"]
            .as_array()
            .expect("samples")
            .iter()
            .map(|value| value.as_f64().expect("sample") as f32)
            .collect::<Vec<_>>();
        let config = case
            .get("options")
            .map(|options| {
                serde_json::from_value::<CaptureMetricsConfig>(options.clone()).expect("options")
            })
            .unwrap_or_default();
        let metrics = capture_metrics(
            &samples,
            case["sampleRate"].as_u64().expect("sampleRate") as u32,
            case["channels"].as_u64().expect("channels") as u16,
            &config,
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let actual = serde_json::to_value(&metrics).expect("serialize");
        for (key, expected) in case["expected"].as_object().expect("expected") {
            let expected = expected.as_f64().expect("numeric expectation");
            let value = actual[key]
                .as_f64()
                .unwrap_or_else(|| panic!("{name}: missing {key}"));
            assert!(
                (value - expected).abs() <= tolerance,
                "{name}: {key} = {value}, expected {expected}"
            );
        }
    }
}
