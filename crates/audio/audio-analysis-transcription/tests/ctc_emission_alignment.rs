use audio_analysis_transcription::{
    align_ctc_emissions, CtcEmissionAlignmentRequest, CtcVocabulary,
};

#[test]
fn public_emission_alignment_retains_words_and_segment_identity_with_model_timing() {
    let request = CtcEmissionAlignmentRequest {
        vocabulary: CtcVocabulary {
            blank_id: 0,
            tokens: ["<pad>", "|", "A", "B"].map(str::to_owned).to_vec(),
        },
        emissions: [0, 2, 0, 1, 3, 0]
            .into_iter()
            .map(|id| {
                let mut row = vec![-7.0; 4];
                row[id] = -0.01;
                row
            })
            .collect(),
        text: "A B".into(),
        segment_index: 7,
        start_seconds: 10.0,
        end_seconds: 13.0,
    };
    let aligned = align_ctc_emissions(request).unwrap();
    assert_eq!(aligned.len(), 2);
    assert_eq!(
        (
            &*aligned[0].text,
            aligned[0].segment_index,
            aligned[0].word_index
        ),
        ("A", 7, 0)
    );
    assert_eq!(
        (aligned[0].start_seconds, aligned[0].end_seconds),
        (10.5, 11.0)
    );
    assert_eq!(
        (aligned[1].start_seconds, aligned[1].end_seconds),
        (12.0, 12.5)
    );
    assert!(aligned.iter().all(|word| word
        .confidence
        .is_some_and(|value| value > 0.98 && value <= 1.0)));
}

fn minimal_request() -> CtcEmissionAlignmentRequest {
    CtcEmissionAlignmentRequest {
        vocabulary: CtcVocabulary {
            blank_id: 0,
            tokens: ["<pad>", "|", "A", "B"].map(str::to_owned).to_vec(),
        },
        emissions: vec![vec![-0.1; 4]; 6],
        text: "A B".into(),
        segment_index: 0,
        start_seconds: 0.0,
        end_seconds: 3.0,
    }
}

#[test]
fn impossible_or_malformed_evidence_never_produces_guessed_word_times() {
    let mut impossible = minimal_request();
    impossible.emissions.truncate(2);
    assert!(align_ctc_emissions(impossible).is_err());
    let mut malformed = minimal_request();
    malformed.emissions[0].pop();
    assert!(align_ctc_emissions(malformed).is_err());
    let mut logits = minimal_request();
    logits.emissions[0][0] = 1.0;
    assert!(align_ctc_emissions(logits).is_err());
    let mut reversed = minimal_request();
    reversed.end_seconds = -1.0;
    assert!(align_ctc_emissions(reversed).is_err());
}

#[test]
fn large_transcript_model_products_fail_before_allocating_an_unbounded_trellis() {
    let mut large = minimal_request();
    large.emissions = vec![vec![-0.1; 4]; 65_536];
    large.text = "A ".repeat(128);
    assert!(align_ctc_emissions(large)
        .unwrap_err()
        .to_string()
        .contains("bounded trellis"));
}
