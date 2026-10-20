#![cfg(any(feature = "pyannote-core", feature = "pyannote-diarization"))]

use audio_analysis_speakers::pyannote::community_embedding_requests;

#[test]
fn portable_embedding_masks_preserve_clean_and_overlap_semantics() {
    let mut scores = vec![0.0; 589 * 3];
    assert!(community_embedding_requests(&scores).unwrap().is_empty());
    for frame in 0..589 {
        scores[frame * 3] = 1.0;
    }
    let clean = community_embedding_requests(&scores).unwrap();
    assert_eq!(clean.len(), 1);
    assert_eq!(clean[0].local_speaker, 0);
    assert_eq!(clean[0].clean_frames, 589);
    assert_eq!(clean[0].mask, vec![1.0; 589]);
    for frame in 0..589 {
        scores[frame * 3 + 1] = 1.0;
    }
    let overlap = community_embedding_requests(&scores).unwrap();
    assert_eq!(overlap.len(), 2);
    assert!(overlap.iter().all(|request| request.clean_frames == 0));
    assert!(overlap.iter().all(|request| request.mask == vec![1.0; 589]));
}

#[test]
fn portable_masks_reject_malformed_model_outputs() {
    for invalid in [
        vec![],
        vec![0.0; 588 * 3],
        vec![f32::NAN; 589 * 3],
        vec![1.1; 589 * 3],
    ] {
        assert!(community_embedding_requests(&invalid).is_err());
    }
}

#[test]
fn portable_processor_fails_closed_on_missing_or_changed_licensed_resources() {
    use audio_analysis_speakers::pyannote::CommunityDiarizationProcessor;
    assert!(CommunityDiarizationProcessor::from_resources(b"{}", b"{}", b"{}", 16_000).is_err());
    assert!(CommunityDiarizationProcessor::from_resources(b"", b"", b"", 0).is_err());
}
