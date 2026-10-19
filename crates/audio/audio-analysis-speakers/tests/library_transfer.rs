use audio_analysis_speakers::{
    SpeakerEmbedding, SpeakerEmbeddingModel, SpeakerId, SpeakerLabel, SpeakerLibrary,
    SpeakerProfile,
};

fn library_with_confirmed_and_draft() -> SpeakerLibrary {
    let mut library = SpeakerLibrary::new();
    for (id, label, status, values) in [
        ("confirmed-a", "Alice", "confirmed", [1.0, 0.0]),
        ("draft-b", "Anonymous B", "draft", [0.0, 1.0]),
    ] {
        let embedding = SpeakerEmbedding::new(
            values,
            SpeakerEmbeddingModel::spectral_baseline(2).unwrap(),
            16_000,
        )
        .unwrap();
        let profile = SpeakerProfile::new(
            SpeakerId::new(id).unwrap(),
            SpeakerLabel::new(label).unwrap(),
        )
        .with_embedding(embedding)
        .unwrap()
        .metadata("status", status);
        library.add_profile(profile).unwrap();
    }
    library
}

#[test]
fn exporting_a_canonical_library_preserves_confirmed_and_draft_identities() {
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("portable-library.json");
    let library = library_with_confirmed_and_draft();
    library.export_new(&destination).unwrap();
    let loaded = SpeakerLibrary::load(&destination).unwrap();
    assert_eq!(loaded, library);
    assert_eq!(
        loaded
            .profile(&SpeakerId::new("draft-b").unwrap())
            .unwrap()
            .metadata_map()
            .get("status")
            .map(String::as_str),
        Some("draft")
    );
}

#[test]
fn exporting_does_not_overwrite_an_existing_destination_or_leave_temporary_files() {
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("portable-library.json");
    std::fs::write(&destination, b"keep this existing file").unwrap();
    assert!(library_with_confirmed_and_draft()
        .export_new(&destination)
        .is_err());
    assert_eq!(
        std::fs::read(&destination).unwrap(),
        b"keep this existing file"
    );
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn exporting_a_new_empty_library_does_not_invent_an_embedding_model() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("library.json");
    SpeakerLibrary::new().export_new(&path).unwrap();
    let library = SpeakerLibrary::load(&path).unwrap();
    assert!(library.is_empty());
    assert!(library.embedding_model().is_none());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(json["profiles"], serde_json::json!([]));
    assert!(json["embedding_model"].is_null());
}

#[test]
fn importing_canonical_identities_is_deterministic_and_preserves_metadata() {
    use audio_analysis_speakers::SpeakerLibraryImportMode;
    let incoming = library_with_confirmed_and_draft();
    let empty = SpeakerLibrary::new();
    let plan = empty
        .prepare_import(&incoming, SpeakerLibraryImportMode::Merge)
        .unwrap();
    assert_eq!(plan.library(), &incoming);
    assert_eq!(plan.summary().added_profiles, 2);
    assert!(plan.summary().changed);
    let repeated = plan
        .library()
        .prepare_import(&incoming, SpeakerLibraryImportMode::Merge)
        .unwrap();
    assert!(!repeated.summary().changed);
    assert_eq!(repeated.summary().unchanged_profiles, 2);
    assert_eq!(repeated.library(), &incoming);
}

#[test]
fn importing_into_a_second_directory_preserves_identity_and_repeated_import_writes_nothing() {
    use audio_analysis_speakers::SpeakerLibraryImportMode;
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    std::fs::create_dir(&first).unwrap();
    std::fs::create_dir(&second).unwrap();
    let source = first.join("library.json");
    let destination = second.join("library.json");
    let backup = second.join("library.before-import.json");
    let trace = second.join("speaker-trace.json");
    std::fs::write(&trace, b"derived trace remains separate").unwrap();
    let incoming = library_with_confirmed_and_draft();
    incoming.export_new(&source).unwrap();
    SpeakerLibrary::new().export_new(&destination).unwrap();
    let plan = SpeakerLibrary::load(&destination)
        .unwrap()
        .prepare_import(
            &SpeakerLibrary::load(&source).unwrap(),
            SpeakerLibraryImportMode::Merge,
        )
        .unwrap();
    let receipt = plan.commit(&destination, &backup).unwrap();
    assert!(receipt.changed);
    assert_eq!(receipt.backup_path.as_deref(), Some(backup.as_path()));
    assert_eq!(SpeakerLibrary::load(&destination).unwrap(), incoming);
    assert!(SpeakerLibrary::load(&backup).unwrap().is_empty());
    let before = std::fs::metadata(&destination).unwrap().modified().unwrap();
    let entries = std::fs::read_dir(&second).unwrap().count();
    let repeated = SpeakerLibrary::load(&destination)
        .unwrap()
        .prepare_import(&incoming, SpeakerLibraryImportMode::Merge)
        .unwrap();
    let receipt = repeated.commit(&destination, &backup).unwrap();
    assert!(!receipt.changed);
    assert!(receipt.backup_path.is_none());
    assert_eq!(
        std::fs::metadata(&destination).unwrap().modified().unwrap(),
        before
    );
    assert_eq!(std::fs::read_dir(&second).unwrap().count(), entries);
    assert_eq!(
        std::fs::read(&trace).unwrap(),
        b"derived trace remains separate"
    );
}

#[test]
fn merge_conflicts_require_explicit_replacement_and_models_cannot_be_converted() {
    use audio_analysis_speakers::{
        SpeakerEmbeddingModelFamily, SpeakerLibraryImportMode, SpeakerProfileUpdate,
    };
    let existing = library_with_confirmed_and_draft();
    let mut conflicting = existing.clone();
    conflicting
        .update_profile(
            &SpeakerId::new("draft-b").unwrap(),
            SpeakerProfileUpdate {
                label: Some("User corrected label".into()),
                metadata: None,
            },
        )
        .unwrap();
    assert!(existing
        .prepare_import(&conflicting, SpeakerLibraryImportMode::Merge)
        .is_err());
    let replacement = existing
        .prepare_import(&conflicting, SpeakerLibraryImportMode::Replace)
        .unwrap();
    assert_eq!(replacement.library(), &conflicting);
    assert_eq!(replacement.summary().replaced_profiles, 1);
    for model in [
        SpeakerEmbeddingModel::new(
            SpeakerEmbeddingModelFamily::SpeechBrain,
            "spectral-speaker-baseline",
            "1",
            2,
        )
        .unwrap(),
        SpeakerEmbeddingModel::new(
            SpeakerEmbeddingModelFamily::SpectralBaseline,
            "other-name",
            "1",
            2,
        )
        .unwrap(),
        SpeakerEmbeddingModel::new(
            SpeakerEmbeddingModelFamily::SpectralBaseline,
            "spectral-speaker-baseline",
            "2",
            2,
        )
        .unwrap(),
        SpeakerEmbeddingModel::spectral_baseline(3).unwrap(),
    ] {
        let dimensions = model.dimensions;
        let embedding = SpeakerEmbedding::new(vec![1.0; dimensions], model, 16_000).unwrap();
        let mut incoming = SpeakerLibrary::new();
        incoming
            .add_profile(
                SpeakerProfile::new(
                    SpeakerId::new("other").unwrap(),
                    SpeakerLabel::new("Other").unwrap(),
                )
                .with_embedding(embedding)
                .unwrap(),
            )
            .unwrap();
        for mode in [
            SpeakerLibraryImportMode::Merge,
            SpeakerLibraryImportMode::Replace,
        ] {
            assert!(existing.prepare_import(&incoming, mode).is_err());
        }
    }
}

#[test]
fn conflicting_labels_are_not_silently_combined_into_an_existing_identity() {
    use audio_analysis_speakers::SpeakerLibraryImportMode;
    let existing = library_with_confirmed_and_draft();
    let mut incoming = SpeakerLibrary::new();
    incoming
        .add_profile(
            SpeakerProfile::new(
                SpeakerId::new("different-stable-id").unwrap(),
                SpeakerLabel::new("Alice").unwrap(),
            )
            .with_embedding(
                SpeakerEmbedding::new(
                    [1.0, 0.0],
                    SpeakerEmbeddingModel::spectral_baseline(2).unwrap(),
                    16_000,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(existing
        .prepare_import(&incoming, SpeakerLibraryImportMode::Merge)
        .is_err());
    let explicit_replacement = existing
        .prepare_import(&incoming, SpeakerLibraryImportMode::Replace)
        .unwrap();
    assert_eq!(explicit_replacement.summary().removed_profiles, 2);
    assert_eq!(explicit_replacement.summary().added_profiles, 1);
    assert_eq!(existing.len(), 2);
}

#[test]
fn a_changed_active_library_invalidates_the_import_preview_without_writing_a_backup() {
    use audio_analysis_speakers::{SpeakerLibraryImportMode, SpeakerProfileUpdate};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("library.json");
    let backup = directory.path().join("backup.json");
    let existing = library_with_confirmed_and_draft();
    existing.export_new(&path).unwrap();
    let plan = existing
        .prepare_import(&SpeakerLibrary::new(), SpeakerLibraryImportMode::Replace)
        .unwrap();
    assert!(!format!("{plan:?}").contains("embeddings"));
    let mut concurrent = existing.clone();
    concurrent
        .update_profile(
            &SpeakerId::new("confirmed-a").unwrap(),
            SpeakerProfileUpdate {
                label: Some("Concurrent correction".into()),
                metadata: None,
            },
        )
        .unwrap();
    concurrent.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let error = plan.commit(&path, &backup).unwrap_err();
    assert!(error.to_string().contains("changed after preflight"));
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert!(!backup.exists());
}

#[test]
fn a_failed_backup_promotion_preserves_both_existing_files_and_cleans_staging() {
    use audio_analysis_speakers::SpeakerLibraryImportMode;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("library.json");
    let backup = directory.path().join("backup.json");
    let existing = library_with_confirmed_and_draft();
    existing.export_new(&path).unwrap();
    std::fs::write(&backup, b"existing backup must survive").unwrap();
    let before = std::fs::read(&path).unwrap();
    let entries = std::fs::read_dir(directory.path()).unwrap().count();
    let plan = existing
        .prepare_import(&SpeakerLibrary::new(), SpeakerLibraryImportMode::Replace)
        .unwrap();
    assert!(plan.commit(&path, &backup).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(
        std::fs::read(&backup).unwrap(),
        b"existing backup must survive"
    );
    assert_eq!(
        std::fs::read_dir(directory.path()).unwrap().count(),
        entries
    );
}

#[test]
fn canonical_writers_fail_closed_when_another_writer_owns_the_library_lock() {
    use audio_analysis_speakers::SpeakerLibraryImportMode;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("library.json");
    let backup = directory.path().join("backup.json");
    let library = library_with_confirmed_and_draft();
    library.export_new(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    // The documented sidecar coordinates all canonical writers across rename.
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(directory.path().join(".library.json.lock"))
        .unwrap();
    lock.try_lock().unwrap();
    assert!(library
        .save(&path)
        .unwrap_err()
        .to_string()
        .contains("busy"));
    assert!(SpeakerLibrary::new().save(&path).is_err());
    let plan = library
        .prepare_import(&SpeakerLibrary::new(), SpeakerLibraryImportMode::Replace)
        .unwrap();
    assert!(plan.commit(&path, &backup).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(!backup.exists());
}
