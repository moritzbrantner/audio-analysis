//! Canonical Speaker Library transfer and guarded persistence operations.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use audio_contracts::{DetectError, Result};

use crate::library::SNAPSHOT_VERSION;
use crate::{SpeakerEmbeddingModel, SpeakerLibrary, SpeakerLibrarySnapshot, SpeakerProfile};

impl SpeakerLibrary {
    /// Exports canonical identities to a new file without replacing an existing destination.
    ///
    /// The destination becomes visible only after the complete snapshot has been
    /// written and synchronized. Product consent and destination authorization
    /// belong to the caller. Embedding vectors are never included in errors.
    pub fn export_new(&self, destination: impl AsRef<Path>) -> Result<()> {
        let destination = destination.as_ref();
        match fs::symlink_metadata(destination) {
            Ok(_) => {
                return Err(transfer_error(
                    destination,
                    std::io::Error::from(std::io::ErrorKind::AlreadyExists),
                ))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(transfer_error(destination, error)),
        }
        let _lock = acquire_library_lock(destination)?;
        let json = self.to_json_string()?;
        write_snapshot(destination, json.as_bytes(), SnapshotPromotion::Create)
    }
}

enum SnapshotPromotion {
    Create,
    Replace,
}

fn write_snapshot(destination: &Path, bytes: &[u8], promotion: SnapshotPromotion) -> Result<()> {
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let filename = destination.file_name().ok_or_else(|| {
        DetectError::InvalidArgument("Speaker Library export requires a filename".to_string())
    })?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            DetectError::Source(format!("Speaker Library export clock unavailable: {error}"))
        })?
        .as_nanos();
    let temporary_path = parent.join(format!(
        ".{}.{}.{nonce}.tmp",
        filename.to_string_lossy(),
        std::process::id()
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temporary_path)
        .map_err(|error| transfer_error(destination, error))?;
    let temporary = TemporarySnapshot(temporary_path);
    let written = file.write_all(bytes).and_then(|_| file.sync_all());
    drop(file);
    written.map_err(|error| transfer_error(destination, error))?;
    // A same-directory hard link atomically installs the fully written snapshot
    // only when the destination is absent; unlike rename it cannot overwrite it.
    let result = match promotion {
        SnapshotPromotion::Create => fs::hard_link(&temporary.0, destination),
        SnapshotPromotion::Replace => fs::rename(&temporary.0, destination),
    };
    result.map_err(|error| transfer_error(destination, error))
}

fn transfer_error(destination: &Path, error: std::io::Error) -> DetectError {
    DetectError::Source(format!(
        "failed to persist Speaker Library `{}`: {error}",
        destination.display()
    ))
}

struct TemporarySnapshot(PathBuf);
impl Drop for TemporarySnapshot {
    fn drop(&mut self) {
        // Best-effort cleanup never removes the promoted destination.
        let _ = fs::remove_file(&self.0);
    }
}

// The private wire DTO extends canonical JSON with model-less empty libraries;
// the existing populated Rust snapshot API and its validation remain compatible.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LibraryDocument {
    version: u32,
    #[serde(default)]
    embedding_model: Option<SpeakerEmbeddingModel>,
    profiles: Vec<SpeakerProfile>,
}

pub(super) fn empty_document_json() -> Result<String> {
    serde_json::to_string_pretty(&LibraryDocument {
        version: SNAPSHOT_VERSION,
        embedding_model: None,
        profiles: Vec::new(),
    })
    .map_err(|error| DetectError::Source(error.to_string()))
}

pub(super) fn parse_document(json: &str) -> Result<SpeakerLibrary> {
    let document = serde_json::from_str::<LibraryDocument>(json)
        .map_err(|error| DetectError::Source(error.to_string()))?;
    match document.embedding_model {
        Some(embedding_model) => SpeakerLibrary::from_snapshot(SpeakerLibrarySnapshot {
            version: document.version,
            embedding_model,
            profiles: document.profiles,
        }),
        None => {
            if document.version != SNAPSHOT_VERSION {
                return Err(DetectError::InvalidArgument(format!(
                    "unsupported speaker library snapshot version `{}`",
                    document.version
                )));
            }
            if !document.profiles.is_empty() {
                return Err(DetectError::InvalidArgument(
                    "a populated Speaker Library requires its embedding model".to_string(),
                ));
            }
            Ok(SpeakerLibrary::new())
        }
    }
}

/// Explicit treatment of the existing identities during a canonical import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SpeakerLibraryImportMode {
    /// Add new identities; reject conflicting existing IDs or labels.
    Merge,
    /// Replace all identities after compatibility validation and caller consent.
    Replace,
}

/// Safe preflight facts, without embedding vectors.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerLibraryImportSummary {
    pub existing_profiles: usize,
    pub incoming_profiles: usize,
    pub added_profiles: usize,
    pub unchanged_profiles: usize,
    pub replaced_profiles: usize,
    pub removed_profiles: usize,
    pub resulting_profiles: usize,
    pub changed: bool,
}

/// A validated import candidate. Constructing it never mutates either library.
pub struct SpeakerLibraryImportPlan {
    baseline: SpeakerLibrary,
    candidate: SpeakerLibrary,
    summary: SpeakerLibraryImportSummary,
}

impl std::fmt::Debug for SpeakerLibraryImportPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpeakerLibraryImportPlan")
            .field("summary", &self.summary)
            .finish_non_exhaustive()
    }
}

impl SpeakerLibraryImportPlan {
    pub fn library(&self) -> &SpeakerLibrary {
        &self.candidate
    }
    pub fn summary(&self) -> &SpeakerLibraryImportSummary {
        &self.summary
    }
}

impl SpeakerLibrary {
    /// Plans a strict import without comparing incompatible vectors or changing identities.
    ///
    /// Identical profiles are retained. Merge conflicts require an explicit
    /// replacement decision or correction of the source library. Metadata is
    /// copied as supplied, including draft/confirmed status.
    pub fn prepare_import(
        &self,
        incoming: &Self,
        mode: SpeakerLibraryImportMode,
    ) -> Result<SpeakerLibraryImportPlan> {
        if let Some(model) = incoming.embedding_model() {
            self.validate_model(model)?;
        }
        validate_unique_labels(incoming)?;
        let mut candidate = match mode {
            SpeakerLibraryImportMode::Merge => {
                validate_unique_labels(self)?;
                self.clone()
            }
            SpeakerLibraryImportMode::Replace => incoming.clone(),
        };
        let mut added_profiles = 0;
        let mut unchanged_profiles = 0;
        let mut replaced_profiles = 0;
        for profile in incoming.profiles() {
            match self.profile(profile.id()) {
                Some(existing) if existing == profile => unchanged_profiles += 1,
                Some(_) => match mode {
                    SpeakerLibraryImportMode::Merge => return Err(DetectError::InvalidArgument(format!("conflicting stable speaker id `{}`; explicitly replace or resolve the source identity", profile.id().as_str()))),
                    SpeakerLibraryImportMode::Replace => replaced_profiles += 1,
                },
                None => {
                    added_profiles += 1;
                    if mode == SpeakerLibraryImportMode::Merge { candidate.add_profile(profile.clone())?; }
                }
            }
        }
        validate_unique_labels(&candidate)?;
        let removed_profiles = match mode {
            SpeakerLibraryImportMode::Merge => 0,
            SpeakerLibraryImportMode::Replace => self
                .profiles()
                .filter(|profile| incoming.profile(profile.id()).is_none())
                .count(),
        };
        let summary = SpeakerLibraryImportSummary {
            existing_profiles: self.len(),
            incoming_profiles: incoming.len(),
            added_profiles,
            unchanged_profiles,
            replaced_profiles,
            removed_profiles,
            resulting_profiles: candidate.len(),
            changed: self != &candidate,
        };
        Ok(SpeakerLibraryImportPlan {
            baseline: self.clone(),
            candidate,
            summary,
        })
    }
}

fn validate_unique_labels(library: &SpeakerLibrary) -> Result<()> {
    let mut labels = std::collections::BTreeSet::new();
    for profile in library.profiles() {
        if !labels.insert(profile.label().as_str()) {
            return Err(DetectError::InvalidArgument(format!("conflicting Speaker Library label `{}`; resolve the source identities before importing", profile.label().as_str())));
        }
    }
    Ok(())
}

/// Outcome of a confirmed import, without embedding data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerLibraryImportReceipt {
    pub changed: bool,
    pub backup_path: Option<PathBuf>,
    pub summary: SpeakerLibraryImportSummary,
}

impl SpeakerLibraryImportPlan {
    /// Commits this reviewed plan only if the active canonical identities still
    /// match its baseline. Cooperating canonical writers share a bounded lock.
    /// A changed existing document is backed up without overwriting a prior file
    /// before its validated replacement is atomically promoted.
    pub fn commit(
        self,
        destination: impl AsRef<Path>,
        backup_destination: impl AsRef<Path>,
    ) -> Result<SpeakerLibraryImportReceipt> {
        let destination = destination.as_ref();
        let backup_destination = backup_destination.as_ref();
        let _lock = acquire_library_lock(destination)?;
        let previous = match fs::read_to_string(destination) {
            Ok(json) => Some(json),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(transfer_error(destination, error)),
        };
        let current = match &previous {
            Some(json) => SpeakerLibrary::from_json_str(json)?,
            None => SpeakerLibrary::new(),
        };
        if current != self.baseline {
            return Err(DetectError::InvalidArgument(
                "Speaker Library changed after preflight; review a fresh import summary"
                    .to_string(),
            ));
        }
        if !self.summary.changed {
            return Ok(SpeakerLibraryImportReceipt {
                changed: false,
                backup_path: None,
                summary: self.summary,
            });
        }
        if destination == backup_destination {
            return Err(DetectError::InvalidArgument(
                "Speaker Library backup destination must differ from the active library"
                    .to_string(),
            ));
        }
        let replacement = self.candidate.to_json_string()?;
        let backup_path = match &previous {
            Some(bytes) => {
                write_snapshot(
                    backup_destination,
                    bytes.as_bytes(),
                    SnapshotPromotion::Create,
                )?;
                Some(backup_destination.to_path_buf())
            }
            None => None,
        };
        let promotion = if previous.is_some() {
            SnapshotPromotion::Replace
        } else {
            SnapshotPromotion::Create
        };
        write_snapshot(destination, replacement.as_bytes(), promotion).map_err(|error| {
            DetectError::Source(format!(
                "{error}; active library was not replaced; prior snapshot backup: {}",
                backup_path
                    .as_deref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "none (new library)".to_string())
            ))
        })?;
        Ok(SpeakerLibraryImportReceipt {
            changed: true,
            backup_path,
            summary: self.summary,
        })
    }
}

fn library_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn acquire_library_lock(path: &Path) -> Result<fs::File> {
    let filename = path.file_name().ok_or_else(|| {
        DetectError::InvalidArgument("Speaker Library path requires a filename".to_string())
    })?;
    let mut lock_name = std::ffi::OsString::from(".");
    lock_name.push(filename);
    lock_name.push(".lock");
    let lock_path = library_parent(path).join(lock_name);
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(&lock_path)
        .map_err(|error| transfer_error(path, error))?;
    file.try_lock().map_err(|error| {
        DetectError::Source(format!(
            "Speaker Library is busy or cannot be locked: {error}"
        ))
    })?;
    Ok(file)
}

pub(super) fn save_legacy(library: &SpeakerLibrary, path: &Path) -> Result<()> {
    let parent = library_parent(path);
    if library.is_empty() {
        match fs::symlink_metadata(parent) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(transfer_error(path, error)),
            Ok(_) => {}
        }
        let _lock = acquire_library_lock(path)?;
        return match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(transfer_error(path, error)),
        };
    }
    fs::create_dir_all(parent).map_err(|error| transfer_error(path, error))?;
    let _lock = acquire_library_lock(path)?;
    let json = library.to_json_string()?;
    let _validated = SpeakerLibrary::from_json_str(&json)?;
    write_snapshot(path, json.as_bytes(), SnapshotPromotion::Replace)
}
