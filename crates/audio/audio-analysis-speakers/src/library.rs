//! Speaker identities, enrolled profiles, identification and the speaker library.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use audio_contracts::{DetectError, Result};

use crate::library_transfer;
use crate::{SpeakerAudio, SpeakerEmbedding, SpeakerEmbeddingExtractor, SpeakerEmbeddingModel};

pub(crate) const SNAPSHOT_VERSION: u32 = 1;

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
/// Identifier for an enrolled speaker.
pub struct SpeakerId(String);

impl SpeakerId {
    /// Creates a new value.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "speaker id must not be empty".to_string(),
            ));
        }
        Ok(Self(value))
    }

    /// Returns this value as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for SpeakerId {
    type Error = DetectError;

    fn try_from(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<String> for SpeakerId {
    type Error = DetectError;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Human-readable speaker label.
pub struct SpeakerLabel(String);

impl SpeakerLabel {
    /// Creates a new value.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(DetectError::InvalidArgument(
                "speaker label must not be empty".to_string(),
            ));
        }
        Ok(Self(value))
    }

    /// Returns this value as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for SpeakerLabel {
    type Error = DetectError;

    fn try_from(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<String> for SpeakerLabel {
    type Error = DetectError;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Enrolled speaker profile.
pub struct SpeakerProfile {
    id: SpeakerId,
    label: SpeakerLabel,
    embeddings: Vec<SpeakerEmbedding>,
    metadata: BTreeMap<String, String>,
}

impl SpeakerProfile {
    /// Creates a new value.
    pub fn new(id: SpeakerId, label: SpeakerLabel) -> Self {
        Self {
            id,
            label,
            embeddings: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Returns this value with embedding.
    pub fn with_embedding(mut self, embedding: SpeakerEmbedding) -> Result<Self> {
        self.add_embedding(embedding)?;
        Ok(self)
    }

    /// Returns this value with metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Adds an embedding.
    pub fn add_embedding(&mut self, embedding: SpeakerEmbedding) -> Result<()> {
        embedding.validate_stored()?;
        if let Some(model) = self.embedding_model() {
            validate_model_compatible(model, embedding.model())?;
        }
        self.embeddings.push(embedding);
        Ok(())
    }

    /// Returns id.
    pub fn id(&self) -> &SpeakerId {
        &self.id
    }

    /// Returns label.
    pub fn label(&self) -> &SpeakerLabel {
        &self.label
    }

    /// Returns embeddings.
    pub fn embeddings(&self) -> &[SpeakerEmbedding] {
        &self.embeddings
    }

    /// Returns metadata.
    pub fn metadata_map(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Returns embedding model.
    pub fn embedding_model(&self) -> Option<&SpeakerEmbeddingModel> {
        self.embeddings.first().map(SpeakerEmbedding::model)
    }
}

/// Mutable identity fields for an existing speaker profile.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SpeakerProfileUpdate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Speaker match result.
pub struct SpeakerMatch {
    /// Matched speaker id.
    pub speaker_id: SpeakerId,
    /// Matched speaker label.
    pub label: SpeakerLabel,
    /// Cosine similarity score.
    pub score: f32,
    /// Difference between this score and the next-best score.
    pub margin: Option<f32>,
    /// Confidence bucket derived from score and margin.
    pub confidence: SpeakerConfidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Speaker match confidence bucket.
pub enum SpeakerConfidence {
    /// Score and margin are comfortably above common thresholds.
    High,
    /// Score passes threshold but leaves less separation.
    Medium,
    /// Score is weak or accepted only by permissive settings.
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Policy for accepted unknown results.
pub enum UnknownSpeakerPolicy {
    /// Return a result with `unknown` set when thresholds fail.
    #[default]
    ReturnUnknown,
    /// Return no best match when thresholds fail.
    NoMatch,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Speaker identification thresholds and result options.
pub struct SpeakerIdentificationOptions {
    /// Minimum accepted best score.
    pub min_score: f32,
    /// Optional minimum best-vs-second margin.
    pub min_margin: Option<f32>,
    /// Maximum ranked results to return.
    pub max_results: usize,
    /// Unknown handling policy.
    pub unknown_policy: UnknownSpeakerPolicy,
}

impl SpeakerIdentificationOptions {
    /// Creates a new value.
    pub fn new(min_score: f32) -> Result<Self> {
        let options = Self {
            min_score,
            ..Self::default()
        };
        options.validate()?;
        Ok(options)
    }

    /// Returns this value with a minimum margin.
    pub fn min_margin(mut self, min_margin: f32) -> Result<Self> {
        self.min_margin = Some(min_margin);
        self.validate()?;
        Ok(self)
    }

    /// Returns this value with max results.
    pub fn max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    /// Returns this value with unknown policy.
    pub fn unknown_policy(mut self, unknown_policy: UnknownSpeakerPolicy) -> Self {
        self.unknown_policy = unknown_policy;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !self.min_score.is_finite() || !(-1.0..=1.0).contains(&self.min_score) {
            return Err(DetectError::InvalidArgument(
                "minimum speaker identification score must be finite and between -1.0 and 1.0"
                    .to_string(),
            ));
        }
        if let Some(margin) = self.min_margin {
            if !margin.is_finite() || margin < 0.0 {
                return Err(DetectError::InvalidArgument(
                    "minimum speaker identification margin must be finite and non-negative"
                        .to_string(),
                ));
            }
        }
        if self.max_results == 0 {
            return Err(DetectError::InvalidArgument(
                "max speaker identification results must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for SpeakerIdentificationOptions {
    fn default() -> Self {
        Self {
            min_score: 0.82,
            min_margin: Some(0.05),
            max_results: 5,
            unknown_policy: UnknownSpeakerPolicy::ReturnUnknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Speaker identification result.
pub struct SpeakerIdentificationResult {
    /// Accepted best match, if any.
    pub best_match: Option<SpeakerMatch>,
    /// Ranked candidate matches.
    pub ranked_matches: Vec<SpeakerMatch>,
    /// Whether the query should be treated as unknown.
    pub unknown: bool,
    /// Best-vs-second score margin.
    pub margin: Option<f32>,
    /// Query embedding model metadata.
    pub embedding_model: SpeakerEmbeddingModel,
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Speaker profile library.
pub struct SpeakerLibrary {
    profiles: BTreeMap<SpeakerId, SpeakerProfile>,
    embedding_model: Option<SpeakerEmbeddingModel>,
}

impl SpeakerLibrary {
    /// Creates a new value.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enrolls a speaker from audio.
    pub fn enroll<E: SpeakerEmbeddingExtractor>(
        &mut self,
        id: SpeakerId,
        label: SpeakerLabel,
        audio: &SpeakerAudio<'_>,
        extractor: &mut E,
    ) -> Result<SpeakerProfile> {
        let embedding = extractor.embed_speaker(audio)?;
        let profile = SpeakerProfile::new(id, label).with_embedding(embedding)?;
        self.add_profile(profile.clone())?;
        Ok(profile)
    }

    /// Adds a profile.
    pub fn add_profile(&mut self, profile: SpeakerProfile) -> Result<()> {
        validate_profile_header(&profile)?;
        if profile.embeddings().is_empty() {
            return Err(DetectError::InvalidArgument(format!(
                "speaker profile `{}` must contain at least one embedding",
                profile.id().as_str()
            )));
        }
        let model = profile.embedding_model().cloned().ok_or_else(|| {
            DetectError::InvalidArgument("speaker profile must contain a model".to_string())
        })?;
        for embedding in profile.embeddings() {
            embedding.validate_stored()?;
            validate_model_compatible(&model, embedding.model())?;
        }
        self.validate_model(&model)?;
        if self.profiles.contains_key(profile.id()) {
            return Err(DetectError::InvalidArgument(
                "duplicate speaker id".to_string(),
            ));
        }
        self.embedding_model = Some(model);
        self.profiles.insert(profile.id().clone(), profile);
        Ok(())
    }

    /// Adds an embedding to an existing speaker profile.
    pub fn add_embedding(&mut self, id: &SpeakerId, embedding: SpeakerEmbedding) -> Result<()> {
        self.validate_model(embedding.model())?;
        let profile = self.profiles.get_mut(id).ok_or_else(|| {
            DetectError::InvalidArgument(format!("unknown speaker id `{}`", id.as_str()))
        })?;
        profile.add_embedding(embedding)
    }

    /// Updates reusable profile identity fields while preserving its stable id and embeddings.
    pub fn update_profile(&mut self, id: &SpeakerId, update: SpeakerProfileUpdate) -> Result<()> {
        if update.label.is_none() && update.metadata.is_none() {
            return Err(DetectError::InvalidArgument(
                "speaker profile update must include a label or metadata".to_string(),
            ));
        }
        let profile = self.profiles.get_mut(id).ok_or_else(|| {
            DetectError::InvalidArgument(format!("unknown speaker id `{}`", id.as_str()))
        })?;
        if let Some(label) = update.label {
            profile.label = SpeakerLabel::new(label)?;
        }
        if let Some(metadata) = update.metadata {
            profile.metadata = metadata;
        }
        Ok(())
    }

    /// Deletes a profile and returns the removed identity, embeddings, and metadata.
    pub fn delete_profile(&mut self, id: &SpeakerId) -> Result<SpeakerProfile> {
        let profile = self.profiles.remove(id).ok_or_else(|| {
            DetectError::InvalidArgument(format!("unknown speaker id `{}`", id.as_str()))
        })?;
        if self.profiles.is_empty() {
            self.embedding_model = None;
        }
        Ok(profile)
    }

    /// Returns profile.
    pub fn profile(&self, id: &SpeakerId) -> Option<&SpeakerProfile> {
        self.profiles.get(id)
    }

    /// Returns profiles.
    pub fn profiles(&self) -> impl Iterator<Item = &SpeakerProfile> {
        self.profiles.values()
    }

    /// Returns len.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Returns whether is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Returns embedding model.
    pub fn embedding_model(&self) -> Option<&SpeakerEmbeddingModel> {
        self.embedding_model.as_ref()
    }

    /// Identifies a speaker embedding.
    pub fn identify(
        &self,
        embedding: &SpeakerEmbedding,
        options: &SpeakerIdentificationOptions,
    ) -> Result<SpeakerIdentificationResult> {
        options.validate()?;
        self.validate_model(embedding.model())?;

        let mut ranked_matches = Vec::new();
        for profile in self.profiles.values() {
            let Some(score) = best_profile_score(profile, embedding)? else {
                continue;
            };
            ranked_matches.push(SpeakerMatch {
                speaker_id: profile.id().clone(),
                label: profile.label().clone(),
                score,
                margin: None,
                confidence: SpeakerConfidence::Low,
            });
        }
        ranked_matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.speaker_id.cmp(&right.speaker_id))
        });

        let margin = match (ranked_matches.first(), ranked_matches.get(1)) {
            (Some(best), Some(second)) => Some(best.score - second.score),
            (Some(_), None) => None,
            _ => None,
        };
        for index in 0..ranked_matches.len() {
            let next_score = ranked_matches.get(index + 1).map(|value| value.score);
            ranked_matches[index].margin =
                next_score.map(|score| ranked_matches[index].score - score);
            ranked_matches[index].confidence =
                confidence_for(ranked_matches[index].score, ranked_matches[index].margin);
        }

        let accepted = ranked_matches.first().is_some_and(|best| {
            best.score >= options.min_score
                && options
                    .min_margin
                    .is_none_or(|minimum| margin.is_some_and(|value| value >= minimum))
        });
        let best_match = if accepted {
            ranked_matches.first().cloned()
        } else {
            None
        };
        let unknown = !accepted && options.unknown_policy == UnknownSpeakerPolicy::ReturnUnknown;

        ranked_matches.truncate(options.max_results);
        Ok(SpeakerIdentificationResult {
            best_match,
            ranked_matches,
            unknown,
            margin,
            embedding_model: embedding.model().clone(),
        })
    }

    /// Converts this value to snapshot.
    pub fn to_snapshot(&self) -> Result<SpeakerLibrarySnapshot> {
        let embedding_model = self.embedding_model.clone().ok_or_else(|| {
            DetectError::InvalidArgument(
                "speaker library snapshot requires at least one profile".to_string(),
            )
        })?;
        Ok(SpeakerLibrarySnapshot {
            version: SNAPSHOT_VERSION,
            embedding_model,
            profiles: self.profiles.values().cloned().collect(),
        })
    }

    /// Builds this value from snapshot.
    pub fn from_snapshot(snapshot: SpeakerLibrarySnapshot) -> Result<Self> {
        if snapshot.version != SNAPSHOT_VERSION {
            return Err(DetectError::InvalidArgument(format!(
                "unsupported speaker library snapshot version `{}`",
                snapshot.version
            )));
        }
        snapshot.embedding_model.validate()?;
        let mut library = Self::new();
        for profile in snapshot.profiles {
            for embedding in profile.embeddings() {
                validate_model_compatible(&snapshot.embedding_model, embedding.model())?;
                if embedding.dimensions() != snapshot.embedding_model.dimensions {
                    return Err(DetectError::InvalidArgument(
                        "speaker library snapshot embedding dimensions did not match model"
                            .to_string(),
                    ));
                }
            }
            library.add_profile(profile)?;
        }
        if library.embedding_model.as_ref() != Some(&snapshot.embedding_model) {
            return Err(DetectError::InvalidArgument(
                "speaker library snapshot model did not match profiles".to_string(),
            ));
        }
        Ok(library)
    }

    /// Converts this value to JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        if self.is_empty() {
            return library_transfer::empty_document_json();
        }
        serde_json::to_string_pretty(&self.to_snapshot()?)
            .map_err(|err| DetectError::Source(err.to_string()))
    }

    /// Builds this value from JSON str.
    pub fn from_json_str(json: &str) -> Result<Self> {
        library_transfer::parse_document(json)
    }

    /// Loads and validates a canonical Speaker Library snapshot from a caller-owned path.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = fs::read_to_string(path).map_err(|error| {
            DetectError::Source(format!(
                "failed to read Speaker Library `{}`: {error}",
                path.display()
            ))
        })?;
        Self::from_json_str(&json).map_err(|error| {
            DetectError::InvalidArgument(format!(
                "Speaker Library `{}` is malformed or incompatible: {error}",
                path.display()
            ))
        })
    }

    /// Loads a Speaker Library when present, preserving the ordinary missing-file state.
    pub fn load_if_present(path: impl AsRef<Path>) -> Result<Option<Self>> {
        let path = path.as_ref();
        match Self::load(path) {
            Ok(library) => Ok(Some(library)),
            Err(DetectError::Source(message)) if message.contains("os error 2") => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Serializes, validates, and atomically promotes this library at a caller-owned path.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        library_transfer::save_legacy(self, path.as_ref())
    }

    pub(crate) fn validate_model(&self, model: &SpeakerEmbeddingModel) -> Result<()> {
        if let Some(expected) = &self.embedding_model {
            validate_model_compatible(expected, model)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
/// Stable populated speaker library snapshot. Canonical JSON additionally
/// supports an empty library with no embedding model.
pub struct SpeakerLibrarySnapshot {
    /// Snapshot format version.
    pub version: u32,
    /// Embedding model used by all profiles.
    pub embedding_model: SpeakerEmbeddingModel,
    /// Speaker profiles.
    pub profiles: Vec<SpeakerProfile>,
}

fn best_profile_score(
    profile: &SpeakerProfile,
    embedding: &SpeakerEmbedding,
) -> Result<Option<f32>> {
    let mut best = None;
    for profile_embedding in profile.embeddings() {
        let score = profile_embedding.cosine_similarity(embedding)?;
        best = Some(best.map(|value: f32| value.max(score)).unwrap_or(score));
    }
    Ok(best)
}

fn validate_profile_header(profile: &SpeakerProfile) -> Result<()> {
    if profile.id().as_str().trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "speaker profile id must not be empty".to_string(),
        ));
    }
    if profile.label().as_str().trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "speaker profile label must not be empty".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_model_compatible(
    expected: &SpeakerEmbeddingModel,
    actual: &SpeakerEmbeddingModel,
) -> Result<()> {
    if expected != actual {
        return Err(DetectError::InvalidArgument(format!(
            "speaker embedding model mismatch: expected {}@{} ({} dims), got {}@{} ({} dims)",
            expected.name,
            expected.version,
            expected.dimensions,
            actual.name,
            actual.version,
            actual.dimensions
        )));
    }
    Ok(())
}

fn confidence_for(score: f32, margin: Option<f32>) -> SpeakerConfidence {
    if score >= 0.90 && margin.is_none_or(|value| value >= 0.10) {
        SpeakerConfidence::High
    } else if score >= 0.82 && margin.is_none_or(|value| value >= 0.05) {
        SpeakerConfidence::Medium
    } else {
        SpeakerConfidence::Low
    }
}
