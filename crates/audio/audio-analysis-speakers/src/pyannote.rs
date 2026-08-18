//! Native pyannote community diarization support.
//!
//! The pipeline shape and defaults mirror `pyannote.audio` community diarization
//! behavior: segmentation windows, speaker masks for embedding extraction,
//! speaker-bound validation, and `SPEAKER_XX` labels. The ONNX model artifacts
//! are caller supplied and are intentionally not bundled in this crate.

pub mod clustering;
pub mod embedding;
pub mod manifest;
pub mod plda;
pub mod reconstruct;
pub mod segmentation;
pub mod trace;
pub mod vbx;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use audio_contracts::{DetectError, Result};
use runtime_onnx::{
    f32_output_by_name_or_index, first_f32_output, single_f32_input, OnnxDimension, OnnxF32Tensor,
    OnnxIoInfo, OnnxRunner, OnnxSession, OnnxSessionMetadata, OnnxTensorElementType,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    AudioRuntime, SpeakerAudio, SpeakerDiarizationResponse, SpeakerEmbedding,
    SpeakerEmbeddingModel, SpeakerEmbeddingModelFamily, SpeakerSegmentPrediction,
};

const SAMPLE_RATE: u32 = 16_000;
const DEFAULT_MANIFEST_FILE: &str = "pyannote_diarization_manifest.json";
const DEFAULT_SEGMENTATION_MODEL_FILE: &str = "segmentation.onnx";
const DEFAULT_EMBEDDING_MODEL_FILE: &str = "embedding.onnx";
const DEFAULT_PLDA_TRANSFORM_FILE: &str = "plda_transform.json";
const DEFAULT_PLDA_MODEL_FILE: &str = "plda_model.json";
const DEFAULT_CLUSTERING_CONFIG_FILE: &str = "clustering.json";
const DEFAULT_MODEL_ID: &str = "pyannote/speaker-diarization-community-1";
const PINNED_SOURCE_REVISION: &str = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee";
const APPROVED_ARTIFACT_SET_SHA256: &str =
    "0a12189874dace9b590af9b09ef4637552006130716db57c35d72f984b36c577";
const DEFAULT_LABEL_FORMAT: &str = "SPEAKER_{:02}";
const ACTIVE_THRESHOLD: f32 = 0.5;
// Pinned segmentation-3.0 receptive-field resolution. The converted model
// produces 589 frames for a 10-second chunk. Pyannote places the first frame
// at half the receptive-field duration and advances by this exact step.
const SEGMENTATION_FRAME_STEP_SECONDS: f64 = 0.016_875;
const SEGMENTATION_FRAME_OFFSET_SECONDS: f64 = 0.030_968_75;
// ceil(400 minimum embedding samples / 160_000 window samples * 589 frames)
const EMBEDDING_MIN_CLEAN_FRAMES: usize = 2;

/// Native pyannote community diarizer configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyannoteCommunityDiarizationConfig {
    pub bundle_path: PathBuf,
    pub manifest_file: Option<String>,
    pub segmentation_model_file: Option<String>,
    pub embedding_model_file: Option<String>,
    pub plda_transform_file: Option<String>,
    pub plda_model_file: Option<String>,
    pub clustering_config_file: Option<String>,
    pub min_speakers: Option<usize>,
    pub max_speakers: Option<usize>,
    pub return_speaker_embeddings: bool,
}

/// Native pyannote diarization output plus stable diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PyannoteCommunityDiarizationResult {
    pub response: SpeakerDiarizationResponse,
    pub diagnostics: Vec<String>,
}

/// ONNX-backed pyannote community diarizer.
#[derive(Debug)]
pub struct PyannoteCommunityDiarizer {
    manifest: PyannoteDiarizationManifest,
    segmentation: OnnxSession,
    embedding: OnnxSession,
    plda: plda::Plda,
    clustering: vbx::VbxConfig,
    config: ResolvedPyannoteConfig,
}

#[derive(Debug, Clone)]
struct ResolvedPyannoteConfig {
    bundle_path: PathBuf,
    manifest_path: PathBuf,
    segmentation_model_path: PathBuf,
    embedding_model_path: PathBuf,
    plda_transform_path: PathBuf,
    plda_model_path: PathBuf,
    clustering_config_path: PathBuf,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    return_speaker_embeddings: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyannoteDiarizationManifest {
    schema_version: u32,
    kind: String,
    source: PyannoteSourceManifest,
    artifact_set_sha256: String,
    files: BTreeMap<String, String>,
    #[serde(default = "default_model_id")]
    model_id: String,
    #[serde(default = "default_sample_rate")]
    sample_rate: u32,
    #[serde(default = "default_label_format")]
    label_format: String,
    segmentation: PyannoteSegmentationManifest,
    embedding: PyannoteEmbeddingManifest,
    clustering: PyannoteClusteringManifest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyannoteSourceManifest {
    model_id: String,
    revision: String,
    license: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyannoteSegmentationManifest {
    #[serde(default = "default_segmentation_input_name")]
    input_name: String,
    #[serde(default = "default_segmentation_output_name")]
    output_name: String,
    duration_seconds: f64,
    #[serde(default = "default_step_ratio")]
    step_ratio: f64,
    #[serde(default = "default_true")]
    powerset: bool,
    frames: usize,
    local_speakers: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyannoteEmbeddingManifest {
    #[serde(default = "default_embedding_waveform_input_name")]
    waveform_input_name: String,
    #[serde(default = "default_embedding_mask_input_name")]
    mask_input_name: String,
    #[serde(default = "default_embedding_output_name")]
    output_name: String,
    dimension: usize,
    mask_frames: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PyannoteClusteringManifest {
    #[serde(default = "default_clustering_kind")]
    kind: String,
    #[serde(default = "default_clustering_threshold")]
    threshold: f64,
    #[serde(default = "default_fa")]
    fa: f64,
    #[serde(default = "default_fb")]
    fb: f64,
    #[serde(default = "default_vbx_iters")]
    max_iters: usize,
    #[serde(default = "default_min_active_ratio")]
    min_active_ratio: f64,
    #[serde(default = "default_true")]
    constrained_assignment: bool,
}

#[derive(Debug, Clone)]
struct SegmentationBatch {
    windows: Vec<SegmentationWindow>,
    total_frames: usize,
    audio_duration_seconds: f64,
}

#[derive(Debug, Clone)]
struct SegmentationWindow {
    start_seconds: f64,
    samples: Vec<f32>,
    scores: Vec<f32>,
}

#[derive(Debug, Clone)]
struct LocalSpeakerEmbedding {
    chunk: usize,
    local_speaker: usize,
    clean_frames: usize,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AssignedLocalSpeaker {
    chunk: usize,
    local_speaker: usize,
    speaker: usize,
}

impl PyannoteCommunityDiarizer {
    /// Builds a pyannote community diarizer from a local ONNX bundle.
    pub fn from_config(config: PyannoteCommunityDiarizationConfig) -> Result<Self> {
        let resolved = resolve_config(config)?;
        let manifest = load_manifest(&resolved.manifest_path)?;
        validate_manifest(&manifest)?;
        validate_bundle_files(&resolved, &manifest)?;
        let plda = plda::Plda::load(&resolved.plda_transform_path, &resolved.plda_model_path)?;
        if plda.input_dimension() != manifest.embedding.dimension {
            return Err(setup_error(
                "PLDA input dimension does not match embedding dimension",
            ));
        }
        let clustering = load_clustering(&resolved.clustering_config_path)?;
        vbx::validate_config(&clustering)?;
        validate_clustering_matches_manifest(&clustering, &manifest.clustering)?;
        let segmentation = OnnxSession::from_file_with_options(
            &resolved.segmentation_model_path,
            runtime_onnx::OnnxSessionOptions::default(),
        )
        .map_err(map_onnx_session_error)?;
        validate_segmentation_metadata(
            &segmentation.metadata().map_err(map_onnx_session_error)?,
            &manifest,
        )?;
        let embedding = OnnxSession::from_file_with_options(
            &resolved.embedding_model_path,
            runtime_onnx::OnnxSessionOptions::default(),
        )
        .map_err(map_onnx_session_error)?;
        validate_embedding_metadata(
            &embedding.metadata().map_err(map_onnx_session_error)?,
            &manifest,
        )?;
        Ok(Self {
            manifest,
            segmentation,
            embedding,
            plda,
            clustering,
            config: resolved,
        })
    }

    /// Runs diarization over 16 kHz mono audio.
    pub fn diarize(
        &mut self,
        audio: &SpeakerAudio<'_>,
    ) -> Result<PyannoteCommunityDiarizationResult> {
        if audio.sample_rate() != SAMPLE_RATE || audio.channels() != 1 {
            return Err(DetectError::InvalidArgument(format!(
                "pyannote diarization requires {SAMPLE_RATE} Hz mono audio, got sample_rate={} channels={}",
                audio.sample_rate(),
                audio.channels()
            )));
        }
        validate_speaker_bounds(self.config.min_speakers, self.config.max_speakers)?;

        let samples = audio.samples();
        let segmentations = self.run_segmentations(samples)?;
        let embeddings = self.run_embeddings(&segmentations)?;
        let vbx = vbx::cluster(
            &embeddings
                .iter()
                .map(|embedding| vbx::VbxEmbedding {
                    chunk: embedding.chunk,
                    local_speaker: embedding.local_speaker,
                    clean_frames: embedding.clean_frames,
                    values: &embedding.values,
                })
                .collect::<Vec<_>>(),
            self.manifest.segmentation.frames,
            self.config.min_speakers,
            self.config.max_speakers,
            &self.plda,
            &self.clustering,
        )?;
        let posterior_iterations = vbx.posterior_iterations;
        let training_embeddings = vbx.training_embeddings;
        let automatic_speakers = vbx.automatic_speakers;
        let retained_speakers = vbx.retained_speakers;
        let assignments = vbx
            .assignments
            .into_iter()
            .map(|assignment| AssignedLocalSpeaker {
                chunk: assignment.chunk,
                local_speaker: assignment.local_speaker,
                speaker: assignment.speaker,
            })
            .collect::<Vec<_>>();
        let segments = reconstruct_segments(&self.manifest, &segmentations, &assignments)?;
        let speaker_count = segments
            .iter()
            .map(|segment| segment.speaker.clone())
            .collect::<BTreeSet<_>>()
            .len();
        let speaker_embeddings = if self.config.return_speaker_embeddings {
            Some(speaker_embedding_map(
                &self.manifest,
                &embeddings,
                &assignments,
            )?)
        } else {
            None
        };
        let diagnostics = vec![
            "diarizationProvider=pyannote".to_string(),
            "diarizationRuntime=onnx".to_string(),
            "pyannoteOnnxGraphOptimization=default".to_string(),
            format!("diarizationModel={}", self.manifest.model_id),
            format!("pyannoteSourceRevision={}", self.manifest.source.revision),
            format!("diarizationSpeakerCount={speaker_count}"),
            format!(
                "diarizationMinSpeakers={}",
                format_optional_usize(self.config.min_speakers)
            ),
            format!(
                "diarizationMaxSpeakers={}",
                format_optional_usize(self.config.max_speakers)
            ),
            format!(
                "pyannoteSegmentationWindows={}",
                segmentations.windows.len()
            ),
            format!("pyannoteSegmentationFrames={}", segmentations.total_frames),
            format!("pyannoteEmbeddingCount={}", embeddings.len()),
            format!(
                "pyannoteEmbeddingDimension={}",
                self.manifest.embedding.dimension
            ),
            "pyannoteClustering=vbx".to_string(),
            format!("pyannoteVbxFa={}", self.clustering.fa),
            format!("pyannoteVbxFb={}", self.clustering.fb),
            format!("pyannoteVbxPosteriorIterations={}", posterior_iterations),
            format!("pyannoteVbxTrainingEmbeddings={training_embeddings}"),
            format!("pyannoteVbxAutomaticSpeakers={automatic_speakers}"),
            format!("pyannoteVbxRetainedSpeakers={retained_speakers}"),
            format!(
                "pyannoteArtifactSetSha256={}",
                self.manifest.artifact_set_sha256
            ),
            "pyannotePldaTransform=applied".to_string(),
            "pyannotePldaModel=applied".to_string(),
            "pyannoteConstrainedAssignment=applied".to_string(),
            "speakerEmbeddingProvider=pyannote-onnx".to_string(),
        ];
        Ok(PyannoteCommunityDiarizationResult {
            response: SpeakerDiarizationResponse {
                accepted: true,
                operation: "audio.speakers.diarize".to_string(),
                model_id: self.manifest.model_id.clone(),
                runtime: AudioRuntime::Onnx,
                segments,
                speaker_embeddings,
                diagnostics: diagnostics.clone(),
            },
            diagnostics,
        })
    }

    fn run_segmentations(&mut self, samples: &[f32]) -> Result<SegmentationBatch> {
        let window_samples =
            seconds_to_samples(self.manifest.segmentation.duration_seconds, SAMPLE_RATE)?;
        let step_samples = seconds_to_samples(
            self.manifest.segmentation.duration_seconds * self.manifest.segmentation.step_ratio,
            SAMPLE_RATE,
        )?
        .max(1);
        let mut windows = Vec::new();
        let mut start = 0_usize;
        loop {
            let mut values = vec![0.0_f32; window_samples];
            let end = (start + window_samples).min(samples.len());
            if start < samples.len() {
                values[..end - start].copy_from_slice(&samples[start..end]);
            }
            let tensor = single_f32_input(
                self.manifest.segmentation.input_name.clone(),
                vec![1, 1, window_samples],
                values.clone(),
            )
            .map_err(map_onnx_runtime_error)?;
            let outputs = self
                .segmentation
                .run(vec![tensor])
                .map_err(map_onnx_runtime_error)?;
            let output =
                f32_output_by_name_or_index(&outputs, &self.manifest.segmentation.output_name, 0)
                    .or_else(|_| first_f32_output(&outputs))
                    .map_err(map_onnx_runtime_error)?;
            let scores = segmentation_scores(output, &self.manifest.segmentation)?;
            windows.push(SegmentationWindow {
                start_seconds: start as f64 / SAMPLE_RATE as f64,
                samples: values,
                scores,
            });
            if end >= samples.len() {
                break;
            }
            start += step_samples;
        }
        Ok(SegmentationBatch {
            total_frames: windows.len() * self.manifest.segmentation.frames,
            audio_duration_seconds: samples.len() as f64 / SAMPLE_RATE as f64,
            windows,
        })
    }

    fn run_embeddings(
        &mut self,
        segmentations: &SegmentationBatch,
    ) -> Result<Vec<LocalSpeakerEmbedding>> {
        let mut embeddings = Vec::new();
        for (chunk, window) in segmentations.windows.iter().enumerate() {
            for local_speaker in 0..self.manifest.segmentation.local_speakers {
                let full_mask = speaker_mask(&self.manifest, window, local_speaker);
                let active_frames = full_mask
                    .iter()
                    .filter(|value| **value > ACTIVE_THRESHOLD)
                    .count();
                let clean_mask = clean_speaker_mask(&self.manifest, window, local_speaker);
                let clean_frames = clean_mask
                    .iter()
                    .filter(|value| **value > ACTIVE_THRESHOLD)
                    .count();
                if active_frames == 0 {
                    continue;
                }
                let mask = if clean_frames > EMBEDDING_MIN_CLEAN_FRAMES {
                    clean_mask
                } else {
                    full_mask
                };
                let waveform = single_f32_input(
                    self.manifest.embedding.waveform_input_name.clone(),
                    vec![1, 1, window.samples.len()],
                    window.samples.clone(),
                )
                .map_err(map_onnx_runtime_error)?;
                let masks = single_f32_input(
                    self.manifest.embedding.mask_input_name.clone(),
                    vec![1, self.manifest.embedding.mask_frames],
                    mask,
                )
                .map_err(map_onnx_runtime_error)?;
                let outputs = self
                    .embedding
                    .run(vec![waveform, masks])
                    .map_err(map_onnx_runtime_error)?;
                let output =
                    f32_output_by_name_or_index(&outputs, &self.manifest.embedding.output_name, 0)
                        .or_else(|_| first_f32_output(&outputs))
                        .map_err(map_onnx_runtime_error)?;
                let values = embedding_values(output, self.manifest.embedding.dimension)?;
                if values.iter().all(|value| value.is_finite()) {
                    embeddings.push(LocalSpeakerEmbedding {
                        chunk,
                        local_speaker,
                        clean_frames,
                        values,
                    });
                }
            }
        }
        Ok(embeddings)
    }
}

fn resolve_config(config: PyannoteCommunityDiarizationConfig) -> Result<ResolvedPyannoteConfig> {
    validate_speaker_bounds(config.min_speakers, config.max_speakers)?;
    let bundle_path = config.bundle_path;
    if !bundle_path.is_dir() {
        return Err(setup_error(
            "pyannote diarization bundle does not exist or is not a directory",
        ));
    }
    let manifest_file = config
        .manifest_file
        .as_deref()
        .unwrap_or(DEFAULT_MANIFEST_FILE);
    let segmentation_model_file = config
        .segmentation_model_file
        .as_deref()
        .unwrap_or(DEFAULT_SEGMENTATION_MODEL_FILE);
    let embedding_model_file = config
        .embedding_model_file
        .as_deref()
        .unwrap_or(DEFAULT_EMBEDDING_MODEL_FILE);
    let plda_transform_file = config
        .plda_transform_file
        .as_deref()
        .unwrap_or(DEFAULT_PLDA_TRANSFORM_FILE);
    let plda_model_file = config
        .plda_model_file
        .as_deref()
        .unwrap_or(DEFAULT_PLDA_MODEL_FILE);
    let clustering_config_file = config
        .clustering_config_file
        .as_deref()
        .unwrap_or(DEFAULT_CLUSTERING_CONFIG_FILE);
    let manifest_path = require_file(&bundle_path.join(manifest_file), "manifest")?;
    let segmentation_model_path = require_file(
        &bundle_path.join(segmentation_model_file),
        "segmentation model",
    )?;
    let embedding_model_path =
        require_file(&bundle_path.join(embedding_model_file), "embedding model")?;
    let plda_transform_path =
        require_file(&bundle_path.join(plda_transform_file), "PLDA transform")?;
    let plda_model_path = require_file(&bundle_path.join(plda_model_file), "PLDA model")?;
    let clustering_config_path = require_file(
        &bundle_path.join(clustering_config_file),
        "clustering config",
    )?;
    Ok(ResolvedPyannoteConfig {
        bundle_path,
        manifest_path,
        segmentation_model_path,
        embedding_model_path,
        plda_transform_path,
        plda_model_path,
        clustering_config_path,
        min_speakers: config.min_speakers,
        max_speakers: config.max_speakers,
        return_speaker_embeddings: config.return_speaker_embeddings,
    })
}

fn require_file(path: &Path, role: &str) -> Result<PathBuf> {
    if path.is_file() {
        Ok(path.to_path_buf())
    } else {
        Err(setup_error(format!(
            "pyannote diarization {role} does not exist or is not a file"
        )))
    }
}

fn load_manifest(path: &Path) -> Result<PyannoteDiarizationManifest> {
    let text = fs::read_to_string(path).map_err(|error| {
        setup_error(format!(
            "failed to read pyannote diarization manifest: {error}"
        ))
    })?;
    serde_json::from_str(&text).map_err(|error| {
        DetectError::InvalidArgument(format!(
            "invalid_request: failed to parse pyannote diarization manifest: {error}"
        ))
    })
}

fn validate_manifest(manifest: &PyannoteDiarizationManifest) -> Result<()> {
    if manifest.schema_version != 1
        || manifest.kind != "pyannote-diarization"
        || manifest.model_id != DEFAULT_MODEL_ID
        || manifest.source.model_id != DEFAULT_MODEL_ID
        || manifest.source.revision != PINNED_SOURCE_REVISION
        || manifest.source.license != "CC-BY-4.0"
        || manifest.artifact_set_sha256 != APPROVED_ARTIFACT_SET_SHA256
    {
        return Err(setup_error(
            "pyannote diarization manifest provenance does not match the approved community bundle",
        ));
    }
    if manifest.sample_rate != SAMPLE_RATE {
        return Err(DetectError::InvalidArgument(format!(
            "invalid_request: pyannote diarization manifest sampleRate must be {SAMPLE_RATE}, got {}",
            manifest.sample_rate
        )));
    }
    if manifest.model_id.trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "invalid_request: pyannote diarization manifest modelId must not be empty".to_string(),
        ));
    }
    if manifest.segmentation.duration_seconds <= 0.0
        || !manifest.segmentation.duration_seconds.is_finite()
        || manifest.segmentation.step_ratio <= 0.0
        || !manifest.segmentation.step_ratio.is_finite()
        || manifest.segmentation.frames == 0
        || manifest.segmentation.local_speakers == 0
    {
        return Err(DetectError::InvalidArgument(
            "invalid_request: pyannote diarization segmentation manifest values are invalid"
                .to_string(),
        ));
    }
    if manifest.embedding.dimension == 0 || manifest.embedding.mask_frames == 0 {
        return Err(DetectError::InvalidArgument(
            "invalid_request: pyannote diarization embedding manifest values are invalid"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_bundle_files(
    resolved: &ResolvedPyannoteConfig,
    manifest: &PyannoteDiarizationManifest,
) -> Result<()> {
    let provenance = resolved.bundle_path.join("MODEL_PROVENANCE.md");
    let license = resolved.bundle_path.join("LICENSE.md");
    let required = [
        (
            "segmentation.onnx",
            resolved.segmentation_model_path.as_path(),
        ),
        ("embedding.onnx", resolved.embedding_model_path.as_path()),
        (
            "plda_transform.json",
            resolved.plda_transform_path.as_path(),
        ),
        ("plda_model.json", resolved.plda_model_path.as_path()),
        ("clustering.json", resolved.clustering_config_path.as_path()),
        ("MODEL_PROVENANCE.md", provenance.as_path()),
        ("LICENSE.md", license.as_path()),
    ];
    for (name, path) in required {
        let expected = manifest.files.get(name).ok_or_else(|| {
            setup_error(format!("manifest does not checksum required file `{name}`"))
        })?;
        if !is_sha256(expected) {
            return Err(setup_error(format!(
                "manifest checksum for `{name}` is not SHA-256"
            )));
        }
        let bytes = fs::read(path)
            .map_err(|error| setup_error(format!("failed to read `{name}`: {error}")))?;
        let actual = format!("{:x}", Sha256::digest(bytes));
        if actual != *expected {
            return Err(setup_error(format!("checksum mismatch for `{name}`")));
        }
    }
    let digest = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&manifest.files)
                .map_err(|error| setup_error(format!("failed to hash manifest files: {error}")))?
        )
    );
    if digest != manifest.artifact_set_sha256 {
        return Err(setup_error(
            "artifactSetSha256 does not match the checksummed artifact set",
        ));
    }
    if resolved
        .bundle_path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.starts_with("sha256-")
                && name != format!("sha256-{}", manifest.artifact_set_sha256)
        })
    {
        return Err(setup_error(
            "checksum-addressed snapshot name does not match artifactSetSha256",
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn load_clustering(path: &Path) -> Result<vbx::VbxConfig> {
    serde_json::from_slice(
        &fs::read(path)
            .map_err(|error| setup_error(format!("failed to read clustering config: {error}")))?,
    )
    .map_err(|error| setup_error(format!("failed to parse clustering config: {error}")))
}

fn validate_clustering_matches_manifest(
    clustering: &vbx::VbxConfig,
    manifest: &PyannoteClusteringManifest,
) -> Result<()> {
    let matches = clustering.kind == manifest.kind
        && clustering.threshold == manifest.threshold
        && clustering.fa == manifest.fa
        && clustering.fb == manifest.fb
        && clustering.max_iters == manifest.max_iters
        && clustering.min_active_ratio == manifest.min_active_ratio
        && clustering.constrained_assignment == manifest.constrained_assignment;
    if !matches {
        return Err(setup_error(
            "clustering.json differs from the manifest VBx configuration",
        ));
    }
    Ok(())
}

fn validate_segmentation_metadata(
    metadata: &OnnxSessionMetadata,
    manifest: &PyannoteDiarizationManifest,
) -> Result<()> {
    validate_onnx_io(
        &metadata.inputs,
        &manifest.segmentation.input_name,
        &[
            1,
            1,
            seconds_to_samples(manifest.segmentation.duration_seconds, SAMPLE_RATE)?,
        ],
        "segmentation input",
    )?;
    validate_onnx_io(
        &metadata.outputs,
        &manifest.segmentation.output_name,
        &[
            1,
            manifest.segmentation.frames,
            manifest.segmentation.local_speakers,
        ],
        "segmentation output",
    )
}

fn validate_embedding_metadata(
    metadata: &OnnxSessionMetadata,
    manifest: &PyannoteDiarizationManifest,
) -> Result<()> {
    let window_samples = seconds_to_samples(manifest.segmentation.duration_seconds, SAMPLE_RATE)?;
    validate_onnx_io(
        &metadata.inputs,
        &manifest.embedding.waveform_input_name,
        &[1, 1, window_samples],
        "embedding waveform input",
    )?;
    validate_onnx_io(
        &metadata.inputs,
        &manifest.embedding.mask_input_name,
        &[1, manifest.embedding.mask_frames],
        "embedding mask input",
    )?;
    validate_onnx_io(
        &metadata.outputs,
        &manifest.embedding.output_name,
        &[1, manifest.embedding.dimension],
        "embedding output",
    )
}

fn validate_onnx_io(
    values: &[OnnxIoInfo],
    expected_name: &str,
    expected_shape: &[usize],
    role: &str,
) -> Result<()> {
    let value = values
        .iter()
        .find(|value| value.name == expected_name)
        .ok_or_else(|| {
            setup_error(format!(
                "{role} `{expected_name}` is missing; available names: {}",
                values
                    .iter()
                    .map(|value| value.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
    if value.element_type != Some(OnnxTensorElementType::F32) {
        return Err(setup_error(format!(
            "{role} `{expected_name}` must use F32"
        )));
    }
    let actual = value
        .dimensions
        .iter()
        .map(|dimension| match dimension {
            OnnxDimension::Fixed(value) => Some(*value),
            OnnxDimension::Symbolic(_) | OnnxDimension::Unknown => None,
        })
        .collect::<Vec<_>>();
    if actual.len() != expected_shape.len()
        || actual
            .iter()
            .zip(expected_shape)
            .any(|(actual, expected)| actual.is_some_and(|actual| actual != *expected))
    {
        return Err(setup_error(format!(
            "{role} `{expected_name}` shape {actual:?} does not match {expected_shape:?}"
        )));
    }
    Ok(())
}

fn validate_speaker_bounds(min_speakers: Option<usize>, max_speakers: Option<usize>) -> Result<()> {
    if min_speakers == Some(0) {
        return Err(DetectError::InvalidArgument(
            "invalid_request: pyannote diarization min_speakers must be greater than zero"
                .to_string(),
        ));
    }
    if max_speakers == Some(0) {
        return Err(DetectError::InvalidArgument(
            "invalid_request: pyannote diarization max_speakers must be greater than zero"
                .to_string(),
        ));
    }
    if let (Some(min), Some(max)) = (min_speakers, max_speakers) {
        if min > max {
            return Err(DetectError::InvalidArgument(
                "invalid_request: pyannote diarization min_speakers must be less than or equal to max_speakers"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

fn seconds_to_samples(seconds: f64, sample_rate: u32) -> Result<usize> {
    if seconds <= 0.0 || !seconds.is_finite() {
        return Err(DetectError::InvalidArgument(
            "invalid_request: pyannote diarization window seconds must be finite and positive"
                .to_string(),
        ));
    }
    Ok((seconds * sample_rate as f64).round() as usize)
}

fn segmentation_scores(
    tensor: &OnnxF32Tensor,
    manifest: &PyannoteSegmentationManifest,
) -> Result<Vec<f32>> {
    let expected = manifest.frames * manifest.local_speakers;
    match tensor.shape.as_slice() {
        [1, frames, speakers] if *frames == manifest.frames && *speakers == manifest.local_speakers => {
            Ok(tensor.values.clone())
        }
        [frames, speakers] if *frames == manifest.frames && *speakers == manifest.local_speakers => {
            Ok(tensor.values.clone())
        }
        [1, values] if *values == expected => Ok(tensor.values.clone()),
        [values] if *values == expected => Ok(tensor.values.clone()),
        shape => Err(DetectError::InvalidArgument(format!(
            "model_output_mismatch: pyannote segmentation output shape {shape:?} does not match expected [1, {}, {}]",
            manifest.frames, manifest.local_speakers
        ))),
    }
}

fn embedding_values(tensor: &OnnxF32Tensor, dimension: usize) -> Result<Vec<f32>> {
    match tensor.shape.as_slice() {
        [dim] if *dim == dimension => Ok(tensor.values.clone()),
        [1, dim] if *dim == dimension => Ok(tensor.values.clone()),
        shape => Err(DetectError::InvalidArgument(format!(
            "model_output_mismatch: pyannote embedding output shape {shape:?} does not match expected [1, {dimension}]"
        ))),
    }
}

fn speaker_mask(
    manifest: &PyannoteDiarizationManifest,
    window: &SegmentationWindow,
    local_speaker: usize,
) -> Vec<f32> {
    let mut mask = vec![0.0_f32; manifest.embedding.mask_frames];
    let copy_frames = manifest
        .segmentation
        .frames
        .min(manifest.embedding.mask_frames);
    for (frame, value) in mask.iter_mut().enumerate().take(copy_frames) {
        let score = window.scores[frame * manifest.segmentation.local_speakers + local_speaker];
        *value = if manifest.segmentation.powerset {
            score
        } else if score > ACTIVE_THRESHOLD {
            1.0
        } else {
            0.0
        };
    }
    mask
}

fn clean_speaker_mask(
    manifest: &PyannoteDiarizationManifest,
    window: &SegmentationWindow,
    local_speaker: usize,
) -> Vec<f32> {
    let mut mask = vec![0.0; manifest.embedding.mask_frames];
    for (frame, output) in mask
        .iter_mut()
        .enumerate()
        .take(manifest.segmentation.frames)
    {
        let mut active = 0_usize;
        let mut selected_active = false;
        for speaker in 0..manifest.segmentation.local_speakers {
            let score = window.scores[frame * manifest.segmentation.local_speakers + speaker];
            if score > ACTIVE_THRESHOLD {
                active += 1;
                if speaker == local_speaker {
                    selected_active = true;
                }
            }
        }
        if active == 1 && selected_active {
            *output = window.scores[frame * manifest.segmentation.local_speakers + local_speaker];
        }
    }
    mask
}

fn reconstruct_segments(
    manifest: &PyannoteDiarizationManifest,
    segmentations: &SegmentationBatch,
    assignments: &[AssignedLocalSpeaker],
) -> Result<Vec<SpeakerSegmentPrediction>> {
    let frame_duration = SEGMENTATION_FRAME_STEP_SECONDS;
    let mut by_local = BTreeMap::new();
    for assignment in assignments {
        by_local.insert((assignment.chunk, assignment.local_speaker), assignment);
    }
    // Stitch the overlapping segmentation windows onto one global frame grid.
    // Emitting each window independently creates duplicate turns and contradicts
    // pyannote's overlap aggregation contract.
    let mut aggregate = BTreeMap::<(usize, usize), (f64, usize)>::new();
    let mut speaker_count = BTreeMap::<usize, (f64, usize)>::new();
    for (chunk, window) in segmentations.windows.iter().enumerate() {
        for frame in 0..manifest.segmentation.frames {
            let frame_center = window.start_seconds
                + SEGMENTATION_FRAME_OFFSET_SECONDS
                + frame as f64 * frame_duration;
            let global_frame = ((frame_center - SEGMENTATION_FRAME_OFFSET_SECONDS) / frame_duration)
                .round() as usize;
            let mut window_speakers = BTreeMap::<usize, f32>::new();
            let mut local_count = 0.0_f64;
            for local_speaker in 0..manifest.segmentation.local_speakers {
                let score =
                    window.scores[frame * manifest.segmentation.local_speakers + local_speaker];
                local_count += f64::from(score);
                if let Some(assignment) = by_local.get(&(chunk, local_speaker)) {
                    window_speakers
                        .entry(assignment.speaker)
                        .and_modify(|current| *current = current.max(score))
                        .or_insert(score);
                }
            }
            for (speaker, score) in window_speakers {
                aggregate
                    .entry((global_frame, speaker))
                    .and_modify(|(sum, count)| {
                        *sum += f64::from(score);
                        *count += 1;
                    })
                    .or_insert((f64::from(score), 1));
            }
            speaker_count
                .entry(global_frame)
                .and_modify(|(sum, count)| {
                    *sum += local_count;
                    *count += 1;
                })
                .or_insert((local_count, 1));
        }
    }
    let total_frames = ((segmentations.audio_duration_seconds - SEGMENTATION_FRAME_OFFSET_SECONDS)
        / frame_duration)
        .ceil()
        .max(0.0) as usize;
    let speakers = assignments
        .iter()
        .map(|assignment| assignment.speaker)
        .collect::<BTreeSet<_>>();
    let mut segments = Vec::new();
    let mut open = BTreeMap::<usize, (f64, f64, f32)>::new();
    for frame in 0..total_frames {
        let frame_start = SEGMENTATION_FRAME_OFFSET_SECONDS + frame as f64 * frame_duration;
        let frame_end = (frame_start + frame_duration).min(segmentations.audio_duration_seconds);
        let count = speaker_count
            .get(&frame)
            .map(|(sum, observations)| (*sum / *observations as f64).round() as usize)
            .unwrap_or(0)
            .min(speakers.len());
        let mut ranked = speakers
            .iter()
            .filter_map(|speaker| {
                // Pyannote reconstruction deliberately uses overlap-add sums
                // (`skip_average=true`) before ranking speakers against the
                // independently averaged instantaneous speaker count.
                aggregate
                    .get(&(frame, *speaker))
                    .map(|(sum, observations)| {
                        (*speaker, *sum as f32, (*sum / *observations as f64) as f32)
                    })
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|(left_speaker, left, _), (right_speaker, right, _)| {
            right
                .partial_cmp(left)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_speaker.cmp(right_speaker))
        });
        let active = ranked
            .into_iter()
            .take(count)
            .map(|(speaker, _, score)| (speaker, score))
            .collect::<BTreeMap<_, _>>();
        let closing = open
            .keys()
            .copied()
            .filter(|speaker| !active.contains_key(speaker))
            .collect::<Vec<_>>();
        for speaker in closing {
            if let Some((start, end, score)) = open.remove(&speaker) {
                push_segment(&mut segments, manifest, speaker, start, end, score)?;
            }
        }
        for (speaker, score) in active {
            open.entry(speaker)
                .and_modify(|entry| {
                    entry.1 = frame_end;
                    entry.2 = entry.2.max(score);
                })
                .or_insert((frame_start, frame_end, score));
        }
    }
    for (speaker, (start, end, score)) in open {
        push_segment(&mut segments, manifest, speaker, start, end, score)?;
    }
    segments.sort_by(|left, right| {
        left.start_seconds
            .partial_cmp(&right.start_seconds)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.speaker.cmp(&right.speaker))
    });
    Ok(segments)
}

fn push_segment(
    segments: &mut Vec<SpeakerSegmentPrediction>,
    manifest: &PyannoteDiarizationManifest,
    speaker: usize,
    start: f64,
    end: f64,
    score: f32,
) -> Result<()> {
    if end <= start {
        return Ok(());
    }
    let label = speaker_label(manifest, speaker);
    if let Some(last) = segments.last_mut() {
        if last.speaker == label && (last.end_seconds as f64 - start).abs() <= 0.02 {
            last.end_seconds = end as f32;
            last.score = Some(last.score.unwrap_or(score).max(score));
            return Ok(());
        }
    }
    segments.push(SpeakerSegmentPrediction {
        speaker: label,
        start_seconds: start as f32,
        end_seconds: end as f32,
        score: Some(score),
    });
    Ok(())
}

fn speaker_embedding_map(
    manifest: &PyannoteDiarizationManifest,
    embeddings: &[LocalSpeakerEmbedding],
    assignments: &[AssignedLocalSpeaker],
) -> Result<BTreeMap<String, SpeakerEmbedding>> {
    let model = SpeakerEmbeddingModel {
        family: SpeakerEmbeddingModelFamily::Pyannote,
        name: manifest.model_id.clone(),
        version: "community-1".to_string(),
        dimensions: manifest.embedding.dimension,
    };
    let mut grouped: BTreeMap<usize, Vec<&LocalSpeakerEmbedding>> = BTreeMap::new();
    for assignment in assignments {
        if let Some(embedding) = embeddings.iter().find(|item| {
            item.chunk == assignment.chunk && item.local_speaker == assignment.local_speaker
        }) {
            grouped
                .entry(assignment.speaker)
                .or_default()
                .push(embedding);
        }
    }
    let mut map = BTreeMap::new();
    for (speaker, items) in grouped {
        let mut values = vec![0.0_f32; manifest.embedding.dimension];
        for item in &items {
            for (index, value) in item.values.iter().enumerate().take(values.len()) {
                values[index] += *value;
            }
        }
        for value in &mut values {
            *value /= items.len() as f32;
        }
        map.insert(
            speaker_label(manifest, speaker),
            SpeakerEmbedding::new(values, model.clone(), SAMPLE_RATE)?,
        );
    }
    Ok(map)
}

fn speaker_label(manifest: &PyannoteDiarizationManifest, speaker: usize) -> String {
    if manifest.label_format == DEFAULT_LABEL_FORMAT {
        format!("SPEAKER_{speaker:02}")
    } else {
        manifest
            .label_format
            .replace("{:02}", &format!("{speaker:02}"))
            .replace("{}", &speaker.to_string())
    }
}

fn format_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn setup_error(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("setup_error: {}", message.into()))
}

fn map_onnx_session_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::InvalidArgument(_) => {
            setup_error("failed to initialize pyannote ONNX model")
        }
        runtime_onnx::OnnxRuntimeError::Io(_) => setup_error("failed to read pyannote ONNX model"),
        _ => DetectError::InvalidArgument(
            "unsupported_runtime: failed to initialize pyannote ONNX model".to_string(),
        ),
    }
}

fn map_onnx_runtime_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    DetectError::InvalidArgument(format!("model_output_mismatch: {error}"))
}

fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}

fn default_sample_rate() -> u32 {
    SAMPLE_RATE
}

fn default_label_format() -> String {
    DEFAULT_LABEL_FORMAT.to_string()
}

fn default_segmentation_input_name() -> String {
    "waveform".to_string()
}

fn default_segmentation_output_name() -> String {
    "segmentations".to_string()
}

fn default_step_ratio() -> f64 {
    0.1
}

fn default_true() -> bool {
    true
}

fn default_embedding_waveform_input_name() -> String {
    "waveform".to_string()
}

fn default_embedding_mask_input_name() -> String {
    "masks".to_string()
}

fn default_embedding_output_name() -> String {
    "embeddings".to_string()
}

fn default_clustering_kind() -> String {
    "vbx".to_string()
}

fn default_clustering_threshold() -> f64 {
    0.6
}

fn default_fa() -> f64 {
    0.07
}

fn default_fb() -> f64 {
    0.8
}

fn default_vbx_iters() -> usize {
    20
}

fn default_min_active_ratio() -> f64 {
    0.2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> PyannoteDiarizationManifest {
        PyannoteDiarizationManifest {
            schema_version: 1,
            kind: "pyannote-diarization".to_string(),
            source: PyannoteSourceManifest {
                model_id: DEFAULT_MODEL_ID.to_string(),
                revision: PINNED_SOURCE_REVISION.to_string(),
                license: "CC-BY-4.0".to_string(),
            },
            artifact_set_sha256: APPROVED_ARTIFACT_SET_SHA256.to_string(),
            files: BTreeMap::new(),
            model_id: DEFAULT_MODEL_ID.to_string(),
            sample_rate: SAMPLE_RATE,
            label_format: DEFAULT_LABEL_FORMAT.to_string(),
            segmentation: PyannoteSegmentationManifest {
                input_name: "waveform".to_string(),
                output_name: "segmentations".to_string(),
                duration_seconds: 1.0,
                step_ratio: 1.0,
                powerset: true,
                frames: 4,
                local_speakers: 2,
            },
            embedding: PyannoteEmbeddingManifest {
                waveform_input_name: "waveform".to_string(),
                mask_input_name: "masks".to_string(),
                output_name: "embeddings".to_string(),
                dimension: 2,
                mask_frames: 4,
            },
            clustering: PyannoteClusteringManifest {
                kind: "vbx".to_string(),
                threshold: 0.6,
                fa: 0.07,
                fb: 0.8,
                max_iters: 20,
                min_active_ratio: 0.2,
                constrained_assignment: true,
            },
        }
    }

    #[test]
    fn manifest_validation_rejects_wrong_sample_rate() {
        let mut manifest = manifest();
        manifest.sample_rate = 8_000;

        let error = validate_manifest(&manifest).expect_err("invalid sample rate");

        assert!(error.to_string().contains("sampleRate"));
    }

    #[test]
    fn manifest_validation_rejects_revision_and_artifact_digest_mismatches() {
        let mut wrong_revision = manifest();
        wrong_revision.source.revision = "unapproved-revision".to_string();
        assert!(validate_manifest(&wrong_revision)
            .expect_err("wrong source revision must fail before inference")
            .to_string()
            .contains("provenance"));

        let mut wrong_digest = manifest();
        wrong_digest.artifact_set_sha256 = "0".repeat(64);
        assert!(validate_manifest(&wrong_digest)
            .expect_err("wrong artifact digest must fail before inference")
            .to_string()
            .contains("provenance"));
    }

    #[test]
    fn speaker_bounds_reject_zero_and_inverted_range() {
        assert!(validate_speaker_bounds(Some(0), None)
            .unwrap_err()
            .to_string()
            .contains("min_speakers"));
        assert!(validate_speaker_bounds(Some(3), Some(2))
            .unwrap_err()
            .to_string()
            .contains("less than or equal"));
    }

    #[test]
    fn speaker_mask_uses_powerset_scores() {
        let manifest = manifest();
        let window = SegmentationWindow {
            start_seconds: 0.0,
            samples: vec![0.0; SAMPLE_RATE as usize],
            scores: vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        };

        assert_eq!(
            speaker_mask(&manifest, &window, 1),
            vec![0.9, 0.8, 0.7, 0.6]
        );
    }

    #[test]
    fn reconstruct_segments_emits_speaker_labels() {
        let manifest = manifest();
        let segmentations = SegmentationBatch {
            total_frames: 4,
            audio_duration_seconds: 1.0,
            windows: vec![SegmentationWindow {
                start_seconds: 0.0,
                samples: vec![0.0; SAMPLE_RATE as usize],
                scores: vec![0.9, 0.1, 0.8, 0.1, 0.1, 0.9, 0.1, 0.8],
            }],
        };
        let assignments = vec![
            AssignedLocalSpeaker {
                chunk: 0,
                local_speaker: 0,
                speaker: 0,
            },
            AssignedLocalSpeaker {
                chunk: 0,
                local_speaker: 1,
                speaker: 1,
            },
        ];

        let segments = reconstruct_segments(&manifest, &segmentations, &assignments).unwrap();

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert_eq!(segments[1].speaker, "SPEAKER_01");
    }

    fn io(name: &str, dimensions: &[usize]) -> OnnxIoInfo {
        OnnxIoInfo {
            name: name.to_string(),
            element_type: Some(OnnxTensorElementType::F32),
            dimensions: dimensions
                .iter()
                .copied()
                .map(OnnxDimension::Fixed)
                .collect(),
        }
    }

    #[test]
    fn segmentation_metadata_requires_three_dimensional_waveform() {
        let manifest = manifest();
        let valid = OnnxSessionMetadata {
            inputs: vec![io("waveform", &[1, 1, 16_000])],
            outputs: vec![io("segmentations", &[1, 4, 2])],
        };
        validate_segmentation_metadata(&valid, &manifest).unwrap();

        let mut rank_two = valid;
        rank_two.inputs[0] = io("waveform", &[1, 16_000]);
        let error = validate_segmentation_metadata(&rank_two, &manifest).unwrap_err();
        assert!(error.to_string().contains("shape"), "{error}");
    }

    #[test]
    fn embedding_metadata_requires_waveform_mask_and_embedding_contract() {
        let manifest = manifest();
        let valid = OnnxSessionMetadata {
            inputs: vec![io("waveform", &[1, 1, 16_000]), io("masks", &[1, 4])],
            outputs: vec![io("embeddings", &[1, 2])],
        };
        validate_embedding_metadata(&valid, &manifest).unwrap();

        let mut wrong_mask = valid.clone();
        wrong_mask.inputs[1] = io("masks", &[1, 3]);
        assert!(validate_embedding_metadata(&wrong_mask, &manifest).is_err());

        let mut wrong_type = valid;
        wrong_type.outputs[0].element_type = Some(OnnxTensorElementType::I64);
        assert!(validate_embedding_metadata(&wrong_type, &manifest).is_err());
    }
}
