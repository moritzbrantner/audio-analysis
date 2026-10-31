//! Windowed speaker diarization and window clustering.

use audio_contracts::{DetectError, Result};

use crate::{
    SpeakerAudio, SpeakerEmbedding, SpeakerEmbeddingExtractor, SpeakerId,
    SpeakerIdentificationOptions, SpeakerLibrary, SpeechSpan, VadOptions, VoiceActivityDetector,
};

/// Trait for speaker diarization implementations.
pub trait SpeakerDiarizer {
    /// Diarizes speaker activity.
    fn diarize(&mut self, audio: &SpeakerAudio<'_>) -> Result<DiarizationResult>;
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Speaker diarization result.
pub struct DiarizationResult {
    /// Diarization segments.
    pub segments: Vec<DiarizationSegment>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Diarization segment.
pub struct DiarizationSegment {
    /// Start in seconds.
    pub start_seconds: f64,
    /// End in seconds.
    pub end_seconds: f64,
    /// Speaker assignment.
    pub speaker: DiarizedSpeaker,
    /// Segment confidence or cluster score.
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Diarized speaker assignment.
pub enum DiarizedSpeaker {
    /// Enrolled speaker.
    Known(SpeakerId),
    /// Unenrolled cluster label.
    Unknown(String),
    /// Speech span too short to attribute to any speaker. Its embedding is
    /// neither computed nor clustered, so it never creates a new speaker.
    Unresolved,
}

#[derive(Debug, Clone, PartialEq)]
/// Simple VAD/window/embed/cluster diarizer.
pub struct WindowedSpeakerDiarizer<E, V> {
    /// Speaker embedding extractor.
    pub extractor: E,
    /// Voice activity detector.
    pub vad: V,
    /// Options passed to the voice activity detector.
    pub vad_options: VadOptions,
    /// Optional enrolled speaker library.
    pub library: Option<SpeakerLibrary>,
    /// Identification options for mapping clusters to known speakers.
    pub identification_options: SpeakerIdentificationOptions,
    /// Average-linkage cosine similarity at or above which agglomerative
    /// clustering keeps merging unknown speaker clusters.
    pub cluster_threshold: f32,
    /// Longest audio window embedded at once. Longer speech spans are split
    /// into equal windows so a turn change inside one span stays separable.
    pub embedding_window_seconds: f64,
    /// Speech a cluster needs before it counts as its own speaker: the larger
    /// of this many seconds and [`Self::min_cluster_fraction`] of all clustered
    /// speech. Thinner clusters are folded into their nearest supported
    /// cluster. If no cluster reaches it, every cluster is kept.
    pub min_cluster_seconds: f64,
    /// Share of all clustered speech a cluster needs; see
    /// [`Self::min_cluster_seconds`].
    pub min_cluster_fraction: f64,
    /// Minimum number of unknown speaker clusters to produce when enough
    /// unknown speech windows are available.
    pub min_speakers: Option<usize>,
    /// Maximum number of unknown speaker clusters to produce.
    pub max_speakers: Option<usize>,
    /// Minimum speech span duration, in seconds, required before a span is
    /// embedded and attributed. Shorter spans are returned as
    /// [`DiarizedSpeaker::Unresolved`] with their observation times intact.
    pub min_speaker_evidence_seconds: f64,
}

/// Default [`WindowedSpeakerDiarizer::min_speaker_evidence_seconds`].
///
/// Tuned on real multi-speaker conversations (issue #152): 0.5 s left a third
/// of genuine sub-second turns and ~10% of speech unresolved on word-timed
/// transcript spans, while spans under ~0.1 s yield non-finite WeSpeaker
/// embeddings. 0.2 s keeps most short turns and still withholds tails.
pub const DEFAULT_MIN_SPEAKER_EVIDENCE_SECONDS: f64 = 0.2;

/// Default [`WindowedSpeakerDiarizer::cluster_threshold`].
///
/// Tuned with WeSpeaker ResNet34-LM on word-timed transcript spans (issue
/// #204): the speaker count is right on the pyannote `sample.wav` (2) and AMI
/// `EN2002c` 0–300 s (3) for thresholds 0.2–0.3.
pub const DEFAULT_CLUSTER_THRESHOLD: f32 = 0.25;

/// Default [`WindowedSpeakerDiarizer::embedding_window_seconds`].
pub const DEFAULT_EMBEDDING_WINDOW_SECONDS: f64 = 3.0;

/// Default [`WindowedSpeakerDiarizer::min_cluster_seconds`].
pub const DEFAULT_MIN_CLUSTER_SECONDS: f64 = 2.0;

/// Default [`WindowedSpeakerDiarizer::min_cluster_fraction`].
pub const DEFAULT_MIN_CLUSTER_FRACTION: f64 = 0.05;

impl<E, V> WindowedSpeakerDiarizer<E, V> {
    /// Creates a new value.
    pub fn new(extractor: E, vad: V) -> Self {
        Self {
            extractor,
            vad,
            vad_options: VadOptions::unpadded(),
            library: None,
            identification_options: SpeakerIdentificationOptions::default(),
            cluster_threshold: DEFAULT_CLUSTER_THRESHOLD,
            embedding_window_seconds: DEFAULT_EMBEDDING_WINDOW_SECONDS,
            min_cluster_seconds: DEFAULT_MIN_CLUSTER_SECONDS,
            min_cluster_fraction: DEFAULT_MIN_CLUSTER_FRACTION,
            min_speakers: None,
            max_speakers: None,
            min_speaker_evidence_seconds: DEFAULT_MIN_SPEAKER_EVIDENCE_SECONDS,
        }
    }

    /// Returns this value with the minimum span duration required to attribute
    /// a speaker. `0.0` attributes every span.
    pub fn min_speaker_evidence_seconds(mut self, seconds: f64) -> Result<Self> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(DetectError::InvalidArgument(
                "speaker diarization min_speaker_evidence_seconds must be finite and non-negative"
                    .to_string(),
            ));
        }
        self.min_speaker_evidence_seconds = seconds;
        Ok(self)
    }

    /// Returns this value with voice activity detection options.
    pub fn vad_options(mut self, vad_options: VadOptions) -> Result<Self> {
        vad_options.validate()?;
        self.vad_options = vad_options;
        Ok(self)
    }

    /// Returns this value with a speaker library.
    pub fn library(mut self, library: SpeakerLibrary) -> Self {
        self.library = Some(library);
        self
    }

    /// Returns this value with cluster threshold.
    pub fn cluster_threshold(mut self, cluster_threshold: f32) -> Result<Self> {
        if !cluster_threshold.is_finite() || !(-1.0..=1.0).contains(&cluster_threshold) {
            return Err(DetectError::InvalidArgument(
                "speaker diarization cluster threshold must be finite and between -1.0 and 1.0"
                    .to_string(),
            ));
        }
        self.cluster_threshold = cluster_threshold;
        Ok(self)
    }

    /// Returns this value with the longest embedded window. Use
    /// `f64::INFINITY` to embed every span whole.
    pub fn embedding_window_seconds(mut self, seconds: f64) -> Result<Self> {
        if seconds.is_nan() || seconds <= 0.0 {
            return Err(DetectError::InvalidArgument(
                "speaker diarization embedding_window_seconds must be positive".to_string(),
            ));
        }
        self.embedding_window_seconds = seconds;
        Ok(self)
    }

    /// Returns this value with the speech a cluster needs to count as its own
    /// speaker. `(0.0, 0.0)` keeps every cluster.
    pub fn min_cluster_speech(mut self, seconds: f64, fraction: f64) -> Result<Self> {
        if !seconds.is_finite() || seconds < 0.0 || !(0.0..=1.0).contains(&fraction) {
            return Err(DetectError::InvalidArgument(
                "speaker diarization min cluster speech needs finite non-negative seconds and a fraction in 0..=1"
                    .to_string(),
            ));
        }
        self.min_cluster_seconds = seconds;
        self.min_cluster_fraction = fraction;
        Ok(self)
    }

    /// Returns this value with unknown-speaker cluster bounds.
    pub fn speaker_bounds(
        mut self,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<Self> {
        validate_speaker_bounds(min_speakers, max_speakers)?;
        self.min_speakers = min_speakers;
        self.max_speakers = max_speakers;
        Ok(self)
    }
}

impl<E, V> SpeakerDiarizer for WindowedSpeakerDiarizer<E, V>
where
    E: SpeakerEmbeddingExtractor,
    V: VoiceActivityDetector,
{
    fn diarize(&mut self, audio: &SpeakerAudio<'_>) -> Result<DiarizationResult> {
        let spans = self.vad.detect_speech(audio, &self.vad_options)?.segments;
        let mono = audio.to_mono()?;
        let mut windows = Vec::new();
        let mut unresolved = Vec::new();

        for (source, span) in spans.into_iter().enumerate() {
            let duration = span.end_seconds - span.start_seconds;
            if duration < self.min_speaker_evidence_seconds {
                unresolved.push(span);
                continue;
            }
            let pieces = (duration / self.embedding_window_seconds).ceil().max(1.0) as usize;
            let step = duration / pieces as f64;
            for piece in 0..pieces {
                let start_seconds = span.start_seconds + piece as f64 * step;
                let end_seconds = if piece + 1 == pieces {
                    span.end_seconds
                } else {
                    start_seconds + step
                };
                let start = (start_seconds * audio.sample_rate() as f64).round() as usize;
                let end = (end_seconds * audio.sample_rate() as f64).round() as usize;
                let segment_samples = &mono[start.min(mono.len())..end.min(mono.len())];
                if segment_samples.is_empty() {
                    continue;
                }
                let segment_audio = SpeakerAudio::mono(segment_samples, audio.sample_rate())?;
                let embedding = self.extractor.embed_speaker(&segment_audio)?;

                let known = self
                    .library
                    .as_ref()
                    .map(|library| library.identify(&embedding, &self.identification_options))
                    .transpose()?
                    .and_then(|result| result.best_match.map(|value| value.speaker_id));

                windows.push(DiarizationWindow {
                    source,
                    span: SpeechSpan {
                        start_seconds,
                        end_seconds,
                        score: span.score,
                    },
                    embedding,
                    known,
                });
            }
        }

        let unknown_labels = cluster_unknown_windows(
            &windows,
            ClusterOptions {
                threshold: self.cluster_threshold,
                min_speakers: self.min_speakers,
                max_speakers: self.max_speakers,
                min_cluster_seconds: self.min_cluster_seconds,
                min_cluster_fraction: self.min_cluster_fraction,
            },
        )?;
        let mut segments: Vec<DiarizationSegment> = Vec::new();
        let mut last_source = None;
        for (index, window) in windows.into_iter().enumerate() {
            let speaker = if let Some(speaker_id) = window.known {
                DiarizedSpeaker::Known(speaker_id)
            } else {
                DiarizedSpeaker::Unknown(
                    unknown_labels[index]
                        .clone()
                        .expect("unknown window has clustered label"),
                )
            };
            // Rejoin windows of one span that landed on the same speaker.
            if last_source == Some(window.source) {
                if let Some(previous) = segments.last_mut().filter(|last| last.speaker == speaker) {
                    previous.end_seconds = window.span.end_seconds;
                    continue;
                }
            }
            last_source = Some(window.source);
            segments.push(DiarizationSegment {
                start_seconds: window.span.start_seconds,
                end_seconds: window.span.end_seconds,
                speaker,
                score: window.span.score,
            });
        }
        segments.extend(unresolved.into_iter().map(|span| DiarizationSegment {
            start_seconds: span.start_seconds,
            end_seconds: span.end_seconds,
            speaker: DiarizedSpeaker::Unresolved,
            score: span.score,
        }));
        segments.sort_by(|left, right| left.start_seconds.total_cmp(&right.start_seconds));

        Ok(DiarizationResult { segments })
    }
}

#[derive(Debug, Clone)]
struct DiarizationWindow {
    /// Index of the VAD span this window was cut from.
    source: usize,
    span: SpeechSpan,
    embedding: SpeakerEmbedding,
    known: Option<SpeakerId>,
}

pub(crate) fn validate_speaker_bounds(
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
) -> Result<()> {
    if min_speakers == Some(0) {
        return Err(DetectError::InvalidArgument(
            "speaker diarization min_speakers must be greater than zero".to_string(),
        ));
    }
    if max_speakers == Some(0) {
        return Err(DetectError::InvalidArgument(
            "speaker diarization max_speakers must be greater than zero".to_string(),
        ));
    }
    if let (Some(min), Some(max)) = (min_speakers, max_speakers) {
        if min > max {
            return Err(DetectError::InvalidArgument(
                "speaker diarization min_speakers must be less than or equal to max_speakers"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

/// Speaker-count controls for [`cluster_unknown_windows`].
#[derive(Debug, Clone, Copy)]
struct ClusterOptions {
    threshold: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    min_cluster_seconds: f64,
    min_cluster_fraction: f64,
}

/// Labels unknown windows with average-linkage agglomerative clustering on
/// cosine similarity, then folds clusters with too little speech into the
/// nearest well-supported cluster.
fn cluster_unknown_windows(
    windows: &[DiarizationWindow],
    options: ClusterOptions,
) -> Result<Vec<Option<String>>> {
    validate_speaker_bounds(options.min_speakers, options.max_speakers)?;
    let unknown = windows
        .iter()
        .enumerate()
        .filter_map(|(index, window)| window.known.is_none().then_some(index))
        .collect::<Vec<_>>();
    let count = unknown.len();
    let mut labels = vec![None; windows.len()];
    if count == 0 {
        return Ok(labels);
    }

    let mut similarity = vec![vec![1.0_f32; count]; count];
    for left in 0..count {
        for right in (left + 1)..count {
            let score = windows[unknown[left]]
                .embedding
                .cosine_similarity(&windows[unknown[right]].embedding)?;
            similarity[left][right] = score;
            similarity[right][left] = score;
        }
    }

    // Agglomerate; `linkage` holds the average similarity between live clusters.
    let min_clusters = options.min_speakers.unwrap_or(1).min(count);
    let max_clusters = options.max_speakers.unwrap_or(count);
    let mut linkage = similarity.clone();
    let mut members = (0..count).map(|index| vec![index]).collect::<Vec<_>>();
    let mut live = count;
    while live > min_clusters {
        let mut best: Option<(f32, usize, usize)> = None;
        for left in (0..count).filter(|index| !members[*index].is_empty()) {
            for right in ((left + 1)..count).filter(|index| !members[*index].is_empty()) {
                if best.is_none_or(|(score, _, _)| linkage[left][right] > score) {
                    best = Some((linkage[left][right], left, right));
                }
            }
        }
        let Some((score, left, right)) = best else {
            break;
        };
        if score < options.threshold && live <= max_clusters {
            break;
        }
        let left_size = members[left].len() as f32;
        let right_size = members[right].len() as f32;
        for other in (0..count).filter(|index| !members[*index].is_empty()) {
            let merged = (linkage[left][other] * left_size + linkage[right][other] * right_size)
                / (left_size + right_size);
            linkage[left][other] = merged;
            linkage[other][left] = merged;
        }
        let moved = std::mem::take(&mut members[right]);
        members[left].extend(moved);
        live -= 1;
    }
    let mut clusters = members
        .into_iter()
        .filter(|cluster| !cluster.is_empty())
        .collect::<Vec<_>>();

    // Fold clusters without enough speech into their nearest supported cluster.
    let seconds = |cluster: &Vec<usize>| {
        cluster
            .iter()
            .map(|index| {
                let span = &windows[unknown[*index]].span;
                span.end_seconds - span.start_seconds
            })
            .sum::<f64>()
    };
    let total = clusters.iter().map(seconds).sum::<f64>();
    let required = options
        .min_cluster_seconds
        .max(options.min_cluster_fraction * total);
    clusters.sort_by(|left, right| seconds(right).total_cmp(&seconds(left)));
    let supported = clusters
        .iter()
        .take_while(|cluster| seconds(cluster) >= required)
        .count();
    let keep = supported.max(min_clusters);
    if supported > 0 && keep < clusters.len() {
        let orphans = clusters.split_off(keep).concat();
        let targets = clusters.clone();
        for orphan in orphans {
            let nearest = targets
                .iter()
                .map(|cluster| {
                    cluster
                        .iter()
                        .map(|member| similarity[orphan][*member])
                        .sum::<f32>()
                        / cluster.len() as f32
                })
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .map(|(index, _)| index)
                .expect("supported clusters are non-empty");
            clusters[nearest].push(orphan);
        }
    }

    let mut cluster_for_window = vec![0; count];
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        for member in cluster {
            cluster_for_window[*member] = cluster_index;
        }
    }
    let mut first_seen = Vec::new();
    for (local, window_index) in unknown.iter().enumerate() {
        let cluster_index = cluster_for_window[local];
        let label_index = match first_seen.iter().position(|value| *value == cluster_index) {
            Some(existing) => existing,
            None => {
                first_seen.push(cluster_index);
                first_seen.len() - 1
            }
        };
        labels[*window_index] = Some(format!("speaker_{label_index}"));
    }
    Ok(labels)
}
