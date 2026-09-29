//! Deterministic VBx clustering for pyannote community diarization.

use std::collections::BTreeMap;

use audio_contracts::{DetectError, Result};
use serde::Deserialize;

use super::plda::Plda;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct VbxConfig {
    pub(super) kind: String,
    pub(super) threshold: f64,
    pub(super) fa: f64,
    pub(super) fb: f64,
    pub(super) max_iters: usize,
    pub(super) min_active_ratio: f64,
    pub(super) constrained_assignment: bool,
}

#[derive(Debug, Clone)]
pub(super) struct VbxEmbedding<'a> {
    pub(super) chunk: usize,
    pub(super) local_speaker: usize,
    pub(super) clean_frames: usize,
    pub(super) values: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct VbxAssignment {
    pub(super) chunk: usize,
    pub(super) local_speaker: usize,
    pub(super) speaker: usize,
}

#[derive(Debug, Clone)]
pub(super) struct VbxResult {
    pub(super) assignments: Vec<VbxAssignment>,
    pub(super) posterior_iterations: usize,
    pub(super) training_embeddings: usize,
    pub(super) automatic_speakers: usize,
    pub(super) retained_speakers: usize,
}

pub(super) fn validate_config(value: &VbxConfig) -> Result<()> {
    if value.kind != "vbx"
        || !value.threshold.is_finite()
        || value.threshold <= 0.0
        || !value.fa.is_finite()
        || value.fa <= 0.0
        || !value.fb.is_finite()
        || value.fb <= 0.0
        || value.max_iters == 0
        || !value.min_active_ratio.is_finite()
        || !(0.0..=1.0).contains(&value.min_active_ratio)
        || !value.constrained_assignment
    {
        return Err(setup_error("VBx clustering configuration is incompatible"));
    }
    Ok(())
}

pub(super) fn cluster(
    embeddings: &[VbxEmbedding<'_>],
    total_frames: usize,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    plda: &Plda,
    config: &VbxConfig,
) -> Result<VbxResult> {
    validate_config(config)?;
    if embeddings.is_empty() {
        return Ok(VbxResult {
            assignments: Vec::new(),
            posterior_iterations: 0,
            training_embeddings: 0,
            automatic_speakers: 0,
            retained_speakers: 0,
        });
    }
    if embeddings
        .iter()
        .any(|embedding| embedding.values.len() != plda.input_dimension())
    {
        return Err(model_error(
            "embedding dimension does not match the PLDA transform",
        ));
    }
    let minimum_clean = (config.min_active_ratio * total_frames as f64).ceil() as usize;
    let mut training = embeddings
        .iter()
        .filter(|embedding| {
            embedding.clean_frames >= minimum_clean
                && embedding.values.iter().all(|value| value.is_finite())
        })
        .collect::<Vec<_>>();
    if training.is_empty() {
        training = embeddings
            .iter()
            .filter(|embedding| embedding.values.iter().all(|value| value.is_finite()))
            .collect();
    }
    if training.is_empty() {
        return Err(model_error("no finite embeddings are available for VBx"));
    }
    if training.len() < 2 {
        return Ok(VbxResult {
            assignments: embeddings
                .iter()
                .map(|embedding| VbxAssignment {
                    chunk: embedding.chunk,
                    local_speaker: embedding.local_speaker,
                    speaker: 0,
                })
                .collect(),
            posterior_iterations: 0,
            training_embeddings: 1,
            automatic_speakers: 1,
            retained_speakers: 1,
        });
    }

    let normalized = training
        .iter()
        .map(|embedding| normalize_f32(embedding.values))
        .collect::<Result<Vec<_>>>()?;
    let initial = centroid_linkage(&normalized, config.threshold)?;
    let features = training
        .iter()
        .map(|embedding| plda.project(embedding.values))
        .collect::<Result<Vec<_>>>()?;
    let (posteriors, priors, iterations) = posterior(
        &features,
        &plda.phi(),
        &initial,
        config.fa,
        config.fb,
        config.max_iters,
    )?;
    let retained = priors
        .iter()
        .enumerate()
        .filter_map(|(index, prior)| (*prior > 1e-7).then_some(index))
        .collect::<Vec<_>>();
    let retained = if retained.is_empty() {
        vec![argmax(&priors)]
    } else {
        retained
    };
    let automatic_speakers = retained.len();
    let mut centroids = retained
        .iter()
        .map(|speaker| {
            weighted_centroid(
                &training,
                posteriors.iter().map(|row| row[*speaker]),
                plda.input_dimension(),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let minimum = min_speakers.unwrap_or(1).min(training.len()).max(1);
    let maximum = max_speakers
        .unwrap_or(training.len())
        .min(training.len())
        .max(1);
    let requested = match (min_speakers, max_speakers) {
        (Some(min), Some(max)) if min == max => Some(min.min(training.len()).max(1)),
        _ if centroids.len() < minimum => Some(minimum),
        _ if centroids.len() > maximum => Some(maximum),
        _ => None,
    };
    let constrained = if let Some(count) = requested.filter(|count| *count != centroids.len()) {
        centroids = deterministic_kmeans(&normalized, count)?;
        centroids = centroids
            .iter()
            .map(|centroid| normalize(centroid))
            .collect::<Result<Vec<_>>>()?;
        false
    } else {
        config.constrained_assignment
    };

    let mut by_chunk = BTreeMap::<usize, Vec<(usize, usize, Vec<f64>)>>::new();
    for (index, embedding) in embeddings.iter().enumerate() {
        by_chunk.entry(embedding.chunk).or_default().push((
            index,
            embedding.local_speaker,
            normalize_f32(embedding.values)?,
        ));
    }
    let mut assignments = Vec::new();
    for (chunk, values) in by_chunk {
        let scores = values
            .iter()
            .map(|(_, _, embedding)| {
                centroids
                    .iter()
                    .map(|centroid| dot(embedding, centroid))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let selected = if constrained {
            constrained_argmax(&scores)
        } else {
            scores
                .iter()
                .enumerate()
                .map(|(speaker, row)| (speaker, argmax(row)))
                .collect()
        };
        for (local_index, speaker) in selected {
            let (_, local_speaker, _) = &values[local_index];
            assignments.push(VbxAssignment {
                chunk,
                local_speaker: *local_speaker,
                speaker,
            });
        }
    }
    assignments.sort_by_key(|item| (item.chunk, item.local_speaker));
    Ok(VbxResult {
        assignments,
        posterior_iterations: iterations,
        training_embeddings: training.len(),
        automatic_speakers,
        retained_speakers: centroids.len(),
    })
}

fn centroid_linkage(values: &[Vec<f64>], threshold: f64) -> Result<Vec<usize>> {
    #[derive(Clone)]
    struct Cluster {
        members: Vec<usize>,
        centroid: Vec<f64>,
    }
    let mut clusters = values
        .iter()
        .enumerate()
        .map(|(index, value)| Cluster {
            members: vec![index],
            centroid: value.clone(),
        })
        .collect::<Vec<_>>();
    loop {
        let mut best: Option<(f64, usize, usize)> = None;
        for left in 0..clusters.len() {
            for right in (left + 1)..clusters.len() {
                let distance = euclidean(&clusters[left].centroid, &clusters[right].centroid)?;
                if distance <= threshold
                    && best.is_none_or(|current| {
                        distance < current.0
                            || (distance == current.0 && (left, right) < (current.1, current.2))
                    })
                {
                    best = Some((distance, left, right));
                }
            }
        }
        let Some((_, left, right)) = best else {
            break;
        };
        let removed = clusters.remove(right);
        clusters[left].members.extend(removed.members);
        clusters[left].members.sort_unstable();
        clusters[left].centroid = mean(
            clusters[left]
                .members
                .iter()
                .map(|index| values[*index].as_slice()),
        )?;
    }
    clusters.sort_by_key(|cluster| cluster.members[0]);
    let mut labels = vec![0; values.len()];
    for (label, cluster) in clusters.iter().enumerate() {
        for &member in &cluster.members {
            labels[member] = label;
        }
    }
    Ok(labels)
}

fn posterior(
    features: &[Vec<f64>],
    phi: &[f64],
    initial: &[usize],
    fa: f64,
    fb: f64,
    max_iters: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, usize)> {
    if features.is_empty()
        || features.iter().any(|row| row.len() != phi.len())
        || initial.len() != features.len()
    {
        return Err(model_error("invalid VBx posterior dimensions"));
    }
    let speakers = initial.iter().copied().max().unwrap_or(0) + 1;
    let smooth = 7.0_f64.exp();
    let denominator = smooth + speakers.saturating_sub(1) as f64;
    let mut gamma = initial
        .iter()
        .map(|label| {
            (0..speakers)
                .map(|speaker| {
                    if speaker == *label {
                        smooth / denominator
                    } else {
                        1.0 / denominator
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut priors = vec![1.0 / speakers as f64; speakers];
    let sqrt_phi = phi.iter().map(|value| value.sqrt()).collect::<Vec<_>>();
    let rho = features
        .iter()
        .map(|row| {
            row.iter()
                .zip(&sqrt_phi)
                .map(|(value, scale)| value * scale)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let constant = features
        .iter()
        .map(|row| {
            -0.5 * (row.iter().map(|value| value * value).sum::<f64>()
                + row.len() as f64 * (2.0 * std::f64::consts::PI).ln())
        })
        .collect::<Vec<_>>();
    for _ in 0..max_iters {
        let counts = (0..speakers)
            .map(|speaker| gamma.iter().map(|row| row[speaker]).sum::<f64>())
            .collect::<Vec<_>>();
        let inv_l = (0..speakers)
            .map(|speaker| {
                phi.iter()
                    .map(|value| 1.0 / (1.0 + fa / fb * counts[speaker] * value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let alpha = (0..speakers)
            .map(|speaker| {
                (0..phi.len())
                    .map(|dimension| {
                        fa / fb
                            * inv_l[speaker][dimension]
                            * gamma
                                .iter()
                                .zip(&rho)
                                .map(|(responsibility, row)| {
                                    responsibility[speaker] * row[dimension]
                                })
                                .sum::<f64>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (frame, row) in gamma.iter_mut().enumerate() {
            let logits = (0..speakers)
                .map(|speaker| {
                    let likelihood = dot(&rho[frame], &alpha[speaker])
                        - 0.5
                            * inv_l[speaker]
                                .iter()
                                .zip(&alpha[speaker])
                                .zip(phi)
                                .map(|((inverse, mean), covariance)| {
                                    (inverse + mean * mean) * covariance
                                })
                                .sum::<f64>()
                        + constant[frame];
                    fa * likelihood + (priors[speaker] + 1e-8).ln()
                })
                .collect::<Vec<_>>();
            *row = softmax(&logits);
        }
        for (speaker, prior) in priors.iter_mut().enumerate() {
            *prior = gamma.iter().map(|row| row[speaker]).sum::<f64>() / features.len() as f64;
        }
    }
    Ok((gamma, priors, max_iters))
}

fn weighted_centroid<'a>(
    embeddings: &[&VbxEmbedding<'a>],
    weights: impl Iterator<Item = f64>,
    dimension: usize,
) -> Result<Vec<f64>> {
    let mut centroid = vec![0.0; dimension];
    let mut total = 0.0;
    for (embedding, weight) in embeddings.iter().zip(weights) {
        total += weight;
        for (output, value) in centroid.iter_mut().zip(embedding.values) {
            *output += weight * f64::from(*value);
        }
    }
    if !total.is_finite() || total <= f64::EPSILON {
        return Err(model_error("VBx posterior has no finite speaker mass"));
    }
    for value in &mut centroid {
        *value /= total;
    }
    normalize(&centroid)
}

fn deterministic_kmeans(values: &[Vec<f64>], count: usize) -> Result<Vec<Vec<f64>>> {
    if count == 0 || values.is_empty() {
        return Err(model_error("K-means requires embeddings and speakers"));
    }
    let mut random = Mt19937::new(42);
    let mut best: Option<(f64, Vec<Vec<f64>>)> = None;
    // Match the pinned `KMeans(n_init=3, random_state=42)` fallback. Keeping
    // center order is significant for all-NaN inactive embeddings because
    // NumPy's argmax deterministically assigns those to center zero.
    for _ in 0..3 {
        let initial = kmeans_plus_plus(values, count, &mut random)?;
        let centroids = kmeans_lloyd(values, initial)?;
        let inertia = values
            .iter()
            .map(|value| {
                centroids
                    .iter()
                    .map(|centroid| {
                        let distance = euclidean(value, centroid).unwrap_or(f64::INFINITY);
                        distance * distance
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .sum::<f64>();
        if best
            .as_ref()
            .is_none_or(|(best_inertia, _)| inertia < *best_inertia)
        {
            best = Some((inertia, centroids));
        }
    }
    Ok(best.expect("non-empty K-means candidates").1)
}

fn kmeans_plus_plus(
    values: &[Vec<f64>],
    count: usize,
    random: &mut Mt19937,
) -> Result<Vec<Vec<f64>>> {
    let first = (random.random_f64() * values.len() as f64).floor() as usize;
    let mut centroids = vec![values[first.min(values.len() - 1)].clone()];
    while centroids.len() < count {
        let closest = values
            .iter()
            .map(|value| {
                centroids
                    .iter()
                    .map(|centroid| {
                        let distance = euclidean(value, centroid).unwrap_or(f64::INFINITY);
                        distance * distance
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect::<Vec<_>>();
        let potential = closest.iter().sum::<f64>();
        let trials = 2 + (count as f64).ln() as usize;
        let mut best: Option<(f64, usize)> = None;
        for _ in 0..trials {
            let target = random.random_f64() * potential;
            let mut cumulative = 0.0;
            let mut candidate = values.len() - 1;
            for (index, distance) in closest.iter().enumerate() {
                cumulative += distance;
                if cumulative >= target {
                    candidate = index;
                    break;
                }
            }
            let candidate_potential = values
                .iter()
                .zip(&closest)
                .map(|(value, current)| {
                    let distance = euclidean(value, &values[candidate]).unwrap_or(f64::INFINITY);
                    current.min(distance * distance)
                })
                .sum::<f64>();
            if best
                .as_ref()
                .is_none_or(|(best_potential, _)| candidate_potential < *best_potential)
            {
                best = Some((candidate_potential, candidate));
            }
        }
        centroids.push(values[best.expect("K-means++ candidate").1].clone());
    }
    Ok(centroids)
}

fn kmeans_lloyd(values: &[Vec<f64>], mut centroids: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    for _ in 0..300 {
        let labels = values
            .iter()
            .map(|value| {
                argmin(
                    &centroids
                        .iter()
                        .map(|centroid| euclidean(value, centroid).unwrap_or(f64::INFINITY))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let mut changed = false;
        for (speaker, centroid) in centroids.iter_mut().enumerate() {
            let members = values
                .iter()
                .zip(&labels)
                .filter_map(|(value, label)| (*label == speaker).then_some(value.as_slice()));
            let members = members.collect::<Vec<_>>();
            if members.is_empty() {
                continue;
            }
            let next = mean(members.into_iter())?;
            changed |= euclidean(centroid, &next)? > 1e-6;
            *centroid = next;
        }
        if !changed {
            break;
        }
    }
    Ok(centroids)
}

struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0_u32; 624];
        state[0] = seed;
        for index in 1..624 {
            state[index] = 1_812_433_253_u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        Self { state, index: 624 }
    }

    fn random_f64(&mut self) -> f64 {
        let high = u64::from(self.next_u32() >> 5);
        let low = u64::from(self.next_u32() >> 6);
        ((high << 26) + low) as f64 / 9_007_199_254_740_992.0
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for index in 0..624 {
                let value = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                let mut twisted = self.state[(index + 397) % 624] ^ (value >> 1);
                if value & 1 != 0 {
                    twisted ^= 0x9908_b0df;
                }
                self.state[index] = twisted;
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }
}

fn constrained_argmax(scores: &[Vec<f64>]) -> Vec<(usize, usize)> {
    fn visit(
        row: usize,
        scores: &[Vec<f64>],
        used: &mut [bool],
        current: &mut Vec<(usize, usize)>,
        total: f64,
        best: &mut (f64, Vec<(usize, usize)>),
    ) {
        if row == scores.len() || current.len() == used.len() {
            if total > best.0 {
                *best = (total, current.clone());
            }
            if row == scores.len() {
                return;
            }
            visit(row + 1, scores, used, current, total, best);
            return;
        }
        if scores.len() > used.len() {
            visit(row + 1, scores, used, current, total, best);
        }
        for speaker in 0..used.len() {
            if used[speaker] {
                continue;
            }
            used[speaker] = true;
            current.push((row, speaker));
            visit(
                row + 1,
                scores,
                used,
                current,
                total + scores[row][speaker],
                best,
            );
            current.pop();
            used[speaker] = false;
        }
    }
    if scores.is_empty() || scores[0].is_empty() {
        return Vec::new();
    }
    let mut best = (f64::NEG_INFINITY, Vec::new());
    visit(
        0,
        scores,
        &mut vec![false; scores[0].len()],
        &mut Vec::new(),
        0.0,
        &mut best,
    );
    best.1
}

fn normalize_f32(values: &[f32]) -> Result<Vec<f64>> {
    normalize(
        &values
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>(),
    )
}

fn normalize(values: &[f64]) -> Result<Vec<f64>> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(model_error("embedding norm must be finite and non-zero"));
    }
    Ok(values.iter().map(|value| value / norm).collect())
}

fn mean<'a>(values: impl Iterator<Item = &'a [f64]>) -> Result<Vec<f64>> {
    let values = values.collect::<Vec<_>>();
    let Some(first) = values.first() else {
        return Err(model_error("cannot average an empty embedding set"));
    };
    if values.iter().any(|value| value.len() != first.len()) {
        return Err(model_error("embedding dimensions differ"));
    }
    let mut output = vec![0.0; first.len()];
    for value in &values {
        for (output, value) in output.iter_mut().zip(*value) {
            *output += value;
        }
    }
    for output in &mut output {
        *output /= values.len() as f64;
    }
    Ok(output)
}

fn euclidean(left: &[f64], right: &[f64]) -> Result<f64> {
    if left.len() != right.len() {
        return Err(model_error("embedding dimensions differ"));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>()
        .sqrt())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = values
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let sum = exponentials.iter().sum::<f64>();
    exponentials.iter().map(|value| value / sum).collect()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.partial_cmp(right)
                .unwrap_or(std::cmp::Ordering::Less)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn argmin(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(left_index, left), (right_index, right)| {
            left.partial_cmp(right)
                .unwrap_or(std::cmp::Ordering::Greater)
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn setup_error(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("setup_error: {}", message.into()))
}

fn model_error(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("model_output_mismatch: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posterior_iterations_materially_change_assignments() {
        let features = vec![vec![2.0, 0.0], vec![1.8, 0.1], vec![-2.0, 0.0]];
        let initial = vec![0, 0, 1];
        let (one, _, _) = posterior(&features, &[2.0, 0.5], &initial, 0.07, 0.8, 1).unwrap();
        let (many, _, _) = posterior(&features, &[2.0, 0.5], &initial, 0.07, 0.8, 20).unwrap();
        assert_ne!(one, many);
    }

    #[test]
    fn constrained_assignment_does_not_reuse_a_speaker_in_one_chunk() {
        let selected = constrained_argmax(&[vec![0.9, 0.8], vec![0.85, 0.1]]);
        assert_eq!(selected, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn clustering_parameters_are_all_controlled() {
        let base = VbxConfig {
            kind: "vbx".to_string(),
            threshold: 0.6,
            fa: 0.07,
            fb: 0.8,
            max_iters: 20,
            min_active_ratio: 0.2,
            constrained_assignment: true,
        };
        validate_config(&base).unwrap();
        let mut fault = base.clone();
        fault.max_iters = 0;
        assert!(validate_config(&fault).is_err());
        let mut fault = base.clone();
        fault.fa = f64::NAN;
        assert!(validate_config(&fault).is_err());
        let mut fault = base;
        fault.constrained_assignment = false;
        assert!(validate_config(&fault).is_err());
    }

    #[test]
    fn kmeans_random_state_matches_numpy_mt19937() {
        let mut random = Mt19937::new(42);
        let expected = [
            0.374_540_118_847_362_5,
            0.950_714_306_409_916_2,
            0.731_993_941_811_405_1,
        ];
        for expected in expected {
            assert!((random.random_f64() - expected).abs() < 1e-15);
        }
    }
}
