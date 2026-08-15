use media_core::Result;

use crate::{invalid_request, CandleWhisperDecodeConfig};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperSearchResult {
    pub token_ids: Vec<u32>,
    pub score: f64,
    pub completed: bool,
    pub forward_calls: usize,
}

impl WhisperSearchResult {
    pub(crate) fn average_log_probability(&self) -> f64 {
        self.score / (self.token_ids.len() + usize::from(self.completed)).max(1) as f64
    }
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    token_ids: Vec<u32>,
    score: f64,
    completed: bool,
    forward_calls: usize,
}

impl SearchCandidate {
    fn scored_token_count(&self) -> usize {
        (self.token_ids.len() + usize::from(self.completed)).max(1)
    }

    fn average_score(&self) -> f64 {
        self.score / self.scored_token_count() as f64
    }

    fn beam_score(&self, length_penalty: f64) -> f64 {
        let length = self.scored_token_count() as f64;
        let penalty = ((5.0 + length) / 6.0).powf(length_penalty);
        self.score / penalty
    }

    fn finish(self) -> WhisperSearchResult {
        WhisperSearchResult {
            token_ids: self.token_ids,
            score: self.score,
            completed: self.completed,
            forward_calls: self.forward_calls,
        }
    }
}

/// Runs the configured non-default search against a request-scoped logits seam.
///
/// The provider receives generated tokens only and returns filtered logits for
/// the next token. Model execution, cache policy, and Whisper-specific logit
/// rules remain behind that seam.
#[cfg(test)]
pub(crate) fn decode_with_config(
    config: &CandleWhisperDecodeConfig,
    eos_token_id: u32,
    max_generated_tokens: usize,
    mut logits_for: impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<WhisperSearchResult> {
    if config.beam_size > 1 {
        return decode_beam_search(config, eos_token_id, max_generated_tokens, &mut logits_for);
    }
    let temperature = config
        .temperature_schedule
        .first()
        .copied()
        .ok_or_else(|| invalid_request("Candle Whisper temperature_schedule must not be empty"))?;
    decode_at_temperature(
        config,
        0,
        temperature,
        eos_token_id,
        max_generated_tokens,
        &mut logits_for,
    )
}

/// Decodes and ranks candidates for one declared temperature. The native
/// fallback controller invokes this in declaration order and decides whether
/// the next temperature is needed from the completed attempt's thresholds.
pub(crate) fn decode_at_temperature(
    config: &CandleWhisperDecodeConfig,
    temperature_index: usize,
    temperature: f64,
    eos_token_id: u32,
    max_generated_tokens: usize,
    logits_for: &mut impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<WhisperSearchResult> {
    if config.beam_size > 1 {
        return decode_beam_search(config, eos_token_id, max_generated_tokens, logits_for);
    }
    let mut candidates = Vec::new();
    let candidate_count = if temperature > 0.0 { config.best_of } else { 1 };
    for candidate_index in 0..candidate_count {
        let seed = derived_seed(config.seed, temperature_index, candidate_index);
        candidates.push(decode_one_candidate(
            temperature,
            seed,
            eos_token_id,
            max_generated_tokens,
            logits_for,
        )?);
    }
    candidates
        .into_iter()
        .max_by(compare_average_candidates)
        .map(SearchCandidate::finish)
        .ok_or_else(|| invalid_request("Candle Whisper decoding produced no candidates"))
}

fn decode_one_candidate(
    temperature: f64,
    seed: u64,
    eos_token_id: u32,
    max_generated_tokens: usize,
    logits_for: &mut impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<SearchCandidate> {
    let mut candidate = SearchCandidate {
        token_ids: Vec::new(),
        score: 0.0,
        completed: false,
        forward_calls: 0,
    };
    let mut rng = SeededRng::new(seed);
    while candidate.token_ids.len() < max_generated_tokens {
        let logits = logits_for(&candidate.token_ids)?;
        candidate.forward_calls += 1;
        let model_log_probabilities = log_probabilities(&logits, 0.0)?;
        let sampling_log_probabilities = if temperature == 0.0 {
            model_log_probabilities.clone()
        } else {
            log_probabilities(&logits, temperature)?
        };
        let token = if temperature == 0.0 {
            argmax_log_probability(&sampling_log_probabilities)
        } else {
            sample_log_probabilities(&sampling_log_probabilities, &mut rng)
        }
        .ok_or_else(|| invalid_request("Candle Whisper logits were fully suppressed"))?;
        candidate.score += model_log_probabilities[token];
        if token as u32 == eos_token_id {
            candidate.completed = true;
            break;
        }
        candidate.token_ids.push(token as u32);
    }
    Ok(candidate)
}

fn decode_beam_search(
    config: &CandleWhisperDecodeConfig,
    eos_token_id: u32,
    max_generated_tokens: usize,
    logits_for: &mut impl FnMut(&[u32]) -> Result<Vec<f32>>,
) -> Result<WhisperSearchResult> {
    let mut active = vec![SearchCandidate {
        token_ids: Vec::new(),
        score: 0.0,
        completed: false,
        forward_calls: 0,
    }];
    let mut completed = Vec::new();
    let completion_target = ((config.beam_size as f64 * config.patience).ceil() as usize).max(1);
    let mut stopped_for_patience = false;

    while !active.is_empty() && active[0].token_ids.len() < max_generated_tokens {
        let mut proposals = Vec::new();
        for candidate in std::mem::take(&mut active) {
            let logits = logits_for(&candidate.token_ids)?;
            let log_probabilities = log_probabilities(&logits, 0.0)?;
            for (token, token_score) in top_log_probabilities(&log_probabilities, config.beam_size)
            {
                let mut next = candidate.clone();
                next.forward_calls += 1;
                next.score += token_score;
                if token as u32 == eos_token_id {
                    next.completed = true;
                } else {
                    next.token_ids.push(token as u32);
                }
                proposals.push(next);
            }
        }

        proposals
            .sort_by(|left, right| compare_beam_candidates(right, left, config.length_penalty));
        let mut next_active = Vec::with_capacity(config.beam_size);
        for (global_rank, proposal) in proposals.into_iter().enumerate() {
            if proposal.completed {
                if global_rank < config.beam_size {
                    completed.push(proposal);
                }
            } else if next_active.len() < config.beam_size {
                next_active.push(proposal);
            }
            if global_rank >= config.beam_size && next_active.len() == config.beam_size {
                break;
            }
        }
        active = next_active;
        if completed.len() >= completion_target {
            stopped_for_patience = true;
            break;
        }
    }

    if !stopped_for_patience {
        completed.extend(active.into_iter().map(|mut candidate| {
            candidate.completed = false;
            candidate
        }));
    }
    completed
        .into_iter()
        .max_by(|left, right| compare_beam_candidates(left, right, config.length_penalty))
        .map(SearchCandidate::finish)
        .ok_or_else(|| invalid_request("Candle Whisper beam search produced no candidates"))
}

fn compare_average_candidates(
    left: &SearchCandidate,
    right: &SearchCandidate,
) -> std::cmp::Ordering {
    left.average_score()
        .total_cmp(&right.average_score())
        .then_with(|| right.token_ids.cmp(&left.token_ids))
}

fn compare_beam_candidates(
    left: &SearchCandidate,
    right: &SearchCandidate,
    length_penalty: f64,
) -> std::cmp::Ordering {
    left.beam_score(length_penalty)
        .total_cmp(&right.beam_score(length_penalty))
        .then_with(|| right.token_ids.cmp(&left.token_ids))
}

fn log_probabilities(logits: &[f32], temperature: f64) -> Result<Vec<f64>> {
    let scale = if temperature == 0.0 { 1.0 } else { temperature };
    let max = logits
        .iter()
        .map(|value| *value as f64)
        .filter(|value| value.is_finite())
        .max_by(f64::total_cmp)
        .ok_or_else(|| invalid_request("Candle Whisper logits were fully suppressed"))?;
    let normalizer = logits
        .iter()
        .map(|value| (*value as f64 - max) / scale)
        .filter(|value| value.is_finite())
        .map(f64::exp)
        .sum::<f64>()
        .ln();
    Ok(logits
        .iter()
        .map(|value| {
            let scaled = (*value as f64 - max) / scale;
            if scaled.is_finite() {
                scaled - normalizer
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect())
}

fn argmax_log_probability(values: &[f64]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
}

fn top_log_probabilities(values: &[f64], count: usize) -> Vec<(usize, f64)> {
    let mut ranked = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_index, left), (right_index, right)| {
        right
            .total_cmp(left)
            .then_with(|| left_index.cmp(right_index))
    });
    ranked.truncate(count);
    ranked
}

fn sample_log_probabilities(values: &[f64], rng: &mut SeededRng) -> Option<usize> {
    let draw = rng.next_unit_f64();
    let mut cumulative = 0.0;
    let mut last = None;
    for (index, log_probability) in values.iter().copied().enumerate() {
        if !log_probability.is_finite() {
            continue;
        }
        last = Some(index);
        cumulative += log_probability.exp();
        if draw < cumulative {
            return Some(index);
        }
    }
    last
}

fn derived_seed(seed: u64, temperature_index: usize, candidate_index: usize) -> u64 {
    seed ^ (temperature_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (candidate_index as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

#[derive(Debug, Clone, Copy)]
struct SeededRng {
    state: u64,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xA076_1D64_78BD_642F,
        }
    }

    fn next_unit_f64(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (bits >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn seeded_sampling_repeats_the_same_sequence() {
        let config = CandleWhisperDecodeConfig {
            temperature_schedule: vec![0.8],
            best_of: 2,
            seed: 17,
            ..CandleWhisperDecodeConfig::default()
        };
        let run = || {
            decode_with_config(&config, 9, 4, |_| {
                Ok(vec![0.0, 1.0, 0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 2.0])
            })
            .unwrap()
        };

        assert_eq!(run(), run());
    }

    #[test]
    fn tiny_positive_temperature_keeps_finite_logits_sampleable() {
        let probabilities = log_probabilities(&[2.0, 1.0], f64::MIN_POSITIVE).unwrap();

        assert_eq!(argmax_log_probability(&probabilities), Some(0));
    }

    #[test]
    fn completed_average_log_probability_includes_the_scored_eos_token() {
        let completed = WhisperSearchResult {
            token_ids: vec![7],
            score: -2.0,
            completed: true,
            forward_calls: 2,
        };
        let unfinished = WhisperSearchResult {
            completed: false,
            ..completed.clone()
        };

        assert_eq!(completed.average_log_probability(), -1.0);
        assert_eq!(unfinished.average_log_probability(), -2.0);
    }

    #[test]
    fn sampling_temperature_does_not_rescale_reported_model_score() {
        let run = |temperature| {
            decode_one_candidate(temperature, 17, 2, 2, &mut |generated| {
                Ok(if generated.is_empty() {
                    vec![10.0, 0.0, f32::NEG_INFINITY]
                } else {
                    vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
                })
            })
            .unwrap()
        };
        let cool = run(0.5);
        let warm = run(2.0);

        assert_eq!(cool.token_ids, vec![0]);
        assert_eq!(warm.token_ids, cool.token_ids);
        assert!((cool.score - warm.score).abs() < 1e-12);
    }

    #[test]
    fn best_of_and_beam_lengths_count_completed_eos() {
        let completed = SearchCandidate {
            token_ids: vec![7],
            score: -1.0,
            completed: true,
            forward_calls: 2,
        };
        let unfinished = SearchCandidate {
            token_ids: vec![7],
            score: -0.75,
            completed: false,
            forward_calls: 1,
        };

        assert_eq!(completed.scored_token_count(), 2);
        assert!(compare_average_candidates(&completed, &unfinished).is_gt());
        assert_eq!(completed.beam_score(0.0), -1.0);
    }

    #[test]
    fn decode_with_config_only_evaluates_the_first_declared_temperature() {
        let config = CandleWhisperDecodeConfig {
            temperature_schedule: vec![0.0, 0.4, 0.8],
            best_of: 2,
            seed: 3,
            ..CandleWhisperDecodeConfig::default()
        };
        let calls = Cell::new(0);

        decode_with_config(&config, 1, 1, |_| {
            calls.set(calls.get() + 1);
            Ok(vec![f32::NEG_INFINITY, 0.0])
        })
        .unwrap();

        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn one_temperature_attempt_ranks_only_its_own_best_of_candidates() {
        let config = CandleWhisperDecodeConfig {
            temperature_schedule: vec![0.0, 0.4],
            best_of: 2,
            seed: 3,
            ..CandleWhisperDecodeConfig::default()
        };
        let calls = Cell::new(0);

        decode_at_temperature(&config, 1, 0.4, 1, 1, &mut |_| {
            calls.set(calls.get() + 1);
            Ok(vec![f32::NEG_INFINITY, 0.0])
        })
        .unwrap();

        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn best_of_keeps_the_candidate_with_the_highest_average_log_probability() {
        let config = CandleWhisperDecodeConfig {
            temperature_schedule: vec![1.0],
            best_of: 32,
            seed: 29,
            ..CandleWhisperDecodeConfig::default()
        };
        let calls = Cell::new(0);

        let result = decode_with_config(&config, 2, 2, |generated| {
            calls.set(calls.get() + 1);
            Ok(if generated.is_empty() {
                vec![2.0, 1.0, f32::NEG_INFINITY]
            } else {
                vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]
            })
        })
        .unwrap();

        assert_eq!(result.token_ids, vec![0]);
        assert!(result.completed);
        assert_eq!(calls.get(), 64);
    }

    #[test]
    fn beam_search_ranks_independent_completed_hypotheses() {
        let config = CandleWhisperDecodeConfig {
            beam_size: 2,
            length_penalty: 0.0,
            ..CandleWhisperDecodeConfig::default()
        };

        let result = decode_with_config(&config, 3, 3, |generated| {
            Ok(match generated {
                [] => vec![2.0, 1.9, f32::NEG_INFINITY, f32::NEG_INFINITY],
                [0] => vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 2.0, 0.0],
                [1] => vec![f32::NEG_INFINITY, f32::NEG_INFINITY, -3.0, 3.0],
                _ => vec![f32::NEG_INFINITY, f32::NEG_INFINITY, -3.0, 3.0],
            })
        })
        .unwrap();

        assert_eq!(result.token_ids, vec![1]);
        assert!(result.completed);
    }

    #[test]
    fn beam_patience_ignores_low_probability_eos_expansions_pruned_globally() {
        let config = CandleWhisperDecodeConfig {
            beam_size: 2,
            patience: 1.0,
            ..CandleWhisperDecodeConfig::default()
        };

        let result = decode_with_config(&config, 3, 3, |generated| {
            Ok(match generated {
                [] => vec![2.0, 1.9, f32::NEG_INFINITY, f32::NEG_INFINITY],
                [0] | [1] => vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 5.0, -5.0],
                [0, 2] | [1, 2] => {
                    vec![f32::NEG_INFINITY, f32::NEG_INFINITY, -5.0, 5.0]
                }
                _ => unreachable!("unexpected beam hypothesis: {generated:?}"),
            })
        })
        .unwrap();

        assert_eq!(result.token_ids.len(), 2);
        assert!(result.completed);
    }

    #[test]
    fn beam_token_limit_ranks_active_hypotheses_with_completed_eos_candidates() {
        let config = CandleWhisperDecodeConfig {
            beam_size: 2,
            patience: 2.0,
            ..CandleWhisperDecodeConfig::default()
        };

        let result = decode_with_config(&config, 3, 2, |generated| {
            Ok(match generated {
                [] => vec![2.0, 1.9, f32::NEG_INFINITY, f32::NEG_INFINITY],
                [0] => vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 3.0, -3.0],
                [1] => vec![f32::NEG_INFINITY, f32::NEG_INFINITY, -3.0, 3.0],
                _ => unreachable!("unexpected beam hypothesis: {generated:?}"),
            })
        })
        .unwrap();

        assert_eq!(result.token_ids, vec![0, 2]);
        assert!(!result.completed);
    }

    #[test]
    fn beam_patience_waits_for_more_completed_hypotheses() {
        let run = |patience| {
            let config = CandleWhisperDecodeConfig {
                beam_size: 2,
                patience,
                ..CandleWhisperDecodeConfig::default()
            };
            let calls = Cell::new(0);
            decode_with_config(&config, 3, 4, |generated| {
                calls.set(calls.get() + 1);
                Ok(match generated {
                    [] => vec![2.0, 1.9, f32::NEG_INFINITY, f32::NEG_INFINITY],
                    [0] => vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, 3.0],
                    [1] => vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 3.0, 2.0],
                    _ => vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 2.0, 3.0],
                })
            })
            .unwrap();
            calls.get()
        };

        assert!(run(2.0) > run(1.0));
    }

    #[test]
    fn beam_size_five_stays_bounded_by_the_generation_limit() {
        let config = CandleWhisperDecodeConfig {
            beam_size: 5,
            ..CandleWhisperDecodeConfig::default()
        };
        let calls = Cell::new(0);

        let result = decode_with_config(&config, 6, 8, |_| {
            calls.set(calls.get() + 1);
            Ok(vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0, f32::NEG_INFINITY])
        })
        .unwrap();

        assert_eq!(result.token_ids.len(), 8);
        assert!(!result.completed);
        assert_eq!(calls.get(), 1 + 5 * 7);
    }

    #[test]
    fn length_penalty_can_prefer_a_longer_beam_hypothesis() {
        let short = SearchCandidate {
            token_ids: vec![1],
            score: -0.8,
            completed: true,
            forward_calls: 2,
        };
        let long = SearchCandidate {
            token_ids: vec![1, 2, 3],
            score: -0.9,
            completed: true,
            forward_calls: 4,
        };

        assert!(compare_beam_candidates(&short, &long, 0.0).is_gt());
        assert!(compare_beam_candidates(&long, &short, 2.0).is_gt());
    }
}
