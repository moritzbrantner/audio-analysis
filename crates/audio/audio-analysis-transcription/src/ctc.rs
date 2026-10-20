//! Model-independent CTC forced alignment shared by native and browser runtimes.
use crate::{invalid_request, model_output_mismatch, AlignedWord};
use media_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CtcVocabulary {
    pub blank_id: usize,
    pub tokens: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(crate) struct CtcAlignedToken {
    pub token_id: usize,
    pub token_index: usize,
    pub frame_index: usize,
    pub score: f32,
}

#[cfg_attr(not(feature = "alignment"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenizedAlignmentChar {
    pub(crate) char_index: usize,
    pub(crate) character: String,
    pub(crate) token_id: usize,
    pub(crate) is_word_delimiter: bool,
}

pub(crate) fn normalize_text_to_aligned_chars(
    text: &str,
    vocab: &CtcVocabulary,
) -> Result<Vec<TokenizedAlignmentChar>> {
    let delimiter = token_id(&vocab.tokens, "|").map(|_| "|");
    let uppercase_hits = vocab
        .tokens
        .iter()
        .filter(|token| token.chars().any(|ch| ch.is_ascii_uppercase()))
        .count();
    let lowercase_hits = vocab
        .tokens
        .iter()
        .filter(|token| token.chars().any(|ch| ch.is_ascii_lowercase()))
        .count();
    let use_uppercase = uppercase_hits >= lowercase_hits;
    let mut chars = Vec::new();
    for (char_index, character) in text.trim().chars().enumerate() {
        if character.is_whitespace() {
            if let Some(delimiter) = delimiter {
                if let Some(id) = token_id(&vocab.tokens, delimiter) {
                    chars.push(TokenizedAlignmentChar {
                        char_index,
                        character: " ".to_string(),
                        token_id: id,
                        is_word_delimiter: true,
                    });
                }
            }
            continue;
        }
        let lowered = character.to_lowercase().collect::<String>();
        let token = if character.is_alphabetic() {
            if use_uppercase {
                character.to_uppercase().collect::<String>()
            } else {
                lowered.clone()
            }
        } else {
            character.to_string()
        };
        let token_id = token_id(&vocab.tokens, &token).unwrap_or(usize::MAX);
        chars.push(TokenizedAlignmentChar {
            char_index,
            character: lowered,
            token_id,
            is_word_delimiter: false,
        });
    }
    if chars.is_empty() {
        return Err(invalid_request(
            "transcript text does not contain any wav2vec2 CTC tokens",
        ));
    }
    Ok(chars)
}

pub(crate) fn token_id(tokens: &[String], token: &str) -> Option<usize> {
    tokens.iter().position(|candidate| candidate == token)
}

#[allow(dead_code)]
pub(crate) fn normalize_alignment_text(text: &str) -> Vec<String> {
    text.chars()
        .filter_map(|character| {
            if character.is_alphanumeric() {
                Some(character.to_lowercase().collect::<String>())
            } else {
                None
            }
        })
        .collect()
}

#[allow(dead_code)]
pub(crate) fn build_ctc_trellis(
    emissions: &[Vec<f32>],
    token_ids: &[usize],
    blank_id: usize,
) -> Result<Vec<Vec<f32>>> {
    validate_emissions(emissions, blank_id, token_ids)?;
    let frame_count = emissions.len();
    let token_count = token_ids.len();
    let mut trellis = vec![vec![f32::NEG_INFINITY; token_count + 1]; frame_count + 1];
    trellis[0][0] = 0.0;
    for frame in 0..frame_count {
        for token_index in 0..=token_count {
            let stay = trellis[frame][token_index] + emissions[frame][blank_id];
            let change = if token_index > 0 {
                trellis[frame][token_index - 1]
                    + emission_token_score(&emissions[frame], token_ids[token_index - 1], blank_id)
            } else {
                f32::NEG_INFINITY
            };
            trellis[frame + 1][token_index] = stay.max(change);
        }
    }
    Ok(trellis)
}

#[allow(dead_code)]
pub(crate) fn backtrack_ctc(
    trellis: &[Vec<f32>],
    emissions: &[Vec<f32>],
    token_ids: &[usize],
    blank_id: usize,
) -> Result<Vec<CtcAlignedToken>> {
    validate_emissions(emissions, blank_id, token_ids)?;
    if trellis.len() != emissions.len() + 1
        || trellis.iter().any(|row| row.len() != token_ids.len() + 1)
    {
        return Err(model_output_mismatch(
            "CTC trellis dimensions are inconsistent",
        ));
    }
    let mut frame = emissions.len();
    let mut token_index = token_ids.len();
    if !trellis[frame][token_index].is_finite() {
        return Err(model_output_mismatch("CTC path is impossible"));
    }
    let mut path = Vec::new();
    while token_index > 0 {
        if frame == 0 {
            return Err(model_output_mismatch("CTC backtracking exhausted frames"));
        }
        let current_frame = frame - 1;
        let token_id = token_ids[token_index - 1];
        let changed = trellis[current_frame][token_index - 1]
            + emission_token_score(&emissions[current_frame], token_id, blank_id);
        let stayed = trellis[current_frame][token_index] + emissions[current_frame][blank_id];
        if changed >= stayed {
            path.push(CtcAlignedToken {
                token_id,
                token_index: token_index - 1,
                frame_index: current_frame,
                score: emission_token_score(&emissions[current_frame], token_id, blank_id).exp(),
            });
            token_index -= 1;
        }
        frame -= 1;
    }
    path.reverse();
    Ok(path)
}

fn emission_token_score(frame: &[f32], token_id: usize, blank_id: usize) -> f32 {
    if token_id == usize::MAX {
        frame
            .iter()
            .enumerate()
            .filter(|(index, score)| *index != blank_id && score.is_finite())
            .map(|(_, score)| *score)
            .max_by(f32::total_cmp)
            .unwrap_or(f32::NEG_INFINITY)
    } else {
        frame[token_id]
    }
}

#[allow(dead_code)]
pub(crate) fn tokens_to_words(
    tokens: &[CtcAlignedToken],
    transcript_words: &[String],
    segment_start: f64,
    segment_end: f64,
    frame_seconds: f64,
) -> Result<Vec<AlignedWord>> {
    tokens_to_segment_words(
        0,
        tokens,
        transcript_words,
        segment_start,
        segment_end,
        frame_seconds,
    )
}

#[allow(dead_code)]
pub(crate) fn tokens_to_segment_words(
    segment_index: u64,
    tokens: &[CtcAlignedToken],
    transcript_words: &[String],
    segment_start: f64,
    segment_end: f64,
    frame_seconds: f64,
) -> Result<Vec<AlignedWord>> {
    if !segment_start.is_finite()
        || !segment_end.is_finite()
        || !frame_seconds.is_finite()
        || segment_end <= segment_start
        || frame_seconds <= 0.0
    {
        return Err(invalid_request("invalid CTC word timing range"));
    }
    let mut cursor = 0;
    let mut aligned = Vec::new();
    for (word_index, word) in transcript_words.iter().enumerate() {
        let token_len = ctc_word_token_len(word).max(1);
        let end_cursor = (cursor + token_len).min(tokens.len());
        let word_tokens = &tokens[cursor..end_cursor];
        let (start_seconds, end_seconds, confidence) =
            if let (Some(first), Some(last)) = (word_tokens.first(), word_tokens.last()) {
                let start = segment_start + first.frame_index as f64 * frame_seconds;
                let end = segment_start + (last.frame_index + 1) as f64 * frame_seconds;
                let confidence = word_tokens.iter().map(|token| token.score).sum::<f32>()
                    / word_tokens.len() as f32;
                (
                    start.clamp(segment_start, segment_end),
                    end.clamp(segment_start, segment_end),
                    Some(confidence),
                )
            } else {
                let width = (segment_end - segment_start) / transcript_words.len().max(1) as f64;
                let start = segment_start + word_index as f64 * width;
                (start, (start + width).min(segment_end), None)
            };
        aligned.push(AlignedWord {
            segment_index,
            word_index,
            text: word.clone(),
            start_seconds,
            end_seconds: end_seconds.max(start_seconds),
            confidence,
        });
        cursor = end_cursor;
        if word_index + 1 < transcript_words.len() && cursor < tokens.len() {
            cursor += 1;
        }
    }
    extend_word_ends_to_delimiter_midpoints(&mut aligned);
    Ok(aligned)
}

fn ctc_word_token_len(word: &str) -> usize {
    word.chars()
        .filter(|character| !character.is_whitespace())
        .count()
}

fn extend_word_ends_to_delimiter_midpoints(words: &mut [AlignedWord]) {
    if words.len() < 2 {
        return;
    }
    for index in 0..words.len() - 1 {
        if ctc_word_token_len(&words[index].text) <= 1 {
            continue;
        }
        let next_start = words[index + 1].start_seconds;
        let end = words[index].end_seconds;
        if next_start > end {
            words[index].end_seconds = end + (next_start - end) * 0.5;
        }
    }
}

fn validate_emissions(emissions: &[Vec<f32>], blank_id: usize, token_ids: &[usize]) -> Result<()> {
    if emissions.is_empty() {
        return Err(model_output_mismatch("CTC emissions are empty"));
    }
    let Some(vocab_size) = emissions.first().map(Vec::len) else {
        return Err(model_output_mismatch("CTC emissions are empty"));
    };
    if vocab_size == 0
        || blank_id >= vocab_size
        || token_ids
            .iter()
            .any(|token| *token != usize::MAX && *token >= vocab_size)
        || emissions
            .iter()
            .any(|frame| frame.len() != vocab_size || frame.iter().any(|score| !score.is_finite()))
    {
        return Err(model_output_mismatch(
            "CTC emission dimensions are inconsistent",
        ));
    }
    Ok(())
}

/// One segment's transcript and model log probabilities. No model/runtime is loaded.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CtcEmissionAlignmentRequest {
    pub vocabulary: CtcVocabulary,
    pub emissions: Vec<Vec<f32>>,
    pub text: String,
    pub segment_index: u64,
    pub start_seconds: f64,
    pub end_seconds: f64,
}

/// Aligns existing text using CTC model evidence; impossible paths fail without
/// interpolated/guessed timing. Trellis allocation is bounded to 16 million cells.
pub fn align_ctc_emissions(request: CtcEmissionAlignmentRequest) -> Result<Vec<AlignedWord>> {
    if !request.start_seconds.is_finite()
        || !request.end_seconds.is_finite()
        || request.start_seconds < 0.0
        || request.end_seconds <= request.start_seconds
        || request.text.len() > 65_536
    {
        return Err(invalid_request("invalid CTC segment text or timing range"));
    }
    let words = request
        .text
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if words.len() > 1 && token_id(&request.vocabulary.tokens, "|").is_none() {
        return Err(invalid_request(
            "multi-word CTC alignment requires a word delimiter token",
        ));
    }
    let tokens = normalize_text_to_aligned_chars(&words.join(" "), &request.vocabulary)?;
    let token_ids = tokens
        .iter()
        .map(|token| token.token_id)
        .collect::<Vec<_>>();
    if (request.emissions.len() + 1)
        .checked_mul(token_ids.len() + 1)
        .is_none_or(|cells| cells > 16_000_000)
    {
        return Err(invalid_request(
            "CTC alignment exceeds the bounded trellis size",
        ));
    }
    if request
        .emissions
        .first()
        .is_some_and(|frame| frame.len() != request.vocabulary.tokens.len())
        || request.emissions.iter().flatten().any(|score| *score > 0.0)
    {
        return Err(model_output_mismatch(
            "CTC vocabulary and finite log probabilities are required",
        ));
    }
    let trellis = build_ctc_trellis(&request.emissions, &token_ids, request.vocabulary.blank_id)?;
    let path = backtrack_ctc(
        &trellis,
        &request.emissions,
        &token_ids,
        request.vocabulary.blank_id,
    )?;
    let frame_seconds =
        (request.end_seconds - request.start_seconds) / request.emissions.len() as f64;
    let aligned = tokens_to_segment_words(
        request.segment_index,
        &path,
        &words,
        request.start_seconds,
        request.end_seconds,
        frame_seconds,
    )?;
    if aligned.iter().any(|word| word.confidence.is_none()) {
        return Err(model_output_mismatch(
            "CTC evidence cannot align every word",
        ));
    }
    Ok(aligned)
}
