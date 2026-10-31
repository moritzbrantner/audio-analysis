//! Transcript contract normalization, chunk offsets and alignment application.

use crate::*;

#[allow(dead_code)] // Shared by tests and optional native/alignment feature modules.
pub(crate) fn transcription_from_segments(
    source: Option<String>,
    language: Option<String>,
    segments: Vec<TranscriptSegmentContract>,
) -> Result<TranscriptionContract> {
    normalize_transcription_contract_with_segment_language(
        TranscriptionContract {
            text: None,
            language,
            segments,
            source,
            ..TranscriptionContract::default()
        },
        true,
    )
}

pub(crate) fn normalize_transcription_contract(
    contract: TranscriptionContract,
) -> Result<TranscriptionContract> {
    normalize_transcription_contract_with_segment_language(contract, false)
}

pub(crate) fn normalize_transcription_contract_with_segment_language(
    contract: TranscriptionContract,
    propagate_language: bool,
) -> Result<TranscriptionContract> {
    let language = normalize_optional_string(contract.language);
    let fallback_language = propagate_language.then_some(language.as_ref()).flatten();
    let segments = contract
        .segments
        .iter()
        .map(|segment| normalize_transcript_segment(segment, fallback_language))
        .collect::<Result<Vec<_>>>()?;
    let mut normalized = TranscriptionContract {
        text: normalize_optional_string(contract.text),
        language,
        segments,
        source: normalize_optional_string(contract.source),
        attributes: contract.attributes,
    };
    if normalized.text.is_none() {
        let joined = normalized
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !joined.is_empty() {
            normalized.text = Some(joined);
        }
    }
    normalized.validate()?;
    Ok(normalized)
}

pub(crate) fn normalize_transcript_segment(
    segment: &TranscriptSegmentContract,
    fallback_language: Option<&String>,
) -> Result<TranscriptSegmentContract> {
    let mut normalized = TranscriptSegmentContract::new(segment.index, segment.text.trim())
        .with_time_range(segment.start_seconds(), segment.end_seconds())?
        .with_confidence(segment.confidence())?;
    normalized.language =
        normalize_optional_string(segment.language.clone()).or_else(|| fallback_language.cloned());
    normalized.speaker = normalize_optional_string(segment.speaker.clone());
    normalized.is_final = segment.is_final;
    normalized.attributes = segment.attributes.clone();

    for word in segment.words() {
        let text = word.text.trim();
        if text.is_empty() {
            continue;
        }
        let mut normalized_word = TranscriptWordContract::new(text)
            .with_time_range(word.start_seconds(), word.end_seconds())?
            .with_confidence(word.confidence())?;
        normalized_word.speaker = normalize_optional_string(word.speaker.clone());
        normalized_word.attributes = word.attributes.clone();
        normalized.push_word(normalized_word)?;
    }
    for character in segment.chars() {
        if character.character.is_empty() {
            continue;
        }
        let mut normalized_character = TranscriptCharContract::new(&character.character)
            .with_time_range(character.start_seconds(), character.end_seconds())?
            .with_confidence(character.confidence())?;
        normalized_character.attributes = character.attributes.clone();
        normalized.push_char(normalized_character)?;
    }
    Ok(normalized)
}

pub(crate) fn rebuild_transcript_segment(
    segment: &TranscriptSegmentContract,
    start_seconds: Option<f64>,
    end_seconds: Option<f64>,
    words: Vec<TranscriptWordContract>,
    chars: Vec<TranscriptCharContract>,
) -> Result<TranscriptSegmentContract> {
    let mut rebuilt = TranscriptSegmentContract::new(segment.index, &segment.text)
        .with_time_range(start_seconds, end_seconds)?
        .with_confidence(segment.confidence())?;
    rebuilt.language = segment.language.clone();
    rebuilt.speaker = segment.speaker.clone();
    rebuilt.is_final = segment.is_final;
    rebuilt.attributes = segment.attributes.clone();
    for word in words {
        rebuilt.push_word(word)?;
    }
    for character in chars {
        rebuilt.push_char(character)?;
    }
    Ok(rebuilt)
}

pub(crate) fn rebuild_transcript_word(
    word: &TranscriptWordContract,
    text: String,
    start_seconds: Option<f64>,
    end_seconds: Option<f64>,
    confidence: Option<f32>,
) -> Result<TranscriptWordContract> {
    let mut rebuilt = TranscriptWordContract::new(text)
        .with_time_range(start_seconds, end_seconds)?
        .with_confidence(confidence)?;
    rebuilt.speaker = word.speaker.clone();
    rebuilt.attributes = word.attributes.clone();
    Ok(rebuilt)
}

pub(crate) fn rebuild_transcript_char(
    character: &TranscriptCharContract,
    value: String,
    start_seconds: Option<f64>,
    end_seconds: Option<f64>,
    confidence: Option<f32>,
) -> Result<TranscriptCharContract> {
    let mut rebuilt = TranscriptCharContract::new(value)
        .with_time_range(start_seconds, end_seconds)?
        .with_confidence(confidence)?;
    rebuilt.attributes = character.attributes.clone();
    Ok(rebuilt)
}

pub(crate) fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub(crate) fn sanitize_confidence(value: Option<f32>) -> Option<f32> {
    value.and_then(|confidence| confidence.is_finite().then(|| confidence.clamp(0.0, 1.0)))
}

pub(crate) fn offset_chunk_local_segments(
    transcript: &mut TranscriptionContract,
    chunks: &[SpeechSpan],
) -> Result<()> {
    if chunks.is_empty() || transcript.segments.len() != chunks.len() {
        return Ok(());
    }
    for (segment, chunk) in transcript.segments.iter_mut().zip(chunks) {
        if segment
            .attributes
            .get("timing")
            .is_some_and(|value| value == "global")
        {
            continue;
        }
        let offset = chunk.start_seconds;
        let words = segment
            .words()
            .iter()
            .map(|word| {
                rebuild_transcript_word(
                    word,
                    word.text.clone(),
                    word.start_seconds().map(|start| start + offset),
                    word.end_seconds().map(|end| end + offset),
                    word.confidence(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut rebuilt = rebuild_transcript_segment(
            segment,
            segment.start_seconds().map(|start| start + offset),
            segment.end_seconds().map(|end| end + offset),
            words,
            segment.chars().to_vec(),
        )?;
        rebuilt
            .attributes
            .insert("timing".to_string(), "global".to_string());
        *segment = rebuilt;
    }
    Ok(())
}

pub(crate) fn apply_alignment_words(
    transcript: &mut TranscriptionContract,
    words: &[AlignedWord],
) -> Result<()> {
    for aligned in words {
        if !aligned.start_seconds.is_finite()
            || !aligned.end_seconds.is_finite()
            || aligned.end_seconds < aligned.start_seconds
        {
            return Err(model_output_mismatch(
                "alignment output contains invalid word timing",
            ));
        }
        let segment_index = transcript
            .segments
            .iter()
            .position(|segment| segment.index == aligned.segment_index)
            .ok_or_else(|| model_output_mismatch("alignment output references unknown segment"))?;
        let segment = &transcript.segments[segment_index];
        let mut words = segment.words().to_vec();
        while words.len() <= aligned.word_index {
            words.push(TranscriptWordContract::new(String::new()));
        }
        let word = &words[aligned.word_index];
        let text = if word.text.trim().is_empty() {
            aligned.text.clone()
        } else {
            word.text.clone()
        };
        words[aligned.word_index] = rebuild_transcript_word(
            word,
            text,
            Some(aligned.start_seconds),
            Some(aligned.end_seconds),
            sanitize_confidence(aligned.confidence),
        )?;
        let timed_words = words
            .iter()
            .filter_map(|word| word.start_seconds().zip(word.end_seconds()))
            .collect::<Vec<_>>();
        let (start_seconds, end_seconds) = match (timed_words.first(), timed_words.last()) {
            (Some((start, _)), Some((_, end))) => (Some(*start), Some(*end)),
            _ => (segment.start_seconds(), segment.end_seconds()),
        };
        transcript.segments[segment_index] = rebuild_transcript_segment(
            segment,
            start_seconds,
            end_seconds,
            words,
            segment.chars().to_vec(),
        )?;
    }
    Ok(())
}

// Independently aligned overlapping ASR intervals can change their relative starts.
// Restore chronology only after attaching every word/character by stable segment ID:
// move complete segments, never timestamps or text independently.
pub(crate) fn order_aligned_segments(transcript: &mut TranscriptionContract) {
    if transcript
        .segments
        .iter()
        .filter_map(|segment| segment.start_seconds())
        .is_sorted_by(|previous, next| previous <= next)
    {
        return;
    }
    transcript.segments.sort_by(|left, right| {
        match (left.start_seconds(), right.start_seconds()) {
            (Some(left), Some(right)) => left.total_cmp(&right),
            (None, Some(_)) => std::cmp::Ordering::Less,
            (Some(_), None) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    });
    transcript.text = Some(
        transcript
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    );
}

pub(crate) fn apply_alignment_chars(
    transcript: &mut TranscriptionContract,
    chars: &[AlignedChar],
) -> Result<()> {
    for aligned in chars {
        if let Some((start, end)) = aligned.start_seconds.zip(aligned.end_seconds) {
            if !start.is_finite() || !end.is_finite() || end < start {
                return Err(model_output_mismatch(
                    "alignment output contains invalid char timing",
                ));
            }
        }
        let segment_index = transcript
            .segments
            .iter()
            .position(|segment| segment.index == aligned.segment_index)
            .ok_or_else(|| model_output_mismatch("alignment output references unknown segment"))?;
        let segment = &transcript.segments[segment_index];
        let mut chars = segment.chars().to_vec();
        while chars.len() <= aligned.char_index {
            chars.push(TranscriptCharContract::new(String::new()));
        }
        let character = &chars[aligned.char_index];
        let value = if character.character.is_empty() {
            aligned.character.clone()
        } else {
            character.character.clone()
        };
        chars[aligned.char_index] = rebuild_transcript_char(
            character,
            value,
            aligned.start_seconds,
            aligned.end_seconds,
            sanitize_confidence(aligned.confidence),
        )?;
        transcript.segments[segment_index] = rebuild_transcript_segment(
            segment,
            segment.start_seconds(),
            segment.end_seconds(),
            segment.words().to_vec(),
            chars,
        )?;
    }
    Ok(())
}
