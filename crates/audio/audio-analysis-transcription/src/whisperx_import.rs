//! WhisperX JSON import without running external tools.

use crate::*;

/// Parses existing WhisperX JSON without running external tools.
pub fn import_whisperx_json(bytes: &[u8]) -> Result<TranscriptionContract> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| DetectError::InvalidArgument(error.to_string()))?;
    let object = value.as_object().ok_or_else(|| {
        DetectError::InvalidArgument("WhisperX JSON must be an object".to_string())
    })?;
    let language = object
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_string);
    let text = object
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_string);
    let source = object
        .get("source")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut attributes = unknown_json_attributes(
        object,
        &["language", "text", "source", "segments", "word_segments"],
    );
    if let Some(count) = object.get("segment_count").and_then(Value::as_u64) {
        attributes.insert("segment_count".to_string(), count.to_string());
    }
    let mut segments = object
        .get("segments")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            DetectError::InvalidArgument("WhisperX JSON must include a segments array".to_string())
        })?
        .iter()
        .enumerate()
        .map(|(position, segment)| whisperx_segment(segment, position as u64, language.clone()))
        .collect::<Result<Vec<_>>>()?;

    if segments.iter().all(|segment| segment.words().is_empty()) {
        if let Some(words) = object.get("word_segments").and_then(Value::as_array) {
            attach_flat_whisperx_words(&mut segments, words)?;
        }
    }

    let mut contract = normalize_transcription_contract(TranscriptionContract {
        text,
        language,
        segments,
        source,
        attributes,
    })?;
    for segment in &mut contract.segments {
        if segment.speaker.is_none() {
            segment.speaker = infer_segment_speaker(segment.words());
        }
    }
    contract.validate_strict()?;
    Ok(contract)
}

pub(crate) fn whisperx_segment(
    value: &Value,
    fallback_index: u64,
    language: Option<String>,
) -> Result<TranscriptSegmentContract> {
    let object = value.as_object().ok_or_else(|| {
        DetectError::InvalidArgument("WhisperX segment must be an object".to_string())
    })?;
    let mut segment = TranscriptSegmentContract::new(
        object
            .get("id")
            .or_else(|| object.get("index"))
            .and_then(Value::as_u64)
            .unwrap_or(fallback_index),
        object
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    )
    .with_time_range(
        number_field(object, &["start", "start_seconds", "startSeconds"]),
        number_field(object, &["end", "end_seconds", "endSeconds"]),
    )?
    .with_confidence(confidence_field(
        object,
        &["confidence", "score", "avg_logprob", "no_speech_prob"],
    ))?;
    segment.language = object
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or(language);
    segment.speaker = speaker_field(object);
    segment.attributes = unknown_json_attributes(
        object,
        &[
            "id",
            "index",
            "start",
            "start_seconds",
            "startSeconds",
            "end",
            "end_seconds",
            "endSeconds",
            "text",
            "language",
            "speaker",
            "speaker_label",
            "speakerLabel",
            "confidence",
            "score",
            "avg_logprob",
            "no_speech_prob",
            "words",
            "word_segments",
            "chars",
            "characters",
        ],
    );
    if let Some(words) = object
        .get("words")
        .or_else(|| object.get("word_segments"))
        .and_then(Value::as_array)
    {
        for word in words {
            segment.push_word(whisperx_word(word)?)?;
        }
    }
    if let Some(characters) = object
        .get("chars")
        .or_else(|| object.get("characters"))
        .and_then(Value::as_array)
    {
        for character in characters {
            segment.push_char(whisperx_char(character)?)?;
        }
    }
    if segment.speaker.is_none() {
        segment.speaker = infer_segment_speaker(segment.words());
    }
    Ok(segment)
}

pub(crate) fn whisperx_word(value: &Value) -> Result<TranscriptWordContract> {
    let object = value.as_object().ok_or_else(|| {
        DetectError::InvalidArgument("WhisperX word must be an object".to_string())
    })?;
    let mut word = TranscriptWordContract::new(
        object
            .get("word")
            .or_else(|| object.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default(),
    )
    .with_time_range(
        number_field(object, &["start", "start_seconds", "startSeconds"]),
        number_field(object, &["end", "end_seconds", "endSeconds"]),
    )?
    .with_confidence(confidence_field(
        object,
        &["confidence", "score", "probability"],
    ))?;
    word.speaker = speaker_field(object);
    word.attributes = unknown_json_attributes(
        object,
        &[
            "word",
            "text",
            "start",
            "start_seconds",
            "startSeconds",
            "end",
            "end_seconds",
            "endSeconds",
            "confidence",
            "score",
            "probability",
            "speaker",
            "speaker_label",
            "speakerLabel",
        ],
    );
    Ok(word)
}

pub(crate) fn whisperx_char(value: &Value) -> Result<TranscriptCharContract> {
    let object = value.as_object().ok_or_else(|| {
        DetectError::InvalidArgument("WhisperX char must be an object".to_string())
    })?;
    let mut character = TranscriptCharContract::new(
        object
            .get("char")
            .or_else(|| object.get("character"))
            .or_else(|| object.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default(),
    )
    .with_time_range(
        number_field(object, &["start", "start_seconds", "startSeconds"]),
        number_field(object, &["end", "end_seconds", "endSeconds"]),
    )?
    .with_confidence(confidence_field(
        object,
        &["confidence", "score", "probability"],
    ))?;
    character.attributes = unknown_json_attributes(
        object,
        &[
            "char",
            "character",
            "text",
            "start",
            "start_seconds",
            "startSeconds",
            "end",
            "end_seconds",
            "endSeconds",
            "confidence",
            "score",
            "probability",
        ],
    );
    Ok(character)
}

pub(crate) fn attach_flat_whisperx_words(
    segments: &mut [TranscriptSegmentContract],
    words: &[Value],
) -> Result<()> {
    for value in words {
        let word = whisperx_word(value)?;
        let midpoint = match (word.start_seconds(), word.end_seconds()) {
            (Some(start), Some(end)) => Some((start + end) * 0.5),
            (Some(start), None) => Some(start),
            (None, Some(end)) => Some(end),
            (None, None) => None,
        };
        let Some(segment) = segments.iter_mut().find(|segment| {
            midpoint
                .zip(segment.start_seconds().zip(segment.end_seconds()))
                .map(|(midpoint, (start, end))| midpoint >= start && midpoint <= end)
                .unwrap_or(false)
        }) else {
            continue;
        };
        segment.push_word(word)?;
    }
    for segment in segments {
        if segment.speaker.is_none() {
            segment.speaker = infer_segment_speaker(segment.words());
        }
    }
    Ok(())
}

pub(crate) fn infer_segment_speaker(words: &[TranscriptWordContract]) -> Option<String> {
    let mut scores: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    for word in words {
        let Some(speaker) = word
            .speaker
            .as_deref()
            .filter(|speaker| !speaker.is_empty())
        else {
            continue;
        };
        let duration = word
            .start_seconds()
            .zip(word.end_seconds())
            .map(|(start, end)| (end - start).max(0.0))
            .filter(|duration| duration.is_finite() && *duration > 0.0)
            .unwrap_or(1.0);
        let entry = scores.entry(speaker).or_insert((0.0, 0));
        entry.0 += duration;
        entry.1 += 1;
    }
    scores
        .into_iter()
        .max_by(|left, right| {
            left.1
                 .0
                .total_cmp(&right.1 .0)
                .then(left.1 .1.cmp(&right.1 .1))
                .then_with(|| right.0.cmp(left.0))
        })
        .map(|(speaker, _)| speaker.to_string())
}

pub(crate) fn speaker_field(object: &serde_json::Map<String, Value>) -> Option<String> {
    object
        .get("speaker")
        .or_else(|| object.get("speaker_label"))
        .or_else(|| object.get("speakerLabel"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

pub(crate) fn unknown_json_attributes(
    object: &serde_json::Map<String, Value>,
    known_fields: &[&str],
) -> BTreeMap<String, String> {
    object
        .iter()
        .filter(|(key, _)| !known_fields.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), json_attribute(value)))
        .collect()
}

pub(crate) fn json_attribute(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

pub(crate) fn number_field(object: &serde_json::Map<String, Value>, names: &[&str]) -> Option<f64> {
    names
        .iter()
        .find_map(|name| object.get(*name).and_then(Value::as_f64))
        .filter(|value| value.is_finite())
}

pub(crate) fn confidence_field(
    object: &serde_json::Map<String, Value>,
    names: &[&str],
) -> Option<f32> {
    for name in names {
        let Some(value) = object.get(*name).and_then(Value::as_f64) else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        return match *name {
            "avg_logprob" => Some(value.exp().clamp(0.0, 1.0) as f32),
            "no_speech_prob" => Some((1.0 - value).clamp(0.0, 1.0) as f32),
            _ => Some(value.clamp(0.0, 1.0) as f32),
        };
    }
    None
}
