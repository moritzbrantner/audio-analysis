//! Silero speech timestamps over a stream of 512-sample model windows.
//!
//! [`SileroSegmenter`] is the Silero `get_speech_timestamps` state machine (hysteresis,
//! minimum speech and silence, maximum-speech splitting, padding) fed one window probability at a
//! time. The finite provider and [`SileroVadStream`] both run it, over the same model runner, so a
//! stream fed the same 16 kHz samples in any chunking finalizes exactly the spans the finite
//! provider returns for the whole buffer.
//!
//! A span is final once nothing later can move it:
//! - its start is fixed when it opens (padding against the previous span is already known);
//! - its end is fixed once a next span is certain (closed, or open and already longer than
//!   `min_speech`), or once no span can start within `2 × speech_pad` of its raw end, because only
//!   a closer next span would shorten the padding.
//!
//! The lookahead after a span's raw end is therefore `min_silence` (to close it) plus
//! `2 × speech_pad`, or up to `min_speech` into a next span that starts within that distance, plus
//! up to one 32 ms model window. A span still open, or still inside that
//! lookahead, is provisional: it may grow, shrink, split, appear or disappear (a span shorter than
//! `min_speech` is dropped when it closes). Finalized spans are never revised.
#![cfg_attr(not(feature = "silero-vad"), allow(dead_code, unreachable_pub))]

use std::collections::VecDeque;

use media_core::{DetectError, Result};
use serde::Serialize;

use crate::silero_vad::{
    SileroProbabilityRunner, SileroTimestampDetector, SILERO_SAMPLE_RATE, SILERO_WINDOW_SAMPLES,
};
use crate::SpeechSpan;

/// One incremental VAD observation: what changed since the previous update of the same stream.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VadStreamUpdate {
    /// Increases by one with every update of a stream, starting at 1.
    pub revision: u64,
    /// Seconds of audio received so far, from the first pushed sample.
    pub received_seconds: f64,
    /// Seconds of audio the model has analyzed: whole 32 ms windows until the stream finishes.
    pub analyzed_seconds: f64,
    /// Speech and non-speech before this time is final and will not change.
    pub stable_until_seconds: f64,
    /// Spans finalized by this update, in order. Earlier updates' spans are never repeated or
    /// revised.
    pub finalized: Vec<FinalizedSpeechSpan>,
    /// The complete provisional window after `stable_until_seconds`; it replaces the previous
    /// update's provisional spans. Empty once the stream has finished.
    pub provisional: Vec<SpeechSpan>,
    /// `true` for the update returned by [`SileroVadStream::finish`].
    pub finished: bool,
}

/// A finalized speech span and its position among the stream's finalized spans.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FinalizedSpeechSpan {
    /// 0 for the first finalized span of a stream, then 1, 2, ...
    pub index: usize,
    #[serde(flatten)]
    pub span: SpeechSpan,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RawSpeech {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// The Silero speech-timestamp state machine over one probability per model window.
pub(crate) struct SileroSegmenter {
    threshold: f32,
    neg_threshold: f32,
    min_speech_samples: f64,
    speech_pad_samples: usize,
    min_silence_samples: f64,
    min_silence_at_max_speech: f64,
    max_speech_samples: f64,
    // Silero's loop state.
    triggered: bool,
    current: RawSpeech,
    temp_end: usize,
    prev_end: usize,
    next_start: usize,
    possible_ends: Vec<(usize, usize)>,
    /// Probabilities consumed so far.
    windows: usize,
    /// Probabilities of windows `first_window..windows` that a span not yet finalized can overlap.
    probabilities: VecDeque<f32>,
    first_window: usize,
    /// Closed raw spans not yet finalized, in order.
    closed: VecDeque<RawSpeech>,
    /// Raw end of the last finalized raw span (dropped ones included), for the next start padding.
    previous_raw_end: Option<usize>,
    /// Padded end of the last finalized span that was emitted.
    last_final_end: usize,
    emitted: usize,
}

impl SileroSegmenter {
    pub(crate) fn new(detector: &SileroTimestampDetector) -> Self {
        let rate = SILERO_SAMPLE_RATE as f64;
        let speech_pad_samples = (rate * detector.speech_pad_ms as f64 / 1000.0) as usize;
        Self {
            threshold: detector.threshold,
            neg_threshold: (detector.threshold - 0.15).max(0.01),
            min_speech_samples: rate * detector.min_speech_duration_ms as f64 / 1000.0,
            speech_pad_samples,
            min_silence_samples: rate * detector.min_silence_duration_ms as f64 / 1000.0,
            min_silence_at_max_speech: rate * 98.0 / 1000.0,
            max_speech_samples: (rate * detector.max_speech_duration_seconds
                - SILERO_WINDOW_SAMPLES as f64
                - 2.0 * speech_pad_samples as f64)
                .max(0.0),
            triggered: false,
            current: RawSpeech::default(),
            temp_end: 0,
            prev_end: 0,
            next_start: 0,
            possible_ends: Vec::new(),
            windows: 0,
            probabilities: VecDeque::new(),
            first_window: 0,
            closed: VecDeque::new(),
            previous_raw_end: None,
            last_final_end: 0,
            emitted: 0,
        }
    }

    /// Consumes the next window probability and returns the spans this finalizes.
    pub(crate) fn push_probability(
        &mut self,
        speech_prob: f32,
        finalized: &mut Vec<FinalizedSpeechSpan>,
    ) -> Result<()> {
        if !speech_prob.is_finite() {
            return Err(DetectError::InvalidArgument(
                "silero VAD probabilities must be finite".to_string(),
            ));
        }
        let cur_sample = SILERO_WINDOW_SAMPLES * self.windows;
        self.probabilities.push_back(speech_prob);
        self.windows += 1;
        self.step(speech_prob, cur_sample);
        self.finalize_ready(None, finalized)?;
        self.drop_unneeded_probabilities();
        Ok(())
    }

    /// Ends the stream after `audio_len_samples` samples and returns the remaining spans.
    pub(crate) fn finish(
        &mut self,
        audio_len_samples: usize,
        finalized: &mut Vec<FinalizedSpeechSpan>,
    ) -> Result<()> {
        if self.windows > 0
            && self.triggered
            && self.current.start < audio_len_samples
            && audio_len_samples.saturating_sub(self.current.start) as f64 > self.min_speech_samples
        {
            self.current.end = audio_len_samples;
            self.closed.push_back(self.current);
        }
        self.triggered = false;
        self.current = RawSpeech::default();
        self.finalize_ready(Some(audio_len_samples), finalized)?;
        self.last_final_end = self.last_final_end.max(audio_len_samples);
        self.probabilities.clear();
        self.first_window = self.windows;
        Ok(())
    }

    /// One iteration of Silero's `get_speech_timestamps` loop.
    fn step(&mut self, speech_prob: f32, cur_sample: usize) {
        if speech_prob >= self.threshold && self.temp_end != 0 {
            let silence_duration = cur_sample.saturating_sub(self.temp_end);
            if silence_duration as f64 > self.min_silence_at_max_speech {
                self.possible_ends.push((self.temp_end, silence_duration));
            }
            self.temp_end = 0;
            if self.next_start < self.prev_end {
                self.next_start = cur_sample;
            }
        }

        if speech_prob >= self.threshold && !self.triggered {
            self.triggered = true;
            self.current = RawSpeech {
                start: cur_sample,
                end: 0,
            };
            return;
        }

        if self.triggered
            && (cur_sample.saturating_sub(self.current.start)) as f64 > self.max_speech_samples
        {
            if let Some((best_end, duration)) = self
                .possible_ends
                .iter()
                .copied()
                .max_by_key(|(_, duration)| *duration)
            {
                self.current.end = best_end;
                self.closed.push_back(self.current);
                self.current = RawSpeech::default();
                self.next_start = best_end + duration;
                if self.next_start < best_end + cur_sample {
                    self.current.start = self.next_start;
                } else {
                    self.triggered = false;
                }
            } else {
                self.current.end = cur_sample;
                self.closed.push_back(self.current);
                self.current = RawSpeech::default();
                self.triggered = false;
            }
            self.prev_end = 0;
            self.next_start = 0;
            self.temp_end = 0;
            self.possible_ends.clear();
            return;
        }

        if speech_prob < self.neg_threshold && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = cur_sample;
            }
            let silence_duration = cur_sample.saturating_sub(self.temp_end);
            if (silence_duration as f64) < self.min_silence_samples {
                return;
            }
            self.current.end = self.temp_end;
            if (self.current.end.saturating_sub(self.current.start)) as f64
                > self.min_speech_samples
            {
                self.closed.push_back(self.current);
            }
            self.current = RawSpeech::default();
            self.prev_end = 0;
            self.next_start = 0;
            self.temp_end = 0;
            self.triggered = false;
            self.possible_ends.clear();
        }
    }

    /// Finalizes every closed span whose padded bounds can no longer change. `audio_len` is the
    /// total length once the stream has finished.
    fn finalize_ready(
        &mut self,
        audio_len: Option<usize>,
        finalized: &mut Vec<FinalizedSpeechSpan>,
    ) -> Result<()> {
        let analyzed = self.windows * SILERO_WINDOW_SAMPLES;
        while let Some(&raw) = self.closed.front() {
            let previous = self.previous_raw_end;
            let pad_reach = raw.end + 2 * self.speech_pad_samples;
            // `usize::MAX` as the length: a later span exists past this one's padding, so the end
            // padding is not clamped to the audio.
            let (start, end) = if let Some(next) = self.closed.get(1) {
                self.padded(previous, raw, Some(next.start), usize::MAX)
            } else if let Some(len) = audio_len {
                self.padded(previous, raw, None, len)
            } else if self.triggered {
                if self.open_span_is_certain() {
                    self.padded(previous, raw, Some(self.current.start), usize::MAX)
                } else if self.current.start >= pad_reach {
                    // Whether or not the open span survives, every later start is too far away
                    // to shorten this span's end padding.
                    self.padded(previous, raw, None, usize::MAX)
                } else {
                    break;
                }
            } else if analyzed >= pad_reach {
                // A later span starts at or after `analyzed`, too far away to shorten the padding.
                self.padded(previous, raw, None, analyzed)
            } else {
                break;
            };
            self.closed.pop_front();
            self.previous_raw_end = Some(raw.end);
            if end > start {
                let span = SpeechSpan::new(
                    start as f64 / SILERO_SAMPLE_RATE as f64,
                    end as f64 / SILERO_SAMPLE_RATE as f64,
                    self.max_probability(start, end),
                )?;
                finalized.push(FinalizedSpeechSpan {
                    index: self.emitted,
                    span,
                });
                self.emitted += 1;
                self.last_final_end = self.last_final_end.max(end);
            }
        }
        Ok(())
    }

    /// Whether the open span will be kept whatever follows: its end can be no earlier than a
    /// pending silence start (or the next window), and Silero keeps a span longer than
    /// `min_speech`; a span cut at the maximum speech duration is always kept.
    fn open_span_is_certain(&self) -> bool {
        let analyzed = self.windows * SILERO_WINDOW_SAMPLES;
        let earliest_end = if self.temp_end != 0 {
            self.temp_end
        } else {
            analyzed
        };
        earliest_end.saturating_sub(self.current.start) as f64 > self.min_speech_samples
    }

    /// Silero's padding of one raw span between its finalized predecessor and its successor.
    fn padded(
        &self,
        previous_raw_end: Option<usize>,
        raw: RawSpeech,
        next_start: Option<usize>,
        len: usize,
    ) -> (usize, usize) {
        let pad = self.speech_pad_samples;
        let start = match previous_raw_end {
            Some(previous_end) => {
                let silence = raw.start.saturating_sub(previous_end);
                if silence < 2 * pad {
                    raw.start.saturating_sub(silence - silence / 2)
                } else {
                    raw.start.saturating_sub(pad)
                }
            }
            None => raw.start.saturating_sub(pad),
        };
        let end = match next_start {
            Some(next_start) if next_start.saturating_sub(raw.end) < 2 * pad => {
                raw.end + next_start.saturating_sub(raw.end) / 2
            }
            _ => (raw.end + pad).min(len),
        };
        (start, end)
    }

    fn max_probability(&self, start: usize, end: usize) -> f32 {
        self.probabilities
            .iter()
            .enumerate()
            .filter_map(|(offset, probability)| {
                let window_start = (self.first_window + offset) * SILERO_WINDOW_SAMPLES;
                let window_end = window_start + SILERO_WINDOW_SAMPLES;
                (window_end > start && window_start < end).then_some(*probability)
            })
            .fold(0.0, f32::max)
    }

    /// The earliest sample a span not yet finalized can start at.
    fn unfinalized_low(&self) -> usize {
        let analyzed = self.windows * SILERO_WINDOW_SAMPLES;
        let mut low = analyzed;
        if let Some(front) = self.closed.front() {
            low = low.min(front.start);
        }
        if self.triggered {
            low = low.min(self.current.start);
        }
        // Start padding moves a start back by at most `speech_pad`.
        low.saturating_sub(self.speech_pad_samples)
    }

    fn drop_unneeded_probabilities(&mut self) {
        let low = self.unfinalized_low();
        while !self.probabilities.is_empty()
            && (self.first_window + 1) * SILERO_WINDOW_SAMPLES <= low
        {
            self.probabilities.pop_front();
            self.first_window += 1;
        }
    }

    pub(crate) fn stable_until_samples(&self) -> usize {
        self.unfinalized_low().max(self.last_final_end)
    }

    /// Model windows whose probabilities are retained (bounded by the open span plus padding).
    pub(crate) fn retained_windows(&self) -> usize {
        self.probabilities.len()
    }

    /// The provisional spans: closed spans in their lookahead and the open span up to the analyzed
    /// audio, padded as they would be if the audio ended now.
    pub(crate) fn provisional(&self) -> Result<Vec<SpeechSpan>> {
        let analyzed = self.windows * SILERO_WINDOW_SAMPLES;
        let mut raws = self.closed.iter().copied().collect::<Vec<_>>();
        if self.triggered && analyzed > self.current.start {
            raws.push(RawSpeech {
                start: self.current.start,
                end: analyzed,
            });
        }
        let mut spans = Vec::with_capacity(raws.len());
        let mut previous_raw_end = self.previous_raw_end;
        for (index, raw) in raws.iter().enumerate() {
            let next_start = raws.get(index + 1).map(|next| next.start);
            let (start, end) = self.padded(previous_raw_end, *raw, next_start, analyzed);
            previous_raw_end = Some(raw.end);
            if end > start {
                spans.push(SpeechSpan::new(
                    start as f64 / SILERO_SAMPLE_RATE as f64,
                    end as f64 / SILERO_SAMPLE_RATE as f64,
                    self.max_probability(start, end),
                )?);
            }
        }
        Ok(spans)
    }
}

/// Window framing shared by the finite provider and the stream: buffers samples into 512-sample
/// windows and zero-pads the final partial window, exactly as the finite provider always has.
pub(crate) struct SileroWindowFeed {
    pub(crate) segmenter: SileroSegmenter,
    pending: Vec<f32>,
    received: usize,
}

impl SileroWindowFeed {
    pub(crate) fn new(detector: &SileroTimestampDetector) -> Self {
        Self {
            segmenter: SileroSegmenter::new(detector),
            pending: Vec::with_capacity(SILERO_WINDOW_SAMPLES),
            received: 0,
        }
    }

    pub(crate) fn received(&self) -> usize {
        self.received
    }

    pub(crate) fn windows(&self) -> usize {
        self.segmenter.windows
    }

    pub(crate) fn push(
        &mut self,
        runner: &mut dyn SileroProbabilityRunner,
        samples: &[f32],
        finalized: &mut Vec<FinalizedSpeechSpan>,
    ) -> Result<()> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(DetectError::InvalidArgument(
                "silero VAD samples must be finite".to_string(),
            ));
        }
        let mut rest = samples;
        while !rest.is_empty() {
            let take = (SILERO_WINDOW_SAMPLES - self.pending.len()).min(rest.len());
            self.pending.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            self.received += take;
            if self.pending.len() == SILERO_WINDOW_SAMPLES {
                let probability = runner.window_probability(&self.pending)?;
                self.pending.clear();
                self.segmenter.push_probability(probability, finalized)?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        runner: &mut dyn SileroProbabilityRunner,
        finalized: &mut Vec<FinalizedSpeechSpan>,
    ) -> Result<()> {
        if !self.pending.is_empty() {
            self.pending.resize(SILERO_WINDOW_SAMPLES, 0.0);
            let probability = runner.window_probability(&self.pending)?;
            self.pending.clear();
            self.segmenter.push_probability(probability, finalized)?;
        }
        self.segmenter.finish(self.received, finalized)
    }
}

/// Incremental Silero VAD over 16 kHz mono PCM, with the finite provider's exact results.
///
/// Push chunks of any length as they are captured; every call returns the spans it finalized and
/// the current provisional window. [`finish`](Self::finish) flushes the final partial window and
/// finalizes everything. Retained state is one partial window, the model's recurrent state, and the
/// probabilities of the spans not yet final (one `f32` per 32 ms, at most
/// `max_speech_duration_seconds + min_speech + 2 × speech_pad`). Each push reads only its own
/// samples.
///
/// After an error the stream must be dropped: the model state no longer matches the input.
pub struct SileroVadStream {
    runner: Box<dyn SileroProbabilityRunner + Send>,
    feed: SileroWindowFeed,
    revision: u64,
    failed: bool,
    finished: bool,
}

impl SileroVadStream {
    pub(crate) fn new_for_runner(
        detector: SileroTimestampDetector,
        mut runner: Box<dyn SileroProbabilityRunner + Send>,
    ) -> Self {
        runner.reset();
        Self {
            runner,
            feed: SileroWindowFeed::new(&detector),
            revision: 0,
            failed: false,
            finished: false,
        }
    }

    /// Loads the Silero model and starts an empty stream.
    #[cfg(feature = "silero-vad")]
    pub fn from_options(options: crate::SileroVadOptions) -> Result<Self> {
        let detector = options.detector();
        let runner = crate::silero_vad::OnnxSileroRunner::from_options(options)?;
        Ok(Self::new_for_runner(detector, Box::new(runner)))
    }

    /// The provider identifier, as the finite provider reports it.
    pub fn provider_id(&self) -> &'static str {
        "silero-vad"
    }

    /// Sample rate the stream requires.
    pub fn sample_rate(&self) -> u32 {
        SILERO_SAMPLE_RATE
    }

    /// Model windows whose probabilities the stream currently retains.
    pub fn retained_windows(&self) -> usize {
        self.feed.segmenter.retained_windows()
    }

    /// Adds the next chunk of 16 kHz mono samples.
    pub fn push(&mut self, samples: &[f32]) -> Result<VadStreamUpdate> {
        self.ensure_usable()?;
        let mut finalized = Vec::new();
        let result = self
            .feed
            .push(self.runner.as_mut(), samples, &mut finalized);
        self.update(result, finalized, false)
    }

    /// Flushes the final partial window and finalizes every remaining span.
    pub fn finish(&mut self) -> Result<VadStreamUpdate> {
        self.ensure_usable()?;
        let mut finalized = Vec::new();
        let result = self.feed.finish(self.runner.as_mut(), &mut finalized);
        self.finished = true;
        self.update(result, finalized, true)
    }

    fn ensure_usable(&self) -> Result<()> {
        if self.failed {
            return Err(DetectError::InvalidArgument(
                "silero VAD stream failed earlier and must be restarted".to_string(),
            ));
        }
        if self.finished {
            return Err(DetectError::InvalidArgument(
                "silero VAD stream has already finished".to_string(),
            ));
        }
        Ok(())
    }

    fn update(
        &mut self,
        result: Result<()>,
        finalized: Vec<FinalizedSpeechSpan>,
        finished: bool,
    ) -> Result<VadStreamUpdate> {
        if let Err(error) = result {
            self.failed = true;
            return Err(error);
        }
        self.revision += 1;
        let rate = SILERO_SAMPLE_RATE as f64;
        let received = self.feed.received();
        let analyzed = if finished {
            received
        } else {
            self.feed.windows() * SILERO_WINDOW_SAMPLES
        };
        Ok(VadStreamUpdate {
            revision: self.revision,
            received_seconds: received as f64 / rate,
            analyzed_seconds: analyzed as f64 / rate,
            stable_until_seconds: (self.feed.segmenter.stable_until_samples().min(received)) as f64
                / rate,
            finalized,
            provisional: if finished {
                Vec::new()
            } else {
                self.feed.segmenter.provisional()?
            },
            finished,
        })
    }
}

#[cfg(test)]
#[path = "silero_stream_reference.rs"]
mod reference;

#[cfg(test)]
mod tests {
    use super::reference::ReferenceTimestamps;
    use super::*;
    use crate::silero_vad::SileroVadTranscriptionProvider;
    use crate::{SpeakerAudio, VadOptions, VoiceActivityDetector};

    /// Deterministic linear congruential generator, so the cases are reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    /// A recurrent stand-in for the model: the probability depends on the window's level and on
    /// the previous windows, so window alignment and state handling both matter.
    struct LevelRunner {
        state: f32,
    }

    impl SileroProbabilityRunner for LevelRunner {
        fn reset(&mut self) {
            self.state = 0.0;
        }

        fn window_probability(&mut self, window: &[f32]) -> Result<f32> {
            assert_eq!(window.len(), SILERO_WINDOW_SAMPLES);
            let level = window.iter().map(|sample| sample.abs()).sum::<f32>() / 512.0;
            self.state = 0.6 * self.state + 0.4 * (level * 4.0).min(1.0);
            Ok(self.state)
        }
    }

    fn detector(
        pad_ms: usize,
        max_speech_seconds: f64,
        min_silence_ms: usize,
    ) -> SileroTimestampDetector {
        SileroTimestampDetector {
            threshold: 0.5,
            max_speech_duration_seconds: max_speech_seconds,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: min_silence_ms,
            speech_pad_ms: pad_ms,
        }
    }

    fn detectors() -> Vec<SileroTimestampDetector> {
        vec![
            detector(30, 30.0, 100),
            detector(0, f64::INFINITY, 0),
            detector(100, 0.45, 100),
            detector(200, 1.5, 300),
            detector(30, 0.3, 50),
        ]
    }

    /// Bursts of speech-like probability with jitter, gaps of varying length and a few spikes.
    fn bursty_probabilities(rng: &mut Lcg, windows: usize) -> Vec<f32> {
        let mut probabilities = Vec::with_capacity(windows);
        let mut speaking = false;
        while probabilities.len() < windows {
            let run = 1 + (rng.next() * 40.0) as usize;
            for _ in 0..run.min(windows - probabilities.len()) {
                let base = if speaking { 0.75 } else { 0.15 };
                probabilities.push((base + (rng.next() - 0.5) * 0.5).clamp(0.0, 1.0));
            }
            speaking = !speaking;
        }
        probabilities
    }

    fn bursty_audio(rng: &mut Lcg, samples: usize) -> Vec<f32> {
        let mut audio = Vec::with_capacity(samples);
        let mut loud = false;
        while audio.len() < samples {
            let run = 200 + (rng.next() * 12_000.0) as usize;
            let amplitude = if loud { 0.4 } else { 0.01 };
            for _ in 0..run.min(samples - audio.len()) {
                audio.push((rng.next() - 0.5) * 2.0 * amplitude);
            }
            loud = !loud;
        }
        audio
    }

    fn segment(
        detector: &SileroTimestampDetector,
        probabilities: &[f32],
        len: usize,
    ) -> Vec<SpeechSpan> {
        let mut segmenter = SileroSegmenter::new(detector);
        let mut finalized = Vec::new();
        for probability in probabilities {
            segmenter
                .push_probability(*probability, &mut finalized)
                .unwrap();
        }
        segmenter.finish(len, &mut finalized).unwrap();
        finalized.into_iter().map(|span| span.span).collect()
    }

    fn stream(detector: SileroTimestampDetector) -> SileroVadStream {
        SileroVadStream::new_for_runner(detector, Box::new(LevelRunner { state: 0.0 }))
    }

    fn finite(detector: SileroTimestampDetector, audio: &[f32]) -> Vec<SpeechSpan> {
        let mut provider = SileroVadTranscriptionProvider::new_for_runner(
            detector,
            Box::new(LevelRunner { state: 0.0 }),
            Vec::new(),
        );
        provider
            .detect_speech(
                &SpeakerAudio::mono(audio, SILERO_SAMPLE_RATE).unwrap(),
                &VadOptions::default(),
            )
            .unwrap()
            .segments
    }

    #[test]
    fn segmenter_matches_the_previous_finite_port() {
        let mut rng = Lcg(7);
        for detector in detectors() {
            for case in 0..60 {
                let windows = (rng.next() * 700.0) as usize;
                let probabilities = bursty_probabilities(&mut rng, windows);
                // The last window may be partial.
                let len =
                    (windows * SILERO_WINDOW_SAMPLES).saturating_sub((rng.next() * 511.0) as usize);
                let expected = detector.reference_timestamps(&probabilities, len).unwrap();
                assert_eq!(
                    segment(&detector, &probabilities, len),
                    expected,
                    "case {case} with {detector:?}"
                );
            }
        }
    }

    #[test]
    fn stream_finalizes_the_finite_spans_in_any_chunking() {
        let mut rng = Lcg(11);
        for detector in detectors() {
            for case in 0..4 {
                let length = 16_000 * (2 + case * 3) + (rng.next() * 511.0) as usize;
                let audio = bursty_audio(&mut rng, length);
                let expected = finite(detector, &audio);
                assert!(!expected.is_empty(), "fixture must contain speech");
                for chunk in [1usize, 160, 511, 512, 513, 1_600, 4_800, length] {
                    let mut stream = stream(detector);
                    let mut finalized = Vec::new();
                    let mut stable_until = 0.0;
                    let mut revision = 0;
                    let mut updates = audio
                        .chunks(chunk)
                        .map(|samples| stream.push(samples).unwrap())
                        .collect::<Vec<_>>();
                    updates.push(stream.finish().unwrap());
                    for update in updates {
                        revision += 1;
                        assert_eq!(update.revision, revision);
                        assert!(update.stable_until_seconds >= stable_until);
                        stable_until = update.stable_until_seconds;
                        for span in &update.finalized {
                            assert_eq!(span.index, finalized.len());
                            assert!(span.span.end_seconds <= stable_until + 1e-12);
                            finalized.push(span.span);
                        }
                        for span in &update.provisional {
                            assert!(span.end_seconds > stable_until);
                        }
                    }
                    assert_eq!(finalized, expected, "chunk {chunk}, {detector:?}");
                }
            }
        }
    }

    #[test]
    fn a_short_provisional_span_disappears_when_it_closes_too_short() {
        let mut stream = stream(detector(30, 30.0, 100));
        // About 130 ms loud (under the 250 ms minimum), then silence.
        let mut audio = vec![0.4_f32; 2_048];
        audio.extend(vec![0.0_f32; 16_000]);
        let loud = stream.push(&audio[..2_048]).unwrap();
        assert_eq!(loud.provisional.len(), 1);
        assert!(loud.finalized.is_empty());
        let quiet = stream.push(&audio[2_048..]).unwrap();
        assert!(quiet.provisional.is_empty());
        assert!(quiet.finalized.is_empty());
        let done = stream.finish().unwrap();
        assert!(done.finalized.is_empty());
        assert_eq!(done.stable_until_seconds, done.received_seconds);
    }

    #[test]
    fn empty_and_silent_streams_finalize_no_speech() {
        let mut empty = stream(detector(30, 30.0, 100));
        let nothing = empty.push(&[]).unwrap();
        assert_eq!(nothing.received_seconds, 0.0);
        let done = empty.finish().unwrap();
        assert!(done.finished && done.finalized.is_empty() && done.provisional.is_empty());

        let mut silent = stream(detector(30, 30.0, 100));
        for _ in 0..100 {
            let update = silent.push(&[0.0; 1_600]).unwrap();
            assert!(update.finalized.is_empty() && update.provisional.is_empty());
            assert!(silent.retained_windows() <= 2);
        }
        let done = silent.finish().unwrap();
        assert!(done.finalized.is_empty());
        assert_eq!(done.received_seconds, 10.0);
    }

    #[test]
    fn retained_probabilities_are_bounded_by_the_open_span() {
        // Continuous speech is split every 2 s; the stream keeps at most that plus the 250 ms
        // needed to know the next span survives and the padding.
        let mut stream = stream(detector(30, 2.0, 100));
        let bound = ((2.0 + 0.25 + 0.06) * 16_000.0 / 512.0) as usize + 2;
        let mut finalized = 0;
        for _ in 0..300 {
            let update = stream.push(&[0.4; 1_600]).unwrap();
            finalized += update.finalized.len();
            assert!(
                stream.retained_windows() <= bound,
                "{}",
                stream.retained_windows()
            );
        }
        assert!(finalized >= 10);
    }

    #[test]
    fn a_failed_or_finished_stream_rejects_more_audio() {
        let mut failed = stream(detector(30, 30.0, 100));
        assert!(failed.push(&[0.0, f32::NAN]).is_err());
        let error = failed.push(&[0.0]).unwrap_err().to_string();
        assert!(error.contains("must be restarted"), "{error}");

        let mut finished = stream(detector(30, 30.0, 100));
        finished.finish().unwrap();
        assert!(finished.push(&[0.0]).is_err());
        assert!(finished.finish().is_err());
    }

    /// Real Silero model: the stream and the finite provider agree exactly on the same samples.
    ///
    /// `NATIVE_WHISPERX_SILERO_ONNX` names the model and `SILERO_VAD_AUDIO` a 16 kHz mono 16-bit
    /// WAV; ONNX Runtime comes from `ORT_DYLIB_PATH`.
    #[cfg(feature = "silero-vad")]
    #[test]
    #[ignore = "requires a caller-owned Silero model and WAV"]
    fn real_silero_stream_matches_the_finite_provider() {
        let model_path = std::path::PathBuf::from(
            std::env::var_os("NATIVE_WHISPERX_SILERO_ONNX").expect("NATIVE_WHISPERX_SILERO_ONNX"),
        );
        let audio_path = std::env::var_os("SILERO_VAD_AUDIO").expect("SILERO_VAD_AUDIO");
        let mut reader = hound::WavReader::open(audio_path).expect("WAV");
        assert_eq!(reader.spec().sample_rate, SILERO_SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        let audio = reader
            .samples::<i16>()
            .map(|sample| f32::from(sample.expect("sample")) / 32_768.0)
            .collect::<Vec<_>>();
        let options = crate::SileroVadOptions {
            model_path,
            input_name: None,
            output_name: None,
            threshold: 0.5,
            max_speech_duration_seconds: 30.0,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
        };
        let mut provider =
            SileroVadTranscriptionProvider::from_options(options.clone(), Vec::new()).unwrap();
        let started = std::time::Instant::now();
        let expected = provider
            .detect_speech(
                &SpeakerAudio::mono(&audio, SILERO_SAMPLE_RATE).unwrap(),
                &VadOptions::default(),
            )
            .unwrap()
            .segments;
        eprintln!(
            "finite: {} spans over {:.1} s audio in {:?}",
            expected.len(),
            audio.len() as f64 / 16_000.0,
            started.elapsed()
        );
        assert!(!expected.is_empty());
        for chunk in [160usize, 512, 1_000, 4_800, 16_000] {
            let mut stream = SileroVadStream::from_options(options.clone()).unwrap();
            let started = std::time::Instant::now();
            let mut finalized = Vec::new();
            let mut max_push = std::time::Duration::ZERO;
            let mut max_retained = 0;
            for samples in audio.chunks(chunk) {
                let push = std::time::Instant::now();
                finalized.extend(stream.push(samples).unwrap().finalized);
                max_push = max_push.max(push.elapsed());
                max_retained = max_retained.max(stream.retained_windows());
            }
            finalized.extend(stream.finish().unwrap().finalized);
            eprintln!(
                "chunk {chunk}: {:?} total, slowest push {max_push:?}, max retained windows {max_retained}",
                started.elapsed()
            );
            let spans = finalized
                .into_iter()
                .map(|span| span.span)
                .collect::<Vec<_>>();
            assert_eq!(spans, expected, "chunk {chunk}");
        }
        for span in &expected {
            eprintln!(
                "{:.3}-{:.3} {:.3}",
                span.start_seconds, span.end_seconds, span.score
            );
        }
    }
}
