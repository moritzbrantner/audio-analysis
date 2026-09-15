#![doc = include_str!("../README.md")]

use audio_analysis_core::FrameSpec;
use audio_analysis_pitch::{AutocorrelationPitchDetector, PitchDetectorConfig};
use audio_analysis_rhythm::track::{analyze_rhythm_track, TrackRhythmConfig};
use audio_contracts::{DetectError, Result};
use audio_generation_midi::karaoke::{
    build_karaoke_chart, KaraokeChart, KaraokeChartBuildOptions, KaraokeMelismaMode,
};
use audio_generation_midi::PitchTrackFrame;
use audio_karaoke_formats::{export_ultrastar_v1, UltraStarV1Metadata};
use media_core::TimedTextWordContract;

/// Authority used to resolve chart tempo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KaraokeTempoSource {
    /// Use `chart_options.tempo_bpm` exactly as supplied by the caller.
    Explicit,
    /// Analyze the full mix with `audio-analysis-rhythm` and use its selected tempo.
    AnalyzeFullMix,
}

/// Configuration for the in-memory song-to-karaoke orchestration slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KaraokePipelineOptions {
    /// How chart tempo is resolved.
    pub tempo_source: KaraokeTempoSource,
    /// Monophonic pitch detector configuration used on the vocal stem.
    pub pitch_detector: PitchDetectorConfig,
    /// Analysis window size for vocal pitch estimation.
    pub pitch_frame_size: usize,
    /// Hop size for vocal pitch estimation and emitted pitch-evidence intervals.
    pub pitch_hop_size: usize,
    /// Whole-track rhythm configuration used when tempo is analyzed.
    pub rhythm: TrackRhythmConfig,
    /// Neutral chart construction options. `tempo_bpm` is authoritative only for explicit tempo.
    pub chart_options: KaraokeChartBuildOptions,
}

impl Default for KaraokePipelineOptions {
    fn default() -> Self {
        let mut chart_options = KaraokeChartBuildOptions::default();
        chart_options.melisma_mode = KaraokeMelismaMode::SplitAcrossPitchNotes;
        Self {
            tempo_source: KaraokeTempoSource::AnalyzeFullMix,
            pitch_detector: PitchDetectorConfig::default(),
            pitch_frame_size: 2_048,
            pitch_hop_size: 512,
            rhythm: TrackRhythmConfig::default(),
            chart_options,
        }
    }
}

impl KaraokePipelineOptions {
    /// Validates all reusable analysis and chart settings.
    pub fn validate(self) -> Result<()> {
        self.pitch_detector.validate()?;
        FrameSpec::new(self.pitch_frame_size, self.pitch_hop_size)?;
        self.rhythm.validate()?;
        self.chart_options.validate()
    }
}

/// Stable evidence describing how the orchestration produced the chart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KaraokePipelineEvidence {
    /// Tempo authority used for the result.
    pub tempo_source: KaraokeTempoSource,
    /// Tempo actually supplied to neutral chart construction.
    pub tempo_bpm: f32,
    /// Rhythm confidence when tempo was analyzed; absent for explicit tempo.
    pub tempo_confidence: Option<f32>,
    /// Number of ranked rhythm tempo candidates retained by the analyzer.
    pub tempo_candidate_count: usize,
    /// Number of tracked beats retained by the whole-track rhythm analyzer.
    pub beat_count: usize,
    /// Number of inferred downbeats retained by the whole-track rhythm analyzer.
    pub downbeat_count: usize,
    /// Number of overlapping vocal analysis windows inspected for pitch.
    pub pitch_analysis_frame_count: usize,
    /// Number of hop-sized pitched evidence intervals passed to chart construction.
    pub pitched_frame_count: usize,
}

/// Result of the first complete in-memory song-to-UltraStar pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct KaraokePipelineResult {
    /// Neutral chart retained as the authoritative karaoke representation.
    pub chart: KaraokeChart,
    /// Downstream UltraStar v1 serialization of the neutral chart.
    pub ultrastar_text: String,
    /// Diagnostics produced by neutral lyric/pitch fusion.
    pub diagnostics: Vec<String>,
    /// Analysis provenance for tempo and pitch orchestration.
    pub evidence: KaraokePipelineEvidence,
}

/// Builds a neutral karaoke chart and UltraStar v1 document from decoded mono samples.
///
/// `full_mix_samples` are used only for whole-track rhythm analysis. They may be empty
/// when `KaraokeTempoSource::Explicit` is selected. `vocal_samples` are always required
/// and are analyzed only by the existing monophonic pitch detector. Lyrics must already
/// be canonically aligned; this function does not transcribe, realign, or syllabify text.
pub fn build_ultrastar_from_samples(
    full_mix_samples: &[f32],
    vocal_samples: &[f32],
    sample_rate: u32,
    lyrics: &[TimedTextWordContract],
    options: KaraokePipelineOptions,
    metadata: &UltraStarV1Metadata,
) -> Result<KaraokePipelineResult> {
    options.validate()?;
    validate_sample_rate(sample_rate)?;
    validate_samples(vocal_samples, "vocal")?;
    if options.tempo_source == KaraokeTempoSource::AnalyzeFullMix {
        validate_samples(full_mix_samples, "full-mix")?;
    } else {
        validate_finite_samples(full_mix_samples, "full-mix")?;
    }

    let tempo = resolve_tempo(full_mix_samples, sample_rate, options)?;
    let (pitch_frames, pitch_analysis_frame_count) = analyze_vocal_pitch(
        vocal_samples,
        sample_rate,
        options.pitch_detector,
        options.pitch_frame_size,
        options.pitch_hop_size,
    )?;

    let mut chart_options = options.chart_options;
    chart_options.tempo_bpm = tempo.bpm;
    let built = build_karaoke_chart(lyrics, &pitch_frames, chart_options)?;
    let ultrastar_text = export_ultrastar_v1(&built.chart, metadata)?;

    Ok(KaraokePipelineResult {
        chart: built.chart,
        ultrastar_text,
        diagnostics: built.diagnostics,
        evidence: KaraokePipelineEvidence {
            tempo_source: options.tempo_source,
            tempo_bpm: tempo.bpm,
            tempo_confidence: tempo.confidence,
            tempo_candidate_count: tempo.candidate_count,
            beat_count: tempo.beat_count,
            downbeat_count: tempo.downbeat_count,
            pitch_analysis_frame_count,
            pitched_frame_count: pitch_frames.len(),
        },
    })
}

#[derive(Debug, Clone, Copy)]
struct ResolvedTempo {
    bpm: f32,
    confidence: Option<f32>,
    candidate_count: usize,
    beat_count: usize,
    downbeat_count: usize,
}

fn resolve_tempo(
    full_mix_samples: &[f32],
    sample_rate: u32,
    options: KaraokePipelineOptions,
) -> Result<ResolvedTempo> {
    match options.tempo_source {
        KaraokeTempoSource::Explicit => Ok(ResolvedTempo {
            bpm: options.chart_options.tempo_bpm,
            confidence: None,
            candidate_count: 0,
            beat_count: 0,
            downbeat_count: 0,
        }),
        KaraokeTempoSource::AnalyzeFullMix => {
            let analysis = analyze_rhythm_track(full_mix_samples, sample_rate, options.rhythm)?;
            let bpm = analysis.bpm.ok_or_else(|| {
                invalid_argument("whole-track rhythm analysis did not select a karaoke tempo")
            })?;
            Ok(ResolvedTempo {
                bpm,
                confidence: Some(analysis.confidence),
                candidate_count: analysis.tempo_candidates.len(),
                beat_count: analysis.beats.len(),
                downbeat_count: analysis.downbeats.len(),
            })
        }
    }
}

fn analyze_vocal_pitch(
    samples: &[f32],
    sample_rate: u32,
    detector_config: PitchDetectorConfig,
    frame_size: usize,
    hop_size: usize,
) -> Result<(Vec<PitchTrackFrame>, usize)> {
    let frame_spec = FrameSpec::new(frame_size, hop_size)?;
    let detector = AutocorrelationPitchDetector::new(detector_config)?;
    let mut pitch_frames = Vec::new();
    let mut analysis_frame_count = 0_usize;

    for (start_sample, frame) in frame_spec.frames(samples) {
        analysis_frame_count += 1;
        let estimate = detector.estimate_samples(frame, sample_rate)?;
        let Some(frequency_hz) = estimate.frequency_hz else {
            continue;
        };

        // The detector sees the full overlapping analysis window. The neutral karaoke
        // builder, however, owns a single non-overlapping lead voice. Represent each
        // estimate by its hop-sized support interval so analysis context does not become
        // overlapping chart evidence.
        let end_sample = start_sample.saturating_add(hop_size).min(samples.len());
        if end_sample <= start_sample {
            continue;
        }
        pitch_frames.push(PitchTrackFrame {
            start_seconds: samples_to_seconds_f32(start_sample, sample_rate)?,
            end_seconds: samples_to_seconds_f32(end_sample, sample_rate)?,
            frequency_hz,
            confidence: estimate.confidence,
        });
    }

    Ok((pitch_frames, analysis_frame_count))
}

fn samples_to_seconds_f32(sample: usize, sample_rate: u32) -> Result<f32> {
    let seconds = sample as f64 / f64::from(sample_rate);
    if !seconds.is_finite() || seconds > f64::from(f32::MAX) {
        return Err(invalid_argument(
            "sample timestamp exceeds the representable karaoke time range",
        ));
    }
    Ok(seconds as f32)
}

fn validate_sample_rate(sample_rate: u32) -> Result<()> {
    if sample_rate == 0 {
        return Err(DetectError::InvalidAudioFormat {
            sample_rate,
            channels: 1,
        });
    }
    Ok(())
}

fn validate_samples(samples: &[f32], label: &str) -> Result<()> {
    if samples.is_empty() {
        return Err(invalid_argument(format!(
            "{label} samples must not be empty"
        )));
    }
    validate_finite_samples(samples, label)
}

fn validate_finite_samples(samples: &[f32], label: &str) -> Result<()> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid_argument(format!(
            "{label} samples must contain only finite values"
        )));
    }
    Ok(())
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lyric(text: &str, start_seconds: f64, end_seconds: f64) -> TimedTextWordContract {
        let timed = TimedTextWordContract::new(text)
            .with_time_range(Some(start_seconds), Some(end_seconds));
        let Ok(timed) = timed else {
            panic!("test lyric range must be valid");
        };
        timed
    }

    fn sine_wave(sample_rate: u32, frequency_hz: f32, seconds: f32) -> Vec<f32> {
        let len = (sample_rate as f32 * seconds) as usize;
        (0..len)
            .map(|sample| {
                let time = sample as f32 / sample_rate as f32;
                (std::f32::consts::TAU * frequency_hz * time).sin() * 0.8
            })
            .collect()
    }

    fn click_track(sample_rate: u32, bpm: f32, seconds: f32) -> Vec<f32> {
        let len = (sample_rate as f32 * seconds) as usize;
        let interval = (sample_rate as f32 * 60.0 / bpm).round() as usize;
        let mut samples = vec![0.0_f32; len];
        for start in (0..len).step_by(interval.max(1)) {
            for sample in samples.iter_mut().skip(start).take(16) {
                *sample = 1.0;
            }
        }
        samples
    }

    fn metadata() -> UltraStarV1Metadata {
        UltraStarV1Metadata::new("Pipeline", "Artist", "song.ogg")
    }

    fn test_options(tempo_source: KaraokeTempoSource) -> KaraokePipelineOptions {
        let mut options = KaraokePipelineOptions::default();
        options.tempo_source = tempo_source;
        options.pitch_frame_size = 512;
        options.pitch_hop_size = 128;
        options.pitch_detector = PitchDetectorConfig {
            min_frequency_hz: 80.0,
            max_frequency_hz: 800.0,
            confidence_threshold: 0.5,
        };
        options.rhythm = TrackRhythmConfig {
            min_bpm: 80.0,
            max_bpm: 160.0,
            fft_size: 256,
            hop_size: 64,
            tempo_candidate_count: 3,
            ..TrackRhythmConfig::default()
        };
        options.chart_options.tempo_bpm = 120.0;
        options.chart_options.min_overlap_ratio = 0.5;
        options.chart_options.min_pitch_note_overlap_ratio = 0.5;
        options
    }

    #[test]
    fn explicit_tempo_builds_chart_without_full_mix_analysis() {
        let sample_rate = 4_000;
        let vocals = sine_wave(sample_rate, 440.0, 2.0);
        let lyrics = vec![lyric("hello", 0.25, 1.5)];

        let result = build_ultrastar_from_samples(
            &[],
            &vocals,
            sample_rate,
            &lyrics,
            test_options(KaraokeTempoSource::Explicit),
            &metadata(),
        );
        let Ok(result) = result else {
            panic!("explicit-tempo pipeline should succeed");
        };

        assert_eq!(result.evidence.tempo_source, KaraokeTempoSource::Explicit);
        assert_eq!(result.evidence.tempo_bpm, 120.0);
        assert_eq!(result.evidence.tempo_confidence, None);
        assert!(result.evidence.pitch_analysis_frame_count > 0);
        assert!(result.evidence.pitched_frame_count > 0);
        assert!(result.ultrastar_text.contains("#BPM:120\n"));
        assert!(result.ultrastar_text.contains("hello"));
    }

    #[test]
    fn analyzed_tempo_uses_whole_track_rhythm_evidence() {
        let sample_rate = 4_000;
        let full_mix = click_track(sample_rate, 120.0, 8.0);
        let vocals = sine_wave(sample_rate, 440.0, 2.0);
        let lyrics = vec![lyric("hello", 0.25, 1.5)];

        let result = build_ultrastar_from_samples(
            &full_mix,
            &vocals,
            sample_rate,
            &lyrics,
            test_options(KaraokeTempoSource::AnalyzeFullMix),
            &metadata(),
        );
        let Ok(result) = result else {
            panic!("analyzed-tempo pipeline should succeed");
        };

        assert_eq!(
            result.evidence.tempo_source,
            KaraokeTempoSource::AnalyzeFullMix
        );
        assert!(result.evidence.tempo_confidence.is_some());
        assert!(result.evidence.tempo_candidate_count > 0);
        assert!(result.evidence.beat_count > 0);
        assert!((80.0..=160.0).contains(&result.evidence.tempo_bpm));
    }

    #[test]
    fn analyzed_tempo_requires_full_mix_samples() {
        let sample_rate = 4_000;
        let vocals = sine_wave(sample_rate, 440.0, 1.0);
        let lyrics = vec![lyric("word", 0.1, 0.8)];

        let error = build_ultrastar_from_samples(
            &[],
            &vocals,
            sample_rate,
            &lyrics,
            test_options(KaraokeTempoSource::AnalyzeFullMix),
            &metadata(),
        );
        let Err(error) = error else {
            panic!("missing full mix must fail");
        };
        assert!(error.to_string().contains("full-mix samples"));
    }

    #[test]
    fn rejects_non_finite_vocal_samples_before_analysis() {
        let lyrics = vec![lyric("word", 0.1, 0.8)];
        let error = build_ultrastar_from_samples(
            &[],
            &[0.0, f32::NAN, 0.0],
            4_000,
            &lyrics,
            test_options(KaraokeTempoSource::Explicit),
            &metadata(),
        );
        let Err(error) = error else {
            panic!("non-finite vocal samples must fail");
        };
        assert!(error.to_string().contains("vocal samples"));
    }
}
