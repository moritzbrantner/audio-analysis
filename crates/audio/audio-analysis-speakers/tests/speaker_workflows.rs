use audio_analysis_recognition::SpectralEmbeddingConfig;
use audio_analysis_speakers::{
    DiarizedSpeaker, EnergyVadConfig, EnergyVoiceActivityDetector, SpeakerAudio, SpeakerDiarizer,
    SpeakerEmbedding, SpeakerEmbeddingExtractor, SpeakerEmbeddingModel,
    SpeakerEmbeddingModelFamily, SpeakerId, SpeakerIdentificationOptions, SpeakerLabel,
    SpeakerLibrary, SpectralSpeakerEmbedder, SpeechSpan, VoiceActivityDetector,
    WindowedSpeakerDiarizer,
};
use audio_contracts::Result;

#[derive(Debug, Clone)]
struct SignEmbedder {
    model: SpeakerEmbeddingModel,
}

impl SignEmbedder {
    fn new() -> Self {
        Self {
            model: SpeakerEmbeddingModel::new(
                SpeakerEmbeddingModelFamily::SpeechBrain,
                "test-sign-speaker",
                "1",
                2,
            )
            .unwrap(),
        }
    }
}

impl SpeakerEmbeddingExtractor for SignEmbedder {
    fn model_info(&self) -> SpeakerEmbeddingModel {
        self.model.clone()
    }

    fn embed_speaker(&mut self, audio: &SpeakerAudio<'_>) -> Result<SpeakerEmbedding> {
        let mean = audio.samples().iter().sum::<f32>() / audio.samples().len() as f32;
        let values = if mean >= 0.0 {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        };
        SpeakerEmbedding::new(values, self.model_info(), audio.sample_rate())
    }
}

#[derive(Debug, Clone)]
struct FixedVad {
    spans: Vec<SpeechSpan>,
}

impl VoiceActivityDetector for FixedVad {
    fn detect_speech(&mut self, _audio: &SpeakerAudio<'_>) -> Result<Vec<SpeechSpan>> {
        Ok(self.spans.clone())
    }
}

fn id(value: &str) -> SpeakerId {
    SpeakerId::new(value).unwrap()
}

fn label(value: &str) -> SpeakerLabel {
    SpeakerLabel::new(value).unwrap()
}

fn sine(freq_hz: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
    let samples = (sample_rate as f32 * seconds) as usize;
    (0..samples)
        .map(|index| {
            let t = index as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5
        })
        .collect()
}

#[test]
fn public_spectral_enrollment_identification_and_snapshot_workflow() {
    let sample_rate = 8_000;
    let samples = sine(220.0, sample_rate, 0.5);
    let audio = SpeakerAudio::mono(&samples, sample_rate).unwrap();
    let mut embedder =
        SpectralSpeakerEmbedder::new(SpectralEmbeddingConfig::new(512, 256, 8).unwrap()).unwrap();
    let mut library = SpeakerLibrary::new();
    library
        .enroll(id("speaker-a"), label("Speaker A"), &audio, &mut embedder)
        .unwrap();

    let query = embedder.embed_speaker(&audio).unwrap();
    let mut options = SpeakerIdentificationOptions::new(0.8).unwrap();
    options.min_margin = None;
    let result = library.identify(&query, &options).unwrap();
    assert_eq!(result.best_match.unwrap().speaker_id.as_str(), "speaker-a");

    let restored = SpeakerLibrary::from_json_str(&library.to_json_string().unwrap()).unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored.embedding_model().unwrap(), &result.embedding_model);
}

#[test]
fn public_vad_finds_speech_and_rejects_quiet_audio() {
    let mut samples = vec![0.0_f32; 100];
    samples.extend(vec![0.25; 200]);
    samples.extend(vec![0.0; 100]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();
    let mut vad = EnergyVoiceActivityDetector::new(EnergyVadConfig {
        rms_threshold: 0.05,
        frame_seconds: 0.04,
        hop_seconds: 0.02,
        min_speech_seconds: 0.05,
        merge_gap_seconds: 0.02,
    })
    .unwrap();
    assert_eq!(vad.detect_speech(&audio).unwrap().len(), 1);

    let quiet = vec![0.0_f32; 400];
    let quiet = SpeakerAudio::mono(&quiet, 1_000).unwrap();
    assert!(vad.detect_speech(&quiet).unwrap().is_empty());
}

#[test]
fn public_diarizer_maps_known_speaker_and_labels_unknown_cluster() {
    let mut samples = vec![0.4_f32; 100];
    samples.extend(vec![-0.4_f32; 100]);
    let audio = SpeakerAudio::mono(&samples, 1_000).unwrap();

    let mut enroll_embedder = SignEmbedder::new();
    let known_audio = SpeakerAudio::mono(&samples[..100], 1_000).unwrap();
    let mut library = SpeakerLibrary::new();
    library
        .enroll(
            id("known"),
            label("Known"),
            &known_audio,
            &mut enroll_embedder,
        )
        .unwrap();

    let mut options = SpeakerIdentificationOptions::new(0.8).unwrap();
    options.min_margin = None;
    let mut diarizer = WindowedSpeakerDiarizer::new(
        SignEmbedder::new(),
        FixedVad {
            spans: vec![
                SpeechSpan::new(0.0, 0.1, 0.9).unwrap(),
                SpeechSpan::new(0.1, 0.2, 0.9).unwrap(),
            ],
        },
    )
    .library(library);
    diarizer.identification_options = options;

    let result = diarizer.diarize(&audio).unwrap();

    assert_eq!(
        result.segments[0].speaker,
        DiarizedSpeaker::Known(id("known"))
    );
    assert_eq!(
        result.segments[1].speaker,
        DiarizedSpeaker::Unknown("speaker_0".to_string())
    );
}
