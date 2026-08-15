pub fn effects_catalog_value() -> serde_json::Value {
    serde_json::json!({
        "streamingEffects": [
            {"type": "gain", "fields": ["linear"]},
            {"type": "distortion", "fields": ["mode", "driveDb", "mix", "outputGainDb"]},
            {"type": "delay", "fields": ["delaySeconds", "feedback", "wet", "dry"]},
            {"type": "echo", "fields": ["delaySeconds", "feedback", "wet", "dry"]},
            {"type": "reverb", "fields": ["roomSize", "damping", "wet", "dry", "width"]},
            {"type": "compressor", "fields": ["thresholdDb", "ratio", "attackMs", "releaseMs", "makeupGainDb", "kneeDb"]},
            {"type": "limiter", "fields": ["ceilingDb", "releaseMs"]},
            {"type": "eq", "fields": ["bands"]},
            {"type": "chorus", "fields": ["baseDelayMs", "depthMs", "rateHz", "feedback", "wet", "dry"]},
            {"type": "flanger", "fields": ["baseDelayMs", "depthMs", "rateHz", "feedback", "wet", "dry"]},
            {"type": "tremolo", "fields": ["rateHz", "depth"]},
            {"type": "pan", "fields": ["position"]},
            {"type": "stereoWidth", "fields": ["width"]}
        ],
        "offlineEdits": ["trim", "reverse", "fade", "normalize", "insertSilence", "delete", "resample", "speed", "pitchShift"],
        "presets": ["VocalClean", "PodcastVoice", "LoFi", "WideChorus", "SmallRoomReverb", "HardLimiter"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effects_catalog_stays_local_and_deterministic() {
        let catalog = effects_catalog_value();
        assert!(catalog["streamingEffects"].as_array().unwrap().len() >= 10);
        assert!(catalog["offlineEdits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|edit| edit == "reverse"));
    }
}
