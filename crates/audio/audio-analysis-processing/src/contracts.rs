use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AudioSamplesRequest {
    pub samples: Vec<f32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub preview_samples: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AudioProcessingChainRequest {
    pub samples: Vec<f32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub chain: Vec<serde_json::Value>,
    pub preview_samples: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AudioMixdownRequest {
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub placements: Vec<AudioPlacementRequest>,
    pub preview_samples: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AudioPlacementRequest {
    pub samples: Vec<f32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub start_seconds: Option<f64>,
    pub gain: Option<f32>,
}
