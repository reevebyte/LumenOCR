use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OcrLine {
    pub text: String,
    pub bounds: Rect,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InstallState {
    NotInstalled,
    Downloading,
    Ready,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub supported: bool,
    pub state: InstallState,
    pub model_version: String,
    pub installed_bytes: u64,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub worker_running: bool,
    pub error: Option<String>,
}
