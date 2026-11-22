use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Cancelled,
    InvalidInput,
    Configuration,
    OcrUnavailable,
    Network,
    InvalidResponse,
    Unsupported,
    Storage,
    Internal,
}

#[derive(Debug, Clone, Error)]
#[error("{message}")]
pub struct LumenError {
    pub kind: ErrorKind,
    pub message: String,
}

impl LumenError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn cancelled() -> Self {
        Self::new(ErrorKind::Cancelled, "OCR 操作已取消")
    }
}

pub type Result<T> = std::result::Result<T, LumenError>;
