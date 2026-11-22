pub mod downloader;
pub mod engine;
pub mod error;
pub mod manager;
pub mod manifest;
pub mod postprocess;
pub mod protocol;
pub mod types;

pub use engine::{LumenEngine, worker_main};
pub use error::{ErrorKind, LumenError, Result};
pub use manager::LumenManager;
pub use manifest::Manifest;
pub use types::{InstallState, OcrLine, Rect, Status};

use std::path::PathBuf;

/// Parse `--ocr-worker --model-dir <path>` from process arguments.
/// Call this at the very start of `main()` before initializing any GUI runtime.
pub fn parse_worker_args() -> Option<PathBuf> {
    let mut args = std::env::args_os().skip(1);
    let mut worker = false;
    let mut model_dir = None;
    while let Some(arg) = args.next() {
        if arg == "--ocr-worker" {
            worker = true;
        } else if arg == "--model-dir" {
            model_dir = args.next().map(PathBuf::from);
        }
    }
    if worker { model_dir } else { None }
}
