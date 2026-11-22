# LumenOCR

**Pure-Rust, zero-OpenCV PP-OCRv5 engine for lightweight desktop apps and CLIs.**

Built for [ScreenTranslate](https://github.com/reevebyte/ScreenTranslate) and reusable by any Rust or Tauri application that needs accurate offline multilingual OCR (Simplified/Traditional Chinese, English, Japanese) without bloating its installer or leaking resident inference memory.

## Highlights

- **Zero C++ build dependencies**: Pure-Rust DBNet postprocessing (connected components, convex hull, rotating-calipers minimum area rectangle, polygon unclip, bilinear perspective crop) and SVTR greedy CTC decoding. No OpenCV required.
- **Tiny binary footprint**: Uses `ort` (`load-dynamic`, `api-20`) so ONNX Runtime and PP-OCRv5 models are never bundled into your executable.
- **On-demand verified downloads**: Downloads models (ModelScope primary + GitHub Release fallback) and prebuilt single-file ONNX Runtime binaries (`assets-v1` primary + official Microsoft NuGet fallback) with strict SHA-256 verification and atomic installation.
- **Process-isolated memory reclamation**: Runs inference inside a same-binary `--ocr-worker` child process over stdin/stdout length-prefixed frames. Reuses warm sessions across consecutive requests and exits cleanly when idle so the OS reclaims 100% of memory.
- **Cross-platform**: Supports Windows x64, Linux x64, macOS Apple Silicon (`aarch64`), and macOS Intel (`x86_64`).

## CLI Usage

```sh
# Download PP-OCRv5 Mobile models + ONNX Runtime for this platform
cargo run -- download

# Recognize text in an image (human-readable or JSON with pixel bounding boxes)
cargo run -- recognize path/to/screenshot.png
cargo run -- recognize path/to/screenshot.png --json

# Inspect installation status or remove downloaded assets
cargo run -- status
cargo run -- remove
```

## Library Usage

```rust
use lumen_ocr::{LumenManager, worker_main};
use std::{path::PathBuf, sync::Arc};
use tokio_util::sync::CancellationToken;

// 1. Intercept `--ocr-worker` at the very start of `main()` before GUI init:
if let Some(dir) = lumen_ocr::parse_worker_args() {
    std::process::exit(worker_main(&dir));
}

// 2. Manage on-demand download & isolated worker inference from your app:
let manager = LumenManager::new(PathBuf::from("./models"));
manager.download(|status| {
    println!("state={:?} {}/{}", status.state, status.downloaded_bytes, status.total_bytes.unwrap_or(0));
}).await?;

let image = Arc::new(image::open("screenshot.png")?.to_rgba8());
let lines = manager.recognize(image, CancellationToken::new()).await?;
for line in lines {
    println!("{} {:?}", line.text, line.bounds);
}

// Worker shuts down automatically after `idle_unload_secs` (default 180s),
// or call `manager.shutdown_worker().await` immediately.
```

## License

MIT License. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for upstream model and runtime licenses.
