use lumen_ocr::{LumenManager, parse_worker_args, worker_main};
use std::{path::PathBuf, sync::Arc};
use tokio_util::sync::CancellationToken;

fn default_model_dir() -> PathBuf {
    if let Some(custom) = std::env::var_os("LUMEN_OCR_DIR") {
        return PathBuf::from(custom);
    }
    if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(xdg).join("lumen-ocr");
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        return PathBuf::from(home).join(".lumen-ocr");
    }
    PathBuf::from(".lumen-ocr")
}

fn print_usage() {
    eprintln!(
        "LumenOCR - Pure-Rust PP-OCRv5 CLI\n\n\
         Usage:\n\
           lumen-ocr download [--model-dir <dir>]\n\
           lumen-ocr recognize <image> [--model-dir <dir>] [--json]\n\
           lumen-ocr status [--model-dir <dir>]\n\
           lumen-ocr remove [--model-dir <dir>]"
    );
}

#[tokio::main]
async fn main() {
    if let Some(dir) = parse_worker_args() {
        std::process::exit(worker_main(&dir));
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help") {
        print_usage();
        return;
    }

    let mut model_dir = default_model_dir();
    let mut json = false;
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" if i + 1 < args.len() => {
                model_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            other => {
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    let Some(cmd) = positional.first().map(String::as_str) else {
        print_usage();
        return;
    };
    let manager = LumenManager::new(model_dir);

    match cmd {
        "download" => {
            let result = manager
                .download(|status| {
                    if let Some(total) = status.total_bytes
                        && total > 0
                    {
                        let pct = (status.downloaded_bytes * 100 / total).min(100);
                        eprint!(
                            "\rDownloading... {pct}% ({:.1} / {:.1} MB)",
                            status.downloaded_bytes as f64 / 1_048_576.0,
                            total as f64 / 1_048_576.0
                        );
                    }
                })
                .await;
            eprintln!();
            match result {
                Ok(status) => println!(
                    "Ready ({}, {:.1} MB)",
                    status.model_version,
                    status.installed_bytes as f64 / 1_048_576.0
                ),
                Err(err) => {
                    eprintln!("Error: {err}");
                    std::process::exit(1);
                }
            }
        }
        "status" => {
            let status = manager.status();
            println!("{}", serde_json::to_string_pretty(&status).unwrap());
        }
        "remove" => match manager.remove().await {
            Ok(_) => println!("Removed local OCR assets."),
            Err(err) => {
                eprintln!("Error: {err}");
                std::process::exit(1);
            }
        },
        "recognize" => {
            let Some(image_path) = positional.get(1) else {
                eprintln!("Missing image path.");
                print_usage();
                std::process::exit(2);
            };
            let img = match image::open(image_path) {
                Ok(i) => Arc::new(i.to_rgba8()),
                Err(err) => {
                    eprintln!("Failed to open {image_path}: {err}");
                    std::process::exit(1);
                }
            };
            match manager.recognize(img, CancellationToken::new()).await {
                Ok(lines) => {
                    if json {
                        println!("{}", serde_json::to_string_pretty(&lines).unwrap());
                    } else {
                        for line in lines {
                            println!("{}", line.text);
                        }
                    }
                    manager.shutdown_worker().await;
                }
                Err(err) => {
                    eprintln!("Error: {err}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            print_usage();
            std::process::exit(2);
        }
    }
}
