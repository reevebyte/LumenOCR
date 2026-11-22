use crate::{
    error::{ErrorKind, LumenError, Result},
    manifest::{AssetKind, AssetSpec, Manifest},
};
use flate2::read::GzDecoder;
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tar::Archive;
use tokio_util::sync::CancellationToken;
use zip::ZipArchive;

const MAX_EXTRACTED_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

pub fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn valid_sha256_hex(expected: &str) -> bool {
    expected.len() == 64 && expected.chars().all(|c| c.is_ascii_hexdigit())
}

fn verify_existing(path: &Path, expected_sha256: &str) -> bool {
    if !path.is_file() || !valid_sha256_hex(expected_sha256) {
        return false;
    }
    sha256_file(path)
        .map(|actual| actual.eq_ignore_ascii_case(expected_sha256))
        .unwrap_or(false)
}

fn parse_proxy_url(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let value = if raw.contains('=') {
        raw.split(';')
            .map(str::trim)
            .find_map(|s| s.strip_prefix("https="))
            .or_else(|| {
                raw.split(';')
                    .map(str::trim)
                    .find_map(|s| s.strip_prefix("http="))
            })?
            .trim()
    } else {
        raw.split(';').next()?.trim()
    };
    if value.is_empty() {
        return None;
    }
    let normalized = if value.contains("://") {
        value.to_string()
    } else {
        format!("http://{value}")
    };
    let parsed = url::Url::parse(&normalized).ok()?;
    matches!(parsed.scheme(), "http" | "https").then(|| parsed.to_string())
}

fn system_proxy_url() -> Option<String> {
    [
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    .into_iter()
    .find_map(|key| std::env::var(key).ok().and_then(|v| parse_proxy_url(&v)))
}

fn build_client(proxy: Option<&str>) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(600))
        .user_agent("LumenOCR-Downloader/0.1");
    builder = match proxy {
        Some(url) => builder.proxy(
            reqwest::Proxy::all(url)
                .map_err(|_| LumenError::new(ErrorKind::Network, "代理地址配置无效"))?,
        ),
        None => builder.no_proxy(),
    };
    builder
        .build()
        .map_err(|_| LumenError::new(ErrorKind::Internal, "无法初始化模型下载器"))
}

fn is_direct_download_url(url: &str, asset: &AssetSpec) -> bool {
    match &asset.kind {
        AssetKind::DirectFile => true,
        AssetKind::TarGzMember { .. } => !url.ends_with(".tgz") && !url.ends_with(".tar.gz"),
        AssetKind::ZipMember { .. } => !url.ends_with(".zip") && !url.ends_with(".nupkg"),
    }
}

fn extract_tar_gz_member(archive_path: &Path, member_suffix: &str, dest_path: &Path) -> Result<()> {
    let file = File::open(archive_path)
        .map_err(|_| LumenError::new(ErrorKind::Storage, "无法读取下载的运行时压缩包"))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    let entries = archive
        .entries()
        .map_err(|_| LumenError::new(ErrorKind::InvalidResponse, "运行时压缩包格式无效"))?;

    for entry in entries {
        let entry = entry
            .map_err(|_| LumenError::new(ErrorKind::InvalidResponse, "无法解析运行时压缩包"))?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry
            .path()
            .map_err(|_| LumenError::new(ErrorKind::InvalidResponse, "运行时压缩包路径无效"))?;
        let normalized = path.to_string_lossy().replace('\\', "/");
        if normalized.contains("..") {
            continue;
        }
        if normalized.ends_with(member_suffix) {
            let mut limited = entry.take(MAX_EXTRACTED_BYTES + 1);
            let mut out = File::create(dest_path)
                .map_err(|_| LumenError::new(ErrorKind::Storage, "无法写入运行时文件"))?;
            let copied = io::copy(&mut limited, &mut out)
                .map_err(|_| LumenError::new(ErrorKind::Storage, "解压运行时文件失败"))?;
            if copied > MAX_EXTRACTED_BYTES {
                return Err(LumenError::new(
                    ErrorKind::InvalidResponse,
                    "运行时文件超出允许大小",
                ));
            }
            out.sync_all()
                .map_err(|_| LumenError::new(ErrorKind::Storage, "保存运行时文件失败"))?;
            return Ok(());
        }
    }

    Err(LumenError::new(
        ErrorKind::InvalidResponse,
        "运行时压缩包中缺少所需的动态库文件",
    ))
}

fn extract_zip_member(archive_path: &Path, member_suffix: &str, dest_path: &Path) -> Result<()> {
    let file = File::open(archive_path)
        .map_err(|_| LumenError::new(ErrorKind::Storage, "无法读取下载的运行时压缩包"))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|_| LumenError::new(ErrorKind::InvalidResponse, "运行时压缩包格式无效"))?;

    for index in 0..archive.len() {
        let entry = archive
            .by_index(index)
            .map_err(|_| LumenError::new(ErrorKind::InvalidResponse, "无法解析运行时压缩包"))?;
        if !entry.is_file() {
            continue;
        }
        let normalized = entry.name().replace('\\', "/");
        if normalized.contains("..") {
            continue;
        }
        if normalized.ends_with(member_suffix) {
            let mut limited = entry.take(MAX_EXTRACTED_BYTES + 1);
            let mut out = File::create(dest_path)
                .map_err(|_| LumenError::new(ErrorKind::Storage, "无法写入运行时文件"))?;
            let copied = io::copy(&mut limited, &mut out)
                .map_err(|_| LumenError::new(ErrorKind::Storage, "解压运行时文件失败"))?;
            if copied > MAX_EXTRACTED_BYTES {
                return Err(LumenError::new(
                    ErrorKind::InvalidResponse,
                    "运行时文件超出允许大小",
                ));
            }
            out.sync_all()
                .map_err(|_| LumenError::new(ErrorKind::Storage, "保存运行时文件失败"))?;
            return Ok(());
        }
    }

    Err(LumenError::new(
        ErrorKind::InvalidResponse,
        "运行时压缩包中缺少所需的动态库文件",
    ))
}

async fn download_stream_to_file<F>(
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    stop: &CancellationToken,
    mut on_chunk: F,
) -> Result<u64>
where
    F: FnMut(u64),
{
    let response = tokio::select! {
        _ = stop.cancelled() => return Err(LumenError::cancelled()),
        res = client.get(url).send() => res.map_err(|_| {
            LumenError::new(ErrorKind::Network, "本地 OCR 资源下载失败，请检查网络或代理")
        })?,
    };
    if !response.status().is_success() {
        return Err(LumenError::new(
            ErrorKind::Network,
            format!(
                "本地 OCR 资源下载失败（HTTP {}）",
                response.status().as_u16()
            ),
        ));
    }

    let mut file = File::create(dest)
        .map_err(|_| LumenError::new(ErrorKind::Storage, "无法创建本地 OCR 临时文件"))?;
    let mut stream = response.bytes_stream();
    let mut written = 0u64;

    while let Some(item) = tokio::select! {
        _ = stop.cancelled() => return Err(LumenError::cancelled()),
        next = stream.next() => next,
    } {
        let chunk =
            item.map_err(|_| LumenError::new(ErrorKind::Network, "下载连接中断，请重试"))?;
        file.write_all(&chunk)
            .map_err(|_| LumenError::new(ErrorKind::Storage, "写入本地 OCR 文件失败"))?;
        written = written.saturating_add(chunk.len() as u64);
        on_chunk(chunk.len() as u64);
    }
    file.sync_all()
        .map_err(|_| LumenError::new(ErrorKind::Storage, "保存本地 OCR 文件失败"))?;
    Ok(written)
}

pub fn cleanup_part_files(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with(".part-") {
            let _ = fs::remove_file(entry.path());
        }
    }
}

pub fn remove_assets(dir: &Path, manifest: &Manifest) -> Result<()> {
    cleanup_part_files(dir);
    for asset in &manifest.assets {
        let path = dir.join(&asset.file_name);
        if path.exists() {
            fs::remove_file(&path).map_err(|_| {
                LumenError::new(
                    ErrorKind::Storage,
                    "无法删除本地 OCR 文件，请关闭占用后重试",
                )
            })?;
        }
    }
    let _ = fs::remove_dir(dir);
    Ok(())
}

pub async fn download_manifest<F>(
    dir: &Path,
    manifest: &Manifest,
    stop: CancellationToken,
    mut on_progress: F,
) -> Result<()>
where
    F: FnMut(DownloadProgress) + Send,
{
    for asset in &manifest.assets {
        if !valid_sha256_hex(&asset.sha256) {
            return Err(LumenError::new(
                ErrorKind::Configuration,
                format!(
                    "本地 OCR 资产 {} 缺少有效的 SHA-256 校验值",
                    asset.file_name
                ),
            ));
        }
    }

    fs::create_dir_all(dir)
        .map_err(|_| LumenError::new(ErrorKind::Storage, "无法创建本地 OCR 模型目录"))?;
    cleanup_part_files(dir);

    let total_bytes: u64 = manifest.assets.iter().map(|a| a.expected_bytes).sum();
    let mut completed_bytes = 0u64;
    on_progress(DownloadProgress {
        downloaded_bytes: 0,
        total_bytes: Some(total_bytes),
    });

    let proxy = system_proxy_url();
    let mut proxy_candidates = Vec::new();
    if let Some(p) = proxy {
        proxy_candidates.push(Some(p));
    }
    proxy_candidates.push(None);

    let result = async {
        for asset in &manifest.assets {
            if stop.is_cancelled() {
                return Err(LumenError::cancelled());
            }
            let target_path = dir.join(&asset.file_name);
            if verify_existing(&target_path, &asset.sha256) {
                completed_bytes = completed_bytes.saturating_add(asset.expected_bytes);
                on_progress(DownloadProgress {
                    downloaded_bytes: completed_bytes.min(total_bytes),
                    total_bytes: Some(total_bytes),
                });
                continue;
            }

            let part_download = dir.join(format!(".part-dl-{}", asset.file_name));
            let part_final = dir.join(format!(".part-out-{}", asset.file_name));
            let mut last_error: Option<LumenError> = None;
            let mut installed = false;

            'urls: for url in &asset.urls {
                for proxy_opt in &proxy_candidates {
                    if stop.is_cancelled() {
                        return Err(LumenError::cancelled());
                    }
                    let _ = fs::remove_file(&part_download);
                    let _ = fs::remove_file(&part_final);

                    let client = match build_client(proxy_opt.as_deref()) {
                        Ok(c) => c,
                        Err(err) => {
                            last_error = Some(err);
                            continue;
                        }
                    };

                    let mut asset_downloaded = 0u64;
                    let mut last_emit = Instant::now();
                    let dl =
                        download_stream_to_file(&client, url, &part_download, &stop, |delta| {
                            asset_downloaded = asset_downloaded.saturating_add(delta);
                            if last_emit.elapsed() >= Duration::from_millis(80) {
                                last_emit = Instant::now();
                                let capped = asset_downloaded.min(asset.expected_bytes);
                                on_progress(DownloadProgress {
                                    downloaded_bytes: completed_bytes
                                        .saturating_add(capped)
                                        .min(total_bytes),
                                    total_bytes: Some(total_bytes),
                                });
                            }
                        })
                        .await;

                    if let Err(err) = dl {
                        if err.kind == ErrorKind::Cancelled {
                            return Err(err);
                        }
                        last_error = Some(err);
                        continue;
                    }

                    let candidate_file: PathBuf = if is_direct_download_url(url, asset) {
                        part_download.clone()
                    } else {
                        let part_dl_clone = part_download.clone();
                        let part_out_clone = part_final.clone();
                        let kind = asset.kind.clone();
                        let extracted = tokio::task::spawn_blocking(move || match kind {
                            AssetKind::DirectFile => Ok(()),
                            AssetKind::TarGzMember { member_suffix } => extract_tar_gz_member(
                                &part_dl_clone,
                                &member_suffix,
                                &part_out_clone,
                            ),
                            AssetKind::ZipMember { member_suffix } => {
                                extract_zip_member(&part_dl_clone, &member_suffix, &part_out_clone)
                            }
                        })
                        .await
                        .map_err(|_| LumenError::new(ErrorKind::Internal, "解压运行时任务异常"))?;
                        if let Err(err) = extracted {
                            last_error = Some(err);
                            continue;
                        }
                        part_final.clone()
                    };

                    let check_path = candidate_file.clone();
                    let actual_sha = tokio::task::spawn_blocking(move || sha256_file(&check_path))
                        .await
                        .map_err(|_| LumenError::new(ErrorKind::Internal, "文件校验任务异常"))?
                        .map_err(|_| LumenError::new(ErrorKind::Storage, "无法读取已下载的文件"))?;

                    if !actual_sha.eq_ignore_ascii_case(&asset.sha256) {
                        last_error = Some(LumenError::new(
                            ErrorKind::InvalidResponse,
                            format!("文件 {} 完整性校验失败（SHA-256 不匹配）", asset.file_name),
                        ));
                        continue;
                    }

                    #[cfg(windows)]
                    if target_path.exists() {
                        let _ = fs::remove_file(&target_path);
                    }
                    fs::rename(&candidate_file, &target_path).map_err(|_| {
                        LumenError::new(ErrorKind::Storage, "无法保存本地 OCR 文件")
                    })?;
                    let _ = fs::remove_file(&part_download);
                    let _ = fs::remove_file(&part_final);
                    installed = true;
                    break 'urls;
                }
            }

            if !installed {
                return Err(last_error.unwrap_or_else(|| {
                    LumenError::new(
                        ErrorKind::Network,
                        format!("下载 {} 失败，请检查网络或代理后重试", asset.file_name),
                    )
                }));
            }

            completed_bytes = completed_bytes.saturating_add(asset.expected_bytes);
            on_progress(DownloadProgress {
                downloaded_bytes: completed_bytes.min(total_bytes),
                total_bytes: Some(total_bytes),
            });
        }
        Ok(())
    }
    .await;

    cleanup_part_files(dir);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    fn sha256_bytes(bytes: &[u8]) -> String {
        hex_lower(&Sha256::digest(bytes))
    }

    #[tokio::test]
    async fn downloads_verifies_sha256_and_extracts_tgz_member() {
        let server = MockServer::start().await;
        let model_bytes = b"fake-onnx-model-payload";
        let lib_bytes = b"fake-onnxruntime-shared-library";

        let mut tar_buf = Vec::new();
        {
            let enc = GzEncoder::new(&mut tar_buf, Compression::default());
            let mut builder = tar::Builder::new(enc);
            let mut header = tar::Header::new_gnu();
            header.set_size(lib_bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(
                    &mut header,
                    "onnxruntime-linux-x64-1.22.1/lib/libonnxruntime.so.1.22.1",
                    &lib_bytes[..],
                )
                .unwrap();
            let enc = builder.into_inner().unwrap();
            enc.finish().unwrap();
        }

        Mock::given(method("GET"))
            .and(path("/model.onnx"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(model_bytes.to_vec()))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/ort.tgz"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(tar_buf))
            .mount(&server)
            .await;

        let manifest = Manifest {
            model_version: "test-v1".into(),
            assets: vec![
                AssetSpec {
                    file_name: "model.onnx".into(),
                    urls: vec![format!("{}/model.onnx", server.uri())],
                    sha256: sha256_bytes(model_bytes),
                    expected_bytes: model_bytes.len() as u64,
                    kind: AssetKind::DirectFile,
                    is_runtime: false,
                },
                AssetSpec {
                    file_name: "libonnxruntime.so".into(),
                    urls: vec![format!("{}/ort.tgz", server.uri())],
                    sha256: sha256_bytes(lib_bytes),
                    expected_bytes: lib_bytes.len() as u64,
                    kind: AssetKind::TarGzMember {
                        member_suffix: "/lib/libonnxruntime.so.1.22.1".into(),
                    },
                    is_runtime: true,
                },
            ],
        };

        let dir = tempfile::tempdir().unwrap();
        download_manifest(dir.path(), &manifest, CancellationToken::new(), |_| {})
            .await
            .unwrap();

        assert_eq!(
            fs::read(dir.path().join("model.onnx")).unwrap(),
            model_bytes
        );
        assert_eq!(
            fs::read(dir.path().join("libonnxruntime.so")).unwrap(),
            lib_bytes
        );

        remove_assets(dir.path(), &manifest).unwrap();
        assert!(!dir.path().join("model.onnx").exists());
    }

    #[tokio::test]
    async fn rejects_tampered_sha256_and_leaves_no_part_files() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/tampered.onnx"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"corrupted-bytes".to_vec()))
            .mount(&server)
            .await;

        let manifest = Manifest {
            model_version: "test-v1".into(),
            assets: vec![AssetSpec {
                file_name: "model.onnx".into(),
                urls: vec![format!("{}/tampered.onnx", server.uri())],
                sha256: "a".repeat(64),
                expected_bytes: 15,
                kind: AssetKind::DirectFile,
                is_runtime: false,
            }],
        };

        let dir = tempfile::tempdir().unwrap();
        let err = download_manifest(dir.path(), &manifest, CancellationToken::new(), |_| {})
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidResponse);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
