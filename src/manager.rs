//! Host-process supervisor for on-demand asset downloads and isolated
//! `--ocr-worker` inference with automatic idle memory reclamation.

use crate::{
    downloader::{download_manifest, remove_assets},
    error::{ErrorKind, LumenError, Result},
    manifest::{
        MODEL_VERSION, Manifest, current_platform_supported, default_manifest, installed_bytes,
        is_installed, total_expected_bytes,
    },
    protocol::{WorkerRequest, WorkerResponse, read_async_frame, write_async_frame},
    types::{InstallState, OcrLine, Status},
};
use image::RgbaImage;
use std::{
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex as AsyncMutex,
};
use tokio_util::sync::CancellationToken;

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    last_used: Instant,
}

impl WorkerProcess {
    async fn shutdown(mut self) {
        if let Ok(bytes) = serde_json::to_vec(&WorkerRequest::Shutdown) {
            let _ = tokio::time::timeout(
                Duration::from_millis(400),
                write_async_frame(&mut self.stdin, &bytes, &[]),
            )
            .await;
        }
        drop(self.stdin);
        let _ = tokio::time::timeout(Duration::from_millis(800), self.child.wait()).await;
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }

    async fn kill_now(mut self) {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
    }
}

#[derive(Default)]
struct DownloadState {
    active: bool,
    cancel: Option<CancellationToken>,
    downloaded_bytes: u64,
    total_bytes: Option<u64>,
    last_error: Option<String>,
}

pub struct LumenManager {
    model_dir: PathBuf,
    worker_exe: Option<PathBuf>,
    worker: AsyncMutex<Option<WorkerProcess>>,
    download_op: AsyncMutex<()>,
    download: Mutex<DownloadState>,
    idle_unload_secs: AtomicU32,
    reaper_started: AtomicBool,
}

impl LumenManager {
    pub fn new(model_dir: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            model_dir,
            worker_exe: None,
            worker: AsyncMutex::new(None),
            download_op: AsyncMutex::new(()),
            download: Mutex::new(DownloadState::default()),
            idle_unload_secs: AtomicU32::new(180),
            reaper_started: AtomicBool::new(false),
        })
    }

    pub fn with_worker_exe(model_dir: PathBuf, worker_exe: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            model_dir,
            worker_exe: Some(worker_exe),
            worker: AsyncMutex::new(None),
            download_op: AsyncMutex::new(()),
            download: Mutex::new(DownloadState::default()),
            idle_unload_secs: AtomicU32::new(180),
            reaper_started: AtomicBool::new(false),
        })
    }

    fn ensure_idle_reaper(self: &Arc<Self>) {
        if self.reaper_started.swap(true, Ordering::AcqRel) {
            return;
        }
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(15));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                let Some(manager) = weak.upgrade() else {
                    break;
                };
                let limit = manager.idle_unload_secs.load(Ordering::Relaxed);
                if limit == 0 {
                    continue;
                }
                let mut guard = manager.worker.lock().await;
                let expired = guard
                    .as_ref()
                    .is_some_and(|w| w.last_used.elapsed() >= Duration::from_secs(limit as u64));
                if expired && let Some(worker) = guard.take() {
                    drop(guard);
                    worker.shutdown().await;
                }
            }
        });
    }

    pub fn set_idle_unload_secs(&self, secs: u32) {
        self.idle_unload_secs.store(secs, Ordering::Relaxed);
    }

    pub fn is_downloading(&self) -> bool {
        self.download
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active
    }

    pub fn status(&self) -> Status {
        let supported = current_platform_supported() || default_manifest().is_some();
        let manifest = default_manifest();
        let installed = manifest
            .as_ref()
            .is_some_and(|m| is_installed(&self.model_dir, m));
        let bytes = manifest
            .as_ref()
            .map_or(0, |m| installed_bytes(&self.model_dir, m));
        let expected = manifest.as_ref().map(total_expected_bytes);
        let worker_running = self
            .worker
            .try_lock()
            .map(|guard| guard.is_some())
            .unwrap_or(true);

        let dl = self.download.lock().unwrap_or_else(|e| e.into_inner());
        let state = if dl.active {
            InstallState::Downloading
        } else if installed {
            InstallState::Ready
        } else if dl.last_error.is_some() {
            InstallState::Error
        } else {
            InstallState::NotInstalled
        };

        Status {
            supported,
            state,
            model_version: manifest
                .map(|m| m.model_version)
                .unwrap_or_else(|| MODEL_VERSION.into()),
            installed_bytes: bytes,
            downloaded_bytes: if dl.active {
                dl.downloaded_bytes
            } else {
                bytes
            },
            total_bytes: if dl.active {
                dl.total_bytes.or(expected)
            } else {
                expected
            },
            worker_running,
            error: if dl.active || installed {
                None
            } else {
                dl.last_error.clone()
            },
        }
    }

    pub async fn download<F>(&self, mut on_status: F) -> Result<Status>
    where
        F: FnMut(Status) + Send,
    {
        let manifest = default_manifest().ok_or_else(|| {
            LumenError::new(
                ErrorKind::Unsupported,
                "当前系统架构暂不支持内置 PP-OCRv5 本地引擎",
            )
        })?;
        self.download_with_manifest(&manifest, &mut on_status).await
    }

    pub async fn download_with_manifest<F>(
        &self,
        manifest: &Manifest,
        on_status: &mut F,
    ) -> Result<Status>
    where
        F: FnMut(Status) + Send,
    {
        let _op_guard = self.download_op.try_lock().map_err(|_| {
            LumenError::new(ErrorKind::InvalidInput, "本地 OCR 模型正在下载中，请稍候")
        })?;
        let cancel = {
            let mut dl = self.download.lock().unwrap_or_else(|e| e.into_inner());
            if dl.active {
                return Err(LumenError::new(
                    ErrorKind::InvalidInput,
                    "本地 OCR 模型正在下载中，请稍候",
                ));
            }
            let token = CancellationToken::new();
            dl.active = true;
            dl.cancel = Some(token.clone());
            dl.downloaded_bytes = 0;
            dl.total_bytes = Some(total_expected_bytes(manifest));
            dl.last_error = None;
            token
        };

        on_status(self.status());

        let outcome = download_manifest(&self.model_dir, manifest, cancel, |progress| {
            {
                let mut dl = self.download.lock().unwrap_or_else(|e| e.into_inner());
                dl.downloaded_bytes = progress.downloaded_bytes;
                dl.total_bytes = progress.total_bytes;
            }
            on_status(self.status());
        })
        .await;

        {
            let mut dl = self.download.lock().unwrap_or_else(|e| e.into_inner());
            dl.active = false;
            dl.cancel = None;
            match &outcome {
                Ok(()) => dl.last_error = None,
                Err(err) if err.kind == ErrorKind::Cancelled => dl.last_error = None,
                Err(err) => dl.last_error = Some(err.message.clone()),
            }
        }

        let final_status = self.status();
        on_status(final_status.clone());
        outcome?;
        Ok(final_status)
    }

    pub fn cancel_download(&self) {
        let dl = self.download.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(token) = &dl.cancel {
            token.cancel();
        }
    }

    pub async fn remove(&self) -> Result<Status> {
        self.cancel_download();
        let _op_guard = self.download_op.lock().await;
        self.shutdown_worker().await;
        if let Some(manifest) = default_manifest() {
            remove_assets(&self.model_dir, &manifest)?;
        }
        {
            let mut dl = self.download.lock().unwrap_or_else(|e| e.into_inner());
            dl.last_error = None;
            dl.downloaded_bytes = 0;
        }
        Ok(self.status())
    }

    pub async fn shutdown_worker(&self) {
        let worker = {
            let mut guard = self.worker.lock().await;
            guard.take()
        };
        if let Some(worker) = worker {
            worker.shutdown().await;
        }
    }

    pub fn shutdown_blocking(&self) {
        if let Ok(mut guard) = self.worker.try_lock()
            && let Some(mut worker) = guard.take()
        {
            let _ = worker.child.start_kill();
        }
    }

    async fn spawn_worker(&self) -> Result<WorkerProcess> {
        let exe = match &self.worker_exe {
            Some(path) => path.clone(),
            None => std::env::current_exe()
                .map_err(|_| LumenError::new(ErrorKind::Internal, "无法定位本地 OCR 推理程序"))?,
        };
        let mut cmd = Command::new(exe);
        cmd.arg("--ocr-worker")
            .arg("--model-dir")
            .arg(&self.model_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        #[cfg(windows)]
        {
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd
            .spawn()
            .map_err(|_| LumenError::new(ErrorKind::OcrUnavailable, "无法启动本地 OCR 推理进程"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| LumenError::new(ErrorKind::Internal, "无法建立 OCR 推理管道"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LumenError::new(ErrorKind::Internal, "无法建立 OCR 推理管道"))?;
        Ok(WorkerProcess {
            child,
            stdin,
            stdout,
            last_used: Instant::now(),
        })
    }

    pub async fn recognize(
        self: &Arc<Self>,
        image: Arc<RgbaImage>,
        stop: CancellationToken,
    ) -> Result<Vec<OcrLine>> {
        if stop.is_cancelled() {
            return Err(LumenError::cancelled());
        }
        self.ensure_idle_reaper();
        let manifest = default_manifest().ok_or_else(|| {
            LumenError::new(
                ErrorKind::Unsupported,
                "当前系统架构暂不支持内置 PP-OCRv5 本地引擎",
            )
        })?;
        if !is_installed(&self.model_dir, &manifest) {
            return Err(LumenError::new(
                ErrorKind::OcrUnavailable,
                "尚未下载 PP-OCRv5 本地模型，请先下载后使用",
            ));
        }

        let mut guard = tokio::select! {
            _ = stop.cancelled() => return Err(LumenError::cancelled()),
            guard = self.worker.lock() => guard,
        };

        let need_spawn = match guard.as_mut() {
            None => true,
            Some(worker) => worker.child.try_wait().ok().flatten().is_some(),
        };
        if need_spawn {
            if let Some(stale) = guard.take() {
                stale.kill_now().await;
            }
            *guard = Some(self.spawn_worker().await?);
        }

        let (width, height) = image.dimensions();
        let raw = image.as_raw();
        let header = serde_json::to_vec(&WorkerRequest::Recognize {
            width,
            height,
            rgba_len: raw.len(),
        })
        .map_err(|_| LumenError::new(ErrorKind::Internal, "无法编码 OCR 请求"))?;

        let worker = guard.as_mut().unwrap();
        let exchange = async {
            write_async_frame(&mut worker.stdin, &header, raw).await?;
            let (resp_header, _) = read_async_frame(&mut worker.stdout).await?;
            serde_json::from_slice::<WorkerResponse>(&resp_header).map_err(|_| {
                LumenError::new(ErrorKind::InvalidResponse, "本地 OCR 推理输出格式无效")
            })
        };

        let result = tokio::select! {
            _ = stop.cancelled() => {
                if let Some(stuck) = guard.take() {
                    stuck.kill_now().await;
                }
                return Err(LumenError::cancelled());
            }
            res = exchange => res,
        };

        match result {
            Ok(WorkerResponse::Lines { lines }) => {
                worker.last_used = Instant::now();
                Ok(lines)
            }
            Ok(WorkerResponse::Error { message }) => {
                worker.last_used = Instant::now();
                Err(LumenError::new(ErrorKind::OcrUnavailable, message))
            }
            Ok(_) => {
                if let Some(broken) = guard.take() {
                    broken.kill_now().await;
                }
                Err(LumenError::new(
                    ErrorKind::InvalidResponse,
                    "本地 OCR 推理响应异常",
                ))
            }
            Err(err) => {
                if let Some(broken) = guard.take() {
                    broken.kill_now().await;
                }
                Err(err)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reports_not_installed_when_models_are_missing() {
        let dir = tempfile::tempdir().unwrap();
        let manager = LumenManager::new(dir.path().to_path_buf());
        let status = manager.status();
        assert_eq!(status.state, InstallState::NotInstalled);
        assert!(!status.worker_running);

        let err = manager
            .recognize(Arc::new(RgbaImage::new(32, 32)), CancellationToken::new())
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::OcrUnavailable);
    }
}
