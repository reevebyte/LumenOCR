//! ONNX Runtime PP-OCRv5 inference engine and `--ocr-worker` subprocess loop.

use crate::{
    manifest::{CLS_MODEL_FILE, DET_MODEL_FILE, DICT_FILE, REC_MODEL_FILE, resolve_runtime_dylib},
    postprocess::{
        crop_quad, ctc_greedy_decode, dbnet_postprocess, det_resize_dims, parse_dictionary,
        rgba_to_rgb,
    },
    protocol::{WorkerRequest, WorkerResponse, read_sync_frame, write_sync_frame},
    types::OcrLine,
};
use image::{RgbImage, RgbaImage, imageops::FilterType};
use ndarray::Array4;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::{
    fs,
    io::{BufReader, BufWriter},
    path::Path,
};

const DET_LIMIT_SIDE_LEN: u32 = 960;
const DET_THRESH: f32 = 0.3;
const DET_BOX_THRESH: f32 = 0.6;
const DET_UNCLIP_RATIO: f32 = 1.5;
const CLS_THRESH: f32 = 0.9;
const REC_IMG_HEIGHT: u32 = 48;
const REC_MAX_WIDTH: u32 = 2048;
const REC_MIN_SCORE: f32 = 0.5;

const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct LumenEngine {
    det: Session,
    cls: Option<Session>,
    cls_dims: (u32, u32),
    rec: Session,
    dict: Vec<String>,
}

impl LumenEngine {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let dylib = resolve_runtime_dylib(model_dir)
            .ok_or_else(|| "未找到 ONNX Runtime 动态库，请重新下载本地 OCR 模型".to_string())?;
        ort::init_from(&dylib)
            .map_err(|e| format!("无法加载 ONNX Runtime 动态库: {e}"))?
            .with_name("lumen-ocr-worker")
            .commit();

        let threads = std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 4))
            .unwrap_or(2);

        let build_session = |path: &Path| -> Result<Session, String> {
            Session::builder()
                .map_err(|e| format!("无法创建推理会话: {e}"))?
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(|e| format!("无法配置推理图优化: {e}"))?
                .with_intra_threads(threads)
                .map_err(|e| format!("无法配置推理线程数: {e}"))?
                .with_inter_threads(1)
                .map_err(|e| format!("无法配置推理并发数: {e}"))?
                .commit_from_file(path)
                .map_err(|e| format!("无法加载模型 {}: {e}", path.display()))
        };

        let det = build_session(&model_dir.join(DET_MODEL_FILE))?;
        let cls_path = model_dir.join(CLS_MODEL_FILE);
        let cls = if cls_path.is_file() {
            Some(build_session(&cls_path)?)
        } else {
            None
        };
        let cls_dims = cls
            .as_ref()
            .and_then(|session| {
                let shape = session.inputs().first()?.dtype().tensor_shape()?;
                let h = *shape.get(2)?;
                let w = *shape.get(3)?;
                (h > 0 && w > 0).then_some((h as u32, w as u32))
            })
            .unwrap_or((80, 160));

        let rec = build_session(&model_dir.join(REC_MODEL_FILE))?;
        let dict = load_dictionary(&rec, &model_dir.join(DICT_FILE))?;

        Ok(Self {
            det,
            cls,
            cls_dims,
            rec,
            dict,
        })
    }

    pub fn recognize(&mut self, image: &RgbaImage) -> Result<Vec<OcrLine>, String> {
        let (orig_w, orig_h) = image.dimensions();
        if orig_w == 0 || orig_h == 0 {
            return Ok(Vec::new());
        }
        let rgb = rgba_to_rgb(image);
        let quads = self.detect(&rgb)?;
        if quads.is_empty() {
            return Ok(Vec::new());
        }

        let mut lines = Vec::with_capacity(quads.len());
        for quad in &quads {
            let Some(mut crop) = crop_quad(&rgb, quad) else {
                continue;
            };
            if self.should_rotate_180(&crop)? {
                crop = image::imageops::rotate180(&crop);
            }
            let (text, score) = self.recognize_crop(&crop)?;
            if text.is_empty() || score < REC_MIN_SCORE {
                continue;
            }
            lines.push(OcrLine {
                text,
                bounds: quad.bounding_rect(orig_w, orig_h),
                confidence: Some(score),
            });
        }
        Ok(lines)
    }

    fn detect(&mut self, rgb: &RgbImage) -> Result<Vec<crate::postprocess::DetectedQuad>, String> {
        let (orig_w, orig_h) = rgb.dimensions();
        let (resize_w, resize_h) = det_resize_dims(orig_w, orig_h, DET_LIMIT_SIDE_LEN);
        let resized = if resize_w == orig_w && resize_h == orig_h {
            rgb.clone()
        } else {
            image::imageops::resize(rgb, resize_w, resize_h, FilterType::Triangle)
        };

        let mut input = Array4::<f32>::zeros((1, 3, resize_h as usize, resize_w as usize));
        for y in 0..resize_h as usize {
            for x in 0..resize_w as usize {
                let p = resized.get_pixel(x as u32, y as u32).0;
                let bgr = [
                    p[2] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[0] as f32 / 255.0,
                ];
                for c in 0..3 {
                    input[[0, c, y, x]] = (bgr[c] - DET_MEAN[c]) / DET_STD[c];
                }
            }
        }

        let tensor =
            TensorRef::from_array_view(&input).map_err(|e| format!("构建检测输入张量失败: {e}"))?;
        let outputs = self
            .det
            .run(ort::inputs![tensor])
            .map_err(|e| format!("文本检测推理失败: {e}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("读取检测输出张量失败: {e}"))?;
        let (map_h, map_w) = match shape.as_ref() {
            [_, _, h, w] => (*h as usize, *w as usize),
            [_, h, w] => (*h as usize, *w as usize),
            _ => return Err(format!("不支持的检测输出维度: {shape:?}")),
        };
        Ok(dbnet_postprocess(
            data,
            map_w,
            map_h,
            orig_w,
            orig_h,
            DET_THRESH,
            DET_BOX_THRESH,
            DET_UNCLIP_RATIO,
        ))
    }

    fn should_rotate_180(&mut self, crop: &RgbImage) -> Result<bool, String> {
        let Some(cls) = self.cls.as_mut() else {
            return Ok(false);
        };
        let (target_h, target_w) = self.cls_dims;
        let (w, h) = crop.dimensions();
        let ratio = w as f32 / h.max(1) as f32;
        let resized_w = ((target_h as f32 * ratio).ceil() as u32).clamp(1, target_w);
        let resized = image::imageops::resize(crop, resized_w, target_h, FilterType::Triangle);

        let mut input = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));
        for y in 0..target_h as usize {
            for x in 0..resized_w as usize {
                let p = resized.get_pixel(x as u32, y as u32).0;
                let bgr = [p[2], p[1], p[0]];
                for c in 0..3 {
                    input[[0, c, y, x]] = (bgr[c] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }

        let tensor =
            TensorRef::from_array_view(&input).map_err(|e| format!("构建方向分类张量失败: {e}"))?;
        let outputs = cls
            .run(ort::inputs![tensor])
            .map_err(|e| format!("方向分类推理失败: {e}"))?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("读取方向分类输出失败: {e}"))?;
        Ok(data.len() >= 2 && data[1] > data[0] && data[1] >= CLS_THRESH)
    }

    fn recognize_crop(&mut self, crop: &RgbImage) -> Result<(String, f32), String> {
        let (w, h) = crop.dimensions();
        let ratio = w as f32 / h.max(1) as f32;
        let target_w = ((REC_IMG_HEIGHT as f32 * ratio).ceil() as u32).clamp(32, REC_MAX_WIDTH);
        let resized = image::imageops::resize(crop, target_w, REC_IMG_HEIGHT, FilterType::Triangle);

        let mut input = Array4::<f32>::zeros((1, 3, REC_IMG_HEIGHT as usize, target_w as usize));
        for y in 0..REC_IMG_HEIGHT as usize {
            for x in 0..target_w as usize {
                let p = resized.get_pixel(x as u32, y as u32).0;
                let bgr = [p[2], p[1], p[0]];
                for c in 0..3 {
                    input[[0, c, y, x]] = (bgr[c] as f32 / 255.0 - 0.5) / 0.5;
                }
            }
        }

        let tensor =
            TensorRef::from_array_view(&input).map_err(|e| format!("构建文本识别张量失败: {e}"))?;
        let outputs = self
            .rec
            .run(ort::inputs![tensor])
            .map_err(|e| format!("文本识别推理失败: {e}"))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("读取文本识别输出失败: {e}"))?;
        let (timesteps, num_classes) = match shape.as_ref() {
            [_, t, c] => (*t as usize, *c as usize),
            [t, c] => (*t as usize, *c as usize),
            _ => return Err(format!("不支持的识别输出维度: {shape:?}")),
        };
        Ok(ctc_greedy_decode(data, timesteps, num_classes, &self.dict))
    }
}

fn load_dictionary(rec: &Session, fallback_path: &Path) -> Result<Vec<String>, String> {
    if let Ok(meta) = rec.metadata()
        && let Some(raw) = meta.custom("character")
        && !raw.trim().is_empty()
    {
        return Ok(parse_dictionary(&raw));
    }
    let raw = fs::read_to_string(fallback_path)
        .map_err(|e| format!("无法读取识别字典 {}: {e}", fallback_path.display()))?;
    Ok(parse_dictionary(&raw))
}

/// Entry point when an executable is spawned with `--ocr-worker --model-dir <path>`.
pub fn worker_main(model_dir: &Path) -> i32 {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());

    let mut engine: Option<LumenEngine> = None;

    let send =
        |writer: &mut BufWriter<std::io::StdoutLock<'_>>, response: &WorkerResponse| -> bool {
            let Ok(bytes) = serde_json::to_vec(response) else {
                return false;
            };
            write_sync_frame(writer, &bytes, &[]).is_ok()
        };

    while let Ok(Some((header, payload))) = read_sync_frame(&mut reader) {
        let Ok(request) = serde_json::from_slice::<WorkerRequest>(&header) else {
            let _ = send(
                &mut writer,
                &WorkerResponse::Error {
                    message: "无效的 OCR 推理请求".into(),
                },
            );
            continue;
        };

        match request {
            WorkerRequest::Ping => {
                if !send(&mut writer, &WorkerResponse::Pong) {
                    break;
                }
            }
            WorkerRequest::Shutdown => {
                let _ = send(&mut writer, &WorkerResponse::Bye);
                break;
            }
            WorkerRequest::Recognize {
                width,
                height,
                rgba_len,
            } => {
                if payload.len() != rgba_len {
                    if !send(
                        &mut writer,
                        &WorkerResponse::Error {
                            message: "图像数据长度不匹配".into(),
                        },
                    ) {
                        break;
                    }
                    continue;
                }
                if engine.is_none() {
                    match LumenEngine::load(model_dir) {
                        Ok(loaded) => engine = Some(loaded),
                        Err(message) => {
                            let _ = send(&mut writer, &WorkerResponse::Error { message });
                            let _ = std::io::Write::flush(&mut writer);
                            return exit_clean(1);
                        }
                    }
                }
                let Some(image) = RgbaImage::from_raw(width, height, payload) else {
                    if !send(
                        &mut writer,
                        &WorkerResponse::Error {
                            message: "图像尺寸与像素缓冲区不匹配".into(),
                        },
                    ) {
                        break;
                    }
                    continue;
                };
                let response = match engine.as_mut().unwrap().recognize(&image) {
                    Ok(lines) => WorkerResponse::Lines { lines },
                    Err(message) => WorkerResponse::Error { message },
                };
                if !send(&mut writer, &response) {
                    break;
                }
            }
        }
    }

    drop(engine);
    let _ = std::io::Write::flush(&mut writer);
    exit_clean(0)
}

fn exit_clean(code: i32) -> i32 {
    // On Linux with `ort`'s `load-dynamic`, glibc runs dlopen'd `libonnxruntime.so`
    // C++ destructors before the binary's `.fini_array` (`ReleaseEnv`), which can
    // trigger a heap abort on exit. Exiting the isolated worker via `_exit` lets
    // the kernel reclaim all memory cleanly with zero coredumps.
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn _exit(status: i32) -> !;
        }
        unsafe { _exit(code) };
    }
    #[cfg(not(target_os = "linux"))]
    code
}
