use crate::{
    error::{ErrorKind, LumenError, Result},
    types::OcrLine,
};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Cap a single IPC frame to 128 MiB so corrupted length prefixes cannot OOM.
pub const MAX_FRAME_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerRequest {
    Ping,
    Recognize {
        width: u32,
        height: u32,
        rgba_len: usize,
    },
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerResponse {
    Pong,
    Lines { lines: Vec<OcrLine> },
    Error { message: String },
    Bye,
}

pub fn write_sync_frame<W: Write>(
    writer: &mut W,
    header: &[u8],
    payload: &[u8],
) -> std::io::Result<()> {
    let header_len = u32::try_from(header.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "header too large"))?;
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "payload too large"))?;
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&payload_len.to_le_bytes())?;
    writer.write_all(header)?;
    if !payload.is_empty() {
        writer.write_all(payload)?;
    }
    writer.flush()
}

pub fn read_sync_frame<R: Read>(reader: &mut R) -> std::io::Result<Option<(Vec<u8>, Vec<u8>)>> {
    let mut len_buf = [0u8; 8];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }
    let header_len = u32::from_le_bytes(len_buf[0..4].try_into().unwrap()) as usize;
    let payload_len = u32::from_le_bytes(len_buf[4..8].try_into().unwrap()) as usize;
    if header_len == 0 || header_len.saturating_add(payload_len) > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid frame size",
        ));
    }
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header)?;
    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        reader.read_exact(&mut payload)?;
    }
    Ok(Some((header, payload)))
}

pub async fn write_async_frame<W: AsyncWrite + Unpin>(
    writer: &mut W,
    header: &[u8],
    payload: &[u8],
) -> Result<()> {
    let io_err = |_| LumenError::new(ErrorKind::OcrUnavailable, "本地 OCR 推理进程通信中断");
    let header_len = u32::try_from(header.len())
        .map_err(|_| LumenError::new(ErrorKind::InvalidInput, "OCR 请求头过大"))?;
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| LumenError::new(ErrorKind::InvalidInput, "OCR 图像过大，请缩小选区"))?;
    writer
        .write_all(&header_len.to_le_bytes())
        .await
        .map_err(io_err)?;
    writer
        .write_all(&payload_len.to_le_bytes())
        .await
        .map_err(io_err)?;
    writer.write_all(header).await.map_err(io_err)?;
    if !payload.is_empty() {
        writer.write_all(payload).await.map_err(io_err)?;
    }
    writer.flush().await.map_err(io_err)?;
    Ok(())
}

pub async fn read_async_frame<R: AsyncRead + Unpin>(reader: &mut R) -> Result<(Vec<u8>, Vec<u8>)> {
    let io_err = |_| LumenError::new(ErrorKind::OcrUnavailable, "本地 OCR 推理进程意外退出");
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf).await.map_err(io_err)?;
    let header_len = u32::from_le_bytes(len_buf[0..4].try_into().unwrap()) as usize;
    let payload_len = u32::from_le_bytes(len_buf[4..8].try_into().unwrap()) as usize;
    if header_len == 0 || header_len.saturating_add(payload_len) > MAX_FRAME_BYTES {
        return Err(LumenError::new(
            ErrorKind::InvalidResponse,
            "本地 OCR 推理进程返回数据无效",
        ));
    }
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header).await.map_err(io_err)?;
    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        reader.read_exact(&mut payload).await.map_err(io_err)?;
    }
    Ok((header, payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rect;
    use std::io::Cursor;

    #[test]
    fn sync_frame_roundtrip_and_eof() {
        let mut buf = Vec::new();
        let req = WorkerRequest::Recognize {
            width: 2,
            height: 1,
            rgba_len: 8,
        };
        let header = serde_json::to_vec(&req).unwrap();
        let payload = vec![1, 2, 3, 4, 5, 6, 7, 8];
        write_sync_frame(&mut buf, &header, &payload).unwrap();

        let mut cursor = Cursor::new(buf);
        let (rx_h, rx_p) = read_sync_frame(&mut cursor).unwrap().unwrap();
        assert_eq!(rx_p, payload);
        match serde_json::from_slice::<WorkerRequest>(&rx_h).unwrap() {
            WorkerRequest::Recognize {
                width,
                height,
                rgba_len,
            } => assert_eq!((width, height, rgba_len), (2, 1, 8)),
            _ => panic!("unexpected request"),
        }
        assert!(read_sync_frame(&mut cursor).unwrap().is_none());
    }

    #[tokio::test]
    async fn async_frame_roundtrip_lines() {
        let resp = WorkerResponse::Lines {
            lines: vec![OcrLine {
                text: "你好 Hello".into(),
                bounds: Rect {
                    x: 10.0,
                    y: 20.0,
                    width: 120.0,
                    height: 24.0,
                },
                confidence: Some(0.97),
            }],
        };
        let header = serde_json::to_vec(&resp).unwrap();
        let mut wire = Vec::new();
        write_async_frame(&mut wire, &header, &[]).await.unwrap();
        let (rx_h, rx_p) = read_async_frame(&mut Cursor::new(wire)).await.unwrap();
        assert!(rx_p.is_empty());
        match serde_json::from_slice::<WorkerResponse>(&rx_h).unwrap() {
            WorkerResponse::Lines { lines } => assert_eq!(lines[0].text, "你好 Hello"),
            _ => panic!("unexpected response"),
        }
    }
}
