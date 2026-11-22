//! Pure-Rust PP-OCRv5 preprocessing and postprocessing (no OpenCV dependency).

use crate::types::Rect;
use image::{Rgb, RgbImage, RgbaImage};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct DetectedQuad {
    /// Four corners ordered clockwise starting from top-left: [tl, tr, br, bl].
    pub corners: [Point2; 4],
    pub score: f32,
}

impl DetectedQuad {
    pub fn bounding_rect(&self, max_w: u32, max_h: u32) -> Rect {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for p in self.corners {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        let x = min_x.clamp(0.0, max_w as f32) as f64;
        let y = min_y.clamp(0.0, max_h as f32) as f64;
        let r = max_x.clamp(0.0, max_w as f32) as f64;
        let b = max_y.clamp(0.0, max_h as f32) as f64;
        Rect {
            x,
            y,
            width: (r - x).max(1.0),
            height: (b - y).max(1.0),
        }
    }
}

pub fn det_resize_dims(width: u32, height: u32, limit_side_len: u32) -> (u32, u32) {
    let max_side = width.max(height).max(1);
    let ratio = if max_side > limit_side_len {
        limit_side_len as f32 / max_side as f32
    } else {
        1.0
    };
    let round32 = |v: f32| -> u32 { ((v / 32.0).round() as u32 * 32).max(32) };
    (
        round32(width as f32 * ratio),
        round32(height as f32 * ratio),
    )
}

pub fn rgba_to_rgb(image: &RgbaImage) -> RgbImage {
    let (w, h) = image.dimensions();
    RgbImage::from_fn(w, h, |x, y| {
        let p = image.get_pixel(x, y).0;
        if p[3] == 255 {
            Rgb([p[0], p[1], p[2]])
        } else {
            let a = p[3] as u16;
            let inv = 255 - a;
            Rgb([
                ((p[0] as u16 * a + 255 * inv) / 255) as u8,
                ((p[1] as u16 * a + 255 * inv) / 255) as u8,
                ((p[2] as u16 * a + 255 * inv) / 255) as u8,
            ])
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub fn dbnet_postprocess(
    prob_map: &[f32],
    map_w: usize,
    map_h: usize,
    orig_w: u32,
    orig_h: u32,
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
) -> Vec<DetectedQuad> {
    if map_w == 0 || map_h == 0 || prob_map.len() < map_w * map_h {
        return Vec::new();
    }

    let mut visited = vec![false; map_w * map_h];
    let mut queue = VecDeque::new();
    let mut quads = Vec::new();

    let scale_x = orig_w as f32 / map_w as f32;
    let scale_y = orig_h as f32 / map_h as f32;

    for sy in 0..map_h {
        for sx in 0..map_w {
            let start = sy * map_w + sx;
            if visited[start] || prob_map[start] < thresh {
                continue;
            }
            visited[start] = true;
            queue.push_back((sx, sy));

            let mut boundary = Vec::new();
            let mut sum_prob = 0.0_f64;
            let mut count = 0usize;

            while let Some((x, y)) = queue.pop_front() {
                let idx = y * map_w + x;
                sum_prob += prob_map[idx] as f64;
                count += 1;

                let mut is_border = false;
                for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= map_w as isize || ny >= map_h as isize {
                        is_border = true;
                        continue;
                    }
                    let nux = nx as usize;
                    let nuy = ny as usize;
                    let nidx = nuy * map_w + nux;
                    if prob_map[nidx] < thresh {
                        is_border = true;
                    } else if !visited[nidx] {
                        visited[nidx] = true;
                        queue.push_back((nux, nuy));
                    }
                }
                if is_border {
                    boundary.push(Point2 {
                        x: x as f32 + 0.5,
                        y: y as f32 + 0.5,
                    });
                }
            }

            if count < 4 || boundary.len() < 3 {
                continue;
            }
            let mean_score = (sum_prob / count as f64) as f32;
            if mean_score < box_thresh {
                continue;
            }

            let hull = convex_hull(&boundary);
            if hull.len() < 3 {
                continue;
            }
            let rect = min_area_rect(&hull);
            if rect.width.min(rect.height) < 3.0 {
                continue;
            }

            let area = rect.width * rect.height;
            let perimeter = 2.0 * (rect.width + rect.height);
            let expand = if perimeter > 1e-4 {
                area * unclip_ratio / perimeter
            } else {
                0.0
            };

            let expanded = RotatedRect {
                cx: rect.cx,
                cy: rect.cy,
                width: rect.width + 2.0 * expand,
                height: rect.height + 2.0 * expand,
                angle: rect.angle,
            };

            let mut corners = order_quad_clockwise(expanded.corners());
            for p in &mut corners {
                p.x = (p.x * scale_x).clamp(0.0, orig_w as f32);
                p.y = (p.y * scale_y).clamp(0.0, orig_h as f32);
            }

            let w_edge = dist(corners[0], corners[1]).max(dist(corners[3], corners[2]));
            let h_edge = dist(corners[0], corners[3]).max(dist(corners[1], corners[2]));
            if w_edge < 4.0 || h_edge < 4.0 {
                continue;
            }

            quads.push(DetectedQuad {
                corners,
                score: mean_score,
            });
            if quads.len() >= 1000 {
                break;
            }
        }
    }

    sort_quads_reading_order(&mut quads);
    quads
}

#[derive(Debug, Clone, Copy)]
struct RotatedRect {
    cx: f32,
    cy: f32,
    width: f32,
    height: f32,
    angle: f32,
}

impl RotatedRect {
    fn corners(self) -> [Point2; 4] {
        let cos = self.angle.cos();
        let sin = self.angle.sin();
        let hw = self.width * 0.5;
        let hh = self.height * 0.5;
        let ux = Point2 {
            x: cos * hw,
            y: sin * hw,
        };
        let uy = Point2 {
            x: -sin * hh,
            y: cos * hh,
        };
        [
            Point2 {
                x: self.cx - ux.x - uy.x,
                y: self.cy - ux.y - uy.y,
            },
            Point2 {
                x: self.cx + ux.x - uy.x,
                y: self.cy + ux.y - uy.y,
            },
            Point2 {
                x: self.cx + ux.x + uy.x,
                y: self.cy + ux.y + uy.y,
            },
            Point2 {
                x: self.cx - ux.x + uy.x,
                y: self.cy - ux.y + uy.y,
            },
        ]
    }
}

fn cross(o: Point2, a: Point2, b: Point2) -> f32 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

fn dist(a: Point2, b: Point2) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

fn convex_hull(points: &[Point2]) -> Vec<Point2> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then_with(|| a.y.total_cmp(&b.y)));
    pts.dedup_by(|a, b| (a.x - b.x).abs() < 1e-4 && (a.y - b.y).abs() < 1e-4);
    if pts.len() <= 2 {
        return pts;
    }
    let mut lower: Vec<Point2> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Point2> = Vec::with_capacity(pts.len());
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn min_area_rect(hull: &[Point2]) -> RotatedRect {
    let mut best_area = f32::INFINITY;
    let mut best = RotatedRect {
        cx: 0.0,
        cy: 0.0,
        width: 0.0,
        height: 0.0,
        angle: 0.0,
    };
    let n = hull.len();
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            continue;
        }
        let cos = dx / len;
        let sin = dy / len;

        let mut min_u = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for &p in hull {
            let u = p.x * cos + p.y * sin;
            let v = -p.x * sin + p.y * cos;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let w = max_u - min_u;
        let h = max_v - min_v;
        let area = w * h;
        if area < best_area {
            best_area = area;
            let cu = 0.5 * (min_u + max_u);
            let cv = 0.5 * (min_v + max_v);
            best = RotatedRect {
                cx: cu * cos - cv * sin,
                cy: cu * sin + cv * cos,
                width: w,
                height: h,
                angle: dy.atan2(dx),
            };
        }
    }
    best
}

pub fn order_quad_clockwise(mut pts: [Point2; 4]) -> [Point2; 4] {
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then_with(|| a.y.total_cmp(&b.y)));
    let (tl, bl) = if pts[0].y <= pts[1].y {
        (pts[0], pts[1])
    } else {
        (pts[1], pts[0])
    };
    let (tr, br) = if pts[2].y <= pts[3].y {
        (pts[2], pts[3])
    } else {
        (pts[3], pts[2])
    };
    [tl, tr, br, bl]
}

pub fn sort_quads_reading_order(quads: &mut [DetectedQuad]) {
    quads.sort_by(|a, b| {
        let ay = a.corners[0].y;
        let by = b.corners[0].y;
        ay.total_cmp(&by)
            .then_with(|| a.corners[0].x.total_cmp(&b.corners[0].x))
    });
    for i in 0..quads.len().saturating_sub(1) {
        let mut j = i;
        while j > 0
            && (quads[j + 1].corners[0].y - quads[j].corners[0].y).abs() < 10.0
            && quads[j + 1].corners[0].x < quads[j].corners[0].x
        {
            quads.swap(j, j + 1);
            j -= 1;
        }
    }
}

pub fn crop_quad(rgb: &RgbImage, quad: &DetectedQuad) -> Option<RgbImage> {
    let [tl, tr, br, bl] = quad.corners;
    let crop_w = dist(tl, tr).max(dist(bl, br)).round() as u32;
    let crop_h = dist(tl, bl).max(dist(tr, br)).round() as u32;
    if crop_w < 2 || crop_h < 2 {
        return None;
    }

    let (img_w, img_h) = rgb.dimensions();
    let max_x = (img_w.saturating_sub(1)) as f32;
    let max_y = (img_h.saturating_sub(1)) as f32;

    let sample_bilinear = |fx: f32, fy: f32| -> Rgb<u8> {
        let x = fx.clamp(0.0, max_x);
        let y = fy.clamp(0.0, max_y);
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(img_w - 1);
        let y1 = (y0 + 1).min(img_h - 1);
        let wx = x - x0 as f32;
        let wy = y - y0 as f32;
        let p00 = rgb.get_pixel(x0, y0).0;
        let p10 = rgb.get_pixel(x1, y0).0;
        let p01 = rgb.get_pixel(x0, y1).0;
        let p11 = rgb.get_pixel(x1, y1).0;
        let mut out = [0u8; 3];
        for c in 0..3 {
            let top = p00[c] as f32 * (1.0 - wx) + p10[c] as f32 * wx;
            let bot = p01[c] as f32 * (1.0 - wx) + p11[c] as f32 * wx;
            out[c] = (top * (1.0 - wy) + bot * wy).round().clamp(0.0, 255.0) as u8;
        }
        Rgb(out)
    };

    let cropped = RgbImage::from_fn(crop_w, crop_h, |x, y| {
        let u = if crop_w > 1 {
            x as f32 / (crop_w - 1) as f32
        } else {
            0.5
        };
        let v = if crop_h > 1 {
            y as f32 / (crop_h - 1) as f32
        } else {
            0.5
        };
        let top_x = tl.x + (tr.x - tl.x) * u;
        let top_y = tl.y + (tr.y - tl.y) * u;
        let bot_x = bl.x + (br.x - bl.x) * u;
        let bot_y = bl.y + (br.y - bl.y) * u;
        sample_bilinear(top_x + (bot_x - top_x) * v, top_y + (bot_y - top_y) * v)
    });

    if crop_h as f32 / crop_w as f32 >= 1.5 {
        Some(image::imageops::rotate270(&cropped))
    } else {
        Some(cropped)
    }
}

pub fn parse_dictionary(raw: &str) -> Vec<String> {
    let mut dict = Vec::with_capacity(19_000);
    dict.push(String::new()); // index 0 = CTC blank
    for line in raw.lines() {
        let token = line.trim_end_matches('\r');
        dict.push(token.to_owned());
    }
    dict.push(" ".into());
    dict
}

pub fn ctc_greedy_decode(
    logits: &[f32],
    timesteps: usize,
    num_classes: usize,
    dict: &[String],
) -> (String, f32) {
    if timesteps == 0 || num_classes == 0 || logits.len() < timesteps * num_classes {
        return (String::new(), 0.0);
    }
    let mut text = String::new();
    let mut prev_idx = 0usize;
    let mut conf_sum = 0.0_f64;
    let mut conf_count = 0usize;

    for t in 0..timesteps {
        let row = &logits[t * num_classes..(t + 1) * num_classes];
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &val) in row.iter().enumerate() {
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        if best_idx != 0 && best_idx != prev_idx {
            if let Some(ch) = dict.get(best_idx) {
                text.push_str(ch);
            }
            conf_sum += best_val.clamp(0.0, 1.0) as f64;
            conf_count += 1;
        }
        prev_idx = best_idx;
    }

    let confidence = if conf_count > 0 {
        (conf_sum / conf_count as f64) as f32
    } else {
        0.0
    };
    (text.trim().to_owned(), confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_resize_caps_and_aligns_to_32() {
        assert_eq!(det_resize_dims(1920, 1080, 960), (960, 544));
        assert_eq!(det_resize_dims(100, 40, 960), (96, 32));
        assert_eq!(det_resize_dims(10, 10, 960), (32, 32));
    }

    #[test]
    fn dbnet_postprocess_detects_synthetic_text_lines_in_reading_order() {
        let (w, h) = (160, 120);
        let mut map = vec![0.0_f32; w * h];
        for y in 70..82 {
            for x in 20..120 {
                map[y * w + x] = 0.92;
            }
        }
        for y in 20..32 {
            for x in 30..130 {
                map[y * w + x] = 0.95;
            }
        }
        let quads = dbnet_postprocess(&map, w, h, 320, 240, 0.3, 0.6, 1.5);
        assert_eq!(quads.len(), 2);
        let r0 = quads[0].bounding_rect(320, 240);
        let r1 = quads[1].bounding_rect(320, 240);
        assert!(r0.y < r1.y);
        assert!(r0.width > 180.0);
        assert!(r1.width > 180.0);
    }

    #[test]
    fn ctc_decoder_collapses_repeats_and_skips_blanks() {
        let dict = parse_dictionary("你\n好\nA\nB");
        let steps = [0, 1, 1, 0, 2, 5, 3, 4];
        let num_classes = dict.len();
        let mut logits = vec![0.0_f32; steps.len() * num_classes];
        for (t, &idx) in steps.iter().enumerate() {
            logits[t * num_classes + idx] = 0.9;
        }
        let (text, conf) = ctc_greedy_decode(&logits, steps.len(), num_classes, &dict);
        assert_eq!(text, "你好 AB");
        assert!((conf - 0.9).abs() < 1e-4);
    }
}
