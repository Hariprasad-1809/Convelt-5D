#!/usr/bin/env python3
"""
JointGuard YOLO Classification - Live Webcam Inspection Program
Robust Moving-Belt Pipeline with Calibrated Confidence & Margin Gating:

Features:
- Default camera source: 1 (configurable via --source)
- Motion Blur Rejection: Rejects blurry crops (Laplacian variance < min_sharpness)
- Configurable Asymmetric Thresholds: DAMAGE_THRESH (0.70) & HEALTHY_THRESH (0.70)
- Confidence Margin Gate: Requires |p_damage - p_healthy| >= CONF_MARGIN (0.15) before assigning DAMAGE/HEALTHY
- Inner Inspection Zone: Only collects sharp frames when joint is inside central ROI zone
- Multi-Frame Majority Voting: Accumulates sharp frames (e.g. 5) for stable decision
- Quality Gate: Returns WAITING if fewer than required sharp frames (3/5) are collected
- Diagnostic Debug Crop Saver: Saves crops to check_yolo/debug_crops/ for root-cause audit
- Live HUD Overlay: Displays Healthy %, Damage %, Sharpness, Valid count, and Final verdict
- End-of-Run Diagnostic Summary: Prints total counts & average confidence scores on exit

Usage:
    python check_yolo/webcam.py
    python check_yolo/webcam.py --source 1 --show-crop --damage-thresh 0.70 --healthy-thresh 0.70 --conf-margin 0.15
"""

import argparse
from collections import deque
import datetime
import json
import os
import sys
import time
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_MODEL_V4_PATH = os.path.join(SCRIPT_DIR, "models", "joint_yolo_classifier_v4.pt")
DEFAULT_MODEL_V2_PATH = os.path.join(SCRIPT_DIR, "models", "joint_yolo_classifier_v2.pt")
DEFAULT_MODEL_V1_PATH = os.path.join(SCRIPT_DIR, "models", "joint_yolo_classifier.pt")
FALLBACK_MODEL_PATH = os.path.join(SCRIPT_DIR, "results", "joint_cls_v4", "weights", "best.pt")

DEFAULT_JSONL_PATH = os.path.join(PROJECT_ROOT, "data", "joint_events.jsonl")
DEBUG_CROPS_DIR = os.path.join(SCRIPT_DIR, "debug_crops")
COLLECTED_DIR = os.path.join(SCRIPT_DIR, "dataset", "collected")
COLLECTED_HEALTHY_DIR = os.path.join(COLLECTED_DIR, "healthy")
COLLECTED_DAMAGE_DIR = os.path.join(COLLECTED_DIR, "damage")


def parse_args():
    parser = argparse.ArgumentParser(
        description="JointGuard Live Webcam: Calibrated Confidence & Margin-Gated Moving-Belt Inspection Pipeline"
    )
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit path to trained YOLO classification weights",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        choices=["v1", "v2", "v4"],
        default="v4",
        help="YOLO model version: 'v4' (default), 'v2', or 'v1'",
    )
    parser.add_argument(
        "--v1", action="store_true", help="Shortcut for v1 model"
    )
    parser.add_argument(
        "--v2", action="store_true", help="Shortcut for v2 model"
    )
    parser.add_argument(
        "--v4", action="store_true", help="Shortcut for v4 model (default)"
    )
    parser.add_argument(
        "--damage-thresh",
        type=float,
        default=0.70,
        help="Minimum confidence threshold required to declare DAMAGE (default: 0.70)",
    )
    parser.add_argument(
        "--healthy-thresh",
        type=float,
        default=0.70,
        help="Minimum confidence threshold required to declare HEALTHY (default: 0.70)",
    )
    parser.add_argument(
        "--conf-margin",
        type=float,
        default=0.15,
        help="Minimum confidence margin between classes |p_damage - p_healthy| (default: 0.15)",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=100.0,
        help="Minimum Laplacian variance for motion blur filter (default: 100.0)",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=5,
        help="Number of valid sharp frames to accumulate for majority voting (default: 5)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="0.15,0.20,0.85,0.80",
        help="Fixed Search ROI as 'x1,y1,x2,y2' (default: 0.15,0.20,0.85,0.80)",
    )
    parser.add_argument(
        "--min-joint-area",
        type=int,
        default=600,
        help="Minimum contour area to consider a joint candidate (default: 600)",
    )
    parser.add_argument(
        "--min-joint-width",
        type=int,
        default=40,
        help="Minimum bounding box width to consider a joint candidate (default: 40)",
    )
    parser.add_argument(
        "--direct-roi",
        action="store_true",
        help="Bypass OpenCV joint detection and feed entire Search ROI directly to YOLO",
    )
    parser.add_argument(
        "--show-crop",
        action="store_true",
        help="Open secondary window displaying the exact image crop sent to YOLO",
    )
    parser.add_argument(
        "--save-debug-crops",
        action="store_true",
        help="Automatically save evaluation crops into check_yolo/debug_crops/ for audit",
    )
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Disable automatic dark/blur crop validation filter",
    )
    parser.add_argument(
        "--jsonl-log",
        type=str,
        default=DEFAULT_JSONL_PATH,
        help=f"Path to output diagnostic JSONL event log file (default: {DEFAULT_JSONL_PATH})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested webcam width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested webcam height (default: 480)",
    )
    return parser.parse_args()


def resolve_model_path(requested_path: Optional[str] = None, version: str = "v4") -> Tuple[str, str]:
    if requested_path:
        if os.path.isfile(requested_path):
            base_lower = os.path.basename(requested_path).lower()
            tag = "v4" if "v4" in base_lower else ("v2" if "v2" in base_lower else "v1")
            return os.path.abspath(requested_path), tag
        raise FileNotFoundError(f"Requested YOLO model file does not exist: {requested_path}")

    if version == "v4":
        if os.path.isfile(DEFAULT_MODEL_V4_PATH):
            return os.path.abspath(DEFAULT_MODEL_V4_PATH), "v4"
        if os.path.isfile(FALLBACK_MODEL_PATH):
            return os.path.abspath(FALLBACK_MODEL_PATH), "v4-fallback"

    if os.path.isfile(DEFAULT_MODEL_V4_PATH):
        return os.path.abspath(DEFAULT_MODEL_V4_PATH), "v4"
    if os.path.isfile(DEFAULT_MODEL_V2_PATH):
        return os.path.abspath(DEFAULT_MODEL_V2_PATH), "v2"
    if os.path.isfile(DEFAULT_MODEL_V1_PATH):
        return os.path.abspath(DEFAULT_MODEL_V1_PATH), "v1"

    raise FileNotFoundError("Trained YOLO model not found in check_yolo/models/")


def parse_roi_string(roi_str: str, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    parts = [float(p.strip()) for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must contain exactly 4 coordinates 'x1,y1,x2,y2', got '{roi_str}'")

    x1, y1, x2, y2 = parts
    if max(x1, y1, x2, y2) <= 1.0:
        px1, py1 = int(x1 * frame_w), int(y1 * frame_h)
        px2, py2 = int(x2 * frame_w), int(y2 * frame_h)
    else:
        px1, py1, px2, py2 = int(x1), int(y1), int(x2), int(y2)

    px1 = max(0, min(frame_w - 10, px1))
    py1 = max(0, min(frame_h - 10, py1))
    px2 = max(px1 + 10, min(frame_w, px2))
    py2 = max(py1 + 10, min(frame_h, py2))
    return px1, py1, px2, py2


def find_joint_in_roi(
    roi_img: np.ndarray,
    min_area: int = 600,
    min_width: int = 40,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Classical OpenCV Joint Detector inside the Fixed Search ROI.
    """
    h, w = roi_img.shape[:2]
    if h < 20 or w < 20:
        return None

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2]
    s_chan = hsv[:, :, 1]

    metallic_highlights = (v_chan >= 180) & (s_chan <= 75)
    if int(np.count_nonzero(metallic_highlights)) < 250:
        return None

    if int(v_chan.max()) < 195:
        return None

    v_med = float(np.median(v_chan))
    v_lower = max(140, min(210, int(v_med + 20)))
    s_upper = 75

    mask_hsv = cv2.inRange(
        hsv, np.array([0, 0, v_lower], dtype=np.uint8), np.array([180, s_upper, 255], dtype=np.uint8)
    )

    gray_med = float(np.median(gray))
    gray_thresh = max(160, int(gray_med + 25))
    _, mask_gray_raw = cv2.threshold(gray, min(235, gray_thresh), 255, cv2.THRESH_BINARY)
    mask_gray = cv2.bitwise_and(mask_gray_raw, (s_chan <= s_upper).astype(np.uint8) * 255)
    mask = cv2.bitwise_or(mask_hsv, mask_gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < min_width or bh < 10:
            continue

        is_side_rail = (
            (bx <= 3 or bx + bw >= w - 3) and
            (bh > 2.5 * bw and bh >= int(0.7 * h))
        )
        if is_side_rail:
            continue

        # Reject narrow static background boundary artifacts pinned right against outer borders
        # (e.g. thin horizontal edge between dark rubber belt and light floor/mount behind it).
        # A real metallic joint has substantial height/thickness or spans across the belt,
        # whereas a background border artifact is a thin edge sliver (bh <= 25) or full-width floor boundary.
        is_bottom_background_edge = (
            (by + bh >= h - 3) and (by > 0.50 * h) and (bh <= 25 or bw >= int(0.85 * w))
        )
        if is_bottom_background_edge:
            continue

        is_top_background_edge = (
            (by <= 3) and (bh <= 25 and by + bh <= 30 and bw >= int(0.85 * w))
        )
        if is_top_background_edge:
            continue

        cand_v = v_chan[by:by+bh, bx:bx+bw]
        cand_s = s_chan[by:by+bh, bx:bx+bw]
        cand_metallic = np.count_nonzero((cand_v >= 175) & (cand_s <= 75))
        if cand_metallic < 150:
            continue

        bright_mask = cand_v >= 150
        bright_s_med = float(np.median(cand_s[bright_mask])) if np.any(bright_mask) else float(np.median(cand_s))
        cand_s_med = float(np.median(cand_s))
        if bright_s_med > 40.0 or cand_s_med > 45.0:
            continue

        sat_frac = float(np.count_nonzero(cand_s > 70)) / float(cand_s.size)
        if sat_frac > 0.25:
            continue

        cand_bgr = roi_img[by:by+bh, bx:bx+bw]
        b, g, r = cv2.split(cand_bgr)
        diff_rg = np.abs(r.astype(int) - g.astype(int))
        diff_gb = np.abs(g.astype(int) - b.astype(int))
        diff_rb = np.abs(r.astype(int) - b.astype(int))
        max_ch_diff = np.maximum(diff_rg, np.maximum(diff_gb, diff_rb))
        ch_diff_med = float(np.median(max_ch_diff[bright_mask])) if np.any(bright_mask) else float(np.median(max_ch_diff))
        if ch_diff_med > 28.0:
            continue

        candidates.append((area, (bx, by, bw, bh)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    bx, by, bw, bh = candidates[0][1]
    bx2, by2 = bx + bw, by + bh

    for _, (fx, fy, fw, fh) in candidates[1:]:
        fx2, fy2 = fx + fw, fy + fh
        if not (fx > bx2 + 35 or fx2 < bx - 35 or fy > by2 + 35 or fy2 < by - 35):
            bx = min(bx, fx)
            by = min(by, fy)
            bx2 = max(bx2, fx2)
            by2 = max(by2, fy2)

    pad_y = int((by2 - by) * 0.12)
    pad_x = int((bx2 - bx) * 0.06)

    final_x = max(0, bx - pad_x)
    final_y = max(0, by - pad_y)
    final_w = min(w - final_x, (bx2 - bx) + 2 * pad_x)
    final_h = min(h - final_y, (by2 - by) + 2 * pad_y)

    return (final_x, final_y, final_w, final_h)


def is_in_inner_inspection_zone(
    rel_bbox: Tuple[int, int, int, int],
    roi_w: int,
    roi_h: int,
    is_currently_inside: bool = False
) -> bool:
    """
    Checks if joint centroid is inside inner inspection zone with hysteresis.
    - Entry Boundary: stricter central 70% zone (prevents early edge triggers)
    - Tracking/Exit Boundary: generous 90% zone (prevents boundary chatter/flicker)
    """
    jx, jy, jw, jh = rel_bbox
    cx = jx + jw / 2.0
    cy = jy + jh / 2.0

    if is_currently_inside:
        # Generous "still tracking" boundary
        xmin = roi_w * 0.03
        xmax = roi_w * 0.97
        ymin = roi_h * 0.04
        ymax = roi_h * 0.96
    else:
        # Stricter entry boundary
        xmin = roi_w * 0.08
        xmax = roi_w * 0.92
        ymin = roi_h * 0.10
        ymax = roi_h * 0.90

    return (xmin <= cx <= xmax) and (ymin <= cy <= ymax)


def calculate_sharpness(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_crop_quality(
    crop: np.ndarray,
    min_sharpness: float = 100.0,
    is_approaching: bool = False
) -> Tuple[bool, str, float]:
    """
    Validates crop quality before sending to YOLO.
    Rejects undersized, bad aspect ratio, dark, bright, or motion-blurred crops.
    """
    h, w = crop.shape[:2]
    if h < 20 or w < 20:
        return False, "TOO SMALL", 0.0

    aspect_ratio = float(h) / float(w)
    # Loosened aspect ratio: allow 0.08 to 4.5 standard, widened to 0.06 to 5.0 while approaching
    max_ar = 5.0 if is_approaching else 4.5
    min_ar = 0.06 if is_approaching else 0.08
    if aspect_ratio > max_ar or aspect_ratio < min_ar:
        return False, "BAD ASPECT RATIO", 0.0

    mean_val = float(np.mean(crop))
    if mean_val < 15.0:
        return False, "TOO DARK", 0.0
    if mean_val > 245.0:
        return False, "TOO BRIGHT", 0.0

    sharpness = calculate_sharpness(crop)
    if sharpness < min_sharpness:
        return False, f"BLURRY ({sharpness:.1f} < {min_sharpness:.1f})", sharpness

    return True, "OK", sharpness


def generate_contact_sheet(joint_id: str, debug_crops_dir: str, crop_history: List[Tuple[np.ndarray, str, float, float]]):
    """
    Generates a grid contact sheet showing up to 20 consecutive crops for joint_id.
    """
    if not crop_history:
        return
    try:
        samples = crop_history[:20]
        n = len(samples)
        cols = 5
        rows = int(np.ceil(n / cols))

        cell_w, cell_h = 160, 160
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for idx, (crop, label, p_h, p_d) in enumerate(samples):
            r, c = idx // cols, idx % cols
            resized = cv2.resize(crop, (cell_w, cell_h))

            color = (0, 220, 0) if label == "HEALTHY" else ((0, 0, 230) if label == "DAMAGE" else (0, 200, 255))
            conf = p_d if label == "DAMAGE" else p_h
            txt = f"F{idx+1}: {label[:3]} {conf*100:.0f}%"
            cv2.rectangle(resized, (0, cell_h - 22), (cell_w, cell_h), (15, 23, 42), -1)
            cv2.putText(resized, txt, (5, cell_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized

        out_fn = os.path.join(debug_crops_dir, f"{joint_id}_contact_sheet.jpg")
        cv2.imwrite(out_fn, grid)
        print(f"[DEBUG SHEET] Generated contact sheet for {joint_id}: {out_fn}")
    except Exception as e:
        print(f"[WARNING] Could not generate contact sheet for {joint_id}: {e}")


def compute_bbox_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


class JointTrack:
    """
    State object representing a single tracked conveyor belt joint (e.g. J01).
    """
    def __init__(self, joint_id: str, raw_bbox: Tuple[int, int, int, int], timestamp: float):
        self.joint_id = joint_id
        self.raw_bbox = raw_bbox
        self.smoothed_bbox = raw_bbox  # (x, y, w, h)
        self.last_centroid = (raw_bbox[0] + raw_bbox[2] / 2.0, raw_bbox[1] + raw_bbox[3] / 2.0)
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.total_frames_seen = 1
        self.total_frames_lost = 0
        self.tracking_status = "VISUALIZED"  # VISUALIZED, TEMPORARILY_LOST, EXITED
        self.inspection_state = "APPROACHING"  # APPROACHING, INSPECTING, CONFIRMED, PASSED

        # Buffer of valid sharp frames: (pred_label, p_healthy, p_damage, sharpness, timestamp)
        self.accumulated_samples: List[Tuple[str, float, float, float, str]] = []
        self.confirmed_label: str = "UNCERTAIN"  # HEALTHY, DAMAGE, UNCERTAIN
        self.confirmed_confidence: float = 0.0
        self.consecutive_contradictory_count: int = 0
        self.crop_history: List[Tuple[np.ndarray, str, float, float]] = []

        # Early-exit & rolling finalization lock
        self.is_early_finalized: bool = False
        self.finalized_at_state: Optional[str] = None
        self.finalized_via: str = "none"  # "early-lock-lifetime-average", "early-lock-recent-window", "fallback-at-PASSED", "none"

        # Robust State Machine Hysteresis & Debounce tracking
        self.is_inside_zone: bool = False
        self.consecutive_outside_frames: int = 0
        self.consecutive_lost_frames: int = 0

    def update_position(self, raw_bbox: Tuple[int, int, int, int], timestamp: float, alpha: float = 0.3):
        self.raw_bbox = raw_bbox
        rx, ry, rw, rh = raw_bbox
        sx, sy, sw, sh = self.smoothed_bbox

        # Exponential Moving Average Bounding Box Smoothing
        nx = int(alpha * rx + (1.0 - alpha) * sx)
        ny = int(alpha * ry + (1.0 - alpha) * sy)
        nw = int(alpha * rw + (1.0 - alpha) * sw)
        nh = int(alpha * rh + (1.0 - alpha) * sh)

        self.smoothed_bbox = (nx, ny, nw, nh)
        self.last_centroid = (nx + nw / 2.0, ny + nh / 2.0)
        self.last_seen_time = timestamp
        self.total_frames_seen += 1
        self.total_frames_lost = 0
        self.consecutive_lost_frames = 0
        self.tracking_status = "VISUALIZED"


class JointGuardStateEngine:
    """
    Robust Joint Inspection State Machine & Multi-Joint Track Manager.
    Features:
    - Persistent Joint Tracking & Association (J01, J02...)
    - EMA Bounding Box Smoothing (alpha = 0.3)
    - Temporal Hold Timeout during dropouts (TRACK_LOST_TIMEOUT = 2.0s)
    - Re-identification of recently lost tracks to prevent ID churn
    - Hysteresis & rolling early finalization for high-confidence DAMAGE
    - Full temporal history accumulation across track lifetime evaluated at PASSED
    - Aspect ratio clamping/fallback to eliminate stalls while joint enters
    - Debounced zone boundary evaluation preventing inside/outside flicker
    - Contact Sheet Generator (debug_crops/J01_contact_sheet.jpg)
    - Clean HUD Overlay & JSONL Event Logging
    """
    def __init__(
        self,
        target_frames: int = 5,
        min_sharpness: float = 100.0,
        damage_thresh: float = 0.70,
        healthy_thresh: float = 0.70,
        conf_margin: float = 0.15,
        track_lost_timeout: float = 2.0,
        bbox_smooth_alpha: float = 0.3,
        jsonl_path: Optional[str] = None,
        save_debug_crops: bool = False,
        debug_crops_dir: Optional[str] = None,
        debounce_outside_frames: int = 5,
        lost_bbox_grace_frames: int = 2,
    ):
        self.target_frames = target_frames
        self.min_sharpness = min_sharpness
        self.damage_thresh = damage_thresh
        self.healthy_thresh = healthy_thresh
        self.conf_margin = conf_margin
        self.track_lost_timeout = track_lost_timeout
        self.bbox_smooth_alpha = bbox_smooth_alpha
        self.jsonl_path = jsonl_path
        self.save_debug_crops = save_debug_crops
        self.debug_crops_dir = debug_crops_dir or DEBUG_CROPS_DIR
        self.debounce_outside_frames = debounce_outside_frames
        self.lost_bbox_grace_frames = lost_bbox_grace_frames

        self.joint_counter = 1
        self.active_track: Optional[JointTrack] = None
        self.recent_tracks: List[JointTrack] = []
        self.rejected_blurry_count = 0
        self.printed_debug_for_track: Dict[str, bool] = {}

        # Diagnostics
        self.diag_total_frames = 0
        self.diag_healthy_count = 0
        self.diag_damage_count = 0
        self.diag_uncertain_count = 0
        self.diag_healthy_confs: List[float] = []
        self.diag_damage_confs: List[float] = []

        if self.save_debug_crops:
            os.makedirs(self.debug_crops_dir, exist_ok=True)

    def _get_next_joint_id(self) -> str:
        jid = f"J{self.joint_counter:02d}"
        self.joint_counter += 1
        return jid

    def _finalize_track_verdict(self, track: JointTrack):
        """
        Evaluates and locks the FINAL verdict based on the full temporal history
        across the joint track's entire lifetime at the moment it transitions to PASSED.
        Guarantees majority-vote consistency: majority DAMAGE never produces HEALTHY.
        """
        if not track.accumulated_samples:
            return

        # If already locked as early-finalized DAMAGE, keep it locked
        if track.is_early_finalized and track.confirmed_label == "DAMAGE":
            return

        n_samples = len(track.accumulated_samples)
        avg_h = float(np.mean([s[1] for s in track.accumulated_samples]))
        avg_d = float(np.mean([s[2] for s in track.accumulated_samples]))
        damage_count = sum(1 for s in track.accumulated_samples if s[0] == "DAMAGE")
        healthy_count = sum(1 for s in track.accumulated_samples if s[0] == "HEALTHY")

        if n_samples >= min(3, self.target_frames):
            # Prioritize Majority Vote Consistency
            if damage_count > healthy_count:
                track.confirmed_label = "DAMAGE"
                track.confirmed_confidence = avg_d
            elif healthy_count > damage_count:
                if avg_d >= self.damage_thresh:
                    track.confirmed_label = "UNCERTAIN"
                    track.confirmed_confidence = max(avg_h, avg_d)
                else:
                    track.confirmed_label = "HEALTHY"
                    track.confirmed_confidence = avg_h
            else:
                # Tie: break tie in favor of safety
                if avg_d >= avg_h:
                    track.confirmed_label = "DAMAGE"
                    track.confirmed_confidence = avg_d
                else:
                    track.confirmed_label = "HEALTHY"
                    track.confirmed_confidence = avg_h

        # Consistency Safeguard: majority DAMAGE must NEVER have confirmed_label == HEALTHY
        if damage_count > healthy_count and track.confirmed_label == "HEALTHY":
            print(f"[CONSISTENCY OVERRIDE] Joint {track.joint_id}: majority was DAMAGE ({damage_count} vs {healthy_count}), overriding contradictory HEALTHY to DAMAGE!")
            track.confirmed_label = "DAMAGE"
            track.confirmed_confidence = avg_d

        track.finalized_via = "fallback-at-PASSED"
        print(f"[FINAL VERDICT] Joint {track.joint_id} finalized via fallback-at-PASSED: {track.confirmed_label} ({track.confirmed_confidence*100:.1f}%)")

    def process_frame(
        self,
        frame: np.ndarray,
        joint_detected: bool,
        raw_bbox_full: Optional[Tuple[int, int, int, int]],
        in_zone: bool,
        yolo_model: Any,
        no_quality_check: bool = False
    ) -> Dict[str, Any]:
        self.diag_total_frames += 1
        now_ts = time.time()
        timestamp = datetime.datetime.now().isoformat()
        reason = "no-change"

        # 1. TRACK ASSOCIATION OR CREATION (WITH RE-IDENTIFICATION & EXTENDED REACH)
        if joint_detected and raw_bbox_full is not None:
            jx, jy, jw, jh = raw_bbox_full
            det_centroid = (jx + jw / 2.0, jy + jh / 2.0)
            matched_track = None

            # A. Check active track
            if self.active_track is not None and self.active_track.tracking_status != "EXITED":
                dist = np.sqrt(
                    (det_centroid[0] - self.active_track.last_centroid[0])**2 +
                    (det_centroid[1] - self.active_track.last_centroid[1])**2
                )
                iou = compute_bbox_iou(raw_bbox_full, self.active_track.smoothed_bbox)
                dx = abs(det_centroid[0] - self.active_track.last_centroid[0])
                dy = abs(det_centroid[1] - self.active_track.last_centroid[1])

                # Match if close, overlapping, or moving along conveyor belt
                if dist <= 250.0 or iou > 0.04 or (dy <= 80.0 and dx <= 260.0):
                    matched_track = self.active_track

            # B. If active track didn't match or was None, check recent tracks (re-identification)
            if matched_track is None and self.recent_tracks:
                for r_track in reversed(self.recent_tracks):
                    if now_ts - r_track.last_seen_time <= 2.5:
                        r_dist = np.sqrt(
                            (det_centroid[0] - r_track.last_centroid[0])**2 +
                            (det_centroid[1] - r_track.last_centroid[1])**2
                        )
                        r_iou = compute_bbox_iou(raw_bbox_full, r_track.smoothed_bbox)
                        r_dx = abs(det_centroid[0] - r_track.last_centroid[0])
                        r_dy = abs(det_centroid[1] - r_track.last_centroid[1])
                        if r_dist <= 250.0 or r_iou > 0.04 or (r_dy <= 80.0 and r_dx <= 260.0):
                            matched_track = r_track
                            break

            if matched_track is not None:
                self.active_track = matched_track
                self.active_track.update_position(raw_bbox_full, now_ts, alpha=self.bbox_smooth_alpha)
            else:
                if self.active_track is not None:
                    self._close_joint_track(self.active_track)
                new_jid = self._get_next_joint_id()
                self.active_track = JointTrack(new_jid, raw_bbox_full, now_ts)
        else:
            # NO JOINT IN ROI
            if self.active_track is not None and self.active_track.tracking_status != "EXITED":
                self.active_track.consecutive_lost_frames += 1
                elapsed_lost = now_ts - self.active_track.last_seen_time
                if elapsed_lost < self.track_lost_timeout:
                    self.active_track.tracking_status = "TEMPORARILY_LOST"
                    self.active_track.total_frames_lost += 1
                else:
                    self.active_track.tracking_status = "EXITED"
                    self._close_joint_track(self.active_track)
                    self.active_track = None

        if self.active_track is None:
            return {
                "joint_id": "NONE",
                "current_label": "NO JOINT IN ROI",
                "p_healthy": 0.0,
                "p_damage": 0.0,
                "confidence_margin": 0.0,
                "sharpness": 0.0,
                "valid_frame": False,
                "valid_frame_count": 0,
                "rejected_blurry_count": self.rejected_blurry_count,
                "final_label": "UNCERTAIN",
                "final_confidence": 0.0,
                "avg_p_healthy": 0.0,
                "avg_p_damage": 0.0,
                "tracking_status": "SEARCHING",
                "inspection_state": "PASSED",
                "is_early_finalized": False,
                "total_frames_seen": 0,
                "total_frames_lost": 0,
                "smoothed_bbox": None,
                "timestamp": timestamp,
                "reason_for_final_change": "no-active-joint",
            }

        track = self.active_track

        # 2. UPDATE INSPECTION STATE & ZONE HANDLING WITH DEBOUNCE
        if in_zone:
            track.consecutive_outside_frames = 0
            track.is_inside_zone = True
            if track.inspection_state == "APPROACHING":
                track.inspection_state = "INSPECTING"
        else:
            if track.is_inside_zone:
                track.consecutive_outside_frames += 1
                if track.consecutive_outside_frames >= self.debounce_outside_frames:
                    track.is_inside_zone = False
                    if track.inspection_state in ("INSPECTING", "CONFIRMED"):
                        track.inspection_state = "PASSED"
                        self._finalize_track_verdict(track)
                        reason = "zone-exit-debounced"
            else:
                if track.inspection_state == "CONFIRMED":
                    track.inspection_state = "PASSED"
                    self._finalize_track_verdict(track)
                elif track.inspection_state != "PASSED":
                    track.inspection_state = "APPROACHING"

        if not joint_detected or not track.is_inside_zone:
            current_lbl = "NOT VISIBLE" if not joint_detected else "OUTSIDE_ZONE"
            visible_bbox = track.smoothed_bbox if (joint_detected or track.consecutive_lost_frames <= self.lost_bbox_grace_frames) else None
            return {
                "joint_id": track.joint_id,
                "current_label": current_lbl,
                "p_healthy": 0.0,
                "p_damage": 0.0,
                "confidence_margin": 0.0,
                "sharpness": 0.0,
                "valid_frame": False,
                "valid_frame_count": len(track.accumulated_samples),
                "rejected_blurry_count": self.rejected_blurry_count,
                "final_label": track.confirmed_label,
                "final_confidence": track.confirmed_confidence,
                "avg_p_healthy": float(np.mean([s[1] for s in track.accumulated_samples])) if track.accumulated_samples else 0.0,
                "avg_p_damage": float(np.mean([s[2] for s in track.accumulated_samples])) if track.accumulated_samples else 0.0,
                "tracking_status": track.tracking_status,
                "inspection_state": track.inspection_state,
                "is_early_finalized": track.is_early_finalized,
                "total_frames_seen": track.total_frames_seen,
                "total_frames_lost": track.total_frames_lost,
                "consecutive_lost_frames": track.consecutive_lost_frames,
                "smoothed_bbox": visible_bbox,
                "timestamp": timestamp,
                "reason_for_final_change": reason if reason != "no-change" else ("temporarily-lost" if not joint_detected else "outside-zone"),
                "finalized_via": getattr(track, "finalized_via", "none"),
            }

        # 3. EXTRACT STABILIZED CROP USING SMOOTHED BBOX (WITH ASPECT-RATIO CLAMPING)
        sx, sy, sw, sh = track.smoothed_bbox
        fh, fw = frame.shape[:2]

        # Clamp/correct bounding box to nearest valid aspect ratio to prevent entry/skew stalls
        # Conveyor joint crops should ideally have h/w in [0.20, 2.5]
        if sw > 0 and sh > 0:
            ar = float(sh) / float(sw)
            if ar < 0.20:
                # Too flat / vertically thin: expand height symmetrically with belt context
                target_h = min(fh, max(sh, int(sw * 0.28)))
                diff_h = target_h - sh
                sy = max(0, sy - diff_h // 2)
                sh = min(fh - sy, target_h)
            elif ar > 3.0:
                # Too narrow / horizontally thin (entering edge): expand width symmetrically
                target_w = min(fw, max(sw, int(sh / 2.0)))
                diff_w = target_w - sw
                sx = max(0, sx - diff_w // 2)
                sw = min(fw - sx, target_w)

        sx = max(0, min(fw - 10, sx))
        sy = max(0, min(fh - 10, sy))
        sw = max(10, min(fw - sx, sw))
        sh = max(10, min(fh - sy, sh))

        joint_crop = frame[sy:sy+sh, sx:sx+sw].copy()

        # Quality Check
        sharpness = calculate_sharpness(joint_crop)
        if no_quality_check:
            is_valid, quality_reason = True, "OK"
        else:
            is_approaching = (track.inspection_state == "APPROACHING")
            is_valid, quality_reason, sharpness = check_crop_quality(
                joint_crop, self.min_sharpness, is_approaching=is_approaching
            )

        if not is_valid:
            if "BLURRY" in quality_reason:
                self.rejected_blurry_count += 1
            current_label = quality_reason
            p_healthy = 0.0
            p_damage = 0.0
            margin = 0.0
            valid_frame = False
            self.diag_uncertain_count += 1
        else:
            # YOLO Classification
            yolo_res = yolo_model(joint_crop, verbose=False)[0]
            p_healthy = 0.0
            p_damage = 0.0
            for idx, name in yolo_res.names.items():
                c_prob = float(yolo_res.probs.data[idx])
                n_str = str(name).strip().lower()
                if n_str == "healthy":
                    p_healthy = c_prob
                elif n_str == "damage":
                    p_damage = c_prob

            margin = abs(p_healthy - p_damage)
            current_label = "DAMAGE" if p_damage >= p_healthy else "HEALTHY"

            if current_label == "DAMAGE":
                self.diag_damage_count += 1
                self.diag_damage_confs.append(p_damage)
            else:
                self.diag_healthy_count += 1
                self.diag_healthy_confs.append(p_healthy)

            valid_frame = True
            # Accumulate all valid frames across track lifetime (never cap at target_frames!)
            track.accumulated_samples.append(
                (current_label, p_healthy, p_damage, sharpness, timestamp)
            )

            # Save diagnostic crop
            if self.save_debug_crops:
                self._save_track_crop(track.joint_id, joint_crop, current_label, p_healthy, p_damage, sharpness)
                track.crop_history.append((joint_crop.copy(), current_label, p_healthy, p_damage))

        # 4. EVALUATE TEMPORAL DECISION, ROLLING EARLY-FINALIZATION & HYSTERESIS
        n_samples = len(track.accumulated_samples)
        avg_h = float(np.mean([s[1] for s in track.accumulated_samples])) if track.accumulated_samples else 0.0
        avg_d = float(np.mean([s[2] for s in track.accumulated_samples])) if track.accumulated_samples else 0.0
        damage_count = sum(1 for s in track.accumulated_samples if s[0] == "DAMAGE")
        healthy_count = sum(1 for s in track.accumulated_samples if s[0] == "HEALTHY")

        # Sliding window evaluation (last 5-8 valid frames)
        # Allows quick recovery from noisy/ambiguous starts once the joint is clearly visible
        recent_window_size = min(8, max(5, self.target_frames))
        recent_samples = track.accumulated_samples[-recent_window_size:] if n_samples >= 5 else []
        recent_avg_d = float(np.mean([s[2] for s in recent_samples])) if recent_samples else 0.0
        recent_damage_count = sum(1 for s in recent_samples if s[0] == "DAMAGE")
        recent_healthy_count = sum(1 for s in recent_samples if s[0] == "HEALTHY")

        # If already early-finalized as DAMAGE, maintain locked verdict
        if track.is_early_finalized and track.confirmed_label == "DAMAGE":
            track.confirmed_confidence = max(avg_d, recent_avg_d)
            track.inspection_state = "CONFIRMED"
        elif n_samples >= min(3, self.target_frames):
            # Evaluate Early Finalization during INSPECTING
            # Path A: Full lifetime average crosses damage threshold
            is_confident_damage_lifetime = (
                (n_samples >= self.target_frames and damage_count >= 3 and avg_d >= self.damage_thresh) or
                (damage_count >= 4 and avg_d >= self.damage_thresh) or
                (damage_count >= 3 and avg_d >= 0.85 and healthy_count == 0)
            )

            # Path B: Recent sliding window (last 5-8 valid frames) crosses damage threshold
            # Resolves Issue 1: recovers quickly from noisy early frames without waiting for PASSED
            is_confident_damage_recent = (
                len(recent_samples) >= 5 and
                recent_damage_count >= 3 and
                recent_damage_count > recent_healthy_count and
                recent_avg_d >= self.damage_thresh
            )

            if is_confident_damage_lifetime:
                track.confirmed_label = "DAMAGE"
                track.confirmed_confidence = avg_d
                track.is_early_finalized = True
                track.finalized_at_state = "INSPECTING"
                track.finalized_via = "early-lock-lifetime-average"
                track.inspection_state = "CONFIRMED"
                reason = "early-finalized-damage-lifetime"
                print(f"[EARLY FINAL VERDICT] Joint {track.joint_id} early-finalized via lifetime average: DAMAGE ({avg_d*100:.1f}%)")
            elif is_confident_damage_recent:
                track.confirmed_label = "DAMAGE"
                track.confirmed_confidence = recent_avg_d
                track.is_early_finalized = True
                track.finalized_at_state = "INSPECTING"
                track.finalized_via = "early-lock-recent-window"
                track.inspection_state = "CONFIRMED"
                reason = "early-finalized-damage-recent-window"
                print(f"[EARLY FINAL VERDICT] Joint {track.joint_id} early-finalized via recent sliding window ({len(recent_samples)} frames): DAMAGE ({recent_avg_d*100:.1f}%)")
            else:
                # Normal rolling candidate evaluation
                if damage_count >= 3 and avg_d >= self.damage_thresh:
                    candidate = "DAMAGE"
                    candidate_conf = avg_d
                elif healthy_count >= 3 and avg_h >= self.healthy_thresh:
                    candidate = "HEALTHY"
                    candidate_conf = avg_h
                elif damage_count > healthy_count and avg_d >= self.damage_thresh:
                    candidate = "DAMAGE"
                    candidate_conf = avg_d
                elif healthy_count > damage_count and avg_h >= self.healthy_thresh:
                    candidate = "HEALTHY"
                    candidate_conf = avg_h
                elif avg_d > avg_h and avg_d >= self.damage_thresh:
                    candidate = "DAMAGE"
                    candidate_conf = avg_d
                elif avg_h > avg_d and avg_h >= self.healthy_thresh and damage_count == 0:
                    candidate = "HEALTHY"
                    candidate_conf = avg_h
                else:
                    candidate = "UNCERTAIN"
                    candidate_conf = max(avg_h, avg_d)

                # Apply Result Hysteresis
                old_label = track.confirmed_label
                if old_label in ("DAMAGE", "HEALTHY"):
                    # SAFETY OVERRIDE: DAMAGE candidate immediately overrides an early provisional HEALTHY label!
                    if candidate == "DAMAGE" and old_label == "HEALTHY":
                        track.confirmed_label = "DAMAGE"
                        track.confirmed_confidence = candidate_conf
                        track.inspection_state = "CONFIRMED"
                        track.consecutive_contradictory_count = 0
                        reason = "damage-overrides-early-healthy"
                    elif candidate not in ("UNCERTAIN", old_label):
                        track.consecutive_contradictory_count += 1
                        if track.consecutive_contradictory_count >= 3:
                            track.confirmed_label = candidate
                            track.confirmed_confidence = candidate_conf
                            track.inspection_state = "CONFIRMED"
                            track.consecutive_contradictory_count = 0
                            reason = "3-consecutive-contradictory-valid-frames"
                    else:
                        track.consecutive_contradictory_count = 0
                        if candidate == old_label:
                            track.confirmed_confidence = candidate_conf
                else:
                    # First confirmation
                    if candidate != "UNCERTAIN":
                        track.confirmed_label = candidate
                        track.confirmed_confidence = candidate_conf
                        track.inspection_state = "CONFIRMED"
                        reason = "temporal-majority"
                    else:
                        track.confirmed_label = "UNCERTAIN"
                        track.confirmed_confidence = candidate_conf

        # Consistency check: Ensure majority DAMAGE never has final_label == "HEALTHY"
        if damage_count > healthy_count and track.confirmed_label == "HEALTHY":
            track.confirmed_label = "DAMAGE"
            track.confirmed_confidence = avg_d
            track.inspection_state = "CONFIRMED"

        # Print debug trace upon target_frames completion
        if len(track.accumulated_samples) >= self.target_frames and not self.printed_debug_for_track.get(track.joint_id, False):
            self._print_5frame_debug_output(track)
            self.printed_debug_for_track[track.joint_id] = True

        visible_bbox = track.smoothed_bbox if (joint_detected or track.consecutive_lost_frames <= self.lost_bbox_grace_frames) else None
        res = {
            "joint_id": track.joint_id,
            "current_label": current_label,
            "p_healthy": p_healthy,
            "p_damage": p_damage,
            "confidence_margin": margin,
            "sharpness": sharpness,
            "valid_frame": valid_frame,
            "valid_frame_count": len(track.accumulated_samples),
            "rejected_blurry_count": self.rejected_blurry_count,
            "final_label": track.confirmed_label,
            "final_confidence": track.confirmed_confidence,
            "avg_p_healthy": avg_h,
            "avg_p_damage": avg_d,
            "tracking_status": track.tracking_status,
            "inspection_state": track.inspection_state,
            "is_early_finalized": track.is_early_finalized,
            "total_frames_seen": track.total_frames_seen,
            "total_frames_lost": track.total_frames_lost,
            "consecutive_lost_frames": track.consecutive_lost_frames,
            "smoothed_bbox": visible_bbox,
            "timestamp": timestamp,
            "reason_for_final_change": reason,
            "finalized_via": getattr(track, "finalized_via", "none"),
        }

        if self.jsonl_path:
            self.log_event(res)

        return res

    def _close_joint_track(self, track: JointTrack):
        """Finalizes track, caches in recent_tracks for re-identification, and generates contact sheet."""
        self._finalize_track_verdict(track)
        if track not in self.recent_tracks:
            self.recent_tracks.append(track)
            if len(self.recent_tracks) > 5:
                self.recent_tracks.pop(0)
        if self.save_debug_crops and track.crop_history:
            generate_contact_sheet(track.joint_id, self.debug_crops_dir, track.crop_history)

    def _save_track_crop(self, jid: str, crop: np.ndarray, label: str, p_h: float, p_d: float, sharpness: float):
        try:
            j_dir = os.path.join(self.debug_crops_dir, jid)
            os.makedirs(j_dir, exist_ok=True)
            cnt = len(os.listdir(j_dir)) + 1
            fn = f"frame_{cnt:03d}_{label}_h{p_h:.2f}_d{p_d:.2f}_s{sharpness:.0f}.jpg"
            cv2.imwrite(os.path.join(j_dir, fn), crop)
        except Exception:
            pass

    def _print_5frame_debug_output(self, track: JointTrack):
        print(f"\n[{track.joint_id}] Frame predictions:")
        for idx, sample in enumerate(track.accumulated_samples, start=1):
            pred_cls, p_h, p_d = sample[0], sample[1], sample[2]
            conf = p_d if pred_cls == "DAMAGE" else p_h
            print(f"F{idx}: {pred_cls} {conf:.2f}")

        damage_count = sum(1 for s in track.accumulated_samples if s[0] == "DAMAGE")
        healthy_count = sum(1 for s in track.accumulated_samples if s[0] == "HEALTHY")
        majority = "DAMAGE" if damage_count > healthy_count else ("HEALTHY" if healthy_count > damage_count else "TIE")

        # Consistency check before printing: majority DAMAGE must NEVER output FINAL: HEALTHY
        if majority == "DAMAGE" and track.confirmed_label == "HEALTHY":
            print(f"[CONSISTENCY WARNING] Majority is DAMAGE ({damage_count} vs {healthy_count}) but track was HEALTHY! Auto-correcting to DAMAGE.")
            track.confirmed_label = "DAMAGE"
            track.confirmed_confidence = float(np.mean([s[2] for s in track.accumulated_samples]))

        print(f"\nMajority: {majority}")
        print(f"Average confidence: {track.confirmed_confidence:.2f}")
        print(f"FINAL: {track.confirmed_label}\n")

    def log_event(self, res: Dict[str, Any]):
        try:
            log_dir = os.path.dirname(self.jsonl_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            log_entry = {
                "timestamp": res["timestamp"],
                "joint_id": res["joint_id"],
                "smoothed_bbox": res["smoothed_bbox"],
                "current_prediction": res["current_label"],
                "healthy_confidence": round(float(res["p_healthy"]), 4),
                "damage_confidence": round(float(res["p_damage"]), 4),
                "sharpness": round(float(res["sharpness"]), 2),
                "tracking_status": res["tracking_status"],
                "inspection_state": res["inspection_state"],
                "buffer_size": res["valid_frame_count"],
                "final_label": res["final_label"],
                "final_confidence": round(float(res["final_confidence"]), 4),
                "reason_for_final_change": res["reason_for_final_change"]
            }
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def print_diagnostic_summary(self):
        avg_h_conf = (sum(self.diag_healthy_confs) / len(self.diag_healthy_confs)) * 100.0 if self.diag_healthy_confs else 0.0
        avg_d_conf = (sum(self.diag_damage_confs) / len(self.diag_damage_confs)) * 100.0 if self.diag_damage_confs else 0.0
        print("\n" + "=" * 65)
        print("JOINTGUARD DIAGNOSTIC SUMMARY")
        print("=" * 65)
        print(f"Total Evaluated Frames:       {self.diag_total_frames}")
        print(f"Total Healthy Predictions:    {self.diag_healthy_count} (Avg Conf: {avg_h_conf:.1f}%)")
        print(f"Total Damage Predictions:     {self.diag_damage_count} (Avg Conf: {avg_d_conf:.1f}%)")
        print(f"Total Uncertain Predictions:  {self.diag_uncertain_count}")
        print(f"Total Blurry Rejected:        {self.rejected_blurry_count}")
        print("=" * 65 + "\n")



def main():
    args = parse_args()

    req_version = "v1" if args.v1 else ("v2" if args.v2 else ("v4" if args.v4 else args.model_version))
    try:
        model_file, model_tag = resolve_model_path(args.model, version=req_version)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    sources_to_try = [args.source]
    for alt in [0, 1]:
        if alt not in sources_to_try:
            sources_to_try.append(alt)

    cap = None
    active_source = args.source
    test_frame = None

    for src in sources_to_try:
        if sys.platform.startswith("win"):
            cap_ds = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if cap_ds and cap_ds.isOpened():
                cap_ds.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap_ds.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                ret_ds, frame_ds = cap_ds.read()
                if ret_ds and frame_ds is not None:
                    cap = cap_ds
                    test_frame = frame_ds
                    active_source = src
                    break
                cap_ds.release()

        cap_std = cv2.VideoCapture(src)
        if cap_std and cap_std.isOpened():
            cap_std.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap_std.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            ret_std, frame_std = cap_std.read()
            if ret_std and frame_std is not None:
                cap = cap_std
                test_frame = frame_std
                active_source = src
                break
            cap_std.release()

    if cap is None or test_frame is None:
        print(f"[ERROR] Could not open or capture from any camera source ({sources_to_try}).", file=sys.stderr)
        sys.exit(1)

    if active_source != args.source:
        print(f"[INFO] Camera source {args.source} unavailable. Automatically connected to camera source {active_source}.")
    args.source = active_source

    frame_h, frame_w = test_frame.shape[:2]

    try:
        rx1, ry1, rx2, ry2 = parse_roi_string(args.roi, frame_w, frame_h)
    except Exception as e:
        print(f"[ERROR] Invalid ROI format: {e}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    roi_w = rx2 - rx1
    roi_h = ry2 - ry1

    print("=" * 65)
    print("JOINTGUARD ROBUST WEBCAM INSPECTION STATE MACHINE")
    print("=" * 65)
    print(f"Camera Source:         {args.source}")
    print(f"Model:                 {model_file} [{model_tag.upper()}]")
    print(f"Damage Threshold:      {args.damage_thresh:.2f}")
    print(f"Healthy Threshold:     {args.healthy_thresh:.2f}")
    print(f"Confidence Margin:     {args.conf_margin:.2f}")
    print(f"Minimum Sharpness:     {args.min_sharpness:.1f} (Laplacian Variance)")
    print(f"Target Sharp Frames:   {args.target_frames} (Majority Voting)")
    print(f"Track Lost Timeout:    2.0s")
    print(f"Bbox Smooth Alpha:     0.3")
    print(f"Save Debug Crops:      {args.save_debug_crops}")
    print(f"JSONL Event Log:       {args.jsonl_log}")
    print("=" * 65)

    from ultralytics import YOLO
    model = YOLO(model_file)
    print("[INFO] YOLO model loaded successfully.")
    print("[KEYBOARD SHORTCUTS] 'h'=save healthy crop | 'd'=save damage crop | 'q'=quit")

    os.makedirs(COLLECTED_HEALTHY_DIR, exist_ok=True)
    os.makedirs(COLLECTED_DAMAGE_DIR, exist_ok=True)

    tracker = JointGuardStateEngine(
        target_frames=args.target_frames,
        min_sharpness=args.min_sharpness,
        damage_thresh=args.damage_thresh,
        healthy_thresh=args.healthy_thresh,
        conf_margin=args.conf_margin,
        track_lost_timeout=2.0,
        bbox_smooth_alpha=0.3,
        jsonl_path=args.jsonl_log,
        save_debug_crops=args.save_debug_crops,
        debug_crops_dir=DEBUG_CROPS_DIR,
    )

    fps_history = deque(maxlen=30)
    window_main = "JointGuard YOLO - Robust State Machine Inspection"
    window_crop = "JointGuard YOLO Input (Smoothed Crop)"

    cv2.namedWindow(window_main, cv2.WINDOW_AUTOSIZE)
    if args.show_crop:
        cv2.namedWindow(window_crop, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            fixed_roi_crop = frame[ry1:ry2, rx1:rx2].copy()

            joint_detected = False
            actual_bbox_full = None
            in_inner_zone = False

            if args.direct_roi:
                joint_detected = True
                actual_bbox_full = (rx1, ry1, roi_w, roi_h)
                in_inner_zone = True
            else:
                rel_bbox = find_joint_in_roi(
                    fixed_roi_crop,
                    min_area=args.min_joint_area,
                    min_width=args.min_joint_width,
                )

                if rel_bbox is not None:
                    jx, jy, jw, jh = rel_bbox
                    fx, fy = rx1 + jx, ry1 + jy
                    actual_bbox_full = (fx, fy, jw, jh)
                    joint_detected = True
                    is_currently_inside = (
                        tracker.active_track is not None and
                        tracker.active_track.is_inside_zone
                    )
                    in_inner_zone = is_in_inner_inspection_zone(
                        rel_bbox, roi_w, roi_h, is_currently_inside=is_currently_inside
                    )

            # Process frame through JointGuardStateEngine
            eval_res = tracker.process_frame(
                frame=frame,
                joint_detected=joint_detected,
                raw_bbox_full=actual_bbox_full,
                in_zone=in_inner_zone,
                yolo_model=model,
                no_quality_check=args.no_quality_check
            )

            # Secondary Crop Window
            if args.show_crop:
                if eval_res["smoothed_bbox"] is not None:
                    sx, sy, sw, sh = eval_res["smoothed_bbox"]
                    fh, fw = frame.shape[:2]
                    sx = max(0, min(fw - 10, sx))
                    sy = max(0, min(fh - 10, sy))
                    sw = max(10, min(fw - sx, sw))
                    sh = max(10, min(fh - sy, sh))
                    crop_view = frame[sy:sy+sh, sx:sx+sw].copy()
                    cv2.putText(
                        crop_view,
                        f"YOLO: {eval_res['current_label']}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0) if eval_res['current_label'] == "HEALTHY" else ((0, 0, 255) if eval_res['current_label'] == "DAMAGE" else (0, 200, 255)),
                        2,
                    )
                else:
                    crop_view = fixed_roi_crop.copy()
                    cv2.putText(
                        crop_view,
                        "SEARCHING...",
                        (20, crop_view.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )
                cv2.imshow(window_crop, crop_view)

            # RENDER HUD DEBUG OVERLAY ON MAIN DISPLAY
            t_now = time.time()
            fps_history.append(1.0 / max(1e-5, t_now - t_start))
            current_fps = float(np.mean(fps_history))

            display_frame = frame.copy()

            # Outer ROI rectangle
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 180, 0), 1)
            cv2.putText(display_frame, "SEARCH ROI", (rx1 + 5, ry1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

            # Inner Inspection Zone rectangle
            iz_x1 = rx1 + int(roi_w * 0.08)
            iz_y1 = ry1 + int(roi_h * 0.10)
            iz_x2 = rx1 + int(roi_w * 0.92)
            iz_y2 = ry1 + int(roi_h * 0.90)
            cv2.rectangle(display_frame, (iz_x1, iz_y1), (iz_x2, iz_y2), (0, 255, 255), 1)
            cv2.putText(display_frame, "INNER INSPECTION ZONE", (iz_x1 + 5, iz_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Draw Smoothed Bounding Box on localized joint (only when detected or within grace period)
            if eval_res["smoothed_bbox"] is not None and eval_res.get("consecutive_lost_frames", 0) <= 2:
                bx, by, bw, bh = eval_res["smoothed_bbox"]
                f_lbl = eval_res["final_label"]
                color = (0, 220, 0) if f_lbl == "HEALTHY" else ((0, 0, 230) if f_lbl == "DAMAGE" else (0, 200, 255))

                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), color, 2)

                badge_txt = f"{eval_res['joint_id']}: {f_lbl}"
                if eval_res["final_confidence"] > 0:
                    badge_txt += f" ({eval_res['final_confidence']*100:.1f}%)"
                if eval_res["tracking_status"] == "TEMPORARILY_LOST":
                    badge_txt += " [LOST]"

                cv2.rectangle(display_frame, (bx, max(0, by - 22)), (bx + 260, max(22, by)), (20, 20, 20), -1)
                cv2.putText(display_frame, badge_txt, (bx + 5, max(14, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # TOP DIAGNOSTIC HUD BOX (REQUIREMENT 12)
            hud_bg_w = 480
            hud_bg_h = 170
            cv2.rectangle(display_frame, (10, 10), (10 + hud_bg_w, 10 + hud_bg_h), (15, 23, 42), -1)
            cv2.rectangle(display_frame, (10, 10), (10 + hud_bg_w, 10 + hud_bg_h), (51, 65, 85), 1)

            # Line 1: Joint ID, State & FPS
            cv2.putText(display_frame, f"JOINT: {eval_res['joint_id']}  |  STATE: {eval_res['inspection_state']}  |  FPS: {current_fps:.1f}",
                        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 2)

            # Line 2: CURRENT Prediction
            hp_pct = eval_res['p_healthy'] * 100.0
            dp_pct = eval_res['p_damage'] * 100.0
            cv2.putText(display_frame, f"CURRENT: {eval_res['current_label']} (H: {hp_pct:.1f}% | D: {dp_pct:.1f}%)",
                        (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1)

            # Line 3: FINAL Confirmed Result & Status
            final_lbl = eval_res['final_label']
            final_conf_pct = eval_res['final_confidence'] * 100.0
            final_col = (74, 222, 128) if final_lbl == "HEALTHY" else ((248, 113, 113) if final_lbl == "DAMAGE" else (251, 191, 36))
            status_txt = "VISUALIZED" if eval_res['tracking_status'] == "VISUALIZED" else ("TEMPORARILY NOT VISIBLE" if eval_res['tracking_status'] == "TEMPORARILY_LOST" else "SEARCHING")
            cv2.putText(display_frame, f"FINAL: {final_lbl} ({final_conf_pct:.1f}%)  |  STATUS: {status_txt}",
                        (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, final_col, 2)

            # Line 4: Track Stats (Seen / Lost)
            cv2.putText(display_frame, f"TRACK: {eval_res['joint_id']}  |  Seen: {eval_res['total_frames_seen']} frames  |  Lost: {eval_res['total_frames_lost']} frames",
                        (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (251, 191, 36), 1)

            # Line 5: Temporal Buffer & Average Confidences
            avg_h_pct = eval_res['avg_p_healthy'] * 100.0
            avg_d_pct = eval_res['avg_p_damage'] * 100.0
            cv2.putText(display_frame, f"TEMPORAL: {eval_res['valid_frame_count']}/{args.target_frames} valid  |  Avg D: {avg_d_pct:.1f}%  |  Avg H: {avg_h_pct:.1f}%",
                        (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (203, 213, 225), 1)

            # Line 6: Sharpness Score
            sharp_val = eval_res['sharpness']
            sharp_col = (74, 222, 128) if sharp_val >= args.min_sharpness else (248, 113, 113)
            cv2.putText(display_frame, f"SHARPNESS: {sharp_val:.1f} (Min: {args.min_sharpness:.1f})",
                        (20, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.46, sharp_col, 1)

            cv2.imshow(window_main, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Quitting webcam inspection...")
                break

    finally:
        tracker.print_diagnostic_summary()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
