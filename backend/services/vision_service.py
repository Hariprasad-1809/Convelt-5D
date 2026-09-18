"""
JointGuard Vision Service (FastAPI Background Integration)

Reuses the OpenCV specular joint localization & YOLOv8 classification pipeline
from check_yolo/webcam.py for continuous live video joint monitoring.

Pipeline Features:
  - Motion Blur Rejection (Laplacian Variance >= min_sharpness)
  - Asymmetric Confidence Thresholding (DAMAGE_THRESH: 0.70, HEALTHY_THRESH: 0.70)
  - Confidence Margin Gating (|p_damage - p_healthy| >= 0.15 required)
  - Inner Inspection Zone Centering (prevents entry/exit border cutoffs)
  - Multi-Frame Accumulation & Majority Voting (target 5 sharp frames)
  - Quality Gate: Returns WAITING if fewer than required sharp frames (3/5) are collected
  - Diagnostic JSONL Event Logger (data/joint_events.jsonl)
  - Live HUD Overlays & MJPEG Streaming Endpoint (/api/v1/vision/stream)

Calculates vision_score:
  - HEALTHY: vision_score = round(final_confidence * 100, 2)
  - DAMAGE:  vision_score = round((1 - final_confidence) * 100, 2)  (Lower score = worse condition!)
  - UNCERTAIN / WAITING / LOW_QUALITY / NO_JOINT: vision_score = None
"""

import asyncio
from collections import deque
import datetime
import json
import os
import sys
import threading
import time
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np

from backend.config import settings

# Attempt importing Ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_JSONL_PATH = os.path.join(PROJECT_ROOT, "data", "joint_events.jsonl")


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


def is_in_inner_inspection_zone(rel_bbox: Tuple[int, int, int, int], roi_w: int, roi_h: int) -> bool:
    """Checks if joint centroid is inside central 70% inner inspection zone."""
    jx, jy, jw, jh = rel_bbox
    cx = jx + jw / 2.0
    cy = jy + jh / 2.0

    xmin = roi_w * 0.08
    xmax = roi_w * 0.92
    ymin = roi_h * 0.10
    ymax = roi_h * 0.90

    return (xmin <= cx <= xmax) and (ymin <= cy <= ymax)


def calculate_sharpness(crop: np.ndarray) -> float:
    """Calculates Laplacian variance as a measure of focus/sharpness."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_crop_quality(crop: np.ndarray, min_sharpness: float = 100.0) -> Tuple[bool, str, float]:
    """Validates crop quality before sending to YOLO."""
    h, w = crop.shape[:2]
    if h < 25 or w < 25:
        return False, "TOO SMALL", 0.0

    aspect_ratio = float(h) / float(w)
    if aspect_ratio > 3.2 or aspect_ratio < 0.15:
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

    def update_position(self, raw_bbox: Tuple[int, int, int, int], timestamp: float, alpha: float = 0.3):
        self.raw_bbox = raw_bbox
        rx, ry, rw, rh = raw_bbox
        sx, sy, sw, sh = self.smoothed_bbox

        nx = int(alpha * rx + (1.0 - alpha) * sx)
        ny = int(alpha * ry + (1.0 - alpha) * sy)
        nw = int(alpha * rw + (1.0 - alpha) * sw)
        nh = int(alpha * rh + (1.0 - alpha) * sh)

        self.smoothed_bbox = (nx, ny, nw, nh)
        self.last_centroid = (nx + nw / 2.0, ny + nh / 2.0)
        self.last_seen_time = timestamp
        self.total_frames_seen += 1
        self.total_frames_lost = 0
        self.tracking_status = "VISUALIZED"


class JointGuardStateEngine:
    """
    Robust Joint Inspection State Machine & Multi-Joint Track Manager.
    Features:
    - Persistent Joint Tracking & Association (J01, J02...)
    - EMA Bounding Box Smoothing (alpha = 0.3)
    - Temporal Hold Timeout during dropouts (TRACK_LOST_TIMEOUT = 1.5s)
    - Hysteresis (3 consecutive valid contradictory predictions required before label flip)
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
        track_lost_timeout: float = 1.5,
        bbox_smooth_alpha: float = 0.3,
        jsonl_path: Optional[str] = None,
        save_debug_crops: bool = False,
        debug_crops_dir: Optional[str] = None,
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
        self.debug_crops_dir = debug_crops_dir or os.path.join(PROJECT_ROOT, "check_yolo", "debug_crops")

        self.joint_counter = 1
        self.active_track: Optional[JointTrack] = None
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

        # 1. TRACK ASSOCIATION OR CREATION
        if joint_detected and raw_bbox_full is not None:
            jx, jy, jw, jh = raw_bbox_full
            det_centroid = (jx + jw / 2.0, jy + jh / 2.0)

            if self.active_track is not None and self.active_track.tracking_status != "EXITED":
                dist = np.sqrt(
                    (det_centroid[0] - self.active_track.last_centroid[0])**2 +
                    (det_centroid[1] - self.active_track.last_centroid[1])**2
                )
                if dist <= 160.0:
                    # Associated with current active joint track
                    self.active_track.update_position(raw_bbox_full, now_ts, alpha=self.bbox_smooth_alpha)
                else:
                    # Centroid distance too large -> Expire old joint, create new
                    self._close_joint_track(self.active_track)
                    new_jid = self._get_next_joint_id()
                    self.active_track = JointTrack(new_jid, raw_bbox_full, now_ts)
            else:
                new_jid = self._get_next_joint_id()
                self.active_track = JointTrack(new_jid, raw_bbox_full, now_ts)
        else:
            # NO JOINT IN ROI
            if self.active_track is not None and self.active_track.tracking_status != "EXITED":
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
                "total_frames_seen": 0,
                "total_frames_lost": 0,
                "smoothed_bbox": None,
                "timestamp": timestamp,
                "reason_for_final_change": "no-active-joint",
            }

        track = self.active_track

        # 2. UPDATE INSPECTION STATE & ZONE HANDLING
        if not in_zone:
            if track.inspection_state == "CONFIRMED":
                track.inspection_state = "PASSED"
            elif track.inspection_state != "PASSED":
                track.inspection_state = "APPROACHING"

            return {
                "joint_id": track.joint_id,
                "current_label": "OUTSIDE_ZONE",
                "p_healthy": 0.0,
                "p_damage": 0.0,
                "confidence_margin": 0.0,
                "sharpness": 0.0,
                "valid_frame": False,
                "valid_frame_count": len(track.accumulated_samples),
                "rejected_blurry_count": self.rejected_blurry_count,
                "final_label": track.confirmed_label,
                "final_confidence": track.confirmed_confidence,
                "avg_p_healthy": 0.0,
                "avg_p_damage": 0.0,
                "tracking_status": track.tracking_status,
                "inspection_state": track.inspection_state,
                "total_frames_seen": track.total_frames_seen,
                "total_frames_lost": track.total_frames_lost,
                "smoothed_bbox": track.smoothed_bbox,
                "timestamp": timestamp,
                "reason_for_final_change": "outside-zone",
            }

        # If inside inspection zone
        if track.inspection_state == "APPROACHING":
            track.inspection_state = "INSPECTING"

        # 3. EXTRACT STABILIZED CROP USING SMOOTHED BBOX
        sx, sy, sw, sh = track.smoothed_bbox
        fh, fw = frame.shape[:2]
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
            is_valid, quality_reason, sharpness = check_crop_quality(joint_crop, self.min_sharpness)

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
            if len(track.accumulated_samples) < self.target_frames:
                track.accumulated_samples.append(
                    (current_label, p_healthy, p_damage, sharpness, timestamp)
                )

            # Save diagnostic crop
            if self.save_debug_crops:
                self._save_track_crop(track.joint_id, joint_crop, current_label, p_healthy, p_damage, sharpness)
                track.crop_history.append((joint_crop.copy(), current_label, p_healthy, p_damage))

        # 4. EVALUATE TEMPORAL DECISION & HYSTERESIS
        avg_h = float(np.mean([s[1] for s in track.accumulated_samples])) if track.accumulated_samples else 0.0
        avg_d = float(np.mean([s[2] for s in track.accumulated_samples])) if track.accumulated_samples else 0.0

        if len(track.accumulated_samples) >= min(3, self.target_frames):
            damage_count = sum(1 for s in track.accumulated_samples if s[0] == "DAMAGE")
            healthy_count = sum(1 for s in track.accumulated_samples if s[0] == "HEALTHY")

            if damage_count >= 3 and avg_d >= self.damage_thresh:
                candidate = "DAMAGE"
                candidate_conf = avg_d
            elif healthy_count >= 3 and avg_h >= self.healthy_thresh:
                candidate = "HEALTHY"
                candidate_conf = avg_h
            else:
                candidate = "UNCERTAIN"
                candidate_conf = max(avg_h, avg_d)

            # Apply Result Hysteresis
            old_label = track.confirmed_label
            if old_label in ("DAMAGE", "HEALTHY"):
                if candidate not in ("UNCERTAIN", old_label):
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
                    reason = "5-frame-majority"
                else:
                    track.confirmed_label = "UNCERTAIN"
                    track.confirmed_confidence = candidate_conf

        # Print debug trace upon 5-frame completion
        if len(track.accumulated_samples) == self.target_frames and not self.printed_debug_for_track.get(track.joint_id, False):
            self._print_5frame_debug_output(track)
            self.printed_debug_for_track[track.joint_id] = True

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
            "total_frames_seen": track.total_frames_seen,
            "total_frames_lost": track.total_frames_lost,
            "smoothed_bbox": track.smoothed_bbox,
            "timestamp": timestamp,
            "reason_for_final_change": reason,
        }

        if self.jsonl_path:
            self.log_event(res)

        return res

    def _close_joint_track(self, track: JointTrack):
        """Generates contact sheet when a joint exits."""
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


# Backward Compatibility Alias
JointEventTracker = JointGuardStateEngine


class VisionService:
    def __init__(self):
        self.camera_index = settings.VISION_CAMERA_INDEX
        self.model_path = settings.VISION_MODEL_PATH
        self.damage_thresh = getattr(settings, "VISION_DAMAGE_THRESH", 0.70)
        self.healthy_thresh = getattr(settings, "VISION_HEALTHY_THRESH", 0.70)
        self.conf_margin = getattr(settings, "VISION_CONF_MARGIN", 0.15)
        self.min_sharpness = getattr(settings, "VISION_MIN_SHARPNESS", 100.0)
        self.target_frames = getattr(settings, "VISION_TARGET_FRAMES", 5)

        self.active_joint_id = "J01"
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.model = None
        self.model_loaded = False
        self.camera_status = "DISCONNECTED"

        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_result: Dict[str, Any] = {
            "type": "vision_update",
            "joint_id": "J01",
            "label": "WAITING",
            "confidence": 0.0,
            "vision_score": None,
            "camera_status": "DISCONNECTED",
            "model_status": "NOT_LOADED",
            "valid_frame_count": 0,
            "rejected_blurry_count": 0,
            "sharpness": 0.0,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        self.tracker = JointGuardStateEngine(
            target_frames=self.target_frames,
            min_sharpness=self.min_sharpness,
            damage_thresh=self.damage_thresh,
            healthy_thresh=self.healthy_thresh,
            conf_margin=self.conf_margin,
            track_lost_timeout=1.5,
            bbox_smooth_alpha=0.3,
            jsonl_path=DEFAULT_JSONL_PATH
        )
        self._lock = threading.Lock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None


    def load_model(self) -> bool:
        if not YOLO_AVAILABLE:
            print("[VISION SERVICE ERROR] Ultralytics package not installed.")
            return False

        if not os.path.isfile(self.model_path):
            print(f"[VISION SERVICE WARN] YOLO model weights not found at {self.model_path}")
            return False

        try:
            self.model = YOLO(self.model_path)
            self.model_loaded = True
            print(f"[VISION SERVICE] YOLO model loaded successfully: {self.model_path}")
            return True
        except Exception as e:
            print(f"[VISION SERVICE ERROR] Failed to load YOLO model: {e}")
            self.model_loaded = False
            return False

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        if self.is_running:
            return

        self._event_loop = loop
        self.is_running = True
        self.load_model()

        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="VisionServiceThread")
        self.thread.start()
        print(f"[VISION SERVICE] Started calibrated vision thread on camera index {self.camera_index}")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.camera_status = "DISCONNECTED"
        print("[VISION SERVICE] Stopped vision thread and released camera.")

    def _open_camera(self) -> bool:
        target_idx = self.camera_index
        print(f"[VISION SERVICE] Attempting to open webcam at target source index {target_idx}...")
        
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        
        for backend in backends:
            cap = cv2.VideoCapture(target_idx, backend)
            if cap and cap.isOpened():
                # Warm up USB webcam to allow driver initialization
                for _ in range(5):
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
                        self.camera_status = "CONNECTED"
                        print(f"[VISION SERVICE] Successfully opened target webcam source index {target_idx}")
                        return True
                    time.sleep(0.05)
                cap.release()
                
        self.camera_status = "DISCONNECTED"
        print(f"[VISION SERVICE ERROR] Failed to open target webcam at source index {target_idx}. Built-in camera (index 0) fallback disabled.")
        return False

    def _worker_loop(self):
        consecutive_failures = 0

        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open_camera():
                    with self._lock:
                        self.camera_status = "DISCONNECTED"
                        self.latest_result["camera_status"] = "DISCONNECTED"
                    time.sleep(3.0)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 5:
                    if self.cap:
                        self.cap.release()
                    with self._lock:
                        self.camera_status = "DISCONNECTED"
                        self.latest_result["camera_status"] = "DISCONNECTED"
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            self._process_frame(frame)
            time.sleep(0.03)

    def _process_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(0.15 * w), int(0.20 * h), int(0.85 * w), int(0.80 * h)
        roi_w, roi_h = x2 - x1, y2 - y1
        roi_img = frame[y1:y2, x1:x2]

        joint_crop_box = find_joint_in_roi(roi_img)

        raw_bbox_full = None
        in_inner_zone = False
        joint_detected = False

        if joint_crop_box is not None:
            bx, by, bw, bh = joint_crop_box
            abs_bx1, abs_by1 = x1 + bx, y1 + by
            raw_bbox_full = (abs_bx1, abs_by1, bw, bh)
            joint_detected = True
            in_inner_zone = is_in_inner_inspection_zone(joint_crop_box, roi_w, roi_h)

        if self.model_loaded and self.model is not None:
            eval_res = self.tracker.process_frame(
                frame=frame,
                joint_detected=joint_detected,
                raw_bbox_full=raw_bbox_full,
                in_zone=in_inner_zone,
                yolo_model=self.model
            )
        else:
            eval_res = {
                "joint_id": "NONE",
                "current_label": "MODEL_NOT_LOADED",
                "p_healthy": 0.0,
                "p_damage": 0.0,
                "confidence_margin": 0.0,
                "sharpness": 0.0,
                "valid_frame": False,
                "valid_frame_count": 0,
                "rejected_blurry_count": self.tracker.rejected_blurry_count,
                "final_label": "UNCERTAIN",
                "final_confidence": 0.0,
                "avg_p_healthy": 0.0,
                "avg_p_damage": 0.0,
                "tracking_status": "DISCONNECTED",
                "inspection_state": "PASSED",
                "total_frames_seen": 0,
                "total_frames_lost": 0,
                "smoothed_bbox": None,
                "timestamp": datetime.datetime.now().isoformat(),
                "reason_for_final_change": "model-not-loaded",
            }

        final_label = eval_res["final_label"]
        final_conf = eval_res["final_confidence"]
        vision_score: Optional[float] = None

        if final_label == "HEALTHY":
            vision_score = round(final_conf * 100.0, 2)
        elif final_label == "DAMAGE":
            vision_score = round((1.0 - final_conf) * 100.0, 2)

        res = {
            "type": "vision_update",
            "joint_id": eval_res["joint_id"],
            "label": final_label,
            "confidence": round(final_conf, 4),
            "vision_score": vision_score,
            "current_label": eval_res["current_label"],
            "p_healthy": round(eval_res["p_healthy"], 4),
            "p_damage": round(eval_res["p_damage"], 4),
            "sharpness": round(eval_res["sharpness"], 2),
            "valid_frame_count": eval_res["valid_frame_count"],
            "rejected_blurry_count": eval_res["rejected_blurry_count"],
            "tracking_status": eval_res["tracking_status"],
            "inspection_state": eval_res["inspection_state"],
            "camera_status": "CONNECTED",
            "model_status": "LOADED" if self.model_loaded else "NOT_LOADED",
            "timestamp": eval_res["timestamp"],
        }

        # Trigger alert on high-confidence damage detection
        if final_label == "DAMAGE" and final_conf >= self.damage_thresh:
            try:
                from backend.database.db import SessionLocal
                from backend.services.alert_service import trigger_vision_damage_alert
                db = SessionLocal()
                try:
                    trigger_vision_damage_alert(db, eval_res["joint_id"], final_conf, vision_score or 8.0)
                finally:
                    db.close()
            except Exception:
                pass

        # HUD Overlays on video frame for MJPEG stream
        vis_frame = frame.copy()
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 180, 0), 1)

        iz_x1 = x1 + int(roi_w * 0.08)
        iz_y1 = y1 + int(roi_h * 0.10)
        iz_x2 = x1 + int(roi_w * 0.92)
        iz_y2 = y1 + int(roi_h * 0.90)
        cv2.rectangle(vis_frame, (iz_x1, iz_y1), (iz_x2, iz_y2), (0, 255, 255), 1)

        if eval_res["smoothed_bbox"] is not None:
            bx, by, bw, bh = eval_res["smoothed_bbox"]
            color = (0, 255, 0) if final_label == "HEALTHY" else ((0, 0, 255) if final_label == "DAMAGE" else (0, 165, 255))
            cv2.rectangle(vis_frame, (bx, by), (bx + bw, by + bh), color, 2)
            txt = f"{eval_res['joint_id']}: {final_label} {final_conf*100:.1f}%" if final_conf > 0 else f"{eval_res['joint_id']}: {final_label}"
            cv2.putText(vis_frame, txt, (bx, max(15, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Top Diagnostic HUD Box
        hud_w, hud_h = 440, 140
        cv2.rectangle(vis_frame, (10, 10), (10 + hud_w, 10 + hud_h), (15, 23, 42), -1)
        cv2.rectangle(vis_frame, (10, 10), (10 + hud_w, 10 + hud_h), (51, 65, 85), 1)

        cv2.putText(vis_frame, f"JOINT: {eval_res['joint_id']} | Final: {final_label}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (248, 250, 252), 2)
        hp_pct = eval_res['p_healthy'] * 100.0
        dp_pct = eval_res['p_damage'] * 100.0
        cv2.putText(vis_frame, f"Current: {eval_res['current_label']} (H: {hp_pct:.1f}% | D: {dp_pct:.1f}%)", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1)
        sharp_col = (74, 222, 128) if eval_res['sharpness'] >= self.min_sharpness else (248, 113, 113)
        cv2.putText(vis_frame, f"Sharpness: {eval_res['sharpness']:.1f} (Min: {self.min_sharpness:.1f})", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sharp_col, 1)
        cv2.putText(vis_frame, f"Valid: {eval_res['valid_frame_count']}/{self.target_frames} | Rejected: {eval_res['rejected_blurry_count']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (251, 191, 36), 1)

        _, jpeg_buf = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = jpeg_buf.tobytes()

        with self._lock:
            self.latest_result = res
            self.latest_frame_jpeg = jpeg_bytes
            self.camera_status = "CONNECTED"

        self._notify_subscribers(res)

    def _notify_subscribers(self, data: Dict[str, Any]):
        if self._event_loop and self._event_loop.is_running():
            try:
                from backend.websocket.telemetry_ws import telemetry_ws
                asyncio.run_coroutine_threadsafe(
                    telemetry_ws.broadcast(data),
                    self._event_loop
                )
            except Exception:
                pass

    def get_latest_result(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.latest_result)

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_frame_jpeg


vision_service = VisionService()
