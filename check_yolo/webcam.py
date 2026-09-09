#!/usr/bin/env python3
"""
JointGuard YOLO Classification - Live Webcam Inspection Program
Two-Stage Detection Pipeline:
    Fixed ROI -> OpenCV searches for joint -> Actual joint bounding box -> YOLO -> HEALTHY / DAMAGE

Features:
- Default camera source: 1 (configurable via --source)
- Fixed Capture ROI: The user-defined inspection window on the conveyor belt
- Classical OpenCV Joint Finder: Detects metallic fasteners/patches within the Fixed ROI
- Actual Joint Bounding Box: Isolates the localized joint region
- YOLOv8n-cls: Classifies the actual joint crop as HEALTHY or DAMAGE
- Temporal Stability: 5-frame sliding window probability smoothing
- Image Quality Filter: Rejects severely dark or blurred frames
- Interactive Secondary Window (--show-crop): Shows the exact joint crop sent to YOLO
- Dataset Collector: Keyboard shortcuts ('h' / 'd') to save real joint samples

Usage:
    python check_yolo/webcam.py
    python check_yolo/webcam.py --source 1 --show-crop
    python check_yolo/webcam.py --roi 0.15,0.20,0.85,0.80
"""

import argparse
from collections import deque
import datetime
import os
import sys
import time
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "joint_yolo_classifier.pt")
FALLBACK_MODEL_PATH = os.path.join(SCRIPT_DIR, "results", "joint_cls", "weights", "best.pt")

COLLECTED_DIR = os.path.join(SCRIPT_DIR, "dataset", "collected")
COLLECTED_HEALTHY_DIR = os.path.join(COLLECTED_DIR, "healthy")
COLLECTED_DAMAGE_DIR = os.path.join(COLLECTED_DIR, "damage")


def parse_args():
    parser = argparse.ArgumentParser(
        description="JointGuard Live Webcam: Fixed ROI -> OpenCV Joint Finder -> YOLO Classification"
    )
    parser.add_argument(
        "--source",
        type=int,
        default=1,
        help="Webcam device index (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained YOLO classification weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.60,
        help="Confidence threshold for certainty (default: 0.60)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="0.15,0.20,0.85,0.80",
        help="Fixed Search ROI as 'x1,y1,x2,y2'. Fractions (0.0-1.0) or pixel integers (default: 0.15,0.20,0.85,0.80)",
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
        help="Bypass OpenCV joint detection and feed the entire Fixed ROI directly to YOLO",
    )
    parser.add_argument(
        "--infer-interval",
        type=int,
        default=2,
        help="Run YOLO inference every N frames to optimize CPU usage (default: 2)",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=5,
        help="Sliding window size for temporal prediction smoothing (default: 5)",
    )
    parser.add_argument(
        "--show-crop",
        action="store_true",
        help="Open secondary window displaying the exact image crop sent to YOLO",
    )
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Disable automatic dark/blur crop validation filter",
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


def resolve_model_path(requested_path: str) -> str:
    """Finds the trained YOLO weights file or throws an informative error."""
    if os.path.isfile(requested_path):
        return os.path.abspath(requested_path)
    if os.path.isfile(FALLBACK_MODEL_PATH):
        return os.path.abspath(FALLBACK_MODEL_PATH)
    # Check results folder
    results_dir = os.path.join(SCRIPT_DIR, "results")
    if os.path.isdir(results_dir):
        for root, _, files in os.walk(results_dir):
            if "best.pt" in files:
                return os.path.abspath(os.path.join(root, "best.pt"))

    raise FileNotFoundError(
        f"Trained YOLO model not found in check_yolo/models/\n"
        f"Expected path: {os.path.abspath(requested_path)}\n"
        f"Please run 'python check_yolo/train.py' to generate the model weights."
    )


def parse_roi_string(roi_str: str, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """
    Parses ROI coordinate string. Supports either normalized fractions (e.g. 0.15,0.20,0.85,0.80)
    or absolute pixel coordinates (e.g. 100,100,540,380).
    """
    parts = [float(p.strip()) for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must contain exactly 4 coordinates 'x1,y1,x2,y2', got '{roi_str}'")

    x1, y1, x2, y2 = parts

    if max(x1, y1, x2, y2) <= 1.0:
        px1 = int(x1 * frame_w)
        py1 = int(y1 * frame_h)
        px2 = int(x2 * frame_w)
        py2 = int(y2 * frame_h)
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

    Steps:
      1. Converts ROI to HSV and Grayscale.
      2. Dynamically thresholds bright/metallic joint elements above median belt intensity.
      3. Performs morphological closing to merge discrete fastener teeth / specular reflections.
      4. Identifies external contours matching conveyor joint physical characteristics.
      5. Merges adjacent fragments and returns the tight bounding box (x, y, w, h) in ROI-relative space.
      6. Returns None if only dark rubber belt is present without a joint.
    """
    h, w = roi_img.shape[:2]
    if h < 20 or w < 20:
        return None

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    v_chan = hsv[:, :, 2]

    # Metallic joints on rubber belt have distinctly higher V intensity
    v_med = float(np.median(v_chan))
    v_lower = max(90, min(210, int(v_med + 20)))

    # Quick pre-filter: if < 1.0% of pixels exceed metallic threshold, no joint is present
    bright_count = np.count_nonzero(v_chan > v_lower)
    if (bright_count / float(h * w)) < 0.010:
        return None

    # Binary mask: HSV V-channel threshold + Grayscale dynamic threshold
    mask_hsv = cv2.inRange(
        hsv, np.array([0, 0, v_lower], dtype=np.uint8), np.array([180, 150, 255], dtype=np.uint8)
    )
    gray_med = float(np.median(gray))
    _, mask_gray = cv2.threshold(gray, max(90, int(gray_med + 25)), 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask_hsv, mask_gray)

    # Morphological closing to bridge fastener teeth / reflection fragments into one joint contour
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
        candidates.append((area, (bx, by, bw, bh)))

    if not candidates:
        return None

    # Select largest candidate and merge any adjacent fragments within 35px
    candidates.sort(key=lambda x: x[0], reverse=True)
    bx, by, bw, bh = candidates[0][1]
    bx2, by2 = bx + bw, by + bh

    for _, (fx, fy, fw, fh) in candidates[1:]:
        fx2, fy2 = fx + fw, fy + fh
        # Check if fragment is close to the main joint plate
        if not (fx > bx2 + 35 or fx2 < bx - 35 or fy > by2 + 35 or fy2 < by - 35):
            bx = min(bx, fx)
            by = min(by, fy)
            bx2 = max(bx2, fx2)
            by2 = max(by2, fy2)

    # Add comfortable padding (12% height, 6% width) to avoid clipping rivets/fasteners
    pad_y = int((by2 - by) * 0.12)
    pad_x = int((bx2 - bx) * 0.06)

    final_x = max(0, bx - pad_x)
    final_y = max(0, by - pad_y)
    final_w = min(w - final_x, (bx2 - bx) + 2 * pad_x)
    final_h = min(h - final_y, (by2 - by) + 2 * pad_y)

    return (final_x, final_y, final_w, final_h)


def check_crop_quality(crop: np.ndarray) -> Tuple[bool, str]:
    """
    Validates crop quality before sending to YOLO.
    Rejects completely dark, severely blurred, or undersized crops.
    """
    h, w = crop.shape[:2]
    if h < 25 or w < 25:
        return False, "TOO SMALL"

    mean_val = float(np.mean(crop))
    if mean_val < 15.0:
        return False, "TOO DARK"
    if mean_val > 245.0:
        return False, "TOO BRIGHT"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 8.0:
        return False, "TOO BLURRED"

    return True, "OK"


def main():
    args = parse_args()

    # 1. Resolve and verify model weights
    try:
        model_file = resolve_model_path(args.model)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Open Camera 1
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam source {args.source}.", file=sys.stderr)
        print(f"[TIP] Verify that the external camera is plugged in.", file=sys.stderr)
        print(f"[TIP] To test with default built-in camera, pass: --source 0", file=sys.stderr)
        sys.exit(1)

    # Set requested resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Read a test frame to get actual dimensions
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print(f"[ERROR] Camera opened but failed to capture a frame from source {args.source}.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    frame_h, frame_w = test_frame.shape[:2]

    # 3. Parse Fixed ROI
    try:
        rx1, ry1, rx2, ry2 = parse_roi_string(args.roi, frame_w, frame_h)
    except Exception as e:
        print(f"[ERROR] Invalid ROI format: {e}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    mode_label = "DIRECT FIXED ROI" if args.direct_roi else "TWO-STAGE (Fixed ROI -> OpenCV Joint Search -> YOLO)"

    # 4. Print Startup Information
    print("=" * 55)
    print("JOINTGUARD YOLO WEBCAM INSPECTION")
    print("=" * 55)
    print(f"Camera source:        {args.source}")
    print(f"Model:                {model_file}")
    print(f"Pipeline Mode:        {mode_label}")
    print(f"Confidence threshold: {args.conf_thresh:.2f}")
    print(f"Fixed Search ROI:     ({rx1}, {ry1}, {rx2}, {ry2}) [resolution: {frame_w}x{frame_h}]")
    print("=" * 55)
    print("[INFO] Camera opened successfully.")

    # 5. Load YOLO model exactly ONCE
    from ultralytics import YOLO

    model = YOLO(model_file)
    print("[INFO] YOLO model loaded successfully.")
    print("[KEYBOARD SHORTCUTS] 'h'=save healthy crop | 'd'=save damage crop | 'q'=quit")

    # Ensure dataset collection directories exist
    os.makedirs(COLLECTED_HEALTHY_DIR, exist_ok=True)
    os.makedirs(COLLECTED_DAMAGE_DIR, exist_ok=True)

    # State tracking
    pred_history = deque(maxlen=args.history_len)
    last_display_pred = "WAITING FOR JOINT"
    last_display_conf = 0.0
    last_display_color = (255, 200, 0)  # Cyan/Yellow neutral

    frame_count = 0
    fps_history = deque(maxlen=30)
    current_fps = 0.0

    save_notification_text = ""
    save_notification_time = 0.0
    last_saved_crop = None

    window_main = "JointGuard YOLO - Live Inspection"
    window_crop = "JointGuard YOLO Input"

    cv2.namedWindow(window_main, cv2.WINDOW_AUTOSIZE)
    if args.show_crop:
        cv2.namedWindow(window_crop, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame capture dropped.")
                time.sleep(0.01)
                continue

            frame_count += 1

            # ==========================================================
            # STEP 1: FIXED ROI (Search Zone)
            # ==========================================================
            fixed_roi_crop = frame[ry1:ry2, rx1:rx2].copy()

            # ==========================================================
            # STEP 2 & 3: OPENCV SEARCHES FOR JOINT -> ACTUAL JOINT BBOX
            # ==========================================================
            joint_detected = False
            actual_joint_crop = None
            actual_bbox_full = None  # (x, y, w, h) in full frame

            if args.direct_roi:
                # Bypass OpenCV detector: treat entire fixed ROI as joint
                joint_detected = True
                actual_joint_crop = fixed_roi_crop
                actual_bbox_full = (rx1, ry1, rx2 - rx1, ry2 - ry1)
            else:
                # Classical OpenCV joint segmentation inside Fixed ROI
                rel_bbox = find_joint_in_roi(
                    fixed_roi_crop,
                    min_area=args.min_joint_area,
                    min_width=args.min_joint_width,
                )

                if rel_bbox is not None:
                    jx, jy, jw, jh = rel_bbox
                    # Map to full-frame coordinates
                    fx, fy = rx1 + jx, ry1 + jy
                    actual_bbox_full = (fx, fy, jw, jh)
                    actual_joint_crop = frame[fy:fy+jh, fx:fx+jw].copy()
                    joint_detected = True

            # ==========================================================
            # STEP 4 & 5: YOLO INFERENCE & HEALTHY / DAMAGE SCORING
            # ==========================================================
            if joint_detected and actual_joint_crop is not None and actual_joint_crop.size > 0:
                last_saved_crop = actual_joint_crop.copy()

                # Quality check
                is_valid, quality_reason = (True, "OK") if args.no_quality_check else check_crop_quality(actual_joint_crop)

                if is_valid:
                    if frame_count % args.infer_interval == 0 or len(pred_history) == 0:
                        # Send ACTUAL JOINT CROP to trained YOLO model
                        yolo_res = model(actual_joint_crop, verbose=False)[0]

                        # Extract probabilities
                        p_healthy = 0.0
                        p_damage = 0.0
                        for idx, name in yolo_res.names.items():
                            c_prob = float(yolo_res.probs.data[idx])
                            if str(name).strip().lower() == "healthy":
                                p_healthy = c_prob
                            elif str(name).strip().lower() == "damage":
                                p_damage = c_prob

                        pred_history.append((p_healthy, p_damage))

                    # Temporal stability smoothing across last N predictions
                    if len(pred_history) > 0:
                        avg_healthy = float(np.mean([p[0] for p in pred_history]))
                        avg_damage = float(np.mean([p[1] for p in pred_history]))

                        if avg_healthy >= avg_damage:
                            smooth_class = "HEALTHY"
                            smooth_conf = avg_healthy
                        else:
                            smooth_class = "DAMAGE"
                            smooth_conf = avg_damage

                        # Threshold evaluation
                        if smooth_conf >= args.conf_thresh:
                            last_display_pred = smooth_class
                            last_display_conf = smooth_conf * 100.0
                            last_display_color = (0, 220, 0) if smooth_class == "HEALTHY" else (0, 0, 230)
                        else:
                            last_display_pred = "UNCERTAIN"
                            last_display_conf = smooth_conf * 100.0
                            last_display_color = (0, 200, 255)  # Amber
                else:
                    last_display_pred = f"INVALID ({quality_reason})"
                    last_display_conf = 0.0
                    last_display_color = (128, 128, 128)
            else:
                # No joint detected by OpenCV in search ROI
                last_display_pred = "NO JOINT IN ROI"
                last_display_conf = 0.0
                last_display_color = (255, 180, 0)  # Cyan
                pred_history.clear()
                last_saved_crop = None

            # ==========================================================
            # STEP 6: SECONDARY CROP WINDOW (--show-crop)
            # ==========================================================
            if args.show_crop:
                if joint_detected and actual_joint_crop is not None and actual_joint_crop.size > 0:
                    crop_view = actual_joint_crop.copy()
                    cv2.rectangle(
                        crop_view,
                        (0, 0),
                        (crop_view.shape[1] - 1, crop_view.shape[0] - 1),
                        last_display_color,
                        3,
                    )
                    # Label
                    cv2.putText(
                        crop_view,
                        f"YOLO INPUT: {last_display_pred}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        last_display_color,
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    crop_view = fixed_roi_crop.copy()
                    cv2.putText(
                        crop_view,
                        "SEARCHING FOR JOINT...",
                        (20, crop_view.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_crop, crop_view)

            # ==========================================================
            # STEP 7: RENDER HUD ON MAIN DISPLAY
            # ==========================================================
            t_now = time.time()
            fps_history.append(1.0 / max(1e-5, t_now - t_start))
            current_fps = float(np.mean(fps_history))

            display_frame = frame.copy()

            # A. Draw Fixed Search ROI (Subtle Frame)
            cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 180, 0), 1)
            cv2.putText(
                display_frame,
                "FIXED SEARCH ROI",
                (rx1 + 5, ry1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

            # B. Draw Actual Joint Bounding Box (when OpenCV detects joint)
            if joint_detected and actual_bbox_full is not None:
                bx, by, bw, bh = actual_bbox_full
                joint_color = last_display_color

                # Main bounding box
                cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), joint_color, 2)

                # Corner accent brackets
                c_len = min(20, min(bw, bh) // 3)
                # Top-left
                cv2.line(display_frame, (bx, by), (bx + c_len, by), joint_color, 4)
                cv2.line(display_frame, (bx, by), (bx, by + c_len), joint_color, 4)
                # Top-right
                cv2.line(display_frame, (bx + bw, by), (bx + bw - c_len, by), joint_color, 4)
                cv2.line(display_frame, (bx + bw, by), (bx + bw, by + c_len), joint_color, 4)
                # Bottom-left
                cv2.line(display_frame, (bx, by + bh), (bx + c_len, by + bh), joint_color, 4)
                cv2.line(display_frame, (bx, by + bh), (bx, by + bh - c_len), joint_color, 4)
                # Bottom-right
                cv2.line(display_frame, (bx + bw, by + bh), (bx + bw - c_len, by + bh), joint_color, 4)
                cv2.line(display_frame, (bx + bw, by + bh), (bx + bw, by + bh - c_len), joint_color, 4)

                # Detection badge
                badge_text = f"JOINT: {last_display_pred}"
                if last_display_conf > 0:
                    badge_text += f" ({last_display_conf:.1f}%)"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_frame, (bx, by - 22), (bx + tw + 10, by), (20, 20, 20), -1)
                cv2.putText(
                    display_frame,
                    badge_text,
                    (bx + 5, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    joint_color,
                    1,
                    cv2.LINE_AA,
                )

            # C. Top Status Bar
            cv2.rectangle(display_frame, (0, 0), (frame_w, 36), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                f"FPS: {current_fps:.1f}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 180),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Source: {args.source} | Pipeline: OpenCV -> YOLO",
                (frame_w - 320, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            # D. Bottom Inspection Banner
            banner_h = 75
            cv2.rectangle(
                display_frame,
                (0, frame_h - banner_h),
                (frame_w, frame_h),
                (15, 15, 15),
                -1,
            )

            # Classification Text
            cv2.putText(
                display_frame,
                last_display_pred,
                (15, frame_h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                last_display_color,
                2,
                cv2.LINE_AA,
            )

            # Confidence Text
            if last_display_conf > 0:
                conf_text = f"Confidence: {last_display_conf:.1f}% (thresh: {args.conf_thresh*100:.0f}%)"
            else:
                conf_text = "Status: Monitoring belt for joint passage"

            cv2.putText(
                display_frame,
                conf_text,
                (15, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Controls Hint (Bottom Right)
            cv2.putText(
                display_frame,
                "[H] Save Healthy | [D] Save Damage | [Q] Quit",
                (frame_w - 330, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

            # E. Save Notification Banner
            if time.time() - save_notification_time < 2.0:
                cv2.rectangle(
                    display_frame,
                    (frame_w // 2 - 220, 45),
                    (frame_w // 2 + 220, 85),
                    (0, 120, 0) if "SAVED" in save_notification_text else (0, 0, 150),
                    -1,
                )
                cv2.putText(
                    display_frame,
                    save_notification_text,
                    (frame_w // 2 - 200, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_main, display_frame)

            # ==========================================================
            # KEYBOARD INTERACTION
            # ==========================================================
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q") or key == 27:
                print("\n[INFO] 'q' pressed. Exiting live inspection.")
                break

            elif key == ord("h") or key == ord("H"):
                crop_to_save = last_saved_crop if last_saved_crop is not None else fixed_roi_crop
                if crop_to_save is not None and crop_to_save.size > 0:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                    filename = f"healthy_{timestamp}.jpg"
                    save_path = os.path.join(COLLECTED_HEALTHY_DIR, filename)
                    cv2.imwrite(save_path, crop_to_save)
                    save_notification_text = f"SAVED HEALTHY JOINT: {filename}"
                    save_notification_time = time.time()
                    print(f"[COLLECTED] Saved healthy joint: {save_path}")
                else:
                    save_notification_text = "CANNOT SAVE: NO JOINT DETECTED"
                    save_notification_time = time.time()

            elif key == ord("d") or key == ord("D"):
                crop_to_save = last_saved_crop if last_saved_crop is not None else fixed_roi_crop
                if crop_to_save is not None and crop_to_save.size > 0:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                    filename = f"damage_{timestamp}.jpg"
                    save_path = os.path.join(COLLECTED_DAMAGE_DIR, filename)
                    cv2.imwrite(save_path, crop_to_save)
                    save_notification_text = f"SAVED DAMAGE JOINT: {filename}"
                    save_notification_time = time.time()
                    print(f"[COLLECTED] Saved damage joint: {save_path}")
                else:
                    save_notification_text = "CANNOT SAVE: NO JOINT DETECTED"
                    save_notification_time = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed cleanly.")


if __name__ == "__main__":
    main()
