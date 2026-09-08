"""
JointGuard Live Test Runner & Pipeline Executable (Phase 1 Classical CV)

Usage:
  python vision/opencv/main.py --source 0 --debug --save-log joint_events.jsonl
  python vision/opencv/main.py --source vision/opencv/test_images/healthy_sample.jpg --debug
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import Optional, List, Dict, Any, Union
import cv2
import numpy as np

# Add project root directory to sys.path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vision.opencv.config import VisionConfig, DEFAULT_CONFIG
from vision.opencv.detector import detect_joint, get_last_detection_mask
from vision.tracking.tracker import JointTracker, TrackEvent


def parse_args():
    parser = argparse.ArgumentParser(description="JointGuard Conveyor Joint Inspection & Tracking Runner")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: webcam index (e.g. 0), video file path (.mp4), or image file path (.jpg)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display OpenCV live window with bounding box, inspection ROI, track ID, state, and vision score"
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Display second OpenCV window showing binary threshold mask (post-morphology, pre-contour-filtering)"
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable Gaussian blur noise reduction on live webcam stream"
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="Target capture FPS for webcam (default 30)"
    )
    parser.add_argument(
        "--save-log",
        type=str,
        default=None,
        help="Path to write JSON-lines log file of finalized classification events"
    )
    return parser.parse_args()


def get_timestamp() -> str:
    """Returns ISO8601 formatted timestamp string."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def draw_debug_overlay(
    frame: np.ndarray,
    config: VisionConfig,
    track_events: Any = None,
    fps: float = 0.0,
    is_static_image: bool = False
) -> np.ndarray:
    """Draws inspection ROI rectangle, capture zone, bounding box, state labels, and measured FPS onto frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Outer Inspection ROI Box (Cyan boundary)
    roi_x1 = int(config.ROI_X_MIN * w)
    roi_y1 = int(config.ROI_Y_MIN * h)
    roi_x2 = int(config.ROI_X_MAX * w)
    roi_y2 = int(config.ROI_Y_MAX * h)
    cv2.rectangle(canvas, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
    cv2.putText(
        canvas,
        "OUTER ROI",
        (roi_x1 + 5, roi_y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 0),
        1,
        cv2.LINE_AA
    )

    # 2. Draw Inner Capture Zone Box (Magenta boundary)
    cz_x1 = int(config.CAPTURE_ZONE_X_MIN * w)
    cz_y1 = int(config.CAPTURE_ZONE_Y_MIN * h)
    cz_x2 = int(config.CAPTURE_ZONE_X_MAX * w)
    cz_y2 = int(config.CAPTURE_ZONE_Y_MAX * h)
    cv2.rectangle(canvas, (cz_x1, cz_y1), (cz_x2, cz_y2), (255, 0, 255), 2)
    cv2.putText(
        canvas,
        "CAPTURE ZONE",
        (cz_x1 + 5, cz_y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 255),
        1,
        cv2.LINE_AA
    )

    # 3. Draw Measured FPS Overlay in top-left
    if fps > 0:
        cv2.putText(
            canvas,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # 4. Draw Bounding Boxes and Track Status for all active tracks
    if track_events is not None:
        event_list = track_events if isinstance(track_events, list) else [track_events]
    else:
        event_list = []

    for te in event_list:
        if te and te.bbox:
            bbox = te.bbox
            x, y, bw, bh = bbox.to_tuple()

            color = (0, 255, 255)  # Yellow default for APPROACHING
            if te.state == "INSPECTING":
                if te.classification:
                    if te.classification.label == "HEALTHY":
                        color = (0, 255, 0)
                    elif te.classification.label == "DAMAGE":
                        color = (0, 0, 255)
                    else:
                        color = (128, 128, 128)  # Grey for INVALID / no signal
                else:
                    color = (0, 255, 0)
            elif te.state == "PASSED":
                color = (128, 128, 128)

            # Draw bounding box and centroid marker
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
            cv2.circle(canvas, bbox.center, 4, (0, 0, 255), -1)

            # Format status string: ID:{track_id} [{state}] | {LABEL} ({vision_score})
            status_str = f"ID:{te.track_id} [{te.state}]"
            if te.classification:
                if te.classification.label == "INVALID":
                    status_str += " | INVALID/no signal"
                else:
                    status_str += f" | {te.classification.label} ({te.classification.vision_score:.1f})"

            text_w = max(150, len(status_str) * 9)
            cv2.rectangle(canvas, (x, max(0, y - 22)), (x + text_w, max(0, y)), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                status_str,
                (x + 5, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

            # Draw raw feature metrics below the box once locked
            if te.classification:
                cls = te.classification
                feat_y = y + bh + 15
                for k, v in cls.features.items():
                    feat_str = f"{k}: {v:.4f}"
                    cv2.putText(canvas, feat_str, (x, min(h - 10, feat_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    feat_y += 15

    return canvas


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    tracker = JointTracker(cfg)

    # Open log file if requested
    log_file = None
    if args.save_log:
        log_dir = os.path.dirname(os.path.abspath(args.save_log))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = open(args.save_log, "a", encoding="utf-8")
        print(f"[INFO] Logging finalized events to: {args.save_log}")

    # Determine input source mode
    source_input = args.source
    is_webcam = False
    is_video = False
    is_image = False

    if source_input.isdigit():
        cap = cv2.VideoCapture(int(source_input))
        is_webcam = True
    elif os.path.isfile(source_input):
        ext = os.path.splitext(source_input)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            is_image = True
        else:
            cap = cv2.VideoCapture(source_input)
            is_video = True
    else:
        print(f"[ERROR] Source input '{source_input}' is invalid.")
        sys.exit(1)

    if is_webcam:
        cap.set(cv2.CAP_PROP_FPS, float(args.target_fps))
        granted_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Requested Target FPS: {args.target_fps} | OpenCV Granted FPS: {granted_fps:.1f}")

    print(f"[INFO] Starting JointGuard CV Runner (Source: {source_input})")
    print("[INFO] Press 'q' in debug window to exit.")

    if is_image:
        frame = cv2.imread(source_input)
        if frame is None:
            print(f"[ERROR] Failed to load image: {source_input}")
            sys.exit(1)

        ts = get_timestamp()
        bbox = detect_joint(frame, cfg)
        track_event = tracker.update(bbox, ts, frame)

        if track_event and track_event.is_new_classification and track_event.classification:
            event_payload = {
                "timestamp": ts,
                "track_id": track_event.track_id,
                "label": track_event.classification.label,
                "vision_score": round(track_event.classification.vision_score, 2),
                "features": {k: round(v, 4) for k, v in track_event.classification.features.items()}
            }
            print("\n[FINALIZED CLASSIFICATION EVENT]")
            print(json.dumps(event_payload, indent=2))
            if log_file:
                log_file.write(json.dumps(event_payload) + "\n")
                log_file.flush()

            gui_available = True
            if args.debug or args.show_mask:
                annotated = draw_debug_overlay(frame, cfg, track_event, is_static_image=True)
                mask = get_last_detection_mask()
                try:
                    if args.debug:
                        cv2.imshow("JointGuard Live Inspection (Debug Mode)", annotated)
                    if args.show_mask and mask is not None:
                        cv2.imshow("JointGuard Binary Mask (Debug)", mask)
                    print("[INFO] Static image displayed. Press any key in window to close...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    print("[WARNING] OpenCV GUI window unavailable in this Python environment.")
                    print("[INFO] Running in headless mode (logging events without live window).")

    else:
        # Video stream / Webcam loop
        gui_available = True
        frame_timestamps = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_video:
                    print("[INFO] End of video stream.")
                break

            now = time.time()
            frame_timestamps.append(now)
            if len(frame_timestamps) > 30:
                frame_timestamps.pop(0)

            measured_fps = 0.0
            if len(frame_timestamps) > 1:
                dt = frame_timestamps[-1] - frame_timestamps[0]
                if dt > 0:
                    measured_fps = float(len(frame_timestamps) - 1) / dt

            if not args.no_denoise and cfg.LIVE_DENOISE_KERNEL[0] > 1:
                frame = cv2.GaussianBlur(frame, cfg.LIVE_DENOISE_KERNEL, 0)

            ts = get_timestamp()
            bbox = detect_joint(frame, cfg)
            track_event = tracker.update(bbox, ts, frame)

            # Log only when a new classification is finalized (locked)
            if track_event and track_event.is_new_classification and track_event.classification:
                event_payload = {
                    "timestamp": ts,
                    "track_id": track_event.track_id,
                    "label": track_event.classification.label,
                    "vision_score": round(track_event.classification.vision_score, 2),
                    "features": {k: round(v, 4) for k, v in track_event.classification.features.items()}
                }
                print(f"[EVENT LOGGED] {json.dumps(event_payload)}")
                if log_file:
                    log_file.write(json.dumps(event_payload) + "\n")
                    log_file.flush()

            if (args.debug or args.show_mask) and gui_available:
                active_events = tracker.get_active_track_events()
                annotated = draw_debug_overlay(frame, cfg, active_events, fps=measured_fps)
                mask = get_last_detection_mask()
                try:
                    if args.debug:
                        cv2.imshow("JointGuard Live Inspection (Debug Mode)", annotated)
                    if args.show_mask and mask is not None:
                        cv2.imshow("JointGuard Binary Mask (Debug)", mask)

                    key = cv2.waitKey(30 if is_video else 1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Quit signal received.")
                        break
                except cv2.error as e:
                    print("[WARNING] OpenCV GUI window unavailable in this Python environment.")
                    print("[INFO] Continuing in headless logging mode...")
                    gui_available = False

        cap.release()
        if (args.debug or args.show_mask) and gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if log_file:
        log_file.close()

    print("[INFO] JointGuard CV runner finished.")


if __name__ == "__main__":
    main()
