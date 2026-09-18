"""
JointGuard Live Test Runner & Pipeline Executable

Runs automated conveyor-belt joint inspection using real-time joint localization,
feature extraction, and a trained machine learning classifier with confidence estimation.

Usage:
  python vision/opencv/main.py --source 0 --debug
  python vision/opencv/main.py --source 0 --debug --show-crop
  python vision/opencv/main.py --source vision/opencv/test_images/healthy_sample.jpg --debug --show-crop
  python vision/opencv/main.py --source vision/opencv/test_images/damage_sample.jpg --debug --show-crop
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import Optional, List, Dict, Any, Union, Tuple
import cv2
import numpy as np

# Add project root directory to sys.path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vision.opencv.config import VisionConfig, DEFAULT_CONFIG
from vision.opencv.detector import detect_joint, get_last_detection_mask, BoundingBox
from vision.opencv.classifier import classify_joint, load_trained_model
from vision.opencv.features import preprocess_joint_crop
from vision.tracking.tracker import JointTracker, TrackEvent


def parse_args():
    parser = argparse.ArgumentParser(description="JointGuard Conveyor Joint Inspection & Tracking Runner")
    parser.add_argument(
        "--source",
        type=str,
        default="1",
        help="Input source: webcam index (e.g. 0 or 1), video file path (.mp4), or image file path (.jpg)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display OpenCV live window with bounding box, inspection ROI, track ID, state, and vision score"
    )
    parser.add_argument(
        "--show-crop",
        action="store_true",
        help="Display separate OpenCV window showing the normalized joint crop sent to the classifier"
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


def save_dataset_crop(crop_img: Optional[np.ndarray], category: str) -> Tuple[bool, str, int, int]:
    """
    Saves the exact current classifier input crop to dataset/{category}/.
    Category must be 'healthy' or 'damage'.
    Returns (success, filepath, healthy_count, damage_count).
    """
    if crop_img is None or crop_img.size == 0 or crop_img.shape[0] < 5 or crop_img.shape[1] < 5:
        return False, "", 0, 0

    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    target_dir = os.path.join(dataset_dir, category)
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"{category}_{timestamp}.jpg"
    filepath = os.path.join(target_dir, filename)

    # Save exact raw crop as high-quality JPEG
    success = cv2.imwrite(filepath, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not success:
        return False, "", 0, 0

    healthy_dir = os.path.join(dataset_dir, "healthy")
    damage_dir = os.path.join(dataset_dir, "damage")
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(damage_dir, exist_ok=True)

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    h_count = len([f for f in os.listdir(healthy_dir) if f.lower().endswith(valid_exts)])
    d_count = len([f for f in os.listdir(damage_dir) if f.lower().endswith(valid_exts)])

    return True, filepath, h_count, d_count


def draw_debug_overlay(
    frame: np.ndarray,
    config: VisionConfig,
    track_events: Any = None,
    fps: float = 0.0,
    is_static_image: bool = False,
    current_bbox: Any = None,
    capture_banner: Optional[str] = None,
    banner_color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draws inspection ROI rectangle, capture zone, bounding box, state labels, and measured FPS onto frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Outer Inspection ROI Box (Cyan boundary)
    roi_x1 = int(config.ROI_X_MIN * w)
    roi_y1 = int(config.ROI_Y_MIN * h)
    roi_x2 = int(config.ROI_X_MAX * w)
    roi_y2 = int(config.ROI_Y_MAX * h)
    cv2.rectangle(canvas, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 1)

    # 2. Draw Inner Capture Zone Box (Magenta boundary)
    cz_x1 = int(config.CAPTURE_ZONE_X_MIN * w)
    cz_y1 = int(config.CAPTURE_ZONE_Y_MIN * h)
    cz_x2 = int(config.CAPTURE_ZONE_X_MAX * w)
    cz_y2 = int(config.CAPTURE_ZONE_Y_MAX * h)
    cv2.rectangle(canvas, (cz_x1, cz_y1), (cz_x2, cz_y2), (255, 0, 255), 2)

    # 3. Draw Top Unified HUD Status Bar (y = 0 to 36)
    cv2.rectangle(canvas, (0, 0), (w, 36), (15, 15, 15), -1)
    cv2.line(canvas, (0, 36), (w, 36), (60, 60, 60), 1)

    # Left: FPS indicator
    if fps > 0:
        cv2.putText(canvas, f"FPS: {fps:.1f}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "INSPECTION ACTIVE", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Right: Clean Legend for Zones
    cv2.rectangle(canvas, (w - 210, 11), (w - 200, 22), (255, 255, 0), -1)
    cv2.putText(canvas, "OUTER ROI", (w - 194, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (w - 110, 11), (w - 100, 22), (255, 0, 255), -1)
    cv2.putText(canvas, "CAPTURE ZONE", (w - 94, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 255), 1, cv2.LINE_AA)

    # 4. Resolve Active Detection / Classification
    if track_events is not None:
        event_list = track_events if isinstance(track_events, list) else [track_events]
    else:
        event_list = []

    active_event = None
    for te in event_list:
        if te and te.bbox:
            active_event = te
            break

    # Determine Authoritative Status String & Color
    # Rules:
    #   - DAMAGED: RED (0, 0, 255)
    #   - HEALTHY: GREEN (0, 255, 0)
    #   - UNCERTAIN (<60%): ORANGE (0, 165, 255)
    #   - NO JOINT DETECTED: GRAY (140, 140, 140)
    if active_event is None and current_bbox is None:
        hud_status = "STATUS: NO JOINT DETECTED"
        hud_color = (140, 140, 140)
    else:
        te = active_event
        cls = te.classification if te else None
        track_id_str = f"ID:{te.track_id} [{te.state}]" if te else "ID:-- [DETECTING]"

        if cls is not None:
            raw_label = cls.label
            conf = cls.features.get("confidence", 0.0)
            conf_pct = conf * 100.0

            if raw_label == "DAMAGE":
                display_label = "DAMAGED"
                hud_color = (0, 0, 255)      # RED
            elif raw_label == "HEALTHY":
                display_label = "HEALTHY"
                hud_color = (0, 255, 0)      # GREEN
            else:
                display_label = "UNCERTAIN"
                hud_color = (0, 165, 255)    # ORANGE

            hud_status = f"{track_id_str}  |  {display_label} ({conf_pct:.1f}%)"
        else:
            hud_status = f"{track_id_str}  |  ANALYZING JOINT..."
            hud_color = (0, 255, 255)        # YELLOW

    # Draw Center Status in Top HUD Bar
    cv2.putText(
        canvas,
        hud_status,
        (120, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        hud_color,
        2,
        cv2.LINE_AA
    )

    # 5. Draw Bounding Box and Details for Active Tracks
    for te in event_list:
        if te and te.bbox:
            bbox = te.bbox
            x, y, bw, bh = bbox.to_tuple()

            cls = te.classification
            if cls:
                if cls.label == "DAMAGE":
                    box_color = (0, 0, 255)       # RED for DAMAGED
                    disp_name = "DAMAGED"
                elif cls.label == "HEALTHY":
                    box_color = (0, 255, 0)       # GREEN for HEALTHY
                    disp_name = "HEALTHY"
                else:
                    box_color = (0, 165, 255)     # ORANGE for UNCERTAIN
                    disp_name = "UNCERTAIN"
                conf_val = cls.features.get("confidence", 0.0) * 100.0
                pill_text = f"ID:{te.track_id} [{te.state}] | {disp_name} ({conf_val:.1f}%)"
            else:
                box_color = (0, 255, 255)         # YELLOW for APPROACHING
                pill_text = f"ID:{te.track_id} [{te.state}]"

            # Draw bounding box rectangle
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), box_color, 2)
            cv2.circle(canvas, bbox.center, 4, (0, 0, 255), -1)

            # Draw pill header: position cleanly above box if space, else inside top edge of box
            text_size, _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tw, th = text_size[0] + 10, text_size[1] + 8

            if y - th > 40:
                py1 = y - th
                py2 = y
            else:
                py1 = y + 2
                py2 = y + 2 + th

            cv2.rectangle(canvas, (x, py1), (x + tw, py2), (15, 15, 15), -1)
            cv2.putText(
                canvas,
                pill_text,
                (x + 5, py2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA
            )

    # 6. Bottom Dataset Capture & Helper Bar (y = h - 28 to h)
    cv2.rectangle(canvas, (0, h - 28), (w, h), (18, 18, 18), -1)
    cv2.line(canvas, (0, h - 28), (w, h - 28), (55, 55, 55), 1)

    hotkey_hint = "[H] Save Healthy  |  [D] Save Damaged  |  [S] Skip  |  [Q] Quit"
    cv2.putText(canvas, hotkey_hint, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    if capture_banner:
        cv2.putText(canvas, capture_banner, (w - 460, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, banner_color, 1, cv2.LINE_AA)

    return canvas


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    tracker = JointTracker(cfg)

    # 1. Preload trained model once on startup with explicit diagnostics
    print("=" * 72)
    print("              JOINTGUARD CONVEYOR JOINT INSPECTION RUNNER")
    print("=" * 72)
    print(f"[STARTUP] Attempting to load trained model from: {cfg.MODEL_PATH}")
    model_loaded = load_trained_model(cfg.MODEL_PATH)
    if not model_loaded:
        print("\n" + "!" * 72)
        print(f"[CRITICAL ERROR] Trained model failed to load from: {cfg.MODEL_PATH}")
        print("Please train the classifier first by running:")
        print("    python vision/opencv/train_classifier.py")
        print("!" * 72 + "\n")
        sys.exit(1)
    else:
        print(f"[STARTUP SUCCESS] PyTorch MobileNetV2 loaded successfully from: {cfg.MODEL_PATH}")
        print(f"[STARTUP INFO]    Architecture       : MobileNetV2 (Pretrained ImageNet + Fine-tuned Binary Head)")
        print(f"[STARTUP INFO]    Class Mapping      : 0 = HEALTHY, 1 = DAMAGE (DAMAGED)")
        print(f"[STARTUP INFO]    Input Tensor Shape : (1, 3, 224, 224) [ImageNet RGB normalized]")
        print(f"[STARTUP INFO]    Confidence Cutoff  : {cfg.CONFIDENCE_THRESHOLD * 100.0:.0f}% (below this -> UNCERTAIN)")
    print("=" * 72)

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
        roi = bbox.crop_roi(frame) if bbox is not None else frame
        classification = classify_joint(roi, cfg)
        track_event = TrackEvent(
            track_id=1,
            state="INSPECTING",
            bbox=bbox,
            classification=classification,
            is_new_classification=True
        )

        if track_event and track_event.is_new_classification and track_event.classification:
            event_payload = {
                "timestamp": ts,
                "track_id": track_event.track_id,
                "label": track_event.classification.label,
                "vision_score": round(track_event.classification.vision_score, 2),
                "features": {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in track_event.classification.features.items()}
            }
            print("\n[FINALIZED CLASSIFICATION EVENT]")
            print(json.dumps(event_payload, indent=2))
            if log_file:
                log_file.write(json.dumps(event_payload) + "\n")
                log_file.flush()

            gui_available = True
            if args.debug or args.show_mask or args.show_crop:
                annotated = draw_debug_overlay(frame, cfg, track_event, is_static_image=True, current_bbox=bbox)
                mask = get_last_detection_mask()
                try:
                    if args.debug:
                        cv2.imshow("JointGuard Live Inspection (Debug Mode)", annotated)
                    if args.show_mask and mask is not None:
                        cv2.imshow("JointGuard Binary Mask (Debug)", mask)
                    if args.show_crop and roi is not None and roi.size > 0:
                        norm_crop = preprocess_joint_crop(roi, (224, 224))
                        display_crop = cv2.resize(norm_crop, (280, 280), interpolation=cv2.INTER_NEAREST)
                        cv2.rectangle(display_crop, (0, 0), (280, 42), (20, 20, 20), -1)

                        conf_pct = classification.features.get('confidence', 0.0) * 100.0
                        if classification.label == "DAMAGE":
                            disp_label = "DAMAGED"
                            text_color = (0, 0, 255)
                        elif classification.label == "HEALTHY":
                            disp_label = "HEALTHY"
                            text_color = (0, 255, 0)
                        else:
                            disp_label = "UNCERTAIN"
                            text_color = (0, 165, 255)

                        crop_label = f"{disp_label} ({conf_pct:.1f}%)"
                        cv2.putText(
                            display_crop,
                            crop_label,
                            (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            text_color,
                            2,
                            cv2.LINE_AA
                        )
                        # Bottom raw probabilities
                        cv2.rectangle(display_crop, (0, 245), (280, 280), (20, 20, 20), -1)
                        p_h = classification.features.get("p_healthy", 0.0) * 100.0
                        p_d = classification.features.get("p_damage", 0.0) * 100.0
                        prob_str = f"P(H): {p_h:.1f}% | P(D): {p_d:.1f}%"
                        cv2.putText(
                            display_crop,
                            prob_str,
                            (10, 268),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (220, 220, 220),
                            1,
                            cv2.LINE_AA
                        )
                        cv2.imshow("JointGuard Classifier Input Crop", display_crop)

                    print("[INFO] Static image displayed. Hotkeys: [H]=Save Healthy, [D]=Save Damaged, [S]=Skip, [Q]/any key=Close")
                    wait_time = 100 if os.environ.get("HEADLESS_TEST", "0") == "1" else 0
                    key = cv2.waitKey(wait_time) & 0xFF
                    if key in (ord('h'), ord('H')):
                        if roi is not None and roi.size > 0:
                            ok, path, h_c, d_c = save_dataset_crop(roi, "healthy")
                            if ok:
                                print(f"[DATASET CAPTURE] Saved HEALTHY crop to: {path}")
                                print(f"                  Dataset Total: {h_c} Healthy, {d_c} Damage ({h_c + d_c} total)")
                    elif key in (ord('d'), ord('D')):
                        if roi is not None and roi.size > 0:
                            ok, path, h_c, d_c = save_dataset_crop(roi, "damage")
                            if ok:
                                print(f"[DATASET CAPTURE] Saved DAMAGE crop to: {path}")
                                print(f"                  Dataset Total: {h_c} Healthy, {d_c} Damage ({h_c + d_c} total)")
                    elif key in (ord('s'), ord('S')):
                        print("[DATASET CAPTURE] Skipped current frame (no image saved).")
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    print("[WARNING] OpenCV GUI window unavailable in this Python environment.")
                    print("[INFO] Running in headless mode (logging events without live window).")

    else:
        # Video stream / Webcam loop
        gui_available = True
        frame_timestamps = []
        capture_banner = None
        banner_color = (0, 255, 0)
        banner_frames_left = 0

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
            current_crop = bbox.crop_roi(frame) if bbox is not None else None
            track_event = tracker.update(bbox, ts, frame)

            # Log only when a new classification is finalized (locked)
            if track_event and track_event.is_new_classification and track_event.classification:
                event_payload = {
                    "timestamp": ts,
                    "track_id": track_event.track_id,
                    "label": track_event.classification.label,
                    "vision_score": round(track_event.classification.vision_score, 2),
                    "features": {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in track_event.classification.features.items()}
                }
                print(f"[EVENT LOGGED] {json.dumps(event_payload)}")
                if log_file:
                    log_file.write(json.dumps(event_payload) + "\n")
                    log_file.flush()

            if (args.debug or args.show_mask or args.show_crop) and gui_available:
                active_events = tracker.get_active_track_events()
                annotated = draw_debug_overlay(
                    frame,
                    cfg,
                    active_events,
                    fps=measured_fps,
                    current_bbox=bbox,
                    capture_banner=capture_banner,
                    banner_color=banner_color
                )
                mask = get_last_detection_mask()
                try:
                    if args.debug:
                        cv2.imshow("JointGuard Live Inspection (Debug Mode)", annotated)
                    if args.show_mask and mask is not None:
                        cv2.imshow("JointGuard Binary Mask (Debug)", mask)
                    if args.show_crop:
                        # Extract crop from detected joint or center belt
                        crop_img = current_crop

                        if crop_img is not None and crop_img.size > 0 and crop_img.shape[0] >= 5 and crop_img.shape[1] >= 5:
                            # Live per-frame classification of the exact visible crop
                            live_res = classify_joint(crop_img, cfg)
                            live_conf = live_res.features.get("confidence", 0.0) * 100.0

                            if live_res.label == "DAMAGE":
                                display_name = "DAMAGED"
                                text_color = (0, 0, 255)      # RED
                            elif live_res.label == "HEALTHY":
                                display_name = "HEALTHY"
                                text_color = (0, 255, 0)      # GREEN
                            else:
                                display_name = "UNCERTAIN"
                                text_color = (0, 165, 255)    # ORANGE

                            crop_label = f"{display_name} ({live_conf:.1f}%)"

                            norm_crop = preprocess_joint_crop(crop_img, (224, 224))
                            display_crop = cv2.resize(norm_crop, (280, 280), interpolation=cv2.INTER_NEAREST)

                            # Header background bar
                            cv2.rectangle(display_crop, (0, 0), (280, 42), (20, 20, 20), -1)
                            cv2.putText(
                                display_crop,
                                crop_label,
                                (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                text_color,
                                2,
                                cv2.LINE_AA
                            )

                            # Bottom telemetry bar with raw class probabilities
                            cv2.rectangle(display_crop, (0, 245), (280, 280), (20, 20, 20), -1)
                            p_h = live_res.features.get("p_healthy", 0.0) * 100.0
                            p_d = live_res.features.get("p_damage", 0.0) * 100.0
                            prob_str = f"P(H): {p_h:.1f}% | P(D): {p_d:.1f}%"
                            cv2.putText(
                                display_crop,
                                prob_str,
                                (10, 268),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (220, 220, 220),
                                1,
                                cv2.LINE_AA
                            )
                            cv2.imshow("JointGuard Classifier Input Crop", display_crop)

                    if banner_frames_left > 0:
                        banner_frames_left -= 1
                        if banner_frames_left == 0:
                            capture_banner = None

                    key = cv2.waitKey(30 if is_video else 1) & 0xFF
                    if key in (ord('h'), ord('H')):
                        if current_crop is not None and current_crop.size > 0:
                            ok, path, h_c, d_c = save_dataset_crop(current_crop, "healthy")
                            if ok:
                                print(f"[DATASET CAPTURE] Saved HEALTHY crop to: {path}")
                                print(f"                  Dataset Total: {h_c} Healthy, {d_c} Damage ({h_c + d_c} total)")
                                capture_banner = f"SAVED: healthy (Total: {h_c}H, {d_c}D)"
                                banner_color = (0, 255, 0)
                                banner_frames_left = 45
                        else:
                            print("[DATASET CAPTURE] Cannot save: No joint currently detected in CAPTURE ZONE.")
                            capture_banner = "CANNOT SAVE: NO JOINT DETECTED"
                            banner_color = (0, 165, 255)
                            banner_frames_left = 45

                    elif key in (ord('d'), ord('D')):
                        if current_crop is not None and current_crop.size > 0:
                            ok, path, h_c, d_c = save_dataset_crop(current_crop, "damage")
                            if ok:
                                print(f"[DATASET CAPTURE] Saved DAMAGE crop to: {path}")
                                print(f"                  Dataset Total: {h_c} Healthy, {d_c} Damage ({h_c + d_c} total)")
                                capture_banner = f"SAVED: damage (Total: {h_c}H, {d_c}D)"
                                banner_color = (0, 0, 255)
                                banner_frames_left = 45
                        else:
                            print("[DATASET CAPTURE] Cannot save: No joint currently detected in CAPTURE ZONE.")
                            capture_banner = "CANNOT SAVE: NO JOINT DETECTED"
                            banner_color = (0, 165, 255)
                            banner_frames_left = 45

                    elif key in (ord('s'), ord('S')):
                        print("[DATASET CAPTURE] Skipped current frame (no image saved).")
                        capture_banner = "SKIPPED FRAME"
                        banner_color = (180, 180, 180)
                        banner_frames_left = 30

                    elif key in (ord('q'), ord('Q')):
                        print("[INFO] Quit signal received.")
                        break
                except cv2.error as e:
                    print("[WARNING] OpenCV GUI window unavailable in this Python environment.")
                    print("[INFO] Continuing in headless logging mode...")
                    gui_available = False

        cap.release()
        if (args.debug or args.show_mask or args.show_crop) and gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if log_file:
        log_file.close()

    print("[INFO] JointGuard CV runner finished.")


if __name__ == "__main__":
    main()
