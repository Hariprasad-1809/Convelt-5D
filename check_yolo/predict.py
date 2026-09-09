#!/usr/bin/env python3
"""
JointGuard YOLO Classification - Inference Prediction Script

Runs standalone YOLO image classification on a single conveyor-belt joint image.
Outputs human-readable classification: HEALTHY, DAMAGE, or UNCERTAIN.

Usage:
    python check_yolo/predict.py path/to/image.jpg
    python check_yolo/predict.py path/to/image.jpg --show
    python check_yolo/predict.py path/to/image.jpg --model check_yolo/models/joint_yolo_classifier.pt
"""

import argparse
import os
import sys
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_V4_PATH = os.path.join(SCRIPT_DIR, "models", "joint_yolo_classifier_v4.pt")
DEFAULT_MODEL_V2_PATH = os.path.join(SCRIPT_DIR, "models", "joint_yolo_classifier_v2.pt")
DEFAULT_MODEL_V1_PATH = os.path.join(SCRIPT_DIR, "models", "joint_yolo_classifier.pt")
DEFAULT_MODEL_PATH = DEFAULT_MODEL_V4_PATH if os.path.isfile(DEFAULT_MODEL_V4_PATH) else (DEFAULT_MODEL_V2_PATH if os.path.isfile(DEFAULT_MODEL_V2_PATH) else DEFAULT_MODEL_V1_PATH)
FALLBACK_MODEL_PATH = os.path.join(SCRIPT_DIR, "results", "joint_cls_v4", "weights", "best.pt")

# Confidence threshold below which a prediction is flagged UNCERTAIN (Step 9)
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict conveyor joint condition (HEALTHY vs DAMAGE) using YOLO classification"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file for classification",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Explicit path to trained YOLO classification model weights (overrides version flags)",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        choices=["v1", "v2", "v4"],
        default="v4",
        help="Model version to use: 'v4' (default), 'v2', or 'v1'",
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="Shortcut to use v1 model",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Shortcut to use v2 model",
    )
    parser.add_argument(
        "--v4",
        action="store_true",
        help="Shortcut to use v4 model (default)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for certainty (default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display preview window with classification overlay",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable automatic joint ROI extraction during inference",
    )
    return parser.parse_args()


def extract_joint_roi(img: np.ndarray) -> np.ndarray:
    """
    Extracts the central belt joint region, eliminating background walls
    and outer conveyor rig frames to focus classification strictly on the joint.
    Matches the preprocessing in prepare_dataset.py.
    """
    h, w = img.shape[:2]

    # If the image is already a tight crop, return as is
    if w / max(1, h) >= 2.0 or h < 350:
        return img

    x_min = int(w * 0.08)
    x_max = int(w * 0.92)
    belt = img[:, x_min:x_max]

    gray = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_energy = np.mean(np.abs(sobel_y), axis=1) + 0.5 * np.mean(gray, axis=1)

    thresh_val = np.percentile(row_energy, 65)
    joint_rows = np.where(row_energy > thresh_val)[0]

    if len(joint_rows) > 0:
        y_min = max(0, int(joint_rows.min() - 0.05 * h))
        y_max = min(h, int(joint_rows.max() + 0.05 * h))

        if (y_max - y_min) < int(h * 0.30):
            center_y = int((y_min + y_max) / 2)
            half = int(h * 0.20)
            y_min = max(0, center_y - half)
            y_max = min(h, center_y + half)
    else:
        y_min = int(h * 0.15)
        y_max = int(h * 0.85)

    return img[y_min:y_max, x_min:x_max]


def resolve_model_path(requested_path: Optional[str] = None, version: str = "v4") -> str:
    """Finds the model weights file or falls back to latest results."""
    if requested_path:
        if os.path.isfile(requested_path):
            return os.path.abspath(requested_path)
        raise FileNotFoundError(f"Model weights not found at '{requested_path}'.")

    if version == "v4":
        if os.path.isfile(DEFAULT_MODEL_V4_PATH):
            return os.path.abspath(DEFAULT_MODEL_V4_PATH)
        if os.path.isfile(FALLBACK_MODEL_PATH):
            return os.path.abspath(FALLBACK_MODEL_PATH)
        if os.path.isfile(DEFAULT_MODEL_V2_PATH):
            return os.path.abspath(DEFAULT_MODEL_V2_PATH)
        if os.path.isfile(DEFAULT_MODEL_V1_PATH):
            return os.path.abspath(DEFAULT_MODEL_V1_PATH)

    if version == "v2":
        if os.path.isfile(DEFAULT_MODEL_V2_PATH):
            return os.path.abspath(DEFAULT_MODEL_V2_PATH)
        fallback_v2 = os.path.join(SCRIPT_DIR, "results", "joint_cls_v2", "weights", "best.pt")
        if os.path.isfile(fallback_v2):
            return os.path.abspath(fallback_v2)
        if os.path.isfile(DEFAULT_MODEL_V1_PATH):
            return os.path.abspath(DEFAULT_MODEL_V1_PATH)

    if version == "v1":
        if os.path.isfile(DEFAULT_MODEL_V1_PATH):
            return os.path.abspath(DEFAULT_MODEL_V1_PATH)
        fallback_v1 = os.path.join(SCRIPT_DIR, "results", "joint_cls", "weights", "best.pt")
        if os.path.isfile(fallback_v1):
            return os.path.abspath(fallback_v1)

    # Look for any best.pt under results
    results_dir = os.path.join(SCRIPT_DIR, "results")
    if os.path.isdir(results_dir):
        for root, _, files in os.walk(results_dir):
            if "best.pt" in files:
                return os.path.abspath(os.path.join(root, "best.pt"))
    raise FileNotFoundError("Model weights not found. Please run train.py first.")


def predict(
    image_path: str,
    model_path: Optional[str] = None,
    conf_thresh: float = DEFAULT_CONFIDENCE_THRESHOLD,
    apply_crop: bool = True,
    show: bool = False,
    version: str = "v4",
) -> Dict[str, Any]:
    """
    Performs inference on an image and returns a dictionary with predictions.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_file = resolve_model_path(model_path, version=version)

    # Lazy import to avoid startup delay
    from ultralytics import YOLO

    model = YOLO(model_file)

    img_raw = cv2.imread(image_path)
    if img_raw is None:
        raise ValueError(f"Could not decode image at: {image_path}")

    img_input = extract_joint_roi(img_raw) if apply_crop else img_raw

    # Run inference
    results = model(img_input, verbose=False)
    res = results[0]

    # Parse classification probabilities
    top1_idx = int(res.probs.top1)
    raw_class_name = str(res.names[top1_idx]).strip().lower()
    raw_conf = float(res.probs.top1conf)

    # Human-readable mapping
    if raw_class_name == "healthy":
        human_class = "HEALTHY"
        class_id = 0
    elif raw_class_name == "damage":
        human_class = "DAMAGE"
        class_id = 1
    else:
        human_class = raw_class_name.upper()
        class_id = top1_idx

    # Uncertainty handling (Step 9)
    is_uncertain = raw_conf < conf_thresh
    display_prediction = "UNCERTAIN" if is_uncertain else human_class

    result_dict = {
        "image_file": os.path.basename(image_path),
        "image_path": image_path,
        "prediction": display_prediction,
        "raw_class": human_class,
        "class_id": class_id,
        "confidence": raw_conf * 100.0,
        "is_uncertain": is_uncertain,
        "threshold": conf_thresh * 100.0,
    }

    # Optional visual preview (Step 8)
    if show:
        display_img = img_raw.copy()
        h, w = display_img.shape[:2]

        color = (0, 200, 0) if display_prediction == "HEALTHY" else (
            (0, 0, 230) if display_prediction == "DAMAGE" else (0, 165, 255)
        )

        header_text = f"{display_prediction} ({raw_conf * 100.0:.1f}%)"
        cv2.rectangle(display_img, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(
            display_img,
            header_text,
            (15, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

        sub_text = f"File: {os.path.basename(image_path)} | Thresh: {conf_thresh*100.0:.0f}%"
        cv2.putText(
            display_img,
            sub_text,
            (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("JointGuard YOLO Classification Preview", display_img)
        print("[PREVIEW] Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_dict


def main():
    args = parse_args()

    req_version = "v1" if args.v1 else ("v2" if args.v2 else ("v4" if args.v4 else args.model_version))
    try:
        res = predict(
            image_path=args.image_path,
            model_path=args.model,
            conf_thresh=args.conf_thresh,
            apply_crop=not args.no_crop,
            show=args.show,
            version=req_version,
        )

        # Output format matching Step 6
        print(f"Image: {res['image_file']}")
        print(f"Prediction: {res['prediction']}")
        print(f"Confidence: {res['confidence']:.1f}%")

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
