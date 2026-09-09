#!/usr/bin/env python3
"""
JointGuard YOLO Classification - Automated Test Script

Tests the trained YOLO model against the two real reference images:
- Healthy: healthy_001.jpg
- Damaged: damage_001.jpg

Prints formatted test results with model probabilities.
Does NOT hardcode any predictions; evaluates live model inference.

Usage:
    python check_yolo/test_model.py
    python check_yolo/test_model.py --model check_yolo/models/joint_yolo_classifier.pt
"""

import argparse
import os
import sys

# Add check_yolo to path for predict helper
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from predict import predict, DEFAULT_MODEL_PATH, resolve_model_path


def find_reference_images():
    """Locates the reference healthy and damaged images."""
    # Priority 1: check_yolo/dataset/train/
    p_train_h = os.path.join(SCRIPT_DIR, "dataset", "train", "healthy", "healthy_001.jpg")
    p_train_d = os.path.join(SCRIPT_DIR, "dataset", "train", "damage", "damage_001.jpg")

    # Priority 2: original vision/opencv/dataset/
    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    p_orig_h = os.path.join(project_root, "jointgurd_proto", "vision", "opencv", "dataset", "healthy", "healthy_001.jpg")
    p_orig_d = os.path.join(project_root, "jointgurd_proto", "vision", "opencv", "dataset", "damage", "damage_001.jpg")

    h_path = p_train_h if os.path.isfile(p_train_h) else p_orig_h
    d_path = p_train_d if os.path.isfile(p_train_d) else p_orig_d

    return h_path, d_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run automated test of YOLO classifier on reference images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained YOLO classification model weights",
    )
    parser.add_argument(
        "--healthy-image",
        type=str,
        default=None,
        help="Custom path to healthy reference image",
    )
    parser.add_argument(
        "--damage-image",
        type=str,
        default=None,
        help="Custom path to damaged reference image",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    default_h, default_d = find_reference_images()
    h_image = args.healthy_image or default_h
    d_image = args.damage_image or default_d

    if not os.path.isfile(h_image):
        print(f"[ERROR] Healthy reference image not found: {h_image}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(d_image):
        print(f"[ERROR] Damaged reference image not found: {d_image}", file=sys.stderr)
        sys.exit(1)

    try:
        model_file = resolve_model_path(args.model)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Perform actual model inference (NO hardcoding)
    res_healthy = predict(h_image, model_path=model_file)
    res_damage = predict(d_image, model_path=model_file)

    # Required output format (Step 7)
    print("================================")
    print("JOINTGUARD YOLO TEST")
    print("================================")
    print("")
    print("Healthy image:")
    print(f"Prediction: {res_healthy['prediction']}")
    print(f"Confidence: {res_healthy['confidence']:.1f}%")
    print("")
    print("Damaged image:")
    print(f"Prediction: {res_damage['prediction']}")
    print(f"Confidence: {res_damage['confidence']:.1f}%")
    print("")
    print("================================")

    # Verification status
    healthy_ok = res_healthy["prediction"] == "HEALTHY"
    damage_ok = res_damage["prediction"] == "DAMAGE"

    if healthy_ok and damage_ok:
        print("[STATUS] Both reference images classified correctly.")
    else:
        print(f"[STATUS] Mismatches detected: Healthy={healthy_ok}, Damage={damage_ok}")


if __name__ == "__main__":
    main()
