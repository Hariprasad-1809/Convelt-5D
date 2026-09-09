"""
JointGuard Trained Classifier Verification Script

Loads the trained machine learning model, evaluates it against healthy and damaged
reference images, asserts classification correctness and confidence levels, and verifies
pipeline integration.

Usage:
    python vision/opencv/test_trained_classifier.py
"""

import os
import sys
import cv2
import numpy as np

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vision.opencv.config import VisionConfig, DEFAULT_CONFIG
from vision.opencv.detector import detect_joint
from vision.opencv.classifier import classify_joint, load_trained_model


def test_classifier_predictions():
    print("=" * 72)
    print("        JOINTGUARD TRAINED CLASSIFIER DIRECT VERIFICATION")
    print("=" * 72)

    cfg = DEFAULT_CONFIG
    loaded = load_trained_model(cfg.MODEL_PATH)
    assert loaded, f"Failed to load trained model from: {cfg.MODEL_PATH}"

    all_passed = True

    # --- Test Suite 1: Direct Image Evaluation (Bypassing Detector) ---
    print("\n[PART 1] Direct Classifier Evaluation (Bypassing Detector on dataset/)")
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    ds_healthy = os.path.join(dataset_dir, "healthy", "healthy_001.jpg")
    ds_damage = os.path.join(dataset_dir, "damage", "damage_001.jpg")

    direct_samples = [
        ("DATASET HEALTHY DIRECT CROP", ds_healthy, (129, 508, 174, 880), "HEALTHY"),
        ("DATASET DAMAGE DIRECT CROP", ds_damage, (312, 703, 174, 880), "DAMAGE")
    ]

    for title, img_path, (y1, y2, x1, x2), expected_label in direct_samples:
        if not os.path.exists(img_path):
            print(f"  [SKIP] File not found: {img_path}")
            continue
        frame = cv2.imread(img_path)
        assert frame is not None, f"Failed to read image at: {img_path}"

        # Direct joint crop bypassing detector & webcam
        joint_crop = frame[y1:y2, x1:x2]
        res = classify_joint(joint_crop, cfg)
        conf = res.features.get("confidence", 0.0)
        p_healthy = res.features.get("p_healthy", 0.0)
        p_damage = res.features.get("p_damage", 0.0)

        print(f"\n--- {title} ---")
        print(f"  File Path        : {os.path.basename(img_path)}")
        print(f"  Image Shape      : {frame.shape}")
        print(f"  Predicted Label  : {res.label} (Expected: {expected_label})")
        print(f"  Display Label    : {res.features.get('display_label', res.label)}")
        print(f"  Confidence       : {conf * 100.0:.2f}%")
        print(f"  Vision Score     : {res.vision_score:.2f}")
        print(f"  P(HEALTHY)       : {p_healthy:.4f} | P(DAMAGE): {p_damage:.4f}")

        is_correct = (res.label == expected_label)
        has_high_conf = (conf >= 0.60)
        if is_correct and has_high_conf:
            print(f"  --> [PASS] Correct classification with high confidence!")
        else:
            print(f"  --> [FAIL] Mismatched or low confidence decision!")
            all_passed = False

    # --- Test Suite 2: End-to-End Detect + Crop + Classify ---
    print("\n[PART 2] End-to-End Detector + Crop + Classifier Pipeline (test_images/)")
    test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
    healthy_path = os.path.join(test_images_dir, "healthy_sample.jpg")
    damage_path = os.path.join(test_images_dir, "damage_sample.jpg")

    samples = [
        ("HEALTHY REFERENCE", healthy_path, "HEALTHY"),
        ("DAMAGED REFERENCE", damage_path, "DAMAGE")
    ]

    for title, img_path, expected_label in samples:
        frame = cv2.imread(img_path)
        assert frame is not None, f"Failed to read image at: {img_path}"

        # 1. Detect Joint
        bbox = detect_joint(frame, cfg)
        assert bbox is not None, f"Joint detection failed for: {title}"

        # 2. Crop Joint
        crop = bbox.crop_roi(frame)
        assert crop.size > 0, f"Joint crop is empty for: {title}"

        # 3. Classify with Trained ML Model
        res = classify_joint(crop, cfg)

        conf = res.features.get("confidence", 0.0)
        p_healthy = res.features.get("p_healthy", 0.0)
        p_damage = res.features.get("p_damage", 0.0)

        print(f"\n--- {title} ---")
        print(f"  File Path        : {os.path.basename(img_path)}")
        print(f"  Detected BBox    : (x={bbox.x}, y={bbox.y}, w={bbox.w}, h={bbox.h}), Aspect={bbox.aspect_ratio:.2f}")
        print(f"  Predicted Label  : {res.label} (Expected: {expected_label})")
        print(f"  Display Label    : {res.features.get('display_label', res.label)}")
        print(f"  Confidence       : {conf * 100.0:.2f}%")
        print(f"  Vision Score     : {res.vision_score:.2f}")
        print(f"  P(HEALTHY)       : {p_healthy:.4f} | P(DAMAGE): {p_damage:.4f}")
        print(f"  Key Features     : surface_roughness={res.features.get('surface_roughness')}, edge_density={res.features.get('edge_density')}")

        is_correct = (res.label == expected_label)
        has_high_conf = (conf >= 0.60)

        if is_correct and has_high_conf:
            print(f"  --> [PASS] Correct classification with high confidence!")
        else:
            print(f"  --> [FAIL] Mismatched or low confidence decision!")
            all_passed = False

    print("\n" + "=" * 72)
    if all_passed:
        print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("  Trained model reliably distinguishes Healthy and Damaged joint images.")
    else:
        print("  ONE OR MORE TESTS FAILED!")
    print("=" * 72)

    return all_passed


if __name__ == "__main__":
    success = test_classifier_predictions()
    if not success:
        sys.exit(1)
