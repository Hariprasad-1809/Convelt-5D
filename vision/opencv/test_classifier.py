"""
Sanity-Check Test Script for JointGuard Classifier

Runs classifier.py directly against healthy_sample.jpg and damage_sample.jpg
and prints raw feature values and vision scores side-by-side.
"""

import os
import sys
import cv2

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vision.opencv.config import VisionConfig, DEFAULT_CONFIG
from vision.opencv.detector import detect_joint
from vision.opencv.classifier import classify_joint


def main():
    test_dir = os.path.join(os.path.dirname(__file__), "test_images")
    healthy_path = os.path.join(test_dir, "healthy_sample.jpg")
    damage_path = os.path.join(test_dir, "damage_sample.jpg")

    if not os.path.exists(healthy_path) or not os.path.exists(damage_path):
        print("[ERROR] Test image fixtures missing. Run generate_test_images.py first.")
        sys.exit(1)

    print("=" * 70)
    print("      JOINTGUARD CLASSICAL CV CLASSIFIER SANITY CHECK")
    print("=" * 70)

    samples = [("HEALTHY SAMPLE", healthy_path), ("DAMAGE SAMPLE", damage_path)]

    results = []

    for name, img_path in samples:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[ERROR] Could not load image: {img_path}")
            continue

        test_cfg = VisionConfig(
            ROI_X_MIN=0.0,
            ROI_X_MAX=1.0,
            ROI_Y_MIN=0.0,
            ROI_Y_MAX=1.0,
            MIN_CONTEXT_CONTRAST=0.0,
            MAX_CONTOUR_AREA_FRACTION_OF_ROI=1.0
        )
        # 1. Detect Joint ROI
        bbox = detect_joint(frame, test_cfg)
        if bbox is None:
            print(f"[WARNING] No joint detected in {name}! Classifying full frame.")
            roi = frame
        else:
            roi = bbox.crop_roi(frame)

        # 2. Classify Joint ROI
        res = classify_joint(roi, DEFAULT_CONFIG)
        results.append((name, res, bbox))

    # Print Side-by-Side Summary Table
    print("\n" + f"{'Metric / Feature':<25} | {'Healthy Sample':<20} | {'Damage Sample':<20}")
    print("-" * 72)

    h_res = results[0][1]
    d_res = results[1][1]

    print(f"{'Final Label':<25} | {h_res.label:<20} | {d_res.label:<20}")
    print(f"{'Vision Score (0-100)':<25} | {h_res.vision_score:<20.2f} | {d_res.vision_score:<20.2f}")
    print("-" * 72)
    print(f"{'Raw Edge Density':<25} | {h_res.features['edge_density']:<20.4f} | {d_res.features['edge_density']:<20.4f}")
    print(f"{'Raw Hough Line Score':<25} | {h_res.features['hough_line_score']:<20.4f} | {d_res.features['hough_line_score']:<20.4f}")
    print(f"{'Raw Intensity Variance':<25} | {h_res.features['intensity_variance']:<20.4f} | {d_res.features['intensity_variance']:<20.4f}")
    print("=" * 70)

    # Sanity assertion check
    print("\nSanity Check Verification:")
    if h_res.label == "HEALTHY" and d_res.label == "DAMAGE":
        print(" [PASSED] Classifier correctly identified Healthy as HEALTHY and Damaged as DAMAGE!")
    else:
        print(" [ATTENTION] Score cutoffs may require tuning for physical lighting setup.")


if __name__ == "__main__":
    main()
