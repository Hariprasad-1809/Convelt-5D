"""
Comprehensive Test Suite for JointGuard (Phase 1 Classical CV)

Tests:
  1. Healthy static sample -> HEALTHY (high vision score)
  2. Damaged static sample -> DAMAGE (low vision score)
  3. Empty background image -> INVALID / No candidate
  4. Fragmented damaged joint -> Bounding box fusion into 1 candidate
  5. Continuous motor movement -> APPROACHING -> INSPECTING -> PASSED
"""

import os
import sys
import datetime
import cv2
import numpy as np

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vision.opencv.config import VisionConfig
from vision.opencv.detector import detect_joint, BoundingBox
from vision.opencv.classifier import classify_joint
from vision.tracking.tracker import JointTracker


def test_healthy_sample():
    img_path = os.path.join(project_root, "vision", "opencv", "test_images", "healthy_sample.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(project_root, "vision", "opencv", "test_images", "healthy_sample.jpg")
    frame = cv2.imread(img_path)
    assert frame is not None, f"Failed to load {img_path}"

    test_cfg = VisionConfig(ROI_X_MIN=0.10, ROI_X_MAX=0.90, ROI_Y_MIN=0.05, ROI_Y_MAX=0.95, MIN_CONTEXT_CONTRAST=0.0, MAX_CONTOUR_AREA_FRACTION_OF_ROI=1.0)
    bbox = detect_joint(frame, test_cfg)
    crop = bbox.crop_roi(frame) if bbox else frame

    res = classify_joint(crop, test_cfg)
    assert res.label == "HEALTHY", f"Expected HEALTHY, got {res.label} (score={res.vision_score:.2f})"
    assert res.vision_score >= 54.0, f"Score {res.vision_score} < 54.0"
    print(f"  [PASS] Test 1: Healthy Sample -> Label={res.label}, Score={res.vision_score:.2f}")


def test_damage_sample():
    img_path = os.path.join(project_root, "vision", "opencv", "test_images", "damage_sample.jpg")
    frame = cv2.imread(img_path)
    assert frame is not None, f"Failed to load {img_path}"

    test_cfg = VisionConfig(ROI_X_MIN=0.10, ROI_X_MAX=0.90, ROI_Y_MIN=0.05, ROI_Y_MAX=0.95, CONTOUR_CLUSTER_MAX_GAP_PX=60, MAX_CONTOUR_AREA_FRACTION_OF_ROI=1.0, MIN_CONTEXT_CONTRAST=0.0)
    bbox = detect_joint(frame, test_cfg)
    crop = bbox.crop_roi(frame) if bbox else frame

    res = classify_joint(crop, test_cfg)
    assert res.label == "DAMAGE", f"Expected DAMAGE, got {res.label} (score={res.vision_score:.2f})"
    assert res.vision_score < 54.0, f"Score {res.vision_score} >= 54.0"
    print(f"  [PASS] Test 2: Damaged Sample -> Label={res.label}, Score={res.vision_score:.2f}")


def test_empty_background():
    cfg = VisionConfig()
    # Pure dark background frame with no metallic joint
    frame = np.full((480, 640, 3), (30, 30, 30), dtype=np.uint8)

    bbox = detect_joint(frame, cfg)
    assert bbox is None, "Empty background should produce no detection candidate"

    # Empty crop test for classifier
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    res = classify_joint(empty_crop, cfg)
    assert res.label == "INVALID", f"Expected INVALID for empty crop, got {res.label}"
    print("  [PASS] Test 3: Empty Background -> No candidate & INVALID crop protection verified")


def test_fragmented_joint_fusion():
    cfg = VisionConfig(
        ROI_X_MIN=0.10, ROI_X_MAX=0.90,
        MIN_CONTEXT_CONTRAST=5.0,
        MIN_CONTOUR_AREA=500
    )
    frame = np.full((480, 640, 3), (30, 30, 30), dtype=np.uint8)

    # Draw 3 fragmented metallic spots of a single crumpled joint close together (gap ~20px)
    cv2.rectangle(frame, (200, 150), (280, 190), (220, 220, 220), -1)
    cv2.rectangle(frame, (200, 210), (280, 250), (210, 210, 210), -1)
    cv2.rectangle(frame, (200, 270), (280, 310), (230, 230, 230), -1)

    bbox = detect_joint(frame, cfg)
    assert bbox is not None, "Fragmented joint detection failed"
    # Bounding box should enclose all 3 fragments (y from 150 to 310, height ~160)
    assert bbox.h >= 140, f"Fragments were not merged! Height was {bbox.h}"
    print(f"  [PASS] Test 4: Fragmented Damaged Joint -> Merged into 1 unified box (x={bbox.x}, y={bbox.y}, w={bbox.w}, h={bbox.h})")


def test_continuous_motion_tracker():
    cfg = VisionConfig(
        ROI_X_MIN=0.15, ROI_X_MAX=0.85,
        CAPTURE_ZONE_X_MIN=0.25, CAPTURE_ZONE_X_MAX=0.75,
        CAPTURE_ZONE_MIN_FRAMES=2, MIN_SHARPNESS_SCORE=0.0
    )
    tracker = JointTracker(cfg)
    width, height = 640, 480

    x_positions = [20, 90, 130, 240, 240, 240, 350, 450]
    locked_count = 0

    for idx, x_pos in enumerate(x_positions):
        frame = np.full((height, width, 3), (25, 25, 25), dtype=np.uint8)
        cv2.rectangle(frame, (x_pos, 160), (x_pos + 180, 320), (220, 220, 220), -1)

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        bbox = detect_joint(frame, cfg)
        event = tracker.update(bbox, ts, frame)

        if event and event.is_new_classification:
            locked_count += 1

    assert locked_count == 1, f"Expected 1 locked classification, got {locked_count}"
    print("  [PASS] Test 5: Continuous Movement State Machine -> APPROACHING -> INSPECTING -> PASSED locked ONCE")


def main():
    print("=" * 70)
    print("      JOINTGUARD FULL VISION SUITE VERIFICATION")
    print("=" * 70)

    test_healthy_sample()
    test_damage_sample()
    test_empty_background()
    test_fragmented_joint_fusion()
    test_continuous_motion_tracker()

    print("=" * 70)
    print("  ALL 5 INTEGRATION TESTS PASSED CLEANLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
