"""
Pipeline Simulation & Integration Test for JointGuard (Phase 1)

Simulates a joint patch moving across the camera field of view frame-by-frame
to test detection, state transitions (APPROACHING -> INSPECTING -> PASSED),
and single-shot classification locking.
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
from vision.opencv.detector import detect_joint
from vision.tracking.tracker import JointTracker


def run_sequence_test():
    print("=" * 70)
    print("      JOINTGUARD PIPELINE & TRACKER STATE MACHINE TEST")
    print("=" * 70)

    cfg = VisionConfig(
        ROI_X_MIN=0.15,
        ROI_X_MAX=0.85,
        CAPTURE_ZONE_X_MIN=0.25,
        CAPTURE_ZONE_X_MAX=0.75,
        CAPTURE_ZONE_MIN_FRAMES=2,
        MIN_SHARPNESS_SCORE=0.0
    )
    tracker = JointTracker(cfg)

    width, height = 640, 480
    bg_color = (25, 25, 25)

    # Joint size
    jw, jh = 200, 160
    jy = 160

    # Simulate joint moving horizontally from x = 10 to x = 500 (across ROI x: 160 to 480)
    # Step 1: Move into left edge (APPROACHING)
    # Step 2: Stop inside ROI for 5 frames (INSPECTING -> locked classification)
    # Step 3: Move out right edge (PASSED)

    x_positions = [
        20, 50, 90, 130,       # Entering (APPROACHING)
        240, 240, 240, 240, 240,# Inside ROI, stationary (INSPECTING)
        350, 450, 550          # Exiting frame (PASSED)
    ]

    events_recorded = []
    classification_trigger_count = 0

    for idx, x_pos in enumerate(x_positions):
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Draw a damaged joint foil patch at x_pos
        foil = np.full((jh, jw, 3), (220, 220, 220), dtype=np.uint8)
        cv2.line(foil, (20, 20), (180, 140), (20, 20, 20), 4)  # Fold crease
        cv2.line(foil, (22, 18), (182, 138), (255, 255, 255), 3)

        y1 = max(0, jy)
        y2 = min(height, jy + jh)
        x1 = max(0, x_pos)
        x2 = min(width, x_pos + jw)

        fw = x2 - x1
        fh = y2 - y1
        if fw > 0 and fh > 0:
            frame[y1:y2, x1:x2] = foil[:fh, :fw]

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        bbox = detect_joint(frame, cfg)
        event = tracker.update(bbox, ts, frame)

        if event:
            events_recorded.append((idx, event.state, event.is_new_classification, event.classification))
            if event.is_new_classification:
                classification_trigger_count += 1
                print(f"  Frame {idx:02d} | x={x_pos:3d} | State: {event.state:<12} | [CLASSIFIED ONCE] Label={event.classification.label}, Score={event.classification.vision_score:.2f}")
            else:
                print(f"  Frame {idx:02d} | x={x_pos:3d} | State: {event.state:<12} | (No re-classification)")

    print("-" * 70)
    print("Verification Results:")
    print(f"  Total frames simulated: {len(x_positions)}")
    print(f"  Classification locked count: {classification_trigger_count}")

    if classification_trigger_count == 1:
        print(" [PASSED] Single-shot classification verified! Joint classified exactly ONCE without double-counting.")
    else:
        print(f" [FAILED] Classification triggered {classification_trigger_count} times instead of 1.")

    print("=" * 70)


if __name__ == "__main__":
    run_sequence_test()
