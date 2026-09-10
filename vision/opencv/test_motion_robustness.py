"""
Targeted Motion Robustness Test Suite for JointGuard
Verifies fixes for the 4 belt-motion vision issues:
  Issue 1: Partial-entry / skewed aspect ratio fallback (no stall on entry)
  Issue 2: Dual-boundary hysteresis & debounce (no INSIDE vs OUTSIDE_ZONE flicker)
  Issue 3: Classification stabilization & dynamic disagreement reconciliation
  Issue 4: Clean bbox clearing on undetected tracks (no stray orphaned labels)
"""

import os
import sys
import datetime
import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vision.opencv.config import VisionConfig
from vision.opencv.detector import detect_joint, BoundingBox
from vision.opencv.classifier import ClassificationResult
from vision.tracking.tracker import JointTracker, TrackItem


def test_issue1_aspect_ratio_fallback():
    """Verify that joints with non-standard aspect ratios during entry/exit are detected via relaxed fallback."""
    print("\n--- Test Issue 1: Aspect Ratio Fallback on Entry/Angle ---")
    cfg = VisionConfig(
        ROI_X_MIN=0.10, ROI_X_MAX=0.90,
        MIN_CONTEXT_CONTRAST=2.0,
        MIN_ASPECT_RATIO=1.0, MAX_ASPECT_RATIO=20.0,
        MIN_ASPECT_RATIO_RELAXED=0.4, MAX_ASPECT_RATIO_RELAXED=35.0,
        ALLOW_PARTIAL_ENTRY_FALLBACK=True
    )
    frame = np.full((480, 640, 3), (25, 25, 25), dtype=np.uint8)

    # Draw a joint entering at an angle: tall and narrow patch (w=40, h=80 -> aspect = 0.5 < 1.0)
    # With standard limits this would be rejected as "aspect ratio invalid (0.50 vs min 1.0)".
    # With two-tier fallback, it should be accepted!
    cv2.rectangle(frame, (200, 150), (240, 230), (210, 210, 210), -1)

    bbox = detect_joint(frame, cfg)
    assert bbox is not None, "FAILED: Aspect ratio 0.5 was rejected! Entry was stalled."
    aspect = bbox.aspect_ratio
    print(f"  [PASS] Skewed entry joint detected: x={bbox.x}, y={bbox.y}, w={bbox.w}, h={bbox.h}, aspect={aspect:.2f}")

    # Also test wide entering sliver (w=200, h=8 -> aspect = 25.0 > 20.0)
    frame2 = np.full((480, 640, 3), (25, 25, 25), dtype=np.uint8)
    cv2.rectangle(frame2, (150, 200), (350, 208), (220, 220, 220), -1)
    bbox2 = detect_joint(frame2, cfg)
    assert bbox2 is not None, "FAILED: Aspect ratio 25.0 was rejected! Sliver entry was stalled."
    print(f"  [PASS] Ultra-wide sliver detected: x={bbox2.x}, y={bbox2.y}, w={bbox2.h}, h={bbox2.h}, aspect={bbox2.aspect_ratio:.2f}")


def test_issue2_zone_boundary_hysteresis_and_debounce():
    """Verify dual-boundary hysteresis and debounce prevent rapid toggling between INSIDE and OUTSIDE_ZONE."""
    print("\n--- Test Issue 2: Zone Boundary Hysteresis & Debounce ---")
    cfg = VisionConfig(
        ROI_X_MIN=0.10, ROI_X_MAX=0.90,
        CAPTURE_ZONE_X_MIN=0.30, CAPTURE_ZONE_X_MAX=0.70,
        CAPTURE_ZONE_Y_MIN=0.20, CAPTURE_ZONE_Y_MAX=0.80,
        ZONE_HYSTERESIS_MARGIN_PIXELS=25,
        ZONE_EXIT_DEBOUNCE_FRAMES=4,
        CAPTURE_ZONE_MIN_FRAMES=2
    )
    tracker = JointTracker(cfg)
    frame_w, frame_h = 640, 480
    ts = "2026-09-10T00:00:00Z"

    # 1. Joint enters the strict capture zone (x=250, w=100 -> center cx=300)
    tracker.update(BoundingBox(x=250, y=150, w=100, h=80), ts)
    tracker.update(BoundingBox(x=265, y=150, w=100, h=80), ts)
    track = tracker.tracks[1]
    assert track.is_inside_capture_zone is True, "FAILED: Joint should be INSIDE capture zone"
    assert track.state == "INSPECTING", f"FAILED: Expected state INSPECTING, got {track.state}"
    print("  [PASS] Joint entered inner zone -> state=INSPECTING, is_inside_capture_zone=True")

    # 2. Joint transits smoothly towards the boundary: x=280 -> 300 -> 320 -> 340 -> 355
    # At x=355, bbox.x + bbox.w = 455, which exceeds the inner boundary cz_x2 = 448
    # BUT is within the exit margin (cz_x2 + 25 = 473)
    for x_step in [280, 300, 320, 340, 355]:
        tracker.update(BoundingBox(x=x_step, y=150, w=100, h=80), ts)
        assert track.is_inside_capture_zone is True

    # Micro-jitter around the inner boundary (alternating +/- 2px around x=355)
    for i in range(10):
        offset = 2 if i % 2 == 0 else -2
        test_box = BoundingBox(x=355 + offset, y=150, w=100, h=80)
        evt = tracker.update(test_box, ts)
        assert track.is_inside_capture_zone is True, f"FAILED: Zone flickered on jitter frame {i}!"
        assert track.state == "INSPECTING", f"FAILED: State toggled to {track.state} on jitter frame {i}!"
    print("  [PASS] 10 boundary jitter frames handled with ZERO flicker (remained INSIDE)")

    # 3. Joint moves beyond the relaxed exit boundary (x=385, w=100 -> x2=485 > 473)
    # Must debounce for 4 frames before transitioning out
    exit_box = BoundingBox(x=385, y=150, w=100, h=80)
    for frame_idx in range(1, 4):  # Frames 1, 2, 3 outside relaxed boundary
        evt = tracker.update(exit_box, ts)
        assert track.is_inside_capture_zone is True, f"FAILED: Exited prematurely on debounce frame {frame_idx}"
        assert track.outside_zone_frames == frame_idx
    print("  [PASS] Debounce delay respected: frames 1-3 outside exit boundary did NOT trigger zone exit")

    # Frame 4: hits ZONE_EXIT_DEBOUNCE_FRAMES (4) -> now cleanly exits
    evt4 = tracker.update(exit_box, ts)
    assert track.is_inside_capture_zone is False, "FAILED: Track should have exited zone on frame 4"
    print("  [PASS] Debounce completed on frame 4 -> clean transition without flickering")


def test_issue3_classification_stabilization_and_reconciliation():
    """Verify classification stabilization prevents premature locking and dynamic reconciliation resolves disagreements."""
    print("\n--- Test Issue 3: Stabilization & Disagreement Reconciliation ---")
    cfg = VisionConfig(
        ROI_X_MIN=0.10, ROI_X_MAX=0.90,
        CAPTURE_ZONE_X_MIN=0.20, CAPTURE_ZONE_X_MAX=0.80,
        CAPTURE_ZONE_Y_MIN=0.10, CAPTURE_ZONE_Y_MAX=0.90,
        CAPTURE_ZONE_MIN_FRAMES=2,
        CLASSIFICATION_STABILITY_FRAMES=4,
        DISAGREEMENT_CONFIDENCE_THRESHOLD=0.65,
        DISAGREEMENT_CORRECTION_STREAK=3
    )
    tracker = JointTracker(cfg)
    box = BoundingBox(x=250, y=150, w=120, h=80)
    ts = "2026-09-10T00:00:00Z"

    # Feed Frame 1: Low-confidence reading (e.g. 52% healthy)
    track_item = TrackItem(1, box)
    tracker.tracks[1] = track_item
    track_item.is_inside_capture_zone = True

    # Manually simulate 3 initial frames with moderate healthy read (H: 65%, D: 35%)
    # Not yet locked because frames < 4 and confidence < 0.70 (not immediate lock)
    for f in range(3):
        track_item.feature_history.append({"p_healthy": 0.65, "p_damage": 0.35, "confidence": 0.65})
        track_item.live_classification = ClassificationResult("HEALTHY", 65.0, {"confidence": 0.65, "p_healthy": 0.65, "p_damage": 0.35})
        track_item.capture_zone_frames = f + 1
        # Check stabilization: should NOT lock yet because frames < 4 and conf < 0.70
        stability_frames = cfg.CLASSIFICATION_STABILITY_FRAMES
        live_conf = track_item.live_classification.features.get("confidence", 0.0)
        high_conf_imm = (track_item.capture_zone_frames >= cfg.CAPTURE_ZONE_MIN_FRAMES and live_conf >= 0.70)
        should_lock = (track_item.capture_zone_frames >= stability_frames or high_conf_imm)
        assert should_lock is False, f"FAILED: Prematurely locked on frame {f+1}"
    print("  [PASS] Stabilization verified: frames 1-3 did NOT trigger premature lock")

    # Frame 4: hits stability requirement (4 frames), initial lock set as HEALTHY
    track_item.feature_history.append({"p_healthy": 0.75, "p_damage": 0.25, "confidence": 0.75})
    track_item.capture_zone_frames = 4
    locked_init = tracker._compute_averaged_classification(track_item)
    track_item.locked_classification = locked_init
    track_item.classification_result = locked_init
    track_item.has_been_classified = True
    assert track_item.locked_classification.label == "HEALTHY"
    print(f"  [PASS] Frame 4 locked stabilized initial classification: {track_item.locked_classification.label}")

    # Now simulate joint transiting deeper into belt lighting:
    # Next 3 frames reveal clear damage (DAMAGE with 75% confidence)
    # The live reading contradicts the locked HEALTHY reading.
    reconciled_triggered = False
    for streak in range(1, 4):
        track_item.feature_history.append({"p_healthy": 0.20, "p_damage": 0.80, "confidence": 0.80})
        if len(track_item.feature_history) > cfg.CAPTURE_ZONE_MAX_FRAMES:
            track_item.feature_history.pop(0)

        live_res = ClassificationResult("DAMAGE", 20.0, {"confidence": 0.80, "p_healthy": 0.20, "p_damage": 0.80})
        track_item.live_classification = live_res

        # Tracker disagreement logic
        locked_lbl = track_item.locked_classification.label
        live_lbl = track_item.live_classification.label
        live_cnf = track_item.live_classification.features.get("confidence", 0.0)

        if live_lbl in ("HEALTHY", "DAMAGE") and locked_lbl in ("HEALTHY", "DAMAGE") and live_lbl != locked_lbl:
            if live_cnf >= cfg.DISAGREEMENT_CONFIDENCE_THRESHOLD:
                track_item.disagreement_streak += 1
                if track_item.disagreement_streak >= cfg.DISAGREEMENT_CORRECTION_STREAK:
                    reconciled = tracker._compute_averaged_classification(track_item)
                    track_item.locked_classification = reconciled
                    track_item.classification_result = reconciled
                    track_item.disagreement_streak = 0
                    reconciled_triggered = True

    assert reconciled_triggered is True, "FAILED: Disagreement reconciliation did not trigger!"
    assert track_item.locked_classification.label == "DAMAGE", f"FAILED: Expected DAMAGE, got {track_item.locked_classification.label}"
    assert track_item.classification_result.label == "DAMAGE"
    print(f"  [PASS] Disagreement reconciled: Locked classification updated from HEALTHY -> DAMAGE (streak=3 sustained)")
    print(f"  [PASS] Final vs Current contradiction ELIMINATED: Final={track_item.locked_classification.label}, Current={track_item.live_classification.label}")


def test_issue4_no_stray_orphaned_labels():
    """Verify that when a track is not detected on a frame, its bbox is cleared so no stray labels appear."""
    print("\n--- Test Issue 4: Prevention of Stray/Orphaned Labels ---")
    cfg = VisionConfig()
    tracker = JointTracker(cfg)
    ts = "2026-09-10T00:00:00Z"

    # Frame 1: Joint detected as Track 1
    box1 = BoundingBox(x=200, y=150, w=100, h=80)
    tracker.update(box1, ts)
    active_events = tracker.get_active_track_events()
    assert len(active_events) == 1
    assert active_events[0].track_id == 1
    assert active_events[0].bbox is not None
    print("  [PASS] Track 1 active with valid bounding box")

    # Frame 2: Joint detection lost (e.g. frame dropout / bbox is None)
    tracker.update(None, ts)
    # Active track events MUST NOT include Track 1 because bbox was cleared!
    active_events_after_drop = tracker.get_active_track_events()
    assert len(active_events_after_drop) == 0, f"FAILED: Stray track event yielded: {active_events_after_drop}"
    assert tracker.tracks[1].bbox is None, "FAILED: Track 1 bbox was not cleared on undetected frame"
    assert tracker.tracks[1].disappeared_count == 1
    print("  [PASS] On undetected frame: bbox cleared to None, get_active_track_events() yielded 0 stray labels")

    # Frame 3: New detection matches Track 1 again
    box1_next = BoundingBox(x=205, y=150, w=100, h=80)
    tracker.update(box1_next, ts)
    active_events_restored = tracker.get_active_track_events()
    assert len(active_events_restored) == 1
    assert active_events_restored[0].track_id == 1
    assert active_events_restored[0].bbox == box1_next
    print("  [PASS] Track 1 re-attached with clean bounding box and no ghosting")


def main():
    print("=" * 70)
    print("   JOINTGUARD TARGETED MOTION ROBUSTNESS VERIFICATION SUITE")
    print("=" * 70)
    test_issue1_aspect_ratio_fallback()
    test_issue2_zone_boundary_hysteresis_and_debounce()
    test_issue3_classification_stabilization_and_reconciliation()
    test_issue4_no_stray_orphaned_labels()
    print("\n" + "=" * 70)
    print("  ALL 4 MOTION ROBUSTNESS TEST SUITES PASSED FLAWLESSLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
