"""
Targeted Verification Suite for Vision Fixes:
1. Recent-window sliding early-lock finalization for noisy starts (Issue 1)
2. Stale bounding box non-rendering after grace period (Issue 2)
3. Static background bottom-edge false-positive rejection (Issue 3)
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.services.vision_service import (
    JointGuardStateEngine,
    JointTrack,
    find_joint_in_roi,
    is_in_inner_inspection_zone,
)


class MockYOLO:
    """Mock YOLO classifier with programmable per-frame probability schedule."""
    def __init__(self, schedule_fn):
        self.schedule_fn = schedule_fn
        self.frame_idx = 0
        self.names = {0: "healthy", 1: "damage"}

    def __call__(self, crop, verbose=False):
        p_h, p_d = self.schedule_fn(self.frame_idx, crop)
        self.frame_idx += 1

        class Probs:
            def __init__(self, h, d):
                self.data = {0: h, 1: d}

        class Result:
            def __init__(self, names, probs):
                self.names = names
                self.probs = probs

        return [Result(self.names, Probs(p_h, p_d))]


def test_issue1_recent_window_early_lock_recovers_from_noisy_start():
    """
    ISSUE 1 REGRESSION TEST:
    A damaged joint starts with 15 noisy frames where p_damage is low (0.35).
    At frame 16-21 (6 frames), the joint becomes clearly visible with DAMAGE 79.4%.
    
    The full-lifetime average is only: (15*0.35 + 6*0.794) / 21 = 0.476 (< 0.70 threshold).
    Under the old logic, this would never early-lock and would wait until PASSED.
    
    Under the new sliding-window logic (last 5-8 frames), the recent-window average is 0.794 >= 0.70.
    The engine must early-lock DAMAGE with finalized_via == 'early-lock-recent-window'.
    """
    engine = JointGuardStateEngine(
        target_frames=5,
        damage_thresh=0.70,
        healthy_thresh=0.70,
        track_lost_timeout=2.0
    )
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 120

    def noisy_then_clear_schedule(idx, crop):
        if idx < 15:
            # 15 frames of noisy ambiguous entry (e.g. edge shadow/glare)
            return (0.65, 0.35)
        else:
            # Clearly visible crack
            return (0.206, 0.794)

    yolo = MockYOLO(noisy_then_clear_schedule)

    results = []
    # Feed 15 noisy frames
    for i in range(15):
        r = engine.process_frame(
            frame,
            joint_detected=True,
            raw_bbox_full=(150 + i * 5, 180, 120, 80),
            in_zone=True,
            yolo_model=yolo,
            no_quality_check=True
        )
        results.append(r)
        # Should not be early finalized yet (damage is low)
        assert r["is_early_finalized"] is False

    # Feed frames 16-21 (6 clear damage frames)
    for i in range(15, 21):
        r = engine.process_frame(
            frame,
            joint_detected=True,
            raw_bbox_full=(150 + i * 5, 180, 120, 80),
            in_zone=True,
            yolo_model=yolo,
            no_quality_check=True
        )
        results.append(r)

    last_res = results[-1]
    # Verify early finalization occurred via recent sliding window
    assert last_res["is_early_finalized"] is True, (
        f"Expected early finalization after 6 sustained damage frames (recent avg 79.4%), "
        f"got is_early_finalized={last_res['is_early_finalized']}"
    )
    assert last_res["final_label"] == "DAMAGE"
    assert last_res["finalized_via"] == "early-lock-recent-window"
    assert last_res["inspection_state"] == "CONFIRMED"
    assert last_res["final_confidence"] >= 0.70


def test_issue2_stale_box_not_rendered_after_grace_period():
    """
    ISSUE 2 REGRESSION TEST:
    A joint is tracked for 5 frames, then goes lost (e.g. leaves field of view).
    Grace period is 2 frames:
      - Lost frame 1: smoothed_bbox is retained (transient dropout buffer)
      - Lost frame 2: smoothed_bbox is retained (transient dropout buffer)
      - Lost frame 3+: smoothed_bbox MUST BE NONE (stale box eliminated)
    
    Verifies that for lost frames 3 to 20+, smoothed_bbox is strictly None,
    preventing any solid bounding box with [LOST] from lingering on screen.
    """
    engine = JointGuardStateEngine(
        target_frames=5,
        track_lost_timeout=2.0,
        lost_bbox_grace_frames=2
    )
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 120

    def dummy_yolo(idx, crop):
        return (0.90, 0.10)

    yolo = MockYOLO(dummy_yolo)

    # 5 frames of active joint
    for i in range(5):
        r = engine.process_frame(
            frame,
            joint_detected=True,
            raw_bbox_full=(100 + i * 10, 150, 100, 80),
            in_zone=True,
            yolo_model=yolo,
            no_quality_check=True
        )
        assert r["smoothed_bbox"] is not None

    # Frame 6: Lost frame 1 (consecutive_lost_frames == 1 <= 2) -> held in grace period
    r_lost1 = engine.process_frame(
        frame,
        joint_detected=False,
        raw_bbox_full=None,
        in_zone=False,
        yolo_model=yolo,
        no_quality_check=True
    )
    assert r_lost1["tracking_status"] == "TEMPORARILY_LOST"
    assert r_lost1["consecutive_lost_frames"] == 1
    assert r_lost1["smoothed_bbox"] is not None, "Expected grace period hold at frame 1"

    # Frame 7: Lost frame 2 (consecutive_lost_frames == 2 <= 2) -> held in grace period
    r_lost2 = engine.process_frame(
        frame,
        joint_detected=False,
        raw_bbox_full=None,
        in_zone=False,
        yolo_model=yolo,
        no_quality_check=True
    )
    assert r_lost2["consecutive_lost_frames"] == 2
    assert r_lost2["smoothed_bbox"] is not None, "Expected grace period hold at frame 2"

    # Frame 8: Lost frame 3 (consecutive_lost_frames == 3 > 2) -> MUST BE NONE
    r_lost3 = engine.process_frame(
        frame,
        joint_detected=False,
        raw_bbox_full=None,
        in_zone=False,
        yolo_model=yolo,
        no_quality_check=True
    )
    assert r_lost3["consecutive_lost_frames"] == 3
    assert r_lost3["smoothed_bbox"] is None, (
        f"STALE BBOX REGRESSION: Expected smoothed_bbox to be None at lost frame 3, got {r_lost3['smoothed_bbox']}"
    )

    # Frames 9 through 20: Joint remains lost -> smoothed_bbox must stay None for ALL frames
    for lost_idx in range(4, 21):
        r_subsequent = engine.process_frame(
            frame,
            joint_detected=False,
            raw_bbox_full=None,
            in_zone=False,
            yolo_model=yolo,
            no_quality_check=True
        )
        assert r_subsequent["smoothed_bbox"] is None, (
            f"STALE BBOX REGRESSION: Expected smoothed_bbox=None at lost frame {lost_idx}, "
            f"got {r_subsequent['smoothed_bbox']}"
        )


def test_issue3_static_background_bottom_edge_rejected():
    """
    ISSUE 3 REGRESSION TEST:
    Verifies that a static contrast edge along the bottom edge of the ROI (the boundary
    between the dark rubber belt and light floor/mount) is rejected as a candidate,
    while a real metallic joint centered on the conveyor belt is detected correctly.
    """
    # Create synthetic ROI crop (h=300, w=400):
    # Conveyor belt rubber is dark (V ~ 60)
    roi_dark_belt = np.ones((300, 400, 3), dtype=np.uint8) * 60

    # 1. Simulate static bright floor along bottom border (y >= 260 to 300)
    # Bright neutral color: V >= 200, S <= 30 (floor/mount behind belt)
    roi_with_floor_edge = roi_dark_belt.copy()
    roi_with_floor_edge[265:300, :] = 220  # Bright white/light floor at bottom

    detected_edge = find_joint_in_roi(roi_with_floor_edge, min_area=400, min_width=30)
    assert detected_edge is None, (
        f"FALSE POSITIVE REGRESSION: Static bottom background edge was accepted as a joint: {detected_edge}"
    )

    # 2. Simulate real metallic joint on the belt (centered at y=100..180, x=150..270)
    # Metallic joint: V >= 200, S <= 40, enclosed by dark rubber belt above and below
    roi_with_real_joint = roi_dark_belt.copy()
    # Fill metallic patch
    roi_with_real_joint[110:170, 140:260] = 210  # Metallic plate in center

    detected_joint = find_joint_in_roi(roi_with_real_joint, min_area=400, min_width=30)
    assert detected_joint is not None, "Real metallic joint on conveyor belt was incorrectly rejected!"
    jx, jy, jw, jh = detected_joint
    # Center should be roughly near (200, 140)
    assert 120 <= jx <= 160, f"Expected jx around 140, got {jx}"
    assert 90 <= jy <= 130, f"Expected jy around 110, got {jy}"
    assert jw >= 100
    assert jh >= 50
