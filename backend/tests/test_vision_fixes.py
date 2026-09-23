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
    Verifies that:
    1. A static contrast edge along the bottom edge of the ROI (the boundary between dark rubber belt
       and light floor/mount) is rejected as a false-positive candidate.
    2. Real metallic joints anywhere in the zone (centered, near bottom edge, or full vertical span)
       ARE detected correctly.
    3. A real joint that remains stationary across multiple consecutive frames maintains valid detection
       without dropping to NOT VISIBLE or losing its bounding box.
    """
    # Create synthetic ROI crop (h=300, w=400): Conveyor belt rubber is dark (V ~ 60)
    roi_dark_belt = np.ones((300, 400, 3), dtype=np.uint8) * 60

    # 1. Negative case: Static bright floor along bottom border (y >= 265 to 300, full width)
    roi_with_floor_edge = roi_dark_belt.copy()
    roi_with_floor_edge[265:300, :] = 220  # Bright white/light floor spanning across bottom

    detected_edge = find_joint_in_roi(roi_with_floor_edge, min_area=400, min_width=30)
    assert detected_edge is None, (
        f"FALSE POSITIVE REGRESSION: Static bottom background edge was accepted as a joint: {detected_edge}"
    )

    # 2. Positive case 1: Real metallic joint centered on belt
    roi_with_real_joint = roi_dark_belt.copy()
    roi_with_real_joint[110:170, 140:260] = 210
    detected_center = find_joint_in_roi(roi_with_real_joint, min_area=400, min_width=30)
    assert detected_center is not None, "Real metallic joint on conveyor belt was incorrectly rejected!"
    jx, jy, jw, jh = detected_center
    assert 120 <= jx <= 160, f"Expected jx around 140, got {jx}"
    assert 90 <= jy <= 130, f"Expected jy around 110, got {jy}"

    # 3. Positive case 2: Real metallic joint near the bottom edge (y=220..280, h=60)
    roi_near_bottom = roi_dark_belt.copy()
    roi_near_bottom[220:280, 140:260] = 210
    detected_bottom = find_joint_in_roi(roi_near_bottom, min_area=400, min_width=30)
    assert detected_bottom is not None, "Real metallic joint near bottom edge was incorrectly rejected!"

    # 4. Positive case 3: Real joint spanning vertically across the belt (y=0..300, as seen on real rig)
    roi_full_vertical = roi_dark_belt.copy()
    roi_full_vertical[0:300, 150:250] = 210
    detected_vertical = find_joint_in_roi(roi_full_vertical, min_area=400, min_width=30)
    assert detected_vertical is not None, "Real metallic joint spanning belt vertically was incorrectly rejected!"

    # 5. Positive case 4: Stationary joint across consecutive frames
    engine = JointGuardStateEngine(target_frames=5, damage_thresh=0.70)
    yolo = MockYOLO(lambda idx, c: (0.95, 0.05))
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 60
    # Joint sits stationary in the middle of inspection zone for 10 frames
    stat_bbox = (200, 150, 180, 180)
    for frame_idx in range(10):
        res = engine.process_frame(
            frame,
            joint_detected=True,
            raw_bbox_full=stat_bbox,
            in_zone=True,
            yolo_model=yolo,
            no_quality_check=True
        )
        assert res["smoothed_bbox"] is not None, f"Stationary joint lost bounding box at frame {frame_idx}"
        assert res["current_label"] != "NOT VISIBLE", f"Stationary joint reported NOT VISIBLE at frame {frame_idx}"
        assert res["consecutive_lost_frames"] == 0

