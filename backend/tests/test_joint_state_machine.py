"""
Unit tests for JointGuard JointGuardStateEngine & Persistent Joint Inspection State Machine
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.services.vision_service import JointGuardStateEngine, JointTrack

class DummyYOLO:
    def __init__(self, p_healthy=0.90, p_damage=0.10):
        self.p_healthy = p_healthy
        self.p_damage = p_damage
        self.names = {0: "healthy", 1: "damage"}

    def __call__(self, crop, verbose=False):
        class Probs:
            def __init__(self, p_h, p_d):
                self.data = {0: p_h, 1: p_d}
        class Result:
            def __init__(self, names, probs):
                self.names = names
                self.probs = probs
        return [Result(self.names, Probs(self.p_healthy, self.p_damage))]


def test_persistent_joint_tracking_and_ema():
    engine = JointGuardStateEngine(target_frames=5, track_lost_timeout=1.5, bbox_smooth_alpha=0.3)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150

    # Frame 1: Detect J01 at (100, 100, 100, 100)
    yolo_h = DummyYOLO(0.95, 0.05)
    res1 = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(100, 100, 100, 100), in_zone=True, yolo_model=yolo_h, no_quality_check=True)
    
    assert res1["joint_id"] == "J01"
    assert res1["smoothed_bbox"] == (100, 100, 100, 100)

    # Frame 2: Slight movement to (110, 105, 102, 98) -> Distance is small
    res2 = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(110, 105, 102, 98), in_zone=True, yolo_model=yolo_h, no_quality_check=True)
    
    assert res2["joint_id"] == "J01"  # Same persistent joint!
    # EMA smoothing: 0.3 * 110 + 0.7 * 100 = 103
    sx, sy, sw, sh = res2["smoothed_bbox"]
    assert sx == 103


def test_temporary_disappearance_hold_period():
    engine = JointGuardStateEngine(target_frames=5, track_lost_timeout=1.5)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    yolo_d = DummyYOLO(0.05, 0.95)

    # Confirm DAMAGE on J01 with 5 frames
    for i in range(5):
        res = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(100, 100, 100, 100), in_zone=True, yolo_model=yolo_d, no_quality_check=True)

    assert res["final_label"] == "DAMAGE"

    # Now simulate 3 frames of NO JOINT IN ROI (joint temporarily lost for 0.2s)
    for i in range(3):
        res_lost = engine.process_frame(frame, joint_detected=False, raw_bbox_full=None, in_zone=False, yolo_model=yolo_d, no_quality_check=True)
        # CRITICAL RULE: FINAL MUST REMAIN DAMAGE (NOT UNCERTAIN!)
        assert res_lost["final_label"] == "DAMAGE"
        assert res_lost["tracking_status"] == "TEMPORARILY_LOST"


def test_result_hysteresis():
    engine = JointGuardStateEngine(target_frames=5, track_lost_timeout=1.5)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    yolo_d = DummyYOLO(0.05, 0.95)
    yolo_h = DummyYOLO(0.95, 0.05)

    # Confirm DAMAGE on J01 with 5 frames
    for i in range(5):
        res = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(100, 100, 100, 100), in_zone=True, yolo_model=yolo_d, no_quality_check=True)

    assert res["final_label"] == "DAMAGE"

    # Now feed 1 frame of HEALTHY -> Hysteresis MUST keep FINAL as DAMAGE
    res_h1 = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(100, 100, 100, 100), in_zone=True, yolo_model=yolo_h, no_quality_check=True)
    assert res_h1["final_label"] == "DAMAGE"

    # Feed 2nd frame of HEALTHY -> Still keep DAMAGE
    res_h2 = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(100, 100, 100, 100), in_zone=True, yolo_model=yolo_h, no_quality_check=True)
    assert res_h2["final_label"] == "DAMAGE"


if __name__ == "__main__":
    test_persistent_joint_tracking_and_ema()
    test_temporary_disappearance_hold_period()
    test_result_hysteresis()
    print("All JointGuardStateEngine unit tests passed successfully!")
