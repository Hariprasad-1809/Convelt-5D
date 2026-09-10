"""
Unit tests for JointGuard JointGuardStateEngine Temporal Aggregation Logic
"""

import sys
import os
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


def test_temporal_aggregation_damage():
    engine = JointGuardStateEngine(target_frames=5, damage_thresh=0.70, healthy_thresh=0.70)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    yolo_d = DummyYOLO(0.15, 0.85)

    for i in range(5):
        res = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(100, 100, 100, 100), in_zone=True, yolo_model=yolo_d, no_quality_check=True)

    assert res["final_label"] == "DAMAGE"
    assert round(res["final_confidence"], 2) == 0.85


def test_temporal_aggregation_healthy():
    engine = JointGuardStateEngine(target_frames=5, damage_thresh=0.70, healthy_thresh=0.70)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    yolo_h = DummyYOLO(0.95, 0.05)

    for i in range(5):
        res = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(100, 100, 100, 100), in_zone=True, yolo_model=yolo_h, no_quality_check=True)

    assert res["final_label"] == "HEALTHY"
    assert round(res["final_confidence"], 2) == 0.95


def test_temporal_aggregation_uncertain_low_avg_conf():
    engine = JointGuardStateEngine(target_frames=5, damage_thresh=0.70, healthy_thresh=0.70)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    yolo_w = DummyYOLO(0.40, 0.60)

    for i in range(5):
        res = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(100, 100, 100, 100), in_zone=True, yolo_model=yolo_w, no_quality_check=True)

    assert res["final_label"] == "UNCERTAIN"


def test_temporal_aggregation_waiting_insufficient_frames():
    engine = JointGuardStateEngine(target_frames=5, damage_thresh=0.70, healthy_thresh=0.70)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    yolo_h = DummyYOLO(0.99, 0.01)

    # Only 2 frames collected
    for i in range(2):
        res = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(100, 100, 100, 100), in_zone=True, yolo_model=yolo_h, no_quality_check=True)

    assert res["final_label"] == "UNCERTAIN"


if __name__ == "__main__":
    test_temporal_aggregation_damage()
    test_temporal_aggregation_healthy()
    test_temporal_aggregation_uncertain_low_avg_conf()
    test_temporal_aggregation_waiting_insufficient_frames()
    print("All JointGuardStateEngine temporal aggregation tests passed successfully!")
