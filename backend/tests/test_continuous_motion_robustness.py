"""
Robust State Machine Inspection Verification Suite
Tests the 4 critical requirements for conveyor belt joint inspection:
1. Genuinely damaged joint overrides early healthy frames to lock FINAL: DAMAGE upon reaching PASSED
2. Genuinely healthy joint locks FINAL: HEALTHY
3. Entering/motion-skewed joint aspect ratio clamping prevents multi-second stalls
4. Zone debounce & hysteresis eliminates OUTSIDE_ZONE flicker and duplicate track IDs
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.services.vision_service import JointGuardStateEngine, JointTrack, is_in_inner_inspection_zone, check_crop_quality


class DynamicDummyYOLO:
    """YOLO mock that dynamically returns probabilities based on a callable or schedule."""
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


def test_damaged_joint_transit_overrides_early_healthy_lock_in():
    """
    ISSUE 1 TEST:
    A damaged joint enters the zone. The first 5 frames only capture the edge of the joint
    (clean belt), resulting in early HEALTHY (76.0%) predictions.
    As the joint transits further (frames 6-35), the visible crack enters the crop,
    producing strong DAMAGE predictions (88.0%).
    Upon reaching PASSED, the FINAL verdict must reflect the full temporal history
    and lock as DAMAGE with high confidence (not the stale early HEALTHY guess).
    """
    engine = JointGuardStateEngine(target_frames=5, damage_thresh=0.70, healthy_thresh=0.70)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150

    def yolo_schedule(idx, crop):
        if idx < 5:
            # First 5 frames: early edge view before crack enters crop
            return (0.76, 0.24)
        else:
            # Frames 6+: crack clearly visible in crop
            return (0.12, 0.88)

    yolo_model = DynamicDummyYOLO(yolo_schedule)

    # Transit through zone (frames 0 to 34 inside zone)
    # Moving along conveyor from x=150 to x=450
    track_ids = set()
    res_list = []

    for f_idx in range(35):
        x_pos = 150 + f_idx * 8
        res = engine.process_frame(
            frame=frame,
            joint_detected=True,
            raw_bbox_full=(x_pos, 200, 100, 80),
            in_zone=True,
            yolo_model=yolo_model,
            no_quality_check=True
        )
        track_ids.add(res["joint_id"])
        res_list.append(res)

    # Verify that during frames 0-4 it had the early reading
    assert res_list[4]["final_label"] == "HEALTHY"
    assert round(res_list[4]["final_confidence"], 2) == 0.76

    # Verify that as the crack transits (frames 6+), candidate becomes DAMAGE
    # and confirmed_label flips to DAMAGE via hysteresis
    assert res_list[-1]["final_label"] == "DAMAGE"
    assert res_list[-1]["avg_p_damage"] > 0.75

    # Now joint leaves the capture zone (5 frames outside zone to trigger debounced PASSED)
    for f_idx in range(5):
        x_pos = 450 + (f_idx + 1) * 8
        res_exit = engine.process_frame(
            frame=frame,
            joint_detected=True,
            raw_bbox_full=(x_pos, 200, 100, 80),
            in_zone=False,
            yolo_model=yolo_model,
            no_quality_check=True
        )

    # CRITICAL CHECK: Upon reaching PASSED, FINAL must be DAMAGE!
    assert res_exit["inspection_state"] == "PASSED", f"Expected PASSED, got {res_exit['inspection_state']}"
    assert res_exit["final_label"] == "DAMAGE", f"Expected FINAL: DAMAGE, got {res_exit['final_label']}"
    assert res_exit["final_confidence"] >= 0.75, f"Expected confidence >= 0.75, got {res_exit['final_confidence']}"
    assert track_ids == {"J01"}, f"Expected exactly 1 stable track ID, got {track_ids}"
    print(f"  [PASS] Test 1: Damaged Joint Transit Overrides Early Healthy -> FINAL: {res_exit['final_label']} ({res_exit['final_confidence']*100:.1f}%)")


def test_healthy_joint_transit_locks_healthy():
    """
    Test continuous transit of a genuinely healthy joint.
    Must lock FINAL: HEALTHY upon reaching PASSED.
    """
    engine = JointGuardStateEngine(target_frames=5, damage_thresh=0.70, healthy_thresh=0.70)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150

    yolo_model = DynamicDummyYOLO(lambda idx, crop: (0.92, 0.08))
    track_ids = set()

    for f_idx in range(30):
        x_pos = 150 + f_idx * 8
        res = engine.process_frame(
            frame=frame,
            joint_detected=True,
            raw_bbox_full=(x_pos, 200, 100, 80),
            in_zone=True,
            yolo_model=yolo_model,
            no_quality_check=True
        )
        track_ids.add(res["joint_id"])

    # Move outside zone to PASSED
    for f_idx in range(5):
        x_pos = 450 + (f_idx + 1) * 8
        res_exit = engine.process_frame(
            frame=frame,
            joint_detected=True,
            raw_bbox_full=(x_pos, 200, 100, 80),
            in_zone=False,
            yolo_model=yolo_model,
            no_quality_check=True
        )

    assert res_exit["inspection_state"] == "PASSED"
    assert res_exit["final_label"] == "HEALTHY"
    assert res_exit["final_confidence"] >= 0.85
    assert track_ids == {"J01"}
    print(f"  [PASS] Test 2: Healthy Joint Transit -> FINAL: {res_exit['final_label']} ({res_exit['final_confidence']*100:.1f}%)")


def test_no_aspect_ratio_classification_gap_during_entry():
    """
    ISSUE 2 TEST:
    When a real joint partially enters frame or is motion-skewed, its bounding box
    has a skewed aspect ratio (e.g. w=22, h=80 -> aspect_ratio = 3.63, or w=80, h=12 -> aspect_ratio = 0.15).
    Previously, check_crop_quality rejected this with BAD ASPECT RATIO, producing seconds
    of zero classification.
    With our aspect-ratio clamping fallback:
    The bounding box is symmetrically clamped/expanded with adjacent belt context,
    enabling immediate valid classification without stalling.
    """
    engine = JointGuardStateEngine(target_frames=5, min_sharpness=50.0)
    # Synthetic frame with belt texture (sharpness > 50)
    frame = np.random.randint(40, 200, (480, 640, 3), dtype=np.uint8)

    yolo_model = DynamicDummyYOLO(lambda idx, crop: (0.15, 0.85))

    # Case A: Narrow vertical sliver entering frame (x=50, y=100, w=20, h=80 -> raw ar = 4.0)
    res_narrow = engine.process_frame(
        frame=frame,
        joint_detected=True,
        raw_bbox_full=(50, 100, 20, 80),
        in_zone=True,
        yolo_model=yolo_model,
        no_quality_check=False
    )

    # Must NOT be rejected as BAD ASPECT RATIO
    assert res_narrow["current_label"] != "BAD ASPECT RATIO", f"Got rejected: {res_narrow['current_label']}"
    assert res_narrow["valid_frame"] is True, f"Frame was not valid: {res_narrow['current_label']}"
    assert res_narrow["p_damage"] == 0.85

    # Case B: Flat horizontal sliver (x=100, y=100, w=100, h=15 -> raw ar = 0.15)
    res_flat = engine.process_frame(
        frame=frame,
        joint_detected=True,
        raw_bbox_full=(100, 100, 100, 15),
        in_zone=True,
        yolo_model=yolo_model,
        no_quality_check=False
    )

    assert res_flat["current_label"] != "BAD ASPECT RATIO", f"Got rejected: {res_flat['current_label']}"
    assert res_flat["valid_frame"] is True, f"Frame was not valid: {res_flat['current_label']}"
    print("  [PASS] Test 3: No Aspect Ratio Gap During Joint Entry (Bounding Box Clamped & Classified)")


def test_no_zone_flicker_or_duplicate_track_ids():
    """
    ISSUE 3 TEST:
    A physical joint transits the zone. Due to specular highlight variations or boundary jitter:
    - 2 frames report in_zone=False in the middle of transit
    - 1 frame drops detection completely (specular blackout: joint_detected=False)
    With zone debounce and dropout hold:
    - ZERO transitions to OUTSIDE_ZONE while transiting
    - Zero duplicate track IDs generated (stays J01 throughout)
    - Reaches PASSED only after 5 consecutive frames outside zone
    """
    engine = JointGuardStateEngine(target_frames=5, debounce_outside_frames=5, track_lost_timeout=1.5)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    yolo_model = DynamicDummyYOLO(lambda idx, crop: (0.10, 0.90))

    observed_labels = []
    observed_jids = set()

    # Frame 0-10: Normal transit inside zone
    for i in range(11):
        x = 150 + i * 10
        r = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(x, 200, 100, 80), in_zone=True, yolo_model=yolo_model, no_quality_check=True)
        observed_labels.append(r["current_label"])
        observed_jids.add(r["joint_id"])

    # Frame 11-12: 2 frames of boundary jitter where raw in_zone is temporarily False
    for i in range(2):
        x = 260 + i * 10
        r = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(x, 200, 100, 80), in_zone=False, yolo_model=yolo_model, no_quality_check=True)
        observed_labels.append(r["current_label"])
        observed_jids.add(r["joint_id"])
        # Debounce MUST NOT switch to OUTSIDE_ZONE or PASSED!
        assert r["current_label"] != "OUTSIDE_ZONE", f"Flickered to OUTSIDE_ZONE at frame {11+i}"
        assert r["inspection_state"] != "PASSED", f"Prematurely transitioned to PASSED at frame {11+i}"

    # Frame 13: 1 frame specular dropout (joint_detected=False)
    r_drop = engine.process_frame(frame, joint_detected=False, raw_bbox_full=None, in_zone=False, yolo_model=yolo_model, no_quality_check=True)
    observed_jids.add(r_drop["joint_id"])
    # Track ID must NOT be lost
    assert r_drop["joint_id"] == "J01"

    # Frame 14-20: Joint detected again inside zone
    for i in range(7):
        x = 290 + i * 10
        r = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(x, 200, 100, 80), in_zone=True, yolo_model=yolo_model, no_quality_check=True)
        observed_labels.append(r["current_label"])
        observed_jids.add(r["joint_id"])

    # Assert exactly 1 joint track ID was produced (NO duplicate IDs from dropout or flicker)
    assert observed_jids == {"J01"}, f"Duplicate track IDs detected: {observed_jids}"

    # Now joint genuinely exits zone: requires 5 consecutive frames
    for i in range(4):
        r_exit_partial = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(400+i*10, 200, 100, 80), in_zone=False, yolo_model=yolo_model, no_quality_check=True)
        assert r_exit_partial["inspection_state"] != "PASSED", f"PASSED before 5 debounced frames at exit frame {i}"

    # 5th frame outside: genuinely transitions to PASSED
    r_passed = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(450, 200, 100, 80), in_zone=False, yolo_model=yolo_model, no_quality_check=True)
    assert r_passed["inspection_state"] == "PASSED"
    assert r_passed["current_label"] == "OUTSIDE_ZONE"
    assert r_passed["final_label"] == "DAMAGE"

    print("  [PASS] Test 4: Zero Zone Flicker & Zero Duplicate Track IDs Verified")


def test_early_strong_damage_finalizes_during_inspecting():
    """
    ISSUE 1 & 2 VERIFICATION:
    A damaged joint with 5 consecutive high-confidence DAMAGE frames (p_d >= 0.70)
    must lock FINAL: DAMAGE immediately while STILL inside the inspection zone (INSPECTING),
    without waiting for the track to reach PASSED.
    """
    engine = JointGuardStateEngine(target_frames=5, damage_thresh=0.70, healthy_thresh=0.70)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    yolo_model = DynamicDummyYOLO(lambda idx, crop: (0.10, 0.90))

    res_history = []
    # Feed 5 frames inside the inspection zone
    for i in range(5):
        x = 200 + i * 10
        r = engine.process_frame(
            frame,
            joint_detected=True,
            raw_bbox_full=(x, 200, 100, 80),
            in_zone=True,
            yolo_model=yolo_model,
            no_quality_check=True
        )
        res_history.append(r)

    # CRITICAL CHECK: At frame 5, still in_zone=True, verdict MUST be early-finalized as DAMAGE!
    fifth_res = res_history[4]
    assert fifth_res["valid_frame_count"] == 5, f"Expected 5 valid frames, got {fifth_res['valid_frame_count']}"
    assert fifth_res["is_early_finalized"] is True, "Expected is_early_finalized to be True at frame 5"
    assert fifth_res["final_label"] == "DAMAGE", f"Expected FINAL: DAMAGE, got {fifth_res['final_label']}"
    assert fifth_res["final_confidence"] >= 0.70, f"Expected conf >= 0.70, got {fifth_res['final_confidence']}"
    assert fifth_res["inspection_state"] == "CONFIRMED", f"Expected CONFIRMED, got {fifth_res['inspection_state']}"

    # Verify that the verdict stays permanently locked as DAMAGE across further in-zone frames
    for i in range(5, 10):
        x = 200 + i * 10
        r_cont = engine.process_frame(
            frame,
            joint_detected=True,
            raw_bbox_full=(x, 200, 100, 80),
            in_zone=True,
            yolo_model=yolo_model,
            no_quality_check=True
        )
        assert r_cont["final_label"] == "DAMAGE", f"Verdict flipped: {r_cont['final_label']}"
        assert r_cont["is_early_finalized"] is True

    print("  [PASS] Test 5: Early Strong DAMAGE Finalizes During INSPECTING (Real-Time Motor Interlock Ready)")


def test_majority_damage_never_produces_final_healthy():
    """
    ISSUE 2 CONTRADICTION VERIFICATION:
    Even when early edge frames exhibit high HEALTHY confidence (e.g. 2 frames of 0.99 Healthy),
    if the majority of frames across the transit are DAMAGE (e.g. 3 frames of 0.70 Damage),
    the final verdict must NEVER contradict the majority vote and output FINAL: HEALTHY.
    """
    engine = JointGuardStateEngine(target_frames=5, damage_thresh=0.70, healthy_thresh=0.70)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150

    def skewed_schedule(idx, crop):
        if idx < 2:
            # 2 edge frames with very high healthy confidence
            return (0.99, 0.01)
        else:
            # 3 frames with genuine damage detection
            return (0.30, 0.70)

    yolo_model = DynamicDummyYOLO(skewed_schedule)

    res_list = []
    for i in range(5):
        x = 200 + i * 10
        r = engine.process_frame(
            frame,
            joint_detected=True,
            raw_bbox_full=(x, 200, 100, 80),
            in_zone=True,
            yolo_model=yolo_model,
            no_quality_check=True
        )
        res_list.append(r)

    # Frame 5: 3 DAMAGE vs 2 HEALTHY -> Majority is DAMAGE!
    final_res = res_list[4]
    assert final_res["final_label"] != "HEALTHY", "CONTRADICTION BUG: Majority is DAMAGE but FINAL was HEALTHY!"
    assert final_res["final_label"] == "DAMAGE", f"Expected FINAL: DAMAGE, got {final_res['final_label']}"

    # Now transit to PASSED outside zone
    for i in range(5):
        r_exit = engine.process_frame(
            frame,
            joint_detected=True,
            raw_bbox_full=(450 + i * 10, 200, 100, 80),
            in_zone=False,
            yolo_model=yolo_model,
            no_quality_check=True
        )

    assert r_exit["inspection_state"] == "PASSED"
    assert r_exit["final_label"] == "DAMAGE", f"Expected PASSED FINAL: DAMAGE, got {r_exit['final_label']}"
    print("  [PASS] Test 6: Majority DAMAGE Never Produces Contradictory FINAL: HEALTHY")


def test_re_identification_prevents_track_churn_on_temporary_dropout():
    """
    ISSUE 3 TRACK FRAGMENTATION VERIFICATION:
    If a joint suffers detection dropout or momentary loss and re-appears in the same area,
    re-identification should preserve the existing joint track ID and its accumulated samples,
    preventing rapid churning (J01 -> J02 -> J03...).
    """
    engine = JointGuardStateEngine(target_frames=5, track_lost_timeout=2.0)
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
    yolo_model = DynamicDummyYOLO(lambda idx, crop: (0.15, 0.85))

    observed_ids = []

    # 1. First 4 frames detected
    for i in range(4):
        x = 200 + i * 10
        r = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(x, 200, 100, 80), in_zone=True, yolo_model=yolo_model, no_quality_check=True)
        observed_ids.append(r["joint_id"])

    # 2. 8 frames of complete detection dropout (e.g. camera glitch or specular blackout)
    for _ in range(8):
        engine.process_frame(frame, joint_detected=False, raw_bbox_full=None, in_zone=False, yolo_model=yolo_model, no_quality_check=True)

    # 3. Detection re-appears at nearby downstream position (x=260)
    r_reappear = engine.process_frame(frame, joint_detected=True, raw_bbox_full=(260, 200, 100, 80), in_zone=True, yolo_model=yolo_model, no_quality_check=True)
    observed_ids.append(r_reappear["joint_id"])

    # Must be re-identified as the SAME joint (J01), preserving previous sample accumulation
    assert r_reappear["joint_id"] == "J01", f"Track fragmented! Expected J01, got {r_reappear['joint_id']}"
    assert r_reappear["valid_frame_count"] == 5, f"Expected 5 accumulated samples, got {r_reappear['valid_frame_count']}"
    assert set(observed_ids) == {"J01"}, f"Multiple track IDs observed: {set(observed_ids)}"
    print("  [PASS] Test 7: Re-Identification Prevents Track Churn on Temporary Dropout")


if __name__ == "__main__":
    print("=" * 70)
    print("  RUNNING JOINTGUARD CONTINUOUS MOTION ROBUSTNESS TEST SUITE")
    print("=" * 70)
    test_damaged_joint_transit_overrides_early_healthy_lock_in()
    test_healthy_joint_transit_locks_healthy()
    test_no_aspect_ratio_classification_gap_during_entry()
    test_no_zone_flicker_or_duplicate_track_ids()
    test_early_strong_damage_finalizes_during_inspecting()
    test_majority_damage_never_produces_final_healthy()
    test_re_identification_prevents_track_churn_on_temporary_dropout()
    print("=" * 70)
    print("  ALL 7 ROBUST STATE MACHINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
