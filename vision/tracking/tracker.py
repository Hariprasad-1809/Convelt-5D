"""
Joint Centroid Tracker & Inspection State Machine for JointGuard (Phase 1)

Tracks detected joints frame-to-frame across state transitions:
  APPROACHING -> INSPECTING -> PASSED

Ensures single-shot classification per joint pass without double counting.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from vision.opencv.config import VisionConfig, DEFAULT_CONFIG
from vision.opencv.detector import BoundingBox
from vision.opencv.classifier import classify_joint, ClassificationResult


@dataclass
class TrackEvent:
    """Represents a tracking update event for a joint."""
    track_id: int
    state: str                          # "APPROACHING", "INSPECTING", "PASSED"
    bbox: Optional[BoundingBox]
    classification: Optional[ClassificationResult] = None
    is_new_classification: bool = False  # True only on the frame classification is locked or reconciled
    live_classification: Optional[ClassificationResult] = None

    def to_dict(self, timestamp: str) -> Dict[str, Any]:
        """Formats the track event into the exact backend JSON structure."""
        res = {
            "timestamp": timestamp,
            "track_id": self.track_id,
            "state": self.state,
            "bbox": self.bbox.to_tuple() if self.bbox else None
        }
        if self.classification:
            res["label"] = self.classification.label
            res["vision_score"] = round(self.classification.vision_score, 2)
            res["features"] = {k: round(v, 4) for k, v in self.classification.features.items()}
        if self.live_classification:
            res["live_label"] = self.live_classification.label
            res["live_confidence"] = round(self.live_classification.features.get("confidence", 0.0), 4)
        return res


class TrackItem:
    """Internal state representation for a single tracked joint object."""

    def __init__(self, track_id: int, initial_bbox: BoundingBox):
        self.track_id: int = track_id
        self.bbox: Optional[BoundingBox] = initial_bbox
        self.last_known_bbox: Optional[BoundingBox] = initial_bbox
        self.state: str = "APPROACHING"
        self.centroid_history: List[Tuple[int, int]] = [initial_bbox.center]
        self.velocity_history: List[float] = []
        self.feature_history: List[Dict[str, float]] = []
        self.capture_zone_frames: int = 0
        self.disappeared_count: int = 0
        self.classification_result: Optional[ClassificationResult] = None
        self.locked_classification: Optional[ClassificationResult] = None
        self.live_classification: Optional[ClassificationResult] = None
        self.disagreement_streak: int = 0
        self.is_inside_capture_zone: bool = False
        self.outside_zone_frames: int = 0
        self.has_been_classified: bool = False
        self.has_been_inspected: bool = False
        self.has_exited_roi: bool = False

    @property
    def current_centroid(self) -> Tuple[int, int]:
        return self.centroid_history[-1]


class JointTracker:
    """Lightweight centroid tracker and single-pass inspection state machine."""

    def __init__(self, config: Optional[VisionConfig] = None):
        self.cfg: VisionConfig = config if config is not None else DEFAULT_CONFIG
        self.next_track_id: int = 1
        self.tracks: Dict[int, TrackItem] = {}
        self.recently_removed_tracks: Dict[int, Dict[str, Any]] = {}

    def _is_touching_frame_edge(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> bool:
        """Checks if bounding box touches or is near frame border (partially occluded)."""
        margin = self.cfg.EDGE_MARGIN_PIXELS
        return (
            bbox.x <= margin or
            bbox.y <= margin or
            (bbox.x + bbox.w) >= (frame_w - margin) or
            (bbox.y + bbox.h) >= (frame_h - margin)
        )

    def _is_inside_roi(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> bool:
        """Checks if joint centroid and bbox are fully within outer inspection ROI."""
        roi_x1 = int(self.cfg.ROI_X_MIN * frame_w)
        roi_y1 = int(self.cfg.ROI_Y_MIN * frame_h)
        roi_x2 = int(self.cfg.ROI_X_MAX * frame_w)
        roi_y2 = int(self.cfg.ROI_Y_MAX * frame_h)

        cx, cy = bbox.center
        centroid_inside = (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2)
        not_edge = not self._is_touching_frame_edge(bbox, frame_w, frame_h)
        return centroid_inside and not_edge

    def _is_contained_in_capture_zone(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> bool:
        """Checks if the joint's FULL bounding box is contained inside the inner capture zone (entry boundary)."""
        cz_x1 = int(self.cfg.CAPTURE_ZONE_X_MIN * frame_w)
        cz_y1 = int(self.cfg.CAPTURE_ZONE_Y_MIN * frame_h)
        cz_x2 = int(self.cfg.CAPTURE_ZONE_X_MAX * frame_w)
        cz_y2 = int(self.cfg.CAPTURE_ZONE_Y_MAX * frame_h)

        contained = (
            bbox.x >= cz_x1 and
            bbox.y >= cz_y1 and
            (bbox.x + bbox.w) <= cz_x2 and
            (bbox.y + bbox.h) <= cz_y2
        )
        not_edge = not self._is_touching_frame_edge(bbox, frame_w, frame_h)
        return contained and not_edge

    def _is_within_exit_capture_zone(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> bool:
        """
        Checks if the joint's bounding box is within the relaxed capture zone (exit boundary with hysteresis).
        Applies ZONE_HYSTERESIS_MARGIN_PIXELS expansion to prevent rapid boundary toggling.
        """
        margin = getattr(self.cfg, "ZONE_HYSTERESIS_MARGIN_PIXELS", 25)
        cz_x1 = max(0, int(self.cfg.CAPTURE_ZONE_X_MIN * frame_w) - margin)
        cz_y1 = max(0, int(self.cfg.CAPTURE_ZONE_Y_MIN * frame_h) - margin)
        cz_x2 = min(frame_w, int(self.cfg.CAPTURE_ZONE_X_MAX * frame_w) + margin)
        cz_y2 = min(frame_h, int(self.cfg.CAPTURE_ZONE_Y_MAX * frame_h) + margin)

        contained = (
            bbox.x >= cz_x1 and
            bbox.y >= cz_y1 and
            (bbox.x + bbox.w) <= cz_x2 and
            (bbox.y + bbox.h) <= cz_y2
        )
        not_edge = not self._is_touching_frame_edge(bbox, frame_w, frame_h)
        return contained and not_edge

    def _is_in_or_near_capture_zone(self, bbox: Optional[BoundingBox], frame_w: int, frame_h: int) -> bool:
        """Checks if bbox is physically inside or near the capture zone."""
        if bbox is None:
            return False
        cx, cy = bbox.center
        cz_x1 = int(self.cfg.CAPTURE_ZONE_X_MIN * frame_w)
        cz_y1 = int(self.cfg.CAPTURE_ZONE_Y_MIN * frame_h)
        cz_x2 = int(self.cfg.CAPTURE_ZONE_X_MAX * frame_w)
        cz_y2 = int(self.cfg.CAPTURE_ZONE_Y_MAX * frame_h)
        margin = int(self.cfg.MAX_CENTROID_DISTANCE)
        return (cz_x1 - margin <= cx <= cz_x2 + margin) and (cz_y1 - margin <= cy <= cz_y2 + margin)

    def _compute_averaged_classification(
        self,
        track: TrackItem,
        current_frame: Optional[np.ndarray] = None,
        use_recent_streak: bool = False
    ) -> ClassificationResult:
        """Averages probabilities and features collected over the capture zone window and computes final classification."""
        if not track.feature_history:
            # Fallback to direct classification of the current bounding box crop instead of failing to INVALID
            if current_frame is not None and track.bbox is not None:
                roi_crop = track.bbox.crop_roi(current_frame)
                if roi_crop.size > 0 and roi_crop.shape[0] >= 5 and roi_crop.shape[1] >= 5:
                    return classify_joint(roi_crop, self.cfg)

            return ClassificationResult(
                label="UNCERTAIN",
                vision_score=50.0,
                features={"edge_density": 0.0, "intensity_variance": 0.0, "confidence": 0.50, "display_label": "UNCERTAIN"}
            )

        disagree_req = getattr(self.cfg, "DISAGREEMENT_CORRECTION_STREAK", 3)
        if use_recent_streak or getattr(track, "disagreement_streak", 0) >= disagree_req:
            samples = track.feature_history[-disagree_req:]
        else:
            samples = track.feature_history

        # Check if trained probabilities exist in feature history
        has_trained_probs = any("p_healthy" in f for f in samples)
        if has_trained_probs:
            avg_p_healthy = float(np.mean([f.get("p_healthy", 0.5) for f in samples]))
            avg_p_damage = float(np.mean([f.get("p_damage", 0.5) for f in samples]))
            conf_threshold = getattr(self.cfg, "CONFIDENCE_THRESHOLD", 0.60)

            if avg_p_healthy >= conf_threshold:
                label = "HEALTHY"
                confidence = avg_p_healthy
                vision_score = round(confidence * 100.0, 2)
            elif avg_p_damage >= conf_threshold:
                label = "DAMAGE"
                confidence = avg_p_damage
                vision_score = round((1.0 - confidence) * 100.0, 2)
            else:
                label = "UNCERTAIN"
                confidence = max(avg_p_healthy, avg_p_damage)
                vision_score = 50.0

            avg_features = {
                "confidence": round(confidence, 4),
                "p_healthy": round(avg_p_healthy, 4),
                "p_damage": round(avg_p_damage, 4),
                "edge_density": round(float(np.mean([f.get("edge_density", 0.0) for f in track.feature_history])), 4),
                "intensity_variance": round(float(np.mean([f.get("intensity_variance", 0.0) for f in track.feature_history])), 2)
            }
            return ClassificationResult(
                label=label,
                vision_score=vision_score,
                features=avg_features
            )

        # Fallback to classical CV feature weighting
        avg_edge = float(np.mean([f.get("edge_density", 0.0) for f in track.feature_history]))
        avg_hough = float(np.mean([f.get("hough_line_score", 0.0) for f in track.feature_history]))
        avg_var = float(np.mean([f.get("intensity_variance", 0.0) for f in track.feature_history]))

        norm_edge_penalty = min(1.0, avg_edge / self.cfg.EDGE_DENSITY_MAX_EXPECTED)
        norm_hough_penalty = min(1.0, avg_hough / 2.0)
        norm_var_penalty = min(1.0, avg_var / self.cfg.VARIANCE_MAX_EXPECTED)

        total_damage_penalty = (
            self.cfg.WEIGHT_EDGE_DENSITY * norm_edge_penalty +
            self.cfg.WEIGHT_HOUGH_LINES * norm_hough_penalty +
            self.cfg.WEIGHT_INTENSITY_VARIANCE * norm_var_penalty
        )

        vision_score = float(max(0.0, min(100.0, 100.0 * (1.0 - total_damage_penalty))))

        # Apply classification threshold with hysteresis
        h_threshold = self.cfg.HEALTHY_THRESHOLD_SCORE
        h_margin = getattr(self.cfg, "THRESHOLD_HYSTERESIS", 5.0) / 2.0

        if vision_score >= (h_threshold + h_margin):
            label = "HEALTHY"
        elif vision_score <= (h_threshold - h_margin):
            label = "DAMAGE"
        else:
            label = "HEALTHY" if vision_score >= h_threshold else "DAMAGE"

        return ClassificationResult(
            label=label,
            vision_score=vision_score,
            features={
                "edge_density": avg_edge,
                "hough_line_score": avg_hough,
                "intensity_variance": avg_var
            }
        )

    def get_active_track_events(self) -> List[TrackEvent]:
        """Returns TrackEvent snapshots for all active tracks updated on the current frame."""
        events = []
        for track_id, track in self.tracks.items():
            if track.bbox is not None and track.disappeared_count == 0:
                events.append(TrackEvent(
                    track_id=track.track_id,
                    state=track.state,
                    bbox=track.bbox,
                    classification=track.classification_result,
                    is_new_classification=False,
                    live_classification=track.live_classification
                ))
        return events

    def update(
        self,
        bbox: Optional[BoundingBox],
        timestamp: str,
        frame: Optional[np.ndarray] = None
    ) -> Optional[TrackEvent]:
        """
        Updates the tracker with the latest detected bounding box under continuous belt motion.
        """
        frame_h, frame_w = (frame.shape[:2]) if frame is not None else (480, 640)

        # Age recently-removed tracks and purge those beyond grace window (10 frames)
        for tid in list(self.recently_removed_tracks.keys()):
            self.recently_removed_tracks[tid]["age"] += 1
            if self.recently_removed_tracks[tid]["age"] > 10:
                del self.recently_removed_tracks[tid]

        # ---------------------------------------------------------------------
        # Case A: No joint detected in this frame
        # ---------------------------------------------------------------------
        if bbox is None:
            retired_event = None
            for track_id, track in list(self.tracks.items()):
                track.disappeared_count += 1
                track.bbox = None  # Issue 4 fix: clear bbox on undetected frames to avoid stray ghost boxes
                # If a track was merely approaching and lost detection, reset it quickly (3 frames)
                # to avoid lingering ghost boxes on screen
                max_allowed = 3 if track.state == "APPROACHING" else self.cfg.MAX_DISAPPEARED_FRAMES
                if track.disappeared_count > max_allowed:
                    if track.has_been_inspected or track.has_been_classified:
                        track.state = "PASSED"
                        retired_event = TrackEvent(
                            track_id=track.track_id,
                            state="PASSED",
                            bbox=None,
                            classification=track.classification_result,
                            is_new_classification=False,
                            live_classification=track.live_classification
                        )

                    # Prevent duplicate IDs caused by transient detection dropouts:
                    # If the track was physically inside or near the capture zone / ROI and did not
                    # genuinely exit the far side of the ROI, save only the single most recent track
                    is_in_or_near = (
                        track.last_known_bbox is not None and
                        (self._is_inside_roi(track.last_known_bbox, frame_w, frame_h) or
                         self._is_in_or_near_capture_zone(track.last_known_bbox, frame_w, frame_h))
                    )
                    if is_in_or_near and not track.has_exited_roi:
                        self.recently_removed_tracks = {
                            track_id: {
                                "track": track,
                                "age": 0
                            }
                        }

                    del self.tracks[track_id]
            return retired_event

        # ---------------------------------------------------------------------
        # Case B: Joint detected -> Match with existing active tracks
        # ---------------------------------------------------------------------
        cx, cy = bbox.center
        best_match_id = None
        min_dist = float("inf")

        for track_id, track in self.tracks.items():
            prev_cx, prev_cy = track.current_centroid
            dist = np.hypot(cx - prev_cx, cy - prev_cy)
            if dist < min_dist and dist <= self.cfg.MAX_CENTROID_DISTANCE:
                min_dist = dist
                best_match_id = track_id

        # Update existing active track, re-attach recently removed track, or initialize new track
        reattached = False
        if best_match_id is not None:
            track = self.tracks[best_match_id]
        else:
            # Check recently-removed tracks (within grace window of 5-10 frames)
            # for a centroid within MAX_CENTROID_DISTANCE of the new detection
            reattached_id = None
            reattached_min_dist = float("inf")
            for tid, record in list(self.recently_removed_tracks.items()):
                rem_track = record["track"]
                prev_cx, prev_cy = rem_track.current_centroid
                dist = np.hypot(cx - prev_cx, cy - prev_cy)
                if dist < reattached_min_dist and dist <= self.cfg.MAX_CENTROID_DISTANCE:
                    reattached_min_dist = dist
                    reattached_id = tid

            if reattached_id is not None:
                track = self.recently_removed_tracks.pop(reattached_id)["track"]
                self.tracks[reattached_id] = track
                reattached = True
                if track.has_been_inspected:
                    track.state = "INSPECTING"
            else:
                # If an active track is already in INSPECTING state inside the capture zone,
                # and this detection is inside the capture zone, match it to that active track
                # rather than spawning a duplicate track ID on the same physical joint
                active_inspecting = [
                    (tid, t) for tid, t in self.tracks.items()
                    if t.state == "INSPECTING" and self._is_contained_in_capture_zone(bbox, frame_w, frame_h)
                ]
                if active_inspecting:
                    best_id, track = active_inspecting[0]
                else:
                    track = TrackItem(self.next_track_id, bbox)
                    self.tracks[self.next_track_id] = track
                    self.next_track_id += 1

        track.bbox = bbox
        track.last_known_bbox = bbox
        track.disappeared_count = 0
        prev_cx, prev_cy = track.current_centroid
        delta_dist = float(np.hypot(cx - prev_cx, cy - prev_cy))
        track.centroid_history.append((cx, cy))

        # Crucial: Any OTHER track in self.tracks was NOT detected on this frame!
        # Increment their disappeared_count, clear bbox to avoid ghost labels (Issue 4),
        # and remove stale tracks so IDs do not accumulate
        for other_id, other_track in list(self.tracks.items()):
            if other_id != track.track_id:
                other_track.disappeared_count += 1
                other_track.bbox = None  # Issue 4 fix: ensure undetected tracks do not render bounding boxes
                max_allowed = 3 if other_track.state == "APPROACHING" else self.cfg.MAX_DISAPPEARED_FRAMES
                if other_track.disappeared_count > max_allowed:
                    if other_track.has_been_inspected or other_track.has_been_classified:
                        other_track.state = "PASSED"
                    del self.tracks[other_id]

        # Velocity smooth motion validation (skip erratic jump check if reattached after transient gap)
        if not reattached and len(track.velocity_history) > 0:
            v_prev = track.velocity_history[-1]
            if abs(delta_dist - v_prev) > self.cfg.MAX_VELOCITY_VARIANCE:
                # Erratic jump detected, reset capture window
                track.capture_zone_frames = 0
        track.velocity_history.append(delta_dist)

        inside_roi = self._is_inside_roi(bbox, frame_w, frame_h)

        # Issue 2 Fix: Dual-boundary hysteresis + exit debounce
        raw_inside_cz = self._is_contained_in_capture_zone(bbox, frame_w, frame_h)
        raw_within_exit_cz = self._is_within_exit_capture_zone(bbox, frame_w, frame_h)
        debounce_limit = getattr(self.cfg, "ZONE_EXIT_DEBOUNCE_FRAMES", 4)

        if not track.is_inside_capture_zone:
            if raw_inside_cz:
                track.is_inside_capture_zone = True
                track.outside_zone_frames = 0
        else:
            if not raw_within_exit_cz:
                track.outside_zone_frames += 1
                if track.outside_zone_frames >= debounce_limit:
                    track.is_inside_capture_zone = False
            else:
                track.outside_zone_frames = 0

        # If completely outside outer ROI, instantly clear zone state
        if not inside_roi:
            track.is_inside_capture_zone = False
            track.outside_zone_frames = 0

        inside_cz = track.is_inside_capture_zone
        is_new_class = False

        if not inside_roi:
            if track.has_been_inspected:
                track.state = "PASSED"
                track.has_exited_roi = True
                if not track.has_been_classified:
                    classified = self._compute_averaged_classification(track, frame)
                    track.classification_result = classified
                    track.locked_classification = classified
                    track.has_been_classified = True
                    is_new_class = True
            else:
                track.state = "APPROACHING"
                track.capture_zone_frames = 0
                track.feature_history.clear()
        else:
            # Inside outer ROI
            if inside_cz:
                track.capture_zone_frames += 1

                # Sample feature snapshot
                if frame is not None:
                    roi_crop = bbox.crop_roi(frame)
                    if roi_crop.size > 0 and roi_crop.shape[0] >= 5 and roi_crop.shape[1] >= 5:
                        snapshot = classify_joint(roi_crop, self.cfg)
                        track.feature_history.append(snapshot.features)
                        track.live_classification = snapshot
                        if not track.has_been_classified:
                            track.classification_result = snapshot
                        if len(track.feature_history) > self.cfg.CAPTURE_ZONE_MAX_FRAMES:
                            track.feature_history.pop(0)

                if track.has_been_inspected or track.capture_zone_frames >= self.cfg.CAPTURE_ZONE_MIN_FRAMES:
                    track.state = "INSPECTING"
                    track.has_been_inspected = True

                    # Issue 3 Fix: Classification Stabilization before locking
                    stability_frames = getattr(self.cfg, "CLASSIFICATION_STABILITY_FRAMES", 4)
                    live_conf = track.live_classification.features.get("confidence", 0.0) if track.live_classification else 0.0
                    high_conf_immediate = (track.capture_zone_frames >= self.cfg.CAPTURE_ZONE_MIN_FRAMES and live_conf >= 0.70)

                    if not track.has_been_classified:
                        if track.capture_zone_frames >= stability_frames or high_conf_immediate:
                            stabilized = self._compute_averaged_classification(track, frame)
                            track.classification_result = stabilized
                            track.locked_classification = stabilized
                            track.has_been_classified = True
                            is_new_class = True
                    else:
                        # Issue 3 Fix: Dynamic Disagreement Reconciliation
                        if track.locked_classification is not None and track.live_classification is not None:
                            locked_label = track.locked_classification.label
                            live_label = track.live_classification.label
                            disagree_thresh = getattr(self.cfg, "DISAGREEMENT_CONFIDENCE_THRESHOLD", 0.65)
                            disagree_streak_req = getattr(self.cfg, "DISAGREEMENT_CORRECTION_STREAK", 3)

                            if live_label in ("HEALTHY", "DAMAGE") and locked_label in ("HEALTHY", "DAMAGE") and live_label != locked_label:
                                if live_conf >= disagree_thresh:
                                    track.disagreement_streak += 1
                                    if track.disagreement_streak >= disagree_streak_req:
                                        # Reconcile locked classification using the sustained streak evidence
                                        reconciled = self._compute_averaged_classification(track, frame, use_recent_streak=True)
                                        track.locked_classification = reconciled
                                        track.classification_result = reconciled
                                        track.disagreement_streak = 0
                                        is_new_class = True
                                else:
                                    track.disagreement_streak = 0
                            else:
                                track.disagreement_streak = 0
            else:
                # Inside outer ROI but outside inner capture zone
                if track.has_been_inspected:
                    track.state = "INSPECTING"
                    if not track.has_been_classified:
                        classified = self._compute_averaged_classification(track, frame)
                        track.classification_result = classified
                        track.locked_classification = classified
                        track.has_been_classified = True
                        is_new_class = True
                else:
                    track.state = "APPROACHING"

        return TrackEvent(
            track_id=track.track_id,
            state=track.state,
            bbox=bbox,
            classification=track.classification_result,
            is_new_classification=is_new_class,
            live_classification=track.live_classification
        )
