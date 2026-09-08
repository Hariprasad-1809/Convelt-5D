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
    is_new_classification: bool = False  # True only on the frame classification is locked

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
        return res


class TrackItem:
    """Internal state representation for a single tracked joint object."""

    def __init__(self, track_id: int, initial_bbox: BoundingBox):
        self.track_id: int = track_id
        self.bbox: BoundingBox = initial_bbox
        self.state: str = "APPROACHING"
        self.centroid_history: List[Tuple[int, int]] = [initial_bbox.center]
        self.velocity_history: List[float] = []
        self.feature_history: List[Dict[str, float]] = []
        self.capture_zone_frames: int = 0
        self.disappeared_count: int = 0
        self.classification_result: Optional[ClassificationResult] = None
        self.has_been_classified: bool = False
        self.has_been_inspected: bool = False

    @property
    def current_centroid(self) -> Tuple[int, int]:
        return self.centroid_history[-1]


class JointTracker:
    """Lightweight centroid tracker and single-pass inspection state machine."""

    def __init__(self, config: Optional[VisionConfig] = None):
        self.cfg: VisionConfig = config if config is not None else DEFAULT_CONFIG
        self.next_track_id: int = 1
        self.tracks: Dict[int, TrackItem] = {}

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
        """Checks if the joint's FULL bounding box is contained inside the inner capture zone."""
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

    def _compute_averaged_classification(self, track: TrackItem) -> ClassificationResult:
        """Averages raw features collected over the capture zone window and computes final classification."""
        if not track.feature_history:
            print("[WARNING] Capture zone feature window empty — returning INVALID classification")
            return ClassificationResult(
                label="INVALID",
                vision_score=0.0,
                features={"edge_density": 0.0, "hough_line_score": 0.0, "intensity_variance": 0.0}
            )

        avg_edge = float(np.mean([f.get("edge_density", 0.0) for f in track.feature_history]))
        avg_hough = float(np.mean([f.get("hough_line_score", 0.0) for f in track.feature_history]))
        avg_var = float(np.mean([f.get("intensity_variance", 0.0) for f in track.feature_history]))

        # Normalize metrics against config maximums
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
                    is_new_classification=False
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

        # ---------------------------------------------------------------------
        # Case A: No joint detected in this frame
        # ---------------------------------------------------------------------
        if bbox is None:
            retired_event = None
            for track_id, track in list(self.tracks.items()):
                track.disappeared_count += 1
                if track.disappeared_count > self.cfg.MAX_DISAPPEARED_FRAMES:
                    if track.has_been_inspected or track.has_been_classified:
                        track.state = "PASSED"
                        retired_event = TrackEvent(
                            track_id=track.track_id,
                            state="PASSED",
                            bbox=None,
                            classification=track.classification_result,
                            is_new_classification=False
                        )
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

        # Update existing track or initialize new track
        if best_match_id is not None:
            track = self.tracks[best_match_id]
        else:
            track = TrackItem(self.next_track_id, bbox)
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        track.bbox = bbox
        track.disappeared_count = 0
        prev_cx, prev_cy = track.current_centroid
        delta_dist = float(np.hypot(cx - prev_cx, cy - prev_cy))
        track.centroid_history.append((cx, cy))

        # Velocity smooth motion validation
        if len(track.velocity_history) > 0:
            v_prev = track.velocity_history[-1]
            if abs(delta_dist - v_prev) > self.cfg.MAX_VELOCITY_VARIANCE:
                # Erratic jump detected, reset capture window
                track.capture_zone_frames = 0
        track.velocity_history.append(delta_dist)

        inside_roi = self._is_inside_roi(bbox, frame_w, frame_h)
        inside_cz = self._is_contained_in_capture_zone(bbox, frame_w, frame_h)

        is_new_class = False

        if not inside_roi:
            if track.has_been_inspected:
                track.state = "PASSED"
                if not track.has_been_classified:
                    track.classification_result = self._compute_averaged_classification(track)
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

                # Sample feature snapshot if sharp enough
                if frame is not None and not track.has_been_classified:
                    roi_crop = bbox.crop_roi(frame)
                    gray_crop = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY) if len(roi_crop.shape) == 3 else roi_crop
                    sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

                    if sharpness >= self.cfg.MIN_SHARPNESS_SCORE:
                        snapshot = classify_joint(roi_crop, self.cfg)
                        track.feature_history.append(snapshot.features)
                        if len(track.feature_history) > self.cfg.CAPTURE_ZONE_MAX_FRAMES:
                            track.feature_history.pop(0)
                    else:
                        print(f"[DEBUG] Frame skipped due to motion blur (sharpness {sharpness:.1f} < {self.cfg.MIN_SHARPNESS_SCORE:.1f})")

                if track.capture_zone_frames >= self.cfg.CAPTURE_ZONE_MIN_FRAMES:
                    track.state = "INSPECTING"
                    track.has_been_inspected = True
                    if not track.has_been_classified:
                        track.classification_result = self._compute_averaged_classification(track)
                        track.has_been_classified = True
                        is_new_class = True
            else:
                # Inside outer ROI but outside inner capture zone
                if track.has_been_inspected:
                    track.state = "INSPECTING"
                    if not track.has_been_classified:
                        track.classification_result = self._compute_averaged_classification(track)
                        track.has_been_classified = True
                        is_new_class = True
                else:
                    track.state = "APPROACHING"

        return TrackEvent(
            track_id=track.track_id,
            state=track.state,
            bbox=bbox,
            classification=track.classification_result,
            is_new_classification=is_new_class
        )
