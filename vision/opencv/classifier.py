"""
Joint Damage Classification Module for JointGuard (Phase 1 Classical CV)

Extracts texture and geometric features from a cropped joint ROI (Edge Density,
Hough Diagonal Crease Detection, Local Intensity Variance) to compute a continuous
flatness / vision score (0 - 100) and assign HEALTHY / DAMAGE labels.

Provides a clean swappable interface (BaseJointClassifier) for Phase 2 YOLO replacement.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from vision.opencv.config import VisionConfig, DEFAULT_CONFIG


@dataclass
class ClassificationResult:
    """Structure holding the classification label, continuous score, and raw feature metrics."""
    label: str                   # "HEALTHY" or "DAMAGE"
    vision_score: float          # Continuous score 0.0 - 100.0 (100 = perfectly flat/healthy)
    features: Dict[str, float]   # Raw extracted feature values for debugging & tuning

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to dict format matching backend event payload."""
        return {
            "label": self.label,
            "vision_score": round(self.vision_score, 2),
            "features": {k: round(v, 4) for k, v in self.features.items()}
        }


class BaseJointClassifier(ABC):
    """Abstract Base Class for joint classification algorithms (Classical CV, YOLO, CNN, etc.)."""

    @abstractmethod
    def classify(self, roi_frame: np.ndarray, config: Optional[VisionConfig] = None) -> ClassificationResult:
        """Classifies a cropped joint ROI image."""
        pass


class ClassicalCVClassifier(BaseJointClassifier):
    """Classical CV feature-based classifier using Canny edges, Hough lines, and intensity variance."""

    def classify(self, roi_frame: np.ndarray, config: Optional[VisionConfig] = None) -> ClassificationResult:
        cfg = config if config is not None else DEFAULT_CONFIG

        if roi_frame is None or roi_frame.size == 0 or roi_frame.shape[0] < 5 or roi_frame.shape[1] < 5:
            # Input validity guard: return INVALID label for empty or unreadable ROI crop
            return ClassificationResult(
                label="INVALID",
                vision_score=0.0,
                features={"edge_density": 0.0, "hough_line_score": 0.0, "intensity_variance": 0.0}
            )

        # Convert ROI to BGR / Grayscale / HSV
        if len(roi_frame.shape) == 3:
            bgr = roi_frame
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_frame.copy()
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        roi_area = float(gray.shape[0] * gray.shape[1])

        # ---------------------------------------------------------------------
        # 1. Feature 1: Patch Continuity & Sub-contour Fragmentation (Physical Foil Structure)
        # ---------------------------------------------------------------------
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_chan = hsv[:, :, 2]
        v_median = float(np.median(v_chan)) if v_chan.size > 0 else 120.0
        v_lower = max(int(cfg.HSV_LOWER[2]), min(210, int(v_median + 20.0)))
        mask = cv2.inRange(hsv, np.array([0, 0, v_lower], dtype=np.uint8), np.array([180, int(cfg.HSV_UPPER[1]), int(cfg.HSV_UPPER[2])], dtype=np.uint8))

        # Focus analysis on metallic foil pixels inside ROI
        pts = cv2.findNonZero(mask)
        if pts is not None:
            fx, fy, fw, fh = cv2.boundingRect(pts)
            foil_crop_gray = gray[fy:fy + fh, fx:fx + fw]
            foil_crop_mask = mask[fy:fy + fh, fx:fx + fw]
        else:
            foil_crop_gray = gray
            foil_crop_mask = mask

        cnts, _ = cv2.findContours(foil_crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_cnts = [c for c in cnts if cv2.contourArea(c) > 500]
        total_mask_px = float(np.count_nonzero(foil_crop_mask))

        if total_mask_px > 0 and large_cnts:
            max_area = max([cv2.contourArea(c) for c in large_cnts])
            raw_patch_continuity = float(max_area / total_mask_px)
        else:
            raw_patch_continuity = 1.0

        raw_frag_count = len(large_cnts)

        # Structural penalty calculation (0.0 = single smooth flat foil, 1.0 = fragmented/split foil)
        cont_penalty = max(0.0, 1.0 - raw_patch_continuity)
        frag_penalty = min(1.0, max(0.0, float(raw_frag_count - 1) * 0.20))
        norm_struct_penalty = min(1.0, 0.6 * cont_penalty + 0.4 * frag_penalty)

        # ---------------------------------------------------------------------
        # 2. Feature 2: Edge Density (Canny Edges on Gaussian Blurred Foil ROI)
        # ---------------------------------------------------------------------
        blurred_gray = cv2.GaussianBlur(foil_crop_gray, (5, 5), 0)
        edges = cv2.Canny(blurred_gray, int(cfg.CANNY_THRESHOLD1), int(cfg.CANNY_THRESHOLD2))
        edge_pixel_count = float(np.count_nonzero(edges))
        raw_edge_density = edge_pixel_count / float(foil_crop_gray.size) if foil_crop_gray.size > 0 else 0.0
        norm_edge_penalty = min(1.0, raw_edge_density / float(cfg.EDGE_DENSITY_MAX_EXPECTED))

        # ---------------------------------------------------------------------
        # 3. Feature 3: Hough Line Crease Detection
        # ---------------------------------------------------------------------
        lines = cv2.HoughLinesP(
            edges,
            cfg.HOUGH_RHO,
            cfg.HOUGH_THETA,
            cfg.HOUGH_THRESHOLD,
            minLineLength=cfg.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=cfg.HOUGH_MAX_LINE_GAP
        )

        hough_line_score = 0.0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.ravel()
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                length = np.sqrt(dx * dx + dy * dy)
                angle_deg = np.degrees(np.abs(np.arctan2(dy, dx)))
                if angle_deg > 90.0:
                    angle_deg = 180.0 - angle_deg

                if cfg.HOUGH_DIAGONAL_MIN_ANGLE <= angle_deg <= cfg.HOUGH_DIAGONAL_MAX_ANGLE:
                    hough_line_score += length

            roi_diag = np.sqrt(foil_crop_gray.shape[0] ** 2 + foil_crop_gray.shape[1] ** 2)
            raw_hough_score = float(hough_line_score / roi_diag) if roi_diag > 0 else 0.0
        else:
            raw_hough_score = 0.0

        norm_hough_penalty = min(1.0, raw_hough_score / 5.0)

        # ---------------------------------------------------------------------
        # 4. Feature 4: Local Intensity Variance (Patchiness)
        # ---------------------------------------------------------------------
        blurred_var = cv2.GaussianBlur(foil_crop_gray, cfg.VARIANCE_BLUR_KERNEL, 0)
        raw_intensity_var = float(np.std(blurred_var)) if blurred_var.size > 0 else 0.0
        norm_var_penalty = min(1.0, max(0.0, (raw_intensity_var - 30.0) / float(cfg.VARIANCE_MAX_EXPECTED - 30.0)))

        # ---------------------------------------------------------------------
        # 5. Feature Combination & Vision Score Calculation
        # ---------------------------------------------------------------------
        # Weighted total damage penalty (0.0 = perfectly smooth/healthy, 1.0 = highly damaged)
        total_damage_penalty = (
            0.50 * norm_struct_penalty +
            0.30 * norm_var_penalty +
            0.10 * norm_edge_penalty +
            0.10 * norm_hough_penalty
        )

        # Convert to 0 - 100 continuous score (100 = perfectly flat/healthy)
        vision_score = max(0.0, min(100.0, 100.0 * (1.0 - total_damage_penalty)))

        # Assign label based on configured threshold
        label = "HEALTHY" if vision_score >= cfg.HEALTHY_THRESHOLD_SCORE else "DAMAGE"

        features = {
            "patch_continuity": float(raw_patch_continuity),
            "fragment_count": float(raw_frag_count),
            "edge_density": float(raw_edge_density),
            "hough_line_score": float(raw_hough_score),
            "intensity_variance": float(raw_intensity_var)
        }

        return ClassificationResult(
            label=label,
            vision_score=float(vision_score),
            features=features
        )


# Default classifier instance
_DEFAULT_CLASSIFIER = ClassicalCVClassifier()


def classify_joint(roi_frame: np.ndarray, config: Optional[VisionConfig] = None) -> ClassificationResult:
    """
    Convenience function matching the swappable classifier interface signature.

    Args:
        roi_frame: BGR or Grayscale cropped joint region.
        config: Optional VisionConfig parameter.

    Returns:
        ClassificationResult containing label, vision_score (0-100), and raw feature dictionary.
    """
    return _DEFAULT_CLASSIFIER.classify(roi_frame, config)
