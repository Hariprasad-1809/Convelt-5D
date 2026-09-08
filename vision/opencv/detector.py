"""
Joint Detection Module for JointGuard (Phase 1 Classical CV)

Detects rectangular metal foil patch joints on a black rubber conveyor belt in BGR frames.
Exposes detect_joint(frame, config) -> Optional[BoundingBox].
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from vision.opencv.config import VisionConfig, DEFAULT_CONFIG


@dataclass
class BoundingBox:
    """Bounding box data structure representing a detected joint region."""
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> int:
        """Center X coordinate."""
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        """Center Y coordinate."""
        return self.y + self.h // 2

    @property
    def center(self) -> Tuple[int, int]:
        """(cx, cy) tuple."""
        return (self.cx, self.cy)

    @property
    def area(self) -> int:
        """Bounding box pixel area."""
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height or height / width, whichever is >= 1.0)."""
        if self.h == 0 or self.w == 0:
            return 0.0
        r = float(self.w) / float(self.h)
        return r if r >= 1.0 else 1.0 / r

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Returns (x, y, w, h)."""
        return (self.x, self.y, self.w, self.h)

    def crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crops the bounding box region from a BGR image frame."""
        h, w = frame.shape[:2]
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(w, self.x + self.w)
        y2 = min(h, self.y + self.h)
        return frame[y1:y2, x1:x2]


_last_detection_mask: Optional[np.ndarray] = None


def get_last_detection_mask() -> Optional[np.ndarray]:
    """Returns the binary threshold mask from the most recent detect_joint call."""
    return _last_detection_mask


def _merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], gap_px: int) -> List[Tuple[int, int, int, int]]:
    """
    Merges candidate bounding boxes in ROI space that overlap or lie within gap_px distance of each other.
    Fuses specular reflection fragments into a single bounding box for the entire joint steel patch.
    """
    if not boxes:
        return []

    rects = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(rects):
            j = i + 1
            while j < len(rects):
                r1 = rects[i]
                r2 = rects[j]

                # Check if r1 and r2 are within gap_px of each other
                if (r1[0] - gap_px <= r2[2] and r2[0] - gap_px <= r1[2] and
                    r1[1] - gap_px <= r2[3] and r2[1] - gap_px <= r1[3]):
                    rects[i] = [
                        min(r1[0], r2[0]),
                        min(r1[1], r2[1]),
                        max(r1[2], r2[2]),
                        max(r1[3], r2[3])
                    ]
                    rects.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    return [(r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in rects]


def detect_joint(frame: np.ndarray, config: Optional[VisionConfig] = None) -> Optional[BoundingBox]:
    """
    Detects a metal patch joint in a single BGR frame within the configured ROI.

    Args:
        frame: Input BGR image (np.ndarray).
        config: Optional VisionConfig instance. Uses DEFAULT_CONFIG if None.

    Returns:
        BoundingBox instance of the single best joint candidate in full-frame coordinates,
        or None if no valid joint detected.
    """
    global _last_detection_mask

    if frame is None or frame.size == 0:
        _last_detection_mask = None
        return None

    cfg = config if config is not None else DEFAULT_CONFIG
    h, w = frame.shape[:2]

    # 1. Compute pixel-space ROI coordinates from normalized config ratios
    rx1 = max(0, int(cfg.ROI_X_MIN * w))
    ry1 = max(0, int(cfg.ROI_Y_MIN * h))
    rx2 = min(w, int(cfg.ROI_X_MAX * w))
    ry2 = min(h, int(cfg.ROI_Y_MAX * h))

    if rx2 <= rx1 or ry2 <= ry1:
        _last_detection_mask = None
        return None

    # Crop frame to Region of Interest (ROI) before thresholding & contour search
    roi_frame = frame[ry1:ry2, rx1:rx2]
    roi_h, roi_w = roi_frame.shape[:2]
    roi_area = float(roi_h * roi_w)

    # Effective max contour area relative to ROI pixel area
    max_roi_fraction = float(getattr(cfg, "MAX_CONTOUR_AREA_FRACTION_OF_ROI", 0.85))
    effective_max_contour_area = min(float(cfg.MAX_CONTOUR_AREA), roi_area * max_roi_fraction)

    # 2. Convert ROI to HSV and generate dynamic metallic brightness mask
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    v_chan = hsv_roi[:, :, 2]
    v_median = float(np.median(v_chan)) if v_chan.size > 0 else 120.0

    # Dynamic threshold: metallic foil must be at least 20 units brighter than median belt in ROI
    v_lower = max(int(cfg.HSV_LOWER[2]), min(210, int(v_median + 20.0)))
    s_upper = int(cfg.HSV_UPPER[1])
    v_upper = int(cfg.HSV_UPPER[2])

    mask_hsv = cv2.inRange(hsv_roi, np.array([0, 0, v_lower], dtype=np.uint8), np.array([180, s_upper, v_upper], dtype=np.uint8))

    # Combine with grayscale thresholding ONLY if explicitly enabled
    if getattr(cfg, "USE_GRAY_FALLBACK", False):
        gray_thresh = max(int(cfg.GRAY_THRESHOLD_MIN), int(np.median(gray_roi) + 40.0))
        _, mask_gray = cv2.threshold(gray_roi, min(240, gray_thresh), cfg.GRAY_THRESHOLD_MAX, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask_hsv, mask_gray)
    else:
        mask = mask_hsv

    # Apply Morphological Closing to remove noise/specular speckle
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.MORPH_KERNEL_SIZE)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Build full-frame mask visualization (ROI mask embedded in full black frame)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = closed_mask
    _last_detection_mask = full_mask
    detect_joint.last_mask = full_mask

    # 3. Find Contours within ROI mask
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect initial candidate fragment boxes
    raw_fragment_boxes = []
    for cnt in contours:
        c_area = cv2.contourArea(cnt)
        if c_area >= 100:  # Min fragment area to consider for bounding box fusion
            bx_roi, by_roi, bw, bh = cv2.boundingRect(cnt)
            if bw > 0 and bh > 0:
                raw_fragment_boxes.append((bx_roi, by_roi, bw, bh))

    # Fuse nearby fragment boxes into unified joint patch bounding boxes
    gap_px = getattr(cfg, "CONTOUR_CLUSTER_MAX_GAP_PX", getattr(cfg, "CONTOUR_MERGE_GAP_PIXELS", 50))
    merged_boxes = _merge_overlapping_boxes(raw_fragment_boxes, gap_px)

    # Full frame grayscale image for Belt Context validation
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    valid_candidates: List[Tuple[float, BoundingBox]] = []

    for bx_roi, by_roi, bw, bh in merged_boxes:
        box_area = bw * bh
        if box_area < cfg.MIN_CONTOUR_AREA or box_area > effective_max_contour_area:
            print(f"[DEBUG] Rejected candidate: area out of bounds ({box_area} vs min {cfg.MIN_CONTOUR_AREA}, max {effective_max_contour_area:.0f})")
            continue

        # Aspect ratio check
        r = float(bw) / float(bh)
        aspect = r if r >= 1.0 else 1.0 / r
        if aspect < cfg.MIN_ASPECT_RATIO or aspect > cfg.MAX_ASPECT_RATIO:
            print(f"[DEBUG] Rejected candidate: aspect ratio invalid ({aspect:.2f} vs min {cfg.MIN_ASPECT_RATIO}, max {cfg.MAX_ASPECT_RATIO})")
            continue

        # Glare / Overexposure Sanity Check (Rejects featureless blown-out glare hotspots e.g. on motor/clamp)
        gray_candidate = gray_roi[by_roi:by_roi + bh, bx_roi:bx_roi + bw]
        if gray_candidate.size > 0:
            blownout_pixels = np.count_nonzero(gray_candidate >= cfg.GRAY_BLOWNOUT_THRESHOLD)
            blownout_fraction = float(blownout_pixels) / float(gray_candidate.size)
            if blownout_fraction > cfg.MAX_BLOWNOUT_PIXEL_FRACTION:
                print(f"[DEBUG] Rejected candidate: blown-out glare ({blownout_fraction * 100.0:.2f}% saturated > {cfg.MAX_BLOWNOUT_PIXEL_FRACTION * 100.0:.1f}%)")
                continue

        # Translate bounding box coordinates from ROI space back to full-frame space
        x_full = bx_roi + rx1
        y_full = by_roi + ry1

        # 4. Relative Context Contrast Validation (Lighting-Invariant):
        # Candidate must be meaningfully brighter than its immediate surrounding top/bottom belt margin
        candidate_mean = float(gray_candidate.mean()) if gray_candidate.size > 0 else 0.0

        margin_h = max(cfg.BELT_CONTEXT_MIN_MARGIN_PIXELS, bh // 2)
        mx1 = max(0, x_full + int(bw * 0.2))
        mx2 = min(w, x_full + int(bw * 0.8))
        if mx2 <= mx1:
            mx1, mx2 = max(0, x_full), min(w, x_full + bw)

        # Top margin strip (above joint)
        top_y1 = max(0, y_full - margin_h)
        top_y2 = y_full
        top_strip = gray_full[top_y1:top_y2, mx1:mx2]

        # Bottom margin strip (below joint)
        bot_y1 = y_full + bh
        bot_y2 = min(h, y_full + bh + margin_h)
        bot_strip = gray_full[bot_y1:bot_y2, mx1:mx2]

        top_mean = float(top_strip.mean()) if top_strip.size > 0 else candidate_mean
        bot_mean = float(bot_strip.mean()) if bot_strip.size > 0 else candidate_mean
        margin_mean = (top_mean + bot_mean) / 2.0
        contrast_delta = candidate_mean - margin_mean

        min_contrast = getattr(cfg, "MIN_CONTEXT_CONTRAST", 15.0)
        if contrast_delta < min_contrast:
            print(f"[DEBUG] Rejected candidate: low context contrast (delta {contrast_delta:.1f} < min {min_contrast:.1f} | candidate={candidate_mean:.1f}, margin={margin_mean:.1f})")
            continue  # Reject candidate that is not brighter than surrounding belt

        bbox = BoundingBox(x=x_full, y=y_full, w=bw, h=bh)
        valid_candidates.append((float(box_area), bbox))

    if not valid_candidates:
        return None

    # Pick the single best candidate (largest valid contour area)
    valid_candidates.sort(key=lambda item: item[0], reverse=True)
    return valid_candidates[0][1]


# Function attribute for mask storage
detect_joint.last_mask = None
