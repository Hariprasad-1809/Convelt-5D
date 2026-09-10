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

    # 1. Candidate search boundary:
    # Strictly limit joint candidate search to CAPTURE ZONE when DETECT_STRICT_CAPTURE_ZONE is True
    if getattr(cfg, "DETECT_STRICT_CAPTURE_ZONE", True):
        rx1 = max(0, int(cfg.CAPTURE_ZONE_X_MIN * w))
        ry1 = max(0, int(cfg.CAPTURE_ZONE_Y_MIN * h))
        rx2 = min(w, int(cfg.CAPTURE_ZONE_X_MAX * w))
        ry2 = min(h, int(cfg.CAPTURE_ZONE_Y_MAX * h))
    else:
        rx1 = max(0, int(cfg.ROI_X_MIN * w))
        ry1 = max(0, int(cfg.ROI_Y_MIN * h))
        rx2 = min(w, int(cfg.ROI_X_MAX * w))
        ry2 = min(h, int(cfg.ROI_Y_MAX * h))

    if rx2 <= rx1 or ry2 <= ry1:
        _last_detection_mask = None
        return None

    # Crop frame to search region before thresholding & contour search
    roi_frame = frame[ry1:ry2, rx1:rx2].copy()
    roi_h, roi_w = roi_frame.shape[:2]
    roi_area = float(roi_h * roi_w)

    # Suppress static side rail reflections if search region begins near the extreme left edge (x < 0.17 * w)
    rail_bound = int(0.17 * w) - rx1
    if rail_bound > 0:
        roi_frame[:, :rail_bound] = 0

    # Effective max contour area relative to ROI pixel area (capped at exact ROI area)
    max_roi_fraction = float(getattr(cfg, "MAX_CONTOUR_AREA_FRACTION_OF_ROI", 1.0))
    effective_max_contour_area = min(float(cfg.MAX_CONTOUR_AREA), roi_area * max_roi_fraction)

    # 2. Convert ROI to HSV and generate dynamic metallic brightness mask
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    v_chan = hsv_roi[:, :, 2]
    v_median = float(np.median(v_chan)) if v_chan.size > 0 else 120.0

    # Dynamic threshold: metallic foil must be brighter than median belt in ROI
    dynamic_delta = float(getattr(cfg, "HSV_DYNAMIC_DELTA", 15.0))
    v_lower = max(int(cfg.HSV_LOWER[2]), min(210, int(v_median + dynamic_delta)))
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

    # Apply Morphological Closing to fuse fragmented foil reflections into unified joint candidate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.MORPH_KERNEL_SIZE)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Build full-frame mask visualization (ROI mask embedded in full black frame)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = closed_mask
    _last_detection_mask = full_mask
    detect_joint.last_mask = full_mask

    # 3. Find Contours within ROI mask
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect initial candidate fragment boxes, excluding border-clinging skinny vertical rails
    raw_fragment_boxes = []
    for cnt in contours:
        c_area = cv2.contourArea(cnt)
        if c_area >= 200:  # Min fragment area to consider for bounding box fusion
            bx_roi, by_roi, bw, bh = cv2.boundingRect(cnt)
            if bw > 0 and bh > 0:
                # Discard skinny vertical side rail strip clinging to ROI outer edge (must be narrow: bw < 40 and bh > 2*bw)
                is_side_rail = (
                    (bx_roi <= 3 or bx_roi + bw >= roi_w - 3) and
                    (bw < 40 and bh > 2 * bw)
                )
                if not is_side_rail:
                    raw_fragment_boxes.append((bx_roi, by_roi, bw, bh))

    # Fuse nearby fragment boxes into unified joint patch bounding boxes
    gap_px = getattr(cfg, "CONTOUR_CLUSTER_MAX_GAP_PX", getattr(cfg, "CONTOUR_MERGE_GAP_PIXELS", 60))
    merged_boxes = _merge_overlapping_boxes(raw_fragment_boxes, gap_px)

    # Full frame grayscale image for Belt Context validation
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    valid_candidates: List[Tuple[float, BoundingBox]] = []
    fallback_candidates: List[Tuple[float, BoundingBox]] = []
    rejected_diagnostics: List[Dict[str, Any]] = []

    for bx_roi, by_roi, bw, bh in merged_boxes:
        box_area = bw * bh
        x_full = bx_roi + rx1
        y_full = by_roi + ry1

        r = float(bw) / float(bh)
        aspect = r if r >= 1.0 else 1.0 / r

        # Area check (standard min area, or relaxed min area for entering edges)
        min_area_thresh = cfg.MIN_CONTOUR_AREA
        if box_area < min_area_thresh or box_area > effective_max_contour_area:
            rejected_diagnostics.append({
                "bbox": (x_full, y_full, bw, bh),
                "area": box_area,
                "aspect": aspect,
                "contrast_delta": 0.0,
                "candidate_mean": 0.0,
                "margin_mean": 0.0,
                "reason": f"area out of bounds ({box_area} vs min {min_area_thresh}, max {effective_max_contour_area:.0f})"
            })
            continue

        # Two-tier aspect ratio evaluation for continuous belt motion:
        # Tier 1: Standard aspect ratio for fully framed joints
        # Tier 2: Relaxed fallback for partially entered or motion-skewed joints
        min_aspect_relaxed = getattr(cfg, "MIN_ASPECT_RATIO_RELAXED", 0.4)
        max_aspect_relaxed = getattr(cfg, "MAX_ASPECT_RATIO_RELAXED", 35.0)
        allow_fallback = getattr(cfg, "ALLOW_PARTIAL_ENTRY_FALLBACK", True)

        is_standard_aspect = (aspect >= cfg.MIN_ASPECT_RATIO and aspect <= cfg.MAX_ASPECT_RATIO)
        is_relaxed_aspect = allow_fallback and (aspect >= min_aspect_relaxed and aspect <= max_aspect_relaxed)

        if not is_standard_aspect and not is_relaxed_aspect:
            rejected_diagnostics.append({
                "bbox": (x_full, y_full, bw, bh),
                "area": box_area,
                "aspect": aspect,
                "contrast_delta": 0.0,
                "candidate_mean": 0.0,
                "margin_mean": 0.0,
                "reason": f"aspect ratio invalid ({aspect:.2f} vs min {cfg.MIN_ASPECT_RATIO}, max {cfg.MAX_ASPECT_RATIO})"
            })
            continue

        # Glare / Overexposure Sanity Check (Rejects featureless blown-out glare hotspots e.g. on motor/clamp)
        gray_candidate = gray_roi[by_roi:by_roi + bh, bx_roi:bx_roi + bw]
        if gray_candidate.size > 0:
            blownout_pixels = np.count_nonzero(gray_candidate >= cfg.GRAY_BLOWNOUT_THRESHOLD)
            blownout_fraction = float(blownout_pixels) / float(gray_candidate.size)
            if blownout_fraction > cfg.MAX_BLOWNOUT_PIXEL_FRACTION:
                rejected_diagnostics.append({
                    "bbox": (x_full, y_full, bw, bh),
                    "area": box_area,
                    "aspect": aspect,
                    "contrast_delta": 0.0,
                    "candidate_mean": float(gray_candidate.mean()),
                    "margin_mean": 0.0,
                    "reason": f"blown-out glare ({blownout_fraction * 100.0:.1f}% saturated > {cfg.MAX_BLOWNOUT_PIXEL_FRACTION * 100.0:.1f}%)"
                })
                continue

        # 4. Relative Context Contrast Validation (Lighting-Invariant):
        # Candidate must be brighter than its immediate surrounding dark belt rubber margin
        candidate_mean = float(gray_candidate.mean()) if gray_candidate.size > 0 else 0.0

        margin_h = max(cfg.BELT_CONTEXT_MIN_MARGIN_PIXELS, min(30, bh // 4))
        mx1 = max(0, x_full + int(bw * 0.2))
        mx2 = min(w, x_full + int(bw * 0.8))
        if mx2 <= mx1:
            mx1, mx2 = max(0, x_full), min(w, x_full + bw)

        # Sample top margin strip from full frame (dark belt rubber)
        top_y1 = max(0, y_full - margin_h)
        top_y2 = y_full
        if (top_y2 - top_y1) >= 3 and (mx2 - mx1) > 2:
            top_strip = gray_full[top_y1:top_y2, mx1:mx2]
            top_mean = float(top_strip.mean()) if top_strip.size > 0 else None
        else:
            top_mean = None

        # Sample bottom margin strip from full frame (dark belt rubber)
        bot_y1 = y_full + bh
        bot_y2 = min(h, y_full + bh + margin_h)
        if (bot_y2 - bot_y1) >= 3 and (mx2 - mx1) > 2:
            bot_strip = gray_full[bot_y1:bot_y2, mx1:mx2]
            bot_mean = float(bot_strip.mean()) if bot_strip.size > 0 else None
        else:
            bot_mean = None

        # Determine belt context brightness (exclude exterior non-belt background strips)
        if top_mean is not None and bot_mean is not None:
            # Both strips on belt: use darker strip representing true dark rubber belt
            margin_mean = min(top_mean, bot_mean)
        elif bot_mean is not None:
            margin_mean = bot_mean
        elif top_mean is not None:
            margin_mean = top_mean
        else:
            # Candidate covers full vertical height of frame; reject uniform background
            margin_mean = candidate_mean

        contrast_delta = candidate_mean - margin_mean
        min_contrast = float(getattr(cfg, "MIN_CONTEXT_CONTRAST", 2.0))

        if contrast_delta < min_contrast:
            rejected_diagnostics.append({
                "bbox": (x_full, y_full, bw, bh),
                "area": box_area,
                "aspect": aspect,
                "contrast_delta": contrast_delta,
                "candidate_mean": candidate_mean,
                "margin_mean": margin_mean,
                "reason": f"low context contrast (delta {contrast_delta:.1f} < min {min_contrast:.1f} | candidate={candidate_mean:.1f}, margin={margin_mean:.1f})"
            })
            continue

        # Score candidate with belt-centrality weighting (favors joint centered in conveyor belt)
        cx_roi = bx_roi + bw / 2.0
        center_offset = abs(cx_roi - (roi_w / 2.0)) / (roi_w / 2.0)
        centrality_factor = max(0.2, 1.0 - 0.5 * center_offset)
        candidate_score = float(box_area) * centrality_factor

        bbox = BoundingBox(x=x_full, y=y_full, w=bw, h=bh)
        if is_standard_aspect:
            valid_candidates.append((candidate_score, bbox))
        else:
            # Fallback candidate for partially entered/skewed joint during belt motion
            fallback_candidates.append((candidate_score * 0.7, bbox))

    # Prefer standard aspect ratio candidates; if none exist (e.g. entering joint), use fallback candidate
    if valid_candidates:
        valid_candidates.sort(key=lambda item: item[0], reverse=True)
        return valid_candidates[0][1]
    elif fallback_candidates:
        fallback_candidates.sort(key=lambda item: item[0], reverse=True)
        return fallback_candidates[0][1]

    if rejected_diagnostics:
        # Log the single top candidate (largest area) with computed contrast & area values
        top_rej = max(rejected_diagnostics, key=lambda c: c["area"])
        print(f"[DEBUG] Top rejected candidate: bbox={top_rej['bbox']} | area={top_rej['area']} (min: {cfg.MIN_CONTOUR_AREA}) | contrast_delta={top_rej['contrast_delta']:.1f} (min: {cfg.MIN_CONTEXT_CONTRAST:.1f} | cand={top_rej['candidate_mean']:.1f}, margin={top_rej['margin_mean']:.1f}) | reason={top_rej['reason']}")
    return None


# Function attribute for mask storage
detect_joint.last_mask = None
