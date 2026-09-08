"""
JointGuard Computer Vision Configuration (Phase 1 Classical CV)

All tunable vision, threshold, tracking, and feature classification parameters are centralized
here to allow easy live tuning against physical webcam setups and lighting conditions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any


@dataclass
class VisionConfig:
    """Configuration class containing all tunable parameters for OpenCV detection & classification."""

    # -------------------------------------------------------------------------
    # 1. DETECTOR TUNABLE THRESHOLDS (Metal Patch on Black Belt)
    # -------------------------------------------------------------------------
    # HSV Threshold range for metallic foil patch (high brightness V, low saturation S)
    # Belt rubber is dark/black (low V); foil reflects light (high V, metallic low/moderate S)
    HSV_LOWER: np.ndarray = field(default_factory=lambda: np.array([0, 0, 160], dtype=np.uint8))
    HSV_UPPER: np.ndarray = field(default_factory=lambda: np.array([180, 65, 255], dtype=np.uint8))

    # Grayscale fallback threshold disabled by default to prevent ambient light blowout
    USE_GRAY_FALLBACK: bool = False
    GRAY_THRESHOLD_MIN: int = 180
    GRAY_THRESHOLD_MAX: int = 255

    # Morphological closing kernel size (fuses reflective foil patch fragments into one patch region)
    MORPH_KERNEL_SIZE: Tuple[int, int] = (15, 15)

    # Live Capture Denoising (Reduces sensor graininess on live webcam stream)
    LIVE_DENOISE_KERNEL: Tuple[int, int] = (3, 3)

    # Bounding Box Contour Filters & Clustering (Joint foil is a wide rectangular patch)
    MIN_CONTOUR_AREA: int = 2500         # Min pixel area of joint foil candidate (filters out tiny noise boxes)
    MAX_CONTOUR_AREA: int = 2000000      # Max pixel area fallback (overridden by ROI area fraction if ROI active)
    MAX_CONTOUR_AREA_FRACTION_OF_ROI: float = 1.0  # Max candidate area relative to ROI area
    MIN_ASPECT_RATIO: float = 1.0        # Aspect ratio = width / height (or height/width, min 1.0 for square patches)
    MAX_ASPECT_RATIO: float = 20.0       # Upper bound for rectangular joint patch shape (allows wide/long metallic foil joints)
    CONTOUR_CLUSTER_MAX_GAP_PX: int = 60 # Max pixel gap between nearby patch fragments to cluster them into one candidate box
    CONTOUR_MERGE_GAP_PIXELS: int = 60   # Alias for backward compatibility

    # Relative Context Contrast Validation (Lighting-Invariant)
    # Checks candidate mean brightness minus surrounding belt margin mean brightness. Must be >= MIN_CONTEXT_CONTRAST
    MIN_CONTEXT_CONTRAST: float = 10.0  # Must be live-calibrated via calibrate.py (min brightness delta above belt)
    BELT_CONTEXT_MIN_MARGIN_PIXELS: int = 10   # Minimum pixel thickness for top/bottom margin strips

    # Glare / Overexposure Protection (Rejects featureless blown-out glare hotspots e.g. on motor/clamp)
    GRAY_BLOWNOUT_THRESHOLD: int = 254         # Grayscale value threshold for blown-out pixels
    MAX_BLOWNOUT_PIXEL_FRACTION: float = 0.85   # Max allowed fraction of near-255 pixels before candidate is rejected as glare

    # Frame Border Margin (in pixels): candidate bbox within this distance from frame edge
    # is considered partially occluded/entering/exiting frame.
    EDGE_MARGIN_PIXELS: int = 15

    # -------------------------------------------------------------------------
    # 2. INSPECTION REGION OF INTEREST (ROI) & CAPTURE ZONE (Continuous Motion)
    # -------------------------------------------------------------------------
    # Normalized outer inspection ROI boundaries [x_min, y_min, x_max, y_max] (0.0 - 1.0)
    # Scaled to focus on the conveyor belt (x: 0.10 to 0.90) and exclude external background
    ROI_X_MIN: float = 0.10
    ROI_Y_MIN: float = 0.05
    ROI_X_MAX: float = 0.90
    ROI_Y_MAX: float = 0.95

    # Inner Capture Zone (centered inside outer ROI for continuous belt motion)
    CAPTURE_ZONE_X_MIN: float = 0.20
    CAPTURE_ZONE_Y_MIN: float = 0.15
    CAPTURE_ZONE_X_MAX: float = 0.80
    CAPTURE_ZONE_Y_MAX: float = 0.85
    CAPTURE_ZONE_MIN_FRAMES: int = 2      # Min consecutive frames inside capture zone to trigger INSPECTING
    CAPTURE_ZONE_MAX_FRAMES: int = 10     # Max frames window to aggregate feature snapshots

    # Motion Blur Awareness & Velocity Validation
    MIN_SHARPNESS_SCORE: float = 30.0     # Min variance of Laplacian score for motion blur rejection
    MAX_VELOCITY_VARIANCE: float = 25.0   # Max allowed centroid speed delta between frames

    # Tracker Centroid Distance Matching
    MAX_CENTROID_DISTANCE: float = 100.0  # Max pixel jump between consecutive frames for same track
    MAX_DISAPPEARED_FRAMES: int = 15      # Frames before an inactive track is retired

    # Target Camera FPS
    TARGET_CAMERA_FPS: int = 30

    # -------------------------------------------------------------------------
    # 3. FEATURE EXTRACTION & DAMAGE CLASSIFICATION (Must be live-calibrated)
    # -------------------------------------------------------------------------
    # Feature 1: Canny Edge Detection (Edge Density)
    CANNY_THRESHOLD1: float = 50.0
    CANNY_THRESHOLD2: float = 150.0
    EDGE_DENSITY_MAX_EXPECTED: float = 0.25  # Calibrated for live webcam & real brushed metallic foil

    # Feature 2: Hough Line Crease Detection
    HOUGH_RHO: float = 1.0
    HOUGH_THETA: float = np.pi / 180.0
    HOUGH_THRESHOLD: int = 15              # Crease detection threshold
    HOUGH_MIN_LINE_LENGTH: int = 15
    HOUGH_MAX_LINE_GAP: int = 5

    # Diagonal crease angles relative to horizontal axis (in degrees, 15 to 75 deg)
    HOUGH_DIAGONAL_MIN_ANGLE: float = 15.0
    HOUGH_DIAGONAL_MAX_ANGLE: float = 75.0

    # Feature 3: Local Intensity Variance (Patchiness / Specular reflections)
    VARIANCE_BLUR_KERNEL: Tuple[int, int] = (7, 7)
    VARIANCE_MAX_EXPECTED: float = 95.0  # Calibrated for live webcam & physical brushed metallic foil

    # Feature Weights in combined score (sum up to 1.0)
    # Intensity variance and structural patch continuity weighted higher to distinguish smooth healthy foil from creased/patchy damaged foil
    WEIGHT_EDGE_DENSITY: float = 0.10
    WEIGHT_HOUGH_LINES: float = 0.10
    WEIGHT_INTENSITY_VARIANCE: float = 0.30

    # Final Classification Score Threshold (0 - 100 score)
    # Score >= HEALTHY_THRESHOLD -> HEALTHY, else DAMAGE
    HEALTHY_THRESHOLD_SCORE: float = 54.0  # Calibrated for physical metallic foil patches (Healthy ~56.5, Damaged ~51.9)
    THRESHOLD_HYSTERESIS: float = 5.0      # Hysteresis band around threshold to prevent flickering label flips


# Global default instance
DEFAULT_CONFIG = VisionConfig()
