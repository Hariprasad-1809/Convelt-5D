"""
JointGuard Feature Extraction & Preprocessing Module

Provides high-speed, lighting-robust texture, gradient, and structural feature
extraction for joint classification. Runs in < 2ms on CPU.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional


def preprocess_joint_crop(crop: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Standardizes a cropped joint region to a consistent canonical input size and format.

    Args:
        crop: Cropped BGR image of the joint region.
        target_size: Desired (width, height) tuple (default 128x128).

    Returns:
        Resized 3-channel BGR image.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    if len(crop.shape) == 2:
        bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    elif crop.shape[2] == 4:
        bgr = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
    else:
        bgr = crop

    # Preserve metallic joint texture using INTER_AREA for downscaling
    h, w = bgr.shape[:2]
    interp = cv2.INTER_AREA if (w >= target_size[0] and h >= target_size[1]) else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, target_size, interpolation=interp)
    return resized


def preprocess_for_mobilenet(crop: np.ndarray, target_size: Tuple[int, int] = (224, 224)):
    """
    Standardizes a cropped joint region into an ImageNet-normalized PyTorch tensor for MobileNetV2.
    """
    import torch
    norm_crop = preprocess_joint_crop(crop, target_size)
    rgb = cv2.cvtColor(norm_crop, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    norm_tensor = (tensor - mean) / std
    return norm_tensor.unsqueeze(0)


def extract_feature_vector(crop: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Extracts a dense, 1D invariant numerical feature vector from a joint crop.

    Features include:
      - 16-bin Grayscale intensity distribution
      - 16-bin HSV Value channel distribution
      - 8-bin HSV Saturation distribution
      - 32-bin 4-Quadrant Gradient Orientation Histograms (structural fold & crease directions)
      - Gradient magnitude distribution moments (mean, std, 25%, 50%, 75%, 90% percentiles)
      - Surface roughness via Laplacian variance
      - Multi-scale Canny edge density
      - Horizontal and vertical intensity profile gradients (sensitive to transverse tears)

    Returns:
        np.ndarray of shape (86,), dtype float32.
    """
    norm_crop = preprocess_joint_crop(crop, target_size)
    gray = cv2.cvtColor(norm_crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(norm_crop, cv2.COLOR_BGR2HSV)

    # 1. Grayscale intensity distribution (16 bins)
    hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist_gray /= (hist_gray.sum() + 1e-6)

    # 2. HSV Value & Saturation distributions (24 bins total)
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    hist_v /= (hist_v.sum() + 1e-6)
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_s /= (hist_s.sum() + 1e-6)

    # 3. Directional Sobel Gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0

    # 4. Global & Quadrant Gradient Orientation Histograms (32 bins: 4 quadrants x 8 bins)
    bin_idx = np.clip((ang / 22.5).astype(int), 0, 7)
    tw, th = target_size
    mid_w, mid_h = tw // 2, th // 2
    quad_hists = []
    for q_mag, q_bin in [
        (mag[:mid_h, :mid_w], bin_idx[:mid_h, :mid_w]),
        (mag[:mid_h, mid_w:], bin_idx[:mid_h, mid_w:]),
        (mag[mid_h:, :mid_w], bin_idx[mid_h:, :mid_w]),
        (mag[mid_h:, mid_w:], bin_idx[mid_h:, mid_w:])
    ]:
        qh = np.bincount(q_bin.flatten(), weights=q_mag.flatten(), minlength=8).astype(np.float32)
        norm = float(np.linalg.norm(qh)) + 1e-6
        quad_hists.append(qh / norm)
    quad_feats = np.concatenate(quad_hists)

    # 5. Gradient magnitude moments (6 features)
    grad_stats = np.array([
        float(mag.mean()),
        float(mag.std()),
        float(np.percentile(mag, 25)),
        float(np.percentile(mag, 50)),
        float(np.percentile(mag, 75)),
        float(np.percentile(mag, 90))
    ], dtype=np.float32)

    # 6. Surface roughness & multi-scale edge density (3 features)
    lap_var = np.array([float(cv2.Laplacian(gray, cv2.CV_32F).var())], dtype=np.float32)
    edge_50_150 = float(np.count_nonzero(cv2.Canny(gray, 50, 150))) / float(gray.size)
    edge_30_100 = float(np.count_nonzero(cv2.Canny(gray, 30, 100))) / float(gray.size)
    edge_stats = np.array([edge_50_150, edge_30_100], dtype=np.float32)

    # 7. Structural Profile Gradients (5 features)
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    row_diffs = np.diff(row_means)
    col_diffs = np.diff(col_means)
    profile_stats = np.array([
        float(row_means.std()),
        float(col_means.std()),
        float(row_means.max() - row_means.min()),
        float(np.abs(row_diffs).mean()),
        float(np.abs(col_diffs).mean())
    ], dtype=np.float32)

    feature_vec = np.concatenate([
        hist_gray,
        hist_v,
        hist_s,
        quad_feats,
        grad_stats,
        lap_var,
        edge_stats,
        profile_stats
    ]).astype(np.float32)

    return feature_vec


def extract_human_readable_features(crop: np.ndarray) -> Dict[str, float]:
    """
    Extracts descriptive texture and edge features for telemetry logging and debugging.
    """
    if crop is None or crop.size == 0:
        return {
            "edge_density": 0.0,
            "intensity_variance": 0.0,
            "surface_roughness": 0.0,
            "gradient_mean": 0.0,
            "profile_variation": 0.0,
            "hough_line_score": 0.0
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(sobel_x, sobel_y)

    lines = cv2.HoughLinesP(edges, 1.0, np.pi / 180.0, 15, minLineLength=15, maxLineGap=5)
    hough_score = float(len(lines)) / 10.0 if lines is not None else 0.0

    return {
        "edge_density": round(float(np.count_nonzero(edges)) / float(gray.size), 4),
        "intensity_variance": round(float(gray.std()), 2),
        "surface_roughness": round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 2),
        "gradient_mean": round(float(mag.mean()), 2),
        "profile_variation": round(float(gray.mean(axis=1).std()), 2),
        "hough_line_score": round(hough_score, 4)
    }
