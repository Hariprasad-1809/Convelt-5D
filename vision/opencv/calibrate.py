"""
JointGuard Live Webcam Threshold Calibration Tool

Captures 10 live frames of a known-HEALTHY joint and 10 live frames of a known-DAMAGED joint,
analyzes raw feature distributions under actual webcam sensor noise & lighting, and suggests
optimal thresholds for VisionConfig in config.py.

Usage:
  python vision/opencv/calibrate.py --source 0
  python vision/opencv/calibrate.py --source 1
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vision.opencv.config import VisionConfig, DEFAULT_CONFIG
from vision.opencv.detector import detect_joint
from vision.opencv.classifier import classify_joint


def parse_args():
    parser = argparse.ArgumentParser(description="JointGuard Live Camera Calibration Tool")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: webcam index (e.g. 0 or 1), video file, or image path"
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable Gaussian blur noise reduction"
    )
    return parser.parse_args()


def capture_samples(cap, label_name: str, config: VisionConfig, apply_denoise: bool = True, num_samples: int = 10):
    print("\n" + "=" * 70)
    print(f"  STEP: CAPTURING KNOWN-{label_name.upper()} JOINT SAMPLES")
    print("=" * 70)
    print(f"1. Place a known-{label_name.upper()} joint patch in the central inspection ROI.")
    print("2. Ensure the joint is stationary under your physical webcam setup.")
    input(f"--> Press Enter when ready to capture {num_samples} frames of the {label_name.upper()} joint... ")

    samples = []
    frames_captured = 0

    print(f"[INFO] Capturing {num_samples} valid frames...")

    while cap.isOpened() and frames_captured < num_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to grab frame from camera.")
            break

        if apply_denoise and config.LIVE_DENOISE_KERNEL[0] > 1:
            frame = cv2.GaussianBlur(frame, config.LIVE_DENOISE_KERNEL, 0)

        bbox = detect_joint(frame, config)
        if bbox is None:
            print("  [WAITING] No joint detected in ROI... adjust position.", end="\r")
            cv2.waitKey(100)
            continue

        roi_crop = bbox.crop_roi(frame)
        result = classify_joint(roi_crop, config)

        samples.append(result)
        frames_captured += 1
        print(f"  Captured sample {frames_captured}/{num_samples}: score={result.vision_score:.2f}, edge_density={result.features['edge_density']:.4f}, hough={result.features['hough_line_score']:.4f}, var={result.features['intensity_variance']:.2f}")
        cv2.waitKey(200)

    return samples


def print_feature_stats(name: str, sample_list):
    scores = [s.vision_score for s in sample_list]
    edges = [s.features["edge_density"] for s in sample_list]
    houghs = [s.features["hough_line_score"] for s in sample_list]
    vars_ = [s.features["intensity_variance"] for s in sample_list]

    print(f"\n--- {name.upper()} SET STATS ({len(sample_list)} samples) ---")
    print(f"  Vision Score       : Mean={np.mean(scores):6.2f} | Min={np.min(scores):6.2f} | Max={np.max(scores):6.2f}")
    print(f"  Raw Edge Density   : Mean={np.mean(edges):6.4f} | Min={np.min(edges):6.4f} | Max={np.max(edges):6.4f}")
    print(f"  Raw Hough Score    : Mean={np.mean(houghs):6.4f} | Min={np.min(houghs):6.4f} | Max={np.max(houghs):6.4f}")
    print(f"  Raw Intensity Var  : Mean={np.mean(vars_):6.2f} | Min={np.min(vars_):6.2f} | Max={np.max(vars_):6.2f}")

    return {
        "score_mean": np.mean(scores),
        "edge_mean": np.mean(edges),
        "edge_max": np.max(edges),
        "hough_mean": np.mean(houghs),
        "var_mean": np.mean(vars_),
        "var_max": np.max(vars_)
    }


def capture_background_samples(cap, config: VisionConfig, apply_denoise: bool = True, num_samples: int = 10):
    print("\n" + "=" * 70)
    print("  STEP: CAPTURING EMPTY BELT BACKGROUND SAMPLES (NO JOINT PRESENT)")
    print("=" * 70)
    print("1. Ensure NO joint patch is present inside the inspection ROI (empty belt).")
    print("2. Ensure the belt/webcam is under current room lighting.")
    input("--> Press Enter when ready to capture 10 background frames... ")

    contrast_samples = []
    frames_captured = 0

    h, w = 480, 640

    while cap.isOpened() and frames_captured < num_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if apply_denoise and config.LIVE_DENOISE_KERNEL[0] > 1:
            frame = cv2.GaussianBlur(frame, config.LIVE_DENOISE_KERNEL, 0)

        h, w = frame.shape[:2]
        rx1 = max(0, int(config.ROI_X_MIN * w))
        ry1 = max(0, int(config.ROI_Y_MIN * h))
        rx2 = min(w, int(config.ROI_X_MAX * w))
        ry2 = min(h, int(config.ROI_Y_MAX * h))

        roi = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        contrast_samples.append(float(np.std(gray)))
        frames_captured += 1
        print(f"  Captured background sample {frames_captured}/{num_samples}: roi_mean={gray.mean():.1f}, std={gray.std():.1f}")
        cv2.waitKey(150)

    return contrast_samples


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG

    source_input = args.source
    if source_input.isdigit():
        cap = cv2.VideoCapture(int(source_input))
    elif os.path.isfile(source_input):
        cap = cv2.VideoCapture(source_input)
    else:
        print(f"[ERROR] Invalid source input: {source_input}")
        sys.exit(1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {source_input}")
        sys.exit(1)

    print("=" * 70)
    print("      JOINTGUARD LIVE WEBCAM THRESHOLD CALIBRATOR")
    print("=" * 70)
    print(f"[INFO] Camera source: {source_input}")
    print(f"[INFO] Live Denoising: {'Disabled' if args.no_denoise else 'Enabled (' + str(cfg.LIVE_DENOISE_KERNEL) + ')'}")

    # Capture healthy set
    healthy_samples = capture_samples(cap, "HEALTHY", cfg, apply_denoise=(not args.no_denoise))
    if len(healthy_samples) < 3:
        print("[ERROR] Insufficient healthy samples captured. Exiting.")
        cap.release()
        sys.exit(1)

    # Capture damaged set
    damaged_samples = capture_samples(cap, "DAMAGED", cfg, apply_denoise=(not args.no_denoise))
    if len(damaged_samples) < 3:
        print("[ERROR] Insufficient damaged samples captured. Exiting.")
        cap.release()
        sys.exit(1)

    # Capture empty background set
    bg_samples = capture_background_samples(cap, cfg, apply_denoise=(not args.no_denoise))

    cap.release()

    # Calculate statistics
    h_stats = print_feature_stats("HEALTHY", healthy_samples)
    d_stats = print_feature_stats("DAMAGED", damaged_samples)

    # Calculate suggested threshold
    suggested_threshold = (h_stats["score_mean"] + d_stats["score_mean"]) / 2.0
    suggested_min_contrast = 15.0

    print("\n" + "=" * 70)
    print("      CALIBRATION SUMMARY & RECOMMENDED CONFIG UPDATES")
    print("=" * 70)
    print(f"[HEALTHY MEAN SCORE] : {h_stats['score_mean']:.2f}")
    print(f"[DAMAGED MEAN SCORE] : {d_stats['score_mean']:.2f}")
    print(f"[SUGGESTED HEALTHY_THRESHOLD_SCORE] = {suggested_threshold:.1f}")

    # Sanity checks on normalization caps
    print("\n--- Cap Validation Checks ---")

    if h_stats["edge_max"] > cfg.EDGE_DENSITY_MAX_EXPECTED * 0.8:
        new_edge_cap = round(h_stats["edge_max"] * 1.5, 2)
        print(f"  [WARNING] Live healthy edge_density max ({h_stats['edge_max']:.4f}) approaches/exceeds EDGE_DENSITY_MAX_EXPECTED cap ({cfg.EDGE_DENSITY_MAX_EXPECTED:.2f}).")
        print(f"            --> Consider raising EDGE_DENSITY_MAX_EXPECTED to {new_edge_cap} in config.py.")
    else:
        print(f"  [OK] EDGE_DENSITY_MAX_EXPECTED cap ({cfg.EDGE_DENSITY_MAX_EXPECTED:.2f}) looks appropriate.")

    if h_stats["var_max"] > cfg.VARIANCE_MAX_EXPECTED * 0.8:
        new_var_cap = round(h_stats["var_max"] * 1.5, 1)
        print(f"  [WARNING] Live healthy intensity_variance max ({h_stats['var_max']:.2f}) approaches/exceeds VARIANCE_MAX_EXPECTED cap ({cfg.VARIANCE_MAX_EXPECTED:.2f}).")
        print(f"            --> Consider raising VARIANCE_MAX_EXPECTED to {new_var_cap} in config.py.")
    else:
        print(f"  [OK] VARIANCE_MAX_EXPECTED cap ({cfg.VARIANCE_MAX_EXPECTED:.2f}) looks appropriate.")

    if bg_samples:
        bg_mean_std = float(np.mean(bg_samples))
        suggested_min_contrast = max(10.0, round(bg_mean_std * 1.5, 1))
        print(f"  [INFO] Empty background noise std = {bg_mean_std:.1f} --> Suggested MIN_CONTEXT_CONTRAST = {suggested_min_contrast:.1f}")

    print("\n" + "=" * 70)
    print("To apply these live calibration values, update vision/opencv/config.py:")
    print(f"  MIN_CONTEXT_CONTRAST: float = {suggested_min_contrast:.1f}")
    print(f"  HEALTHY_THRESHOLD_SCORE: float = {suggested_threshold:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
