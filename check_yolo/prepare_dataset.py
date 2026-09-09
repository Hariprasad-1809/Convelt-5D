#!/usr/bin/env python3
"""
JointGuard YOLO Classification - Dataset Preparation Script

Copies reference conveyor-belt joint images (healthy and damaged), applies
conservative, physically plausible data augmentation, and organizes them
into standard Ultralytics YOLO classification directory format:

check_yolo/dataset/
├── train/
│   ├── healthy/
│   │   ├── healthy_001.jpg (original reference)
│   │   └── healthy_aug_001.jpg ... (conservative augmentations)
│   └── damage/
│       ├── damage_001.jpg (original reference)
│       └── damage_aug_001.jpg ... (conservative augmentations)
└── val/
    ├── healthy/
    │   └── healthy_001.jpg (reference validation)
    └── damage/
        └── damage_001.jpg (reference validation)

IMPORTANT:
- No data leakage: augmented images are strictly kept in train/.
- The validation split is a structural placeholder required by YOLO's training loop.
  It is NOT an independent statistical validation benchmark.
"""

import argparse
import os
import shutil
import sys
import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DEFAULT_SRC_HEALTHY = os.path.join(
    PROJECT_ROOT, "jointgurd_proto", "vision", "opencv", "dataset", "healthy", "healthy_001.jpg"
)
DEFAULT_SRC_DAMAGE = os.path.join(
    PROJECT_ROOT, "jointgurd_proto", "vision", "opencv", "dataset", "damage", "damage_001.jpg"
)
DEFAULT_DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and augment dataset for JointGuard YOLO classification experiment"
    )
    parser.add_argument(
        "--source-healthy",
        type=str,
        default=DEFAULT_SRC_HEALTHY,
        help="Path to source healthy joint reference image",
    )
    parser.add_argument(
        "--source-damage",
        type=str,
        default=DEFAULT_SRC_DAMAGE,
        help="Path to source damaged joint reference image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="Target dataset root directory",
    )
    parser.add_argument(
        "--num-augmented",
        type=int,
        default=25,
        help="Number of conservative augmented samples to generate per class (default: 25)",
    )
    parser.add_argument(
        "--crop-joint",
        action="store_true",
        default=True,
        help="Focus on the conveyor belt joint region, removing outer wall/frame background (default: True)",
    )
    parser.add_argument(
        "--no-crop",
        dest="crop_joint",
        action="store_false",
        help="Disable automatic joint cropping; use raw full camera frame",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentation",
    )
    return parser.parse_args()


def extract_joint_roi(img: np.ndarray) -> np.ndarray:
    """
    Extracts the central belt joint region, eliminating background walls,
    outer rig frames, and uninformative upper/lower belt expanses while
    preserving all metallic plates, rivets, and damage irregularities.
    """
    h, w = img.shape[:2]

    # If the image is already a tight crop (wide aspect ratio or small height), keep as is
    if w / max(1, h) >= 2.0 or h < 350:
        return img

    # 1. Exclude outer 8% margins containing rig frames / background walls
    x_min = int(w * 0.08)
    x_max = int(w * 0.92)
    belt = img[:, x_min:x_max]

    # 2. Identify the joint band using horizontal edge energy and metallic brightness
    gray = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_energy = np.mean(np.abs(sobel_y), axis=1) + 0.5 * np.mean(gray, axis=1)

    # Pick the high-energy band containing the joint
    thresh_val = np.percentile(row_energy, 65)
    joint_rows = np.where(row_energy > thresh_val)[0]

    if len(joint_rows) > 0:
        y_min = max(0, int(joint_rows.min() - 0.05 * h))
        y_max = min(h, int(joint_rows.max() + 0.05 * h))

        # Enforce minimum height (at least 30% of original height)
        if (y_max - y_min) < int(h * 0.30):
            center_y = int((y_min + y_max) / 2)
            half = int(h * 0.20)
            y_min = max(0, center_y - half)
            y_max = min(h, center_y + half)
    else:
        # Fallback: center 70%
        y_min = int(h * 0.15)
        y_max = int(h * 0.85)

    return img[y_min:y_max, x_min:x_max]


def apply_conservative_augmentation(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Applies conservative, physically plausible augmentations:
      - Small rotation ([-6 to +6 degrees])
      - Slight translation ([-4% to +4%])
      - Small scale variation ([0.95 to 1.05])
      - Mild brightness adjustment ([-15 to +15 intensity levels])
      - Mild contrast adjustment ([0.90 to 1.12])
      - Mild Gaussian sensor noise (25% probability)
      - Mild Gaussian blur (25% probability)

    Strictly avoids unrealistic transformations that alter joint geometry.
    """
    h, w = img.shape[:2]

    # 1. Geometric Affine Transform
    angle = rng.uniform(-6.0, 6.0)
    scale = rng.uniform(0.95, 1.05)
    tx = rng.uniform(-0.04 * w, 0.04 * w)
    ty = rng.uniform(-0.04 * h, 0.04 * h)

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    aug = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 2. Lighting & Contrast Variation
    alpha = rng.uniform(0.90, 1.12)
    beta = rng.uniform(-15.0, 15.0)
    aug = np.clip(alpha * aug.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 3. Mild Sensor Noise (25% probability)
    if rng.rand() < 0.25:
        noise = rng.normal(0, rng.uniform(2.0, 5.0), aug.shape).astype(np.float32)
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Mild Gaussian Blur (25% probability, simulates slight camera focus/motion)
    if rng.rand() < 0.25:
        aug = cv2.GaussianBlur(aug, (3, 3), rng.uniform(0.5, 1.0))

    return aug


def main():
    args = parse_args()

    print("=" * 60)
    print("JOINTGUARD YOLO - DATASET PREPARATION")
    print("=" * 60)

    # 1. Validate source files
    if not os.path.isfile(args.source_healthy):
        print(f"[ERROR] Source healthy image not found: {args.source_healthy}")
        sys.exit(1)
    if not os.path.isfile(args.source_damage):
        print(f"[ERROR] Source damaged image not found: {args.source_damage}")
        sys.exit(1)

    print(f"[SOURCE] Healthy image: {args.source_healthy}")
    print(f"[SOURCE] Damaged image: {args.source_damage}")
    print(f"[TARGET] Dataset dir:   {args.output_dir}")
    print(f"[CONFIG] Crop joint:    {args.crop_joint}")
    print(f"[CONFIG] Augmentations: {args.num_augmented} per class")

    # 2. Setup directory hierarchy
    train_healthy_dir = os.path.join(args.output_dir, "train", "healthy")
    train_damage_dir = os.path.join(args.output_dir, "train", "damage")
    val_healthy_dir = os.path.join(args.output_dir, "val", "healthy")
    val_damage_dir = os.path.join(args.output_dir, "val", "damage")

    for d in [train_healthy_dir, train_damage_dir, val_healthy_dir, val_damage_dir]:
        os.makedirs(d, exist_ok=True)

    # 3. Read and process source images
    healthy_raw = cv2.imread(args.source_healthy)
    damage_raw = cv2.imread(args.source_damage)

    if healthy_raw is None:
        print(f"[ERROR] Failed to load image: {args.source_healthy}")
        sys.exit(1)
    if damage_raw is None:
        print(f"[ERROR] Failed to load image: {args.source_damage}")
        sys.exit(1)

    if args.crop_joint:
        healthy_base = extract_joint_roi(healthy_raw)
        damage_base = extract_joint_roi(damage_raw)
        print(f"[PREPROCESS] Healthy cropped: {healthy_raw.shape} -> {healthy_base.shape}")
        print(f"[PREPROCESS] Damaged cropped: {damage_raw.shape} -> {damage_base.shape}")
    else:
        healthy_base = healthy_raw
        damage_base = damage_raw
        print("[PREPROCESS] Using raw images without joint cropping")

    # 4. Save original reference images in train/
    dst_orig_healthy = os.path.join(train_healthy_dir, "healthy_001.jpg")
    dst_orig_damage = os.path.join(train_damage_dir, "damage_001.jpg")
    cv2.imwrite(dst_orig_healthy, healthy_base)
    cv2.imwrite(dst_orig_damage, damage_base)
    print(f"[SAVED] Original reference: {dst_orig_healthy}")
    print(f"[SAVED] Original reference: {dst_orig_damage}")

    # 5. Generate conservative augmented samples in train/
    rng = np.random.RandomState(args.seed)

    print(f"\n[AUGMENTATION] Generating {args.num_augmented} augmented healthy samples...")
    for i in range(1, args.num_augmented + 1):
        aug_img = apply_conservative_augmentation(healthy_base, rng)
        out_path = os.path.join(train_healthy_dir, f"healthy_aug_{i:03d}.jpg")
        cv2.imwrite(out_path, aug_img)

    print(f"[AUGMENTATION] Generating {args.num_augmented} augmented damaged samples...")
    for i in range(1, args.num_augmented + 1):
        aug_img = apply_conservative_augmentation(damage_base, rng)
        out_path = os.path.join(train_damage_dir, f"damage_aug_{i:03d}.jpg")
        cv2.imwrite(out_path, aug_img)

    # 6. Populate val/ with reference samples
    # NOTE: Strictly avoid data leakage. No augmented images are placed in val/.
    val_dst_healthy = os.path.join(val_healthy_dir, "healthy_001.jpg")
    val_dst_damage = os.path.join(val_damage_dir, "damage_001.jpg")
    cv2.imwrite(val_dst_healthy, healthy_base)
    cv2.imwrite(val_dst_damage, damage_base)
    print(f"[SAVED] Validation reference: {val_dst_healthy}")
    print(f"[SAVED] Validation reference: {val_dst_damage}")

    # 7. Print summary & data leakage disclaimer
    train_h_count = len(os.listdir(train_healthy_dir))
    train_d_count = len(os.listdir(train_damage_dir))
    val_h_count = len(os.listdir(val_healthy_dir))
    val_d_count = len(os.listdir(val_damage_dir))

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Train Healthy: {train_h_count} images (1 original + {train_h_count - 1} augmented)")
    print(f"  Train Damage:  {train_d_count} images (1 original + {train_d_count - 1} augmented)")
    print(f"  Val Healthy:   {val_h_count} images (reference sample)")
    print(f"  Val Damage:    {val_d_count} images (reference sample)")
    print("=" * 60)
    print("\n[IMPORTANT NOTICE: DATA LEAKAGE & VALIDATION]")
    print("Augmented images are confined exclusively to train/ to prevent leakage.")
    print("Because only two original reference images currently exist, val/ contains")
    print("unaugmented copies of these reference samples to satisfy YOLO's loop.")
    print("A statistically independent validation dataset is NOT yet available.")
    print("High validation metrics reflect prototype fit, NOT production generalization.")
    print("=" * 60)


if __name__ == "__main__":
    main()
