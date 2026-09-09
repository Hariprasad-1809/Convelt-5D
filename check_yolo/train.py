#!/usr/bin/env python3
"""
JointGuard YOLO Classification - Training Script

Trains a lightweight Ultralytics YOLO classification model (yolov8n-cls)
to classify conveyor-belt joint images as HEALTHY (0) vs DAMAGE (1).

Usage:
    python check_yolo/train.py
    python check_yolo/train.py --epochs 20 --batch 8 --lr0 0.001
"""

import argparse
import os
import shutil
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_OUTPUT_MODEL = os.path.join(MODELS_DIR, "joint_yolo_classifier.pt")
PRETRAINED_MODEL = "yolov8n-cls.pt"


def check_dependencies():
    """Validates that all required packages are importable."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python (cv2)")
    try:
        import ultralytics
    except ImportError:
        missing.append("ultralytics")

    if missing:
        print(f"[ERROR] Missing required dependencies: {', '.join(missing)}", file=sys.stderr)
        print("Please run: pip install -r check_yolo/requirements.txt", file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train JointGuard YOLO classification model for conveyor belt joints"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DATASET_DIR,
        help=f"Path to dataset directory (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Input image resolution (default: 224)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on ('cpu' or '0' for CUDA GPU, default: 'cpu')",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=DEFAULT_OUTPUT_MODEL,
        help=f"Target path for best model weights (default: {DEFAULT_OUTPUT_MODEL})",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="joint_cls",
        help="Name of the training run subfolder in results (default: 'joint_cls')",
    )
    return parser.parse_args()


def validate_dataset(dataset_dir: str):
    """Verifies that the dataset structure exists and contains training images."""
    train_healthy = os.path.join(dataset_dir, "train", "healthy")
    train_damage = os.path.join(dataset_dir, "train", "damage")
    val_healthy = os.path.join(dataset_dir, "val", "healthy")
    val_damage = os.path.join(dataset_dir, "val", "damage")

    required_dirs = [train_healthy, train_damage, val_healthy, val_damage]
    missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]

    if missing_dirs:
        print(f"[WARNING] Dataset structure incomplete. Preparing dataset automatically...")
        import prepare_dataset
        prepare_dataset.main()

    h_train_files = [f for f in os.listdir(train_healthy) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    d_train_files = [f for f in os.listdir(train_damage) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not h_train_files or not d_train_files:
        print("[ERROR] Training folders are empty! Please run prepare_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[DATASET] Healthy training samples: {len(h_train_files)}")
    print(f"[DATASET] Damaged training samples: {len(d_train_files)}")


def main():
    check_dependencies()
    args = parse_args()

    print("=" * 65)
    print("JOINTGUARD YOLO CLASSIFIER - TRAINING PIPELINE")
    print("=" * 65)
    print(f"[CONFIG] Model architecture:  YOLOv8n-cls (Nano classification)")
    print(f"[CONFIG] Dataset location:    {args.dataset_dir}")
    print(f"[CONFIG] Target epochs:       {args.epochs}")
    print(f"[CONFIG] Batch size:          {args.batch}")
    print(f"[CONFIG] Image resolution:    {args.imgsz}x{args.imgsz}")
    print(f"[CONFIG] Compute device:      {args.device}")
    print(f"[CONFIG] Target output model: {args.output_model}")
    print("=" * 65)

    # 1. Verify dataset structure
    validate_dataset(args.dataset_dir)

    # 2. Ensure destination directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 3. Import Ultralytics
    from ultralytics import YOLO

    # Check for local pretrained weights or download automatically
    local_pt = os.path.abspath(PRETRAINED_MODEL)
    root_pt = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", PRETRAINED_MODEL))
    if os.path.isfile(local_pt):
        weights_source = local_pt
    elif os.path.isfile(root_pt):
        weights_source = root_pt
    else:
        weights_source = PRETRAINED_MODEL

    print(f"\n[MODEL] Loading pretrained backbone: {weights_source}")
    model = YOLO(weights_source)

    # 4. Start fine-tuning / transfer learning
    start_time = time.time()
    print(f"[TRAIN] Launching transfer learning for {args.epochs} epochs on {args.device.upper()}...")

    results = model.train(
        data=os.path.abspath(args.dataset_dir),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        device=args.device,
        project=os.path.abspath(RESULTS_DIR),
        name=args.experiment_name,
        exist_ok=True,
        verbose=True,
    )

    elapsed_time = time.time() - start_time
    print(f"\n[TRAIN] Training finished in {elapsed_time:.1f} seconds ({elapsed_time / 60.0:.2f} min).")

    # 5. Locate and save best model weights
    exp_dir = os.path.join(RESULTS_DIR, args.experiment_name)
    best_weights_path = os.path.join(exp_dir, "weights", "best.pt")

    if not os.path.isfile(best_weights_path):
        # Fallback to last.pt if best.pt is not created
        best_weights_path = os.path.join(exp_dir, "weights", "last.pt")

    if os.path.isfile(best_weights_path):
        out_dest = os.path.abspath(args.output_model)
        os.makedirs(os.path.dirname(out_dest), exist_ok=True)
        shutil.copy2(best_weights_path, out_dest)
        print(f"[SAVED] Best model saved to: {out_dest} ({os.path.getsize(out_dest) / 1e6:.2f} MB)")
    else:
        print(f"[WARNING] Weights file not found at expected path: {best_weights_path}")

    # 6. Honest reporting and limitations (Step 5, Step 17)
    print("\n" + "=" * 65)
    print("TRAINING EVALUATION & HONEST DATASET LIMITATIONS")
    print("=" * 65)
    print("[STATUS] Prototype classification pipeline is working.")
    print("")
    print("[CRITICAL NOTICE: NOT PRODUCTION READY]")
    print("- Current dataset is based on only TWO real reference images (1 healthy, 1 damage).")
    print("- Augmented samples allowed the model to train and learn joint features,")
    print("  but augmentation does NOT demonstrate real-world generalization.")
    print("- High training accuracy reflects fitting to the prototype samples.")
    print("- A statistically meaningful independent validation dataset is required")
    print("  before this classifier can be considered reliable in production.")
    print("=" * 65)
    print(f"Results, loss curves, and confusion matrix saved to:")
    print(f"  {exp_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
