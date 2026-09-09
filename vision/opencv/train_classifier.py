"""
JointGuard Conveyor Joint Classifier Training Script (PyTorch MobileNetV2)

Trains a lightweight MobileNetV2 transfer-learning classifier on healthy vs damaged
conveyor joint reference images with physically realistic data augmentation.

Usage:
    python vision/opencv/train_classifier.py
"""

import argparse
import datetime
import json
import os
import sys
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from vision.opencv.config import VisionConfig, DEFAULT_CONFIG
from vision.opencv.detector import detect_joint
from vision.opencv.features import preprocess_joint_crop


DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "joint_classifier.pth")
DEFAULT_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

CLASS_MAPPING = {
    0: "HEALTHY",
    1: "DAMAGE"
}
LABEL_TO_ID = {v: k for k, v in CLASS_MAPPING.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="JointGuard PyTorch MobileNetV2 Classifier Training")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DATASET_DIR,
        help="Path to dataset directory containing healthy/ and damage/ folders"
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Destination path for serialized .pth model"
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default=DEFAULT_METADATA_PATH,
        help="Destination path for model metadata JSON"
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=100,
        help="Number of augmented variations to generate per source image"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of fine-tuning training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for classification head"
    )
    return parser.parse_args()


def load_raw_images(folder_path: str) -> List[Tuple[str, np.ndarray]]:
    """Loads all valid image files from a given directory."""
    images = []
    if not os.path.exists(folder_path):
        return images

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            full_path = os.path.join(folder_path, fname)
            img = cv2.imread(full_path)
            if img is not None:
                images.append((fname, img))
    return images


def extract_joint_crop_from_sample(image: np.ndarray, config: VisionConfig, fname: str = "") -> np.ndarray:
    """
    Locates and crops the joint region from an input sample.
    - If the image is already a pre-cropped joint (e.g. captured live via 'h'/'d', or has crop aspect ratio),
      returns image directly.
    - If it's a full conveyor belt frame (e.g. reference photos like healthy_001.jpg, damage_001.jpg),
      runs detect_joint to extract the joint bbox.
    """
    fname_lower = fname.lower()
    # 1. Live dataset capture crops (from main.py press 'h'/'d')
    if "crop" in fname_lower or fname_lower.startswith("healthy_20") or fname_lower.startswith("damage_20"):
        return image

    h, w = image.shape[:2]
    aspect = float(w) / max(1.0, float(h))

    # 2. For full conveyor belt scene photos, run detect_joint
    bbox = detect_joint(image, config)
    if bbox is not None:
        crop = bbox.crop_roi(image)
        if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
            return crop

    # 3. If already a tight joint crop (wide aspect ratio >= 1.4 or small height <= 250), use as-is
    if aspect >= 1.4 or h <= 250 or w <= 450:
        return image

    # 4. Fallback for uncropped full frames where detector found no candidate: center 70%
    cy1, cy2 = int(h * 0.15), int(h * 0.85)
    cx1, cx2 = int(w * 0.15), int(w * 0.85)
    return image[cy1:cy2, cx1:cx2]


def apply_realistic_augmentation(crop: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Applies physically realistic data augmentations to simulate real-world webcam conditions:
      - Small rotation: [-8 to +8 degrees]
      - Small translation: [-8% to +8%]
      - Small scaling: [0.92 to 1.08]
      - Brightness & Contrast adjustment
      - Mild Gaussian blur (simulating slight focus / conveyor motion)
      - Mild additive Gaussian sensor noise
    """
    h, w = crop.shape[:2]

    # 1. Geometric Affine Transform (Rotation, Translation, Scale)
    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-0.08 * w, 0.08 * w)
    ty = rng.uniform(-0.08 * h, 0.08 * h)

    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rot_mat[0, 2] += tx
    rot_mat[1, 2] += ty

    aug = cv2.warpAffine(crop, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 2. Lighting & Contrast Variations
    alpha = rng.uniform(0.85, 1.18)
    beta = rng.uniform(-18.0, 18.0)
    aug = np.clip(alpha * aug.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 3. Mild Sensor Noise (25% probability)
    if rng.rand() < 0.25:
        noise = rng.normal(0, rng.uniform(2.0, 6.0), aug.shape).astype(np.float32)
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Mild Conveyor Motion Blur (30% probability)
    if rng.rand() < 0.30:
        ksize = rng.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), rng.uniform(0.5, 1.2))

    return aug


class JointDataset(Dataset):
    """PyTorch Dataset returning ImageNet-normalized tensors and class labels."""

    def __init__(self, samples: List[Tuple[np.ndarray, int]], target_size=(224, 224)):
        self.samples = samples
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop, label = self.samples[idx]
        norm_crop = preprocess_joint_crop(crop, self.target_size)
        rgb = cv2.cvtColor(norm_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_rgb = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(norm_rgb).permute(2, 0, 1).float()
        return tensor, torch.tensor(label, dtype=torch.long)


def build_mobilenet_v2(num_classes=2) -> nn.Module:
    """Builds a MobileNetV2 network with fine-tuned head for binary joint classification."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze base feature extractor to prevent overfitting small datasets
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze last inverted residual block for domain adaptation
    for param in model.features[-1].parameters():
        param.requires_grad = True

    # Custom classification head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, num_classes)
    )
    return model


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    rng = np.random.RandomState(42)
    torch.manual_seed(42)

    print("=" * 72)
    print("      JOINTGUARD PYTORCH MOBILENET-V2 CLASSIFIER BUILDER")
    print("=" * 72)
    print(f"[INFO] Dataset Directory  : {args.dataset_dir}")
    print(f"[INFO] Model Output Path  : {args.output_model}")
    print(f"[INFO] Metadata Output    : {args.output_metadata}")

    healthy_dir = os.path.join(args.dataset_dir, "healthy")
    damage_dir = os.path.join(args.dataset_dir, "damage")

    healthy_images = load_raw_images(healthy_dir)
    damage_images = load_raw_images(damage_dir)

    print(f"[INFO] Loaded {len(healthy_images)} Healthy base images from: {healthy_dir}")
    print(f"[INFO] Loaded {len(damage_images)} Damaged base images from: {damage_dir}")

    if len(healthy_images) == 0 or len(damage_images) == 0:
        print("[ERROR] Dataset must contain at least 1 image in healthy/ and 1 in damage/.")
        sys.exit(1)

    N_total = len(healthy_images) + len(damage_images)
    H_count = len(healthy_images)
    D_count = len(damage_images)

    dynamic_warning = (
        f"WARNING: Training dataset currently contains {N_total} original images "
        f"({H_count} healthy, {D_count} damage). Small datasets may not generalize well "
        f"to new camera angles, lighting, or distances not represented in the training images."
    )
    print("\n" + "!" * 72)
    print(dynamic_warning)
    print("!" * 72 + "\n")

    # Extract base crops
    print("\n[STEP 1] Locating and cropping joint regions from base samples...")
    base_samples: List[Tuple[str, int, np.ndarray, np.ndarray]] = []

    for fname, img in healthy_images:
        crop = extract_joint_crop_from_sample(img, cfg, fname)
        base_samples.append((fname, LABEL_TO_ID["HEALTHY"], crop, img))
        print(f"  [HEALTHY] {fname}: crop shape={crop.shape}")

    for fname, img in damage_images:
        crop = extract_joint_crop_from_sample(img, cfg, fname)
        base_samples.append((fname, LABEL_TO_ID["DAMAGE"], crop, img))
        print(f"  [DAMAGE]  {fname}: crop shape={crop.shape}")

    # Generate Augmented Training Set
    # Balance number of augmentations so both classes have equal representation
    max_count = max(len(healthy_images), len(damage_images))
    healthy_samples_per_img = max(15, int(args.samples_per_image * (max_count / len(healthy_images))))
    damage_samples_per_img = max(15, int(args.samples_per_image * (max_count / len(damage_images))))

    print(f"\n[STEP 2] Generating realistic variations per base image...")
    print(f"  Augmentations per image -> Healthy: {healthy_samples_per_img}, Damage: {damage_samples_per_img}")
    train_samples = []

    for fname, label_id, base_crop, _ in base_samples:
        train_samples.append((base_crop, label_id))
        n_aug = healthy_samples_per_img if label_id == 0 else damage_samples_per_img
        for _ in range(n_aug):
            aug_crop = apply_realistic_augmentation(base_crop, rng)
            train_samples.append((aug_crop, label_id))

    healthy_count = sum(1 for _, l in train_samples if l == 0)
    damage_count = sum(1 for _, l in train_samples if l == 1)
    print(f"  Total training instances: {len(train_samples)} (Healthy: {healthy_count}, Damage: {damage_count})")

    train_dataset = JointDataset(train_samples, target_size=cfg.CLASSIFIER_INPUT_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Build model
    print("\n[STEP 3] Initializing MobileNetV2 with ImageNet pretrained weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mobilenet_v2(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Train model
    print(f"\n[STEP 4] Training classification head for {args.epochs} epochs on {device}...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc * 100.0:.1f}%")

    # Evaluate on full training set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    train_acc = accuracy_score(all_targets, all_preds)

    # Holdout validation on original base reference images
    base_eval_samples = [(crop, label_id) for _, label_id, crop, _ in base_samples]
    val_dataset = JointDataset(base_eval_samples, target_size=cfg.CLASSIFIER_INPUT_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, val_preds = torch.max(outputs, 1)
            val_preds = val_preds.cpu().numpy()
            val_targets = labels.numpy()

    base_acc = accuracy_score(val_targets, val_preds)
    prec = precision_score(val_targets, val_preds, zero_division=0)
    rec = recall_score(val_targets, val_preds, zero_division=0)
    f1 = f1_score(val_targets, val_preds, zero_division=0)
    cm = confusion_matrix(val_targets, val_preds, labels=[0, 1])

    # Check for separate validation images in dataset/validation/
    val_healthy_dir = os.path.join(args.dataset_dir, "validation", "healthy")
    val_damage_dir = os.path.join(args.dataset_dir, "validation", "damage")
    val_healthy_imgs = load_raw_images(val_healthy_dir)
    val_damage_imgs = load_raw_images(val_damage_dir)
    separate_val_count = len(val_healthy_imgs) + len(val_damage_imgs)

    print("\n" + "=" * 72)
    print("                    TRAINING RESULTS")
    print("=" * 72)
    print(f"  Training Accuracy (on augmented set) : {train_acc * 100.0:.1f}%")
    print(f"  Base Samples Holdout Accuracy        : {base_acc * 100.0:.1f}%")
    print(f"  Precision (Damage)                   : {prec:.4f}")
    print(f"  Recall (Damage)                      : {rec:.4f}")
    print(f"  F1 Score (Damage)                    : {f1:.4f}")
    print("\n  Confusion Matrix (Rows=True [Healthy, Damage], Cols=Pred [Healthy, Damage]):")
    print(f"    [[{cm[0, 0]:2d} (TN), {cm[0, 1]:2d} (FP)]")
    print(f"     [{cm[1, 0]:2d} (FN), {cm[1, 1]:2d} (TP)]]")

    if separate_val_count > 0:
        print(f"\n[INFO] Evaluated on {separate_val_count} separate validation images.")
    else:
        print("\n" + "!" * 72)
        print(f"  {dynamic_warning}")
        print("!" * 72)

    # Save model
    os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
    model.cpu()
    torch.save(model.state_dict(), args.output_model)
    print(f"\n[SAVED] Trained MobileNetV2 state dict saved to: {args.output_model}")

    # Also save full model .pt
    pt_path = os.path.splitext(args.output_model)[0] + ".pt"
    torch.save(model, pt_path)
    print(f"[SAVED] Complete PyTorch model saved to: {pt_path}")

    # Save metadata
    metadata: Dict[str, Any] = {
        "model_architecture": "MobileNetV2 (Pretrained ImageNet + Fine-tuned Binary Head)",
        "framework": f"PyTorch {torch.__version__}",
        "input_size": list(cfg.CLASSIFIER_INPUT_SIZE),
        "class_mapping": CLASS_MAPPING,
        "classes": ["HEALTHY", "DAMAGE"],
        "num_base_images": {
            "healthy": len(healthy_images),
            "damage": len(damage_images)
        },
        "total_training_instances": int(len(train_samples)),
        "metrics": {
            "train_accuracy": round(float(train_acc), 4),
            "base_sample_accuracy": round(float(base_acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist()
        },
        "trained_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "warning": dynamic_warning
    }

    with open(args.output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Metadata saved to: {args.output_metadata}")
    print("=" * 72)


if __name__ == "__main__":
    main()
