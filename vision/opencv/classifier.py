"""
Joint Damage Classification Module for JointGuard

Integrates a trained machine learning classifier with probability confidence
estimation, robust low-confidence handling, and fallback to classical feature extraction.
Maintains the swappable BaseJointClassifier interface.
"""

import os
import sys
import cv2
import joblib
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    TORCH_AVAILABLE = False

from vision.opencv.config import VisionConfig, DEFAULT_CONFIG
from vision.opencv.features import (
    extract_feature_vector,
    extract_human_readable_features,
    preprocess_for_mobilenet
)


@dataclass
class ClassificationResult:
    """Structure holding the classification label, continuous score, and raw feature metrics."""
    label: str                   # "HEALTHY", "DAMAGE", or "UNCERTAIN" / "INVALID"
    vision_score: float          # Continuous score 0.0 - 100.0 (100 = perfectly flat/healthy)
    features: Dict[str, float]   # Raw extracted feature values for telemetry & debugging

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to dict format matching backend event payload."""
        return {
            "label": self.label,
            "vision_score": round(self.vision_score, 2),
            "features": {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in self.features.items()}
        }


class BaseJointClassifier(ABC):
    """Abstract Base Class for joint classification algorithms."""

    @abstractmethod
    def classify(self, roi_frame: np.ndarray, config: Optional[VisionConfig] = None) -> ClassificationResult:
        """Classifies a cropped joint ROI image."""
        pass


def _build_mobilenet_v2(num_classes: int = 2):
    """Builds the MobileNetV2 architecture with binary classification head."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, num_classes)
    )
    return model


class TrainableJointClassifier(BaseJointClassifier):
    """
    Trainable image-based joint classifier using MobileNetV2 transfer learning.
    Loads trained model weights once and executes real-time inference on CPU (< 10ms).
    Falls back gracefully to classical CV if model weights are not present.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._load_attempted = False
        self._fallback_classifier = ClassicalCVClassifier()

    def _resolve_model_path(self, config_path: str) -> str:
        """Resolves relative or absolute path to the serialized model file."""
        if os.path.isabs(config_path) and os.path.exists(config_path):
            return config_path
        # Check relative to module directory
        module_rel = os.path.join(os.path.dirname(__file__), config_path)
        if os.path.exists(module_rel):
            return module_rel
        # Check current working directory
        if os.path.exists(config_path):
            return config_path
        return module_rel

    def load_model(self, path: Optional[str] = None) -> bool:
        """Loads the serialized model into memory once."""
        target_path = path or self.model_path or DEFAULT_CONFIG.MODEL_PATH
        resolved_path = self._resolve_model_path(target_path)

        # If .pth requested but doesn't exist, also check for .pt or vice-versa
        if not os.path.exists(resolved_path):
            base_no_ext = os.path.splitext(resolved_path)[0]
            for alt_ext in [".pth", ".pt", ".joblib"]:
                alt_path = base_no_ext + alt_ext
                if os.path.exists(alt_path):
                    resolved_path = alt_path
                    break

        if not os.path.exists(resolved_path):
            print("\n" + "*" * 74)
            print(f"[CRITICAL ERROR] Trained joint classifier NOT FOUND at: {resolved_path}")
            print("Please train the neural network model by running:")
            print("    python vision/opencv/train_classifier.py")
            print("Falling back to Classical CV feature classifier.")
            print("*" * 74 + "\n")
            self.model = None
            self._load_attempted = True
            return False

        try:
            if resolved_path.endswith(".pth"):
                if TORCH_AVAILABLE:
                    model = _build_mobilenet_v2(num_classes=2)
                    state_dict = torch.load(resolved_path, map_location="cpu", weights_only=True)
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.model = model
                else:
                    alt_joblib = os.path.splitext(resolved_path)[0] + ".joblib"
                    if os.path.exists(alt_joblib):
                        self.model = joblib.load(alt_joblib)
                        resolved_path = alt_joblib
                    else:
                        print(f"[INFO] PyTorch runtime unavailable; using Classical CV fallback for {resolved_path}")
                        self.model = None
                        self._load_attempted = True
                        return False
            elif resolved_path.endswith(".pt"):
                if TORCH_AVAILABLE:
                    model = torch.load(resolved_path, map_location="cpu", weights_only=False)
                    model.eval()
                    self.model = model
                else:
                    alt_joblib = os.path.splitext(resolved_path)[0] + ".joblib"
                    if os.path.exists(alt_joblib):
                        self.model = joblib.load(alt_joblib)
                        resolved_path = alt_joblib
                    else:
                        print(f"[INFO] PyTorch runtime unavailable; using Classical CV fallback for {resolved_path}")
                        self.model = None
                        self._load_attempted = True
                        return False
            elif resolved_path.endswith((".joblib", ".pkl")):
                self.model = joblib.load(resolved_path)
            else:
                # Attempt PyTorch load first, then joblib
                if TORCH_AVAILABLE:
                    try:
                        model = _build_mobilenet_v2(num_classes=2)
                        state_dict = torch.load(resolved_path, map_location="cpu", weights_only=True)
                        model.load_state_dict(state_dict)
                        model.eval()
                        self.model = model
                    except Exception:
                        self.model = joblib.load(resolved_path)
                else:
                    self.model = joblib.load(resolved_path)

            self.model_path = resolved_path
            self._load_attempted = True
            print(f"[MODEL SUCCESS] Loaded trained classifier from: {resolved_path}")
            print(f"[MODEL DETAILS] Architecture    : PyTorch MobileNetV2 (ImageNet Pretrained)")
            print(f"[MODEL DETAILS] Class Mapping   : 0 = HEALTHY, 1 = DAMAGE (DAMAGED)")
            print(f"[MODEL DETAILS] Preprocessing   : 224x224 RGB, ImageNet mean/std normalized")
            return True
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to load model from {resolved_path}: {e}")
            traceback.print_exc()
            self.model = None
            self._load_attempted = True
            return False

    def classify(self, roi_frame: np.ndarray, config: Optional[VisionConfig] = None) -> ClassificationResult:
        cfg = config if config is not None else DEFAULT_CONFIG

        if roi_frame is None or roi_frame.size == 0:
            return ClassificationResult(
                label="INVALID",
                vision_score=0.0,
                features={"edge_density": 0.0, "intensity_variance": 0.0, "confidence": 0.0, "display_label": "INVALID"}
            )

        if roi_frame.shape[0] < 5 or roi_frame.shape[1] < 5:
            return ClassificationResult(
                label="UNCERTAIN",
                vision_score=50.0,
                features={"edge_density": 0.0, "intensity_variance": 0.0, "confidence": 0.50, "display_label": "UNCERTAIN"}
            )

        # Lazy load model if not yet attempted
        if not self._load_attempted:
            self.load_model(cfg.MODEL_PATH)

        if self.model is None:
            # Fallback to classical CV
            return self._fallback_classifier.classify(roi_frame, cfg)

        try:
            if TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module):
                # PyTorch MobileNetV2 Inference
                norm_tensor = preprocess_for_mobilenet(roi_frame, cfg.CLASSIFIER_INPUT_SIZE)
                with torch.no_grad():
                    outputs = self.model(norm_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    p_healthy = float(probs[0].item())
                    p_damage = float(probs[1].item())
            else:
                # Scikit-Learn / Joblib feature vector fallback
                feat_vec = extract_feature_vector(roi_frame, cfg.CLASSIFIER_INPUT_SIZE)
                probs = self.model.predict_proba([feat_vec])[0]
                p_healthy = float(probs[0])
                p_damage = float(probs[1])
        except Exception as e:
            import traceback
            print(f"[ERROR] Classifier Inference Exception: {e}")
            traceback.print_exc()
            return self._fallback_classifier.classify(roi_frame, cfg)

        conf_threshold = getattr(cfg, "CONFIDENCE_THRESHOLD", 0.60)

        # Determine label and confidence mapping to vision_score (0 - 100)
        # Mapping Formula:
        #   - HEALTHY: vision_score = round(confidence * 100.0, 2)  [e.g. 0.91 -> 91.0]
        #   - DAMAGE : vision_score = round((1.0 - confidence) * 100.0, 2) [e.g. 0.87 -> 13.0]
        #   - UNCERTAIN (< 0.60 conf): vision_score = 50.0
        if p_healthy >= conf_threshold:
            label = "HEALTHY"
            confidence = p_healthy
            vision_score = round(confidence * 100.0, 2)
        elif p_damage >= conf_threshold:
            label = "DAMAGE"
            confidence = p_damage
            vision_score = round((1.0 - confidence) * 100.0, 2)
        else:
            label = "UNCERTAIN"
            confidence = max(p_healthy, p_damage)
            vision_score = 50.0

        telemetry_feats = extract_human_readable_features(roi_frame)
        telemetry_feats["confidence"] = round(confidence, 4)
        telemetry_feats["p_healthy"] = round(p_healthy, 4)
        telemetry_feats["p_damage"] = round(p_damage, 4)
        telemetry_feats["display_label"] = "DAMAGED" if label == "DAMAGE" else label

        return ClassificationResult(
            label=label,
            vision_score=float(vision_score),
            features=telemetry_feats
        )


class ClassicalCVClassifier(BaseJointClassifier):
    """Fallback Classical CV feature-based classifier using Canny edges, Hough lines, and intensity variance."""

    def classify(self, roi_frame: np.ndarray, config: Optional[VisionConfig] = None) -> ClassificationResult:
        cfg = config if config is not None else DEFAULT_CONFIG

        if roi_frame is None or roi_frame.size == 0 or roi_frame.shape[0] < 5 or roi_frame.shape[1] < 5:
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

        # 1. Feature 1: Patch Continuity & Sub-contour Fragmentation
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_chan = hsv[:, :, 2]
        v_median = float(np.median(v_chan)) if v_chan.size > 0 else 120.0
        v_lower = max(int(cfg.HSV_LOWER[2]), min(210, int(v_median + 20.0)))
        mask = cv2.inRange(hsv, np.array([0, 0, v_lower], dtype=np.uint8), np.array([180, int(cfg.HSV_UPPER[1]), int(cfg.HSV_UPPER[2])], dtype=np.uint8))

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
        cont_penalty = max(0.0, 1.0 - raw_patch_continuity)
        frag_penalty = min(1.0, max(0.0, float(raw_frag_count - 1) * 0.20))
        norm_struct_penalty = min(1.0, 0.6 * cont_penalty + 0.4 * frag_penalty)

        # 2. Feature 2: Edge Density
        blurred_gray = cv2.GaussianBlur(foil_crop_gray, (5, 5), 0)
        edges = cv2.Canny(blurred_gray, int(cfg.CANNY_THRESHOLD1), int(cfg.CANNY_THRESHOLD2))
        edge_pixel_count = float(np.count_nonzero(edges))
        raw_edge_density = edge_pixel_count / float(foil_crop_gray.size) if foil_crop_gray.size > 0 else 0.0
        norm_edge_penalty = min(1.0, raw_edge_density / float(cfg.EDGE_DENSITY_MAX_EXPECTED))

        # 3. Feature 3: Hough Line Crease Detection
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

        # 4. Feature 4: Local Intensity Variance
        blurred_var = cv2.GaussianBlur(foil_crop_gray, cfg.VARIANCE_BLUR_KERNEL, 0)
        raw_intensity_var = float(np.std(blurred_var)) if blurred_var.size > 0 else 0.0
        norm_var_penalty = min(1.0, max(0.0, (raw_intensity_var - 30.0) / float(cfg.VARIANCE_MAX_EXPECTED - 30.0)))

        # 5. Combined Score
        total_damage_penalty = (
            0.50 * norm_struct_penalty +
            0.30 * norm_var_penalty +
            0.10 * norm_edge_penalty +
            0.10 * norm_hough_penalty
        )

        vision_score = max(0.0, min(100.0, 100.0 * (1.0 - total_damage_penalty)))
        label = "HEALTHY" if vision_score >= cfg.HEALTHY_THRESHOLD_SCORE else "DAMAGE"

        features = {
            "patch_continuity": float(raw_patch_continuity),
            "fragment_count": float(raw_frag_count),
            "edge_density": float(raw_edge_density),
            "hough_line_score": float(raw_hough_score),
            "intensity_variance": float(raw_intensity_var),
            "confidence": 0.75
        }

        return ClassificationResult(
            label=label,
            vision_score=float(vision_score),
            features=features
        )


# Global trained classifier instance (shared across frames to avoid reloading)
_TRAINED_CLASSIFIER = TrainableJointClassifier()


def load_trained_model(model_path: Optional[str] = None) -> bool:
    """Preloads the trained model on system startup."""
    return _TRAINED_CLASSIFIER.load_model(model_path)


def classify_joint(roi_frame: np.ndarray, config: Optional[VisionConfig] = None) -> ClassificationResult:
    """
    Main classification entrypoint matching the existing system signature.

    Args:
        roi_frame: BGR cropped joint region.
        config: Optional VisionConfig parameter.

    Returns:
        ClassificationResult containing label, vision_score (0-100), and raw features dict.
    """
    return _TRAINED_CLASSIFIER.classify(roi_frame, config)
