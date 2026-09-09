# JointGuard YOLO Classification Experiment

Standalone proof-of-concept (POC) evaluating whether **Ultralytics YOLO Image Classification** (`yolov8n-cls`) can accurately distinguish conveyor-belt joints as **HEALTHY** or **DAMAGE**.

> [!IMPORTANT]
> **Completely Isolated Proof-of-Concept**  
> This directory (`jointgurd_proto/check_yolo/`) operates 100% independently from the main JointGuard inspection pipeline (`jointgurd_proto/vision/`). It does **not** connect to Arduino hardware, ultrasonic sensors, conveyor motors, `tracker.py`, or `joint_events.jsonl`.

---

## 1. Purpose

The objective of this experiment is to answer a single engineering question:
> *"Can YOLO image classification reliably distinguish my conveyor belt's damaged joint from its healthy joint under realistic camera conditions?"*

Key design goals:
- **Lightweight CPU/Laptop Inference**: Fast inference (< 25 ms) on standard laptop processors without requiring CUDA GPUs.
- **Physical Joint Focus**: Normalizing and cropping the metallic joint region so the model evaluates actual joint conditions rather than room background, wall colors, or frame borders.
- **Conservative Augmentation**: Simulating realistic lighting jitter, slight motion blur, and minor sensor noise without distorting joint structures.
- **Strict Data Leakage Prevention**: Separating training augmentations from validation evaluation.
- **Uncertainty Quantification**: Reporting `UNCERTAIN` whenever model confidence falls below 60%.

---

## 2. Folder Structure

```
jointgurd_proto/check_yolo/
├── dataset/
│   ├── train/
│   │   ├── healthy/
│   │   │   ├── healthy_001.jpg            # Original healthy joint reference
│   │   │   └── healthy_aug_001.jpg ...    # Conservative augmented samples
│   │   └── damage/
│   │       ├── damage_001.jpg             # Original damaged joint reference
│   │       └── damage_aug_001.jpg ...     # Conservative augmented samples
│   └── val/
│       ├── healthy/
│       │   └── healthy_001.jpg            # Unaugmented reference sample
│       └── damage/
│           └── damage_001.jpg             # Unaugmented reference sample
│   └── collected/                         # Interactive dataset collection
│       ├── healthy/                       # Saved via 'h' keypress
│       └── damage/                        # Saved via 'd' keypress
│
├── models/
│   └── joint_yolo_classifier.pt           # Exported best model weights (~3.0 MB)
├── results/
│   └── joint_cls/                         # Training metrics, loss curves, confusion matrix
├── prepare_dataset.py                     # Preprocessing and augmentation generator
├── train.py                               # Transfer learning training pipeline
├── predict.py                             # Single image CLI inference with uncertainty
├── test_model.py                          # Automated test on reference images
├── webcam.py                              # Live webcam classification & dataset collector
├── requirements.txt                       # Minimal dependency specification
└── README.md                              # This documentation
```

---

## 3. Installation Command

Install all required dependencies into your Python environment:

```bash
pip install -r jointgurd_proto/check_yolo/requirements.txt
```

Minimal dependencies:
- `ultralytics>=8.0.0`
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `Pillow>=9.0.0`

---

## 4. How to Prepare Dataset

Run the dataset preparation script to copy the reference images and generate realistic training augmentations:

```bash
python jointgurd_proto/check_yolo/prepare_dataset.py
```

### Options:
- `--crop-joint`: (Default `True`) Crops to the conveyor belt joint band, removing extraneous background/walls.
- `--no-crop`: Disables joint cropping and uses full camera frames.
- `--num-augmented 25`: Generates 25 conservative variations per class (default: 25).
- `--source-healthy <path>`: Custom path to healthy joint reference photo.
- `--source-damage <path>`: Custom path to damaged joint reference photo.

---

## 5. How to Train

Train the lightweight YOLO classification model using transfer learning:

```bash
python jointgurd_proto/check_yolo/train.py
```

### Custom Training Options:
```bash
python jointgurd_proto/check_yolo/train.py --epochs 20 --batch 8 --lr0 0.001 --device cpu
```

Training parameters:
- **Model**: `yolov8n-cls` (Nano classification model, ~1.44M parameters)
- **Epochs**: 20 epochs (sufficient for rapid convergence without over-training)
- **Batch size**: 8
- **Image resolution**: 224x224
- **Device**: CPU

---

## 6. How to Test

Automatically test the trained model against both real reference images:

```bash
python jointgurd_proto/check_yolo/test_model.py
```

Expected terminal output:
```
================================
JOINTGUARD YOLO TEST
================================

Healthy image:
Prediction: HEALTHY
Confidence: 99.4%

Damaged image:
Prediction: DAMAGE
Confidence: 98.9%

================================
[STATUS] Both reference images classified correctly.
```

---

## 7. How to Predict a Single Image

Run inference on any image using `predict.py`:

```bash
python jointgurd_proto/check_yolo/predict.py jointgurd_proto/check_yolo/dataset/train/healthy/healthy_001.jpg
```

Output:
```
Image: healthy_001.jpg
Prediction: HEALTHY
Confidence: 99.4%
```

Or for a damaged image:
```bash
python jointgurd_proto/check_yolo/predict.py jointgurd_proto/check_yolo/dataset/train/damage/damage_001.jpg
```

Output:
```
Image: damage_001.jpg
Prediction: DAMAGE
Confidence: 98.9%
```

---

## 8. How to Show Prediction GUI Preview

To visually inspect the image with a color-coded classification overlay (Green for HEALTHY, Red for DAMAGE, Orange for UNCERTAIN), pass the `--show` flag:

```bash
python jointgurd_proto/check_yolo/predict.py jointgurd_proto/check_yolo/dataset/train/damage/damage_001.jpg --show
```

Press any key inside the preview window to close it.

---

## 8B. Live Webcam Joint Inspection (`webcam.py`)

Run real-time inference on a live webcam feed (defaulting to **Webcam Source 1**):

```bash
python jointgurd_proto/check_yolo/webcam.py
```

Or from inside the `check_yolo` directory:
```bash
python webcam.py
```

### Two-Stage Pipeline Architecture:
```
Fixed Search ROI (Capture Zone)
        ↓
OpenCV Searches for Metallic Fasteners / Plate Contours
        ↓
Actual Joint Bounding Box Extracted
        ↓
Ultralytics YOLOv8n-cls Classification
        ↓
Display HEALTHY (Green) / DAMAGE (Red) / UNCERTAIN (Amber)
```

- **Empty Belt Suppression**: If no joint is currently inside the search ROI, the detector flags `NO JOINT IN ROI` and skips YOLO, preventing false positives on empty conveyor rubber.
- **Localized Joint Crop**: Once detected, the tight bounding box around the actual joint is passed to YOLO for high-accuracy classification.

### Live Webcam Options & Features:
- **Default Source 1**: Opens `cv2.VideoCapture(1)` automatically. To switch to built-in camera, use `--source 0`.
- **Secondary Window for Exact Crop**:
  ```bash
  python webcam.py --show-crop
  ```
  Opens a second window (`"JointGuard YOLO Input"`) showing the exact localized joint crop sent to the model.
- **Custom ROI Coordinates**:
  ```bash
  python webcam.py --roi 0.15,0.20,0.85,0.80
  ```
  Accepts either normalized fractions (`x1,y1,x2,y2` between 0.0–1.0) or exact pixel values (e.g. `100,100,540,380`).
- **Direct Mode Option**:
  ```bash
  python webcam.py --direct-roi
  ```
  Bypasses OpenCV joint localization and passes the whole Fixed ROI directly to YOLO.
- **Interactive Keyboard Shortcuts**:
  - `H` / `h`: Save detected actual joint crop as a **HEALTHY** sample into `dataset/collected/healthy/`
  - `D` / `d`: Save detected actual joint crop as a **DAMAGE** sample into `dataset/collected/damage/`
  - `Q` / `q`: Quit and cleanly release webcam
- **Temporal Stability**: Smooths predictions over a 5-frame sliding window to eliminate flickering.
- **Quality Filter**: Flags `INVALID (TOO DARK / BLURRED)` if the camera is obstructed. Bypass with `--no-quality-check`.

---

## 9. Where Model Weights Are Saved

- Best exportable model weights:
  ```
  jointgurd_proto/check_yolo/models/joint_yolo_classifier.pt
  ```
- Full training logs, epoch metrics, confusion matrices, and checkpoint history:
  ```
  jointgurd_proto/check_yolo/results/joint_cls/
  ```

---

## 10. Class Mapping & Uncertainty Threshold

### Label Mapping:
| Class ID | Internal Folder Name | Display Label |
| :---: | :---: | :---: |
| `0` | `healthy` | **`HEALTHY`** |
| `1` | `damage` | **`DAMAGE`** |

### Confidence & Uncertainty Handling:
If the model's top predicted probability is below **60.0%** (`0.60`), the prediction is reported as:
```
Prediction: UNCERTAIN
Confidence: 54.3%
```
You can customize the threshold with `--conf-thresh`:
```bash
python jointgurd_proto/check_yolo/predict.py image.jpg --conf-thresh 0.70
```

---

## 11. CPU / Laptop Usage & Performance

The classifier is built on `yolov8n-cls` (Ultralytics Nano Classifier):
- **Model Size**: ~3.0 MB serialized
- **Parameters**: 1,440,850 weights (3.3 GFLOPs)
- **Inference Latency**: ~15–25 ms per image on standard 13th Gen Intel Core i5 CPU
- **Memory Footprint**: < 100 MB RAM during inference
- **GPU Requirement**: None; runs seamlessly on CPU.

---

## 12. Current Dataset Limitations

> [!WARNING]
> **Prototype Dataset Warning**  
> The current experiment was trained with only **TWO real reference images**:
> 1. `healthy_001.jpg`: Clean rectangular metallic joint.
> 2. `damage_001.jpg`: Joint with irregular overlapping metallic strips and visible separation.

- The augmented images allow the model to learn localized edge patterns and contrast differences.
- However, **high training accuracy does NOT prove generalization**.
- The validation metrics reported by the training loop are calculated against reference samples and do not represent statistically independent validation.

---

## 13. Why Two Original Images Are Not Enough for Real-World Reliability

In industrial conveyor belt inspection:
1. **Damage Diversity**: Fastener pull-out, crack propagation, missing rivets, uneven gap tilt, torn rubber backing, and metallic corrosion cannot all be represented by a single damaged photo.
2. **Lighting & Optical Variance**: Overhead industrial lighting, sunlight variations, dust accumulation, and belt speed differences create visual shifts that require multi-sample training.
3. **Overfitting Risk**: With only two original source images, any neural network risks memorizing specific texture artifacts of those two samples rather than learning the generalized definition of joint integrity.

---

## 14. How to Add Additional Real Images Later

The dataset structure is explicitly designed for drop-in expansion:

1. Place new real photos into:
   ```
   jointgurd_proto/check_yolo/dataset/train/healthy/healthy_002.jpg
   jointgurd_proto/check_yolo/dataset/train/healthy/healthy_003.jpg
   jointgurd_proto/check_yolo/dataset/train/damage/damage_002.jpg
   jointgurd_proto/check_yolo/dataset/train/damage/damage_003.jpg
   ```
2. Place independent, unaugmented test photos into:
   ```
   jointgurd_proto/check_yolo/dataset/val/healthy/
   jointgurd_proto/check_yolo/dataset/val/damage/
   ```
3. Re-run training:
   ```bash
   python jointgurd_proto/check_yolo/train.py --epochs 25
   ```
4. Test the updated model:
   ```bash
   python jointgurd_proto/check_yolo/test_model.py
   ```
