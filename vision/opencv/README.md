# JointGuard OpenCV Vision & Machine Learning Inspection Module

Automated conveyor-belt joint inspection pipeline using computer vision and **PyTorch MobileNetV2** transfer learning. Detects metallic conveyor belt joints, normalizes the localized joint region, and classifies the joint as either **HEALTHY** or **DAMAGE** with real-time confidence scoring on laptop CPU.

---

## A. How to Install Dependencies

The pipeline runs on standard Python (3.10+) on CPU without requiring a GPU.

### 1. PyTorch & Torchvision (CPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. OpenCV & Supporting Libraries
```bash
pip install opencv-python numpy scikit-learn scipy
```

> [!NOTE]
> On Windows systems where path length limits might affect pip, PyTorch packages can be installed into an abbreviated folder (e.g. `C:\Users\<user>\.torch_pkgs`) and linked via a `.pth` file in `site-packages`.

---

## B. How to Create and Add Dataset Images

The dataset is organized by class in the `dataset/` directory:

```
vision/opencv/dataset/
├── healthy/
│   ├── healthy_001.jpg
│   └── (drop additional healthy joint photos here...)
├── damage/
│   ├── damage_001.jpg
│   └── (drop additional damaged joint photos here...)
└── validation/
    ├── healthy/
    └── damage/
```

### Adding New Reference Images (Zero Code Changes Needed):
1. Capture photos of conveyor belt joints under representative plant/webcam lighting.
2. Ensure the photo captures the metallic joint in the belt corridor.
3. Save healthy conveyor belt joints into `dataset/healthy/` (e.g. `healthy_002.jpg`, `healthy_003.jpg`).
4. Save split, torn, creased, or separated joints into `dataset/damage/` (e.g. `damage_002.jpg`, `damage_003.jpg`).
5. (Optional) Place true independent validation images into `dataset/validation/healthy/` and `dataset/validation/damage/`.
6. Rerun `python vision/opencv/train_classifier.py` — it will automatically discover all images.

---

## C. How to Train the Classifier

Run the automated MobileNetV2 transfer-learning script:

```bash
python vision/opencv/train_classifier.py
```

### Optional Arguments:
- `--samples-per-image 100`: Number of realistic variations generated per base image (default: 100).
- `--epochs 15`: Number of fine-tuning epochs for the classification head (default: 15).
- `--batch-size 16`: Training batch size (default: 16).
- `--lr 0.001`: Learning rate for Adam optimizer (default: 1e-3).
- `--output-model models/joint_classifier.pth`: Target state dict model path.
- `--output-metadata models/model_metadata.json`: Target metadata path.

During training, physically realistic augmentations are applied:
- Small rotations (`[-8°, +8°]`)
- Translations (`[-8%, +8%]`)
- Small scaling (`[0.92, 1.08]`)
- Contrast and brightness jitter
- Mild Gaussian sensor noise and conveyor motion blur
- **No horizontal flipping** (prevents invalidating physical joint orientation)
- **No synthetic fabrication** of physical splits or tears

---

## D. Where the Trained Model is Saved

- **PyTorch State Dict**: `vision/opencv/models/joint_classifier.pth` (loaded by default at runtime)
- **PyTorch Full Model**: `vision/opencv/models/joint_classifier.pt`
- **Model Metadata & Metrics**: `vision/opencv/models/model_metadata.json`
  - Records architecture (`MobileNetV2`), input dimensions `(224, 224)`, training metrics, and class mapping:
    - `0 = HEALTHY`
    - `1 = DAMAGE`

---

## E. How to Test a Healthy Image

Test static image classification directly:

```bash
# Terminal headless event output:
python vision/opencv/main.py --source vision/opencv/test_images/healthy_sample.jpg

# Interactive live GUI window with bounding box and score overlay:
python vision/opencv/main.py --source vision/opencv/test_images/healthy_sample.jpg --debug

# Interactive GUI with joint crop debug window (proves model sees only joint, not background):
python vision/opencv/main.py --source vision/opencv/test_images/healthy_sample.jpg --debug --show-crop
```

---

## F. How to Test a Damaged Image

```bash
# Terminal headless event output:
python vision/opencv/main.py --source vision/opencv/test_images/damage_sample.jpg

# Interactive live GUI window with bounding box and score overlay:
python vision/opencv/main.py --source vision/opencv/test_images/damage_sample.jpg --debug

# Interactive GUI with joint crop debug window:
python vision/opencv/main.py --source vision/opencv/test_images/damage_sample.jpg --debug --show-crop
```

You can also run the direct verification test script:

```bash
python vision/opencv/test_trained_classifier.py
```

---

## G. How to Run Webcam Inspection

To run real-time conveyor inspection using your webcam:

```bash
# Default primary camera (Index 0):
python vision/opencv/main.py --source 0 --debug

# External USB camera (Index 1):
python vision/opencv/main.py --source 1 --debug

# With normalized joint crop debug window:
python vision/opencv/main.py --source 0 --debug --show-crop

# Save finalized events to JSON Lines log:
python vision/opencv/main.py --source 0 --debug --save-log joint_events.jsonl
```

### Live Window Controls & Visual Feedback:
- **`q`**: Exit inspection cleanly.
- **Outer ROI Box (Cyan)**: Active belt inspection zone (`0.18` to `0.85` width).
- **Inner Capture Zone (Magenta)**: Feature aggregation and classification locking trigger window.
- **Bounding Box Colors**:
  - **GREEN**: `HEALTHY`
  - **RED**: `DAMAGE`
  - **YELLOW**: `APPROACHING` / inspecting in progress
  - **GRAY**: `UNCERTAIN` / `INVALID` / `PASSED`
- **Live Status Format**:
  `ID:1 [INSPECTING] | HEALTHY (99.8%)` or `ID:1 [INSPECTING] | DAMAGE (99.4%)`

---

## H. Confidence Score Mapping Formula

The classifier predicts softmax probabilities $P(\text{HEALTHY})$ and $P(\text{DAMAGE})$:

$$\text{Confidence Threshold} = 0.60$$

1. **HEALTHY Decision** (when $P(\text{HEALTHY}) \ge 0.60$):
   $$\text{label} = \text{"HEALTHY"}$$
   $$\text{vision\_score} = \text{round}(P(\text{HEALTHY}) \times 100.0, 2)$$
   *(e.g., $P(\text{HEALTHY}) = 0.91 \implies \text{vision\_score} = 91.00$)*

2. **DAMAGE Decision** (when $P(\text{DAMAGE}) \ge 0.60$):
   $$\text{label} = \text{"DAMAGE"}$$
   $$\text{vision\_score} = \text{round}((1.0 - P(\text{DAMAGE})) \times 100.0, 2)$$
   *(e.g., $P(\text{DAMAGE}) = 0.87 \implies \text{vision\_score} = 13.00$)*

3. **UNCERTAIN Decision** (when $\max(P(\text{HEALTHY}), P(\text{DAMAGE})) < 0.60$):
   $$\text{label} = \text{"UNCERTAIN"}$$
   $$\text{vision\_score} = 50.00$$

---

## I. Explanation of the Full Inspection Pipeline

```
WEBCAM / INPUT STREAM
       ↓
1. Joint Localization & ROI (`detector.py`)
   - Restricts search to conveyor belt corridor (0.18 - 0.85 frame width)
   - Filters out reflective side-rail guides (x < 0.17 * w) and exterior glare
   - Connected component clustering fuses fractured metallic plates into one joint box
       ↓
2. Joint Crop & Preprocessing (`features.py`)
   - Tightly crops the detected metallic joint region
   - Standardizes to 224×224 RGB image
   - Applies ImageNet mean/std normalization for MobileNetV2
       ↓
3. Trained MobileNetV2 Classifier (`classifier.py`)
   - Pretrained ImageNet feature extractor + fine-tuned classification head
   - Runs CPU forward pass (< 10 ms)
   - Evaluates P(HEALTHY) vs P(DAMAGE) with 60% confidence cutoff
       ↓
4. Temporal Aggregation & State Machine (`tracker.py`)
   - Tracks joint: APPROACHING → INSPECTING → PASSED
   - Aggregates softmax probabilities across 5–10 frames during INSPECTING
   - Eliminates single-frame visual flicker
       ↓
5. Live OpenCV Overlay & Event Dispatch (`main.py`)
   - Dynamic colored bounding boxes (Green / Red / Yellow / Gray)
   - Status header: ID:1 [INSPECTING] | HEALTHY (99.8%)
   - Dispatches structured event JSON on classification lock
```

---

## J. Explicit Prototype Dataset Limitation Warning

```
WARNING: Training dataset currently contains only two original images.
Results are suitable for prototype testing only. More labelled real
images are required for reliable deployment.
```

> [!WARNING]
> The current training dataset relies on two physical reference base images (`healthy_sample.jpg` and `damage_sample.jpg`), expanded via physically realistic augmentation to train the MobileNetV2 classification head.
>
> While this prototype reliably demonstrates the computer vision workflow, neural network inference, temporal tracking, and interface integration, **production deployment requires gathering and labeling additional physical conveyor joint images** across diverse lighting, wear stages, and belt speeds. Simply place new photos in `dataset/healthy/` and `dataset/damage/` and re-run `python vision/opencv/train_classifier.py`.
