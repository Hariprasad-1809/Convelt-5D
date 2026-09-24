# Convelt-5D — Conveyor-Belt Joint Health Monitoring System

[![FastAPI](https://img.shields.io/badge/FastAPI-%3E%3D0.110-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=Python)](https://python.org/)
[![React](https://img.shields.io/badge/React-19.2-61DAFB.svg?style=flat&logo=React)](https://reactjs.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics%3E%3D8.0-FF6600.svg?style=flat)](https://ultralytics.com/)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy%3E%3D2.0-003B57.svg?style=flat&logo=SQLite)](https://www.sqlite.org/)
[![Pytest](https://img.shields.io/badge/Tests-44%20passed-brightgreen.svg?style=flat&logo=pytest)](https://docs.pytest.org/)

> **Smart India Hackathon (SIH) Prototype** — An IoT + computer-vision predictive maintenance platform for industrial conveyor belt joints. Fuses real-time camera-based joint inspection (YOLOv8 classification) with multi-sensor hardware telemetry (temperature, vibration, hall-effect) via an Arduino UNO to detect damaged joints and automatically stop the motor before failure, while streaming live status to a React monitoring dashboard.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Key Features](#key-features)
4. [Tech Stack](#tech-stack)
5. [Repository Structure](#repository-structure)
6. [Getting Started](#getting-started)
7. [Usage](#usage)
8. [Model and Dataset Notes](#model-and-dataset-notes)
9. [Testing](#testing)
10. [API Reference](#api-reference)
11. [Known Limitations and Future Work](#known-limitations-and-future-work)
12. [Contributors](#contributors)
13. [License](#license)

---

## Overview

Conveyor belt joint failures are a leading cause of unplanned downtime in mining, manufacturing, and logistics operations. A joint splice failure at speed can damage the entire conveyor structure and any material being transported.

Convelt-5D addresses this by combining two independent monitoring channels:

- **Vision channel**: A fixed camera continuously classifies belt joints as `HEALTHY` or `DAMAGE` using a YOLOv8n-cls model trained on real fixed-rig reference images, with OpenCV-based metallic joint localisation.
- **Sensor channel**: An Arduino UNO polls a DS18B20 temperature sensor and an Adafruit MPU6050 IMU (vibration/acceleration), reporting live telemetry to the Python backend over USB serial at 9600 baud.

When the vision pipeline confirms a `DAMAGE` verdict (majority vote across at least 3 sharp frames), the backend automatically sends a `STOP` command over serial to the Arduino, which immediately halts the motor PWM output. An operator can resume the belt via a dashboard button or the `/api/v1/vision/resume` endpoint.

> **Scope**: This is a working engineering prototype built for the Smart India Hackathon (SIH). It is **not** certified industrial safety equipment.

---

## System Architecture

```
Conveyor Belt Hardware
+-- Arduino UNO Sensor Node (USB Serial, 9600 baud)
|   +-- DS18B20  - temperature (degrees C), pin 4
|   +-- MPU6050  - 3-axis acceleration / vibration RMS (I2C)
|   +-- A3144    - hall-effect magnetic event detector (joint marker)
|   +-- Motor    - PWM output, pin 9 (START/STOP/SPEED buttons: pins 5-8)
|   +-- Serial output: CSV telemetry lines; listens for STOP / RESUME commands
|
+-- Webcam
    +-- Vision Pipeline (backend background thread, decoupled from streaming)
        +-- OpenCV - specular highlight detection -> joint bounding-box crop
        +-- Sharpness / quality gate (Laplacian variance >= 100)
        +-- YOLOv8n-cls (joint_yolo_classifier_v4.pt) - HEALTHY / DAMAGE
        +-- Multi-frame majority vote (target 5 sharp frames, early-exit at strong confidence)
        +-- On confirmed DAMAGE -> serial_service.send_command("STOP")
            |
            +-- FastAPI Backend (Uvicorn, port 8000)
                +-- REST API  - joints, sensors, health, alerts, history, simulation, vision
                +-- WebSocket - /api/v1/ws/telemetry (live sensor + vision broadcasts)
                +-- MJPEG stream - /api/v1/vision/stream (~30 FPS, HUD overlays)
                +-- SQLAlchemy / SQLite - telemetry persistence & alert history
                +-- Simulation engine - 5-joint state machine for offline demo
                    |
                    +-- React + Vite Dashboard (localhost:5173)
                        +-- REST polling every 2 s + live WebSocket telemetry
                        +-- VisionResultPanel - live MJPEG feed + detection overlay
                        +-- Recharts - real-time temperature & vibration sparklines
                        +-- Motor interlock controls - STOP / Resume buttons
                        +-- 7 pages: Dashboard, Joint Monitoring, Sensor Data,
                                     Alerts, History/Reports, About, Settings
```

**Key design decision — decoupled vision pipeline**: The camera and YOLO inference run in a dedicated background thread at full frame rate, independent of whether anyone is watching the live stream endpoint. This avoids the latency and reliability problems that arise when raw video is streamed to the frontend before inference. The frontend receives detection *results* (verdict + confidence), not raw video.

---

## Key Features

### Implemented

- **Real-time joint detection** — OpenCV metallic-highlight contour localisation extracts a tight joint crop from the fixed ROI; YOLOv8n-cls classifies it as `HEALTHY`, `DAMAGE`, or `UNCERTAIN` (below 60% confidence threshold).
- **Robust joint state machine** — tracks each joint through `APPROACHING -> INSPECTING -> CONFIRMED -> PASSED` with:
  - Multi-frame accumulation and majority vote (target 5 sharp frames)
  - Early-exit confidence locking when strong evidence arrives before zone exit
  - Zone-boundary hysteresis / debouncing (entry zone 70%, tracking zone 90%) to prevent flicker
  - IoU-based re-identification across temporary dropouts
  - Stale-track cleanup on timeout
- **Automatic motor safety interlock** — confirmed `DAMAGE` verdict triggers `serial_service.send_command("STOP")`, halting motor PWM. A one-shot guard prevents serial spam for the same joint. The `motor_stop_confirmed` flag surfaces to the frontend.
- **Manual operator controls** — `POST /api/v1/vision/stop` (emergency stop) and `POST /api/v1/vision/resume` (resume with guard reset); corresponding buttons in the React dashboard.
- **4-channel weighted health fusion scoring**:

  ```
  health_score = 0.40 x vision_score
               + 0.30 x magnetic_score
               + 0.20 x vibration_score
               + 0.10 x temperature_score
  ```

  | Score  | Risk Level | Meaning                                 |
  |--------|------------|-----------------------------------------|
  | 70-100 | `LOW`      | Healthy; routine cycle                  |
  | 40-69  | `MEDIUM`   | Warning; schedule inspection            |
  | 0-39   | `HIGH`     | Critical; stop conveyor                 |
  | Any `None` sensor | `UNKNOWN` | Missing data — never assumed healthy |

- **Strict UNKNOWN handling** — if *any* sensor reading is `None` (disconnected), `health_score` becomes `None`, `risk_level` becomes `UNKNOWN`, and `is_sufficient_data` is `false`. Missing data is never silently assumed to be zero or healthy.
- **Live WebSocket telemetry** — backend broadcasts sensor readings and vision updates to the React dashboard via `/api/v1/ws/telemetry`; serial telemetry is parsed and rebroadcast immediately on arrival.
- **MJPEG live camera stream** — `/api/v1/vision/stream` delivers annotated frames (~30 FPS) with HUD overlays (verdict, confidence, bounding box, zone state, sharpness score). Stream encoding is gated behind an active viewer counter to save CPU when no one is watching.
- **7-page React dashboard** — Dashboard, Joint Monitoring, Sensor Data, Alerts, History/Reports, About, Settings — custom dark SCADA CSS design system (no Tailwind).
- **Simulation engine** — 5-joint simulation (`J01`-`J05`) with distinct sensor profiles (healthy, vibration warning, high-temp critical, sensor-disconnected UNKNOWN, nominal) for offline demo without hardware.
- **REST API with Swagger UI** — full interactive docs at `http://127.0.0.1:8000/docs`.
- **SQLite persistence** — all sensor readings and alerts stored in `convelt-5d.db`.
- **Arduino auto-detection** — `serial_service` auto-scans COM ports for CH340/FTDI/Arduino VID:PID, skipping Bluetooth ports; reconnects automatically every 3 seconds on disconnect.
- **Diagnostic tooling** — `data/joint_events.jsonl` JSONL event log; per-joint crop contact-sheet grids saved to `data/debug_crops/`.

### Planned / Future Work

See the [Known Limitations and Future Work](#known-limitations-and-future-work) section.

---

## Tech Stack

| Layer | Technology | Version (verified from project files) |
|-------|-----------|---------------------------------------|
| **Firmware** | Arduino C++ / PlatformIO | `platform = atmelavr`, `board = uno` |
| **Firmware libs** | OneWire, DallasTemperature, Adafruit MPU6050 | via `platformio.ini` lib_deps |
| **Backend language** | Python | 3.10+ (tested on 3.13) |
| **Backend framework** | FastAPI + Uvicorn | `>=0.110.0` / `>=0.28.0` |
| **ORM / DB** | SQLAlchemy + SQLite | `>=2.0.28` |
| **Data validation** | Pydantic v2 | `>=2.6.4` |
| **Serial comms** | PySerial | `>=3.5` |
| **WebSockets** | websockets | `>=12.0` |
| **Computer vision** | OpenCV | `>=4.8.0` |
| **ML inference** | Ultralytics YOLOv8 + PyTorch | `>=8.0.0` / `>=2.0.0` |
| **Image processing** | NumPy, Pillow | `>=1.24.0` / `>=9.0.0` |
| **Testing** | Pytest | `>=8.0.2` (9.1.1 in use) |
| **Frontend framework** | React + Vite | `19.2.8` / `8.2.2` |
| **Routing** | react-router-dom | `7.18.3` |
| **Charts** | Recharts | `3.10.1` |
| **Icons** | lucide-react | `1.42.0` |
| **Linting** | oxlint | `1.79.0` |
| **Styling** | Vanilla CSS (custom SCADA design system) | — |

---

## Repository Structure

```
jointgurd_proto/
+-- backend/                        # FastAPI REST + WebSocket backend
|   +-- main.py                     # Server entry point, lifespan, CORS, router registration
|   +-- config.py                   # Centralised thresholds, weights, serial & vision config
|   +-- api/
|   |   +-- joints.py               # GET/PATCH /joints, /joints/{id}
|   |   +-- sensors.py              # GET /joints/{id}/sensors, /sensors/raw
|   |   +-- health.py               # GET /joints/{id}/health, /joints/{id}/risk, /health/overview
|   |   +-- alerts.py               # GET /alerts, /alerts/history, POST /alerts/{id}/ack
|   |   +-- history.py              # GET /history/{id}
|   |   +-- simulation.py           # POST /simulation/start|stop|reset|step|inject, GET /status
|   |   +-- vision.py               # GET /api/v1/vision/status|latest|stream, POST /resume|/stop
|   +-- services/
|   |   +-- vision_service.py       # YOLOv8 pipeline, joint state machine, MJPEG stream, motor interlock
|   |   +-- serial_service.py       # Arduino UNO serial reader, parser, auto-reconnect, WS broadcast
|   |   +-- simulation_service.py   # 5-joint simulation engine (state machine, sensor profiles)
|   |   +-- scoring.py              # Per-channel score functions (vibration, temperature, magnetic, vision)
|   |   +-- health_score.py         # Weighted fusion + strict UNKNOWN handling
|   |   +-- alert_service.py        # Alert rule evaluator and DB logger
|   +-- database/
|   |   +-- db.py                   # SQLAlchemy engine, session factory, init_db()
|   |   +-- crud.py                 # DB read/write helpers
|   +-- models/
|   |   +-- orm_models.py           # SQLAlchemy table definitions
|   |   +-- schemas.py              # Pydantic v2 request/response schemas
|   +-- websocket/
|   |   +-- telemetry_ws.py         # WebSocket ConnectionManager, /api/v1/ws/telemetry endpoint
|   +-- tests/                      # Automated test suite (44 tests, all passing)
|   |   +-- test_scoring.py         # Sensor scoring unit tests (5)
|   |   +-- test_serial_parser.py   # Arduino telemetry parsing tests (13)
|   |   +-- test_motor_stop_on_damage.py    # Motor interlock + serial tests (9)
|   |   +-- test_joint_state_machine.py     # Vision state machine tests (3)
|   |   +-- test_vision_tracker.py          # Temporal aggregation tests (4)
|   |   +-- test_continuous_motion_robustness.py  # Zone flicker / re-ID tests (7)
|   |   +-- test_vision_fixes.py            # Stale-box / noise-recovery tests (3)
|   |   +-- test_api.py                     # REST integration test (script only)
|   +-- requirements.txt            # Backend Python dependencies
|
+-- frontend/                       # React 19 + Vite SCADA dashboard
|   +-- src/
|   |   +-- App.jsx                 # Router setup - 7 pages
|   |   +-- index.css               # Dark SCADA design system tokens & global styles
|   |   +-- components/
|   |   |   +-- dashboard/          # Dashboard-specific widgets
|   |   |   +-- layout/             # AppLayout, Sidebar, Header
|   |   |   +-- ui/                 # Reusable primitives (MetricCard, Modal, etc.)
|   |   +-- context/
|   |   |   +-- SimulationContext.jsx  # Central state: REST polling, WebSocket, simulation controls
|   |   +-- data/                   # constants.js (API_BASE, JOINT_IDS), mockData.js
|   |   +-- hooks/                  # Custom React hooks
|   |   +-- pages/
|   |       +-- Dashboard.jsx       # Overview: health scores, vision feed, motor status
|   |       +-- JointMonitoring.jsx # Per-joint detail, zone state, component scores
|   |       +-- SensorData.jsx      # Real-time Recharts: temperature + vibration
|   |       +-- Alerts.jsx          # Active & historical alert list with acknowledge
|   |       +-- HistoryReports.jsx  # Historical health trend data
|   |       +-- AboutProject.jsx    # Project info, architecture, team
|   |       +-- Settings.jsx        # Configuration panel
|   +-- package.json                # Node dependencies and dev scripts
|
+-- check_yolo/                     # Standalone YOLOv8 training toolchain & webcam inference
|   +-- dataset/
|   |   +-- train/healthy/          # 68 training images (1 orig + 25 close-up aug + 35 fixed-rig aug + 6 webcam)
|   |   +-- train/damage/           # 67 training images (1 orig + 25 close-up aug + 35 fixed-rig aug + 5 webcam)
|   |   +-- val/                    # 1 reference image per class (unaugmented)
|   |   +-- collected/              # Live webcam captures (h/d keyboard shortcuts)
|   +-- models/
|   |   +-- joint_yolo_classifier.pt    # v1 - close-up baseline (~3.0 MB)
|   |   +-- joint_yolo_classifier_v2.pt # v2 - multi-distance
|   |   +-- joint_yolo_classifier_v4.pt # v4 - fixed-rig calibrated, CURRENT DEFAULT (~3.0 MB)
|   +-- results/                    # Training metrics, loss curves, confusion matrices
|   +-- prepare_dataset.py          # Image augmentation generator
|   +-- train.py                    # Transfer-learning training pipeline (yolov8n-cls)
|   +-- predict.py                  # Single-image CLI inference
|   +-- test_model.py               # Automated reference image test
|   +-- webcam.py                   # Live webcam inference + dataset collection tool
|   +-- requirements.txt            # Vision-only dependencies
|
+-- esp32/sensor_node/              # Arduino UNO firmware (PlatformIO; dir name is a misnomer)
|   +-- sensor_node.ino             # Firmware: DS18B20, MPU6050, A3144, motor PWM, serial I/O
|   +-- platformio.ini              # Board: uno, platform: atmelavr, baud: 9600
|
+-- vision/                         # CV placeholders for future ByteTrack / MFL work
+-- data/                           # Runtime storage (joint_events.jsonl)
+-- ml/                             # ML experiment stubs (future)
+-- scripts/                        # Utility scripts
+-- requirements.txt                # Root pointer -> backend/requirements.txt
+-- .env.example                    # Environment variable template
+-- yolov8n-cls.pt                  # Base YOLOv8 nano classifier weights
+-- convelt-5d.db                   # SQLite database (auto-created on first run)
```

---

## Getting Started

### Prerequisites

- **Python** 3.10 or newer
- **Node.js** 18 or newer + npm
- **Arduino IDE** or **PlatformIO** (for firmware upload; optional if using simulation only)
- A webcam — optional; the system runs without one using the simulation engine

### 1. Clone and set up the environment

```bash
git clone https://github.com/Hariprasad-1809/jointgurd_proto.git
cd jointgurd_proto

# Create and activate a virtual environment (recommended)
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
# Installs everything specified in backend/requirements.txt
pip install -r requirements.txt
```

> **Note**: `ultralytics` pulls in PyTorch. On a CPU-only machine this may take several minutes. No CUDA GPU is required — inference runs on CPU.

### 3. Configure the environment (optional)

```bash
cp .env.example .env
```

Edit `.env` to set your Arduino serial port and camera index:

```ini
SERIAL_PORT=COM5             # Windows; use /dev/ttyUSB0 on Linux
SERIAL_BAUD=9600
VISION_CAMERA_INDEX=0        # 0 = built-in, 1 = first external USB webcam
VISION_MODEL_PATH=check_yolo/models/joint_yolo_classifier_v4.pt
VISION_CONF_THRESH=0.60
```

The backend auto-detects the Arduino port by scanning for CH340/FTDI/Arduino USB VID:PID, so `SERIAL_PORT` is only a fallback.

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Flash the Arduino firmware (hardware path only)

From the `esp32/sensor_node/` directory using the PlatformIO CLI:

```bash
pio run --target upload
```

Or open the project in the PlatformIO IDE and click "Upload". The firmware targets `board = uno` (Arduino UNO / ATmega328P). Monitor serial output at 9600 baud to confirm DS18B20 and MPU6050 initialisation messages.

---

## Usage

### Run the full system

**Terminal 1 — start the backend:**

```bash
# From the project root (jointgurd_proto/)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

> **Windows tip**: Always use `python -m uvicorn` rather than `uvicorn` directly to ensure the module resolves correctly even if Python `Scripts/` is not on your PATH.

**Terminal 2 — start the frontend:**

```bash
cd frontend
npm run dev
```

- **React dashboard**: http://localhost:5173
- **Swagger UI (API docs)**: http://127.0.0.1:8000/docs
- **ReDoc (alternative docs)**: http://127.0.0.1:8000/redoc

On startup the backend will:
1. Initialise (or migrate) the SQLite database.
2. Run one simulation step to populate baseline readings immediately.
3. Attempt to open the configured serial port; retries every 3 s if not found.
4. Attempt to open the webcam and load the YOLO model.

If neither the Arduino nor the camera is connected, the system still runs correctly using the simulation engine. Vision status shows `DISCONNECTED`.

### Standalone YOLO webcam test (no backend required)

```bash
# Live inference with v4 model (default: camera index 1)
python check_yolo/webcam.py

# Use built-in camera (index 0):
python check_yolo/webcam.py --source 0

# Show the exact joint crop sent to YOLO in a second window:
python check_yolo/webcam.py --source 0 --show-crop

# Use the v2 model:
python check_yolo/webcam.py --v2

# Custom search ROI (normalised x1,y1,x2,y2 fractions):
python check_yolo/webcam.py --roi 0.15,0.20,0.85,0.80

# Bypass OpenCV joint localisation and pass whole ROI to YOLO:
python check_yolo/webcam.py --direct-roi
```

**Keyboard shortcuts in the webcam window:**

| Key | Action |
|-----|--------|
| `H` | Save current joint crop as HEALTHY sample to `dataset/collected/healthy/` |
| `D` | Save current joint crop as DAMAGE sample to `dataset/collected/damage/` |
| `Q` | Quit and release the camera |

### Test single image prediction

```bash
python check_yolo/predict.py check_yolo/dataset/train/healthy/healthy_001.jpg
python check_yolo/predict.py check_yolo/dataset/train/damage/damage_001.jpg --show
```

### Retrain the classifier

```bash
# Step 1: Regenerate the augmented training dataset
python check_yolo/prepare_dataset.py --num-augmented 25

# Step 2: Train for 20 epochs on CPU
python check_yolo/train.py --epochs 20 --batch 8 --lr0 0.001 --device cpu
```

### Simulation controls (offline demo without hardware)

The simulation engine cycles 5 joints (`J01`-`J05`) through `APPROACHING -> INSPECTING -> PASSED` every 2 seconds with distinct sensor profiles.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/simulation/start` | Start the background simulation loop |
| `POST` | `/simulation/stop` | Pause the loop |
| `POST` | `/simulation/reset` | Reset cycle counter and clear overrides |
| `POST` | `/simulation/step` | Advance exactly one step |
| `POST` | `/simulation/inject` | Inject custom sensor values for a joint |
| `GET` | `/simulation/status` | Returns running state and cycle index |

**PowerShell — inject a joint failure:**

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/simulation/inject `
  -ContentType "application/json" `
  -Body '{"joint_id": "J03", "temperature": 68.5, "vibration": 2.8, "vision_score": 35.0}'
```

**Bash / WSL equivalent:**

```bash
curl -X POST http://127.0.0.1:8000/simulation/inject \
  -H "Content-Type: application/json" \
  -d '{"joint_id": "J03", "temperature": 68.5, "vibration": 2.8, "vision_score": 35.0}'
```

---

## Model and Dataset Notes

### Model details

| Property | Value |
|----------|-------|
| Architecture | YOLOv8n-cls (nano classifier), ~1.44 M parameters |
| Base weights | `yolov8n-cls.pt` (ImageNet pretrained) |
| Training | Transfer learning, 20 epochs, batch 8, image 224x224, CPU |
| Active model | `check_yolo/models/joint_yolo_classifier_v4.pt` (~3.0 MB) |
| Confidence threshold | 0.60 (configurable via `VISION_CONF_THRESH`) |
| Classes | `healthy`, `damage` |
| Uncertainty | Predictions below 0.60 are returned as `UNCERTAIN` — no motor interlock triggered |

**Vision score mapping**: `HEALTHY -> vision_score = confidence * 100`; `DAMAGE -> vision_score = (1 - confidence) * 100`. A *lower* score always means worse condition.

### Three model versions

| Version | File | Notes |
|---------|------|-------|
| v1 | `joint_yolo_classifier.pt` | Close-up reference images only. Fails on fixed-rig frames. |
| v2 | `joint_yolo_classifier_v2.pt` | Added webcam healthy crops. Over-fit to healthy (zero webcam damage samples). |
| **v4** | `joint_yolo_classifier_v4.pt` | **Current default.** Balanced fixed-rig crops (68H/67D). Correctly classifies all fixed-rig test frames. |

### Dataset composition (v4 — verified from filesystem)

| Split | Class | Count | Composition |
|-------|-------|-------|-------------|
| Train | `healthy` | **68** | 1 original + 25 close-up aug + 6 webcam + 1 fixed-rig original + 35 fixed-rig aug |
| Train | `damage` | **67** | 1 original + 25 close-up aug + 5 webcam + 1 fixed-rig original + 35 fixed-rig aug |
| Val | `healthy` | 1 | Unaugmented reference image |
| Val | `damage` | 1 | Unaugmented reference image |

**Total training images: 135 — this is a prototype-scale dataset.**

### Accuracy on fixed-rig test frames

| Test image | Condition | v4 Result | Confidence |
|------------|-----------|-----------|------------|
| `healthy_001.jpg` | Close-up solid plate | HEALTHY | 100.0% |
| `damage_001.jpg` | Close-up cracked plate | DAMAGE | 100.0% |
| `fixed_healthy_01.jpg` | Fixed rig, solid plate | HEALTHY | 100.0% |
| `fixed_healthy_02.jpg` | Fixed rig, plate + edge shadow | HEALTHY (UNCERTAIN in live UI) | 54.0% |
| `fixed_damage_01.jpg` | Fixed rig, vertical split | DAMAGE | 100.0% |
| `fixed_damage_02.jpg` | Fixed rig, vertical split | DAMAGE | 100.0% |

`fixed_healthy_02.jpg` (54.0%) falls just below the 60% threshold and triggers `UNCERTAIN` because a prominent shadow creates high-contrast vertical gradients resembling crack features. In live operation, 5-frame temporal smoothing partially mitigates this.

### Honest limitations

- **135 total training images is prototype-scale.** The model has not been evaluated across different belt types, joint materials, lighting conditions, or camera models.
- All data was captured in a **single controlled indoor environment**. Out-of-distribution performance is unknown.
- The validation set contains **only 2 images (one per class)** — insufficient for meaningful accuracy claims.
- This demonstrates feasibility. Do not use it for safety-critical production decisions.

---

## Testing

Run the full automated test suite from the project root:

```bash
python -m pytest backend/tests -v
```

**Verified result (Python 3.13.14, pytest 9.1.1):**

```
44 passed in 35.51s
```

### Test file breakdown

| File | Tests | What it covers |
|------|-------|----------------|
| `test_scoring.py` | 5 | Individual sensor scoring functions + weighted UNKNOWN fusion |
| `test_serial_parser.py` | 13 | Arduino telemetry line parsing, Pydantic schema validation |
| `test_motor_stop_on_damage.py` | 9 | Motor interlock, serial byte correctness, one-shot guard, viewer count gate, API endpoints |
| `test_joint_state_machine.py` | 3 | EMA bbox smoothing, temporary-disappearance hold, result hysteresis |
| `test_vision_tracker.py` | 4 | Temporal aggregation: damage, healthy, uncertain, insufficient-frames paths |
| `test_continuous_motion_robustness.py` | 7 | Zone flicker, re-ID on dropout, early-lock override, no-duplicate-track-IDs |
| `test_vision_fixes.py` | 3 | Noisy-start recovery, stale-box timeout, background edge rejection |
| `test_api.py` | — | REST integration test (run as script, not collected by pytest by default) |

To run a specific file:

```bash
python -m pytest backend/tests/test_scoring.py -v
python -m pytest backend/tests/test_motor_stop_on_damage.py -v
python -m pytest backend/tests/test_continuous_motion_robustness.py -v
```

---

## API Reference

Interactive documentation is available at `http://127.0.0.1:8000/docs`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | System info, version, active configuration |
| `GET` | `/joints` | List all 5 joints with current status |
| `GET` | `/joints/{id}` | Single joint detail (zone state, risk, scores) |
| `GET` | `/joints/{id}/sensors` | Latest raw sensor readings |
| `GET` | `/joints/{id}/health` | Health score + risk level |
| `GET` | `/health/overview` | Overview across all joints |
| `GET` | `/alerts` | Active unacknowledged alerts |
| `GET` | `/alerts/history` | All historical alerts |
| `POST` | `/alerts/{id}/ack` | Acknowledge an alert |
| `GET` | `/history/{id}` | Historical health trend for a joint |
| `POST` | `/simulation/start` | Start simulation loop |
| `POST` | `/simulation/stop` | Pause simulation |
| `POST` | `/simulation/reset` | Reset simulation |
| `POST` | `/simulation/step` | Advance one step |
| `POST` | `/simulation/inject` | Inject custom sensor values |
| `GET` | `/simulation/status` | Simulation running state |
| `GET` | `/api/v1/vision/status` | Camera connection + model load state |
| `GET` | `/api/v1/vision/latest` | Latest YOLO detection result |
| `GET` | `/api/v1/vision/stream` | Live MJPEG stream (usable as `<img>` src) |
| `POST` | `/api/v1/vision/stop` | Emergency motor STOP |
| `POST` | `/api/v1/vision/resume` | Resume motor + reset interlock guard |
| `WS` | `/api/v1/ws/telemetry` | WebSocket: live sensor + vision broadcasts |

---

## Known Limitations and Future Work

### Current limitations

- **Small training dataset** — 135 total training images, single environment, single camera. The model is not validated for production deployment.
- **Controlled lighting dependency** — the OpenCV specular-highlight joint localiser relies on metallic shine. Performance degrades under low-light conditions or with non-metallic joint types.
- **Single camera / single belt segment** — one joint is inspected at a time. Multi-camera support is not implemented.
- **A3144 is a demo magnetic event detector, not an MFL system** — it detects a physical magnet marker, not the structural integrity of steel cords.
- **The firmware target is Arduino UNO, not ESP32** — `platformio.ini` specifies `board = uno` (ATmega328P). The directory name `esp32/` is a misnomer from an earlier design revision.
- **Prototype, not certified** — academic / hackathon-grade engineering, not certified to IEC 62061, ISO 13849, or any industrial safety standard.
- **Vision score is simulated when camera is absent** — without a webcam, `vision_score` comes from the simulation engine's per-joint profiles, not actual YOLO inference.

### Future work

- Expand the training dataset across diverse belt materials, joint types, and lighting conditions.
- Add automatic camera exposure and white-balance normalisation for lighting robustness.
- Integrate a proper electromagnetic or magnetic flux leakage (MFL) sensor for steel-cord inspection.
- Migrate the frontend data layer fully to WebSocket push to eliminate REST polling.
- Add multi-camera support for monitoring multiple belt segments simultaneously.
- Implement ML-based multi-sensor fusion to replace the deterministic rule-based scoring.
- Containerise the backend with Docker for reproducible deployment.
- Add role-based access control to the operator dashboard.

---

## Contributors

| Name | Role |
|------|------|
| **Hariprasad** | Lead developer — backend, vision pipeline, Arduino firmware, system integration |

---

## License

No `LICENSE` file is present in this repository. All rights reserved by the author until a license is explicitly added.
