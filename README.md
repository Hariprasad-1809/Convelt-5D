# JointGuard — Conveyor-Belt Joint Health Monitoring System (Phase 1 Prototype)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0+-61DAFB.svg?style=flat&logo=React)](https://reactjs.org/)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy-003B57.svg?style=flat&logo=SQLite)](https://www.sqlite.org/)
[![ESP32](https://img.shields.io/badge/ESP32-C%2B%2B%20Firmware-E7352C.svg?style=flat&logo=Espressif)](https://www.espressif.com/)

JointGuard treats every conveyor belt joint as an individually monitored asset. This repository contains the **PHASE 1** student/SIH working prototype built using REST APIs, a rule-based weighted fusion scoring engine, an inspection zone state machine simulation, a React frontend dashboard, and an ESP32 firmware structure.

---

> [!IMPORTANT]
> **SYSTEM-WIDE SYSTEM DISCLAIMERS**:
> 1. **Rule-Based Weighted Fusion**: Current health scoring is strictly a **RULE-BASED WEIGHTED FUSION** (not machine learning).
> 2. **A3144 Hall Effect Sensor**: The A3144 sensor is a **MAGNETIC-EVENT DEMONSTRATION** sensor (detecting physical magnet pulses over joints), **NOT** an industrial electromagnetic/MFL steel-cord inspection system.
> 3. **Vision Pipeline Status**: In Phase 1, `vision_score` is a **SIMULATED OR MANUALLY SUPPLIED** score (0-100). YOLOv8 surface defect detection is postponed to Phase 2.
> 4. **Communication Protocol**: Phase 1 operates via **REST APIs only**. WebSocket communication is postponed to Phase 2.

---

## 📁 Repository Structure

```
JointGuard/
├── frontend/                      # React (Vite + Tailwind CSS + Recharts + Lucide Icons)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ConveyorTwin.jsx  # Digital Twin inspection zone visualizer
│   │   │   ├── JointDetailModal.jsx # Health score gauge & component breakdown
│   │   │   ├── SensorCards.jsx   # Live raw sensor telemetry grid
│   │   │   ├── AlertsPanel.jsx   # Active alerts & searchable log history
│   │   │   ├── TrendCharts.jsx   # Recharts historical trend lines
│   │   │   ├── SimulationControls.jsx # Start/Stop/Reset/Step/Inject modal
│   │   │   └── AboutModal.jsx    # SEE->TRACK->SENSE->FUSE->PREDICT->ACT pipeline & architecture
│   │   ├── App.jsx               # Dashboard entrypoint with REST auto-polling
│   │   └── index.css             # Glassmorphism & animated conveyor styles
│   └── package.json
├── backend/
│   ├── main.py                    # FastAPI application entrypoint
│   ├── config.py                  # Centralized scoring weights, baselines & thresholds
│   ├── api/                       # REST API route handlers
│   │   ├── joints.py
│   │   ├── sensors.py
│   │   ├── health.py
│   │   ├── alerts.py
│   │   ├── history.py
│   │   └── simulation.py
│   ├── services/
│   │   ├── scoring.py             # Individual sensor scoring functions
│   │   ├── health_score.py        # Weighted fusion & strict UNKNOWN risk classifier
│   │   ├── alert_service.py       # Alert generator & DB logging
│   │   └── simulation_service.py  # Multi-joint (J01-J05) inspection state machine
│   ├── database/
│   │   ├── db.py                  # SQLAlchemy engine & session setup (SQLite default)
│   │   └── crud.py                # Database queries
│   ├── models/
│   │   ├── schemas.py             # Pydantic data transfer schemas
│   │   └── orm_models.py          # SQLAlchemy table schemas
│   ├── tests/                     # Automated unit and API integration tests
│   │   ├── test_scoring.py
│   │   └── test_api.py
│   └── websocket/                 # EMPTY placeholder package stub for Phase 2
├── vision/                        # EMPTY placeholder (yolo/, opencv/, tracking/ stubs for Phase 2)
├── ml/                            # EMPTY placeholder (preprocessing.py, feature_engineering.py stubs)
├── esp32/
│   └── sensor_node/
│       └── sensor_node.ino        # ESP32 C++ firmware with pin mapping & safety notes
├── data/                          # Data storage folders (.gitkeep)
├── .env.example                   # Environment configuration template
├── requirements.txt               # Backend Python dependencies
└── README.md                      # Documentation
```

---

## 🧮 Weighted Fusion Scoring Formula & Rules

$$\text{health\_score} = 0.40 \times \text{vision\_score} + 0.30 \times \text{magnetic\_score} + 0.20 \times \text{vibration\_score} + 0.10 \times \text{temperature\_score}$$

### Component Rules (`backend/config.py`):
1. **Vision Score (40%)**: Simulated / manually injected value (0-100).
2. **Magnetic Score (30%)**: A3144 Hall sensor pulse matching expected inspection zone pulse (100 if match, 20 if mismatch).
3. **Vibration Score (20%)**: MPU6050 RMS acceleration deviation from nominal baseline (0.50 m/s²). Linear drop to 70 at +20% deviation, 40 at +50% deviation.
4. **Temperature Score (10%)**: DS18B20 °C. Normal 100 below 45°C; warning linear drop to 40 between 45-60°C; critical below 40 above 60°C.
5. **Strict UNKNOWN Handling**: If **ANY** component reading is missing or disconnected (`None`), `health_score` becomes `None` (`UNKNOWN`), `risk_level` becomes `UNKNOWN`, and `is_sufficient_data` becomes `false`. Zero is **never** silently substituted.

### Risk Level Classification:
- **70 – 100**: `LOW` (Green)
- **40 – 69**: `MEDIUM` (Yellow)
- **0 – 39**: `HIGH` (Red)
- **Missing Sensor**: `UNKNOWN` (Grey / insufficient data)

---

## ⚡ Quick Start & Run Commands

### 1. Prerequisites
- **Python 3.10+**
- **Node.js v18+ & npm**

### 2. Backend Setup & Launch
```bash
# From workspace root
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI backend server (Port 8000)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Documentation (Swagger UI): `http://localhost:8000/docs`
- Root Info & Disclaimers: `http://localhost:8000/`

### 3. Frontend Setup & Launch
```bash
# Open a new terminal in jointguard_prototype/frontend/
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server (Port 5173)
npm run dev
```
- Open browser at: `http://localhost:5173/`

---

## 🧪 Simulation Controls & Manual Value Injection

The background simulation engine manages 5 joints (**J01, J02, J03, J04, J05**) moving through an inspection zone state machine (`APPROACHING` → `INSPECTING` → `PASSED`).

### Simulation REST Endpoints:
- `POST /simulation/start` — Start background simulation loop (cycles every 2.0s)
- `POST /simulation/stop` — Pause background simulation
- `POST /simulation/reset` — Reset cycle count and clear overrides
- `POST /simulation/step` — Trigger a single simulation step manually
- `POST /simulation/inject` — Manually inject sensor/vision values into a joint:

```json
// Example: POST http://localhost:8000/simulation/inject
{
  "joint_id": "J01",
  "vibration": 0.85,
  "temperature": 62.5,
  "vision_score": 35.0
}
```

---

## 🔌 Phase 2 Future Developer Integration Architecture

This codebase was structured so Phase 2 features can be dropped in without redesigning services or data models:

1. **Swapping Simulated Vision for YOLOv8 Output**:
   - The REST endpoint `POST /simulation/inject` and pure function `vision_score(val)` in `backend/services/scoring.py` take a 0-100 float.
   - When the YOLOv8 surface defect model in `vision/yolo/` is ready, it simply calls `POST /simulation/inject` with `{"joint_id": "J0x", "vision_score": yolo_confidence}`. No scoring or fusion code changes are required.

2. **Swapping REST Polling for WebSocket**:
   - The REST route handlers in `backend/api/` are thin wrappers around pure service functions (`calculate_health_score`, `crud.create_sensor_reading`).
   - In Phase 2, a WebSocket router in `backend/websocket/` can ingest raw JSON packets from ESP32 nodes, pass them to `calculate_health_score()`, and broadcast the response to WebSocket clients using the exact same `HealthScoreResponse` schema.

---

## 🚀 Phase 2 Development Roadmap (Not Implemented in Phase 1)

- [ ] **Vision Pipeline**: Integrate OpenCV frame capture, YOLOv8 defect detection weights, and ByteTrack joint object tracking in `vision/`.
- [ ] **WebSocket Real-Time Gateway**: Ingest ESP32 telemetry & push live risk updates to physical LEDs and web clients via WebSocket in `backend/websocket/`.
- [ ] **Machine Learning Predictive Engine**: Train Random Forest / XGBoost classifiers and LSTM degradation models in `ml/`.
- [ ] **Industrial MFL / EM Sensor Integration**: Upgrade magnetic event demo to industrial electromagnetic MFL steel-cord inspection hardware.
- [ ] **PLC / SCADA Integration**: Connect high-risk alerts to emergency conveyor motor trip relays.
