# JointGuard — Conveyor-Belt Joint Health Monitoring System (Phase 1 Prototype)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=Python)](https://python.org/)
[![React](https://img.shields.io/badge/React-19.2+-61DAFB.svg?style=flat&logo=React)](https://reactjs.org/)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy-003B57.svg?style=flat&logo=SQLite)](https://www.sqlite.org/)
[![Pytest](https://img.shields.io/badge/Pytest-Passing-brightgreen.svg?style=flat&logo=Pytest)](https://docs.pytest.org/)

JointGuard treats every conveyor belt joint as an individually monitored asset. This repository contains the **PHASE 1** student / Smart India Hackathon (SIH) working prototype featuring:
- **FastAPI REST API backend** with telemetry persistence, alert generation, and simulation engine.
- **Rule-based weighted sensor fusion scoring** across 4 orthogonal inspection channels.
- **React 19 + Vite SCADA industrial frontend** with real-time telemetry, interactive 3-joint inspection, and simulation controls.
- **ESP32 sensor node firmware structure** with hardware pin mappings.

---

> [!IMPORTANT]
> ### System-Wide Phase 1 Architectural Disclaimers
> 1. **Rule-Based Weighted Fusion**: Current health scoring is strictly a **deterministic, rule-based weighted fusion** (not machine learning/deep learning).
> 2. **A3144 Hall Effect Sensor**: The A3144 sensor is a **magnetic-event demonstration sensor** (detecting physical magnet markers over joints), **NOT** an industrial electromagnetic/MFL steel-cord inspection system.
> 3. **Vision Pipeline Status**: In Phase 1, `vision_score` is a **simulated or manually injected score** (0–100). YOLOv8 computer vision detection is scheduled for Phase 2.
> 4. **Communication Protocol**: Phase 1 operates via **REST API polling** (2.0-second interval). WebSocket push communication is planned for Phase 2.

---

## 📁 Repository Structure

```
jointgurd_proto/
├── backend/                       # FastAPI REST API Backend
│   ├── main.py                    # Server entrypoint & lifespan management
│   ├── config.py                  # Thresholds, baselines, weights & CORS settings
│   ├── api/                       # REST endpoint route handlers
│   │   ├── joints.py              # /joints, /joints/{id}
│   │   ├── sensors.py             # /joints/{id}/sensors, /sensors/raw
│   │   ├── health.py              # /health, /health/overview
│   │   ├── alerts.py              # /alerts, /alerts/history, /alerts/{id}/ack
│   │   ├── history.py             # /history/{id}
│   │   └── simulation.py          # /simulation/start, /stop, /reset, /step, /inject
│   ├── services/
│   │   ├── scoring.py             # Component sensor scoring functions
│   │   ├── health_score.py        # Weighted fusion & strict UNKNOWN risk classifier
│   │   ├── alert_service.py       # Alert rule evaluator & DB logging
│   │   └── simulation_service.py  # Multi-joint inspection state machine
│   ├── database/
│   │   ├── db.py                  # SQLAlchemy engine & session setup (SQLite)
│   │   └── crud.py                # Database queries and insert operations
│   ├── models/
│   │   ├── schemas.py             # Pydantic v2 data validation schemas
│   │   └── orm_models.py          # SQLAlchemy table models
│   ├── tests/                     # Automated unit and API integration tests
│   │   ├── test_scoring.py        # Scoring math & UNKNOWN tests
│   │   └── test_api.py            # FastAPI TestClient endpoint tests
│   ├── requirements.txt           # Backend Python package requirements
│   └── websocket/                 # Architecture stub for Phase 2 WebSocket layer
├── frontend/                      # React 19 + Vite SCADA Dashboard
│   ├── src/
│   │   ├── components/            # UI components (MetricCard, Modal, SensorChart, etc.)
│   │   ├── context/               # SimulationContext (REST API auto-polling)
│   │   ├── data/                  # Central constants (JOINT_IDS: J01, J02, J03)
│   │   ├── pages/                 # Multi-page views (Dashboard, Joints, Sensors, etc.)
│   │   └── index.css              # Dark SCADA design system tokens & styles
│   ├── package.json               # Frontend dependencies & scripts
│   └── README.md                  # Frontend documentation
├── esp32/                         # Sensor node firmware (Arduino/C++)
│   └── sensor_node/
│       └── sensor_node.ino        # ESP32 pin definitions (DS18B20, MPU6050, A3144)
├── ml/                            # Phase 2 ML placeholder stubs
├── vision/                        # Phase 2 OpenCV / YOLOv8 placeholder stubs
├── data/                          # Data storage
├── requirements.txt               # Root convenience requirements pointer
└── README.md                      # System documentation
```

---

## 🧮 4-Channel Weighted Fusion Scoring Formula

$$\text{health\_score} = 0.40 \times \text{vision\_score} + 0.30 \times \text{magnetic\_score} + 0.20 \times \text{vibration\_score} + 0.10 \times \text{temperature\_score}$$

### Component Rules (`backend/config.py`):
1. **Vision Score (40%)**: Simulated or manually injected visual condition score (0–100).
2. **Magnetic Score (30%)**: A3144 Hall-effect pulse detection matching expected inspection zone pulse (100 if matched, 20 if mismatched).
3. **Vibration Score (20%)**: MPU6050 RMS acceleration deviation from nominal baseline (0.50 m/s²). Linear degradation to 70 at +20% deviation, 40 at +50% deviation.
4. **Temperature Score (10%)**: DS18B20 reading in °C. Nominal (100) below 45°C; warning linear drop to 40 between 45–60°C; critical below 40 above 60°C.
5. **Strict UNKNOWN Handling**: If **ANY** sensor reading is disconnected or missing (`None`), `health_score` immediately becomes `None` (`UNKNOWN`), `risk_level` becomes `UNKNOWN`, and `is_sufficient_data` becomes `false`. Missing data is **never** silently assumed to be zero or healthy.

### Risk Level Tiers:
| Score Range | Risk Level | Status Badge | Action Required |
| :---: | :---: | :---: | :--- |
| **70 – 100** | `LOW` | `NORMAL` (Green) | Belt joint healthy; routine inspection cycle. |
| **40 – 69** | `MEDIUM` | `MEDIUM` (Yellow) | Minor thermal / vibrational warning; schedule check. |
| **0 – 39** | `HIGH` | `HIGH` (Red) | Critical joint degradation; stop conveyor inspection. |
| *Missing Sensor* | `UNKNOWN` | `UNKNOWN` (Grey) | Sensor disconnected / telemetry drop. |

---

## ⚡ Commands to Run Backend & Frontend

### 1. Backend Setup & Run

Open a terminal in the project directory:

```bash
# 1. Navigate to jointgurd_proto
cd jointgurd_proto

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install backend dependencies
python -m pip install -r backend/requirements.txt

# 4. Run the FastAPI backend server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

> **Windows Tip**: Always use `python -m uvicorn ...` instead of `uvicorn ...` directly. This ensures Python resolves the module correctly even if the Python `Scripts/` directory is not added to your Windows user `PATH`.

- **API Documentation (Interactive Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Alternative Docs (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Root Status & Disclaimers**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

#### Running Backend Automated Tests:
```bash
python -m pytest backend/tests
```

---

### 2. Frontend Setup & Run

Open a **second terminal**:

```bash
# 1. Navigate to the frontend directory
cd jointgurd_proto/frontend

# 2. Install Node dependencies (if not already installed)
npm install

# 3. Start the Vite development server
npm run dev
```

- **Frontend Dashboard URL**: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Simulation Controls & Manual Override Endpoints

The background simulation engine cycles **3 joints (`J01`, `J02`, `J03`)** through an inspection state machine (`APPROACHING` → `INSPECTING` → `PASSED`).

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/simulation/start` | Starts the automatic background loop (cycles every 2.0s). |
| `POST` | `/simulation/stop` | Pauses the background simulation loop. |
| `POST` | `/simulation/reset` | Resets cycle counter and clears any manual sensor overrides. |
| `POST` | `/simulation/step` | Advances the simulation state machine by exactly one step. |
| `POST` | `/simulation/inject` | Manually injects custom sensor readings or failure values. |
| `GET` | `/simulation/status` | Returns current simulation running state and cycle index. |

### Example: Injecting a Joint Failure via cURL / PowerShell

```bash
curl -X POST http://127.0.0.1:8000/simulation/inject \
  -H "Content-Type: application/json" \
  -d '{"joint_id": "J03", "temperature": 68.5, "vibration": 2.8, "vision_score": 35.0}'
```

The scoring engine will immediately evaluate `J03`, reclassify its status to `HIGH`, trigger an alert in `/alerts`, and update the live dashboard in real time!
