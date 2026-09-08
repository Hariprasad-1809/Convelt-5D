# JointGuard Skills Matrix

## Backend Engineering

- FastAPI is used in [backend/main.py](backend/main.py) to expose the REST API, wire CORS, and bootstrap the simulation lifecycle.
- REST API design is implemented across [backend/api/](backend/api) with thin route modules for joints, sensors, health, alerts, history, and simulation controls.
- Pydantic validation is defined in [backend/models/schemas.py](backend/models/schemas.py) to keep request and response payloads typed and consistent.
- SQLAlchemy modeling is handled in [backend/database/db.py](backend/database/db.py) and [backend/models/orm_models.py](backend/models/orm_models.py) to persist joints, readings, and alerts in SQLite.
- Service-layer architecture is implemented in [backend/services/](backend/services) so scoring, alerting, and simulation logic stay separate from route handlers.

## Sensor & Embedded Systems

- ESP32 firmware structure lives in [esp32/sensor_node/sensor_node.ino](esp32/sensor_node/sensor_node.ino) and mirrors the joint telemetry fields used by the backend.
- MPU6050 I2C vibration sensing is represented in the firmware and backend scoring path as the vibration input used for health classification.
- DS18B20 1-Wire temperature sensing is used as the thermal telemetry input that feeds the weighted score and live dashboard cards.
- A3144 Hall-style digital sensing is used as the magnetic-event demo input and is surfaced in the UI as the magnetic/Hall channel.
- Fault-safe UNKNOWN handling is enforced when sensor data is missing so the dashboard and scoring pipeline do not invent values.

## Systems Design

- Rule-based weighted fusion is implemented in [backend/services/health_score.py](backend/services/health_score.py) to combine vision, magnetic, vibration, and temperature inputs into one health score.
- Threshold-based risk classification is defined in [backend/config.py](backend/config.py) and used to map scores into LOW, MEDIUM, HIGH, or UNKNOWN states.
- The inspection-zone state machine is handled in [backend/services/simulation_service.py](backend/services/simulation_service.py) to move five joints through APPROACHING, INSPECTING, and PASSED.
- The architecture is modular enough that manual overrides and future vision/WebSocket layers can plug into the same service boundaries without changing the UI contract.

## Data & Simulation

- Simulated telemetry generation is implemented in [backend/services/simulation_service.py](backend/services/simulation_service.py) to produce healthy, warning, critical, and unknown joint states.
- Historical storage is written through the SQLite reading tables in [backend/models/orm_models.py](backend/models/orm_models.py) so the trend chart has a persistent time series.
- Joint state tracking is stored with belt position, zone state, and status so the conveyor twin can animate real joint movement.

## Frontend Engineering

- The frontend is a React 19 SPA built with Vite in [frontend/src/App.jsx](frontend/src/App.jsx).
- The styling stack is Tailwind CSS v4 plus custom glass/neon theme tokens and glow utilities in [frontend/src/index.css](frontend/src/index.css).
- The dark industrial dashboard design uses semi-transparent panels, neon risk glows, scanline/grid backgrounds, and animated status accents across the components.
- Recharts powers the live historical visualizations in [frontend/src/components/TrendCharts.jsx](frontend/src/components/TrendCharts.jsx) with threshold lines and dark chart styling.
- The UI polls REST endpoints on a timer in [frontend/src/App.jsx](frontend/src/App.jsx) to keep the digital twin, alerts, sensors, and simulation state in sync.
- Responsive digital-twin layouts are implemented across [frontend/src/components/](frontend/src/components) so the dashboard reads cleanly on projector and laptop resolutions.

## Planned / Future Skills (Not Yet Implemented)

- YOLO object detection is planned for [vision/yolo/](vision/yolo) to detect belt damage and surface anomalies from camera frames.
- ByteTrack multi-object tracking is planned for [vision/tracking/](vision/tracking) to preserve joint identity across frames.
- WebSocket telemetry streaming is planned for [backend/websocket/](backend/websocket) to replace REST polling with live push updates.
- Machine-learning prediction models are planned in [ml/](ml) for Random Forest, XGBoost, and sequence-based degradation forecasting.
- Industrial EM/MFL sensing is planned as a later hardware upgrade beyond the current Hall-sensor demonstration path.
- PLC/SCADA integration is planned for future actuation and alarm handoff once the prototype moves beyond the REST demo phase.
