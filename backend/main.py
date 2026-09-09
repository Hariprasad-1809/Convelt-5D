"""
JointGuard FastAPI Entrypoint (Phase 1 Prototype)

REST API server providing joint monitoring, scoring, alert management, and simulation controls.

DISCLAIMER:
- Health scoring is RULE-BASED WEIGHTED FUSION (not machine learning).
- A3144 Hall sensor is a MAGNETIC-EVENT DEMONSTRATION sensor (not industrial EM/MFL).
- Vision score is SIMULATED or MANUALLY INJECTED in Phase 1.
"""

import os
import sys

# Ensure parent directory (jointgurd_proto) is on sys.path so 'backend' package imports always succeed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database.db import engine, Base
from backend.services.simulation_service import simulation_engine
from backend.api import joints, sensors, health, alerts, history, simulation

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure DB tables are created & run initial simulation step
    Base.metadata.create_all(bind=engine)
    simulation_engine.step()
    if settings.DEBUG:
        simulation_engine.start()
    yield
    # Shutdown: Stop simulation thread
    simulation_engine.stop()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="JointGuard Conveyor-Belt Joint Health Monitoring Backend (Phase 1 REST API)",
    lifespan=lifespan
)

# Enable CORS for React frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST Routers
app.include_router(joints.router)
app.include_router(sensors.router)
app.include_router(health.router)
app.include_router(alerts.router)
app.include_router(history.router)
app.include_router(simulation.router)

@app.get("/")
def root_info():
    """System information & explicit Phase 1 architectural disclaimers."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "phase": "PHASE 1 (REST API Prototype)",
        "disclaimers": {
            "scoring": "RULE-BASED WEIGHTED FUSION (0.40 Vision + 0.30 Magnetic + 0.20 Vibration + 0.10 Temp)",
            "hall_sensor": "A3144 Magnetic Event Demonstration Sensor (not industrial EM/MFL)",
            "vision": "Simulated / Manually Injected Score (YOLOv8 postponed to Phase 2)",
            "websocket": "REST API Polling in Phase 1 (WebSocket real-time layer postponed to Phase 2)"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
