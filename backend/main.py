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

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database.db import engine, Base, init_db
from backend.services.simulation_service import simulation_engine
from backend.services.serial_service import serial_service
from backend.services.vision_service import vision_service
from backend.websocket import telemetry_ws
from backend.api import joints, sensors, health, alerts, history, simulation, vision

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure DB tables & columns are initialized & run initial simulation step
    init_db()
    simulation_engine.step()
    if settings.DEBUG:
        simulation_engine.start()

    # Start Arduino UNO Serial Service and Vision Service on main event loop
    loop = asyncio.get_running_loop()
    serial_service.start(loop)
    vision_service.start(loop)

    yield
    # Shutdown: Stop vision service, serial reader & simulation engine threads
    vision_service.stop()
    serial_service.stop()
    simulation_engine.stop()

from uvicorn.protocols.utils import ClientDisconnected


class SuppressClientDisconnectASGI:
    """ASGI middleware that silently catches ClientDisconnected & CancelledError on streaming endpoints."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        try:
            await self.app(scope, receive, send)
        except (ClientDisconnected, asyncio.CancelledError):
            pass
        except BaseException as e:
            err_name = type(e).__name__
            if err_name in ("ExceptionGroup", "BaseExceptionGroup"):
                sub_excs = getattr(e, "exceptions", [])
                if sub_excs and all(
                    isinstance(sub, (ClientDisconnected, asyncio.CancelledError)) or type(sub).__name__ in ("ClientDisconnected", "CancelledError")
                    for sub in sub_excs
                ):
                    return
            raise


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="JointGuard Conveyor-Belt Joint Health Monitoring Backend (Arduino UNO Serial + YOLO Camera + WebSockets)",
    lifespan=lifespan
)

app.add_middleware(SuppressClientDisconnectASGI)

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

# Register REST & WebSocket Routers
app.include_router(joints.router)
app.include_router(sensors.router)
app.include_router(health.router)
app.include_router(alerts.router)
app.include_router(history.router)
app.include_router(simulation.router)
app.include_router(telemetry_ws.router)
app.include_router(vision.router)



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
