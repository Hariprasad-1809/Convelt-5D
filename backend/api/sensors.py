"""
Sensors REST API Router
GET /joints/{joint_id}/sensors — latest raw sensor readings for a joint
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.database import crud
from backend.models.schemas import SensorReadingBase

router = APIRouter(prefix="/joints", tags=["Sensors"])

@router.get("/{joint_id}/sensors", response_model=SensorReadingBase)
def get_latest_sensors(joint_id: str, db: Session = Depends(get_db)):
    """Fetch latest raw sensor telemetry for a joint."""
    joint = crud.get_joint(db, joint_id)
    if not joint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Joint {joint_id} not found")

    latest = crud.get_latest_reading(db, joint_id)
    if not latest:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No sensor readings recorded for joint {joint_id}")

    return SensorReadingBase(
        device_id=latest.device_id,
        joint_id=latest.joint_id,
        belt_position=latest.belt_position,
        vibration=latest.vibration,
        temperature=latest.temperature,
        hall_event=latest.hall_event,
        magnetic_value=latest.magnetic_value,
        vision_score=latest.vision_score
    )

from backend.services.serial_service import serial_service

@router.get("/hardware/status")
def get_hardware_status():
    """Fetch current Arduino UNO serial connection status and configuration."""
    return {
        "device_id": serial_service.device_id,
        "port": serial_service.port,
        "baud": serial_service.baud,
        "connection_status": serial_service.connection_status,
        "serial_status": serial_service.connection_status,
        "websocket_status": "ONLINE",
        "last_updated": serial_service.last_updated.isoformat() if serial_service.last_updated else None,
        "serial_reader_running": serial_service._running,
        "latest_telemetry": serial_service.latest_telemetry
    }

@router.get("/hardware/rawlog")
def get_hardware_rawlog():
    """Return last 100 raw serial lines received from Arduino (for diagnostics).
    
    If this list is empty → Arduino is not sending data (check USB cable / firmware).
    If lines are present but no telemetry parses → check line format vs parser regex.
    """
    lines = list(serial_service.raw_serial_lines)
    return {
        "port": serial_service.port,
        "connection_status": serial_service.connection_status,
        "raw_line_count": len(lines),
        "raw_lines": lines[-50:],  # Last 50 for readability
        "hint": "If raw_lines is empty, Arduino is not sending serial data. "
                "If lines exist but sensor values are null, the parser regex may not match."
    }
