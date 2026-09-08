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
