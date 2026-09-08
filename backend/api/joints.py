"""
Joints REST API Router
GET /joints — list all joints summary
GET /joints/{joint_id} — full details for a single joint
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.database import crud
from backend.models.schemas import JointSummaryResponse, JointDetailResponse, HealthScoreResponse, ComponentScores, SensorReadingBase

router = APIRouter(prefix="/joints", tags=["Joints"])

@router.get("", response_model=List[JointSummaryResponse])
def list_joints(db: Session = Depends(get_db)):
    """Fetch summary list of all 5 monitored conveyor joints."""
    joints = crud.get_all_joints(db)
    summaries = []
    for j in joints:
        latest = crud.get_latest_reading(db, j.joint_id)
        summaries.append(JointSummaryResponse(
            joint_id=j.joint_id,
            name=j.name,
            belt_position=j.belt_position,
            zone_state=j.zone_state,
            status=j.status,
            health_score=latest.health_score if latest else None,
            risk_level=latest.risk_level if latest else "UNKNOWN",
            alert_type=latest.alert_type if latest else None,
            last_updated=latest.timestamp if latest else None
        ))
    return summaries


@router.get("/{joint_id}", response_model=JointDetailResponse)
def get_joint_detail(joint_id: str, db: Session = Depends(get_db)):
    """Fetch complete detail for a specific joint (scores, raw metrics, risk, zone state)."""
    joint = crud.get_joint(db, joint_id)
    if not joint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Joint {joint_id} not found")

    latest = crud.get_latest_reading(db, joint_id)

    current_health = None
    latest_sensors = None

    if latest:
        is_sufficient = (
            latest.vision_score is not None and
            latest.magnetic_score is not None and
            latest.vibration_score is not None and
            latest.temperature_score is not None
        )
        current_health = HealthScoreResponse(
            joint_id=joint_id,
            timestamp=latest.timestamp,
            health_score=latest.health_score,
            risk_level=latest.risk_level,
            is_sufficient_data=is_sufficient,
            component_scores=ComponentScores(
                vision_score=latest.vision_score,
                magnetic_score=latest.magnetic_score,
                vibration_score=latest.vibration_score,
                temperature_score=latest.temperature_score
            )
        )
        latest_sensors = SensorReadingBase(
            device_id=latest.device_id,
            joint_id=latest.joint_id,
            belt_position=latest.belt_position,
            vibration=latest.vibration,
            temperature=latest.temperature,
            hall_event=latest.hall_event,
            magnetic_value=latest.magnetic_value,
            vision_score=latest.vision_score
        )

    return JointDetailResponse(
        joint_id=joint.joint_id,
        name=joint.name,
        belt_position=joint.belt_position,
        zone_state=joint.zone_state,
        status=joint.status,
        current_health=current_health,
        latest_sensors=latest_sensors
    )
