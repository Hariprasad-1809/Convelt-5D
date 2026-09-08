"""
Health REST API Router
GET /joints/{joint_id}/health — current health score + component breakdown
GET /joints/{joint_id}/risk — current risk level
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.database import crud
from backend.models.schemas import HealthScoreResponse, ComponentScores

router = APIRouter(prefix="/joints", tags=["Health & Risk"])

@router.get("/{joint_id}/health", response_model=HealthScoreResponse)
def get_joint_health(joint_id: str, db: Session = Depends(get_db)):
    """Fetch current weighted health score and individual component scores."""
    joint = crud.get_joint(db, joint_id)
    if not joint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Joint {joint_id} not found")

    latest = crud.get_latest_reading(db, joint_id)
    if not latest:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No telemetry readings for joint {joint_id}")

    is_sufficient = (
        latest.vision_score is not None and
        latest.magnetic_score is not None and
        latest.vibration_score is not None and
        latest.temperature_score is not None
    )

    return HealthScoreResponse(
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

@router.get("/{joint_id}/risk")
def get_joint_risk(joint_id: str, db: Session = Depends(get_db)):
    """Fetch current risk level (LOW, MEDIUM, HIGH, or UNKNOWN)."""
    joint = crud.get_joint(db, joint_id)
    if not joint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Joint {joint_id} not found")

    latest = crud.get_latest_reading(db, joint_id)
    risk_level = latest.risk_level if latest else "UNKNOWN"
    health_score = latest.health_score if latest else None

    return {
        "joint_id": joint_id,
        "risk_level": risk_level,
        "health_score": health_score,
        "status": joint.status
    }
