"""
History REST API Router
GET /history/{joint_id} — historical trend data for graphing
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.database import crud
from backend.models.schemas import SensorReadingResponse

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/{joint_id}", response_model=List[SensorReadingResponse])
def get_joint_history(
    joint_id: str,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Fetch historical sensor readings and component scores for a joint (for trend charting)."""
    joint = crud.get_joint(db, joint_id)
    if not joint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Joint {joint_id} not found")

    history = crud.get_reading_history(db, joint_id, limit=limit)
    # Reverse so trend flows chronologically from past to present
    history.reverse()
    return history
