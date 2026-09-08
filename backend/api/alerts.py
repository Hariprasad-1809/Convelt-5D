"""
Alerts REST API Router
GET /alerts — current active alerts
GET /alerts/history — full alert history
"""

from typing import List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.database.db import get_db
from backend.database import crud
from backend.models.schemas import AlertResponse

router = APIRouter(prefix="/alerts", tags=["Alerts"])

@router.get("", response_model=List[AlertResponse])
def get_active_alerts(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    """Fetch active/recent alerts (MEDIUM, HIGH, or UNKNOWN warnings)."""
    alerts = crud.get_active_alerts(db, limit=limit)
    return alerts

@router.get("/history", response_model=List[AlertResponse])
def get_alert_history(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    """Fetch historical log of all generated joint health alerts."""
    history = crud.get_alert_history(db, limit=limit)
    return history
