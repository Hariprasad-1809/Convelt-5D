"""
JointGuard CRUD Operations
Database queries for Joints, Readings, Alerts, and History.
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from backend.models.orm_models import Joint, SensorReading, AlertHistory

# ------------------------------------------------------------------------------
# Joints CRUD
# ------------------------------------------------------------------------------
def get_all_joints(db: Session) -> List[Joint]:
    return db.query(Joint).order_by(Joint.joint_id).all()

def get_joint(db: Session, joint_id: str) -> Optional[Joint]:
    return db.query(Joint).filter(Joint.joint_id == joint_id).first()

def create_or_update_joint(
    db: Session,
    joint_id: str,
    name: str,
    belt_position: float,
    zone_state: str = "APPROACHING",
    status: str = "ACTIVE"
) -> Joint:
    joint = get_joint(db, joint_id)
    if not joint:
        joint = Joint(
            joint_id=joint_id,
            name=name,
            belt_position=belt_position,
            zone_state=zone_state,
            status=status
        )
        db.add(joint)
    else:
        joint.name = name
        joint.belt_position = belt_position
        joint.zone_state = zone_state
        joint.status = status
    db.commit()
    db.refresh(joint)
    return joint


# ------------------------------------------------------------------------------
# Sensor Readings & Telemetry CRUD
# ------------------------------------------------------------------------------
def create_sensor_reading(db: Session, reading_data: dict) -> SensorReading:
    db_reading = SensorReading(**reading_data)
    db.add(db_reading)
    db.commit()
    db.refresh(db_reading)
    return db_reading

def get_latest_reading(db: Session, joint_id: str) -> Optional[SensorReading]:
    return (
        db.query(SensorReading)
        .filter(SensorReading.joint_id == joint_id)
        .order_by(desc(SensorReading.timestamp))
        .first()
    )

def get_reading_history(db: Session, joint_id: str, limit: int = 50) -> List[SensorReading]:
    return (
        db.query(SensorReading)
        .filter(SensorReading.joint_id == joint_id)
        .order_by(desc(SensorReading.timestamp))
        .limit(limit)
        .all()
    )


# ------------------------------------------------------------------------------
# Alerts CRUD
# ------------------------------------------------------------------------------
def create_alert(db: Session, alert_data: dict) -> AlertHistory:
    db_alert = AlertHistory(**alert_data)
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def get_active_alerts(db: Session, limit: int = 20) -> List[AlertHistory]:
    """Returns recent MEDIUM, HIGH, or UNKNOWN alerts."""
    return (
        db.query(AlertHistory)
        .order_by(desc(AlertHistory.timestamp))
        .limit(limit)
        .all()
    )

def get_alert_history(db: Session, limit: int = 100) -> List[AlertHistory]:
    return (
        db.query(AlertHistory)
        .order_by(desc(AlertHistory.timestamp))
        .limit(limit)
        .all()
    )
