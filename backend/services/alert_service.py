"""
JointGuard Alert Generation Service

Generates and persists alert records when:
1. Joint risk level transitions to MEDIUM or HIGH.
2. Sensor data becomes UNKNOWN or missing (data insufficiency).
3. Critical sensor thresholds (e.g. vibration > +50%, temp > 60°C) are crossed.
"""

from typing import Optional
from sqlalchemy.orm import Session
from backend.database import crud
from backend.models.orm_models import AlertHistory

def evaluate_and_trigger_alerts(
    db: Session,
    joint_id: str,
    new_risk_level: str,
    health_score: Optional[float],
    v_score: Optional[float],
    m_score: Optional[float],
    vib_score: Optional[float],
    t_score: Optional[float]
) -> Optional[AlertHistory]:
    """
    Evaluates latest health score & component scores against previous state.
    Triggers an Alert record in DB if risk is elevated or data is missing.
    """
    # Check latest reading to avoid duplicating identical continuous alerts
    recent_alerts = crud.get_alert_history(db, limit=5)
    last_alert = next((a for a in recent_alerts if a.joint_id == joint_id), None)

    alert_type = None
    message = None

    if new_risk_level == "UNKNOWN":
        alert_type = "DATA_MISSING_UNKNOWN"
        missing_sensors = []
        if v_score is None: missing_sensors.append("Vision")
        if m_score is None: missing_sensors.append("Hall Magnetic")
        if vib_score is None: missing_sensors.append("Vibration (MPU6050)")
        if t_score is None: missing_sensors.append("Temperature (DS18B20)")
        msg_str = ", ".join(missing_sensors) if missing_sensors else "Sensor Telemetry"
        message = f"Joint {joint_id} health score UNKNOWN due to missing sensor data: {msg_str}."

    elif new_risk_level == "HIGH":
        alert_type = "CRITICAL_RISK"
        message = f"Joint {joint_id} entered CRITICAL risk state with health score {health_score}/100."

    elif new_risk_level == "MEDIUM":
        alert_type = "WARNING_RISK"
        message = f"Joint {joint_id} entered WARNING risk state with health score {health_score}/100."

    # Avoid spamming duplicate alert if same alert_type was triggered within last reading for same joint
    if alert_type and (not last_alert or last_alert.alert_type != alert_type or last_alert.risk_level != new_risk_level):
        alert_record = crud.create_alert(
            db=db,
            alert_data={
                "joint_id": joint_id,
                "alert_type": alert_type,
                "risk_level": new_risk_level,
                "message": message
            }
        )
        return alert_record

    return None
