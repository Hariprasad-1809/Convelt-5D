"""
JointGuard Pydantic Schemas (API Data Transfer Objects)
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# System & Scoring Meta Disclaimers
# ------------------------------------------------------------------------------
class SystemMeta(BaseModel):
    scoring_method: str = Field(default="RULE_BASED_WEIGHTED_FUSION", description="Strict disclaimer: Not machine learning")
    magnetic_sensor: str = Field(default="A3144_MAGNETIC_EVENT_DEMO", description="Strict disclaimer: Demo hall sensor, not MFL")
    vision_mode: str = Field(default="SIMULATED_OR_MANUAL", description="Phase 1 status: YOLO vision simulated")


# ------------------------------------------------------------------------------
# Raw Sensor & Manual Injection Models
# ------------------------------------------------------------------------------
class SensorReadingBase(BaseModel):
    device_id: str = "ARDUINO_UNO_01"
    joint_id: str
    belt_position: float = 0.0
    vibration: Optional[float] = Field(None, description="MPU6050 RMS m/s^2 deviation")
    temperature: Optional[float] = Field(None, description="DS18B20 °C")
    hall_event: Optional[bool] = Field(None, description="A3144 Hall event detected")
    magnetic_value: Optional[float] = Field(None, description="Optional HMC5883L vector value")
    vision_score: Optional[float] = Field(None, description="Simulated / Manual vision score 0-100")
    motor_speed: Optional[int] = Field(None, description="Motor speed percentage (0-100%)")
    motor_running: Optional[bool] = Field(None, description="Motor running state")
    connection_status: str = Field(default="CONNECTED", description="Hardware serial connection status")

class ArduinoTelemetryPayload(BaseModel):
    type: str = "sensor_telemetry"
    data_source: str = "LIVE_HARDWARE"
    device_id: str = "ARDUINO_UNO_01"
    joint_id: str = "J01"
    timestamp: datetime
    temperature: Optional[float] = None
    temperature_status: str = "NORMAL"
    vibration: Optional[float] = None
    vibration_status: str = "NORMAL"
    hall_detected: Optional[bool] = False
    motor_speed: Optional[int] = 0
    motor_running: Optional[bool] = False
    health_score: Optional[float] = None
    risk_level: str = "LOW"
    connection_status: str = "CONNECTED"

class SerialStatusPayload(BaseModel):
    type: str = "connection_status"
    device_id: str = "ARDUINO_UNO_01"
    port: str = "COM4"
    baud: int = 9600
    connection_status: str = "DISCONNECTED"
    message: str = ""
    timestamp: datetime


class SimulationInjectRequest(BaseModel):
    joint_id: str
    vibration: Optional[float] = None
    temperature: Optional[float] = None
    hall_event: Optional[bool] = None
    magnetic_value: Optional[float] = None
    vision_score: Optional[float] = None


# ------------------------------------------------------------------------------
# Component & Health Score Breakdown
# ------------------------------------------------------------------------------
class ComponentScores(BaseModel):
    vision_score: Optional[float] = Field(None, description="Simulated/Manual 0-100 or None")
    magnetic_score: Optional[float] = Field(None, description="A3144 Hall event match score 0-100 or None")
    vibration_score: Optional[float] = Field(None, description="MPU6050 deviation score 0-100 or None")
    temperature_score: Optional[float] = Field(None, description="DS18B20 score 0-100 or None")

class HealthScoreResponse(BaseModel):
    joint_id: str
    timestamp: datetime
    health_score: Optional[float] = Field(None, description="Weighted fusion score 0-100 or None if UNKNOWN")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, or UNKNOWN")
    is_sufficient_data: bool
    component_scores: ComponentScores
    disclaimer: str = "RULE_BASED_WEIGHTED_FUSION (not machine learning)"


# ------------------------------------------------------------------------------
# Reading & Joint Detailed Responses
# ------------------------------------------------------------------------------
class SensorReadingResponse(SensorReadingBase):
    id: int
    timestamp: datetime
    vibration_score: Optional[float] = None
    temperature_score: Optional[float] = None
    magnetic_score: Optional[float] = None
    health_score: Optional[float] = None
    risk_level: str
    alert_type: Optional[str] = None

    class Config:
        from_attributes = True

class JointDetailResponse(BaseModel):
    joint_id: str
    name: str
    belt_position: float
    zone_state: str # APPROACHING, INSPECTING, PASSED
    status: str
    current_health: Optional[HealthScoreResponse] = None
    latest_sensors: Optional[SensorReadingBase] = None

class JointSummaryResponse(BaseModel):
    joint_id: str
    name: str
    belt_position: float
    zone_state: str
    status: str
    health_score: Optional[float] = None
    risk_level: str
    alert_type: Optional[str] = None
    last_updated: Optional[datetime] = None

    class Config:
        from_attributes = True


# ------------------------------------------------------------------------------
# Alerts & History
# ------------------------------------------------------------------------------
class AlertResponse(BaseModel):
    id: int
    timestamp: datetime
    joint_id: str
    alert_type: str
    risk_level: str
    message: str

    class Config:
        from_attributes = True


# ------------------------------------------------------------------------------
# Simulation State Response
# ------------------------------------------------------------------------------
class SimulationStatusResponse(BaseModel):
    is_running: bool
    interval_seconds: float
    current_cycle: int
    active_joints: List[str]
    meta: SystemMeta = Field(default_factory=SystemMeta)
