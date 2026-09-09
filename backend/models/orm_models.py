"""
JointGuard SQLAlchemy ORM Models
Defines tables for Joints, Telemetry Sensor Readings, and Alert History.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Boolean, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.database.db import Base

class Joint(Base):
    __tablename__ = "joints"

    joint_id = Column(String(20), primary_key=True, index=True) # e.g. J01, J02
    name = Column(String(50), nullable=False)                    # e.g. "Conveyor Joint J01"
    belt_position = Column(Float, nullable=False, default=0.0)    # Position along conveyor belt (meters)
    zone_state = Column(String(20), nullable=False, default="APPROACHING") # APPROACHING, INSPECTING, PASSED
    status = Column(String(20), nullable=False, default="ACTIVE") # ACTIVE, OFFLINE, MAINTENANCE

    readings = relationship("SensorReading", back_populates="joint", cascade="all, delete-orphan")
    alerts = relationship("AlertHistory", back_populates="joint", cascade="all, delete-orphan")


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    device_id = Column(String(50), nullable=False, default="ESP32_NODE_01")
    joint_id = Column(String(20), ForeignKey("joints.joint_id"), nullable=False, index=True)
    belt_position = Column(Float, nullable=False)

    # Raw Sensor Readings
    vibration = Column(Float, nullable=True)        # MPU6050 RMS m/s^2 (nullable if sensor UNKNOWN)
    temperature = Column(Float, nullable=True)      # DS18B20 °C (nullable if sensor UNKNOWN)
    hall_event = Column(Boolean, nullable=True)     # A3144 Hall event detected (nullable if UNKNOWN)
    magnetic_value = Column(Float, nullable=True)  # HMC5883L optional magnetic vector (nullable)
    vision_score = Column(Float, nullable=True)    # Simulated / Manual vision score 0-100 (nullable)
    motor_speed = Column(Integer, nullable=True)   # Motor speed percentage (0-100)
    motor_running = Column(Boolean, nullable=True) # Motor execution state (True/False)


    # Component Scores (0-100 or UNKNOWN/Null)
    vibration_score = Column(Float, nullable=True)
    temperature_score = Column(Float, nullable=True)
    magnetic_score = Column(Float, nullable=True)

    # Overall Fusion & Risk
    health_score = Column(Float, nullable=True)    # Weighted fusion score 0-100 or Null if UNKNOWN
    risk_level = Column(String(20), nullable=False) # LOW, MEDIUM, HIGH, UNKNOWN
    alert_type = Column(String(50), nullable=True)  # e.g., NONE, VIBRATION_HIGH, TEMP_WARNING, DATA_MISSING

    joint = relationship("Joint", back_populates="readings")


class AlertHistory(Base):
    __tablename__ = "alert_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    joint_id = Column(String(20), ForeignKey("joints.joint_id"), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    risk_level = Column(String(20), nullable=False) # MEDIUM, HIGH, UNKNOWN
    message = Column(Text, nullable=False)

    joint = relationship("Joint", back_populates="alerts")
