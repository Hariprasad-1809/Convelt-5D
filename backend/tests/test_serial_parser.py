"""
Unit Tests for JointGuard Arduino UNO Serial Parser & Telemetry Pipeline
"""

import pytest
from datetime import datetime, timezone
from backend.services.serial_service import SerialParser, SerialReaderService
from backend.models.schemas import ArduinoTelemetryPayload, SerialStatusPayload

def test_parse_temperature_line_normal():
    frame = {}
    line = "Temperature : 28.50 C [NORMAL]"
    result = SerialParser.parse_line(line, frame)
    assert result["temperature"] == 28.50
    assert result["temperature_status"] == "NORMAL"

def test_parse_temperature_line_high():
    frame = {}
    line = "Temperature : 52.10 C [HIGH]"
    result = SerialParser.parse_line(line, frame)
    assert result["temperature"] == 52.10
    assert result["temperature_status"] == "HIGH"

def test_parse_temperature_line_error():
    frame = {}
    line = "Temperature : SENSOR ERROR"
    result = SerialParser.parse_line(line, frame)
    assert result["temperature"] is None
    assert result["temperature_status"] == "ERROR"

def test_parse_vibration_line_normal():
    frame = {}
    line = "Vibration   : 0.15 m/s2 [NORMAL]"
    result = SerialParser.parse_line(line, frame)
    assert result["vibration"] == 0.15
    assert result["vibration_status"] == "NORMAL"

def test_parse_vibration_line_error():
    frame = {}
    line = "Vibration   : SENSOR ERROR"
    result = SerialParser.parse_line(line, frame)
    assert result["vibration"] is None
    assert result["vibration_status"] == "ERROR"

def test_parse_hall_sensor_magnet_detected():
    frame = {}
    line = "Hall Sensor : MAGNET DETECTED"
    result = SerialParser.parse_line(line, frame)
    assert result["hall_detected"] is True

def test_parse_hall_sensor_no_magnet():
    frame = {}
    line = "Hall Sensor : NO MAGNET"
    result = SerialParser.parse_line(line, frame)
    assert result["hall_detected"] is False

def test_parse_motor_speed():
    frame = {}
    line = "Motor Speed : 45%"
    result = SerialParser.parse_line(line, frame)
    assert result["motor_speed"] == 45

def test_parse_motor_running():
    frame = {}
    line = "Motor       : RUNNING"
    result = SerialParser.parse_line(line, frame)
    assert result["motor_running"] is True

def test_parse_motor_stopped():
    frame = {}
    line = "Motor       : STOPPED"
    result = SerialParser.parse_line(line, frame)
    assert result["motor_running"] is False

def test_complete_telemetry_block_accumulation():
    raw_lines = [
        "================================",
        "Temperature : 31.20 C [NORMAL]",
        "Vibration   : 0.42 m/s2 [NORMAL]",
        "Hall Sensor : NO MAGNET",
        "Motor Speed : 30%",
        "Motor       : RUNNING",
        "================================"
    ]
    frame = {}
    for line in raw_lines:
        if not line.startswith("==="):
            frame = SerialParser.parse_line(line, frame)

    assert frame["temperature"] == 31.20
    assert frame["vibration"] == 0.42
    assert frame["hall_detected"] is False
    assert frame["motor_speed"] == 30
    assert frame["motor_running"] is True

def test_arduino_telemetry_schema_validation():
    payload = ArduinoTelemetryPayload(
        timestamp=datetime.now(timezone.utc),
        temperature=28.50,
        vibration=0.15,
        hall_detected=False,
        motor_speed=30,
        motor_running=True,
        health_score=92.0,
        risk_level="LOW",
        connection_status="CONNECTED"
    )
    assert payload.device_id == "ARDUINO_UNO_01"
    assert payload.temperature == 28.50
    assert payload.motor_running is True

def test_serial_status_payload_validation():
    payload = SerialStatusPayload(
        connection_status="DISCONNECTED",
        message="Port COM4 unreachable",
        timestamp=datetime.now(timezone.utc)
    )
    assert payload.port == "COM4"
    assert payload.baud == 9600
    assert payload.connection_status == "DISCONNECTED"
