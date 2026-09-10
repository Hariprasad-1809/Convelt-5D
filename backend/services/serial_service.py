"""
JointGuard Arduino UNO Serial Reader & Parser Service

Runs a non-blocking background thread reading live sensor streams from USB Serial (COM3 @ 9600 baud).
Parses raw sensor text, computes health scores, stores readings in SQLite DB, evaluates alerts,
and broadcasts live telemetry to WebSocket clients.
"""

import asyncio
import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False

from backend.config import settings
from backend.database.db import SessionLocal
from backend.models.orm_models import SensorReading, Joint
from backend.services.scoring import (
    vibration_score,
    temperature_score,
    magnetic_score,
    vision_score,
)
from backend.services.health_score import calculate_health_score, classify_risk_level
from backend.services.alert_service import evaluate_and_trigger_alerts
from backend.websocket.telemetry_ws import broadcast_telemetry_sync

logger = logging.getLogger("jointguard.serial")

def find_arduino_port(target_port: str) -> str:
    """Find active COM port matching target or auto-detect Arduino board, ignoring Bluetooth serial links."""
    if not SERIAL_AVAILABLE or serial is None:
        return target_port

    try:
        ports = list(serial.tools.list_ports.comports())
    except Exception as e:
        logger.warning(f"[SERIAL] Error enumerating ports: {e}")
        return target_port

    if not ports:
        return target_port

    # 1. Top priority: Auto-detect genuine USB Arduino / CH340 / FTDI hardware (explicitly non-Bluetooth)
    for p in ports:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if "bluetooth" in desc or "bthenum" in hwid:
            continue
        if any(k in desc or k in hwid for k in ("arduino", "ch340", "usb serial", "usb-serial", "ftdi", "vid:pid=2341")):
            logger.info(f"[SERIAL] Auto-detected Arduino on {p.device} ({p.description})")
            return p.device

    # 2. Check if configured target_port is physically connected and not Bluetooth
    for p in ports:
        if p.device.upper() == target_port.upper():
            desc = (p.description or "").lower()
            if "bluetooth" not in desc:
                return p.device

    # 3. Fallback to any non-Bluetooth port
    for p in ports:
        desc = (p.description or "").lower()
        if "bluetooth" not in desc:
            logger.info(f"[SERIAL] Falling back to non-bluetooth port {p.device} ({p.description})")
            return p.device

    # 4. Fallback to target port
    return target_port


class SerialParser:
    """Tolerant parser for Arduino UNO text output formatting."""

    @staticmethod
    def parse_line(line: str, current_frame: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = line.strip()
        if not cleaned:
            return current_frame

        # 1. Temperature Parsing
        # Matches: "Temperature : 28.50 C [NORMAL]" or "Temperature : SENSOR ERROR"
        if "temperature" in cleaned.lower():
            if "sensor error" in cleaned.lower():
                current_frame["temperature"] = None
                current_frame["temperature_status"] = "ERROR"
            else:
                match = re.search(r"(\d+\.?\d*)", cleaned)
                if match:
                    current_frame["temperature"] = float(match.group(1))
                    upper_line = cleaned.upper()
                    current_frame["temperature_status"] = "HIGH" if ("HIGH" in upper_line or "DANGER" in upper_line) else "NORMAL"

        # 2. Vibration Parsing
        # Matches: "Vibration : 0.15 m/s2 [NORMAL]" or "Vibration : SENSOR ERROR"
        elif "vibration" in cleaned.lower():
            if "sensor error" in cleaned.lower():
                current_frame["vibration"] = None
                current_frame["vibration_status"] = "ERROR"
            else:
                match = re.search(r"(\d+\.?\d*)", cleaned)
                if match:
                    current_frame["vibration"] = float(match.group(1))
                    upper_line = cleaned.upper()
                    if "HIGH" in upper_line or "DANGER" in upper_line:
                        current_frame["vibration_status"] = "HIGH"
                    elif "WARNING" in upper_line:
                        current_frame["vibration_status"] = "WARNING"
                    else:
                        current_frame["vibration_status"] = "NORMAL"

        # 3. Hall Sensor Parsing
        # Matches: "Hall Sensor : MAGNET DETECTED" or "Hall Sensor : NO MAGNET"
        elif "hall sensor" in cleaned.lower() or "hall" in cleaned.lower():
            if "magnet detected" in cleaned.lower():
                current_frame["hall_detected"] = True
            elif "no magnet" in cleaned.lower():
                current_frame["hall_detected"] = False

        # 4. Motor Speed Parsing
        # Matches: "Motor Speed : 30%"
        elif "motor speed" in cleaned.lower():
            match = re.search(r"(\d+)", cleaned)
            if match:
                current_frame["motor_speed"] = int(match.group(1))

        # 5. Motor Running Status Parsing
        # Matches: "Motor : RUNNING" or "Motor : STOPPED"
        elif "motor" in cleaned.lower() and "speed" not in cleaned.lower():
            if "running" in cleaned.lower():
                current_frame["motor_running"] = True
            elif "stopped" in cleaned.lower():
                current_frame["motor_running"] = False

        return current_frame


class SerialReaderService:
    def __init__(self):
        self.port = settings.SERIAL_PORT
        self.baud = settings.SERIAL_BAUD
        self.device_id = settings.DEVICE_ID
        self.joint_id = settings.DEFAULT_JOINT_ID

        self.connection_status = "DISCONNECTED" # CONNECTED, DISCONNECTED, RECONNECTING, ERROR
        self.last_updated: Optional[datetime] = None
        self.latest_telemetry: Dict[str, Any] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._serial_conn: Optional[serial.Serial] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the background serial reader thread."""
        if self._running:
            return
        
        self._async_loop = loop
        self._running = True

        if not SERIAL_AVAILABLE:
            logger.warning("[SERIAL] pyserial package not installed; serial service is disabled.")
            self.connection_status = "DISCONNECTED"
            self._notify_status("DISCONNECTED", "pyserial not installed")
            return

        logger.info(f"[SERIAL] Configured port: {self.port}")
        logger.info(f"[SERIAL] Baud rate: {self.baud}")
        logger.info(f"[SERIAL] Attempting connection to Arduino UNO...")
        self._thread = threading.Thread(target=self._run_loop, name="SerialReaderThread", daemon=True)
        self._thread.start()

    def stop(self):
        """Stop serial thread and close connection."""
        self._running = False
        if self._serial_conn and self._serial_conn.is_open:
            try:
                self._serial_conn.close()
            except Exception as e:
                logger.warning(f"[SERIAL] Error closing port: {e}")
        self.connection_status = "DISCONNECTED"

    def _run_loop(self):
        """Main thread loop handling connection, reading, and reconnection logic."""
        while self._running:
            try:
                available = list(serial.tools.list_ports.comports())
                if available:
                    logger.info(f"[SERIAL] Available ports:")
                    for p in available:
                        logger.info(f"  {p.device} - {p.description}")

                active_port = find_arduino_port(self.port)
                logger.info(f"[SERIAL] Using {active_port}")
                
                self._serial_conn = serial.Serial()
                self._serial_conn.port = active_port
                self._serial_conn.baudrate = self.baud
                self._serial_conn.timeout = 1.5
                self._serial_conn.dtr = False
                self._serial_conn.open()
                self.port = active_port
                time.sleep(1) # Give Arduino serial buffer time to stabilize
                
                self.connection_status = "CONNECTED"
                logger.info(f"[SERIAL] Arduino UNO connected on {self.port}")
                self._notify_status("CONNECTED", f"Serial port {self.port} opened successfully")

                current_frame: Dict[str, Any] = {}
                last_pkt_time = time.time()

                while self._running and self._serial_conn.is_open:
                    try:
                        raw_line = self._serial_conn.readline()
                        if not raw_line:
                            # Check 60s timeout if no telemetry received
                            if time.time() - last_pkt_time > 60.0:
                                logger.warning(f"[SERIAL] Hardware telemetry timeout (>60s without serial frame)")
                                logger.warning(f"[SERIAL] Ensure PlatformIO / Arduino Serial Monitor is CLOSED and USB cable is secure.")
                                self.connection_status = "DISCONNECTED"
                                self._notify_status("DISCONNECTED", "Hardware timeout: No serial data received for 60s")
                                break
                            continue

                        last_pkt_time = time.time()
                        line = raw_line.decode('utf-8', errors='ignore').strip()
                        if not line:
                            continue

                        if self.connection_status != "CONNECTED":
                            self.connection_status = "CONNECTED"
                            self._notify_status("CONNECTED", f"Serial telemetry restored on {self.port}")

                        # Check for banner line indicating frame boundary
                        if line.startswith("==="):
                            if current_frame and ("temperature" in current_frame or "vibration" in current_frame):
                                self._process_completed_frame(current_frame)
                                current_frame = {}
                        else:
                            current_frame = SerialParser.parse_line(line, current_frame)

                    except (serial.SerialException, OSError) as read_err:
                        logger.warning(f"[SERIAL] Read error / disconnection: {read_err}")
                        break

            except (serial.SerialException, OSError) as conn_err:
                logger.warning(f"[SERIAL] FAILED TO OPEN {self.port}: {conn_err}")
                logger.warning(f"[SERIAL] {self.port} may be busy/in use by another application (e.g. PlatformIO / VS Code Serial Monitor).")
                logger.warning(f"[SERIAL] Close PlatformIO Serial Monitor and retry. Retrying in {settings.SERIAL_RECONNECT_INTERVAL_SECONDS}s...")
                self.connection_status = "DISCONNECTED"
                self._notify_status("DISCONNECTED", f"Unable to open {self.port} (Port busy or disconnected)")
            
            finally:
                if self._serial_conn and self._serial_conn.is_open:
                    try:
                        self._serial_conn.close()
                    except Exception:
                        pass

            # Wait before attempting reconnect
            if self._running:
                time.sleep(settings.SERIAL_RECONNECT_INTERVAL_SECONDS)

    def _process_completed_frame(self, frame: Dict[str, Any]):
        """Process completed telemetry frame: compute score, update DB, trigger alerts, and broadcast."""
        now = datetime.now(timezone.utc)
        self.last_updated = now

        temp = frame.get("temperature")
        vib = frame.get("vibration")
        hall = frame.get("hall_detected", False)
        motor_speed = frame.get("motor_speed", 0)
        motor_running = frame.get("motor_running", False)

        # Log individual parsed values
        if temp is not None:
            logger.info(f"[SERIAL] Temperature = {temp:.2f} C")
        else:
            logger.info(f"[SERIAL] Temperature = SENSOR ERROR")

        if vib is not None:
            logger.info(f"[SERIAL] Vibration = {vib:.2f} m/s2")
        else:
            logger.info(f"[SERIAL] Vibration = SENSOR ERROR")

        logger.info(f"[SERIAL] Hall = {'MAGNET DETECTED' if hall else 'NO MAGNET'}")
        logger.info(f"[SERIAL] Motor Speed = {motor_speed}%")
        logger.info(f"[SERIAL] Motor = {'RUNNING' if motor_running else 'STOPPED'}")

        # 1. Calculate Component Scores (0-100) and Health Score via fusion engine
        v_score = vision_score(85.0)
        m_score = 100.0 if not hall else 80.0 # Magnetic baseline score
        vib_sc = vibration_score(vib)
        temp_sc = temperature_score(temp)

        health_val, risk_level, _ = calculate_health_score(
            v_score=v_score,
            m_score=m_score,
            vib_score=vib_sc,
            t_score=temp_sc
        )
        health_score = health_val if health_val is not None else 85.0
        if risk_level == "UNKNOWN":
            risk_level = classify_risk_level(health_score)

        # 2. Store reading in Database & update target joint record
        db = SessionLocal()
        try:
            # Ensure target joint exists
            joint = db.query(Joint).filter(Joint.joint_id == self.joint_id).first()
            if not joint:
                joint = Joint(joint_id=self.joint_id, name=f"Conveyor Joint {self.joint_id}", belt_position=0.0)
                db.add(joint)
                db.commit()

            joint.health_score = health_score
            joint.risk_level = risk_level
            joint.temperature = temp
            joint.vibration = vib
            joint.last_updated = now.isoformat()

            reading = SensorReading(
                device_id=self.device_id,
                joint_id=self.joint_id,
                belt_position=joint.belt_position or 0.0,
                vibration=vib,
                temperature=temp,
                hall_event=hall,
                vision_score=85.0,
                vibration_score=vib_sc,
                temperature_score=temp_sc,
                magnetic_score=m_score,
                health_score=health_score,
                risk_level=risk_level,
                alert_type="NONE" if risk_level == "LOW" else f"{risk_level}_RISK",
                motor_speed=motor_speed,
                motor_running=motor_running
            )
            db.add(reading)

            # 3. Evaluate Alerts
            evaluate_and_trigger_alerts(db, self.joint_id, risk_level, health_score, v_score, m_score, vib_sc, temp_sc)
            db.commit()

        except Exception as db_err:
            db.rollback()
            logger.error(f"[SERIAL DB] Error storing telemetry: {db_err}")
        finally:
            db.close()

        # 4. Construct Telemetry Packet
        payload = {
            "type": "sensor_telemetry",
            "data_source": "LIVE_HARDWARE",
            "device_id": self.device_id,
            "joint_id": self.joint_id,
            "port": self.port,
            "baud": self.baud,
            "timestamp": now.isoformat(),
            "temperature": temp,
            "temperature_status": frame.get("temperature_status", "NORMAL"),
            "vibration": vib,
            "vibration_status": frame.get("vibration_status", "NORMAL"),
            "hall_detected": hall,
            "motor_speed": motor_speed,
            "motor_running": motor_running,
            "health_score": health_score,
            "risk_level": risk_level,
            "connection_status": "CONNECTED",
            "serial_status": "CONNECTED",
            "websocket_status": "CONNECTED",
        }


        self.latest_telemetry = payload
        logger.info(f"[SERIAL] Telemetry broadcast")

        # 5. Broadcast to WebSockets
        if self._async_loop:
            broadcast_telemetry_sync(self._async_loop, payload)

    def _notify_status(self, status: str, message: str):

        """Broadcast connection status changes to WebSocket clients."""
        payload = {
            "type": "connection_status",
            "device_id": self.device_id,
            "port": self.port,
            "baud": self.baud,
            "connection_status": status,
            "serial_status": status,
            "websocket_status": "CONNECTED",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if self._async_loop:
            broadcast_telemetry_sync(self._async_loop, payload)

# Global Serial Service Singleton
serial_service = SerialReaderService()
