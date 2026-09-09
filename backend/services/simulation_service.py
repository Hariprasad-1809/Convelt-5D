"""
JointGuard Simulation Service

Simulates 5 conveyor joints (J01-J05) cycling through the inspection zone:
APPROACHING -> INSPECTING -> PASSED

Generates distinct sensor profiles per joint for interactive prototype demo:
- J01: Healthy baseline
- J02: Moderate vibration warning
- J03: High temperature & severe degradation critical alert
- J04: Sensor disconnection (vibration=None) -> Strict UNKNOWN state
- J05: Baseline healthy / dynamic noise

Supports start, stop, reset, step, and manual value injection override.
"""

import math
import random
import threading
import time
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.database.db import SessionLocal, engine, Base
from backend.database import crud
from backend.services.scoring import (
    vibration_score,
    temperature_score,
    magnetic_score,
    vision_score
)
from backend.services.health_score import calculate_health_score
from backend.services.alert_service import evaluate_and_trigger_alerts

# Define initial 5 joints configuration
DEFAULT_JOINTS = [
    {"joint_id": "J01", "name": "Conveyor Belt Joint #1 (Healthy)", "belt_pos": 2.5},
    {"joint_id": "J02", "name": "Conveyor Belt Joint #2 (Vib Warning)", "belt_pos": 7.5},
    {"joint_id": "J03", "name": "Conveyor Belt Joint #3 (High Temp Crit)", "belt_pos": 12.5},
    {"joint_id": "J04", "name": "Conveyor Belt Joint #4 (Sensor UNKNOWN)", "belt_pos": 17.5},
    {"joint_id": "J05", "name": "Conveyor Belt Joint #5 (Nominal)", "belt_pos": 22.5},
]

ZONE_STATES = ["APPROACHING", "INSPECTING", "PASSED"]

class SimulationEngine:
    def __init__(self):
        self.is_running: bool = False
        self.cycle_count: int = 0
        self.interval_seconds: float = 2.0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.manual_overrides: Dict[str, Dict[str, Any]] = {}
        self.joint_states: Dict[str, int] = {j["joint_id"]: idx for idx, j in enumerate(DEFAULT_JOINTS)}

        # Initialize DB tables on engine start
        Base.metadata.create_all(bind=engine)
        self._seed_initial_joints()

    def _seed_initial_joints(self):
        """Ensure J01 to J05 exist in DB."""
        db: Session = SessionLocal()
        try:
            for j in DEFAULT_JOINTS:
                crud.create_or_update_joint(
                    db=db,
                    joint_id=j["joint_id"],
                    name=j["name"],
                    belt_position=j["belt_pos"],
                    zone_state="APPROACHING",
                    status="ACTIVE"
                )
        finally:
            db.close()

    def set_manual_override(self, joint_id: str, override_data: Dict[str, Any]):
        """Store manual injection overrides for testing / future YOLO input."""
        with self._lock:
            if joint_id not in self.manual_overrides:
                self.manual_overrides[joint_id] = {}
            for k, v in override_data.items():
                if v is not None:
                    self.manual_overrides[joint_id][k] = v

    def clear_manual_override(self, joint_id: Optional[str] = None):
        with self._lock:
            if joint_id:
                self.manual_overrides.pop(joint_id, None)
            else:
                self.manual_overrides.clear()

    def step(self) -> Dict[str, Any]:
        """Executes one simulation step across all 5 joints."""
        db: Session = SessionLocal()
        try:
            with self._lock:
                self.cycle_count += 1
                results = []

                for j_info in DEFAULT_JOINTS:
                    jid = j_info["joint_id"]
                    
                    # Update inspection state machine: APPROACHING -> INSPECTING -> PASSED
                    current_zone_idx = (self.joint_states[jid] + self.cycle_count) % len(ZONE_STATES)
                    zone_state = ZONE_STATES[current_zone_idx]
                    
                    # Belt position advances smoothly
                    belt_pos = (j_info["belt_pos"] + self.cycle_count * 1.5) % 30.0
                    crud.create_or_update_joint(db, jid, j_info["name"], round(belt_pos, 2), zone_state)

                    # Generate simulated raw sensor readings based on joint profile
                    is_inspecting = (zone_state == "INSPECTING")
                    
                    from backend.services.serial_service import serial_service
                    if jid == "J01" and serial_service.connection_status == "CONNECTED":
                        latest = crud.get_latest_reading(db, "J01")
                        if latest and latest.temperature is not None:
                            raw_data = {
                                "vibration": latest.vibration,
                                "temperature": latest.temperature,
                                "hall_event": latest.hall_event,
                                "hall_event_expected": is_inspecting,
                                "magnetic_value": latest.magnetic_value,
                                "vision_score": latest.vision_score or 85.0
                            }
                        else:
                            raw_data = self._generate_joint_telemetry(jid, is_inspecting)
                    else:
                        raw_data = self._generate_joint_telemetry(jid, is_inspecting)

                    # Apply manual override if specified
                    if jid in self.manual_overrides:
                        for k, v in self.manual_overrides[jid].items():
                            raw_data[k] = v

                    # Compute individual component scores
                    v_sc = vision_score(raw_data["vision_score"])
                    m_sc = magnetic_score(raw_data["hall_event_expected"], raw_data["hall_event"])
                    vib_sc = vibration_score(raw_data["vibration"])
                    t_sc = temperature_score(raw_data["temperature"])

                    # Compute overall health fusion & risk classification
                    h_score, risk_lvl, suff = calculate_health_score(v_sc, m_sc, vib_sc, t_sc)

                    # Save reading to DB
                    db_reading = crud.create_sensor_reading(
                        db=db,
                        reading_data={
                            "joint_id": jid,
                            "belt_position": round(belt_pos, 2),
                            "vibration": raw_data["vibration"],
                            "temperature": raw_data["temperature"],
                            "hall_event": raw_data["hall_event"],
                            "magnetic_value": raw_data["magnetic_value"],
                            "vision_score": raw_data["vision_score"],
                            "vibration_score": vib_sc,
                            "temperature_score": t_sc,
                            "magnetic_score": m_sc,
                            "health_score": h_score,
                            "risk_level": risk_lvl,
                            "alert_type": "NONE" if risk_lvl == "LOW" else risk_lvl
                        }
                    )

                    # Trigger alert if risk changes or data missing
                    evaluate_and_trigger_alerts(
                        db=db,
                        joint_id=jid,
                        new_risk_level=risk_lvl,
                        health_score=h_score,
                        v_score=v_sc,
                        m_score=m_sc,
                        vib_score=vib_sc,
                        t_score=t_sc
                    )

                    results.append({
                        "joint_id": jid,
                        "zone_state": zone_state,
                        "health_score": h_score,
                        "risk_level": risk_lvl,
                        "raw_sensors": raw_data
                    })

                return {
                    "cycle": self.cycle_count,
                    "processed_joints": len(results),
                    "details": results
                }
        finally:
            db.close()

    def _generate_joint_telemetry(self, joint_id: str, is_inspecting: bool) -> Dict[str, Any]:
        """Generates realistic simulated sensor profiles per joint."""
        # Expected hall event is True when passing inspection zone magnet
        expected_hall = is_inspecting

        noise = random.uniform(-0.02, 0.02)

        if joint_id == "J01":
            # Baseline Healthy
            return {
                "vibration": round(0.50 + noise, 3),
                "temperature": round(36.5 + random.uniform(-0.5, 0.5), 1),
                "hall_event": expected_hall,
                "hall_event_expected": expected_hall,
                "magnetic_value": 42.5 if expected_hall else 12.0,
                "vision_score": round(96.0 + random.uniform(-2, 2), 1)
            }
        elif joint_id == "J02":
            # Moderate vibration deviation (+30% dev)
            return {
                "vibration": round(0.65 + noise, 3), # Baseline 0.50 -> 0.65 is +30% dev
                "temperature": round(44.0 + random.uniform(-0.5, 0.5), 1),
                "hall_event": expected_hall,
                "hall_event_expected": expected_hall,
                "magnetic_value": 41.0 if expected_hall else 11.5,
                "vision_score": round(72.0 + random.uniform(-3, 3), 1)
            }
        elif joint_id == "J03":
            # Critical High Temperature & Severe Vibration
            return {
                "vibration": round(0.78 + noise, 3), # +56% dev
                "temperature": round(63.5 + random.uniform(-1.0, 1.0), 1), # >60°C Critical
                "hall_event": expected_hall,
                "hall_event_expected": expected_hall,
                "magnetic_value": 38.0 if expected_hall else 10.0,
                "vision_score": round(35.0 + random.uniform(-4, 4), 1)
            }
        elif joint_id == "J04":
            # UNKNOWN Test Joint: Vibration sensor disconnected / missing (None)
            return {
                "vibration": None, # Missing vibration sensor -> UNKNOWN score
                "temperature": round(37.2 + random.uniform(-0.5, 0.5), 1),
                "hall_event": expected_hall,
                "hall_event_expected": expected_hall,
                "magnetic_value": None,
                "vision_score": round(85.0 + random.uniform(-2, 2), 1)
            }
        else: # J05
            # Nominal joint with mild fluctuations
            return {
                "vibration": round(0.52 + noise, 3),
                "temperature": round(38.0 + random.uniform(-0.5, 0.5), 1),
                "hall_event": expected_hall,
                "hall_event_expected": expected_hall,
                "magnetic_value": 43.0 if expected_hall else 12.0,
                "vision_score": round(90.0 + random.uniform(-2, 2), 1)
            }

    def start(self):
        with self._lock:
            if self.is_running:
                return
            self.is_running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self.is_running = False

    def reset(self):
        with self._lock:
            self.cycle_count = 0
            self.manual_overrides.clear()
            self.joint_states = {j["joint_id"]: idx for idx, j in enumerate(DEFAULT_JOINTS)}

    def _run_loop(self):
        while True:
            with self._lock:
                if not self.is_running:
                    break
            self.step()
            time.sleep(self.interval_seconds)

# Global singleton simulation engine
simulation_engine = SimulationEngine()
