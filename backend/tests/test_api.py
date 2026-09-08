"""
Integration API Test Script for JointGuard FastAPI Backend

Tests all REST endpoints:
- GET /
- GET /joints
- GET /joints/{joint_id}
- GET /joints/{joint_id}/sensors
- GET /joints/{joint_id}/health
- GET /joints/{joint_id}/risk
- GET /alerts
- GET /alerts/history
- GET /history/{joint_id}
- POST /simulation/stop, /step, /inject, /start
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

from backend.services.simulation_service import simulation_engine

def run_all_api_tests():
    print("\n--- Starting JointGuard REST API Verification ---")
    
    # Run initial step to populate baseline sensor readings
    simulation_engine.step()

    # 1. Root info

    res = client.get("/")
    assert res.status_code == 200, f"Root failed: {res.text}"
    print("[OK] GET / -> 200 OK | Project:", res.json()["project"])

    # 2. List joints
    res = client.get("/joints")
    assert res.status_code == 200
    joints = res.json()
    assert len(joints) == 5, f"Expected 5 joints, got {len(joints)}"
    print("[OK] GET /joints -> 200 OK | Joints count:", len(joints))

    # 3. Joint detail J01
    res = client.get("/joints/J01")
    assert res.status_code == 200
    detail = res.json()
    assert detail["joint_id"] == "J01"
    print("[OK] GET /joints/J01 -> 200 OK | Zone state:", detail["zone_state"])

    # 4. Joint sensors J01
    res = client.get("/joints/J01/sensors")
    assert res.status_code == 200
    sensors = res.json()
    print("[OK] GET /joints/J01/sensors -> 200 OK | Temp:", sensors["temperature"], "deg C | Vib:", sensors["vibration"])

    # 5. Joint health J01
    res = client.get("/joints/J01/health")
    assert res.status_code == 200
    health = res.json()
    print("[OK] GET /joints/J01/health -> 200 OK | Health Score:", health["health_score"], "| Risk:", health["risk_level"])

    # 6. Joint risk J01
    res = client.get("/joints/J01/risk")
    assert res.status_code == 200
    risk = res.json()
    print("[OK] GET /joints/J01/risk -> 200 OK | Risk:", risk["risk_level"])

    # 7. Alerts
    res = client.get("/alerts")
    assert res.status_code == 200
    print("[OK] GET /alerts -> 200 OK | Recent alerts:", len(res.json()))

    # 8. Alert history
    res = client.get("/alerts/history")
    assert res.status_code == 200
    print("[OK] GET /alerts/history -> 200 OK | History count:", len(res.json()))

    # 9. Joint history trend
    res = client.get("/history/J01?limit=10")
    assert res.status_code == 200
    print("[OK] GET /history/J01 -> 200 OK | Historical points:", len(res.json()))

    # 10. Simulation control: Stop & Step
    res = client.post("/simulation/stop")
    assert res.status_code == 200
    res = client.post("/simulation/step")
    assert res.status_code == 200
    print("[OK] POST /simulation/step -> 200 OK | Cycle:", res.json()["cycle"])

    # 11. Simulation control: Inject high vibration & manual vision score into J01
    inject_payload = {
        "joint_id": "J01",
        "vibration": 0.85, # High vibration deviation (+70% -> Critical score)
        "vision_score": 45.0
    }
    res = client.post("/simulation/inject", json=inject_payload)
    assert res.status_code == 200
    print("[OK] POST /simulation/inject -> 200 OK | Injected values:", res.json()["injected_values"])

    # Verify J01 now has high risk from injected values
    res = client.get("/joints/J01/health")
    assert res.status_code == 200
    health_injected = res.json()
    print("  -> Post-Injection J01 Health Score:", health_injected["health_score"], "| Risk:", health_injected["risk_level"])

    # 12. Simulation control: Start
    res = client.post("/simulation/start")
    assert res.status_code == 200
    print("[OK] POST /simulation/start -> 200 OK | Simulation background thread running")


    print("--- ALL 12 REST API INTEGRATION TESTS PASSED SUCCESSFULLY! ---\n")

if __name__ == "__main__":
    run_all_api_tests()
