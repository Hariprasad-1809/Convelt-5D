"""
Simulation Control REST API Router
POST /simulation/start — start background simulation loop
POST /simulation/stop  — pause simulation
POST /simulation/reset — reset cycles and manual overrides
POST /simulation/step  — trigger single simulation step manually
POST /simulation/inject — set manual sensor / vision override for a joint
"""

from fastapi import APIRouter, HTTPException, status
from backend.services.simulation_service import simulation_engine
from backend.models.schemas import SimulationStatusResponse, SimulationInjectRequest

router = APIRouter(prefix="/simulation", tags=["Simulation Control"])

@router.get("/status", response_model=SimulationStatusResponse)
def get_simulation_status():
    """Get current simulation status, cycle count, and active joint IDs."""
    return SimulationStatusResponse(
        is_running=simulation_engine.is_running,
        interval_seconds=simulation_engine.interval_seconds,
        current_cycle=simulation_engine.cycle_count,
        active_joints=["J01", "J02", "J03", "J04", "J05"]
    )

@router.post("/start", response_model=SimulationStatusResponse)
def start_simulation():
    """Start background multi-joint inspection simulation."""
    simulation_engine.start()
    return get_simulation_status()

@router.post("/stop", response_model=SimulationStatusResponse)
def stop_simulation():
    """Pause background simulation loop."""
    simulation_engine.stop()
    return get_simulation_status()

@router.post("/reset", response_model=SimulationStatusResponse)
def reset_simulation():
    """Reset simulation cycle counter and clear all manual value overrides."""
    simulation_engine.reset()
    return get_simulation_status()

@router.post("/step")
def step_simulation():
    """Execute a single simulation step across all joints immediately."""
    result = simulation_engine.step()
    return result

@router.post("/inject")
def inject_sensor_value(payload: SimulationInjectRequest):
    """
    Manually inject a sensor value or vision score for a specific joint.
    Used for testing, fault injection, and as the Phase 2 YOLO hook.
    """
    if payload.joint_id not in ["J01", "J02", "J03", "J04", "J05"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid joint_id '{payload.joint_id}'. Must be one of J01-J05."
        )

    override_dict = payload.model_dump(exclude_unset=True, exclude={"joint_id"})
    simulation_engine.set_manual_override(payload.joint_id, override_dict)
    
    # Execute a step to immediately apply override
    step_result = simulation_engine.step()

    return {
        "message": f"Successfully injected sensor override for joint {payload.joint_id}",
        "injected_values": override_dict,
        "step_result": step_result
    }
