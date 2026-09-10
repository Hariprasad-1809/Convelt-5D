"""
JointGuard Vision API Endpoints (Phase 1 + YOLO Integration)

Routes for camera status, latest detection, and live MJPEG stream rendering.
"""

import asyncio
from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import StreamingResponse

from backend.services.vision_service import vision_service
from backend.config import settings

router = APIRouter(prefix="/api/v1/vision", tags=["Vision System"])


@router.get("/status")
def get_vision_status():
    """Returns camera connection status and YOLO model load state."""
    latest = vision_service.get_latest_result()
    return {
        "camera_connected": vision_service.camera_status == "CONNECTED",
        "camera_status": vision_service.camera_status,
        "model_loaded": vision_service.model_loaded,
        "active_joint": vision_service.active_joint_id,
        "camera_index": vision_service.camera_index,
        "conf_threshold": settings.VISION_CONF_THRESH,
        "last_detection": latest,
    }


@router.get("/latest")
def get_latest_vision():
    """Returns latest YOLO vision detection result."""
    return vision_service.get_latest_result()


async def _generate_mjpeg_stream():
    """Generator function that yields JPEG frames as an MJPEG stream."""
    vision_service.increment_stream_viewers()
    try:
        while True:
            frame_bytes = vision_service.get_latest_jpeg()
            if frame_bytes is None:
                await asyncio.sleep(0.1)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            await asyncio.sleep(0.03)  # ~30 FPS stream rate
    except (asyncio.CancelledError, Exception):
        # Client disconnected cleanly
        pass
    finally:
        vision_service.decrement_stream_viewers()


@router.get("/stream")
async def get_vision_stream():
    """
    Live MJPEG camera video stream.
    Directly renderable in React frontend via <img src="http://127.0.0.1:8000/api/v1/vision/stream" />
    """
    return StreamingResponse(
        _generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/resume")
def resume_motor():
    """
    Sends RESUME command to Arduino conveyor motor and resets the damage auto-stop interlock guard.
    """
    from backend.services.serial_service import serial_service
    sent = serial_service.send_command("RESUME")
    vision_service.reset_stop_guard()
    return {
        "status": "ok" if sent else "error",
        "command": "RESUME",
        "serial_sent": sent,
        "motor_stop_triggered": False,
        "message": "Conveyor motor resume signal sent to Arduino" if sent else "Failed to send RESUME signal over serial"
    }


@router.post("/stop")
def stop_motor():
    """
    Manual emergency stop for Arduino conveyor motor.
    """
    from backend.services.serial_service import serial_service
    sent = serial_service.send_command("STOP")
    return {
        "status": "ok" if sent else "error",
        "command": "STOP",
        "serial_sent": sent,
        "message": "Conveyor motor stop signal sent to Arduino" if sent else "Failed to send STOP signal over serial"
    }


