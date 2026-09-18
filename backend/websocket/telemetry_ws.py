"""
JointGuard WebSocket Telemetry Broadcast Manager

Manages live WebSocket client connections and broadcasts real-time Arduino UNO 
sensor telemetry and hardware connection status events to React clients.
"""

import asyncio
import json
import logging
from typing import List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("jointguard.websocket")
router = APIRouter(prefix="/api/v1/ws", tags=["websocket"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"[WS] Client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"[WS] Client disconnected. Remaining: {len(self.active_connections)}")


    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        
        # Serialize datetime objects to ISO format string if needed
        json_data = json.dumps(message, default=str)
        disconnected_clients = []

        for connection in self.active_connections:
            try:
                await connection.send_text(json_data)
            except Exception as e:
                logger.warning(f"Error sending message to WebSocket client: {e}")
                disconnected_clients.append(connection)

        for client in disconnected_clients:
            self.disconnect(client)

manager = ConnectionManager()
telemetry_ws = manager  # Global export alias

def broadcast_telemetry_sync(loop: asyncio.AbstractEventLoop, payload: dict):
    """Safely schedule a broadcast onto the main asyncio loop from background threads."""
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), loop)

@router.websocket("/telemetry")
@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Immediately push initial hardware telemetry or connection status to the newly connected client
        try:
            from backend.services.serial_service import serial_service
            if serial_service.latest_telemetry:
                await websocket.send_text(json.dumps(serial_service.latest_telemetry, default=str))
            else:
                from datetime import datetime, timezone
                await websocket.send_text(json.dumps({
                    "type": "connection_status",
                    "device_id": serial_service.device_id,
                    "port": serial_service.port,
                    "baud": serial_service.baud,
                    "connection_status": serial_service.connection_status,
                    "serial_status": serial_service.connection_status,
                    "websocket_status": "CONNECTED",
                    "message": f"Connected to gateway on {serial_service.port}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, default=str))
        except Exception as init_err:
            logger.debug(f"[WS] Non-fatal error sending initial telemetry frame: {init_err}")

        while True:
            # Keep connection alive & handle incoming client pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        manager.disconnect(websocket)

