"""
Automated Test Suite for JointGuard Vision Motor Interlock & Pipeline Decoupling

Tests:
1. SerialReaderService.send_command() command dispatch and error handling.
2. VisionService automated STOP command emission on confirmed DAMAGE detection.
3. One-shot interlock guard (prevents repeated serial stop spamming for the same joint).
4. Operator RESUME / STOP API endpoints (/api/v1/vision/resume, /api/v1/vision/stop).
5. Gating of JPEG stream encoding behind active viewer count.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.serial_service import serial_service
from backend.services.vision_service import vision_service


class TestMotorStopOnDamage(unittest.TestCase):

    def setUp(self):
        vision_service.reset_stop_guard()
        self.client = TestClient(app)

    def tearDown(self):
        vision_service.reset_stop_guard()

    def test_serial_send_command_success(self):
        """Verify send_command writes newline-terminated bytes and flushes."""
        mock_conn = MagicMock()
        with patch.object(serial_service, "_serial_conn", mock_conn):
            # Test STOP
            result = serial_service.send_command("STOP")
            self.assertTrue(result)
            mock_conn.write.assert_called_with(b"STOP\n")
            mock_conn.flush.assert_called_once()

            # Test RESUME
            mock_conn.reset_mock()
            result = serial_service.send_command("RESUME")
            self.assertTrue(result)
            mock_conn.write.assert_called_with(b"RESUME\n")
            mock_conn.flush.assert_called_once()

    def test_serial_send_command_disconnected(self):
        """Verify send_command returns False cleanly when port is disconnected."""
        with patch.object(serial_service, "_serial_conn", None):
            result = serial_service.send_command("STOP")
            self.assertFalse(result)

    def test_serial_send_command_exception_handled(self):
        """Verify exceptions during serial write are caught without raising."""
        mock_conn = MagicMock()
        mock_conn.write.side_effect = IOError("COM5 write timeout")
        with patch.object(serial_service, "_serial_conn", mock_conn):
            result = serial_service.send_command("STOP")
            self.assertFalse(result)

    def test_vision_damage_triggers_motor_stop_once(self):
        """
        Verify that when a joint is confirmed as DAMAGE:
        1. serial_service.send_command("STOP") is called.
        2. vision_service.motor_stop_triggered is set to True.
        3. Subsequent frames for the same joint do NOT spam STOP (one-shot guard).
        """
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_eval_res = {
            "joint_id": "J01",
            "current_label": "DAMAGE",
            "p_healthy": 0.05,
            "p_damage": 0.95,
            "confidence_margin": 0.90,
            "sharpness": 180.0,
            "valid_frame": True,
            "valid_frame_count": 5,
            "rejected_blurry_count": 0,
            "final_label": "DAMAGE",
            "final_confidence": 0.95,
            "avg_p_healthy": 0.05,
            "avg_p_damage": 0.95,
            "tracking_status": "VISUALIZED",
            "inspection_state": "CONFIRMED",
            "total_frames_seen": 15,
            "total_frames_lost": 0,
            "smoothed_bbox": (100, 100, 120, 120),
            "timestamp": "2026-09-10T12:00:00",
            "reason_for_final_change": "temporal-majority",
        }

        with patch.object(vision_service, "model_loaded", True), \
             patch.object(vision_service, "model", MagicMock()), \
             patch.object(vision_service.tracker, "process_frame", return_value=mock_eval_res), \
             patch.object(serial_service, "send_command", return_value=True) as mock_send_cmd:

            # Frame 1: Confirmed DAMAGE -> Should trigger STOP
            vision_service._process_frame(dummy_frame)
            self.assertEqual(mock_send_cmd.call_count, 1)
            mock_send_cmd.assert_called_with("STOP")
            self.assertTrue(vision_service.motor_stop_triggered)
            self.assertEqual(vision_service.last_stopped_joint_id, "J01")

            # Frame 2: Same joint J01 still DAMAGE -> Should NOT call STOP again
            vision_service._process_frame(dummy_frame)
            self.assertEqual(mock_send_cmd.call_count, 1, "STOP command should only be sent once per damaged joint")

            # Reset guard (e.g. operator resumed conveyor)
            vision_service.reset_stop_guard()
            self.assertFalse(vision_service.motor_stop_triggered)
            self.assertEqual(len(vision_service.stopped_damage_joint_ids), 0)

            # Frame 3: After reset, should be able to trigger again if needed
            vision_service._process_frame(dummy_frame)
            self.assertEqual(mock_send_cmd.call_count, 2)

    def test_vision_healthy_does_not_trigger_motor_stop(self):
        """Verify that HEALTHY detections never trigger a motor stop."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_eval_res = {
            "joint_id": "J01",
            "current_label": "HEALTHY",
            "p_healthy": 0.94,
            "p_damage": 0.06,
            "confidence_margin": 0.88,
            "sharpness": 210.0,
            "valid_frame": True,
            "valid_frame_count": 5,
            "rejected_blurry_count": 0,
            "final_label": "HEALTHY",
            "final_confidence": 0.94,
            "avg_p_healthy": 0.94,
            "avg_p_damage": 0.06,
            "tracking_status": "VISUALIZED",
            "inspection_state": "CONFIRMED",
            "total_frames_seen": 15,
            "total_frames_lost": 0,
            "smoothed_bbox": (100, 100, 120, 120),
            "timestamp": "2026-09-10T12:00:00",
            "reason_for_final_change": "temporal-majority",
        }

        with patch.object(vision_service, "model_loaded", True), \
             patch.object(vision_service, "model", MagicMock()), \
             patch.object(vision_service.tracker, "process_frame", return_value=mock_eval_res), \
             patch.object(serial_service, "send_command", return_value=True) as mock_send_cmd:

            vision_service._process_frame(dummy_frame)
            mock_send_cmd.assert_not_called()
            self.assertFalse(vision_service.motor_stop_triggered)

    def test_api_vision_resume_and_stop_endpoints(self):
        """Verify POST /api/v1/vision/resume and /api/v1/vision/stop endpoints."""
        with patch.object(serial_service, "send_command", return_value=True) as mock_cmd:
            # Test POST /api/v1/vision/resume
            res_resume = self.client.post("/api/v1/vision/resume")
            self.assertEqual(res_resume.status_code, 200)
            data = res_resume.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["command"], "RESUME")
            self.assertFalse(data["motor_stop_triggered"])
            mock_cmd.assert_called_with("RESUME")

            # Test POST /api/v1/vision/stop
            mock_cmd.reset_mock()
            res_stop = self.client.post("/api/v1/vision/stop")
            self.assertEqual(res_stop.status_code, 200)
            data_stop = res_stop.json()
            self.assertEqual(data_stop["status"], "ok")
            self.assertEqual(data_stop["command"], "STOP")
            mock_cmd.assert_called_with("STOP")

    def test_viewer_count_and_stream_encoding_gate(self):
        """Verify viewer count tracking and gating of JPEG encoding."""
        # Viewer count increments and decrements cleanly
        self.assertEqual(vision_service.stream_viewer_count, 0)
        vision_service.increment_stream_viewers()
        self.assertEqual(vision_service.stream_viewer_count, 1)
        vision_service.decrement_stream_viewers()
        self.assertEqual(vision_service.stream_viewer_count, 0)

        # When 0 viewers, _process_frame should not update latest_frame_jpeg
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vision_service.latest_frame_jpeg = b"sentinel"

        mock_eval = {
            "joint_id": "J01",
            "current_label": "HEALTHY",
            "p_healthy": 0.9,
            "p_damage": 0.1,
            "confidence_margin": 0.8,
            "sharpness": 150.0,
            "valid_frame": True,
            "valid_frame_count": 3,
            "rejected_blurry_count": 0,
            "final_label": "HEALTHY",
            "final_confidence": 0.9,
            "avg_p_healthy": 0.9,
            "avg_p_damage": 0.1,
            "tracking_status": "VISUALIZED",
            "inspection_state": "INSPECTING",
            "total_frames_seen": 10,
            "total_frames_lost": 0,
            "smoothed_bbox": None,
            "timestamp": "2026-09-10T12:00:00",
            "reason_for_final_change": "test",
        }

        with patch.object(vision_service, "model_loaded", True), \
             patch.object(vision_service, "model", MagicMock()), \
             patch.object(vision_service.tracker, "process_frame", return_value=mock_eval):
            vision_service._process_frame(dummy_frame)
            # JPEG bytes should remain untouched because viewers == 0
            self.assertEqual(vision_service.latest_frame_jpeg, b"sentinel")

            # Now connect a viewer
            vision_service.increment_stream_viewers()
            vision_service._process_frame(dummy_frame)
            # Now JPEG bytes should have been newly encoded
            self.assertNotEqual(vision_service.latest_frame_jpeg, b"sentinel")
            vision_service.decrement_stream_viewers()


if __name__ == "__main__":
    unittest.main()
