"""
Unit tests for JointGuard Individual Sensor Scoring & Health Fusion
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.services.scoring import (
    vibration_score,
    temperature_score,
    magnetic_score,
    vision_score
)
from backend.services.health_score import calculate_health_score


def test_vibration_score():
    # Baseline 0.50 -> 100
    assert vibration_score(0.50, baseline=0.50) == 100.0
    # +20% deviation (0.60) -> 70.0
    assert vibration_score(0.60, baseline=0.50) == 70.0
    # +50% deviation (0.75) -> 40.0
    assert vibration_score(0.75, baseline=0.50) == 40.0
    # Missing value -> None (UNKNOWN)
    assert vibration_score(None) is None

def test_temperature_score():
    # Normal below 45°C
    assert temperature_score(35.0) == 100.0
    assert temperature_score(45.0) == 100.0
    # Warning 52.5°C -> mid-way (70.0)
    assert temperature_score(52.5) == 70.0
    # 60°C -> 40.0
    assert temperature_score(60.0) == 40.0
    # Missing value -> None
    assert temperature_score(None) is None

def test_magnetic_score():
    # Match -> 100
    assert magnetic_score(True, True) == 100.0
    assert magnetic_score(False, False) == 100.0
    # Mismatch -> 20
    assert magnetic_score(True, False) == 20.0
    # Missing value -> None
    assert magnetic_score(None, True) is None

def test_vision_score():
    assert vision_score(85.5) == 85.5
    assert vision_score(None) is None

def test_health_fusion():
    # Healthy case: 100 on all -> 100.0, LOW
    score, risk, suff = calculate_health_score(100.0, 100.0, 100.0, 100.0)
    assert score == 100.0
    assert risk == "LOW"
    assert suff is True

    # Degraded case: vision=50, magnetic=100, vib=70, temp=100
    # 0.40*50 + 0.30*100 + 0.20*70 + 0.10*100 = 20 + 30 + 14 + 10 = 74.0 (LOW)
    score, risk, suff = calculate_health_score(50.0, 100.0, 70.0, 100.0)
    assert score == 74.0
    assert risk == "LOW"

    # Degraded case: vision=30, magnetic=20, vib=40, temp=40
    # 0.40*30 + 0.30*20 + 0.20*40 + 0.10*40 = 12 + 6 + 8 + 4 = 30.0 (HIGH)
    score, risk, suff = calculate_health_score(30.0, 20.0, 40.0, 40.0)
    assert score == 30.0
    assert risk == "HIGH"

    # UNKNOWN handling: ANY missing component makes overall score None, risk UNKNOWN
    score, risk, suff = calculate_health_score(100.0, None, 100.0, 100.0)
    assert score is None
    assert risk == "UNKNOWN"
    assert suff is False

if __name__ == "__main__":
    print("Running quick test_scoring.py...")
    test_vibration_score()
    test_temperature_score()
    test_magnetic_score()
    test_vision_score()
    test_health_fusion()
    print("All individual scoring and health fusion tests passed successfully!")

