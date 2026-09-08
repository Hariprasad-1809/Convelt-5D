"""
JointGuard Individual Sensor Scoring Functions

Each function calculates a 0-100 score or returns None if sensor data is UNKNOWN/missing.
Scoring is pure and independent of API / ORM layers for clean unit-testing.

DISCLAIMER:
- Health scoring is RULE-BASED WEIGHTED FUSION (not machine learning).
- A3144 Hall sensor is a MAGNETIC-EVENT DEMONSTRATION sensor (not industrial EM/MFL).
- Temperature and vibration thresholds are prototype demo values.
"""

from typing import Optional
from backend.config import settings

def vibration_score(vibration_val: Optional[float], baseline: Optional[float] = None) -> Optional[float]:
    """
    Calculates vibration score based on RMS deviation from baseline.
    - 100: Near baseline
    - ~70: At warning threshold (+20% deviation)
    - <=40: At critical threshold (+50% deviation)
    - Returns None if vibration_val is None.
    """
    if vibration_val is None:
        return None

    b = baseline if baseline is not None else settings.VIBRATION_BASELINE_RMS
    if b <= 0:
        b = 0.50

    # Calculate absolute relative deviation
    dev_pct = abs(vibration_val - b) / b

    warn_thresh = settings.VIBRATION_WARN_DEVIATION_PCT  # 0.20 (+20%)
    crit_thresh = settings.VIBRATION_CRIT_DEVIATION_PCT  # 0.50 (+50%)

    if dev_pct <= warn_thresh:
        # Score linearly between 100 and 70
        score = 100.0 - (dev_pct / warn_thresh) * 30.0
    elif dev_pct <= crit_thresh:
        # Score linearly between 70 and 40
        ratio = (dev_pct - warn_thresh) / (crit_thresh - warn_thresh)
        score = 70.0 - ratio * 30.0
    else:
        # Beyond critical deviation: drop from 40 down to 0
        ratio = min((dev_pct - crit_thresh) / crit_thresh, 1.0)
        score = max(0.0, 40.0 - ratio * 40.0)

    return round(score, 2)


def temperature_score(temp_c: Optional[float]) -> Optional[float]:
    """
    Calculates temperature score from DS18B20 °C reading.
    - 100: Normal below 45°C
    - 100 -> 40: Warning zone 45°C to 60°C
    - <40 -> 0: Critical zone above 60°C
    - Returns None if temp_c is None.
    
    Note: Prototype demo thresholds, not certified industrial limits.
    """
    if temp_c is None:
        return None

    normal_max = settings.TEMP_NORMAL_MAX_C   # 45.0 °C
    warning_max = settings.TEMP_WARNING_MAX_C # 60.0 °C

    if temp_c <= normal_max:
        return 100.0
    elif temp_c <= warning_max:
        # Linear drop from 100.0 down to 40.0
        ratio = (temp_c - normal_max) / (warning_max - normal_max)
        score = 100.0 - ratio * 60.0
        return round(score, 2)
    else:
        # Above warning_max: drop from 40.0 down to 0.0
        excess = temp_c - warning_max
        score = max(0.0, 40.0 - excess * 2.0)
        return round(score, 2)


def magnetic_score(hall_event_expected: Optional[bool], hall_event_actual: Optional[bool]) -> Optional[float]:
    """
    Calculates magnetic event match score from A3144 Hall effect sensor.
    - 100: Event detected when expected (or no event when not expected)
    - 20: Missing event (expected pulse but none detected) or unexpected extra pulse
    - Returns None if hall_event_actual is None or hall_event_expected is None.

    Note: A3144 is a magnetic-event demonstration sensor, not an industrial MFL system.
    """
    if hall_event_actual is None or hall_event_expected is None:
        return None

    if hall_event_actual == hall_event_expected:
        return 100.0
    else:
        # Event mismatch (missing pulse or unexpected trigger)
        return 20.0


def vision_score(simulated_val: Optional[float]) -> Optional[float]:
    """
    Returns simulated or manually injected vision score (0.0 to 100.0).
    Returns None if vision_val is None (UNKNOWN).
    
    Note: Phase 1 interface stub. In Phase 2, this function will accept confidence
    scores output by the YOLOv8 surface inspection vision pipeline.
    """
    if simulated_val is None:
        return None

    clamped = max(0.0, min(100.0, float(simulated_val)))
    return round(clamped, 2)
