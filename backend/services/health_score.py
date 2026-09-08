"""
JointGuard Weighted Fusion Health Scoring & Risk Classification Engine

Calculates overall health_score and assigns risk_level.

FORMULA:
  health_score = 0.40 * vision_score + 0.30 * magnetic_score + 0.20 * vibration_score + 0.10 * temperature_score

CRITICAL RULE FOR UNKNOWN:
  If ANY component score is None / UNKNOWN (missing sensor data), overall health_score becomes None,
  risk_level becomes "UNKNOWN", and is_sufficient_data becomes False.
  DO NOT silently substitute zero or skip missing components!
"""

from typing import Optional, Dict, Any, Tuple
from backend.config import settings

def classify_risk_level(health_score: Optional[float]) -> str:
    """
    Classifies health_score into risk levels:
    - 70-100: LOW (Green)
    - 40-69:  MEDIUM (Yellow)
    - 0-39:   HIGH (Red)
    - None:   UNKNOWN (Grey / insufficient data)
    """
    if health_score is None:
        return "UNKNOWN"

    if health_score >= settings.RISK_LOW_MIN_SCORE: # 70.0
        return "LOW"
    elif health_score >= settings.RISK_MEDIUM_MIN_SCORE: # 40.0
        return "MEDIUM"
    else:
        return "HIGH"


def calculate_health_score(
    v_score: Optional[float],
    m_score: Optional[float],
    vib_score: Optional[float],
    t_score: Optional[float]
) -> Tuple[Optional[float], str, bool]:
    """
    Calculates weighted fusion score, risk_level, and data sufficiency boolean.
    
    Returns:
        (health_score, risk_level, is_sufficient_data)
    """
    # Strict UNKNOWN check: If any component is None, health_score is None
    if v_score is None or m_score is None or vib_score is None or t_score is None:
        return None, "UNKNOWN", False

    # Weighted fusion sum
    weighted = (
        settings.WEIGHT_VISION * v_score +
        settings.WEIGHT_MAGNETIC * m_score +
        settings.WEIGHT_VIBRATION * vib_score +
        settings.WEIGHT_TEMPERATURE * t_score
    )

    final_score = round(max(0.0, min(100.0, weighted)), 2)
    risk = classify_risk_level(final_score)

    return final_score, risk, True
