"""
JointGuard Centralized Configuration Module (Phase 1 Prototype)

All weights, baseline parameters, risk classification boundaries, and sensor thresholds
are centralized here to remain easily tunable without changing application logic.

DISCLAIMER:
- Health scoring is RULE-BASED WEIGHTED FUSION (not machine learning).
- A3144 Hall sensor is a MAGNETIC-EVENT DEMONSTRATION sensor (not industrial EM/MFL).
- Vision score is SIMULATED or MANUALLY INJECTED in Phase 1.
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "JointGuard Prototype API"
    VERSION: str = "1.0.0 (Phase 1 Prototype)"
    DEBUG: bool = True
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./jointguard.db")
    
    # --------------------------------------------------------------------------
    # WEIGHTED FUSION SCORING WEIGHTS (Must sum to 1.0)
    # --------------------------------------------------------------------------
    WEIGHT_VISION: float = 0.40
    WEIGHT_MAGNETIC: float = 0.30
    WEIGHT_VIBRATION: float = 0.20
    WEIGHT_TEMPERATURE: float = 0.10
    
    # --------------------------------------------------------------------------
    # SENSOR THRESHOLDS & BASELINES (Configurable Demo Thresholds)
    # --------------------------------------------------------------------------
    # Vibration (MPU6050 RMS Deviation)
    VIBRATION_BASELINE_RMS: float = 0.50       # Nominal vibration level (m/s^2)
    VIBRATION_WARN_DEVIATION_PCT: float = 0.20 # +20% deviation starts warning zone
    VIBRATION_CRIT_DEVIATION_PCT: float = 0.50 # +50% deviation starts critical zone
    
    # Temperature (DS18B20 °C)
    TEMP_NORMAL_MAX_C: float = 45.0   # 100 score up to 45°C
    TEMP_WARNING_MAX_C: float = 60.0  # 45°C to 60°C warning linear degradation
                                      # >60°C critical low score
    
    # --------------------------------------------------------------------------
    # RISK LEVEL CLASSIFICATION BOUNDARIES
    # --------------------------------------------------------------------------
    RISK_LOW_MIN_SCORE: float = 70.0    # 70 - 100: LOW (Green)
    RISK_MEDIUM_MIN_SCORE: float = 40.0 # 40 - 69:  MEDIUM (Yellow)
                                        # 0  - 39:  HIGH (Red)
                                        # UNKNOWN: Grey / offline state
    
    # --------------------------------------------------------------------------
    # SERIAL & ARDUINO UNO HARDWARE CONFIGURATION
    # --------------------------------------------------------------------------
    SERIAL_PORT: str = os.getenv("SERIAL_PORT", "COM4")

    SERIAL_BAUD: int = int(os.getenv("SERIAL_BAUD", "9600"))
    DEVICE_ID: str = os.getenv("DEVICE_ID", "ARDUINO_UNO_01")
    DEFAULT_JOINT_ID: str = os.getenv("DEFAULT_JOINT_ID", "J01")
    SERIAL_AUTO_RECONNECT: bool = True
    SERIAL_RECONNECT_INTERVAL_SECONDS: float = 3.0

    # --------------------------------------------------------------------------
    # VISION & YOLO CAMERA CONFIGURATION
    # --------------------------------------------------------------------------
    VISION_CAMERA_INDEX: int = int(os.getenv("VISION_CAMERA_INDEX", "1"))
    VISION_MODEL_PATH: str = os.getenv(
        "VISION_MODEL_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "check_yolo", "models", "joint_yolo_classifier_v4.pt"))
    )
    VISION_CONF_THRESH: float = float(os.getenv("VISION_CONF_THRESH", "0.60"))
    VISION_ROI: str = os.getenv("VISION_ROI", "0.15,0.20,0.85,0.80")
    VISION_INFER_INTERVAL: int = int(os.getenv("VISION_INFER_INTERVAL", "2"))
    VISION_HISTORY_LEN: int = int(os.getenv("VISION_HISTORY_LEN", "5"))

    # --------------------------------------------------------------------------
    # SYSTEM DISCLAIMERS & LABELS
    # --------------------------------------------------------------------------
    SCORING_TYPE: str = "RULE_BASED_WEIGHTED_FUSION"
    HALL_SENSOR_TYPE: str = "A3144_DEMO_MAGNETIC_EVENT"
    VISION_MODE: str = "YOLO_V8_LIVE_WEBCAM"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()


