/**
 * JointGuard — Central Constants
 *
 * All configurable values are centralized here.
 * Thresholds are PLACEHOLDER values for frontend demonstration only.
 * Replace with validated values derived from real prototype sensor data and testing.
 */

// ─── Backend API Base URL ──────────────────────────────────────────────────
export const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ─── Joint Identifiers (3 Joints Monitored) ──────────────────────────────────
export const JOINT_IDS = ['J01', 'J02', 'J03'];

export const JOINT_LABELS = {
  J01: 'Joint 01',
  J02: 'Joint 02',
  J03: 'Joint 03',
};

export const JOINT_LOCATIONS = {
  J01: 'Conveyor Belt — Position A (Feed End)',
  J02: 'Conveyor Belt — Position B (Mid-Section)',
  J03: 'Conveyor Belt — Position C (Discharge End)',
};

// ─── Status Levels & Mappings ─────────────────────────────────────────────────
export const STATUS = {
  NORMAL: 'NORMAL',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
  UNKNOWN: 'UNKNOWN',
};

export const RISK_TO_STATUS = {
  LOW: 'NORMAL',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
  UNKNOWN: 'UNKNOWN',
};

export const STATUS_TO_RISK = {
  NORMAL: 'LOW',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
  UNKNOWN: 'UNKNOWN',
};

// ─── Status Colors (CSS variable names for reference) ─────────────────────────
export const STATUS_COLORS = {
  NORMAL:  '#16a34a',  // darkened one shade — matches --status-normal on white bg
  MEDIUM:  '#d97706',  // darkened one shade — matches --status-medium on white bg
  HIGH:    '#dc2626',  // darkened one shade — matches --status-high on white bg
  UNKNOWN: '#64748b',
};

// ─── Sensor Thresholds ────────────────────────────────────────────────────────
// ⚠ PLACEHOLDER VALUES — Not scientifically validated.
// Replace with values derived from real JointGuard prototype sensor data and field testing.
// Structure intentionally centralized so Settings page can override at runtime.
export const DEFAULT_THRESHOLDS = {
  temperature: {
    // Unit: °C (Celsius)
    normalMax: 45,  // ≤ 45°C  → NORMAL   [PLACEHOLDER]
    mediumMax: 60,  // ≤ 60°C  → MEDIUM   [PLACEHOLDER]
    // > 60°C → HIGH                       [PLACEHOLDER]
  },
  vibration: {
    // Unit: m/s² (acceleration magnitude)
    normalMax: 2.5, // ≤ 2.5   → NORMAL   [PLACEHOLDER]
    mediumMax: 5.0, // ≤ 5.0   → MEDIUM   [PLACEHOLDER]
    // > 5.0  → HIGH                       [PLACEHOLDER]
  },
};

// ─── Sensor Metadata ──────────────────────────────────────────────────────────
export const SENSOR_INFO = {
  temperature: {
    name: 'DS18B20',
    unit: '°C',
    label: 'Temperature',
    description: '1-Wire digital temperature sensor',
    range: '-55°C to +125°C',
    resolution: '0.0625°C',
  },
  vibration: {
    name: 'MPU6050',
    unit: 'm/s²',
    label: 'Vibration',
    description: '6-axis IMU (accelerometer + gyroscope)',
    range: '±2g to ±16g',
    resolution: '16-bit ADC',
  },
};

// ─── System Configuration ─────────────────────────────────────────────────────
export const SYSTEM_CONFIG = {
  defaultRefreshRateMs: 3000,   // Live data polling interval (ms)
  historyPointCount: 30,         // Number of data points to keep in chart history
  esp32Model: 'ESP32-WROOM-32D',
  firmwareVersion: 'v1.0.0-prototype',
  backendUrl: 'http://localhost:8000', // FastAPI backend (not yet implemented)
  databaseType: 'PostgreSQL',          // Future DB
};

// ─── Navigation Items ─────────────────────────────────────────────────────────
export const NAV_ITEMS = [
  { id: 'dashboard',       label: 'Dashboard',        path: '/' },
  { id: 'joint-monitoring',label: 'Joint Monitoring', path: '/joints' },
  { id: 'sensor-data',     label: 'Sensor Data',      path: '/sensors' },
  { id: 'alerts',          label: 'Alerts',           path: '/alerts' },
  { id: 'history',         label: 'History & Reports',path: '/history' },
  { id: 'about',           label: 'About Project',    path: '/about' },
  { id: 'settings',        label: 'Settings',         path: '/settings' },
];

// ─── Condition Score Thresholds ───────────────────────────────────────────────
// ⚠ PLACEHOLDER — Score derived from a simple weighted formula for demo purposes.
// Replace with validated scoring model after real testing.
export const CONDITION_SCORE_THRESHOLDS = {
  good: 75,    // ≥ 75 → Good condition    [PLACEHOLDER]
  fair: 50,    // ≥ 50 → Fair condition    [PLACEHOLDER]
  // < 50 → Poor condition                  [PLACEHOLDER]
};
