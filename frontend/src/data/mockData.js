/**
 * JointGuard — Centralized Mock Data Layer
 *
 * All data in this file is MOCK / SIMULATED for frontend demonstration.
 * Replace API calls by swapping these functions with fetch('/api/...') calls.
 *
 * Structure mirrors the expected FastAPI response shape for easy migration.
 */

import { STATUS, DEFAULT_THRESHOLDS, JOINT_IDS } from './constants';

// ─── Threshold Helper ─────────────────────────────────────────────────────────
/**
 * Determine STATUS from a value and thresholds config.
 * @param {number} value
 * @param {{ normalMax: number, mediumMax: number }} thresholds
 * @returns {string} STATUS.NORMAL | STATUS.MEDIUM | STATUS.HIGH
 */
export function getValueStatus(value, thresholds) {
  if (value <= thresholds.normalMax) return STATUS.NORMAL;
  if (value <= thresholds.mediumMax) return STATUS.MEDIUM;
  return STATUS.HIGH;
}

/**
 * Determine overall joint STATUS from temperature and vibration.
 * Overall status is the worse of the two.
 */
export function getJointStatus(temperature, vibration, thresholds = DEFAULT_THRESHOLDS) {
  const tempStatus = getValueStatus(temperature, thresholds.temperature);
  const vibStatus  = getValueStatus(vibration,  thresholds.vibration);
  const order = [STATUS.NORMAL, STATUS.MEDIUM, STATUS.HIGH];
  return order[Math.max(order.indexOf(tempStatus), order.indexOf(vibStatus))];
}

/**
 * Compute a simple condition score (0–100) from sensor values.
 * ⚠ PLACEHOLDER formula — not scientifically validated.
 * Replace with validated scoring model after real sensor testing.
 */
export function computeConditionScore(temperature, vibration, thresholds = DEFAULT_THRESHOLDS) {
  const tempMax = thresholds.temperature.mediumMax;
  const vibMax  = thresholds.vibration.mediumMax;
  const tempScore = Math.max(0, 100 - ((temperature / tempMax) * 50));
  const vibScore  = Math.max(0, 100 - ((vibration  / vibMax)  * 50));
  return Math.round((tempScore + vibScore) / 2);
}

// ─── Time-Series Generator ────────────────────────────────────────────────────
/**
 * Generate an array of historical readings for chart display.
 * @param {number} baseValue  - Anchor value
 * @param {number} count      - Number of data points
 * @param {number} variance   - Max random drift ±
 * @param {number} minVal     - Floor value
 * @param {number} maxVal     - Ceiling value
 */
function generateTimeSeries(baseValue, count = 30, variance = 2, minVal = 0, maxVal = 100) {
  const now = Date.now();
  const intervalMs = 3000; // 3-second intervals
  return Array.from({ length: count }, (_, i) => {
    const t = now - (count - 1 - i) * intervalMs;
    const drift = (Math.random() - 0.5) * 2 * variance;
    const value = Math.max(minVal, Math.min(maxVal, baseValue + drift));
    return {
      timestamp: t,
      time: new Date(t).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      value: parseFloat(value.toFixed(2)),
    };
  });
}

// ─── Mock Joint Snapshot Data ─────────────────────────────────────────────────
// ⚠ All values are MOCK — not from real sensors.
// Chosen to demonstrate different status levels across the 3 joints.
export const MOCK_JOINT_SNAPSHOTS = {
  J01: {
    id: 'J01',
    label: 'Joint 01',
    location: 'Conveyor Belt — Position A (Feed End)',
    temperature: 38.4,   // °C  → NORMAL  [MOCK]
    vibration:   1.8,    // m/s² → NORMAL [MOCK]
    temperatureHistory: generateTimeSeries(38.4, 30, 2.5, 15, 80),
    vibrationHistory:   generateTimeSeries(1.8,  30, 0.4, 0,  10),
    lastUpdated: new Date().toISOString(),
  },
  J02: {
    id: 'J02',
    label: 'Joint 02',
    location: 'Conveyor Belt — Position B (Mid-Section)',
    temperature: 52.7,   // °C  → MEDIUM  [MOCK]
    vibration:   3.9,    // m/s² → MEDIUM [MOCK]
    temperatureHistory: generateTimeSeries(52.7, 30, 3.0, 15, 80),
    vibrationHistory:   generateTimeSeries(3.9,  30, 0.6, 0,  10),
    lastUpdated: new Date().toISOString(),
  },
  J03: {
    id: 'J03',
    label: 'Joint 03',
    location: 'Conveyor Belt — Position C (Discharge End)',
    temperature: 67.1,   // °C  → HIGH    [MOCK]
    vibration:   6.3,    // m/s² → HIGH   [MOCK]
    temperatureHistory: generateTimeSeries(67.1, 30, 4.0, 15, 85),
    vibrationHistory:   generateTimeSeries(6.3,  30, 0.8, 0,  12),
    lastUpdated: new Date().toISOString(),
  },
};

// ─── System Status Snapshot ───────────────────────────────────────────────────
export const MOCK_SYSTEM_STATUS = {
  systemOnline: true,
  monitoringStarted: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(), // 4 hours ago
  lastDataUpdate: new Date().toISOString(),
  esp32: {
    connected: true,
    ipAddress: '192.168.1.105',
    signalStrength: -52, // dBm
    uptime: '4h 12m',
    model: 'ESP32-WROOM-32D',
    firmware: 'v1.0.0-prototype',
  },
  sensors: {
    ds18b20: { connected: true, readingsOk: true, faultCode: null },
    mpu6050: { connected: true, readingsOk: true, faultCode: null },
  },
};

// ─── System-wide Aggregates ───────────────────────────────────────────────────
export function getMockSystemAggregates() {
  const joints = Object.values(MOCK_JOINT_SNAPSHOTS);
  const temps  = joints.map(j => j.temperature);
  const vibs   = joints.map(j => j.vibration);
  return {
    temperature: {
      min: Math.min(...temps),
      max: Math.max(...temps),
      avg: parseFloat((temps.reduce((a, b) => a + b, 0) / temps.length).toFixed(1)),
      current: temps[temps.length - 1],
    },
    vibration: {
      min: Math.min(...vibs),
      max: Math.max(...vibs),
      avg: parseFloat((vibs.reduce((a, b) => a + b, 0) / vibs.length).toFixed(2)),
      current: vibs[vibs.length - 1],
    },
  };
}

// ─── Data Quality ─────────────────────────────────────────────────────────────
export const MOCK_DATA_QUALITY = {
  totalReadings: 1842,
  validReadings: 1836,
  qualityPercent: 99.7,
  lastError: null,
};

// ─── Raw Sensor Table (last 20 readings per joint) ───────────────────────────
export function getMockRawReadings(jointId = null, count = 20) {
  const now = Date.now();
  const joints = jointId ? [jointId] : JOINT_IDS;
  const rows = [];
  joints.forEach(jid => {
    const snap = MOCK_JOINT_SNAPSHOTS[jid];
    for (let i = count - 1; i >= 0; i--) {
      const t = now - i * 3000;
      const tempDrift = (Math.random() - 0.5) * 4;
      const vibDrift  = (Math.random() - 0.5) * 0.8;
      rows.push({
        id: `${jid}-${t}`,
        timestamp: new Date(t).toISOString(),
        time: new Date(t).toLocaleTimeString(),
        jointId: jid,
        temperature: parseFloat((snap.temperature + tempDrift).toFixed(2)),
        vibration:   parseFloat((snap.vibration   + vibDrift).toFixed(3)),
      });
    }
  });
  return rows.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
}
