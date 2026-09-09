/**
 * JointGuard — Mock Alert Data
 *
 * All alerts are MOCK data for frontend demonstration.
 * Durations are calculated from start/resolution timestamps — not hardcoded.
 * System-generated alerts are read-only.
 */

import { STATUS } from './constants';

// ─── Duration Calculator ───────────────────────────────────────────────────────
/**
 * Calculate human-readable duration between two ISO timestamps.
 * @param {string} startIso
 * @param {string|null} endIso - null means ongoing
 * @returns {string}
 */
export function calcDuration(startIso, endIso = null) {
  const start = new Date(startIso).getTime();
  const end   = endIso ? new Date(endIso).getTime() : Date.now();
  const diffMs = end - start;
  const totalSecs = Math.floor(diffMs / 1000);
  const hours   = Math.floor(totalSecs / 3600);
  const minutes = Math.floor((totalSecs % 3600) / 60);
  const seconds = totalSecs % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

// ─── Mock Alerts ─────────────────────────────────────────────────────────────
// ⚠ MOCK DATA — Timestamps offset from current time for realistic demo display.
const now = Date.now();
const ago = (ms) => new Date(now - ms).toISOString();

export const MOCK_ALERTS = [
  // ── Active Alerts ────────────────────────────────────────────────────────
  {
    id: 'ALT-001',
    jointId: 'J03',
    parameter: 'temperature',
    level: STATUS.HIGH,
    reading: 67.1,
    unit: '°C',
    message: 'Temperature exceeded HIGH threshold on J03',
    startedAt: ago(42 * 60 * 1000),       // 42 minutes ago
    resolvedAt: null,
    acknowledged: false,
    acknowledgedBy: null,
    acknowledgedAt: null,
    active: true,
    source: 'system',                       // read-only
  },
  {
    id: 'ALT-002',
    jointId: 'J03',
    parameter: 'vibration',
    level: STATUS.HIGH,
    reading: 6.3,
    unit: 'm/s²',
    message: 'Vibration exceeded HIGH threshold on J03',
    startedAt: ago(38 * 60 * 1000),        // 38 minutes ago
    resolvedAt: null,
    acknowledged: true,
    acknowledgedBy: 'Operator',
    acknowledgedAt: ago(30 * 60 * 1000),
    active: true,
    source: 'system',
  },
  {
    id: 'ALT-003',
    jointId: 'J02',
    parameter: 'temperature',
    level: STATUS.MEDIUM,
    reading: 52.7,
    unit: '°C',
    message: 'Temperature in MEDIUM range on J02',
    startedAt: ago(15 * 60 * 1000),        // 15 minutes ago
    resolvedAt: null,
    acknowledged: false,
    acknowledgedBy: null,
    acknowledgedAt: null,
    active: true,
    source: 'system',
  },
  {
    id: 'ALT-004',
    jointId: 'J02',
    parameter: 'vibration',
    level: STATUS.MEDIUM,
    reading: 3.9,
    unit: 'm/s²',
    message: 'Vibration in MEDIUM range on J02',
    startedAt: ago(12 * 60 * 1000),        // 12 minutes ago
    resolvedAt: null,
    acknowledged: false,
    acknowledgedBy: null,
    acknowledgedAt: null,
    active: true,
    source: 'system',
  },

  // ── Resolved Alerts (History) ─────────────────────────────────────────────
  {
    id: 'ALT-005',
    jointId: 'J03',
    parameter: 'vibration',
    level: STATUS.HIGH,
    reading: 7.1,
    unit: 'm/s²',
    message: 'Vibration exceeded HIGH threshold on J03',
    startedAt: ago(3 * 60 * 60 * 1000),   // 3 hours ago
    resolvedAt: ago(2.5 * 60 * 60 * 1000), // Resolved after 30 min
    acknowledged: true,
    acknowledgedBy: 'Operator',
    acknowledgedAt: ago(2.8 * 60 * 60 * 1000),
    active: false,
    source: 'system',
  },
  {
    id: 'ALT-006',
    jointId: 'J02',
    parameter: 'temperature',
    level: STATUS.MEDIUM,
    reading: 55.3,
    unit: '°C',
    message: 'Temperature in MEDIUM range on J02',
    startedAt: ago(5 * 60 * 60 * 1000),
    resolvedAt: ago(4.3 * 60 * 60 * 1000),
    acknowledged: true,
    acknowledgedBy: 'Operator',
    acknowledgedAt: ago(4.8 * 60 * 60 * 1000),
    active: false,
    source: 'system',
  },
  {
    id: 'ALT-007',
    jointId: 'J01',
    parameter: 'vibration',
    level: STATUS.MEDIUM,
    reading: 3.1,
    unit: 'm/s²',
    message: 'Vibration in MEDIUM range on J01',
    startedAt: ago(6 * 60 * 60 * 1000),
    resolvedAt: ago(5.5 * 60 * 60 * 1000),
    acknowledged: true,
    acknowledgedBy: 'Operator',
    acknowledgedAt: ago(5.8 * 60 * 60 * 1000),
    active: false,
    source: 'system',
  },
  {
    id: 'ALT-008',
    jointId: 'J01',
    parameter: 'temperature',
    level: STATUS.NORMAL,
    reading: 41.2,
    unit: '°C',
    message: 'Temperature returned to NORMAL range on J01',
    startedAt: ago(7 * 60 * 60 * 1000),
    resolvedAt: ago(6.8 * 60 * 60 * 1000),
    acknowledged: true,
    acknowledgedBy: 'System',
    acknowledgedAt: ago(6.8 * 60 * 60 * 1000),
    active: false,
    source: 'system',
  },
];

// ─── Important Events (Bookmarks) ────────────────────────────────────────────
export const MOCK_EVENTS = [
  {
    id: 'EVT-001',
    timestamp: ago(42 * 60 * 1000),
    type: 'alert',
    level: STATUS.HIGH,
    jointId: 'J03',
    description: 'HIGH temperature alert triggered on J03',
  },
  {
    id: 'EVT-002',
    timestamp: ago(4 * 60 * 60 * 1000),
    type: 'system',
    level: STATUS.NORMAL,
    jointId: null,
    description: 'JointGuard monitoring system started',
  },
  {
    id: 'EVT-003',
    timestamp: ago(3 * 60 * 60 * 1000),
    type: 'alert',
    level: STATUS.HIGH,
    jointId: 'J03',
    description: 'HIGH vibration alert triggered on J03 — resolved in 30 min',
  },
  {
    id: 'EVT-004',
    timestamp: ago(5 * 60 * 60 * 1000),
    type: 'alert',
    level: STATUS.MEDIUM,
    jointId: 'J02',
    description: 'MEDIUM temperature alert on J02 — resolved after 42 min',
  },
];
