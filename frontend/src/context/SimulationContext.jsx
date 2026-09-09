/**
 * JointGuard — Central Simulation & Backend Integration Context
 *
 * Connects to the FastAPI REST backend:
 *   - Polls /joints, /joints/{id}/sensors, /joints/{id}, /alerts, /simulation/status
 *   - Binds real multi-joint telemetry (J01-J03)
 *   - Exposes simulation engine controls: Start, Stop, Step, Reset, Inject Override
 *   - Manages connection status (apiConnected) with graceful offline fallback
 */

import {
  createContext, useContext, useState, useEffect, useCallback, useRef,
} from 'react';
import {
  API_BASE, JOINT_IDS, JOINT_LABELS, JOINT_LOCATIONS,
  STATUS, RISK_TO_STATUS,
} from '../data/constants';
import { computeConditionScore } from '../data/mockData';

const SimulationContext = createContext(null);

const POLL_INTERVAL_MS = 2000;
const HISTORY_MAX = 30;

function nowISO() {
  return new Date().toISOString();
}

function timeLabel(date = new Date()) {
  return new Intl.DateTimeFormat([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  }).format(date);
}

// ─── Initial Baseline State (for instant render before first REST response) ────
function buildInitialJoints() {
  const initial = {};
  const defaultValues = {
    J01: { temp: 36.5, accel: 1.55, risk: 'LOW', score: 88, zone: 'APPROACHING' },
    J02: { temp: 39.0, accel: 1.80, risk: 'LOW', score: 85, zone: 'APPROACHING' },
    J03: { temp: 42.5, accel: 2.10, risk: 'MEDIUM', score: 72, zone: 'INSPECTING' },
  };

  JOINT_IDS.forEach((id) => {
    const d = defaultValues[id] || { temp: 37.0, accel: 1.6, risk: 'LOW', score: 85, zone: 'PASSED' };
    const now = Date.now();
    const tempHist = [];
    const accelHist = [];

    for (let i = 0; i < 15; i++) {
      const ts = now - (15 - i) * POLL_INTERVAL_MS;
      const t = d.temp + (Math.random() - 0.5) * 1.2;
      const a = d.accel + (Math.random() - 0.5) * 0.15;
      const label = timeLabel(new Date(ts));
      tempHist.push({ timestamp: ts, time: label, value: parseFloat(t.toFixed(2)) });
      accelHist.push({ timestamp: ts, time: label, value: parseFloat(a.toFixed(3)) });
    }

    initial[id] = {
      id,
      joint_id: id,
      label: JOINT_LABELS[id] || `Joint ${id}`,
      location: JOINT_LOCATIONS[id] || `Conveyor Joint ${id}`,
      temperature: d.temp,
      vibration: d.accel,
      hall_event: false,
      magnetic_value: 0.05,
      vision_score: 85.0,
      status: RISK_TO_STATUS[d.risk] || STATUS.NORMAL,
      risk_level: d.risk,
      conditionScore: d.score,
      health_score: d.score,
      belt_position: 0,
      zone_state: d.zone,
      component_scores: {
        vision_score: 85.0,
        magnetic_score: 90.0,
        vibration_score: 88.0,
        temperature_score: 92.0,
      },
      temperatureHistory: tempHist,
      vibrationHistory: accelHist,
      lastUpdated: nowISO(),
    };
  });

  return initial;
}

export function SimulationProvider({ children }) {
  const [joints, setJoints] = useState(buildInitialJoints);
  const [selectedJointId, setSelectedJointId] = useState('J01');
  const [selectedJointDetail, setSelectedJointDetail] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [alertHistory, setAlertHistory] = useState([]);
  const [simStatus, setSimStatus] = useState({
    is_running: true,
    current_cycle: 0,
    interval_seconds: 2.0,
    active_joints: JOINT_IDS,
  });
  const [apiConnected, setApiConnected] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [updateCount, setUpdateCount] = useState(0);

  const isPollingRef = useRef(false);

  // ─── Fetch All Data from FastAPI Backend ──────────────────────────────────────
  const fetchAllData = useCallback(async () => {
    if (isPollingRef.current) return;
    isPollingRef.current = true;

    try {
      // 1. Fetch joints summary
      const jointsRes = await fetch(`${API_BASE}/joints`);
      if (!jointsRes.ok) {
        setApiConnected(false);
        return;
      }

      const allJoints = await jointsRes.json();
      const rawJoints = allJoints.filter(j => JOINT_IDS.includes(j.joint_id));
      const updatedJoints = { ...joints };

      // 2. Enrich joints with raw sensors and historical trends
      await Promise.all(
        rawJoints.map(async (j) => {
          const jid = j.joint_id;
          let sensorData = null;
          let historyData = [];

          try {
            const sRes = await fetch(`${API_BASE}/joints/${jid}/sensors`);
            if (sRes.ok) sensorData = await sRes.json();
          } catch {
            // keep null
          }

          try {
            const hRes = await fetch(`${API_BASE}/history/${jid}?limit=${HISTORY_MAX}`);
            if (hRes.ok) historyData = await hRes.json();
          } catch {
            // keep empty
          }

          const prevJoint = updatedJoints[jid] || {};
          const currentTemp = sensorData?.temperature ?? j.temperature ?? prevJoint.temperature ?? 36.5;
          const currentVib = sensorData?.vibration ?? j.vibration ?? prevJoint.vibration ?? 1.5;
          const currentRisk = j.risk_level || 'LOW';
          const uiStatus = RISK_TO_STATUS[currentRisk] || STATUS.NORMAL;
          const score = j.health_score != null ? Math.round(j.health_score) : computeConditionScore(currentTemp, currentVib);

          // Format chart histories
          let tempHistory = prevJoint.temperatureHistory || [];
          let vibHistory = prevJoint.vibrationHistory || [];

          if (historyData && historyData.length > 0) {
            tempHistory = historyData.map((h) => ({
              timestamp: new Date(h.timestamp).getTime(),
              time: timeLabel(new Date(h.timestamp)),
              value: parseFloat((h.temperature ?? 0).toFixed(2)),
            }));
            vibHistory = historyData.map((h) => ({
              timestamp: new Date(h.timestamp).getTime(),
              time: timeLabel(new Date(h.timestamp)),
              value: parseFloat((h.vibration ?? 0).toFixed(3)),
            }));
          } else {
            const now = Date.now();
            const label = timeLabel(new Date(now));
            tempHistory = [
              ...tempHistory.slice(-(HISTORY_MAX - 1)),
              { timestamp: now, time: label, value: parseFloat(currentTemp.toFixed(2)) },
            ];
            vibHistory = [
              ...vibHistory.slice(-(HISTORY_MAX - 1)),
              { timestamp: now, time: label, value: parseFloat(currentVib.toFixed(3)) },
            ];
          }

          updatedJoints[jid] = {
            id: jid,
            joint_id: jid,
            label: JOINT_LABELS[jid] || `Joint ${jid}`,
            location: JOINT_LOCATIONS[jid] || `Conveyor Joint ${jid}`,
            temperature: parseFloat(Number(currentTemp).toFixed(2)),
            vibration: parseFloat(Number(currentVib).toFixed(3)),
            hall_event: sensorData?.hall_event ?? prevJoint.hall_event ?? false,
            magnetic_value: sensorData?.magnetic_value ?? prevJoint.magnetic_value ?? null,
            vision_score: sensorData?.vision_score ?? prevJoint.vision_score ?? 85.0,
            status: uiStatus,
            risk_level: currentRisk,
            conditionScore: score,
            health_score: j.health_score ?? score,
            belt_position: j.belt_position ?? prevJoint.belt_position ?? 0,
            zone_state: j.zone_state || prevJoint.zone_state || 'APPROACHING',
            temperatureHistory: tempHistory,
            vibrationHistory: vibHistory,
            lastUpdated: j.last_updated || nowISO(),
          };
        }),
      );

      setJoints(updatedJoints);

      // 3. Fetch selected joint detail if available
      if (selectedJointId) {
        try {
          const detailRes = await fetch(`${API_BASE}/joints/${selectedJointId}`);
          if (detailRes.ok) {
            const detail = await detailRes.json();
            setSelectedJointDetail(detail);
            if (updatedJoints[selectedJointId]) {
              updatedJoints[selectedJointId].component_scores = detail.current_health?.component_scores;
            }
          }
        } catch (e) {
          console.debug('Error fetching joint detail:', e);
        }
      }

      // 4. Fetch Active Alerts & History
      try {
        const [alertsRes, historyRes] = await Promise.all([
          fetch(`${API_BASE}/alerts`),
          fetch(`${API_BASE}/alerts/history`),
        ]);

        if (alertsRes.ok) {
          const active = await alertsRes.json();
          // Normalize active alerts
          const normalizedActive = active.map((a) => ({
            id: `ALT-${a.id}`,
            backendId: a.id,
            jointId: a.joint_id,
            parameter: a.alert_type || 'system',
            level: RISK_TO_STATUS[a.risk_level] || a.risk_level,
            reading: a.risk_level,
            unit: '',
            message: a.message,
            startedAt: a.timestamp,
            resolvedAt: null,
            acknowledged: false,
            acknowledgedBy: null,
            acknowledgedAt: null,
            active: true,
            source: 'system',
          }));

          let normalizedHistory = [];
          if (historyRes.ok) {
            const hist = await historyRes.json();
            normalizedHistory = hist.map((a) => ({
              id: `HIST-${a.id}`,
              backendId: a.id,
              jointId: a.joint_id,
              parameter: a.alert_type || 'system',
              level: RISK_TO_STATUS[a.risk_level] || a.risk_level,
              reading: a.risk_level,
              unit: '',
              message: a.message,
              startedAt: a.timestamp,
              resolvedAt: a.timestamp,
              acknowledged: true,
              acknowledgedBy: 'System',
              acknowledgedAt: a.timestamp,
              active: false,
              source: 'system',
            }));
          }

          setAlerts([...normalizedActive, ...normalizedHistory]);
          setAlertHistory(normalizedHistory);
        }
      } catch (e) {
        console.debug('Error fetching alerts:', e);
      }

      // 5. Fetch simulation status
      try {
        const simRes = await fetch(`${API_BASE}/simulation/status`);
        if (simRes.ok) {
          setSimStatus(await simRes.json());
        }
      } catch (e) {
        console.debug('Error fetching simulation status:', e);
      }

      setApiConnected(true);
      setLastUpdated(nowISO());
      setUpdateCount((c) => c + 1);
    } catch (err) {
      console.warn('FastAPI backend connection error:', err);
      setApiConnected(false);
    } finally {
      isPollingRef.current = false;
    }
  }, [selectedJointId, joints]);

  // Polling loop
  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchAllData]);

  // ─── Simulation Controls ──────────────────────────────────────────────────────
  const runSimulationAction = useCallback(async (actionFn) => {
    try {
      await actionFn();
      await fetchAllData();
    } catch (err) {
      console.error('Simulation control error:', err);
    }
  }, [fetchAllData]);

  const startSimulation = useCallback(async () => {
    await runSimulationAction(() => fetch(`${API_BASE}/simulation/start`, { method: 'POST' }));
  }, [runSimulationAction]);

  const stopSimulation = useCallback(async () => {
    await runSimulationAction(() => fetch(`${API_BASE}/simulation/stop`, { method: 'POST' }));
  }, [runSimulationAction]);

  const resetSimulation = useCallback(async () => {
    await runSimulationAction(() => fetch(`${API_BASE}/simulation/reset`, { method: 'POST' }));
  }, [runSimulationAction]);

  const stepSimulation = useCallback(async () => {
    await runSimulationAction(() => fetch(`${API_BASE}/simulation/step`, { method: 'POST' }));
  }, [runSimulationAction]);

  const injectOverride = useCallback(async (payload) => {
    await runSimulationAction(() =>
      fetch(`${API_BASE}/simulation/inject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }),
    );
  }, [runSimulationAction]);

  // ─── Alert Operator Actions ───────────────────────────────────────────────────
  const acknowledge = useCallback((alertId, operator = 'Operator') => {
    setAlerts((prev) =>
      prev.map((a) =>
        a.id === alertId && !a.acknowledged
          ? { ...a, acknowledged: true, acknowledgedBy: operator, acknowledgedAt: nowISO() }
          : a,
      ),
    );
  }, []);

  const resolve = useCallback((alertId) => {
    const ts = nowISO();
    setAlerts((prev) =>
      prev.map((a) =>
        a.id === alertId && a.active
          ? { ...a, active: false, resolvedAt: ts, acknowledged: true }
          : a,
      ),
    );
  }, []);

  // ─── Derived Alert Values ─────────────────────────────────────────────────────
  const activeAlerts = alerts.filter((a) => a.active);
  const resolvedAlerts = alerts.filter((a) => !a.active);
  const summary = {
    totalActive: activeAlerts.length,
    highCount: activeAlerts.filter((a) => a.level === 'HIGH').length,
    mediumCount: activeAlerts.filter((a) => a.level === 'MEDIUM').length,
    normalCount: activeAlerts.filter((a) => a.level === 'NORMAL').length,
  };

  const value = {
    // Sensor & joint data
    joints,
    selectedJointId,
    setSelectedJointId,
    selectedJointDetail,
    lastUpdated,
    updateCount,
    apiConnected,

    // Simulation controls & status
    simStatus,
    startSimulation,
    stopSimulation,
    resetSimulation,
    stepSimulation,
    injectOverride,
    refreshData: fetchAllData,

    // Alert data & actions
    alerts,
    activeAlerts,
    resolvedAlerts,
    alertHistory,
    acknowledge,
    resolve,
    summary,
  };

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
}

export function useSimulation() {
  const ctx = useContext(SimulationContext);
  if (!ctx) throw new Error('useSimulation must be used inside <SimulationProvider>');
  return ctx;
}
