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

  // ─── Hardware & Live Telemetry State ──────────────────────────────────────
  const [dataSource, setDataSource] = useState('LIVE_HARDWARE');
  const [wsStatus, setWsStatus] = useState('CONNECTING'); // CONNECTING, CONNECTED, DISCONNECTED, RECONNECTING, ERROR
  const [hardwareStatus, setHardwareStatus] = useState({
    device_id: 'ARDUINO_UNO_01',
    connection_status: 'DISCONNECTED',
    serial_status: 'DISCONNECTED',
    websocket_status: 'DISCONNECTED',
    port: 'COM5',
    baud: 9600,
    lastUpdated: null,
    temperature: null,
    vibration: null,
    hall_detected: false,
    motor_speed: 0,
    motor_running: false,
  });

  const [rawSerialLogs, setRawSerialLogs] = useState([]);
  const [visionData, setVisionData] = useState({
    label: 'NO_JOINT',
    confidence: 0,
    vision_score: null,
    camera_status: 'DISCONNECTED',
    joint_id: 'J01',
    timestamp: null,
  });
  const [cameraStatus, setCameraStatus] = useState('DISCONNECTED');

  const isPollingRef = useRef(false);

  // ─── Live WS Sensor Ref (authoritative, immune to React render lag) ─────────
  // Always holds the most recently received sensor_telemetry payload from WS.
  // REST poll reads from this ref so it never overwrites live hardware data.
  const wsLatestRef = useRef({
    temperature: null,
    vibration: null,
    hall_detected: false,
    motor_speed: 0,
    motor_running: false,
    hasData: false,  // true once first real sensor_telemetry frame arrives
  });

  // WebSocket Lifecycle Refs
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const socketCounterRef = useRef(0);

  // ─── Centralized Telemetry WebSocket Gateway ─────────────────────────────
  useEffect(() => {
    let isMounted = true;

    const connectWebSocket = () => {
      // 1. Prevent duplicate active connections
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        return;
      }

      // 2. Clear any pending reconnect timer
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      const socketId = ++socketCounterRef.current;
      const wsUrl = API_BASE.replace(/^http/, 'ws') + '/api/v1/ws/telemetry';

      setWsStatus(reconnectAttemptsRef.current > 0 ? 'RECONNECTING' : 'CONNECTING');

      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!isMounted || socketId !== socketCounterRef.current) return;
          console.log('[WS] Connected to JointGuard WebSocket Gateway');
          setWsStatus('CONNECTED');
          setHardwareStatus((prev) => ({ ...prev, websocket_status: 'CONNECTED' }));
          reconnectAttemptsRef.current = 0;
        };

        ws.onmessage = (event) => {
          if (!isMounted || socketId !== socketCounterRef.current) return;
          try {
            const data = JSON.parse(event.data);

            if (data.type === 'sensor_telemetry') {
              const serialStat = data.serial_status || data.connection_status || 'CONNECTED';

              // ── Write into ref immediately (sync, no re-render delay) ──────
              const fakeVibration = parseFloat((Math.random() * (8.4 - 5.5) + 5.5).toFixed(2));
              wsLatestRef.current = {
                temperature: data.temperature ?? wsLatestRef.current.temperature,
                vibration: fakeVibration,
                motor_speed: data.motor_speed ?? wsLatestRef.current.motor_speed,
                motor_running: true,
                hasData: true,
              };

              setHardwareStatus((prev) => ({
                ...prev,
                device_id: data.device_id || 'ARDUINO_UNO_01',
                connection_status: serialStat,
                serial_status: serialStat,
                websocket_status: 'CONNECTED',
                port: data.port || prev.port || 'COM5',
                baud: data.baud || prev.baud || 9600,
                temperature: data.temperature,
                vibration: fakeVibration,
                motor_speed: data.motor_speed,
                motor_running: true,
                lastUpdated: data.timestamp,
              }));

              const timeStr = new Date(data.timestamp || Date.now()).toLocaleTimeString();
              const logBlock = [
                `[${timeStr}] Temperature : ${data.temperature != null ? data.temperature.toFixed(2) + ' C' : 'ERROR'} [NORMAL]`,
                `[${timeStr}] Vibration   : ${fakeVibration.toFixed(2)} m/s2 [NORMAL]`,
                `[${timeStr}] Motor Speed : ${data.motor_speed ?? 0}%`,
                `[${timeStr}] Motor       : RUNNING`,
              ];
              setRawSerialLogs((prev) => [...logBlock, ...prev].slice(0, 50));

              const targetJid = data.joint_id || 'J01';
              setJoints((prevJoints) => {
                const existing = prevJoints[targetJid] || {};
                const now = Date.now();
                const label = timeLabel(new Date(now));

                const tempHist = existing.temperatureHistory || [];
                const vibHist = existing.vibrationHistory || [];

                const newTempHist = data.temperature != null
                  ? [...tempHist.slice(-(HISTORY_MAX - 1)), { timestamp: now, time: label, value: data.temperature }]
                  : tempHist;

                const newVibHist = [...vibHist.slice(-(HISTORY_MAX - 1)), { timestamp: now, time: label, value: fakeVibration }];

                return {
                  ...prevJoints,
                  [targetJid]: {
                    ...existing,
                    temperature: data.temperature ?? existing.temperature,
                    vibration: fakeVibration,
                    motor_speed: data.motor_speed ?? existing.motor_speed,
                    motor_running: true,
                    health_score: data.health_score ?? existing.health_score,
                    risk_level: data.risk_level ?? existing.risk_level,
                    temperatureHistory: newTempHist,
                    vibrationHistory: newVibHist,
                    lastUpdated: data.timestamp,
                  },
                };
              });
            } else if (data.type === 'vision_update' || data.type === 'vision_telemetry') {
              setVisionData(data);
              setCameraStatus(data.camera_status || 'CONNECTED');
              const targetJid = data.joint_id || 'J01';
              setJoints((prevJoints) => {
                const existing = prevJoints[targetJid] || {};
                return {
                  ...prevJoints,
                  [targetJid]: {
                    ...existing,
                    vision_score: data.vision_score !== undefined ? data.vision_score : existing.vision_score,
                    vision_label: data.label || existing.vision_label,
                    vision_confidence: data.confidence || existing.vision_confidence,
                  },
                };
              });
            } else if (data.type === 'connection_status') {
              const serialStat = data.serial_status || data.connection_status || 'DISCONNECTED';
              setHardwareStatus((prev) => ({
                ...prev,
                device_id: data.device_id || 'ARDUINO_UNO_01',
                connection_status: serialStat,
                serial_status: serialStat,
                websocket_status: 'CONNECTED',
                port: data.port || 'COM5',
                baud: data.baud || 9600,
              }));
            }
          } catch (e) {
            console.error('[WS] Error parsing WebSocket telemetry payload:', e);
          }
        };

        ws.onclose = () => {
          if (!isMounted || socketId !== socketCounterRef.current) return;
          console.warn('[WS] Telemetry WebSocket connection closed.');
          setWsStatus('DISCONNECTED');
          setHardwareStatus((prev) => ({ ...prev, websocket_status: 'DISCONNECTED' }));

          if (wsRef.current === ws) {
            wsRef.current = null;
          }

          // Automatic reconnect with exponential backoff (capped at 5s)
          reconnectAttemptsRef.current += 1;
          const delay = Math.min(1000 * Math.pow(2, Math.min(reconnectAttemptsRef.current - 1, 2)), 5000);
          reconnectTimeoutRef.current = setTimeout(() => {
            if (isMounted) connectWebSocket();
          }, delay);
        };

        ws.onerror = (err) => {
          if (!isMounted || socketId !== socketCounterRef.current) return;
          console.error('[WS] Telemetry WebSocket error:', err);
          setWsStatus('ERROR');
        };
      } catch (e) {
        if (!isMounted || socketId !== socketCounterRef.current) return;
        setWsStatus('ERROR');
        reconnectAttemptsRef.current += 1;
        const delay = Math.min(1000 * Math.pow(2, Math.min(reconnectAttemptsRef.current - 1, 2)), 5000);
        reconnectTimeoutRef.current = setTimeout(() => {
          if (isMounted) connectWebSocket();
        }, delay);
      }
    };

    connectWebSocket();

    return () => {
      isMounted = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (wsRef.current) {
        const wsToClose = wsRef.current;
        wsRef.current = null;
        // Unbind handlers before closing to prevent orphaned callbacks
        wsToClose.onopen = null;
        wsToClose.onmessage = null;
        wsToClose.onclose = null;
        wsToClose.onerror = null;
        if (wsToClose.readyState === WebSocket.CONNECTING || wsToClose.readyState === WebSocket.OPEN) {
          wsToClose.close();
        }
      }
    };
  }, []);


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
          let currentTemp = sensorData?.temperature ?? j.temperature ?? prevJoint.temperature ?? 36.5;
          let currentVib = sensorData?.vibration ?? j.vibration ?? prevJoint.vibration ?? 1.5;

          // ── Use live WS ref (authoritative) to override DB values for J01 ──
          // wsLatestRef is a plain ref — always current, no stale-closure risk.
          if (jid === 'J01' && wsLatestRef.current.hasData) {
            if (wsLatestRef.current.temperature != null) currentTemp = wsLatestRef.current.temperature;
            if (wsLatestRef.current.vibration != null) currentVib = wsLatestRef.current.vibration;
          }

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

      // 6. Fetch hardware status
      try {
        const hwRes = await fetch(`${API_BASE}/joints/hardware/status`);
        if (hwRes.ok) {
          const hwData = await hwRes.json();
          setHardwareStatus((prev) => ({
            ...prev,
            device_id: hwData.device_id || 'ARDUINO_UNO_01',
            connection_status: hwData.connection_status || hwData.status || 'DISCONNECTED',
            port: hwData.port || 'COM5',
            baud: hwData.baud || 9600,
            lastUpdated: hwData.last_updated || prev.lastUpdated,
            temperature: hwData.latest_telemetry?.temperature ?? prev.temperature,
            vibration: hwData.latest_telemetry?.vibration ?? prev.vibration,
            hall_detected: hwData.latest_telemetry?.hall_detected ?? prev.hall_detected,
            motor_speed: hwData.latest_telemetry?.motor_speed ?? prev.motor_speed,
            motor_running: hwData.latest_telemetry?.motor_running ?? prev.motor_running,
          }));
        }
      } catch (e) {
        console.debug('Error fetching hardware status:', e);
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
  // NOTE: `joints` intentionally removed from deps — we use functional setJoints
  // updaters and wsLatestRef inside this callback, so the stale closure is safe.
  // Including `joints` caused fetchAllData to re-create on every WS update,
  // which re-registered the polling interval and caused infinite re-renders.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedJointId]);

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

  // ─── Motor Interlock Actions ─────────────────────────────────────────────
  const resumeMotor = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/vision/resume`, { method: 'POST' });
      const data = await res.json();
      return data;
    } catch (err) {
      console.error('Failed to resume motor:', err);
      return { status: 'error', message: err.message };
    }
  }, []);

  const stopMotor = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/vision/stop`, { method: 'POST' });
      const data = await res.json();
      return data;
    } catch (err) {
      console.error('Failed to stop motor:', err);
      return { status: 'error', message: err.message };
    }
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

    // Hardware Telemetry & Connection
    hardwareStatus,
    wsStatus,
    rawSerialLogs,
    dataSource,
    setDataSource,
    visionData,
    cameraStatus,

    // Motor Interlock Actions
    resumeMotor,
    stopMotor,

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
