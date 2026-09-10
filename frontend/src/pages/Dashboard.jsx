import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Thermometer, Activity, Link2, Radio,
  ChevronRight,
} from 'lucide-react';
import { useLiveSensorData } from '../hooks/useLiveSensorData';
import { useAlerts } from '../hooks/useAlerts';
import {
  getValueStatus, computeConditionScore, MOCK_SYSTEM_STATUS,
} from '../data/mockData';
import { STATUS, DEFAULT_THRESHOLDS, JOINT_IDS } from '../data/constants';
import { calcDuration, MOCK_EVENTS } from '../data/mockAlerts';
import MetricCard from '../components/ui/MetricCard';
import StatusBadge from '../components/ui/StatusBadge';
import SensorChart from '../components/ui/SensorChart';
import AlertBanner from '../components/dashboard/AlertBanner';
import SimulationControls from '../components/dashboard/SimulationControls';
import Modal from '../components/ui/Modal';
import CameraVision from '../components/ui/CameraVision';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getOverallStatus(joints) {
  const statuses = Object.values(joints).map(j => j.status);
  if (statuses.includes(STATUS.HIGH))   return STATUS.HIGH;
  if (statuses.includes(STATUS.MEDIUM)) return STATUS.MEDIUM;
  return STATUS.NORMAL;
}

function scoreColor(score) {
  if (score >= 75) return 'var(--status-normal)';
  if (score >= 50) return 'var(--status-medium)';
  return 'var(--status-high)';
}

// ─── Joint Quick-View Modal ───────────────────────────────────────────────────
function JointQuickModal({ jointId, joints, onClose, onInspect }) {
  if (!jointId) return null;
  const joint = joints[jointId];
  if (!joint) return null;

  const tempSt  = getValueStatus(joint.temperature, DEFAULT_THRESHOLDS.temperature);
  const accelSt = getValueStatus(joint.vibration,   DEFAULT_THRESHOLDS.vibration);
  const score   = joint.conditionScore ?? computeConditionScore(joint.temperature, joint.vibration);
  const sc      = scoreColor(score);

  return (
    <Modal isOpen={!!jointId} onClose={onClose} title={`${joint.label ?? jointId} — Quick View`} size="sm">
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {/* Status + location + zone state */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <StatusBadge status={joint.status} />
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{joint.location}</span>
          </div>
          {joint.zone_state && (
            <span
              style={{
                fontSize: '10px',
                fontWeight: 700,
                letterSpacing: '0.05em',
                padding: '2px 8px',
                borderRadius: '100px',
                background: joint.zone_state === 'INSPECTING' ? 'var(--accent-dim)' : 'var(--bg-elevated)',
                border: joint.zone_state === 'INSPECTING' ? '1px solid var(--accent-border)' : '1px solid var(--border-subtle)',
                color: joint.zone_state === 'INSPECTING' ? 'var(--accent)' : 'var(--text-muted)',
              }}
            >
              ZONE: {joint.zone_state}
            </span>
          )}
        </div>

        {/* 4 Sensor Channels */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
          {/* Temperature */}
          <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', textAlign: 'center' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 4 }}>
              Temp · DS18B20 (10%)
            </div>
            <div style={{ fontSize: 'var(--text-lg)', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>
              {joint.temperature.toFixed(1)}°C
            </div>
            <div style={{ marginTop: 4 }}><StatusBadge status={tempSt} size="sm" /></div>
          </div>

          {/* Acceleration */}
          <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', textAlign: 'center' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 4 }}>
              Accel · MPU6050 (20%)
            </div>
            <div style={{ fontSize: 'var(--text-lg)', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>
              {joint.vibration.toFixed(2)} m/s²
            </div>
            <div style={{ marginTop: 4 }}><StatusBadge status={accelSt} size="sm" /></div>
          </div>

          {/* Magnetic / Hall */}
          <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', textAlign: 'center' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 4 }}>
              Magnetic · A3144 (30%)
            </div>
            <div style={{ fontSize: 'var(--text-base)', fontFamily: 'var(--font-mono)', fontWeight: 700, color: joint.hall_event ? 'var(--accent)' : 'var(--text-secondary)' }}>
              {joint.hall_event ? 'PULSE' : 'IDLE'}
            </div>
            <div style={{ height: '100%', width: `${score}%`, background: sc, borderRadius: 2 }} />
          </div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: 6 }}>
            Rule-based weighted fusion (0.40 Vision + 0.30 Magnetic + 0.20 Vibration + 0.10 Temp)
          </div>
        </div>

        {/* Mini temp trend */}
        <div>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>Temperature Trend</div>
          <SensorChart
            data={joint.temperatureHistory?.slice(-15) ?? []}
            color="var(--status-high)"
            unit="°C"
            height={100}
            refLines={DEFAULT_THRESHOLDS.temperature}
          />
        </div>

        <button
          className="btn btn-secondary"
          style={{ width: '100%', justifyContent: 'center' }}
          onClick={() => { onClose(); onInspect(jointId); }}
        >
          Full Inspection — {jointId}
          <ChevronRight size={14} />
        </button>
      </div>
    </Modal>
  );
}

// ─── Divider ──────────────────────────────────────────────────────────────────
function Divider() {
  return <div style={{ width: 1, alignSelf: 'stretch', background: 'var(--border-subtle)', margin: '0 4px' }} />;
}

// ─── Status Row Stat ──────────────────────────────────────────────────────────
function StatItem({ label, value, mono = false }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>
        {label}
      </span>
      <span style={{
        fontSize: 'var(--text-sm)',
        fontWeight: 600,
        color: 'var(--text-primary)',
        fontFamily: mono ? 'var(--font-mono)' : undefined,
      }}>
        {value}
      </span>
    </div>
  );
}

// ─── System Health Item ───────────────────────────────────────────────────────
function HealthItem({ label, value, detail, ok }) {
  return (
    <div style={{
      padding: '12px',
      background: 'var(--bg-elevated)',
      border: `1px solid ${ok ? 'var(--status-normal-border)' : 'var(--border-subtle)'}`,
      borderRadius: 'var(--radius)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <div style={{
          width: 6, height: 6, borderRadius: '50%',
          background: ok ? 'var(--status-normal)' : 'var(--text-muted)',
          flexShrink: 0,
        }} />
        <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
          {label}
        </span>
      </div>
      <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: ok ? 'var(--status-normal)' : 'var(--text-muted)' }}>
        {value}
      </div>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: 2 }}>{detail}</div>
    </div>
  );
}

// ─── Dashboard ────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const navigate = useNavigate();
  const { joints, lastUpdated, hardwareStatus, apiConnected, visionData, cameraStatus } = useLiveSensorData();
  const { activeAlerts, summary } = useAlerts();

  const [quickViewId, setQuickViewId] = useState(null);
  const [selectedChartJoint, setSelectedChartJoint] = useState('J01');

  const overallStatus = getOverallStatus(joints);
  const allJoints     = Object.values(joints);

  // Worst-case readings for top metric cards
  const worstTemp   = Math.max(...allJoints.map(j => j.temperature));
  const worstAccel  = Math.max(...allJoints.map(j => j.vibration));
  const avgTemp     = allJoints.reduce((s, j) => s + j.temperature, 0) / allJoints.length;
  const avgAccel    = allJoints.reduce((s, j) => s + j.vibration,   0) / allJoints.length;
  const tempStatus  = getValueStatus(worstTemp,  DEFAULT_THRESHOLDS.temperature);
  const accelStatus = getValueStatus(worstAccel, DEFAULT_THRESHOLDS.vibration);
  const connStatus  = MOCK_SYSTEM_STATUS.esp32.connected ? STATUS.NORMAL : STATUS.HIGH;

  // Selected joint for live charts on dashboard
  const chartJoint        = joints[selectedChartJoint] || joints['J01'] || allJoints[0];
  const monitoringDuration = calcDuration(MOCK_SYSTEM_STATUS.monitoringStarted);

  const recentEvents = [...MOCK_EVENTS].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)).slice(0, 4);

  return (
    <div className="page-content">

      {/* ── 1. Alert Banner ──────────────────────────────── */}
      <AlertBanner activeAlerts={activeAlerts} />

      {/* ── Simulation Controls Bar ──────────────────────── */}
      <SimulationControls />

      {/* ── 2. Overall Belt Status Row ───────────────────── */}
      <div
        className="card"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
          marginBottom: '16px',
          padding: '14px 20px',
        }}
      >
        {/* Left: core stats */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', align: 'center', gap: '10px' }}>
            <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>Belt Status</span>
            <StatusBadge status={overallStatus} />
          </div>
          <Divider />
          <StatItem label="Duration"       value={monitoringDuration} />
          <Divider />
          <StatItem label="Active Alerts"  value={summary.totalActive} />
          <Divider />
          <StatItem label="Joints"         value={`${JOINT_IDS.length} monitored`} />
          <Divider />
          <StatItem label="Last Update"    value={lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : '—'} mono />
        </div>

        {/* Right: about button */}
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => navigate('/about')}
        >
          About Project
        </button>
      </div>

      {/* ── 3. Sensor Metric Cards ───────────────────────── */}
      <div className="grid-4" style={{ marginBottom: '20px' }}>
        <MetricCard
          icon={<Thermometer size={15} />}
          label="Peak Temperature"
          value={worstTemp}
          unit="°C"
          status={tempStatus}
          subtext={`Avg across joints: ${avgTemp.toFixed(1)}°C`}
        />
        <MetricCard
          icon={<Activity size={15} />}
          label="Peak Accel · MPU6050"
          value={worstAccel}
          unit="m/s²"
          status={accelStatus}
          subtext={`Avg: ${avgAccel.toFixed(2)} m/s²  (raw accel, mock)`}
        />
        <MetricCard
          icon={<Link2 size={15} />}
          label="Belt Condition"
          value={overallStatus}
          unit=""
          status={overallStatus}
          subtext="Worst-case joint status"
        />
        <MetricCard
          icon={<Radio size={15} />}
          label="Hardware Node (UNO)"
          value={hardwareStatus?.connection_status === 'CONNECTED' ? 'Connected' : (apiConnected ? `Ready (${hardwareStatus?.port || 'COM5'})` : 'Offline')}
          unit=""
          status={hardwareStatus?.connection_status === 'CONNECTED' ? STATUS.NORMAL : STATUS.HIGH}
          subtext={`${hardwareStatus?.port || 'COM5'} @ ${hardwareStatus?.baud || 9600} Baud · USB Serial Gateway`}
        />
      </div>

      {/* ── Live YOLO Camera Vision Inspector Component ── */}
      <div style={{ marginBottom: '20px' }}>
        <CameraVision visionData={visionData} cameraStatus={cameraStatus} activeJoint={selectedChartJoint} />
      </div>

      {/* ── 4. Live Charts with Joint Selector (J01 - J03) ── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-secondary)' }}>
            Live Telemetry:
          </span>
          {JOINT_IDS.map((id) => (
            <button
              key={id}
              onClick={() => setSelectedChartJoint(id)}
              className={`btn btn-sm ${selectedChartJoint === id ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '4px 14px', fontSize: '11px', fontWeight: selectedChartJoint === id ? 700 : 500 }}
            >
              {id}
            </button>
          ))}
        </div>
        <button className="btn btn-ghost btn-sm" onClick={() => navigate('/joints')}>
          Joint Monitoring Page →
        </button>
      </div>

      <div className="grid-2" style={{ marginBottom: '20px' }}>
        <div className="card">
          <div className="section-header">
            <div>
              <span className="section-title">Temperature — {selectedChartJoint}</span>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginLeft: 8 }}>DS18B20</span>
            </div>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--status-high)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {chartJoint?.temperature?.toFixed(1)} °C
            </span>
          </div>
          <SensorChart
            data={chartJoint?.temperatureHistory ?? []}
            color="var(--status-high)"
            unit="°C"
            height={160}
            refLines={DEFAULT_THRESHOLDS.temperature}
          />
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: 8 }}>
            ⚠ Reference lines use PLACEHOLDER thresholds
          </div>
        </div>

        <div className="card">
          <div className="section-header">
            <div>
              <span className="section-title">Acceleration — {selectedChartJoint}</span>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)', marginLeft: 8 }}>MPU6050 · raw accel · mock</span>
            </div>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {chartJoint?.vibration?.toFixed(2)} m/s²
            </span>
          </div>
          <SensorChart
            data={chartJoint?.vibrationHistory ?? []}
            color="var(--accent)"
            unit="m/s²"
            height={160}
            refLines={DEFAULT_THRESHOLDS.vibration}
          />
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: 8 }}>
            ⚠ Reference lines use PLACEHOLDER thresholds
          </div>
        </div>
      </div>

      {/* ── 5. Joint Condition Summary (3 Joints) ─────────── */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="section-header">
          <div>
            <span className="section-title">Joint Condition Summary ({JOINT_IDS.length} Joints)</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: 8 }}>
              Click any joint card for quick inspection modal
            </span>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/joints')}>
            View All Details →
          </button>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
          {JOINT_IDS.map(id => {
            const j = joints[id];
            if (!j) return null;
            const sc = scoreColor(j.conditionScore ?? 50);
            const score = j.conditionScore ?? 50;
            const tempSt  = getValueStatus(j.temperature, DEFAULT_THRESHOLDS.temperature);
            const accelSt = getValueStatus(j.vibration,   DEFAULT_THRESHOLDS.vibration);

            return (
              <div
                key={id}
                onClick={() => setQuickViewId(id)}
                style={{
                  padding: '14px',
                  background: 'var(--bg-elevated)',
                  borderRadius: 'var(--radius)',
                  cursor: 'pointer',
                  borderLeft: `3px solid ${
                    j.status === STATUS.HIGH   ? 'var(--status-high)'   :
                    j.status === STATUS.MEDIUM ? 'var(--status-medium)' :
                    'var(--status-normal)'
                  }`,
                  borderTop: '1px solid var(--border-subtle)',
                  borderRight: '1px solid var(--border-subtle)',
                  borderBottom: '1px solid var(--border-subtle)',
                  transition: 'background var(--transition), transform var(--transition)',
                }}
                onMouseOver={e => {
                  e.currentTarget.style.background = 'var(--bg-hover)';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }}
                onMouseOut={e => {
                  e.currentTarget.style.background = 'var(--bg-elevated)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
                title={`Click to open quick inspection for ${id}`}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <span style={{ fontWeight: 700, fontSize: 'var(--text-base)', color: 'var(--text-primary)' }}>{id}</span>
                  <StatusBadge status={j.status} />
                </div>

                {/* Score bar */}
                <div style={{ marginBottom: 6 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Health Score</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', fontWeight: 700, color: sc }}>{score}</span>
                  </div>
                  <div style={{ height: 5, background: 'var(--bg-overlay)', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${score}%`, background: sc, borderRadius: 3 }} />
                  </div>
                </div>

                {/* Parameter badges */}
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '11px', color: 'var(--text-muted)' }}>
                    <Thermometer size={11} />
                    <span style={{ fontFamily: 'var(--font-mono)' }}>{j.temperature.toFixed(1)}°C</span>
                    <StatusBadge status={tempSt} size="sm" showDot={false} />
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '11px', color: 'var(--text-muted)' }}>
                    <Activity size={11} />
                    <span style={{ fontFamily: 'var(--font-mono)' }}>{j.vibration.toFixed(2)}</span>
                    <StatusBadge status={accelSt} size="sm" showDot={false} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── 7 & 8. Active Alerts + System Health ─────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>

        {/* Active Alerts */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="section-header">
            <span className="section-title">Active Alerts</span>
            <button className="btn btn-ghost btn-sm" onClick={() => navigate('/alerts')}>View All →</button>
          </div>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
            <span className="badge badge-high">{summary.highCount} HIGH</span>
            <span className="badge badge-medium">{summary.mediumCount} MEDIUM</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {activeAlerts.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)', textAlign: 'center', padding: '24px 0' }}>
                No active alerts
              </div>
            ) : (
              activeAlerts.slice(0, 5).map(alert => (
                <div
                  key={alert.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '9px 10px',
                    background: 'var(--bg-elevated)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius)',
                  }}
                >
                  <StatusBadge status={alert.level} size="sm" showDot={false} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {alert.jointId} — {alert.parameter === 'temperature' ? 'Temperature' : 'Acceleration'}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {alert.reading} {alert.unit} · {calcDuration(alert.startedAt)} ago
                    </div>
                  </div>
                  {!alert.acknowledged && (
                    <span style={{ fontSize: '9px', color: 'var(--status-medium)', fontWeight: 700, letterSpacing: '0.05em' }}>NEW</span>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* System Health */}
        <div className="card">
          <div className="section-header">
            <span className="section-title">System Health</span>
            <button className="btn btn-ghost btn-sm" onClick={() => navigate('/settings')}>Details →</button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <HealthItem label="ESP32 Controller" value={MOCK_SYSTEM_STATUS.esp32.connected ? 'Connected' : 'Disconnected'} detail={MOCK_SYSTEM_STATUS.esp32.ipAddress} ok={MOCK_SYSTEM_STATUS.esp32.connected} />
            <HealthItem label="DS18B20 Temp Sensor" value="Operational" detail="Temperature readings OK" ok />
            <HealthItem label="MPU6050 IMU" value="Operational" detail="Acceleration readings OK" ok />
            <HealthItem label="FastAPI Backend" value="Not Yet Active" detail="Planned — future integration" ok={false} />
            <HealthItem label="Database" value="Not Yet Active" detail="Planned — future integration" ok={false} />
          </div>
        </div>
      </div>

      {/* ── 9. Recent Events ─────────────────────────────── */}
      <div className="card">
        <div className="section-header">
          <span className="section-title">Recent Events</span>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>View History →</button>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          {recentEvents.map((evt, i) => {
            const evtColor =
              evt.level === STATUS.HIGH   ? 'var(--status-high)'   :
              evt.level === STATUS.MEDIUM ? 'var(--status-medium)' :
              'var(--accent)';
            return (
              <div
                key={evt.id}
                style={{ display: 'flex', gap: '14px', paddingBottom: i < recentEvents.length - 1 ? '12px' : 0 }}
              >
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
                  <div style={{ width: 7, height: 7, borderRadius: '50%', background: evtColor, marginTop: 5 }} />
                  {i < recentEvents.length - 1 && (
                    <div style={{ width: 1, flex: 1, background: 'var(--border-subtle)', marginTop: 4 }} />
                  )}
                </div>
                <div style={{ flex: 1, paddingBottom: i < recentEvents.length - 1 ? 4 : 0 }}>
                  <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-primary)', marginBottom: 2 }}>
                    {evt.description}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {new Date(evt.timestamp).toLocaleString()}{evt.jointId ? ` · ${evt.jointId}` : ''}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Joint Quick-View Modal ───────────────────────── */}
      <JointQuickModal
        jointId={quickViewId}
        joints={joints}
        onClose={() => setQuickViewId(null)}
        onInspect={() => navigate('/joints')}
      />
    </div>
  );
}
