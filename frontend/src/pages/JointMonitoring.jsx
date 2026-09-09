import { useState } from 'react';
import { Thermometer, Activity, ChevronRight, AlertTriangle } from 'lucide-react';
import { useLiveSensorData } from '../hooks/useLiveSensorData';
import { getValueStatus, computeConditionScore } from '../data/mockData';
import { STATUS, DEFAULT_THRESHOLDS, JOINT_IDS } from '../data/constants';
import StatusBadge from '../components/ui/StatusBadge';
import SensorChart from '../components/ui/SensorChart';
import Modal from '../components/ui/Modal';

// Condition is expressed as the joint status level (NORMAL/MEDIUM/HIGH), not Good/Fair/Poor.
function scoreColor(score) {
  if (score >= 75) return 'var(--status-normal)';
  if (score >= 50) return 'var(--status-medium)';
  return 'var(--status-high)';
}

function ThresholdRow({ label, value, unit, thresholds }) {
  const status = getValueStatus(value, thresholds);
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '8px 0',
      borderBottom: '1px solid var(--border-subtle)',
      fontSize: 'var(--text-sm)',
    }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
          {value.toFixed(2)} {unit}
        </span>
        <StatusBadge status={status} size="sm" />
      </div>
    </div>
  );
}

function InspectModal({ joint, onClose }) {
  if (!joint) return null;
  const tempStatus  = getValueStatus(joint.temperature, DEFAULT_THRESHOLDS.temperature);
  const accelStatus = getValueStatus(joint.vibration,   DEFAULT_THRESHOLDS.vibration);
  const score       = joint.conditionScore ?? computeConditionScore(joint.temperature, joint.vibration);
  const sc          = scoreColor(score);

  return (
    <Modal isOpen={!!joint} onClose={onClose} title={`Inspect ${joint.label ?? joint.id}`} size="lg">
      {/* Current Readings */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div className="section-title">Current Readings (4-Channel)</div>
          {joint.zone_state && (
            <span
              style={{
                fontSize: '11px',
                fontWeight: 700,
                letterSpacing: '0.06em',
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
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
          {/* Temp */}
          <div className="card-elevated" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Temperature · DS18B20
            </div>
            <div style={{ fontSize: 'var(--text-2xl)', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>
              {joint.temperature.toFixed(1)}
              <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)', fontWeight: 400 }}> °C</span>
            </div>
            <div style={{ marginTop: 6 }}><StatusBadge status={tempStatus} /></div>
          </div>

          {/* Accel */}
          <div className="card-elevated" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Accel · MPU6050
            </div>
            <div style={{ fontSize: 'var(--text-2xl)', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>
              {joint.vibration.toFixed(2)}
              <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)', fontWeight: 400 }}> m/s²</span>
            </div>
            <div style={{ marginTop: 6 }}><StatusBadge status={accelStatus} /></div>
          </div>

          {/* Magnetic / Hall */}
          <div className="card-elevated" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Magnetic · A3144 Pulse
            </div>
            <div style={{ fontSize: 'var(--text-2xl)', fontFamily: 'var(--font-mono)', fontWeight: 700, color: joint.hall_event ? 'var(--accent)' : 'var(--text-secondary)' }}>
              {joint.hall_event ? 'PULSE' : 'IDLE'}
            </div>
            <div style={{ marginTop: 6, fontSize: '11px', color: 'var(--text-muted)' }}>
              {joint.magnetic_value != null ? `${joint.magnetic_value.toFixed(2)} mT` : 'Passing Sensor'}
            </div>
          </div>

          {/* Vision */}
          <div className="card-elevated" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Vision Score (Simulated)
            </div>
            <div style={{ fontSize: 'var(--text-2xl)', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent)' }}>
              {joint.vision_score != null ? joint.vision_score.toFixed(0) : '85'}
              <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)', fontWeight: 400 }}> / 100</span>
            </div>
            <div style={{ marginTop: 6, fontSize: '11px', color: 'var(--accent)' }}>
              Phase 1 Injected
            </div>
          </div>
        </div>
      </div>

      {/* Threshold Comparison */}
      <div style={{ marginBottom: '20px' }}>
        <div className="section-title" style={{ marginBottom: '12px' }}>
          Threshold Comparison
          <span style={{ fontSize: 'var(--text-xs)', fontWeight: 400, color: 'var(--status-medium)', marginLeft: 8 }}>⚠ Demonstration thresholds</span>
        </div>
        <div className="card-elevated">
          <ThresholdRow label="Temperature · DS18B20" value={joint.temperature} unit="°C"   thresholds={DEFAULT_THRESHOLDS.temperature} />
          <ThresholdRow label="Acceleration · MPU6050"  value={joint.vibration}   unit="m/s²" thresholds={DEFAULT_THRESHOLDS.vibration} />
          <div style={{ paddingTop: '8px', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            Temp: NORMAL ≤{DEFAULT_THRESHOLDS.temperature.normalMax}°C · MEDIUM ≤{DEFAULT_THRESHOLDS.temperature.mediumMax}°C · HIGH &gt;{DEFAULT_THRESHOLDS.temperature.mediumMax}°C
          </div>
        </div>
      </div>

      {/* Condition Analysis */}
      <div style={{ marginBottom: '20px' }}>
        <div className="section-title" style={{ marginBottom: '12px' }}>
          Weighted Fusion Health Score
        </div>
        <div className="card-elevated" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ textAlign: 'center', flexShrink: 0 }}>
            <div style={{ fontSize: 'var(--text-3xl)', fontWeight: 700, fontFamily: 'var(--font-mono)', color: sc }}>
              {score}
            </div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>/ 100</div>
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Risk Level:</span>
              <StatusBadge status={joint.status} />
            </div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              Rule-based weighted fusion: 40% Vision + 30% Magnetic + 20% Vibration + 10% Temperature.
              Explicitly deterministic in Phase 1 (ML/DL model postponed to Phase 2).
            </div>
          </div>
        </div>
      </div>

      {/* Temperature Trend */}
      <div style={{ marginBottom: '20px' }}>
        <div className="section-title" style={{ marginBottom: '12px' }}>Temperature Trend</div>
        <div className="card-elevated">
          <SensorChart
            data={joint.temperatureHistory ?? []}
            color="var(--status-high)"
            unit="°C"
            height={140}
            refLines={DEFAULT_THRESHOLDS.temperature}
          />
        </div>
      </div>

      {/* Acceleration Trend */}
      <div>
        <div className="section-title" style={{ marginBottom: '12px' }}>Acceleration Trend
          <span style={{ fontSize: '10px', fontWeight: 400, color: 'var(--text-muted)', marginLeft: 8 }}>MPU6050 · raw accel magnitude · mock</span>
        </div>
        <div className="card-elevated">
          <SensorChart
            data={joint.vibrationHistory ?? []}
            color="var(--accent)"
            unit="m/s²"
            height={140}
            refLines={DEFAULT_THRESHOLDS.vibration}
          />
        </div>
      </div>

      {/* Future ML placeholder */}
      <div style={{ marginTop: '20px', padding: '12px', background: 'var(--accent-dim)', border: '1px solid var(--accent-border)', borderRadius: 'var(--radius)', fontSize: 'var(--text-xs)', color: 'var(--accent)' }}>
        <strong>Future ML Integration:</strong> Predictive maintenance risk score and deterioration forecast will appear here once the ML model is integrated.
      </div>
    </Modal>
  );
}

export default function JointMonitoring() {
  const { joints } = useLiveSensorData();
  const [selectedJoint, setSelectedJoint] = useState(null);

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Joint Monitoring</h1>
        <p className="page-subtitle">Individual status and sensor readings for each conveyor belt joint</p>
      </div>

      <div style={{ fontSize: 'var(--text-xs)', color: 'var(--status-medium)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: 6 }}>
        <AlertTriangle size={12} />
        Thresholds shown are PLACEHOLDER values — not validated limits. Replace with real sensor data after testing.
      </div>

      <div className="grid-3">
        {(Object.keys(joints).length > 0 ? Object.keys(joints) : JOINT_IDS).map(id => {
          const joint = joints[id];
          if (!joint) return null;

          const tempSt  = getValueStatus(joint.temperature, DEFAULT_THRESHOLDS.temperature);
          const accelSt = getValueStatus(joint.vibration,   DEFAULT_THRESHOLDS.vibration);
          const score   = joint.conditionScore ?? computeConditionScore(joint.temperature, joint.vibration);
          const sc      = scoreColor(score);

          return (
            <div
              key={id}
              className="card"
              style={{
                borderTop: `3px solid ${
                  joint.status === STATUS.HIGH   ? 'var(--status-high)'   :
                  joint.status === STATUS.MEDIUM ? 'var(--status-medium)' :
                  'var(--status-normal)'
                }`,
              }}
            >
              {/* Header */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div>
                  <div style={{ fontSize: 'var(--text-lg)', fontWeight: 700, color: 'var(--text-primary)' }}>{id}</div>
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{joint.location}</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {joint.zone_state && (
                    <span
                      style={{
                        fontSize: '9px',
                        fontWeight: 700,
                        padding: '2px 6px',
                        borderRadius: '4px',
                        background: joint.zone_state === 'INSPECTING' ? 'var(--accent-dim)' : 'var(--bg-elevated)',
                        color: joint.zone_state === 'INSPECTING' ? 'var(--accent)' : 'var(--text-muted)',
                        border: '1px solid var(--border-subtle)',
                      }}
                    >
                      {joint.zone_state}
                    </span>
                  )}
                  <StatusBadge status={joint.status} />
                </div>
              </div>

              {/* Readings */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '16px' }}>
                <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <Thermometer size={12} color="var(--text-muted)" />
                    <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontWeight: 600 }}>TEMP · DS18B20</span>
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-base)', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {joint.temperature.toFixed(1)}°C
                  </div>
                  <StatusBadge status={tempSt} size="sm" />
                </div>
                <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <Activity size={12} color="var(--text-muted)" />
                    <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontWeight: 600 }}>ACCEL · MPU6050</span>
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-base)', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {joint.vibration.toFixed(2)} m/s²
                  </div>
                  <StatusBadge status={accelSt} size="sm" />
                </div>
              </div>

              {/* Condition Score */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', marginBottom: '16px' }}>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>Condition Score <span style={{ color: 'var(--status-medium)' }}>⚠PLACEHOLDER</span></span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: sc }}>{score}</span>
                  <StatusBadge status={joint.status} size="sm" />
                </div>
              </div>

              {/* Score bar */}
              <div style={{ height: 4, background: 'var(--bg-overlay)', borderRadius: 2, marginBottom: '16px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${score}%`, background: sc, borderRadius: 2, transition: 'width 0.5s ease' }} />
              </div>

              {/* Inspect button */}
              <button
                className="btn btn-secondary"
                style={{ width: '100%', justifyContent: 'center' }}
                onClick={() => setSelectedJoint(joint)}
                aria-label={`Inspect ${id}`}
              >
                Inspect {id}
                <ChevronRight size={14} />
              </button>
            </div>
          );
        })}
      </div>

      <InspectModal joint={selectedJoint} onClose={() => setSelectedJoint(null)} />
    </div>
  );
}
