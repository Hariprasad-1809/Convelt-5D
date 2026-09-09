import { useState } from 'react';
import { Save, AlertTriangle, RefreshCw } from 'lucide-react';
import { useSimulation } from '../context/SimulationContext';
import { DEFAULT_THRESHOLDS, SYSTEM_CONFIG, SENSOR_INFO, API_BASE } from '../data/constants';
import { MOCK_SYSTEM_STATUS } from '../data/mockData';

function SettingRow({ label, value, children, note }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', padding: '14px 0', borderBottom: '1px solid var(--border-subtle)', gap: '16px', flexWrap: 'wrap' }}>
      <div style={{ flex: '1 1 200px' }}>
        <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>{label}</div>
        {note && <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{note}</div>}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {children ?? <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>{value}</span>}
      </div>
    </div>
  );
}

function InfoRow({ label, value, mono = false }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border-subtle)', fontSize: 'var(--text-sm)' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: 'var(--text-primary)', fontFamily: mono ? 'var(--font-mono)' : undefined, fontSize: mono ? 'var(--text-xs)' : undefined }}>{value}</span>
    </div>
  );
}

function ThresholdInput({ label, value, onChange, unit }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <label style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>{label}</label>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <input
          type="number"
          className="input"
          value={value}
          onChange={e => onChange(parseFloat(e.target.value) || 0)}
          step="0.5"
          style={{ width: 100 }}
          aria-label={label}
        />
        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{unit}</span>
      </div>
    </div>
  );
}

export default function Settings() {
  const { apiConnected, simStatus } = useSimulation();
  const [thresholds, setThresholds] = useState(DEFAULT_THRESHOLDS);
  const [refreshRate, setRefreshRate] = useState(SYSTEM_CONFIG.defaultRefreshRateMs / 1000);
  const [saved, setSaved] = useState(false);

  function setTemp(key, val) {
    setThresholds(t => ({ ...t, temperature: { ...t.temperature, [key]: val } }));
  }
  function setVib(key, val) {
    setThresholds(t => ({ ...t, vibration: { ...t.vibration, [key]: val } }));
  }

  function handleSave() {
    // In a real app: update global config context / localStorage / API call
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  function handleReset() {
    setThresholds(DEFAULT_THRESHOLDS);
    setRefreshRate(SYSTEM_CONFIG.defaultRefreshRateMs / 1000);
  }

  const esp32 = MOCK_SYSTEM_STATUS.esp32;

  return (
    <div className="page-content" style={{ maxWidth: 860 }}>
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
        <p className="page-subtitle">Configure sensor thresholds, display options, and system information</p>
      </div>

      {/* Threshold Warning */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '12px 16px', background: 'var(--status-medium-dim)', border: '1px solid var(--status-medium-border)', borderRadius: 'var(--radius)', marginBottom: '20px', fontSize: 'var(--text-sm)' }}>
        <AlertTriangle size={14} color="var(--status-medium)" style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <span style={{ fontWeight: 600, color: 'var(--status-medium)' }}>PLACEHOLDER Thresholds: </span>
          <span style={{ color: 'var(--text-secondary)' }}>
            The values below are demonstration-only placeholder values, not validated limits. Replace with real values derived from field testing with your prototype sensors before using in production.
          </span>
        </div>
      </div>

      {/* Sensor Thresholds */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="section-header">
          <span className="section-title">Sensor Thresholds ⚠ PLACEHOLDER</span>
          <button className="btn btn-ghost btn-sm" onClick={handleReset}>
            <RefreshCw size={12} /> Reset Defaults
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          {/* Temperature */}
          <div>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: 6 }}>
              Temperature (DS18B20) · °C
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <ThresholdInput
                label="Normal max (≤ NORMAL)"
                value={thresholds.temperature.normalMax}
                onChange={v => setTemp('normalMax', v)}
                unit="°C"
              />
              <ThresholdInput
                label="Medium max (≤ MEDIUM)"
                value={thresholds.temperature.mediumMax}
                onChange={v => setTemp('mediumMax', v)}
                unit="°C"
              />
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--status-high)', paddingLeft: 2 }}>
                HIGH: &gt; {thresholds.temperature.mediumMax}°C (auto)
              </div>
            </div>
          </div>

          {/* Vibration */}
          <div>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: 6 }}>
              Vibration (MPU6050) · m/s²
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <ThresholdInput
                label="Normal max (≤ NORMAL)"
                value={thresholds.vibration.normalMax}
                onChange={v => setVib('normalMax', v)}
                unit="m/s²"
              />
              <ThresholdInput
                label="Medium max (≤ MEDIUM)"
                value={thresholds.vibration.mediumMax}
                onChange={v => setVib('mediumMax', v)}
                unit="m/s²"
              />
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--status-high)', paddingLeft: 2 }}>
                HIGH: &gt; {thresholds.vibration.mediumMax} m/s² (auto)
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Dashboard Settings */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="section-title" style={{ marginBottom: '12px' }}>Dashboard Settings</div>
        <SettingRow label="Live Data Refresh Rate" note="How often sensor readings update on screen">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <select
              className="select"
              style={{ width: 'auto' }}
              value={refreshRate}
              onChange={e => setRefreshRate(Number(e.target.value))}
              aria-label="Refresh rate"
            >
              {[1, 2, 3, 5, 10].map(s => <option key={s} value={s}>{s}s</option>)}
            </select>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>seconds</span>
          </div>
        </SettingRow>
      </div>

      {/* Save Button */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '24px' }}>
        <button className="btn btn-primary" onClick={handleSave}>
          <Save size={14} />
          {saved ? 'Saved!' : 'Save Settings'}
        </button>
        <button className="btn btn-ghost" onClick={handleReset}>Reset to Defaults</button>
      </div>

      {/* System Information */}
      <div className="grid-2">
        <div className="card">
          <div className="section-title" style={{ marginBottom: '12px' }}>System Information</div>
          <InfoRow label="System Status" value={apiConnected ? 'Online (FastAPI Connected)' : 'Offline / Standby'} />
          <InfoRow label="Backend API" value={apiConnected ? `FastAPI Active · Cycle ${simStatus?.current_cycle ?? 0}` : 'Server Unreachable'} />
          <InfoRow label="Backend URL" value={API_BASE} mono />
          <InfoRow label="Database" value="SQLite / SQLAlchemy (backend/database/jointguard.db)" />
          <InfoRow label="History Points" value={`${SYSTEM_CONFIG.historyPointCount} readings`} />
        </div>

        <div className="card">
          <div className="section-title" style={{ marginBottom: '12px' }}>ESP32 Information</div>
          <InfoRow label="Model" value={esp32.model} />
          <InfoRow label="Firmware" value={esp32.firmware} mono />
          <InfoRow label="IP Address" value={esp32.ipAddress} mono />
          <InfoRow label="Signal Strength" value={`${esp32.signalStrength} dBm`} mono />
          <InfoRow label="Uptime" value={esp32.uptime} />
          <InfoRow label="Connected" value={esp32.connected ? 'Yes' : 'No'} />
        </div>

        <div className="card">
          <div className="section-title" style={{ marginBottom: '12px' }}>Sensor Specifications</div>
          {Object.values(SENSOR_INFO).map(s => (
            <div key={s.name} style={{ paddingBottom: '12px', marginBottom: '12px', borderBottom: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 'var(--text-sm)', marginBottom: 4 }}>{s.name} — {s.label}</div>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', lineHeight: 1.7 }}>
                <div>Unit: {s.unit}</div>
                <div>Range: {s.range}</div>
                <div>Resolution: {s.resolution}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Future ML */}
        <div className="card" style={{ border: '1px dashed var(--border-default)' }}>
          <div className="section-title" style={{ marginBottom: '12px', color: 'var(--accent)' }}>Future: ML Configuration</div>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', lineHeight: 1.8 }}>
            When the ML model is integrated, settings will appear here:
            <ul style={{ paddingLeft: '16px', marginTop: '8px', lineHeight: 2 }}>
              <li>Model endpoint URL</li>
              <li>Prediction refresh interval</li>
              <li>Confidence threshold for alerts</li>
              <li>Risk level display settings</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
