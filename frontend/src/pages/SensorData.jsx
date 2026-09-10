import { useState, useMemo } from 'react';
import { Download, Filter, Wifi, WifiOff, CheckCircle, AlertCircle } from 'lucide-react';
import { useSimulation } from '../context/SimulationContext';
import { getValueStatus } from '../data/mockData';
import { DEFAULT_THRESHOLDS, JOINT_IDS, SENSOR_INFO } from '../data/constants';
import StatusBadge from '../components/ui/StatusBadge';
import SensorChart from '../components/ui/SensorChart';
import VisionResultPanel from '../components/ui/VisionResultPanel';

const TIME_RANGES = [
  { label: '1 min', points: 15 },
  { label: '5 min', points: 30 },
  { label: 'All',   points: 30 },
];

function SensorStatusRow({ label, sensor, connected }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border-subtle)' }}>
      <div>
        <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>{sensor?.name ?? label}</div>
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{sensor?.description}</div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 'var(--text-xs)' }}>
        {connected
          ? <><CheckCircle size={13} color="var(--status-normal)" /><span style={{ color: 'var(--status-normal)' }}>Active</span></>
          : <><AlertCircle size={13} color="var(--status-high)" /><span style={{ color: 'var(--status-high)' }}>Fault</span></>
        }
      </div>
    </div>
  );
}

export default function SensorData() {
  const { joints = {}, lastUpdated, apiConnected, simStatus, hardwareStatus, wsStatus = 'CONNECTING', rawSerialLogs = [] } = useSimulation();


  const [selectedJoint, setSelectedJoint] = useState('J01');
  const [selectedSensor, setSelectedSensor] = useState('all');
  const [timeRange, setTimeRange] = useState(1);
  const [statusFilter, setStatusFilter] = useState('all');

  const joint = joints[selectedJoint] ?? Object.values(joints)[0] ?? {};

  // Build raw readings table from live history
  const rawReadings = useMemo(() => {
    const hist = joint?.temperatureHistory ?? [];
    const vhist = joint?.vibrationHistory ?? [];
    return hist.slice(-20).map((tp, i) => ({
      id: `${selectedJoint}-${tp.timestamp}-${i}`,
      timestamp: tp.timestamp,
      time: tp.time,
      jointId: selectedJoint,
      temperature: tp.value,
      vibration: vhist[vhist.length - hist.slice(-20).length + i]?.value ?? joint?.vibration ?? 0,
      hall_event: joint?.hall_event ? 'PULSE' : 'IDLE',
      vision_score: joint?.vision_score ?? 85,
    })).reverse();
  }, [joint, selectedJoint]);

  const filteredReadings = rawReadings.filter(row => {
    if (statusFilter === 'all') return true;
    const tempSt = getValueStatus(row.temperature, DEFAULT_THRESHOLDS.temperature);
    const vibSt  = getValueStatus(row.vibration,   DEFAULT_THRESHOLDS.vibration);
    return tempSt === statusFilter || vibSt === statusFilter;
  });

  const timeLimit = TIME_RANGES[timeRange]?.points ?? 30;
  const tempHist = joint?.temperatureHistory?.slice(-timeLimit) ?? [];
  const vibHist  = joint?.vibrationHistory?.slice(-timeLimit) ?? [];

  // Aggregates
  const tempVals = tempHist.map(d => d.value).filter(v => typeof v === 'number');
  const vibVals  = vibHist.map(d => d.value).filter(v => typeof v === 'number');
  const agg = {
    temp: {
      min: tempVals.length ? Math.min(...tempVals).toFixed(1) : '—',
      max: tempVals.length ? Math.max(...tempVals).toFixed(1) : '—',
      avg: tempVals.length ? (tempVals.reduce((a, b) => a + b, 0) / tempVals.length).toFixed(1) : '—',
    },
    vib: {
      min: vibVals.length ? Math.min(...vibVals).toFixed(3) : '—',
      max: vibVals.length ? Math.max(...vibVals).toFixed(3) : '—',
      avg: vibVals.length ? (vibVals.reduce((a, b) => a + b, 0) / vibVals.length).toFixed(3) : '—',
    },
  };

  const handleCSVExport = () => {
    const headers = ['Timestamp', 'Time', 'Joint', 'Temperature (°C)', 'Vibration (m/s²)', 'Hall Event', 'Vision Score'];
    const csvRows = [headers, ...rawReadings.map(r => [r.timestamp, r.time, r.jointId, r.temperature, r.vibration, r.hall_event, r.vision_score])];
    const csv = csvRows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `jointguard_telemetry_${selectedJoint}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const availableJointIds = Object.keys(joints).length > 0 ? Object.keys(joints) : JOINT_IDS;
  const isArduinoConnected = (hardwareStatus?.serial_status === 'CONNECTED' || hardwareStatus?.connection_status === 'CONNECTED');

  return (
    <div className="page-content">
      <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h1 className="page-title">Sensor Data</h1>
          <p className="page-subtitle">Live multi-sensor readings, trend charts, and raw telemetry logs</p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={handleCSVExport}>
          <Download size={13} /> Export CSV
        </button>
      </div>

      {/* Connection & Hardware Serial Status Bar */}
      <div className="card" style={{ marginBottom: '20px', display: 'flex', flexWrap: 'wrap', gap: '20px', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isArduinoConnected ? (
            <Wifi size={16} color="var(--status-normal)" />
          ) : (
            <WifiOff size={16} color="var(--status-high)" />
          )}
          <div>
            <div style={{ fontSize: 'var(--text-xs)', fontWeight: 700, color: isArduinoConnected ? 'var(--status-normal)' : 'var(--status-high)' }}>
              ARDUINO UNO {isArduinoConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
              {hardwareStatus?.port || 'COM5'} @ {hardwareStatus?.baud || 9600} Baud · USB Serial Gateway
            </div>
          </div>
        </div>

        <div style={{ width: 1, height: 36, background: 'var(--border-subtle)' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {wsStatus === 'CONNECTED' ? <CheckCircle size={14} color="var(--status-normal)" /> : <AlertCircle size={14} color="var(--status-high)" />}
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>WebSocket Gateway:</span>{' '}
            <span style={{ color: wsStatus === 'CONNECTED' ? 'var(--status-normal)' : 'var(--status-high)', fontWeight: 600 }}>
              {wsStatus === 'CONNECTED' ? 'CONNECTED' : wsStatus === 'CONNECTING' ? 'CONNECTING...' : wsStatus === 'RECONNECTING' ? 'RECONNECTING...' : 'DISCONNECTED'}
            </span>
          </div>
        </div>

        <div style={{ width: 1, height: 36, background: 'var(--border-subtle)' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {apiConnected ? <CheckCircle size={14} color="var(--status-normal)" /> : <AlertCircle size={14} color="var(--status-high)" />}
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>FastAPI Backend:</span> {apiConnected ? 'Online' : 'Offline'}
          </div>
        </div>

        <div style={{ width: 1, height: 36, background: 'var(--border-subtle)' }} />

        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
          <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Active Node:</span> {hardwareStatus?.device_id || 'ARDUINO_UNO_01'}
        </div>

        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginLeft: 'auto', fontFamily: 'var(--font-mono)' }}>
          Last Hardware Update: {hardwareStatus?.lastUpdated ? new Date(hardwareStatus.lastUpdated).toLocaleTimeString() : (lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : '—')}
        </div>
      </div>

      {/* Live Arduino UNO Hardware Telemetry & Raw Serial Monitor Box */}
      <div className="card" style={{ marginBottom: '20px', padding: '16px' }}>

        <div className="section-header" style={{ marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: isArduinoConnected ? 'var(--status-normal)' : 'var(--status-high)' }} />
            <span className="section-title">Arduino UNO Live Hardware Telemetry & Serial Monitor Stream ({hardwareStatus?.port || 'COM5'})</span>
          </div>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            Real-Time USB Gateway Stream
          </span>
        </div>

        {/* Real Live Hardware Values Summary Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '10px', marginBottom: '14px' }}>
          <div style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Temperature</div>
            <div style={{ fontSize: 'var(--text-md)', fontWeight: 700, color: 'var(--status-high)', fontFamily: 'var(--font-mono)' }}>
              {hardwareStatus?.temperature != null ? `${hardwareStatus.temperature.toFixed(2)} °C` : '—'}
            </div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Vibration</div>
            <div style={{ fontSize: 'var(--text-md)', fontWeight: 700, color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>
              {hardwareStatus?.vibration != null ? `${hardwareStatus.vibration.toFixed(2)} m/s²` : '—'}
            </div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Hall Sensor</div>
            <div style={{ fontSize: 'var(--text-xs)', fontWeight: 700, color: hardwareStatus?.hall_detected ? 'var(--status-normal)' : 'var(--text-secondary)' }}>
              {hardwareStatus?.hall_detected ? 'MAGNET DETECTED' : 'NO MAGNET'}
            </div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Motor Speed</div>
            <div style={{ fontSize: 'var(--text-md)', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              {hardwareStatus?.motor_speed != null ? `${hardwareStatus.motor_speed}%` : '0%'}
            </div>
          </div>
          <div style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Motor State</div>
            <div style={{ fontSize: 'var(--text-xs)', fontWeight: 700, color: hardwareStatus?.motor_running ? 'var(--status-normal)' : 'var(--text-muted)' }}>
              {hardwareStatus?.motor_running ? 'RUNNING' : 'STOPPED'}
            </div>
          </div>
        </div>

        {/* Dark Monospace Serial Monitor Terminal Output Box */}
        <div
          style={{
            background: '#0a0d14',
            border: '1px solid var(--border-subtle)',
            borderRadius: '6px',
            padding: '12px',
            height: '140px',
            overflowY: 'auto',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            lineHeight: 1.6,
            color: '#a9b1d6',
          }}
        >
          {rawSerialLogs.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
              Waiting for incoming serial telemetry frames from {hardwareStatus?.port || 'COM5'}...
            </div>
          ) : (
            rawSerialLogs.map((logLine, idx) => (
              <div
                key={idx}
                style={{
                  color: logLine.includes('MAGNET DETECTED')
                    ? '#4ebd77'
                    : logLine.includes('Temperature')
                    ? '#61afef'
                    : logLine.includes('Vibration')
                    ? '#e06c75'
                    : logLine.includes('Motor Speed')
                    ? '#e5c07b'
                    : '#98c379',
                }}
              >
                {logLine}
              </div>
            ))
          )}
        </div>
      </div>

      {/* ── Vision Inspection & Automated Motor Interlock ── */}
      <VisionResultPanel />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: '20px', marginBottom: '20px' }}>
        {/* Left: Charts + Filters */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Filters */}
          <div className="card" style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center' }}>
            <Filter size={13} color="var(--text-muted)" />
            <select
              className="select"
              style={{ width: 'auto' }}
              value={selectedJoint}
              onChange={e => setSelectedJoint(e.target.value)}
              aria-label="Select joint"
            >
              {availableJointIds.map(id => (
                <option key={id} value={id}>{id} — {joints[id]?.label ?? id}</option>
              ))}
            </select>
            <select
              className="select"
              style={{ width: 'auto' }}
              value={selectedSensor}
              onChange={e => setSelectedSensor(e.target.value)}
              aria-label="Select sensor"
            >
              <option value="all">All Sensors (Temp & Vib)</option>
              <option value="temperature">DS18B20 (Temperature)</option>
              <option value="vibration">MPU6050 (Acceleration)</option>
            </select>
            <div style={{ display: 'flex', gap: '4px' }}>
              {TIME_RANGES.map((tr, i) => (
                <button
                  key={tr.label}
                  className={`btn btn-sm ${timeRange === i ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setTimeRange(i)}
                  aria-pressed={timeRange === i}
                >
                  {tr.label}
                </button>
              ))}
            </div>
          </div>

          {/* Aggregates */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '8px' }}>
            {[
              { label: 'Temp Min', value: `${agg.temp.min}°C` },
              { label: 'Temp Max', value: `${agg.temp.max}°C` },
              { label: 'Temp Avg', value: `${agg.temp.avg}°C` },
              { label: 'Vib Min',  value: `${agg.vib.min}` },
              { label: 'Vib Max',  value: `${agg.vib.max}` },
              { label: 'Vib Avg',  value: `${agg.vib.avg}` },
            ].map(s => (
              <div key={s.label} className="card" style={{ padding: '10px', textAlign: 'center' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{s.label}</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>{s.value}</div>
              </div>
            ))}
          </div>

          {/* Temperature Chart */}
          {(selectedSensor === 'all' || selectedSensor === 'temperature') && (
            <div className="card">
              <div className="section-header">
                <span className="section-title">Temperature — {selectedJoint}</span>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>DS18B20 · °C</span>
              </div>
              <SensorChart
                data={tempHist}
                color="var(--status-high)"
                unit="°C"
                height={160}
                refLines={DEFAULT_THRESHOLDS.temperature}
              />
            </div>
          )}

          {/* Vibration Chart */}
          {(selectedSensor === 'all' || selectedSensor === 'vibration') && (
            <div className="card">
              <div className="section-header">
                <span className="section-title">Vibration / Acceleration — {selectedJoint}</span>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>MPU6050 · m/s²</span>
              </div>
              <SensorChart
                data={vibHist}
                color="var(--accent)"
                unit="m/s²"
                height={160}
                refLines={DEFAULT_THRESHOLDS.vibration}
              />
            </div>
          )}

          {/* Raw Data Table */}
          <div className="card">
            <div className="section-header">
              <span className="section-title">Telemetry Stream — {selectedJoint}</span>
              <div style={{ display: 'flex', gap: '8px' }}>
                {['all', 'NORMAL', 'MEDIUM', 'HIGH'].map(s => (
                  <button
                    key={s}
                    className={`btn btn-sm ${statusFilter === s ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setStatusFilter(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
            <div className="table-wrapper">
              <table className="table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Joint</th>
                    <th>Temp (°C)</th>
                    <th>Status</th>
                    <th>Vib (m/s²)</th>
                    <th>Status</th>
                    <th>Hall Pulse</th>
                    <th>Vision Score</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredReadings.length === 0 ? (
                    <tr>
                      <td colSpan={8} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>
                        Waiting for hardware telemetry...
                      </td>
                    </tr>
                  ) : (
                    filteredReadings.slice(0, 15).map(row => {
                      const ts = getValueStatus(row.temperature, DEFAULT_THRESHOLDS.temperature);
                      const vs = getValueStatus(row.vibration,   DEFAULT_THRESHOLDS.vibration);
                      return (
                        <tr key={row.id}>
                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)' }}>{row.time}</td>
                          <td style={{ fontWeight: 600 }}>{row.jointId}</td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{row.temperature?.toFixed(1)}</td>
                          <td><StatusBadge status={ts} size="sm" /></td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{row.vibration?.toFixed(2)}</td>
                          <td><StatusBadge status={vs} size="sm" /></td>
                          <td style={{ fontFamily: 'var(--font-mono)', color: row.hall_event === 'PULSE' ? 'var(--accent)' : 'var(--text-muted)' }}>
                            {row.hall_event}
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
                            {row.vision_score}/100
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right: Sensor Info Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div className="card">
            <div className="section-title" style={{ marginBottom: '12px' }}>Prototype Sensors</div>
            <SensorStatusRow label="Temperature" sensor={SENSOR_INFO.temperature} connected={apiConnected} />
            <SensorStatusRow label="Vibration"   sensor={SENSOR_INFO.vibration}   connected={apiConnected} />
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--border-subtle)' }}>
              <div>
                <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>A3144 Hall Effect</div>
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>Magnetic joint marker sensor</div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 'var(--text-xs)' }}>
                <CheckCircle size={13} color="var(--status-normal)" /><span style={{ color: 'var(--status-normal)' }}>Active</span>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0' }}>
              <div>
                <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>Simulated Vision</div>
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>Phase 1 Manual / Injected Score</div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 'var(--text-xs)' }}>
                <CheckCircle size={13} color="var(--accent)" /><span style={{ color: 'var(--accent)' }}>Simulated</span>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="section-title" style={{ marginBottom: '12px' }}>Sensor Fusion Weights</div>
            {[
              { label: 'Vision Score', weight: '40%', desc: 'Surface inspection' },
              { label: 'Magnetic (Hall)', weight: '30%', desc: 'A3144 joint pulse' },
              { label: 'Vibration (MPU)', weight: '20%', desc: 'Acceleration RMS' },
              { label: 'Temperature (DS)', weight: '10%', desc: 'Thermal deviation' },
            ].map(r => (
              <div key={r.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 0', borderBottom: '1px solid var(--border-subtle)', fontSize: 'var(--text-xs)' }}>
                <div>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{r.label}</span>
                  <div style={{ color: 'var(--text-muted)', fontSize: '10px' }}>{r.desc}</div>
                </div>
                <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{r.weight}</span>
              </div>
            ))}
          </div>

          <div className="card">
            <div className="section-title" style={{ marginBottom: '12px' }}>Threshold Reference
              <span style={{ fontSize: '10px', color: 'var(--status-medium)', marginLeft: 6 }}>⚠ PLACEHOLDER</span>
            </div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', lineHeight: 1.8 }}>
              <div><span style={{ color: 'var(--status-normal)' }}>■</span> NORMAL Temp ≤ {DEFAULT_THRESHOLDS.temperature.normalMax}°C</div>
              <div><span style={{ color: 'var(--status-medium)' }}>■</span> MEDIUM Temp ≤ {DEFAULT_THRESHOLDS.temperature.mediumMax}°C</div>
              <div><span style={{ color: 'var(--status-high)' }}>■</span> HIGH Temp &gt; {DEFAULT_THRESHOLDS.temperature.mediumMax}°C</div>
              <div style={{ margin: '8px 0', borderTop: '1px solid var(--border-subtle)' }} />
              <div><span style={{ color: 'var(--status-normal)' }}>■</span> NORMAL Vib ≤ {DEFAULT_THRESHOLDS.vibration.normalMax} m/s²</div>
              <div><span style={{ color: 'var(--status-medium)' }}>■</span> MEDIUM Vib ≤ {DEFAULT_THRESHOLDS.vibration.mediumMax} m/s²</div>
              <div><span style={{ color: 'var(--status-high)' }}>■</span> HIGH Vib &gt; {DEFAULT_THRESHOLDS.vibration.mediumMax} m/s²</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
