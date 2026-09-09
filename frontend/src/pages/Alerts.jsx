import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle, XCircle, Filter, Link2 } from 'lucide-react';
import { useAlerts } from '../hooks/useAlerts';
import { calcDuration } from '../data/mockAlerts';
import { JOINT_IDS } from '../data/constants';
import StatusBadge from '../components/ui/StatusBadge';
import Modal from '../components/ui/Modal';

// ─── Display helpers ──────────────────────────────────────────────────────────

/**
 * Maps the internal parameter key to a human-readable display label.
 * The 'vibration' key holds raw MPU6050 acceleration data — not a
 * validated vibration metric — so it is labelled accordingly.
 */
function getParamLabel(param) {
  if (param === 'vibration') return 'MPU6050 Acceleration';
  if (param === 'temperature') return 'Temperature · DS18B20';
  return param;
}

/**
 * Derives a plain-English cause string from the alert record.
 * ⚠ Thresholds referenced here are PLACEHOLDER values — not validated limits.
 */
function getAlertCause(alert) {
  const paramName = alert.parameter === 'vibration'
    ? 'MPU6050 acceleration'
    : 'temperature';
  return `${paramName.charAt(0).toUpperCase() + paramName.slice(1)} reading `
    + `(${alert.reading}\u202f${alert.unit}) crossed the current `
    + `\u26A0\uFE0F PLACEHOLDER ${alert.level} threshold — not a validated limit.`;
}

function AlertRow({ alert, onAcknowledge, onResolve, onSelect }) {
  const duration = calcDuration(alert.startedAt, alert.resolvedAt);
  return (
    <tr onClick={() => onSelect(alert)} style={{ cursor: 'pointer' }}>
      <td>
        <StatusBadge status={alert.level} size="sm" />
      </td>
      <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{alert.jointId}</td>
      <td>{getParamLabel(alert.parameter)}</td>
      <td style={{ fontFamily: 'var(--font-mono)' }}>{alert.reading} {alert.unit}</td>
      <td style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)' }}>
        {new Date(alert.startedAt).toLocaleString()}
      </td>
      <td style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)' }}>
        {alert.resolvedAt ? new Date(alert.resolvedAt).toLocaleString() : '—'}
      </td>
      <td>{duration}</td>
      <td>
        {alert.active
          ? <span style={{ color: 'var(--status-high)', fontSize: 'var(--text-xs)', fontWeight: 600 }}>ACTIVE</span>
          : <span style={{ color: 'var(--status-normal)', fontSize: 'var(--text-xs)', fontWeight: 600 }}>RESOLVED</span>
        }
      </td>
      <td onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', gap: '4px' }}>
          {alert.active && !alert.acknowledged && (
            <button className="btn btn-sm btn-secondary" onClick={() => onAcknowledge(alert.id)} aria-label={`Acknowledge ${alert.id}`}>ACK</button>
          )}
          {alert.active && (
            <button className="btn btn-sm btn-secondary" onClick={() => onResolve(alert.id)} aria-label={`Resolve ${alert.id}`} style={{ color: 'var(--status-normal)' }}>
              <CheckCircle size={11} />
            </button>
          )}
        </div>
      </td>
    </tr>
  );
}

function AlertDetailModal({ alert, onClose, onAcknowledge, onResolve }) {
  const navigate = useNavigate();
  if (!alert) return null;

  const cause = getAlertCause(alert);
  const duration = calcDuration(alert.startedAt, alert.resolvedAt);

  const detailRows = [
    { label: 'Alert ID',    value: alert.id,                                                    mono: true  },
    { label: 'Joint',       value: alert.jointId,                                               mono: false },
    { label: 'Parameter',   value: getParamLabel(alert.parameter),                              mono: false },
    { label: 'Reading',     value: `${alert.reading}\u202f${alert.unit}`,                       mono: true  },
    { label: 'Level',       value: alert.level,                                                 mono: false },
    { label: 'Started',     value: new Date(alert.startedAt).toLocaleString(),                  mono: true  },
    { label: 'Resolved',    value: alert.resolvedAt ? new Date(alert.resolvedAt).toLocaleString() : 'Not yet resolved', mono: true },
    { label: 'Duration',    value: duration,                                                    mono: true  },
    { label: 'Status',      value: alert.active ? 'ACTIVE' : 'RESOLVED',                       mono: false },
    { label: 'Acknowledged',value: alert.acknowledged ? `Yes — by ${alert.acknowledgedBy}` : 'No', mono: false },
    { label: 'Source',      value: 'System-generated · read-only',                             mono: false },
  ];

  return (
    <Modal isOpen={!!alert} onClose={onClose} title={`Alert Detail — ${alert.jointId}`} size="md">
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

        {/* Status badge row */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <StatusBadge status={alert.level} />
          {alert.active
            ? <span style={{ fontSize: 'var(--text-xs)', color: 'var(--status-high)', fontWeight: 700, letterSpacing: '0.05em' }}>ACTIVE</span>
            : <span style={{ fontSize: 'var(--text-xs)', color: 'var(--status-normal)', fontWeight: 700, letterSpacing: '0.05em' }}>RESOLVED</span>
          }
        </div>

        {/* Cause block — prominently placed */}
        <div style={{
          padding: '12px 14px',
          background: 'var(--bg-elevated)',
          border: '1px solid var(--status-medium-border)',
          borderLeft: '3px solid var(--status-medium)',
          borderRadius: 'var(--radius)',
        }}>
          <div style={{ fontSize: 'var(--text-xs)', fontWeight: 700, color: 'var(--status-medium)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
            Cause
          </div>
          <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            {cause}
          </div>
        </div>

        {/* Detail rows */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          {detailRows.map(row => (
            <div
              key={row.label}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                gap: '12px',
                fontSize: 'var(--text-sm)',
                borderBottom: '1px solid var(--border-subtle)',
                padding: '8px 0',
              }}
            >
              <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>{row.label}</span>
              <span style={{
                color: 'var(--text-primary)',
                fontFamily: row.mono ? 'var(--font-mono)' : undefined,
                fontSize: row.mono ? 'var(--text-xs)' : undefined,
                textAlign: 'right',
              }}>
                {row.value}
              </span>
            </div>
          ))}
        </div>

        {/* Alert message */}
        {alert.message && (
          <div style={{
            padding: '10px 12px',
            background: 'var(--bg-elevated)',
            borderRadius: 'var(--radius)',
            fontSize: 'var(--text-sm)',
            color: 'var(--text-secondary)',
            lineHeight: 1.55,
          }}>
            {alert.message}
          </div>
        )}

        {/* Actions */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', paddingTop: '2px' }}>
          {alert.active && !alert.acknowledged && (
            <button
              className="btn btn-secondary"
              onClick={() => { onAcknowledge(alert.id); onClose(); }}
            >
              <CheckCircle size={14} /> Acknowledge
            </button>
          )}
          {alert.active && (
            <button
              className="btn btn-secondary"
              style={{ color: 'var(--status-normal)' }}
              onClick={() => { onResolve(alert.id); onClose(); }}
            >
              <XCircle size={14} /> Mark Resolved
            </button>
          )}
          <button
            className="btn btn-ghost"
            onClick={() => { onClose(); navigate('/joints'); }}
            style={{ marginLeft: 'auto' }}
          >
            <Link2 size={14} /> View Joint
          </button>
        </div>

      </div>
    </Modal>
  );
}

export default function Alerts() {
  const { activeAlerts, resolvedAlerts, acknowledge, resolve, summary } = useAlerts();
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [filterJoint,  setFilterJoint]    = useState('all');
  const [filterLevel,  setFilterLevel]    = useState('all');
  const [filterParam,  setFilterParam]    = useState('all');
  const [tab,          setTab]            = useState('active');

  // Count alerts resolved within the current calendar day
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);
  const resolvedToday = resolvedAlerts.filter(
    a => a.resolvedAt && new Date(a.resolvedAt) >= todayStart
  ).length;

  const displayed = (tab === 'active' ? activeAlerts : resolvedAlerts).filter(a => {
    if (filterJoint !== 'all' && a.jointId !== filterJoint) return false;
    if (filterLevel !== 'all' && a.level !== filterLevel)   return false;
    if (filterParam !== 'all' && a.parameter !== filterParam) return false;
    return true;
  });

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Alerts</h1>
        <p className="page-subtitle">Monitor, acknowledge, and resolve system alerts</p>
      </div>

      {/* Summary Cards */}
      <div className="grid-4" style={{ marginBottom: '20px' }}>
        {[
          { label: 'Total Active',    value: summary.totalActive, cls: '' },
          { label: 'HIGH',            value: summary.highCount,   cls: 'badge-high' },
          { label: 'MEDIUM',          value: summary.mediumCount, cls: 'badge-medium' },
          { label: 'Resolved Today',  value: resolvedToday,       cls: '' },
        ].map(s => (
          <div key={s.label} className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>{s.label}</div>
            <div style={{ fontSize: 'var(--text-3xl)', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Tab + Filters */}
      <div className="card" style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
          <div style={{ display: 'flex', gap: '4px' }}>
            {['active', 'history'].map(t => (
              <button
                key={t}
                className={`btn btn-sm ${tab === t ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setTab(t)}
                aria-pressed={tab === t}
              >
                {t === 'active' ? `Active (${activeAlerts.length})` : `History (${resolvedAlerts.length})`}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
            <Filter size={13} color="var(--text-muted)" />
            <select className="select" style={{ width: 'auto' }} value={filterJoint} onChange={e => setFilterJoint(e.target.value)} aria-label="Filter by joint">
              <option value="all">All Joints</option>
              {JOINT_IDS.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
            <select className="select" style={{ width: 'auto' }} value={filterLevel} onChange={e => setFilterLevel(e.target.value)} aria-label="Filter by level">
              <option value="all">All Levels</option>
              <option value="HIGH">HIGH</option>
              <option value="MEDIUM">MEDIUM</option>
              <option value="NORMAL">NORMAL</option>
            </select>
            <select className="select" style={{ width: 'auto' }} value={filterParam} onChange={e => setFilterParam(e.target.value)} aria-label="Filter by parameter">
              <option value="all">All Parameters</option>
              <option value="temperature">Temperature · DS18B20</option>
              <option value="vibration">MPU6050 Acceleration</option>
              <option value="magnetic">Magnetic · A3144</option>
              <option value="vision">Vision Score</option>
              <option value="system">System / Health</option>
            </select>
          </div>
        </div>

        <div className="table-wrapper">
          <table aria-label={`${tab === 'active' ? 'Active' : 'Resolved'} alerts table`}>
            <thead>
              <tr>
                <th>Level</th>
                <th>Joint</th>
                <th>Parameter</th>
                <th>Reading</th>
                <th>Started</th>
                <th>Resolved</th>
                <th>Duration</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {displayed.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
                    No {tab} alerts
                  </td>
                </tr>
              ) : (
                displayed.map(alert => (
                  <AlertRow
                    key={alert.id}
                    alert={alert}
                    onAcknowledge={acknowledge}
                    onResolve={resolve}
                    onSelect={setSelectedAlert}
                  />
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <AlertDetailModal
        alert={selectedAlert}
        onClose={() => setSelectedAlert(null)}
        onAcknowledge={acknowledge}
        onResolve={resolve}
      />
    </div>
  );
}
