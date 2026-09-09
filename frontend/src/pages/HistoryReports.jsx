import { useState } from 'react';
import { Plus, Edit2, Trash2, Search, Download } from 'lucide-react';
import { useSimulation } from '../context/SimulationContext';
import { MOCK_ALERTS, MOCK_EVENTS, calcDuration } from '../data/mockAlerts';
import { JOINT_IDS } from '../data/constants';
import StatusBadge from '../components/ui/StatusBadge';
import Modal from '../components/ui/Modal';

// ─── Operator Maintenance Log (editable) ──────────────────────────────────────
const INITIAL_MAINTENANCE_LOG = [
  {
    id: 'MNT-001',
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    jointId: 'J03',
    operator: 'Operator A',
    action: 'Lubricated joint bearing',
    notes: 'Applied lubricant to J03 bearing. Condition appeared dry. Will monitor temperature response.',
    source: 'operator',
  },
  {
    id: 'MNT-002',
    timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
    jointId: 'J02',
    operator: 'Operator B',
    action: 'Visual inspection — no defects found',
    notes: 'Carried out scheduled visual inspection. No visible wear or damage on J02 joint splice.',
    source: 'operator',
  },
];

function LogForm({ initial, onSave, onCancel }) {
  const [form, setForm] = useState(initial ?? { jointId: 'J01', operator: '', action: '', notes: '' });
  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      <div>
        <label style={{ display: 'block', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Joint</label>
        <select className="select" value={form.jointId} onChange={set('jointId')} aria-label="Select joint">
          {JOINT_IDS.map(id => <option key={id} value={id}>{id}</option>)}
        </select>
      </div>
      <div>
        <label style={{ display: 'block', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Operator Name</label>
        <input className="input" value={form.operator} onChange={set('operator')} placeholder="e.g. Operator A" aria-label="Operator name" />
      </div>
      <div>
        <label style={{ display: 'block', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Action Taken</label>
        <input className="input" value={form.action} onChange={set('action')} placeholder="e.g. Lubricated bearing" aria-label="Action taken" />
      </div>
      <div>
        <label style={{ display: 'block', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Notes</label>
        <textarea
          className="input"
          value={form.notes}
          onChange={set('notes')}
          rows={4}
          placeholder="Additional observations..."
          aria-label="Maintenance notes"
          style={{ resize: 'vertical' }}
        />
      </div>
      <div style={{ display: 'flex', gap: '8px', paddingTop: '4px' }}>
        <button className="btn btn-primary" onClick={() => onSave(form)} disabled={!form.operator || !form.action}>
          Save Record
        </button>
        <button className="btn btn-ghost" onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}

export default function HistoryReports() {
  const [maintenanceLog, setMaintenanceLog] = useState(INITIAL_MAINTENANCE_LOG);
  const [view,        setView]        = useState('table');   // 'table' | 'timeline'
  const [search,      setSearch]      = useState('');
  const [filterJoint, setFilterJoint] = useState('all');
  const [filterLevel, setFilterLevel] = useState('all');
  const [editModal,   setEditModal]   = useState(null);      // null | 'new' | recordObj
  const [tab,         setTab]         = useState('alerts');  // 'alerts' | 'maintenance' | 'events'

  const { alerts, alertHistory: liveAlertHistory } = useSimulation();

  // Alert history (prefer backend telemetry logs, fallback to demo seeds)
  const sourceAlerts = (liveAlertHistory && liveAlertHistory.length > 0)
    ? liveAlertHistory
    : (alerts && alerts.length > 0)
      ? alerts
      : MOCK_ALERTS;

  const alertHistory = sourceAlerts.map(a => ({
    ...a,
    duration: calcDuration(a.startedAt, a.resolvedAt),
  }));

  const filteredAlerts = alertHistory.filter(a => {
    if (filterJoint !== 'all' && a.jointId !== filterJoint) return false;
    if (filterLevel !== 'all' && a.level   !== filterLevel) return false;
    if (search) {
      const q = search.toLowerCase();
      return a.jointId.toLowerCase().includes(q) || a.parameter.toLowerCase().includes(q) || a.message.toLowerCase().includes(q);
    }
    return true;
  });

  const filteredMaint = maintenanceLog.filter(r => {
    if (filterJoint !== 'all' && r.jointId !== filterJoint) return false;
    if (search) {
      const q = search.toLowerCase();
      return r.jointId.toLowerCase().includes(q) || r.action.toLowerCase().includes(q) || r.notes.toLowerCase().includes(q);
    }
    return true;
  });

  function saveRecord(form) {
    if (editModal === 'new') {
      setMaintenanceLog(prev => [{
        id: `MNT-${String(prev.length + 1).padStart(3,'0')}`,
        timestamp: new Date().toISOString(),
        ...form,
        source: 'operator',
      }, ...prev]);
    } else {
      setMaintenanceLog(prev => prev.map(r => r.id === editModal.id ? { ...r, ...form } : r));
    }
    setEditModal(null);
  }

  function deleteRecord(id) {
    if (window.confirm('Delete this maintenance record?')) {
      setMaintenanceLog(prev => prev.filter(r => r.id !== id));
    }
  }

  function exportCSV() {
    const rows = [
      ['Type', 'Timestamp', 'Joint', 'Level/Action', 'Parameter/Operator', 'Reading/Notes', 'Status'],
      ...filteredAlerts.map(a => ['Alert', a.startedAt, a.jointId, a.level, a.parameter, a.reading, a.active ? 'Active' : 'Resolved']),
      ...filteredMaint.map(r => ['Maintenance', r.timestamp, r.jointId, r.action, r.operator, r.notes, 'Logged']),
    ];
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `jointguard_history_${Date.now()}.csv`;
    a.click(); URL.revokeObjectURL(url);
  }

  return (
    <div className="page-content">
      <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h1 className="page-title">History & Reports</h1>
          <p className="page-subtitle">Alert history (read-only) and operator maintenance log</p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-secondary btn-sm" onClick={exportCSV}><Download size={13} /> Export CSV</button>
        </div>
      </div>

      {/* Summary */}
      <div className="grid-4" style={{ marginBottom: '20px' }}>
        {[
          { label: 'Total Alerts', value: alertHistory.length },
          { label: 'Active Alerts', value: alertHistory.filter(a => a.active).length },
          { label: 'Maintenance Records', value: maintenanceLog.length },
          { label: 'Events Logged', value: MOCK_EVENTS.length },
        ].map(s => (
          <div key={s.label} className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 600 }}>{s.label}</div>
            <div style={{ fontSize: 'var(--text-2xl)', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Tabs + Filters */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
        {[
          { id: 'alerts', label: 'Alert History' },
          { id: 'maintenance', label: 'Maintenance Log' },
          { id: 'events', label: 'Events' },
        ].map(t => (
          <button key={t.id} className={`btn btn-sm ${tab === t.id ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setTab(t.id)} aria-pressed={tab === t.id}>{t.label}</button>
        ))}

        <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <div style={{ position: 'relative' }}>
            <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', pointerEvents: 'none' }} />
            <input className="input" style={{ paddingLeft: 30, width: 200 }} placeholder="Search..." value={search} onChange={e => setSearch(e.target.value)} aria-label="Search history" />
          </div>
          <select className="select" style={{ width: 'auto' }} value={filterJoint} onChange={e => setFilterJoint(e.target.value)} aria-label="Filter by joint">
            <option value="all">All Joints</option>
            {JOINT_IDS.map(id => <option key={id} value={id}>{id}</option>)}
          </select>
          {tab === 'alerts' && (
            <select className="select" style={{ width: 'auto' }} value={filterLevel} onChange={e => setFilterLevel(e.target.value)} aria-label="Filter by level">
              <option value="all">All Levels</option>
              <option value="HIGH">HIGH</option>
              <option value="MEDIUM">MEDIUM</option>
              <option value="NORMAL">NORMAL</option>
            </select>
          )}
          {tab === 'alerts' && (
            <div style={{ display: 'flex', gap: '4px' }}>
              <button className={`btn btn-sm ${view === 'table' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setView('table')}>Table</button>
              <button className={`btn btn-sm ${view === 'timeline' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setView('timeline')}>Timeline</button>
            </div>
          )}
        </div>
      </div>

      {/* Alert History Tab */}
      {tab === 'alerts' && (
        <div className="card">
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: 6 }}>
            🔒 System-generated alert records are read-only.
          </div>

          {view === 'table' ? (
            <div className="table-wrapper">
              <table aria-label="Alert history table">
                <thead>
                  <tr>
                    <th>Level</th><th>Joint</th><th>Parameter</th>
                    <th>Reading</th><th>Started</th><th>Duration</th><th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAlerts.map(a => (
                    <tr key={a.id}>
                      <td><StatusBadge status={a.level} size="sm" /></td>
                      <td style={{ fontWeight: 600 }}>{a.jointId}</td>
                      <td style={{ textTransform: 'capitalize' }}>{a.parameter}</td>
                      <td style={{ fontFamily: 'var(--font-mono)' }}>{a.reading} {a.unit}</td>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)' }}>{new Date(a.startedAt).toLocaleString()}</td>
                      <td>{a.duration}</td>
                      <td>
                        {a.active
                          ? <span style={{ color: 'var(--status-high)', fontSize: 'var(--text-xs)', fontWeight: 600 }}>ACTIVE</span>
                          : <span style={{ color: 'var(--status-normal)', fontSize: 'var(--text-xs)', fontWeight: 600 }}>RESOLVED</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            /* Timeline view */
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
              {filteredAlerts.sort((a, b) => new Date(b.startedAt) - new Date(a.startedAt)).map((a, i) => (
                <div key={a.id} style={{ display: 'flex', gap: '16px', paddingBottom: '16px', position: 'relative' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: a.level === 'HIGH' ? 'var(--status-high)' : a.level === 'MEDIUM' ? 'var(--status-medium)' : 'var(--status-normal)', marginTop: 4 }} />
                    {i < filteredAlerts.length - 1 && <div style={{ width: 1, flex: 1, background: 'var(--border-subtle)', marginTop: 4 }} />}
                  </div>
                  <div style={{ flex: 1, paddingBottom: 4 }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: 4 }}>
                      <StatusBadge status={a.level} size="sm" />
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 'var(--text-sm)' }}>{a.jointId} — {a.parameter}</span>
                    </div>
                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginBottom: 2 }}>
                      {new Date(a.startedAt).toLocaleString()} · {a.duration}
                    </div>
                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>{a.message}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Maintenance Log Tab */}
      {tab === 'maintenance' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <button className="btn btn-primary btn-sm" onClick={() => setEditModal('new')}>
              <Plus size={13} /> Add Maintenance Record
            </button>
          </div>

          <div className="card">
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginBottom: '12px' }}>
              ✏ Operator-created records are editable.
            </div>
            {filteredMaint.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>No maintenance records</div>
            ) : (
              filteredMaint.map(r => (
                <div key={r.id} style={{ padding: '14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', marginBottom: '10px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <span style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 'var(--text-sm)' }}>{r.jointId}</span>
                      <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>by {r.operator}</span>
                    </div>
                    <div style={{ display: 'flex', gap: '4px' }}>
                      <button className="btn btn-ghost btn-sm btn-icon" onClick={() => setEditModal(r)} aria-label="Edit record"><Edit2 size={12} /></button>
                      <button className="btn btn-ghost btn-sm btn-icon" onClick={() => deleteRecord(r.id)} aria-label="Delete record" style={{ color: 'var(--status-high)' }}><Trash2 size={12} /></button>
                    </div>
                  </div>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 'var(--text-sm)', marginBottom: 4 }}>{r.action}</div>
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)', marginBottom: 6 }}>{r.notes}</div>
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{new Date(r.timestamp).toLocaleString()}</div>
                </div>
              ))
            )}
          </div>

          {/* Future before/after maintenance section */}
          <div className="card" style={{ border: '1px dashed var(--border-default)' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--accent)', fontWeight: 600, marginBottom: 6 }}>Future Feature — Before / After Maintenance</div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
              This section will display sensor trend comparisons before and after a maintenance record, once real sensor data is collected. The data structure is already in place.
            </div>
          </div>
        </div>
      )}

      {/* Events Tab */}
      {tab === 'events' && (
        <div className="card">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
            {MOCK_EVENTS.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)).map((evt, i) => (
              <div key={evt.id} style={{ display: 'flex', gap: '16px', paddingBottom: '16px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: evt.level === 'HIGH' ? 'var(--status-high)' : evt.level === 'MEDIUM' ? 'var(--status-medium)' : 'var(--accent)', marginTop: 4 }} />
                  {i < MOCK_EVENTS.length - 1 && <div style={{ width: 1, flex: 1, background: 'var(--border-subtle)', marginTop: 4 }} />}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>{evt.description}</div>
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {new Date(evt.timestamp).toLocaleString()} {evt.jointId ? `· ${evt.jointId}` : ''}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Add/Edit Modal */}
      <Modal isOpen={!!editModal} onClose={() => setEditModal(null)} title={editModal === 'new' ? 'Add Maintenance Record' : 'Edit Maintenance Record'} size="sm">
        <LogForm
          initial={editModal !== 'new' ? editModal : undefined}
          onSave={saveRecord}
          onCancel={() => setEditModal(null)}
        />
      </Modal>
    </div>
  );
}
