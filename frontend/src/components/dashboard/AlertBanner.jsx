import { useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';

/**
 * AlertBanner — shows a dismissable top-of-page banner for critical alerts.
 * @param {Array}  activeAlerts  - active alert objects
 */
export default function AlertBanner({ activeAlerts }) {
  const [dismissed, setDismissed] = useState(false);

  const highAlerts = activeAlerts.filter(a => a.level === 'HIGH');
  if (!highAlerts.length || dismissed) return null;

  const joints = [...new Set(highAlerts.map(a => a.jointId))].join(', ');

  return (
    <div
      role="alert"
      aria-live="assertive"
      style={{
        background: 'var(--status-high-dim)',
        border: '1px solid var(--status-high-border)',
        borderRadius: 'var(--radius)',
        padding: '10px 16px',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        marginBottom: '20px',
      }}
    >
      <AlertTriangle size={16} color="var(--status-high)" style={{ flexShrink: 0 }} aria-hidden="true" />
      <div style={{ flex: 1, fontSize: 'var(--text-sm)' }}>
        <span style={{ fontWeight: 600, color: 'var(--status-high)' }}>
          HIGH ALERT —{' '}
        </span>
        <span style={{ color: 'var(--text-primary)' }}>
          {highAlerts.length} critical condition{highAlerts.length > 1 ? 's' : ''} detected on {joints}. Inspect immediately.
        </span>
      </div>
      <button
        onClick={() => setDismissed(true)}
        aria-label="Dismiss alert banner"
        style={{
          background: 'none',
          border: 'none',
          color: 'var(--text-muted)',
          cursor: 'pointer',
          padding: '2px',
          display: 'flex',
          flexShrink: 0,
        }}
      >
        <X size={16} />
      </button>
    </div>
  );
}
