import StatusBadge from './StatusBadge';

/**
 * MetricCard — displays a sensor metric with icon, label, value, unit, and status.
 * @param {ReactNode} icon
 * @param {string}    label
 * @param {number}    value
 * @param {string}    unit
 * @param {string}    status      - STATUS level
 * @param {string}    [subtext]   - Small note below value
 * @param {boolean}   [showStatus]
 * @param {string}    [accentColor] - Override border accent
 */
export default function MetricCard({
  icon,
  label,
  value,
  unit,
  status,
  subtext,
  showStatus = true,
  accentColor,
}) {
  const accentStyle = accentColor
    ? { borderLeftColor: accentColor, borderLeftWidth: '3px', borderLeftStyle: 'solid' }
    : {};

  return (
    <div className="card" style={accentStyle} aria-label={`${label}: ${value} ${unit}`}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {icon && (
            <span style={{ color: 'var(--text-muted)', display: 'flex' }} aria-hidden="true">
              {icon}
            </span>
          )}
          <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
            {label}
          </span>
        </div>
        {showStatus && status && <StatusBadge status={status} size="sm" />}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
        <span style={{ fontSize: 'var(--text-2xl)', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
          {typeof value === 'number' ? value.toFixed(value % 1 === 0 ? 0 : 1) : value}
        </span>
        <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>{unit}</span>
      </div>

      {subtext && (
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginTop: '4px' }}>
          {subtext}
        </div>
      )}
    </div>
  );
}
