import { STATUS } from '../../data/constants';

const DOT_STYLE = {
  display: 'inline-block',
  width: '6px',
  height: '6px',
  borderRadius: '50%',
  flexShrink: 0,
};

const CONFIG = {
  [STATUS.NORMAL]: { label: 'Normal', className: 'badge-normal', dotColor: 'var(--status-normal)' },
  [STATUS.MEDIUM]: { label: 'Medium', className: 'badge-medium', dotColor: 'var(--status-medium)' },
  [STATUS.HIGH]:   { label: 'High',   className: 'badge-high',   dotColor: 'var(--status-high)' },
};

/**
 * StatusBadge — renders a colored badge for NORMAL / MEDIUM / HIGH status.
 * @param {string}  status    - STATUS.NORMAL | STATUS.MEDIUM | STATUS.HIGH
 * @param {boolean} showDot   - Whether to show the indicator dot
 * @param {string}  size      - 'sm' | 'default'
 */
export default function StatusBadge({ status, showDot = true, size = 'default' }) {
  const cfg = CONFIG[status] ?? CONFIG[STATUS.NORMAL];

  return (
    <span
      className={`badge ${cfg.className}`}
      style={size === 'sm' ? { fontSize: '0.6rem', padding: '1px 6px' } : {}}
      aria-label={`Status: ${cfg.label}`}
    >
      {showDot && (
        <span
          style={{ ...DOT_STYLE, background: cfg.dotColor }}
          aria-hidden="true"
        />
      )}
      {cfg.label}
    </span>
  );
}
