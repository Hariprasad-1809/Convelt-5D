import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Menu, Wifi, WifiOff, Activity, AlertTriangle } from 'lucide-react';
import { NAV_ITEMS } from '../../data/constants';
import { useSimulation } from '../../context/SimulationContext';

function useClock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

export default function Header({ onMobileToggle }) {
  const location = useLocation();
  const time = useClock();
  const { lastUpdated, apiConnected, simStatus } = useSimulation();

  const currentPage = NAV_ITEMS.find(item =>
    item.path === '/'
      ? location.pathname === '/'
      : location.pathname.startsWith(item.path),
  );

  const formattedUpdate = lastUpdated
    ? new Date(lastUpdated).toLocaleTimeString('en-US', {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      })
    : 'Waiting for telemetry';

  return (
    <header className="header" role="banner">
      <div className="header-left">
        <button
          className="header-mobile-toggle"
          onClick={onMobileToggle}
          aria-label="Open navigation menu"
        >
          <Menu size={18} />
        </button>
        <span className="header-breadcrumb">
          {currentPage?.label ?? 'JointGuard'}
        </span>
      </div>

      <div className="header-right">
        {/* Architecture Phase & Backend Badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '3px 10px',
            borderRadius: '100px',
            background: apiConnected ? 'rgba(6, 182, 212, 0.08)' : 'rgba(239, 68, 68, 0.08)',
            border: apiConnected ? '1px solid rgba(6, 182, 212, 0.25)' : '1px solid rgba(239, 68, 68, 0.25)',
            fontSize: '10px',
            fontWeight: 700,
            letterSpacing: '0.07em',
            color: apiConnected ? 'var(--accent)' : 'var(--status-high)',
            textTransform: 'uppercase',
            flexShrink: 0,
          }}
          title={apiConnected ? 'Connected to FastAPI REST Backend' : 'Backend is currently offline or unreachable'}
        >
          <Activity size={11} />
          {apiConnected ? `PHASE 1 REST · Cycle ${simStatus?.current_cycle ?? 0}` : 'REST Offline'}
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            padding: '3px 9px',
            borderRadius: '100px',
            background: 'rgba(245,158,11,0.08)',
            border: '1px solid rgba(245,158,11,0.2)',
            fontSize: '10px',
            fontWeight: 700,
            letterSpacing: '0.07em',
            color: 'var(--status-medium)',
            textTransform: 'uppercase',
            flexShrink: 0,
          }}
          title="A3144 Hall sensor demonstration. Vision scores simulated or injected in Phase 1."
        >
          <AlertTriangle size={10} />
          A3144 demo
        </div>

        <div
          className="header-update"
          title={`Last sensor update: ${formattedUpdate}`}
          style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}
        >
          Updated {formattedUpdate}
        </div>

        <div className="header-system-status">
          <span
            className={`status-dot ${apiConnected ? '' : 'offline'}`}
            aria-hidden="true"
          />
          {apiConnected ? <Wifi size={13} /> : <WifiOff size={13} color="var(--status-high)" />}
          <span>{apiConnected ? 'System Online' : 'Offline'}</span>
        </div>

        <time
          className="header-time"
          dateTime={time.toISOString()}
          aria-label="Current time"
        >
          {time.toLocaleTimeString('en-US', {
            hour: '2-digit', minute: '2-digit', second: '2-digit',
          })}
        </time>
      </div>
    </header>
  );
}
