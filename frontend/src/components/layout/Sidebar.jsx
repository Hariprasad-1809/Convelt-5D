import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, Activity, Radio, Bell, History,
  Info, Settings, ChevronLeft, ChevronRight, Cpu,
} from 'lucide-react';
import { NAV_ITEMS } from '../../data/constants';
import { useAlerts } from '../../hooks/useAlerts';

const ICONS = {
  'dashboard':        LayoutDashboard,
  'joint-monitoring': Activity,
  'sensor-data':      Radio,
  'alerts':           Bell,
  'history':          History,
  'about':            Info,
  'settings':         Settings,
};

export default function Sidebar({ collapsed, onToggle, mobileOpen, onMobileClose }) {
  const location = useLocation();
  const { summary } = useAlerts();

  return (
    <>
      {/* Mobile overlay */}
      <div
        className={`sidebar-overlay ${mobileOpen ? 'visible' : ''}`}
        onClick={onMobileClose}
        aria-hidden="true"
      />

      <aside
        className={`sidebar ${collapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`}
        role="navigation"
        aria-label="Main navigation"
      >
        {/* Brand */}
        <NavLink to="/" className="sidebar-brand" onClick={onMobileClose}>
          <div className="sidebar-logo" aria-hidden="true">
            <Cpu size={18} color="#0a0d12" strokeWidth={2.5} />
          </div>
          {!collapsed && (
            <div className="sidebar-brand-text">
              <div className="sidebar-brand-name">JointGuard</div>
              <div className="sidebar-brand-sub">Belt Joint Monitor</div>
            </div>
          )}
        </NavLink>

        {/* Nav */}
        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => {
            const Icon = ICONS[item.id];
            const isActive = item.path === '/'
              ? location.pathname === '/'
              : location.pathname.startsWith(item.path);
            const hasAlert = item.id === 'alerts' && summary.highCount > 0;

            return (
              <NavLink
                key={item.id}
                to={item.path}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={onMobileClose}
                title={collapsed ? item.label : undefined}
                aria-label={item.label}
                aria-current={isActive ? 'page' : undefined}
              >
                {Icon && <Icon className="nav-item-icon" />}
                {!collapsed && (
                  <span className="nav-item-label">{item.label}</span>
                )}
                {hasAlert && !collapsed && (
                  <span className="nav-alert-dot" aria-label="Active alerts" />
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Footer toggle (desktop only) */}
        <div className="sidebar-footer">
          <button
            className="sidebar-toggle"
            onClick={onToggle}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed
              ? <ChevronRight size={16} />
              : <ChevronLeft size={16} />
            }
          </button>
        </div>
      </aside>
    </>
  );
}
