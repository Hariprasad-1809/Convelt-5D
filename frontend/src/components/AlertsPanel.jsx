import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Bell, CircleAlert, HelpCircle, Search, ShieldAlert, Clock3 } from 'lucide-react';

function alertTone(risk) {
  switch (risk) {
    case 'HIGH':
      return { border: 'risk-left-high glow-high', chip: 'border-risk-high bg-risk-high-soft text-risk-high', icon: <ShieldAlert className="h-4 w-4 text-risk-high" />, label: 'Critical' };
    case 'MEDIUM':
      return { border: 'risk-left-med glow-med', chip: 'border-risk-medium bg-risk-medium-soft text-risk-medium', icon: <AlertTriangle className="h-4 w-4 text-risk-medium" />, label: 'Warning' };
    default:
      return { border: 'risk-left-unknown', chip: 'border-risk-unknown bg-risk-unknown-soft text-risk-unknown', icon: <HelpCircle className="h-4 w-4 text-risk-unknown" />, label: 'Unknown' };
  }
}

function formatTime(value) {
  if (!value) return 'No timestamp';
  return new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(value));
}

export default function AlertsPanel({ activeAlerts, alertHistory }) {
  const [activeTab, setActiveTab] = useState('active');
  const [searchTerm, setSearchTerm] = useState('');

  const source = activeTab === 'active' ? activeAlerts : alertHistory;

  const displayedAlerts = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    return source.filter((alert) => {
      if (!query) return true;
      return [alert.joint_id, alert.alert_type, alert.message]
        .filter(Boolean)
        .some((field) => field.toLowerCase().includes(query));
    });
  }, [source, searchTerm]);

  return (
    <section className="panel panel-strong overflow-hidden">
      <div className="border-b border-accent/20 px-5 py-5 sm:px-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-start gap-3">
              <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-risk-medium/30 bg-risk-medium-soft text-risk-medium glow-med">
                <Bell className="h-5 w-5" />
              </div>
              <div>
                <div className="section-kicker">Alerts and history</div>
                <h3 className="mt-2 text-xl font-bold tracking-tight text-main font-display">Risk event feed</h3>
                <p className="mt-2 max-w-2xl text-sm text-muted">
                  A scannable list of active warnings and historical alerts with risk color, joint ID, type, and timestamp.
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={() => setActiveTab('active')} className={`rounded-full border px-4 py-2 text-xs font-bold transition font-display ${activeTab === 'active' ? 'border-accent bg-accent-soft text-accent glow-cyan' : 'border-surface bg-surface-raised text-muted hover:border-accent hover:text-main'}`}>
              Active ({activeAlerts.length})
            </button>
            <button type="button" onClick={() => setActiveTab('history')} className={`rounded-full border px-4 py-2 text-xs font-bold transition font-display ${activeTab === 'history' ? 'border-accent bg-accent-soft text-accent glow-cyan' : 'border-surface bg-surface-raised text-muted hover:border-accent hover:text-main'}`}>
              History ({alertHistory.length})
            </button>
          </div>
        </div>

        <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_auto]">
          <div className="relative">
            <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
            <input
              type="text"
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Filter by joint, alert type, or message text"
              className="input-shell w-full rounded-2xl py-3 pl-11 pr-4 text-sm outline-none transition focus:border-accent"
            />
          </div>
          <div className="glass-chip justify-center px-4 py-3 text-center">
            <CircleAlert className="h-3.5 w-3.5 text-muted" />
            {displayedAlerts.length} visible
          </div>
        </div>
      </div>

      <div className="max-h-[34rem] space-y-3 overflow-y-auto px-5 py-5 pr-3 sm:px-6">
        <AnimatePresence mode="popLayout">
          {!displayedAlerts.length ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="rounded-3xl border border-surface bg-surface px-6 py-10 text-center text-sm text-muted"
            >
              No alerts match the current filter.
            </motion.div>
          ) : (
            displayedAlerts.map((alert) => {
              const tone = alertTone(alert.risk_level);
              return (
                <motion.article
                  key={alert.id}
                  layout
                  initial={{ opacity: 0, x: -16, scale: 0.97 }}
                  animate={{ opacity: 1, x: 0, scale: 1 }}
                  exit={{ opacity: 0, x: 16, scale: 0.97 }}
                  transition={{ duration: 0.25, ease: 'easeOut' }}
                  className={`panel interactive-surface rounded-3xl border ${tone.border} px-4 py-4`}
                >
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div className="flex items-start gap-3">
                      <div className={`inline-flex h-11 w-11 items-center justify-center rounded-2xl border ${tone.chip}`}>
                        {tone.icon}
                      </div>
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-lg font-bold tracking-tight text-main font-display">{alert.joint_id}</span>
                          <span className={`rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.24em] font-display ${tone.chip}`}>{tone.label}</span>
                          <span className="rounded-full border border-surface bg-surface-raised px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.24em] text-muted font-display">
                            {alert.alert_type}
                          </span>
                        </div>
                        <p className="mt-2 max-w-4xl text-sm leading-relaxed text-main font-body">{alert.message}</p>
                      </div>
                    </div>

                    <div className="inline-flex items-center gap-2 self-start rounded-full border border-surface bg-surface-raised px-3 py-2 text-xs text-muted font-mono">
                      <Clock3 className="h-3.5 w-3.5 text-muted" />
                      {formatTime(alert.timestamp)}
                    </div>
                  </div>
                </motion.article>
              );
            })
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
