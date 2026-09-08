import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, Magnet, Thermometer, Waves } from 'lucide-react';

function riskTone(risk) {
  switch (risk) {
    case 'LOW':
      return 'border-risk-low bg-risk-low-soft text-risk-low glow-low';
    case 'MEDIUM':
      return 'border-risk-medium bg-risk-medium-soft text-risk-medium glow-med';
    case 'HIGH':
      return 'border-risk-high bg-risk-high-soft text-risk-high glow-high';
    default:
      return 'border-risk-unknown bg-risk-unknown-soft text-risk-unknown';
  }
}

function riskLabel(risk) {
  switch (risk) {
    case 'LOW':
      return 'LOW risk';
    case 'MEDIUM':
      return 'MEDIUM risk';
    case 'HIGH':
      return 'HIGH risk';
    default:
      return 'UNKNOWN risk';
  }
}

function numericOrUnknown(value, suffix = '') {
  if (value === null || value === undefined) {
    return 'UNKNOWN';
  }
  return `${value}${suffix}`;
}

function componentCard({ title, subtitle, value, accentClass, icon, simulated = false, rawValue }) {
  return (
    <div className={`panel interactive-surface flex flex-col gap-3 border px-4 py-4 ${accentClass}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className={`inline-flex h-11 w-11 items-center justify-center rounded-2xl border ${accentClass}`}>
            {icon}
          </div>
          <div>
            <div className="text-sm font-bold text-main font-display">{title}</div>
            <div className="mt-1 text-xs text-muted font-body">{subtitle}</div>
          </div>
        </div>
        {simulated ? <span className="simulated-badge px-2 py-1">SIMULATED</span> : null}
      </div>
      <div className="flex items-end justify-between gap-3 border-t border-surface pt-3">
        <div>
          <div className={`metric-value text-3xl font-display ${value === 'UNKNOWN' ? 'text-muted' : 'text-main'}`}>{value}</div>
          <div className="meta-text mt-1">Raw value {rawValue}</div>
        </div>
        <div className="rounded-2xl border border-surface bg-surface-raised px-3 py-2 text-right">
          <div className="text-[10px] uppercase tracking-[0.28em] text-muted font-display">Weight</div>
          <div className="mt-1 font-mono text-sm font-bold text-main">{subtitle}</div>
        </div>
      </div>
    </div>
  );
}

export default function JointDetailModal({ jointDetail }) {
  const risk = jointDetail?.current_health?.risk_level || 'UNKNOWN';

  if (!jointDetail) {
    return (
      <section className="panel panel-strong min-h-[28rem] p-6">
        <div className="h-full min-h-[24rem] rounded-[1.35rem] border border-surface bg-surface-raised p-6 text-center">
          <div className="mx-auto mt-16 h-16 w-16 rounded-full border border-accent/30 bg-accent-soft text-accent glow-cyan animate-pulse" />
          <div className="mt-6 text-lg font-bold text-main font-display">Selected joint telemetry loading</div>
          <div className="mx-auto mt-2 max-w-md text-sm text-muted">
            Waiting for the REST detail endpoint to provide health, component scores, and latest sensor data.
          </div>
        </div>
      </section>
    );
  }

  const health = jointDetail.current_health;
  const sensors = jointDetail.latest_sensors;
  const comp = health?.component_scores;
  const score = health?.health_score;
  const scoreValue = typeof score === 'number' ? score : null;
  const scorePct = scoreValue === null ? 0 : Math.max(0, Math.min(scoreValue, 100));

  const gaugeClass = riskTone(risk);
  const conicColor = risk === 'LOW' ? '#10b981' : risk === 'MEDIUM' ? '#f59e0b' : risk === 'HIGH' ? '#f43f5e' : '#475569';

  return (
    <AnimatePresence mode="wait">
      <motion.section
        key={jointDetail.joint_id}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.25 }}
        className="panel panel-strong overflow-hidden"
      >
        <div className="flex flex-col gap-4 border-b border-accent/20 px-5 py-5 sm:px-6 xl:flex-row xl:items-start xl:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-3">
              <div>
                <div className="section-kicker">Selected joint detail</div>
                <div className="mt-2 flex flex-wrap items-center gap-3">
                  <h2 className="text-2xl font-bold tracking-tight text-main font-display sm:text-3xl">{jointDetail.joint_id}</h2>
                  <span className={`rounded-full border px-3 py-1 text-[10px] font-bold uppercase tracking-[0.24em] font-display ${gaugeClass}`}>{riskLabel(risk)}</span>
                </div>
              </div>
            </div>
            <div className="mt-3 text-base text-main font-body">{jointDetail.name}</div>
            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted">
              <span className="glass-chip">Position {jointDetail.belt_position}m</span>
              <span className="glass-chip">Zone {jointDetail.zone_state}</span>
              <span className="glass-chip">{health?.timestamp ? new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(health.timestamp)) : 'No timestamp'}</span>
              <span className={`glass-chip ${health?.is_sufficient_data ? 'border-risk-low bg-risk-low-soft text-risk-low' : 'border-risk-medium bg-risk-medium-soft text-risk-medium'}`}>
                {health?.is_sufficient_data ? '4/4 active channels' : 'Insufficient data'}
              </span>
              {sensors?.device_id ? <span className="glass-chip">Device {sensors.device_id}</span> : null}
            </div>
          </div>

          <div className={`panel flex w-full max-w-[22rem] items-center justify-center border p-4 sm:p-5 ${gaugeClass}`}>
            <div
              className="relative flex h-52 w-52 items-center justify-center rounded-full transition-all duration-500"
              style={{ background: `conic-gradient(${conicColor} ${scorePct}%, rgba(3, 13, 31, 0.95) 0)` }}
            >
              <div className="flex h-[11.5rem] w-[11.5rem] flex-col items-center justify-center rounded-full border border-surface bg-app text-center shadow-[inset_0_0_20px_rgba(0,0,0,0.8)]">
                <div className="section-kicker">Health score</div>
                <div className={`metric-value mt-2 text-5xl font-display ${scoreValue === null ? 'text-muted' : risk === 'LOW' ? 'text-risk-low' : risk === 'MEDIUM' ? 'text-risk-medium' : risk === 'HIGH' ? 'text-risk-high' : 'text-main'}`}>
                  {scoreValue === null ? 'N/A' : scoreValue}
                </div>
                <div className="mt-1 text-xs text-muted">out of 100</div>
                <div className="mt-4 rounded-full border border-surface bg-surface-raised px-3 py-1 text-[10px] font-bold uppercase tracking-[0.24em] text-muted font-display">
                  {riskLabel(risk)}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid gap-4 px-5 py-5 sm:px-6 xl:grid-cols-[0.95fr_1.05fr]">
          <div className="space-y-4">
            {componentCard({
              title: 'Vision surface score',
              subtitle: '40% of total',
              value: numericOrUnknown(comp?.vision_score, '/100'),
              rawValue: numericOrUnknown(sensors?.vision_score, '/100'),
              accentClass: 'border-accent/30 bg-accent-soft text-accent',
              icon: <Eye className="h-5 w-5 text-accent" />,
              simulated: true,
            })}
            {componentCard({
              title: 'Magnetic event score',
              subtitle: '30% of total',
              value: numericOrUnknown(comp?.magnetic_score, '/100'),
              rawValue: numericOrUnknown(sensors?.magnetic_value, ''),
              accentClass: 'border-surface bg-surface-raised text-main',
              icon: <Magnet className="h-5 w-5 text-main" />,
            })}
          </div>

          <div className="space-y-4">
            {componentCard({
              title: 'Vibration score',
              subtitle: '20% of total',
              value: numericOrUnknown(comp?.vibration_score, '/100'),
              rawValue: numericOrUnknown(sensors?.vibration, ' m/s²'),
              accentClass: 'border-risk-medium/30 bg-risk-medium-soft text-risk-medium',
              icon: <Waves className="h-5 w-5 text-risk-medium" />,
            })}
            {componentCard({
              title: 'Temperature score',
              subtitle: '10% of total',
              value: numericOrUnknown(comp?.temperature_score, '/100'),
              rawValue: numericOrUnknown(sensors?.temperature, ' °C'),
              accentClass: 'border-risk-high/30 bg-risk-high-soft text-risk-high',
              icon: <Thermometer className="h-5 w-5 text-risk-high" />,
            })}

            <div className="panel border border-surface bg-surface/60 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="section-kicker">Raw sensor snapshot</div>
                <span className="glass-chip">Latest sensor endpoint</span>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-surface bg-surface-raised px-4 py-3 text-sm text-main">
                  <div className="meta-text">Hall state</div>
                  <div className="mt-1 text-lg font-bold font-display">{sensors?.hall_event === null || sensors?.hall_event === undefined ? 'UNKNOWN' : sensors.hall_event ? 'PULSE' : 'IDLE'}</div>
                </div>
                <div className="rounded-2xl border border-surface bg-surface-raised px-4 py-3 text-sm text-main">
                  <div className="meta-text">Magnetic value</div>
                  <div className="mt-1 text-lg font-bold font-display">{numericOrUnknown(sensors?.magnetic_value, '')}</div>
                </div>
                <div className="rounded-2xl border border-surface bg-surface-raised px-4 py-3 text-sm text-main">
                  <div className="meta-text">Vibration raw</div>
                  <div className="mt-1 text-lg font-bold font-display">{numericOrUnknown(sensors?.vibration, ' m/s²')}</div>
                </div>
                <div className="rounded-2xl border border-surface bg-surface-raised px-4 py-3 text-sm text-main">
                  <div className="meta-text">Temperature raw</div>
                  <div className="mt-1 text-lg font-bold font-display">{numericOrUnknown(sensors?.temperature, ' °C')}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.section>
    </AnimatePresence>
  );
}
