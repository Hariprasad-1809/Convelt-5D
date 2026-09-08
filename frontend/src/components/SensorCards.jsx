import React from 'react';
import { Eye, Magnet, Thermometer, Waves } from 'lucide-react';

function valueOrUnknown(value, suffix = '') {
  if (value === null || value === undefined) {
    return 'UNKNOWN';
  }
  return `${value}${suffix}`;
}

function channelCard({ title, value, subtitle, icon, accentClass, badge, rawValue, dimmed = false }) {
  return (
    <div className={`panel interactive-surface flex min-h-[11.5rem] flex-col justify-between border p-4 ${dimmed ? 'border-risk-unknown bg-risk-unknown-soft text-risk-unknown opacity-65' : accentClass}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <div className={`inline-flex h-11 w-11 items-center justify-center rounded-2xl border ${dimmed ? 'border-risk-unknown/30 bg-risk-unknown-soft' : accentClass}`}>
            {icon}
          </div>
          <div>
            <div className="text-sm font-bold text-main font-display">{title}</div>
            <div className="mt-1 text-xs text-muted font-body">{subtitle}</div>
          </div>
        </div>
        {badge ? <span className="simulated-badge px-2 py-1">SIMULATED</span> : null}
      </div>

      <div className="mt-3 rounded-3xl border border-surface bg-surface-raised px-4 py-4">
        <div className="meta-text">Value</div>
        <div className="mt-2 flex items-end justify-between gap-3">
          <div className={`metric-value text-3xl font-display ${value === 'UNKNOWN' || dimmed ? 'text-muted' : 'text-main'}`}>{value}</div>
          <div className="text-right text-xs text-muted font-mono">{rawValue}</div>
        </div>
      </div>
    </div>
  );
}

export default function SensorCards({ jointDetail }) {
  const sensors = jointDetail?.latest_sensors;
  const scores = jointDetail?.current_health?.component_scores;

  return (
    <section className="panel panel-strong overflow-hidden">
      <div className="border-b border-accent/20 px-5 py-5 sm:px-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="section-kicker">Sensor summary</div>
            <h3 className="mt-2 text-xl font-bold tracking-tight text-main font-display">Selected joint channels</h3>
            <p className="mt-2 max-w-2xl text-sm text-muted">
              The four monitored inputs are shown as dedicated cards with unknown-state handling and explicit simulated-field badges.
            </p>
          </div>
          <div className="glass-chip">Latest endpoint data</div>
        </div>
      </div>

      {!jointDetail ? (
        <div className="p-6">
          <div className="rounded-3xl border border-surface bg-surface-raised p-8 text-center text-sm text-muted">
            Sensor cards will appear after the selected joint detail payload arrives.
          </div>
        </div>
      ) : (
        <div className="grid gap-4 px-5 py-5 sm:px-6 lg:grid-cols-2">
          {channelCard({
            title: 'Vibration',
            subtitle: 'MPU6050 / I2C',
            value: valueOrUnknown(sensors?.vibration, ' m/s²'),
            rawValue: scores?.vibration_score !== null && scores?.vibration_score !== undefined ? `weighted ${scores.vibration_score}/100` : 'weighted UNKNOWN',
            icon: <Waves className="h-5 w-5 text-risk-medium" />,
            accentClass: 'border-risk-medium/30 bg-risk-medium-soft text-risk-medium',
            dimmed: sensors?.vibration === null || sensors?.vibration === undefined,
          })}

          {channelCard({
            title: 'Temperature',
            subtitle: 'DS18B20 / 1-Wire',
            value: valueOrUnknown(sensors?.temperature, ' °C'),
            rawValue: scores?.temperature_score !== null && scores?.temperature_score !== undefined ? `weighted ${scores.temperature_score}/100` : 'weighted UNKNOWN',
            icon: <Thermometer className="h-5 w-5 text-risk-high" />,
            accentClass: 'border-risk-high/30 bg-risk-high-soft text-risk-high',
            dimmed: sensors?.temperature === null || sensors?.temperature === undefined,
          })}

          {channelCard({
            title: 'Magnetic / Hall',
            subtitle: 'A3144 pulse marker',
            value: sensors?.hall_event === null || sensors?.hall_event === undefined ? 'UNKNOWN' : sensors.hall_event ? 'PULSE' : 'IDLE',
            rawValue: valueOrUnknown(sensors?.magnetic_value, ''),
            icon: <Magnet className="h-5 w-5 text-main" />,
            accentClass: 'border-surface bg-surface-raised text-main',
            dimmed: sensors?.hall_event === null || sensors?.hall_event === undefined,
          })}

          {channelCard({
            title: 'Vision',
            subtitle: 'Simulated / injected score',
            value: valueOrUnknown(sensors?.vision_score, '/100'),
            rawValue: scores?.vision_score !== null && scores?.vision_score !== undefined ? `weighted ${scores.vision_score}/100` : 'weighted UNKNOWN',
            icon: <Eye className="h-5 w-5 text-accent" />,
            accentClass: 'border-accent/30 bg-accent-soft text-accent',
            badge: true,
            dimmed: sensors?.vision_score === null || sensors?.vision_score === undefined,
          })}
        </div>
      )}
    </section>
  );
}
