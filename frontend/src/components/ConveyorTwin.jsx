import React from 'react';
import { motion } from 'framer-motion';
import { Activity, Shield, ShieldAlert } from 'lucide-react';

function riskClasses(risk) {
  switch (risk) {
    case 'LOW':
      return 'border-risk-low text-risk-low bg-risk-low-soft';
    case 'MEDIUM':
      return 'border-risk-medium text-risk-medium bg-risk-medium-soft';
    case 'HIGH':
      return 'border-risk-high text-risk-high bg-risk-high-soft';
    default:
      return 'border-risk-unknown text-risk-unknown bg-risk-unknown-soft';
  }
}

function zoneTone(zone) {
  switch (zone) {
    case 'APPROACHING':
      return 'border-accent bg-accent-soft text-accent shadow-[0_0_18px_rgba(0,212,255,0.15)]';
    case 'INSPECTING':
      return 'border-accent bg-accent-soft text-accent glow-cyan';
    case 'PASSED':
      return 'border-risk-low bg-risk-low-soft text-risk-low';
    default:
      return 'border-risk-unknown bg-risk-unknown-soft text-risk-unknown';
  }
}

function getStateLabel(zone) {
  switch (zone) {
    case 'APPROACHING':
      return 'Approaching inspection zone';
    case 'INSPECTING':
      return 'Inside inspection zone';
    case 'PASSED':
      return 'Past inspection zone';
    default:
      return 'State unknown';
  }
}

export default function ConveyorTwin({ joints, selectedJointId, onSelectJoint, isRunning }) {
  return (
    <section className="panel panel-strong overflow-hidden scan-line">
      <div className="flex flex-col gap-4 border-b border-accent/20 px-5 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-start gap-3">
          <div className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-accent/30 bg-accent-soft text-accent glow-cyan">
            <Activity className={`h-5 w-5 ${isRunning ? 'animate-pulse' : ''}`} />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="text-xl font-bold tracking-tight text-main font-display sm:text-2xl">Digital-twin conveyor</h2>
              <span className="glass-chip border-accent/30 text-accent">J01-J05 live sonar map</span>
            </div>
            <p className="mt-1 max-w-3xl text-sm text-muted">
              Joints progress through APPROACHING, INSPECTING, and PASSED states while risk is color-coded for SCADA visual inspection.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
          <span className="glass-chip inline-flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-accent" />
            APPROACHING
          </span>
          <span className="glass-chip inline-flex items-center gap-2 border-accent/40 text-accent">
            <span className="h-2 w-2 rounded-full bg-accent shadow-[0_0_12px_rgba(0,212,255,0.8)]" />
            INSPECTING
          </span>
          <span className="glass-chip inline-flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-risk-low" />
            PASSED
          </span>
          <span className={`glass-chip inline-flex items-center gap-2 ${isRunning ? 'border-risk-low bg-risk-low-soft text-risk-low' : 'border-risk-unknown bg-risk-unknown-soft text-risk-unknown'}`}>
            {isRunning ? 'Live motion' : 'Paused'}
          </span>
        </div>
      </div>

      <div className="conveyor-rail px-5 py-5 sm:px-6">
        <div className="mb-5 rounded-3xl border border-accent/20 bg-surface p-4">
          <div className={`conveyor-track ${isRunning ? '' : 'paused'} h-4 rounded-full opacity-90`} />
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <div className="rounded-2xl border border-surface bg-surface-raised px-4 py-3">
              <div className="section-kicker">Approaching</div>
              <div className="mt-1 text-sm text-main">Before the inspection zone</div>
            </div>
            <div className="rounded-2xl border border-accent/30 bg-accent-soft px-4 py-3 glow-cyan">
              <div className="section-kicker text-accent">Inspecting</div>
              <div className="mt-1 text-sm text-main">Active sensor and vision window</div>
            </div>
            <div className="rounded-2xl border border-risk-low/40 bg-risk-low-soft px-4 py-3">
              <div className="section-kicker text-risk-low">Passed</div>
              <div className="mt-1 text-sm text-main">Exited the inspection zone</div>
            </div>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-5">
          {joints.map((joint, index) => {
            const selected = joint.joint_id === selectedJointId;
            const zone = joint.zone_state || 'UNKNOWN';
            const risk = joint.risk_level || 'UNKNOWN';
            const isInspecting = zone === 'INSPECTING';

            const glowClass = selected
              ? risk === 'LOW'
                ? 'glow-low border-risk-low'
                : risk === 'MEDIUM'
                ? 'glow-med border-risk-medium'
                : risk === 'HIGH'
                ? 'glow-high border-risk-high'
                : 'glow-cyan border-accent'
              : 'hover:border-accent/40';

            return (
              <motion.button
                key={joint.joint_id}
                type="button"
                layout
                transition={{ duration: 0.35, ease: 'easeOut' }}
                onClick={() => onSelectJoint(joint.joint_id)}
                className={`relative panel interactive-surface flex min-h-[15.5rem] flex-col justify-between border p-4 text-left outline-none ${riskClasses(risk)} ${glowClass} ${selected ? 'ring-1 ring-accent/60 scale-[1.015]' : ''}`}
              >
                {/* Sonar Pulse animation for currently INSPECTING joint */}
                {isInspecting && (
                  <motion.div
                    className="pointer-events-none absolute inset-0 rounded-[1.4rem] border-2 border-accent"
                    initial={{ scale: 0.98, opacity: 0.9 }}
                    animate={{ scale: [0.98, 1.08, 1.14], opacity: [0.9, 0.4, 0] }}
                    transition={{ duration: 1.8, repeat: Infinity, ease: 'easeOut' }}
                  />
                )}

                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.28em] text-muted font-display">Joint {String(index + 1).padStart(2, '0')}</div>
                    <div className="mt-1 text-2xl font-bold tracking-tight text-main font-display">{joint.joint_id}</div>
                    <div className="mt-1 text-sm text-muted">{joint.name}</div>
                  </div>
                  <div className={`relative z-10 rounded-full border px-3 py-1 text-[10px] font-bold uppercase tracking-[0.24em] font-display ${zoneTone(zone)}`}>
                    {zone}
                  </div>
                </div>

                <div className="mt-4 space-y-3">
                  <div className="rounded-3xl border border-surface bg-slate-950/70 p-4">
                    <div className="section-kicker">Health score</div>
                    <div className="mt-2 flex items-end gap-2">
                      <div className={`metric-value text-4xl font-display ${risk === 'LOW' ? 'text-risk-low' : risk === 'MEDIUM' ? 'text-risk-medium' : risk === 'HIGH' ? 'text-risk-high' : 'text-risk-unknown'}`}>
                        {joint.health_score ?? '—'}
                      </div>
                      <div className="pb-1 text-xs text-muted">/ 100</div>
                    </div>
                  </div>

                  <div className={`rounded-2xl border px-3 py-2 text-xs font-semibold ${risk === 'LOW' ? 'border-risk-low bg-risk-low-soft text-risk-low' : risk === 'MEDIUM' ? 'border-risk-medium bg-risk-medium-soft text-risk-medium' : risk === 'HIGH' ? 'border-risk-high bg-risk-high-soft text-risk-high' : 'border-risk-unknown bg-risk-unknown-soft text-risk-unknown'}`}>
                    {risk} risk
                  </div>
                </div>

                <div className="mt-4 space-y-2 text-xs text-muted">
                  <div className="flex items-center justify-between gap-3 border-t border-surface pt-3">
                    <span className="meta-text">Position</span>
                    <span className="font-mono text-main">{joint.belt_position}m</span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="meta-text">State</span>
                    <span className="font-mono text-main">{getStateLabel(zone)}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="meta-text">Updated</span>
                    <span className="font-mono text-main">
                      {joint.last_updated ? new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(joint.last_updated)) : '—'}
                    </span>
                  </div>
                  {joint.alert_type && joint.alert_type !== 'NONE' ? (
                    <div className="inline-flex items-center gap-2 rounded-full border border-risk-high bg-risk-high-soft px-3 py-1 text-[10px] font-bold uppercase tracking-[0.24em] text-risk-high">
                      <ShieldAlert className="h-3.5 w-3.5" />
                      Alert active
                    </div>
                  ) : (
                    <div className="inline-flex items-center gap-2 rounded-full border border-surface bg-surface-raised px-3 py-1 text-[10px] font-bold uppercase tracking-[0.24em] text-main">
                      <Shield className="h-3.5 w-3.5" />
                      Monitoring
                    </div>
                  )}
                </div>
              </motion.button>
            );
          })}
        </div>

        {!joints.length && (
          <div className="mt-4 rounded-3xl border border-surface bg-surface px-8 py-10 text-center text-sm text-muted">
            Conveyor telemetry data is loading...
          </div>
        )}
      </div>
    </section>
  );
}
