import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, Pause, Play, RefreshCw, SkipForward, Sliders, X } from 'lucide-react';

export default function SimulationControls({ isRunning, onStart, onStop, onReset, onStep, onInject }) {
  const [showInjectModal, setShowInjectModal] = useState(false);
  const [targetJoint, setTargetJoint] = useState('J01');
  const [vibrationVal, setVibrationVal] = useState('');
  const [tempVal, setTempVal] = useState('');
  const [visionVal, setVisionVal] = useState('');
  const [hallVal, setHallVal] = useState('default');
  const [statusMessage, setStatusMessage] = useState('');
  const [pendingAction, setPendingAction] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const triggerAction = async (label, action) => {
    setPendingAction(label);
    setStatusMessage(`${label} sent`);
    try {
      await action();
      setStatusMessage(`${label} completed`);
    } catch (error) {
      console.error(error);
      setStatusMessage(`${label} failed`);
    } finally {
      setPendingAction(null);
      setTimeout(() => setStatusMessage(''), 1800);
    }
  };

  const handleInjectSubmit = async (event) => {
    event.preventDefault();
    const payload = {
      joint_id: targetJoint,
      vibration: vibrationVal !== '' ? parseFloat(vibrationVal) : undefined,
      temperature: tempVal !== '' ? parseFloat(tempVal) : undefined,
      vision_score: visionVal !== '' ? parseFloat(visionVal) : undefined,
      hall_event: hallVal === 'true' ? true : hallVal === 'false' ? false : hallVal === 'none' ? null : undefined,
    };

    setIsSubmitting(true);
    try {
      await onInject(payload);
      setStatusMessage(`Injected into ${targetJoint}`);
      setShowInjectModal(false);
      setTimeout(() => setStatusMessage(''), 1800);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="panel panel-strong overflow-hidden scan-line">
      <div className="flex flex-col gap-4 border-b border-accent/20 px-5 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-3">
          <div className={`h-3.5 w-3.5 rounded-full ${isRunning ? 'bg-risk-low glow-low' : 'bg-risk-high glow-high'}`} />
          <div>
            <div className="section-kicker">Simulation engine</div>
            <div className="mt-1 text-lg font-bold tracking-tight text-main font-display">
              {isRunning ? 'Running auto cycle' : 'Paused'}
            </div>
            <div className="mt-1 text-sm text-muted">REST-driven conveyor state machine with manual stepping and injection hooks.</div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            type="button"
            onClick={() => triggerAction('Start', onStart)}
            disabled={pendingAction !== null}
            className="button-accent inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-bold disabled:opacity-60 font-display"
          >
            <Play className="h-3.5 w-3.5" />
            Start
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            type="button"
            onClick={() => triggerAction('Stop', onStop)}
            disabled={pendingAction !== null}
            className="button-accent inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-bold disabled:opacity-60 font-display"
          >
            <Pause className="h-3.5 w-3.5" />
            Stop
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            type="button"
            onClick={() => triggerAction('Step', onStep)}
            disabled={pendingAction !== null}
            className="button-accent inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-bold disabled:opacity-60 font-display"
          >
            <SkipForward className="h-3.5 w-3.5" />
            Step
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            type="button"
            onClick={() => triggerAction('Reset', onReset)}
            disabled={pendingAction !== null}
            className="button-secondary inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-bold disabled:opacity-60 font-display"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Reset
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            type="button"
            onClick={() => setShowInjectModal(true)}
            className="button-accent inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-bold font-display glow-cyan"
          >
            <Sliders className="h-3.5 w-3.5" />
            Inject override
          </motion.button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 px-5 py-4 text-xs text-muted sm:px-6">
        <span className={`glass-chip ${isRunning ? 'border-risk-low bg-risk-low-soft text-risk-low' : 'border-risk-high bg-risk-high-soft text-risk-high'}`}>
          {isRunning ? 'Auto cycle enabled' : 'Manual control enabled'}
        </span>
        <span className="glass-chip">Joint stepping and injection use unchanged REST endpoints</span>
        {statusMessage ? <span className="glass-chip border-accent bg-accent-soft text-accent">{statusMessage}</span> : null}
        {pendingAction ? <span className="glass-chip">Processing {pendingAction.toLowerCase()}...</span> : null}
      </div>

      <AnimatePresence>
        {showInjectModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-app/80 px-4 py-6 backdrop-blur-md">
            <motion.div
              initial={{ opacity: 0, scale: 0.94, y: 12 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.94, y: 12 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
              className="panel panel-strong w-full max-w-2xl overflow-hidden border border-accent/30 shadow-[0_30px_80px_rgba(0,0,0,0.85)] glow-cyan"
            >
              <div className="flex items-start justify-between gap-3 border-b border-accent/20 px-5 py-5 sm:px-6">
                <div>
                  <div className="section-kicker">Manual injection</div>
                  <h3 className="mt-2 text-xl font-bold tracking-tight text-main font-display">Simulated telemetry override</h3>
                  <p className="mt-2 text-sm text-muted">This writes directly to the existing POST /simulation/inject contract.</p>
                </div>
                <button
                  type="button"
                  onClick={() => setShowInjectModal(false)}
                  className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-surface bg-surface-raised text-muted transition hover:border-accent hover:text-accent"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="border-b border-accent/20 px-5 py-4 sm:px-6">
                <span className="simulated-badge px-3 py-2">SIMULATED DATA</span>
              </div>

              <form onSubmit={handleInjectSubmit} className="grid gap-4 px-5 py-5 sm:px-6 md:grid-cols-2">
                <label className="space-y-2 text-sm text-main">
                  <span className="section-kicker">Target joint</span>
                  <select
                    value={targetJoint}
                    onChange={(event) => setTargetJoint(event.target.value)}
                    className="input-shell w-full rounded-2xl px-4 py-3 text-sm outline-none transition focus:border-accent font-mono"
                  >
                    <option value="J01">J01</option>
                    <option value="J02">J02</option>
                    <option value="J03">J03</option>
                    <option value="J04">J04</option>
                    <option value="J05">J05</option>
                  </select>
                </label>

                <label className="space-y-2 text-sm text-main">
                  <span className="section-kicker">Vibration RMS (m/s²)</span>
                  <input
                    type="number"
                    step="0.01"
                    value={vibrationVal}
                    onChange={(event) => setVibrationVal(event.target.value)}
                    placeholder="0.85"
                    className="input-shell w-full rounded-2xl px-4 py-3 text-sm outline-none transition focus:border-accent font-mono"
                  />
                </label>

                <label className="space-y-2 text-sm text-main">
                  <span className="section-kicker">Temperature (°C)</span>
                  <input
                    type="number"
                    step="0.1"
                    value={tempVal}
                    onChange={(event) => setTempVal(event.target.value)}
                    placeholder="62.5"
                    className="input-shell w-full rounded-2xl px-4 py-3 text-sm outline-none transition focus:border-accent font-mono"
                  />
                </label>

                <label className="space-y-2 text-sm text-main">
                  <span className="section-kicker">Vision score</span>
                  <input
                    type="number"
                    step="0.1"
                    value={visionVal}
                    onChange={(event) => setVisionVal(event.target.value)}
                    placeholder="35.0"
                    className="input-shell w-full rounded-2xl px-4 py-3 text-sm outline-none transition focus:border-accent font-mono"
                  />
                </label>

                <label className="space-y-2 text-sm text-main md:col-span-2">
                  <span className="section-kicker">Hall pulse state</span>
                  <select
                    value={hallVal}
                    onChange={(event) => setHallVal(event.target.value)}
                    className="input-shell w-full rounded-2xl px-4 py-3 text-sm outline-none transition focus:border-accent font-mono"
                  >
                    <option value="default">Default</option>
                    <option value="true">Force pulse detected</option>
                    <option value="false">Force pulse missing</option>
                    <option value="none">Force disconnected / unknown</option>
                  </select>
                </label>

                <div className="md:col-span-2 flex items-center justify-between gap-3 border-t border-surface pt-4">
                  <div className="text-xs text-muted">The override closes automatically after a successful POST.</div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setShowInjectModal(false)}
                      className="button-secondary rounded-full px-4 py-2 text-xs font-bold font-display"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="button-accent inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-bold disabled:opacity-60 font-display glow-cyan"
                    >
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      {isSubmitting ? 'Injecting...' : 'Inject override'}
                    </button>
                  </div>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </section>
  );
}
