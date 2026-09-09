import { useState } from 'react';
import {
  Play, Pause, SkipForward, RefreshCw, Sliders,
  Activity, Check,
} from 'lucide-react';
import { useSimulation } from '../../context/SimulationContext';
import { JOINT_IDS } from '../../data/constants';
import Modal from '../ui/Modal';

export default function SimulationControls() {
  const {
    simStatus,
    apiConnected,
    startSimulation,
    stopSimulation,
    resetSimulation,
    stepSimulation,
    injectOverride,
  } = useSimulation();

  const [showInjectModal, setShowInjectModal] = useState(false);
  const [targetJoint, setTargetJoint] = useState('J01');
  const [vibrationVal, setVibrationVal] = useState('');
  const [tempVal, setTempVal] = useState('');
  const [visionVal, setVisionVal] = useState('');
  const [hallVal, setHallVal] = useState('default');
  const [statusMessage, setStatusMessage] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [pendingAction, setPendingAction] = useState(null);

  const isRunning = simStatus?.is_running ?? false;
  const currentCycle = simStatus?.current_cycle ?? 0;

  const triggerAction = async (label, actionFn) => {
    setPendingAction(label);
    setStatusMessage(`${label} triggered...`);
    try {
      await actionFn();
      setStatusMessage(`${label} completed`);
    } catch (err) {
      console.error(err);
      setStatusMessage(`${label} failed`);
    } finally {
      setPendingAction(null);
      setTimeout(() => setStatusMessage(''), 2500);
    }
  };

  const handleInjectSubmit = async (e) => {
    e.preventDefault();
    const payload = {
      joint_id: targetJoint,
      vibration: vibrationVal !== '' ? parseFloat(vibrationVal) : undefined,
      temperature: tempVal !== '' ? parseFloat(tempVal) : undefined,
      vision_score: visionVal !== '' ? parseFloat(visionVal) : undefined,
      hall_event:
        hallVal === 'true'
          ? true
          : hallVal === 'false'
            ? false
            : hallVal === 'none'
              ? null
              : undefined,
    };

    setIsSubmitting(true);
    try {
      await injectOverride(payload);
      setStatusMessage(`Injected override into ${targetJoint}`);
      setShowInjectModal(false);
      setTimeout(() => setStatusMessage(''), 2500);
    } catch (err) {
      console.error(err);
      setStatusMessage('Override injection failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      className="card"
      style={{
        marginBottom: '20px',
        padding: '16px 20px',
        borderLeft: isRunning
          ? '3px solid var(--status-normal)'
          : '3px solid var(--status-medium)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '14px',
        }}
      >
        {/* Left: Engine status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span
              style={{
                display: 'inline-block',
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: isRunning ? 'var(--status-normal)' : 'var(--status-medium)',
                boxShadow: isRunning
                  ? '0 0 10px rgba(34, 197, 94, 0.6)'
                  : '0 0 10px rgba(245, 158, 11, 0.4)',
              }}
            />
            <div>
              <div style={{ fontSize: 'var(--text-sm)', fontWeight: 700, color: 'var(--text-primary)' }}>
                {isRunning ? 'Simulation Engine Active' : 'Simulation Engine Paused'}
              </div>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                {isRunning ? 'Automated inspection cycles running' : 'Manual step & inspection mode'}
              </div>
            </div>
          </div>

          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '3px 10px',
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              fontSize: 'var(--text-xs)',
              fontFamily: 'var(--font-mono)',
              color: 'var(--accent)',
            }}
          >
            <Activity size={12} />
            Cycle {currentCycle}
          </div>

          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '3px 10px',
              background: apiConnected ? 'rgba(34, 197, 94, 0.08)' : 'rgba(239, 68, 68, 0.08)',
              border: apiConnected
                ? '1px solid rgba(34, 197, 94, 0.25)'
                : '1px solid rgba(239, 68, 68, 0.25)',
              borderRadius: 'var(--radius-sm)',
              fontSize: '11px',
              fontWeight: 600,
              color: apiConnected ? 'var(--status-normal)' : 'var(--status-high)',
            }}
          >
            {apiConnected ? 'REST Connected' : 'REST Offline'}
          </div>

          {statusMessage && (
            <div
              style={{
                fontSize: 'var(--text-xs)',
                color: 'var(--accent)',
                padding: '2px 8px',
                background: 'var(--accent-dim)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--accent-border)',
              }}
            >
              {statusMessage}
            </div>
          )}
        </div>

        {/* Right: Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          {isRunning ? (
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => triggerAction('Pause', stopSimulation)}
              disabled={pendingAction !== null}
              title="Pause automated cycle loop"
            >
              <Pause size={13} />
              Pause
            </button>
          ) : (
            <button
              className="btn btn-primary btn-sm"
              onClick={() => triggerAction('Start', startSimulation)}
              disabled={pendingAction !== null}
              title="Start automated cycle loop"
            >
              <Play size={13} />
              Start
            </button>
          )}

          <button
            className="btn btn-secondary btn-sm"
            onClick={() => triggerAction('Step', stepSimulation)}
            disabled={pendingAction !== null}
            title="Step simulation by 1 cycle immediately"
          >
            <SkipForward size={13} />
            Step
          </button>

          <button
            className="btn btn-ghost btn-sm"
            onClick={() => triggerAction('Reset', resetSimulation)}
            disabled={pendingAction !== null}
            title="Reset cycle counter and clear overrides"
          >
            <RefreshCw size={13} />
            Reset
          </button>

          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setShowInjectModal(true)}
            style={{
              borderColor: 'var(--accent-border)',
              color: 'var(--accent)',
              background: 'var(--accent-dim)',
            }}
            title="Inject manual telemetry or sensor fault"
          >
            <Sliders size={13} />
            Inject Override
          </button>
        </div>
      </div>

      {/* ─── Inject Override Modal ─── */}
      <Modal
        isOpen={showInjectModal}
        onClose={() => setShowInjectModal(false)}
        title="Simulated Telemetry Override"
        size="md"
      >
        <form onSubmit={handleInjectSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div
            style={{
              padding: '10px 14px',
              background: 'rgba(245, 158, 11, 0.08)',
              border: '1px solid rgba(245, 158, 11, 0.25)',
              borderRadius: 'var(--radius)',
              fontSize: 'var(--text-xs)',
              color: 'var(--status-medium)',
              lineHeight: 1.6,
            }}
          >
            <strong>Phase 1 Injection Hook:</strong> Injects manual sensor telemetry directly into the backend
            simulation engine (POST /simulation/inject) to test threshold alarms, component weighted fusion, and fault responses.
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 'var(--text-xs)',
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  marginBottom: 6,
                  textTransform: 'uppercase',
                }}
              >
                Target Joint
              </label>
              <select
                className="select"
                value={targetJoint}
                onChange={(e) => setTargetJoint(e.target.value)}
                style={{ width: '100%' }}
              >
                {JOINT_IDS.map((id) => (
                  <option key={id} value={id}>
                    {id} — Conveyor Joint
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 'var(--text-xs)',
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  marginBottom: 6,
                  textTransform: 'uppercase',
                }}
              >
                Vibration RMS (m/s²)
              </label>
              <input
                type="number"
                step="0.01"
                className="input"
                value={vibrationVal}
                onChange={(e) => setVibrationVal(e.target.value)}
                placeholder="e.g. 5.80"
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 'var(--text-xs)',
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  marginBottom: 6,
                  textTransform: 'uppercase',
                }}
              >
                Temperature (°C)
              </label>
              <input
                type="number"
                step="0.1"
                className="input"
                value={tempVal}
                onChange={(e) => setTempVal(e.target.value)}
                placeholder="e.g. 64.5"
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 'var(--text-xs)',
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  marginBottom: 6,
                  textTransform: 'uppercase',
                }}
              >
                Vision Score (0–100)
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="100"
                className="input"
                value={visionVal}
                onChange={(e) => setVisionVal(e.target.value)}
                placeholder="e.g. 35.0"
                style={{ width: '100%' }}
              />
            </div>
          </div>

          <div>
            <label
              style={{
                display: 'block',
                fontSize: 'var(--text-xs)',
                fontWeight: 600,
                color: 'var(--text-muted)',
                marginBottom: 6,
                textTransform: 'uppercase',
              }}
            >
              Hall Pulse Event (A3144)
            </label>
            <select
              className="select"
              value={hallVal}
              onChange={(e) => setHallVal(e.target.value)}
              style={{ width: '100%' }}
            >
              <option value="default">Default (Normal Magnet Passing)</option>
              <option value="true">Force Pulse Detected (True)</option>
              <option value="false">Force Pulse Missing (False)</option>
              <option value="none">Force Disconnected / Unknown (Null)</option>
            </select>
          </div>

          <div
            style={{
              display: 'flex',
              justifyContent: 'flex-end',
              alignItems: 'center',
              gap: '10px',
              paddingTop: '8px',
              borderTop: '1px solid var(--border-subtle)',
            }}
          >
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => setShowInjectModal(false)}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="btn btn-primary"
              style={{ minWidth: '130px', justifyContent: 'center' }}
            >
              <Check size={14} />
              {isSubmitting ? 'Injecting...' : 'Apply Override'}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
