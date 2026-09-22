import React, { useState } from 'react';
import {
  ShieldCheck, ShieldAlert, AlertTriangle, CheckCircle2,
  Play, Square, RefreshCw, Cpu, Activity, Zap, AlertOctagon
} from 'lucide-react';
import { useSimulation } from '../../context/SimulationContext';

export default function VisionResultPanel({
  visionData: propVisionData,
  hardwareStatus: propHardwareStatus,
  resumeMotor: propResumeMotor,
  stopMotor: propStopMotor,
  cameraStatus: propCameraStatus
}) {
  const context = useSimulation();

  const visionData = propVisionData || context.visionData || {};
  const hardwareStatus = propHardwareStatus || context.hardwareStatus || {};
  const resumeMotor = propResumeMotor || context.resumeMotor;
  const stopMotor = propStopMotor || context.stopMotor;
  const cameraStatus = propCameraStatus || context.cameraStatus || 'DISCONNECTED';

  const [resuming, setResuming] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [actionMessage, setActionMessage] = useState(null);

  const label = visionData.label || 'WAITING';
  const confidence = visionData.confidence != null ? (visionData.confidence * 100).toFixed(1) : '0.0';
  const jointId = visionData.joint_id || 'J01';
  const inspectionState = visionData.inspection_state || 'WAITING';
  const pHealthy = visionData.p_healthy != null ? (visionData.p_healthy * 100).toFixed(1) : '0.0';
  const pDamage = visionData.p_damage != null ? (visionData.p_damage * 100).toFixed(1) : '0.0';
  const validFrames = visionData.valid_frame_count || 0;
  const rejectedFrames = visionData.rejected_blurry_count || 0;
  const sharpness = visionData.sharpness != null ? visionData.sharpness.toFixed(1) : '0.0';
  const fps = visionData.fps || 0;
  const latency = visionData.latency_ms || 0;

  const isMotorRunning = hardwareStatus.motor_running;
  const isStopTriggered = visionData.motor_stop_triggered || (label === 'DAMAGE' && !isMotorRunning);
  const isHardwareConfirmed = visionData.motor_stop_confirmed || (hardwareStatus.motor_running === false && isStopTriggered);
  const cmdStatus = visionData.motor_command_status || (isStopTriggered ? 'CONFIRMED' : 'IDLE');

  const handleResume = async () => {
    if (!resumeMotor) return;
    setResuming(true);
    setActionMessage(null);
    try {
      const res = await resumeMotor();
      if (res?.status === 'ok') {
        setActionMessage('Conveyor motor resumed successfully');
      } else {
        setActionMessage(res?.message || 'Resume signal sent');
      }
    } catch (e) {
      setActionMessage('Failed to resume motor');
    } finally {
      setResuming(false);
      setTimeout(() => setActionMessage(null), 4000);
    }
  };

  const handleStop = async () => {
    if (!stopMotor) return;
    setStopping(true);
    setActionMessage(null);
    try {
      const res = await stopMotor();
      if (res?.status === 'ok') {
        setActionMessage('Conveyor motor stopped');
      } else {
        setActionMessage(res?.message || 'Stop signal sent');
      }
    } catch (e) {
      setActionMessage('Failed to stop motor');
    } finally {
      setStopping(false);
      setTimeout(() => setActionMessage(null), 4000);
    }
  };

  // Verdict style config
  const getVerdictStyle = () => {
    switch (label) {
      case 'HEALTHY':
        return {
          bg: 'rgba(34, 197, 94, 0.12)',
          border: 'rgba(34, 197, 94, 0.35)',
          color: '#4ade80',
          badgeText: 'HEALTHY JOINT',
          icon: CheckCircle2,
        };
      case 'DAMAGE':
        return {
          bg: 'rgba(239, 68, 68, 0.15)',
          border: 'rgba(239, 68, 68, 0.45)',
          color: '#f87171',
          badgeText: 'DAMAGE DETECTED',
          icon: ShieldAlert,
        };
      case 'UNCERTAIN':
        return {
          bg: 'rgba(245, 158, 11, 0.12)',
          border: 'rgba(245, 158, 11, 0.35)',
          color: '#fbbf24',
          badgeText: 'UNCERTAIN VERDICT',
          icon: AlertTriangle,
        };
      default:
        return {
          bg: 'rgba(100, 116, 139, 0.12)',
          border: 'rgba(100, 116, 139, 0.3)',
          color: '#94a3b8',
          badgeText: 'WAITING FOR JOINT',
          icon: Activity,
        };
    }
  };

  const verdict = getVerdictStyle();
  const VerdictIcon = verdict.icon;

  return (
    <div
      className="card"
      style={{
        marginBottom: '20px',
        padding: '20px',
        background: 'var(--card-bg, #1e293b)',
        borderRadius: '12px',
        border: isStopTriggered ? '1px solid rgba(239, 68, 68, 0.6)' : '1px solid var(--border-subtle, #334155)',
        boxShadow: isStopTriggered ? '0 0 20px rgba(239, 68, 68, 0.2)' : '0 4px 6px -1px rgba(0, 0, 0, 0.15)',
        transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
      }}
    >
      {/* 1. Header Bar with System Status Pills */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          marginBottom: '18px',
          borderBottom: '1px solid var(--border-subtle, #334155)',
          paddingBottom: '14px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              padding: '8px',
              borderRadius: '8px',
              background: 'rgba(99, 102, 241, 0.15)',
              color: '#818cf8',
              display: 'flex',
            }}
          >
            <Cpu size={20} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary, #f8fafc)' }}>
                YOLOv8 Edge Vision Inspection
              </h3>
              <span
                style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  padding: '2px 8px',
                  borderRadius: '100px',
                  background: 'rgba(56, 189, 248, 0.12)',
                  color: '#38bdf8',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                }}
              >
                AUTOMATED MOTOR INTERLOCK
              </span>
            </div>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted, #94a3b8)' }}>
              Temporal 5-Frame Inspection · OpenCV ROI Tracking · Zero-Lag Decoupled Backend Pipeline
            </span>
          </div>
        </div>

        {/* Status Indicators */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          {/* Camera Status */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '20px',
              fontSize: '11px',
              fontWeight: 600,
              background: cameraStatus === 'CONNECTED' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              color: cameraStatus === 'CONNECTED' ? '#4ade80' : '#f87171',
              border: `1px solid ${cameraStatus === 'CONNECTED' ? 'rgba(34, 197, 94, 0.25)' : 'rgba(239, 68, 68, 0.25)'}`,
            }}
          >
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: cameraStatus === 'CONNECTED' ? '#4ade80' : '#f87171',
                boxShadow: cameraStatus === 'CONNECTED' ? '0 0 6px #4ade80' : 'none',
              }}
            />
            {cameraStatus === 'CONNECTED' ? 'CAMERA ACTIVE' : 'CAMERA OFFLINE'}
          </div>

          {/* Inference Speed */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '20px',
              fontSize: '11px',
              fontFamily: 'var(--font-mono, monospace)',
              background: 'var(--bg-elevated, #0f172a)',
              color: '#38bdf8',
              border: '1px solid var(--border-subtle, #334155)',
            }}
            title="Backend processing framerate & latency"
          >
            <Zap size={12} />
            <span>{fps > 0 ? `${fps.toFixed(1)} FPS` : 'STANDBY'}</span>
            <span style={{ color: 'var(--text-muted, #64748b)' }}>·</span>
            <span>{latency > 0 ? `${latency}ms` : '0ms'}</span>
          </div>

          {/* Motor Status */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              borderRadius: '20px',
              fontSize: '11px',
              fontWeight: 600,
              background: isMotorRunning ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              color: isMotorRunning ? '#4ade80' : '#f87171',
              border: `1px solid ${isMotorRunning ? 'rgba(34, 197, 94, 0.25)' : 'rgba(239, 68, 68, 0.25)'}`,
            }}
          >
            <Activity size={12} />
            <span>MOTOR: {isMotorRunning ? 'RUNNING' : 'STOPPED'}</span>
          </div>
        </div>
      </div>

      {/* 2. Critical Motor Stopped Banner (Appears when Damage is Confirmed) */}
      {isStopTriggered && (
        <div
          style={{
            background: 'linear-gradient(90deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.25) 100%)',
            border: '1px solid rgba(239, 68, 68, 0.6)',
            borderRadius: '10px',
            padding: '16px 20px',
            marginBottom: '18px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '14px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div
              style={{
                width: '42px',
                height: '42px',
                borderRadius: '50%',
                background: 'rgba(239, 68, 68, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#f87171',
                flexShrink: 0,
              }}
            >
              <AlertOctagon size={24} />
            </div>
            <div>
              <div style={{ fontSize: '1rem', fontWeight: 700, color: '#fca5a5', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                <span>🚨 DAMAGED JOINT DETECTED ON {jointId}</span>
                <span
                  style={{
                    fontSize: '11px',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    background: '#ef4444',
                    color: '#ffffff',
                    fontWeight: 700,
                  }}
                >
                  MOTOR INTERLOCK TRIPPED
                </span>
                {isHardwareConfirmed ? (
                  <span
                    style={{
                      fontSize: '11px',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      background: 'rgba(34, 197, 94, 0.2)',
                      border: '1px solid rgba(34, 197, 94, 0.4)',
                      color: '#4ade80',
                      fontWeight: 700,
                    }}
                  >
                    ✔ HARDWARE STOP CONFIRMED
                  </span>
                ) : cmdStatus === 'SENT' ? (
                  <span
                    style={{
                      fontSize: '11px',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      background: 'rgba(251, 191, 36, 0.2)',
                      border: '1px solid rgba(251, 191, 36, 0.4)',
                      color: '#fbbf24',
                      fontWeight: 700,
                    }}
                  >
                    ⏳ STOP DISPATCHED (AWAITING ACK)
                  </span>
                ) : null}
              </div>
              <div style={{ fontSize: '0.85rem', color: '#fecaca', marginTop: '2px' }}>
                {isHardwareConfirmed
                  ? `Arduino confirmed 'COMMAND RECEIVED: STOP' (Motor STOPPED). Inspect joint ${jointId} before resuming conveyor.`
                  : `STOP signal transmitted over serial to Arduino UNO. Awaiting hardware confirmation loop...`}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={handleResume}
              disabled={resuming}
              className="btn btn-primary"
              style={{
                background: '#22c55e',
                borderColor: '#16a34a',
                color: '#ffffff',
                fontWeight: 700,
                fontSize: '0.85rem',
                padding: '8px 18px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                cursor: resuming ? 'not-allowed' : 'pointer',
                boxShadow: '0 0 12px rgba(34, 197, 94, 0.4)',
              }}
            >
              <Play size={16} fill="#ffffff" />
              {resuming ? 'Resuming Motor...' : 'Resume Conveyor Motor'}
            </button>
          </div>
        </div>
      )}

      {/* 3. Main Decision & Telemetry Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '16px',
          marginBottom: '16px',
        }}
      >
        {/* Joint State & Final Decision Box */}
        <div
          style={{
            background: verdict.bg,
            border: `1px solid ${verdict.border}`,
            borderRadius: '10px',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: '12px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted, #94a3b8)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Current Target Joint
              </span>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginTop: '4px' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'var(--font-mono, monospace)', color: '#f8fafc' }}>
                  {jointId}
                </span>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: '6px',
                    background: 'var(--bg-elevated, #0f172a)',
                    border: '1px solid var(--border-subtle, #334155)',
                    color: inspectionState === 'INSPECTING' ? '#38bdf8' : (inspectionState === 'CONFIRMED' ? '#4ade80' : '#94a3b8'),
                  }}
                >
                  STATE: {inspectionState}
                </span>
              </div>
            </div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                borderRadius: '8px',
                background: 'var(--bg-elevated, #0f172a)',
                border: `1px solid ${verdict.border}`,
                color: verdict.color,
                fontWeight: 700,
                fontSize: '0.85rem',
              }}
            >
              <VerdictIcon size={18} />
              <span>{verdict.badgeText}</span>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', marginTop: '4px' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '10px', borderRadius: '8px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted, #94a3b8)', textTransform: 'uppercase' }}>Locked Verdict</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 700, color: verdict.color, fontFamily: 'var(--font-mono, monospace)', marginTop: '2px' }}>
                {label}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted, #94a3b8)', marginTop: '2px' }}>
                Confidence: <strong style={{ color: '#f8fafc' }}>{confidence}%</strong>
              </div>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '10px', borderRadius: '8px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted, #94a3b8)', textTransform: 'uppercase' }}>Vision Health Score</div>
              <div
                style={{
                  fontSize: '1.2rem',
                  fontWeight: 700,
                  color: label === 'DAMAGE' ? '#f87171' : (label === 'HEALTHY' ? '#4ade80' : '#fbbf24'),
                  fontFamily: 'var(--font-mono, monospace)',
                  marginTop: '2px',
                }}
              >
                {visionData.vision_score != null ? `${Math.round(visionData.vision_score)} / 100` : '—'}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted, #94a3b8)', marginTop: '2px' }}>
                Channel Weight: <strong>40%</strong>
              </div>
            </div>
          </div>
        </div>

        {/* Temporal Frame Aggregation & Sharpness Filter */}
        <div
          style={{
            background: 'var(--bg-elevated, #0f172a)',
            border: '1px solid var(--border-subtle, #334155)',
            borderRadius: '10px',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: '12px',
          }}
        >
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted, #94a3b8)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Temporal Frame Aggregation
              </span>
              <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono, monospace)', color: '#38bdf8', fontWeight: 600 }}>
                {validFrames} / 5 Valid Frames
              </span>
            </div>

            {/* Progress Bar for Valid Frames */}
            <div style={{ height: '8px', background: 'var(--bg-overlay, #1e293b)', borderRadius: '4px', overflow: 'hidden', marginBottom: '12px' }}>
              <div
                style={{
                  height: '100%',
                  width: `${Math.min(100, (validFrames / 5) * 100)}%`,
                  background: validFrames >= 5 ? '#22c55e' : '#38bdf8',
                  borderRadius: '4px',
                  transition: 'width 0.25s ease',
                }}
              />
            </div>

            {/* Class Probabilities Bar */}
            <div style={{ fontSize: '10px', color: 'var(--text-muted, #94a3b8)', marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
              <span>HEALTHY: {pHealthy}%</span>
              <span>DAMAGE: {pDamage}%</span>
            </div>
            <div style={{ height: '6px', display: 'flex', borderRadius: '3px', overflow: 'hidden', background: '#334155', marginBottom: '12px' }}>
              <div style={{ width: `${pHealthy}%`, background: '#22c55e', transition: 'width 0.2s' }} />
              <div style={{ width: `${pDamage}%`, background: '#ef4444', transition: 'width 0.2s' }} />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
            <div style={{ background: 'var(--card-bg, #1e293b)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-subtle, #334155)' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted, #94a3b8)' }}>Laplacian Sharpness</div>
              <div
                style={{
                  fontSize: '0.95rem',
                  fontWeight: 700,
                  fontFamily: 'var(--font-mono, monospace)',
                  color: parseFloat(sharpness) >= 100 ? '#4ade80' : '#f87171',
                  marginTop: '2px',
                }}
              >
                {sharpness} <span style={{ fontSize: '10px', color: 'var(--text-muted, #64748b)' }}>(≥100)</span>
              </div>
            </div>

            <div style={{ background: 'var(--card-bg, #1e293b)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-subtle, #334155)' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted, #94a3b8)' }}>Rejected Blurry</div>
              <div style={{ fontSize: '0.95rem', fontWeight: 700, fontFamily: 'var(--font-mono, monospace)', color: '#fbbf24', marginTop: '2px' }}>
                {rejectedFrames} frames
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Manual Conveyor Controls & Safety Status Footer */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          paddingTop: '12px',
          borderTop: '1px solid var(--border-subtle, #334155)',
          fontSize: '12px',
          color: 'var(--text-muted, #94a3b8)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ShieldCheck size={16} color={isMotorRunning ? '#4ade80' : '#f87171'} />
          <span>
            {isMotorRunning ? (
              <span style={{ color: '#4ade80' }}>
                Motor Interlock Armed — High-confidence damage triggers immediate hardware shutdown.
              </span>
            ) : (
              <span style={{ color: '#f87171' }}>
                Motor Interlock Engaged — Conveyor stopped. Operator must verify joint and clear fault.
              </span>
            )}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {actionMessage && (
            <span style={{ color: '#38bdf8', fontSize: '11px', fontWeight: 600 }}>{actionMessage}</span>
          )}

          {!isMotorRunning ? (
            <button
              onClick={handleResume}
              disabled={resuming}
              className="btn btn-sm btn-primary"
              style={{
                background: '#22c55e',
                borderColor: '#16a34a',
                color: '#ffffff',
                padding: '4px 14px',
                fontSize: '11px',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <Play size={12} fill="#ffffff" />
              {resuming ? 'Resuming...' : 'Resume Motor'}
            </button>
          ) : (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="btn btn-sm btn-secondary"
              style={{
                color: '#f87171',
                borderColor: 'rgba(239, 68, 68, 0.4)',
                padding: '4px 14px',
                fontSize: '11px',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <Square size={12} fill="#f87171" />
              {stopping ? 'Stopping...' : 'Emergency Stop'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
