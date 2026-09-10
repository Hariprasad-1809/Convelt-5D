import React, { useState } from 'react';
import { Camera, CameraOff, AlertTriangle, CheckCircle2, ShieldAlert, HelpCircle } from 'lucide-react';

/**
 * CameraVision — Live Camera Video Stream & Vision Classification Card
 * Renders MJPEG stream from FastAPI backend (/api/v1/vision/stream)
 * and displays live YOLO classification badges.
 */
export default function CameraVision({ visionData, cameraStatus, activeJoint = 'J01' }) {
  const [imgError, setImgError] = useState(false);

  const isConnected = cameraStatus === 'CONNECTED' && !imgError;
  const label = visionData?.label || 'NO_JOINT';
  const confidence = visionData?.confidence ? (visionData.confidence * 100).toFixed(1) : null;
  const visionScore = visionData?.vision_score != null ? Math.round(visionData.vision_score) : null;

  // Status Badge Styling
  const getBadgeStyle = () => {
    switch (label) {
      case 'HEALTHY':
        return { bg: 'rgba(34, 197, 94, 0.15)', color: '#4ade80', border: 'rgba(34, 197, 94, 0.3)', icon: CheckCircle2 };
      case 'DAMAGE':
        return { bg: 'rgba(239, 68, 68, 0.15)', color: '#f87171', border: 'rgba(239, 68, 68, 0.3)', icon: ShieldAlert };
      case 'UNCERTAIN':
        return { bg: 'rgba(245, 158, 11, 0.15)', color: '#fbbf24', border: 'rgba(245, 158, 11, 0.3)', icon: AlertTriangle };
      default:
        return { bg: 'rgba(148, 163, 184, 0.15)', color: '#94a3b8', border: 'rgba(148, 163, 184, 0.3)', icon: HelpCircle };
    }
  };

  const badge = getBadgeStyle();
  const BadgeIcon = badge.icon;

  const streamUrl = "http://127.0.0.1:8000/api/v1/vision/stream";

  return (
    <div style={{
      background: 'var(--card-bg, #1e293b)',
      borderRadius: '12px',
      border: '1px solid var(--card-border, #334155)',
      padding: '20px',
      color: '#f8fafc',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.2)',
    }}>
      {/* Card Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            padding: '8px',
            borderRadius: '8px',
            background: 'rgba(99, 102, 241, 0.15)',
            color: '#818cf8',
            display: 'flex'
          }}>
            <Camera size={20} />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>YOLOv8 Joint Vision Inspector</h3>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Real-time OpenCV Localization & Deep Learning</span>
          </div>
        </div>

        {/* Connection Indicator */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '4px 12px',
          borderRadius: '20px',
          fontSize: '0.75rem',
          fontWeight: 600,
          background: isConnected ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
          color: isConnected ? '#4ade80' : '#f87171',
          border: `1px solid ${isConnected ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
        }}>
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: isConnected ? '#4ade80' : '#f87171',
            boxShadow: isConnected ? '0 0 8px #4ade80' : 'none'
          }} />
          {isConnected ? 'CAMERA CONNECTED' : 'CAMERA OFFLINE'}
        </div>
      </div>

      {/* Main Grid Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 1fr) 240px', gap: '20px', alignItems: 'center' }}>
        
        {/* Stream / Preview Window */}
        <div style={{
          position: 'relative',
          width: '100%',
          height: '240px',
          borderRadius: '8px',
          overflow: 'hidden',
          background: '#0f172a',
          border: '1px solid #334155',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          {isConnected ? (
            <img
              src={streamUrl}
              alt="Live JointGuard Camera Stream"
              onError={() => setImgError(true)}
              onLoad={() => setImgError(false)}
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            />
          ) : (
            <div style={{ textAlign: 'center', color: '#64748b', padding: '20px' }}>
              <CameraOff size={44} style={{ marginBottom: '8px', opacity: 0.5 }} />
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#94a3b8' }}>Live Camera Video Stream Offline</div>
              <div style={{ fontSize: '0.75rem', marginTop: '4px' }}>Ensure USB camera is connected (Index 1/0)</div>
            </div>
          )}

          {/* Active Joint Overlay Badge */}
          <div style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            background: 'rgba(15, 23, 42, 0.85)',
            backdropFilter: 'blur(4px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            padding: '4px 10px',
            borderRadius: '6px',
            fontSize: '0.8rem',
            fontWeight: 700,
            color: '#38bdf8'
          }}>
            ACTIVE JOINT: {activeJoint}
          </div>
        </div>

        {/* Live Metrics Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          
          {/* Classification Badge */}
          <div style={{
            padding: '14px',
            borderRadius: '8px',
            background: badge.bg,
            border: `1px solid ${badge.border}`,
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
          }}>
            <BadgeIcon size={28} style={{ color: badge.color }} />
            <div>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Classification
              </div>
              <div style={{ fontSize: '1.25rem', fontWeight: 700, color: badge.color }}>
                {label}
              </div>
            </div>
          </div>

          {/* Confidence Metric */}
          <div style={{
            padding: '10px 12px',
            borderRadius: '8px',
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid #334155',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Class Probabilities</span>
            <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>
              <span style={{ color: '#4ade80' }}>H: {visionData?.p_healthy != null ? (visionData.p_healthy * 100).toFixed(1) + '%' : '—'}</span> | {' '}
              <span style={{ color: '#f87171' }}>D: {visionData?.p_damage != null ? (visionData.p_damage * 100).toFixed(1) + '%' : '—'}</span>
            </span>
          </div>


          {/* Valid & Blurry Frames Metric */}
          <div style={{
            padding: '10px 12px',
            borderRadius: '8px',
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid #334155',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Sharp Frames (Blur Filter)</span>
            <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#fbbf24' }}>
              {visionData?.valid_frame_count ?? 0}/5 <span style={{ fontSize: '0.75rem', color: '#f87171' }}>({visionData?.rejected_blurry_count ?? 0} rej)</span>
            </span>
          </div>

          {/* Vision Score Metric */}
          <div style={{
            padding: '10px 12px',
            borderRadius: '8px',
            background: 'rgba(30, 41, 59, 0.6)',
            border: '1px solid #334155',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Vision Channel Score</span>
            <span style={{
              fontSize: '1rem',
              fontWeight: 700,
              color: visionScore != null ? (visionScore >= 70 ? '#4ade80' : visionScore >= 40 ? '#fbbf24' : '#f87171') : '#94a3b8'
            }}>
              {visionScore != null ? `${visionScore} / 100` : 'WAITING'}
            </span>
          </div>


        </div>
      </div>
    </div>
  );
}
