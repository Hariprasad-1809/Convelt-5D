import { useState, useMemo } from 'react';
import { STATUS, STATUS_COLORS } from '../../data/constants';

/**
 * ConveyorVisualization — industrial SVG belt diagram supporting dynamic joints (J01-J05).
 *
 * @param {Object}   joints       - Object keyed by joint ID with live sensor data
 * @param {Function} onJointClick - callback(jointId)
 */
export default function ConveyorVisualization({ joints = {}, onJointClick }) {
  const [hoveredId, setHoveredId] = useState(null);

  const availableJointIds = useMemo(() => {
    const keys = Object.keys(joints);
    return keys.length > 0 ? keys : ['J01', 'J02', 'J03', 'J04', 'J05'];
  }, [joints]);

  const jointPositions = useMemo(() => {
    const total = availableJointIds.length;
    const startX = 115;
    const endX = 485;
    const step = total > 1 ? (endX - startX) / (total - 1) : 0;

    return availableJointIds.map((id, index) => ({
      id,
      cx: Math.round(startX + index * step),
      cy: 96,
    }));
  }, [availableJointIds]);

  function getStatusColor(status) {
    return STATUS_COLORS[status] ?? STATUS_COLORS[STATUS.NORMAL];
  }

  const isClickable = !!onJointClick;

  return (
    <div
      style={{ width: '100%', position: 'relative' }}
      role="img"
      aria-label="Conveyor belt digital-twin diagram with interactive joints. Click any joint to inspect."
    >
      <svg
        viewBox="0 0 600 220"
        width="100%"
        style={{ display: 'block' }}
        aria-hidden="true"
      >
        {/* ── Belt structure ───────────────────────────────── */}

        {/* Idler supports (vertical stanchions under each joint node) */}
        {jointPositions.map(({ id, cx }) => (
          <line
            key={`stanchion-${id}`}
            x1={cx}
            y1="106"
            x2={cx}
            y2="148"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="3"
            strokeLinecap="round"
          />
        ))}

        {/* Base rail */}
        <rect
          x="45"
          y="148"
          width="510"
          height="4"
          rx="2"
          fill="rgba(255,255,255,0.06)"
        />

        {/* Support feet */}
        {[70, 175, 300, 425, 530].map(x => (
          <line
            key={x}
            x1={x}
            y1="148"
            x2={x}
            y2="155"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="2"
          />
        ))}

        {/* Upper belt surface */}
        <rect
          x="55"
          y="88"
          width="490"
          height="16"
          rx="2"
          fill="rgba(255,255,255,0.03)"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth="1"
        />

        {/* Belt cross-cleats */}
        {Array.from({ length: 24 }, (_, i) => (
          <line
            key={i}
            x1={65 + i * 20}
            y1="88"
            x2={65 + i * 20}
            y2="104"
            stroke="rgba(255,255,255,0.04)"
            strokeWidth="1"
          />
        ))}

        {/* Direction chevrons */}
        {[160, 260, 360, 450].map(x => (
          <g key={x}>
            <polyline
              points={`${x - 4},92 ${x},96 ${x - 4},100`}
              fill="none"
              stroke="rgba(34,211,238,0.2)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <polyline
              points={`${x},92 ${x + 4},96 ${x},100`}
              fill="none"
              stroke="rgba(34,211,238,0.12)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </g>
        ))}

        {/* ── End pulleys ──────────────────────────────────── */}
        {/* Drive pulley (left — feed) */}
        <ellipse
          cx="55"
          cy="96"
          rx="20"
          ry="30"
          fill="rgba(255,255,255,0.04)"
          stroke="rgba(255,255,255,0.10)"
          strokeWidth="1.5"
        />
        <ellipse
          cx="55"
          cy="96"
          rx="12"
          ry="22"
          fill="rgba(255,255,255,0.02)"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="1"
        />

        {/* Tail pulley (right — discharge) */}
        <ellipse
          cx="545"
          cy="96"
          rx="20"
          ry="30"
          fill="rgba(255,255,255,0.04)"
          stroke="rgba(255,255,255,0.10)"
          strokeWidth="1.5"
        />
        <ellipse
          cx="545"
          cy="96"
          rx="12"
          ry="22"
          fill="rgba(255,255,255,0.02)"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="1"
        />

        {/* Lower belt return */}
        <path
          d="M55 116 Q300 136 545 116"
          fill="none"
          stroke="rgba(255,255,255,0.03)"
          strokeWidth="12"
          strokeLinecap="round"
        />

        {/* End labels */}
        <text
          x="55"
          y="172"
          textAnchor="middle"
          fill="rgba(255,255,255,0.25)"
          fontSize="8"
          fontFamily="Inter, sans-serif"
          letterSpacing="0.08em"
        >
          FEED END
        </text>
        <text
          x="545"
          y="172"
          textAnchor="middle"
          fill="rgba(255,255,255,0.25)"
          fontSize="8"
          fontFamily="Inter, sans-serif"
          letterSpacing="0.08em"
        >
          DISCHARGE
        </text>

        {/* ── Joint Nodes ──────────────────────────────────── */}
        {jointPositions.map(({ id, cx, cy }) => {
          const joint = joints?.[id];
          const status = joint?.status ?? STATUS.NORMAL;
          const zoneState = joint?.zone_state || 'PASSED';
          const isInspecting = zoneState === 'INSPECTING';
          const color = getStatusColor(status);
          const isHovered = hoveredId === id;
          const outerR = isHovered ? 17 : 15;
          const innerR = isHovered ? 11 : 9;

          return (
            <g
              key={id}
              onClick={() => isClickable && onJointClick(id)}
              onMouseEnter={() => setHoveredId(id)}
              onMouseLeave={() => setHoveredId(null)}
              style={{ cursor: isClickable ? 'pointer' : 'default' }}
              role={isClickable ? 'button' : undefined}
              tabIndex={isClickable ? 0 : undefined}
              aria-label={`${id}: ${status}${joint ? ` — ${joint.temperature?.toFixed(1)}°C` : ''} — click to inspect`}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') onJointClick?.(id);
              }}
            >
              {/* Inspection Zone indicator ring */}
              {isInspecting && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={outerR + 9}
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth="1.5"
                  strokeDasharray="3 3"
                  opacity={0.8}
                />
              )}

              {/* Outer glow ring — expands on hover */}
              <circle
                cx={cx}
                cy={cy}
                r={outerR + 6}
                fill="none"
                stroke={color}
                strokeWidth="1"
                opacity={isHovered ? 0.35 : isInspecting ? 0.25 : 0.1}
                style={{ transition: 'all 0.15s ease' }}
              />

              {/* Main ring */}
              <circle
                cx={cx}
                cy={cy}
                r={outerR}
                fill="rgba(10,13,18,0.95)"
                stroke={color}
                strokeWidth={isHovered ? 2.5 : 2}
                style={{ transition: 'all 0.15s ease' }}
              />

              {/* Inner fill */}
              <circle
                cx={cx}
                cy={cy}
                r={innerR}
                fill={color}
                opacity={isHovered ? 0.3 : 0.15}
                style={{ transition: 'all 0.15s ease' }}
              />

              {/* Center dot */}
              <circle
                cx={cx}
                cy={cy}
                r={isHovered ? 4.5 : 3.5}
                fill={color}
                style={{ transition: 'r 0.15s ease' }}
              />

              {/* Temperature reading — above node */}
              {joint && (
                <text
                  x={cx}
                  y={cy - 25}
                  textAnchor="middle"
                  fill="rgba(255,255,255,0.6)"
                  fontSize="9"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {joint.temperature?.toFixed(1)}°C
                </text>
              )}

              {/* Acceleration reading — above temp */}
              {joint && (
                <text
                  x={cx}
                  y={cy - 35}
                  textAnchor="middle"
                  fill="rgba(255,255,255,0.35)"
                  fontSize="7.5"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {joint.vibration?.toFixed(2)} m/s²
                </text>
              )}

              {/* Joint ID — below node */}
              <text
                x={cx}
                y={cy + 28}
                textAnchor="middle"
                fill={color}
                fontSize="10"
                fontWeight="700"
                fontFamily="Inter, sans-serif"
                letterSpacing="0.06em"
              >
                {id}
              </text>

              {/* Status / Zone label */}
              <text
                x={cx}
                y={cy + 38}
                textAnchor="middle"
                fill={color}
                fontSize="7"
                opacity={isHovered ? 1 : 0.7}
                fontFamily="Inter, sans-serif"
                letterSpacing="0.05em"
                fontWeight="600"
                style={{ transition: 'opacity 0.15s ease' }}
              >
                {status}
              </text>

              {/* Zone state chip */}
              {joint?.zone_state && (
                <text
                  x={cx}
                  y={cy + 48}
                  textAnchor="middle"
                  fill={isInspecting ? 'var(--accent)' : 'rgba(255,255,255,0.3)'}
                  fontSize="6.5"
                  fontFamily="Inter, sans-serif"
                  fontWeight={isInspecting ? '700' : '500'}
                  letterSpacing="0.04em"
                >
                  {joint.zone_state}
                </text>
              )}

              {/* Inspect hint — only on hover */}
              {isHovered && isClickable && (
                <text
                  x={cx}
                  y={cy + 58}
                  textAnchor="middle"
                  fill="rgba(34,211,238,0.85)"
                  fontSize="6.5"
                  fontFamily="Inter, sans-serif"
                  letterSpacing="0.04em"
                >
                  click to inspect
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
