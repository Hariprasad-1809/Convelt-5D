import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';

/**
 * Custom tooltip matching the dark theme.
 */
function CustomTooltip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--bg-overlay)',
      border: '1px solid var(--border-default)',
      borderRadius: 'var(--radius)',
      padding: '8px 12px',
      fontSize: 'var(--text-xs)',
      fontFamily: 'var(--font-mono)',
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ color: payload[0].color, fontWeight: 600 }}>
        {Number(payload[0].value).toFixed(2)} {unit}
      </div>
    </div>
  );
}

/**
 * SensorChart — recharts line chart wrapper for sensor time-series data.
 * @param {Array}  data        - Array of { time, value }
 * @param {string} color       - Line color (CSS value)
 * @param {string} unit        - Data unit for tooltip
 * @param {number} [height]    - Chart height in px
 * @param {object} [refLines]  - { normalMax, mediumMax } for threshold reference lines
 * @param {string} [dataKey]   - Data key to plot (default: 'value')
 */
export default function SensorChart({
  data = [],
  color = 'var(--accent)',
  unit = '',
  height = 180,
  refLines = null,
  dataKey = 'value',
}) {
  const allValues = data.map(d => d[dataKey]).filter(v => v != null);
  const min = allValues.length ? Math.min(...allValues) : 0;
  const max = allValues.length ? Math.max(...allValues) : 100;
  const padding = (max - min) * 0.15 || 5;
  const yMin = Math.max(0, min - padding);
  const yMax = max + padding;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(255,255,255,0.04)"
          vertical={false}
        />
        <XAxis
          dataKey="time"
          tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={[yMin, yMax]}
          tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
          tickLine={false}
          axisLine={false}
          tickFormatter={v => Number(v).toFixed(1)}
          width={48}
        />
        <Tooltip content={<CustomTooltip unit={unit} />} />

        {/* Reference lines for thresholds */}
        {refLines?.normalMax != null && (
          <ReferenceLine
            y={refLines.normalMax}
            stroke="var(--status-medium)"
            strokeDasharray="4 4"
            strokeWidth={1}
            label={{ value: 'MED', fill: 'var(--status-medium)', fontSize: 9, position: 'insideTopRight' }}
          />
        )}
        {refLines?.mediumMax != null && (
          <ReferenceLine
            y={refLines.mediumMax}
            stroke="var(--status-high)"
            strokeDasharray="4 4"
            strokeWidth={1}
            label={{ value: 'HIGH', fill: 'var(--status-high)', fontSize: 9, position: 'insideTopRight' }}
          />
        )}

        <Line
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 4, stroke: color, strokeWidth: 2, fill: 'var(--bg-base)' }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
