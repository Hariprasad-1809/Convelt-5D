import React from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { AlertTriangle, RefreshCw, ShieldCheck, Waves } from 'lucide-react';

function formatTime(value) {
  return new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(value));
}

export default function TrendCharts({ jointId, historyData, onRefresh }) {
  const formattedData = historyData.map((item) => ({
    time: formatTime(item.timestamp),
    health_score: item.health_score ?? 0,
    vision_score: item.vision_score ?? 0,
    vibration_score: item.vibration_score ?? 0,
    temperature_score: item.temperature_score ?? 0,
    raw_vibration: item.vibration ?? 0,
    raw_temperature: item.temperature ?? 0,
  }));

  return (
    <section className="panel panel-strong overflow-hidden chart-panel scan-line">
      <div className="border-b border-accent/20 px-5 py-5 sm:px-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-start gap-3">
              <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-accent/30 bg-accent-soft text-accent glow-cyan">
                <Waves className="h-5 w-5" />
              </div>
              <div>
                <div className="section-kicker">Historical trends</div>
                <h3 className="mt-2 text-xl font-bold tracking-tight text-main font-display">Joint {jointId} trend analysis</h3>
                <p className="mt-2 max-w-2xl text-sm text-muted">
                  The score chart includes explicit threshold reference lines at 70 and 40 so crossings are clear during visual telemetry inspection.
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
            <span className="glass-chip inline-flex items-center gap-2 border-risk-low/30 text-risk-low">
              <ShieldCheck className="h-3.5 w-3.5" />
              LOW at 70+
            </span>
            <span className="glass-chip inline-flex items-center gap-2 border-risk-medium/30 text-risk-medium">
              <AlertTriangle className="h-3.5 w-3.5" />
              MEDIUM at 40-69
            </span>
            <span className="glass-chip inline-flex items-center gap-2 border-risk-high/30 text-risk-high">
              <AlertTriangle className="h-3.5 w-3.5" />
              HIGH below 40
            </span>
            <button
              type="button"
              onClick={onRefresh}
              className="button-accent inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-bold"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Refresh data
            </button>
          </div>
        </div>
      </div>

      {!formattedData.length ? (
        <div className="px-5 py-8 sm:px-6">
          <div className="rounded-3xl border border-surface bg-surface-raised px-6 py-10 text-center text-sm text-muted">
            No history yet. Start or step the simulation to populate the chart.
          </div>
        </div>
      ) : (
        <div className="grid gap-6 px-5 py-5 sm:px-6 xl:grid-cols-2">
          <div className="rounded-3xl border border-surface bg-surface/80 p-4">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="section-kicker">Health score trend</div>
                <div className="mt-1 text-sm text-muted font-body">Overall health and simulated vision score over time</div>
              </div>
              <span className="simulated-badge px-2 py-1">SIMULATED</span>
            </div>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={formattedData} margin={{ top: 10, right: 18, left: -8, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.08)" />
                  <XAxis dataKey="time" stroke="#607d9b" tickLine={false} axisLine={false} minTickGap={18} tick={{ fill: '#e0f0ff', fontSize: 11 }} label={{ value: 'Time', position: 'insideBottomRight', offset: -2, fill: '#607d9b', fontSize: 11 }} />
                  <YAxis domain={[0, 100]} stroke="#607d9b" tickLine={false} axisLine={false} tick={{ fill: '#e0f0ff', fontSize: 11 }} label={{ value: 'Score', angle: -90, position: 'insideLeft', fill: '#607d9b', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(1,7,18,0.95)', borderColor: 'rgba(0,212,255,0.3)', borderRadius: '16px', fontSize: '12px', boxShadow: '0 0 20px rgba(0,212,255,0.2)' }}
                    labelStyle={{ color: '#e0f0ff', fontFamily: 'Space Grotesk' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '11px', color: '#e0f0ff' }} />
                  <ReferenceLine y={70} stroke="#10b981" strokeDasharray="5 5" label={{ value: 'LOW 70', fill: '#10b981', fontSize: 10, position: 'insideTopRight' }} />
                  <ReferenceLine y={40} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: 'MEDIUM 40', fill: '#f59e0b', fontSize: 10, position: 'insideTopRight' }} />
                  <Line type="monotone" dataKey="health_score" name="Overall health score" stroke="#00d4ff" strokeWidth={3} dot={{ r: 3, fill: '#00d4ff' }} activeDot={{ r: 5, stroke: '#e0f0ff', strokeWidth: 1.5 }} />
                  <Line type="monotone" dataKey="vision_score" name="Vision score (simulated)" stroke="#0099bb" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-3xl border border-surface bg-surface/80 p-4">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="section-kicker">Raw physical sensors</div>
                <div className="mt-1 text-sm text-muted font-body">MPU6050 vibration and DS18B20 temperature response</div>
              </div>
              <span className="glass-chip">Hardware telemetry</span>
            </div>
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={formattedData} margin={{ top: 10, right: 18, left: -8, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.08)" />
                  <XAxis dataKey="time" stroke="#607d9b" tickLine={false} axisLine={false} minTickGap={18} tick={{ fill: '#e0f0ff', fontSize: 11 }} label={{ value: 'Time', position: 'insideBottomRight', offset: -2, fill: '#607d9b', fontSize: 11 }} />
                  <YAxis stroke="#607d9b" tickLine={false} axisLine={false} tick={{ fill: '#e0f0ff', fontSize: 11 }} label={{ value: 'Raw reading', angle: -90, position: 'insideLeft', fill: '#607d9b', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(1,7,18,0.95)', borderColor: 'rgba(0,212,255,0.3)', borderRadius: '16px', fontSize: '12px', boxShadow: '0 0 20px rgba(0,212,255,0.2)' }}
                    labelStyle={{ color: '#e0f0ff', fontFamily: 'Space Grotesk' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '11px', color: '#e0f0ff' }} />
                  <Line type="monotone" dataKey="raw_vibration" name="Vibration (m/s²)" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: '#fef3c7', strokeWidth: 1 }} />
                  <Line type="monotone" dataKey="raw_temperature" name="Temperature (°C)" stroke="#f43f5e" strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: '#ffe4e6', strokeWidth: 1 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
