import { Cpu, Thermometer, Activity, Server, Database, Brain, Layers, Magnet, Eye, AlertTriangle } from 'lucide-react';

const TECH_STACK = [
  { label: 'ESP32-WROOM-32D',  role: 'Microcontroller',         icon: Cpu,        color: 'var(--accent)' },
  { label: 'DS18B20',           role: 'Temperature (10% Weight)',icon: Thermometer, color: 'var(--status-high)' },
  { label: 'MPU6050',           role: 'IMU / Accel (20% Weight)',icon: Activity,   color: 'var(--status-medium)' },
  { label: 'A3144 Hall Effect', role: 'Magnetic Pulse (30% Weight)', icon: Magnet, color: 'var(--accent)' },
  { label: 'Vision Scoring',    role: 'Visual Analysis (40% Weight)', icon: Eye,    color: '#38bdf8' },
  { label: 'FastAPI REST API',  role: 'Backend Engine & Simulation', icon: Server,  color: '#a78bfa' },
  { label: 'SQLite / SQLAlchemy', role: 'Telemetry & Alert Storage', icon: Database, color: '#60a5fa' },
  { label: 'React + Vite',      role: 'SCADA Digital Twin Dashboard', icon: Layers,  color: '#34d399' },
  { label: 'YOLOv8 + ByteTrack',role: 'Vision ML Model (Phase 2)', icon: Brain,     color: '#f472b6' },
];

function Section({ title, children }) {
  return (
    <section style={{ marginBottom: '28px' }}>
      <h2 style={{ fontSize: 'var(--text-base)', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '12px', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
        {title}
      </h2>
      <div style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
        {children}
      </div>
    </section>
  );
}

export default function AboutProject() {
  return (
    <div className="page-content" style={{ maxWidth: 900 }}>
      {/* Hero */}
      <div className="card" style={{ marginBottom: '24px', padding: '28px', borderTop: '3px solid var(--accent)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '14px' }}>
          <div style={{ width: 44, height: 44, background: 'linear-gradient(135deg, var(--accent), #0e7490)', borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Cpu size={22} color="#0a0d12" strokeWidth={2.5} />
          </div>
          <div>
            <h1 style={{ fontSize: 'var(--text-2xl)', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>JointGuard</h1>
            <div style={{ fontSize: 'var(--text-sm)', color: 'var(--accent)' }}>IoT & Vision-Assisted Conveyor Belt Joint Health Monitoring System</div>
          </div>
        </div>
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', lineHeight: 1.8, maxWidth: 700 }}>
          JointGuard is a digital-twin industrial monitoring system designed to inspect, detect, and track the condition of conveyor belt joints in real time using multi-modal IoT sensor fusion, an ESP32 microcontroller, a FastAPI simulation backend, and an industrial SCADA dashboard.
        </p>
      </div>

      {/* Phase 1 Disclaimers */}
      <div className="card" style={{ marginBottom: '24px', borderLeft: '3px solid var(--status-medium)', background: 'rgba(245, 158, 11, 0.04)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
          <AlertTriangle size={16} color="var(--status-medium)" />
          <h3 style={{ fontSize: 'var(--text-sm)', fontWeight: 700, color: 'var(--status-medium)' }}>
            Phase 1 Prototype Architectural Disclaimers
          </h3>
        </div>
        <ul style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)', lineHeight: 2, paddingLeft: '20px' }}>
          <li>
            <strong>Health Scoring:</strong> Rule-based weighted fusion formula (0.40 Vision + 0.30 Magnetic + 0.20 Vibration + 0.10 Temperature) — explicitly deterministic, not machine learning in Phase 1.
          </li>
          <li>
            <strong>Hall Sensor:</strong> A3144 Hall effect sensor is a magnetic-event demonstration sensor for detecting joint pulse markers — not an industrial electromagnetic (EM/MFL) steel cord scanner.
          </li>
          <li>
            <strong>Vision Analysis:</strong> Vision score is simulated or manually injected via the REST API — full camera pipeline with YOLOv8/ByteTrack is slated for Phase 2.
          </li>
          <li>
            <strong>Telemetry Streaming:</strong> Periodic REST API polling (2.0s interval) — WebSocket streaming is planned for Phase 2.
          </li>
        </ul>
      </div>

      {/* Problem */}
      <Section title="The Problem">
        <p>
          Conveyor belt joints are among the most failure-prone components in mining, thermal power, and industrial logistics operations. They are subject to continuous mechanical tension, thermal friction, and cyclic acceleration shock. Uncaught failures cause:
        </p>
        <ul style={{ marginTop: '10px', paddingLeft: '20px', lineHeight: 2.2 }}>
          <li>Unplanned production halts and massive downtime expenses</li>
          <li>Severe material spillage, belt snapping, and catastrophic safety hazards</li>
          <li>Costly emergency splicing and equipment damage</li>
          <li>Inability to detect hidden splice degradation through visual human walk-bys alone</li>
        </ul>
      </Section>

      {/* Proposed Solution */}
      <Section title="Proposed Solution">
        <p>
          JointGuard instrumentally tracks every joint (J01–J03) continuously as it cycles through the conveyor circuit. Real-time telemetry is fused across four orthogonal channels into a weighted health score with three actionable risk tiers:
        </p>
        <div style={{ display: 'flex', gap: '12px', marginTop: '12px', flexWrap: 'wrap' }}>
          <div style={{ padding: '8px 14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', border: '1px solid var(--status-normal-border)' }}>
            <span style={{ color: 'var(--status-normal)', fontWeight: 700 }}>NORMAL / LOW:</span> Belt splice healthy (Score ≥ 75)
          </div>
          <div style={{ padding: '8px 14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', border: '1px solid var(--status-medium-border)' }}>
            <span style={{ color: 'var(--status-medium)', fontWeight: 700 }}>MEDIUM RISK:</span> Moderate thermal/vibrational warning (Score 50–74)
          </div>
          <div style={{ padding: '8px 14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', border: '1px solid var(--status-high-border)' }}>
            <span style={{ color: 'var(--status-high)', fontWeight: 700 }}>HIGH RISK / CRITICAL:</span> Urgent splice inspection required (Score &lt; 50)
          </div>
        </div>
      </Section>

      {/* How It Works */}
      <Section title="How It Works">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginTop: '4px' }}>
          {[
            { step: '1', title: 'Sensors & Hall Marker', desc: 'DS18B20 monitors splice heat. MPU6050 tracks vibration. A3144 registers magnetic pulse when a joint passes.' },
            { step: '2', title: 'ESP32 IoT Node', desc: 'Microcontroller samples physical sensors and transmits JSON telemetry packets over Wi-Fi.' },
            { step: '3', title: 'FastAPI Backend', desc: 'REST server with SQLite persistence, automated 5-joint simulation, manual fault injection, and scoring engine.' },
            { step: '4', title: 'SCADA Web Console', desc: 'This React dashboard polls telemetry every 2.0s, charts trends, and enables operator alert acknowledgment.' },
          ].map(s => (
            <div key={s.step} style={{ padding: '14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--accent)', marginBottom: 6, letterSpacing: '0.1em' }}>STEP {s.step}</div>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>{s.title}</div>
              <div style={{ fontSize: 'var(--text-xs)', lineHeight: 1.7 }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* Tech Stack */}
      <Section title="Technology Stack">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px', marginTop: '4px' }}>
          {TECH_STACK.map(t => {
            const Icon = t.icon;
            return (
              <div key={t.label} style={{ padding: '12px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius)', border: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                <Icon size={16} color={t.color} style={{ flexShrink: 0, marginTop: 2 }} />
                <div>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: 'var(--text-xs)' }}>{t.label}</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5 }}>{t.role}</div>
                </div>
              </div>
            );
          })}
        </div>
      </Section>
    </div>
  );
}
