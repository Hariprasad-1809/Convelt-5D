import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, Blocks, BrainCircuit, Camera, CircuitBoard, Cpu, Eye, Activity, ShieldCheck, Sparkles, Workflow, X, Thermometer } from 'lucide-react';

const pipelineStages = [
  {
    title: 'SEE',
    subtitle: 'USB Webcam capture',
    status: 'Future Phase',
    description: 'Frame capture and preprocessing will feed the vision branch in the next integration stage.',
    icon: <Camera className="h-5 w-5 text-accent" />,
    tone: 'border-accent/30 bg-accent-soft text-accent',
    implemented: false,
  },
  {
    title: 'TRACK',
    subtitle: 'YOLO + ByteTrack',
    status: 'Future Phase',
    description: 'Joint persistence and surface-damage detection remain planned for the next vision stack.',
    icon: <Eye className="h-5 w-5 text-accent" />,
    tone: 'border-accent/30 bg-accent-soft text-accent',
    implemented: false,
  },
  {
    title: 'SENSE',
    subtitle: 'ESP32 inputs',
    status: 'Implemented',
    description: 'The current build reads MPU6050 vibration, DS18B20 temperature, and A3144 Hall events.',
    icon: <Cpu className="h-5 w-5 text-risk-low" />,
    tone: 'border-risk-low/30 bg-risk-low-soft text-risk-low',
    implemented: true,
  },
  {
    title: 'FUSE',
    subtitle: 'Rule-based scoring',
    status: 'Implemented',
    description: 'Health scoring is a weighted fusion of the current sensor inputs and the simulated vision field.',
    icon: <Workflow className="h-5 w-5 text-risk-low" />,
    tone: 'border-risk-low/30 bg-risk-low-soft text-risk-low',
    implemented: true,
  },
  {
    title: 'PREDICT',
    subtitle: 'ML layer',
    status: 'Future Phase',
    description: 'Random Forest, XGBoost, and LSTM/GRU failure prediction are planned, not active yet.',
    icon: <BrainCircuit className="h-5 w-5 text-accent" />,
    tone: 'border-accent/30 bg-accent-soft text-accent',
    implemented: false,
  },
  {
    title: 'ACT',
    subtitle: 'Alerts and controls',
    status: 'Implemented',
    description: 'The current prototype exposes alerts and simulation control actions over the REST backend.',
    icon: <Sparkles className="h-5 w-5 text-risk-low" />,
    tone: 'border-risk-low/30 bg-risk-low-soft text-risk-low',
    implemented: true,
  },
];

const hardwareItems = [
  { name: 'ESP32', detail: 'Prototype edge controller', icon: <Cpu className="h-5 w-5 text-accent" /> },
  { name: 'USB Webcam', detail: 'Future vision source', icon: <Camera className="h-5 w-5 text-accent" /> },
  { name: 'MPU6050', detail: 'I2C vibration sensor', icon: <Activity className="h-5 w-5 text-risk-medium" /> },
  { name: 'DS18B20', detail: '1-Wire temperature sensor', icon: <Thermometer className="h-5 w-5 text-risk-high" /> },
  { name: 'A3144', detail: 'Hall pulse marker', icon: <ShieldCheck className="h-5 w-5 text-main" /> },
  { name: 'Magnet', detail: 'Joint marker target', icon: <Eye className="h-5 w-5 text-main" /> },
  { name: 'Breadboard', detail: 'Rapid prototype wiring', icon: <Blocks className="h-5 w-5 text-main" /> },
  { name: 'LEDs', detail: 'Status feedback outputs', icon: <Sparkles className="h-5 w-5 text-risk-low" /> },
  { name: 'Resistors', detail: 'Current limiting and pullups', icon: <CircuitBoard className="h-5 w-5 text-main" /> },
];

const futureItems = ['YOLO', 'ByteTrack', 'WebSocket', 'ML/DL', 'industrial EM/MFL', 'PLC', 'SCADA'];

export default function AboutModal({ isOpen, onClose }) {
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-app/85 px-4 py-6 backdrop-blur-xl sm:px-6">
          <div className="mx-auto max-w-6xl">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 16 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 16 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
              className="panel panel-strong overflow-hidden border border-accent/30 shadow-[0_40px_100px_rgba(0,0,0,0.85)] glow-cyan"
            >
              <div className="flex items-start justify-between gap-3 border-b border-accent/20 px-5 py-5 sm:px-6">
                <div>
                  <div className="section-kicker">About the project</div>
                  <h2 className="mt-2 text-2xl font-bold tracking-tight text-main font-display sm:text-3xl">JointGuard system blueprint</h2>
                  <p className="mt-2 max-w-3xl text-sm text-muted">
                    This page explains the current prototype surface and the planned extensions without changing the live REST-driven dashboard.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={onClose}
                  className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-surface bg-surface-raised text-muted transition hover:border-accent hover:text-accent"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="space-y-8 px-5 py-6 sm:px-6">
                <section className="rounded-3xl border border-risk-medium/30 bg-risk-medium-soft p-5 glow-med">
                  <div className="flex items-center gap-2 text-sm font-bold text-risk-medium font-display">
                    <ShieldCheck className="h-4 w-4" />
                    Current prototype notes
                  </div>
                  <div className="mt-4 grid gap-3 md:grid-cols-3">
                    <div className="rounded-2xl border border-surface bg-surface-raised p-4 text-sm text-main">
                      Health scoring is a weighted fusion of current telemetry inputs and simulated vision score.
                    </div>
                    <div className="rounded-2xl border border-surface bg-surface-raised p-4 text-sm text-main">
                      A3144 is used as a Hall pulse demo sensor in this phase, not an industrial magnetic-flux-leakage system.
                    </div>
                    <div className="rounded-2xl border border-surface bg-surface-raised p-4 text-sm text-main">
                      The dashboard is REST-polling only in the current build; WebSocket streaming remains future work.
                    </div>
                  </div>
                </section>

                <section>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="section-kicker">Pipeline</div>
                      <h3 className="mt-2 text-xl font-bold tracking-tight text-main font-display">SEE → TRACK → SENSE → FUSE → PREDICT → ACT</h3>
                    </div>
                    <span className="glass-chip">Implemented and future phases shown together</span>
                  </div>
                  <div className="mt-4 grid gap-3 xl:grid-cols-6">
                    {pipelineStages.map((stage, index) => (
                      <div key={stage.title} className={`panel interactive-surface flex min-h-[13rem] flex-col justify-between border p-4 ${stage.tone} ${stage.implemented ? 'glow-low border-risk-low/40' : 'border-surface bg-surface-raised'}`}>
                        <div>
                          <div className="flex items-center justify-between gap-3">
                            <div className="text-[11px] font-mono tracking-[0.24em] text-muted">0{index + 1}</div>
                            <span className={`rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.24em] font-display ${stage.status === 'Implemented' ? 'border-risk-low/30 bg-risk-low-soft text-risk-low' : 'border-accent/30 bg-accent-soft text-accent'}`}>
                              {stage.status}
                            </span>
                          </div>
                          <div className={`mt-4 inline-flex h-11 w-11 items-center justify-center rounded-2xl border ${stage.implemented ? 'border-risk-low/30 bg-risk-low-soft' : 'border-accent/30 bg-accent-soft'}`}>
                            {stage.icon}
                          </div>
                          <div className="mt-4 text-lg font-bold tracking-tight text-main font-display">{stage.title}</div>
                          <div className="mt-1 text-sm text-muted">{stage.subtitle}</div>
                        </div>
                        <p className="mt-4 text-sm leading-relaxed text-main font-body">{stage.description}</p>
                      </div>
                    ))}
                  </div>
                </section>

                <section>
                  <div className="flex items-center gap-3">
                    <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-accent/30 bg-accent-soft text-accent glow-cyan">
                      <Workflow className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="section-kicker">System architecture</div>
                      <h3 className="mt-1 text-xl font-bold tracking-tight text-main font-display">Current flow versus future flow</h3>
                    </div>
                  </div>

                  <div className="mt-4 grid gap-4 xl:grid-cols-2">
                    <div className="panel border border-risk-low/30 bg-risk-low-soft p-5 glow-low">
                      <div className="flex items-center justify-between gap-3">
                        <div className="text-sm font-bold text-risk-low font-display">Current phase flow</div>
                        <span className="glass-chip border-risk-low/30 text-risk-low">Live prototype</span>
                      </div>
                      <div className="mt-4 space-y-3 text-sm text-main">
                        <div className="rounded-2xl border border-surface bg-surface-raised p-3">ESP32 sensors and manual overrides</div>
                        <div className="flex justify-center text-risk-low"><ArrowRight className="h-4 w-4 rotate-90" /></div>
                        <div className="rounded-2xl border border-surface bg-surface-raised p-3">FastAPI REST endpoints</div>
                        <div className="flex justify-center text-risk-low"><ArrowRight className="h-4 w-4 rotate-90" /></div>
                        <div className="rounded-2xl border border-surface bg-surface-raised p-3">Rule-based health scoring and alert generation</div>
                        <div className="flex justify-center text-risk-low"><ArrowRight className="h-4 w-4 rotate-90" /></div>
                        <div className="rounded-2xl border border-surface bg-surface-raised p-3">SQLite persistence and historical storage</div>
                        <div className="flex justify-center text-risk-low"><ArrowRight className="h-4 w-4 rotate-90" /></div>
                        <div className="rounded-2xl border border-accent/30 bg-accent-soft p-3 font-bold text-accent font-display">React live console with REST polling</div>
                      </div>
                    </div>

                    <div className="panel border border-accent/20 bg-surface-raised p-5">
                      <div className="flex items-center justify-between gap-3">
                        <div className="text-sm font-bold text-accent font-display">Future full architecture</div>
                        <span className="glass-chip border-accent/30 text-accent">Not yet implemented</span>
                      </div>
                      <div className="mt-4 space-y-3 text-sm text-main">
                        <div className="rounded-2xl border border-surface bg-surface/80 p-3">USB Webcam and industrial camera feed</div>
                        <div className="flex justify-center text-accent"><ArrowRight className="h-4 w-4 rotate-90" /></div>
                        <div className="rounded-2xl border border-surface bg-surface/80 p-3">YOLO damage detection and ByteTrack persistence</div>
                        <div className="flex justify-center text-accent"><ArrowRight className="h-4 w-4 rotate-90" /></div>
                        <div className="rounded-2xl border border-surface bg-surface/80 p-3">WebSocket live telemetry gateway</div>
                        <div className="flex justify-center text-accent"><ArrowRight className="h-4 w-4 rotate-90" /></div>
                        <div className="rounded-2xl border border-surface bg-surface/80 p-3">Predictive ML models and industrial sensing extensions</div>
                        <div className="flex justify-center text-accent"><ArrowRight className="h-4 w-4 rotate-90" /></div>
                        <div className="rounded-2xl border border-accent/30 bg-accent-soft p-3 font-bold text-accent font-display">PLC / SCADA actuation and protocol gateways</div>
                      </div>
                    </div>
                  </div>
                </section>

                <section>
                  <div className="flex items-center gap-3">
                    <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-accent/30 bg-accent-soft text-accent">
                      <Cpu className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="section-kicker">Prototype hardware</div>
                      <h3 className="mt-1 text-xl font-bold tracking-tight text-main font-display">Component manifest</h3>
                    </div>
                  </div>

                  <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {hardwareItems.map((item) => (
                      <div key={item.name} className="panel interactive-surface border border-surface bg-surface-raised p-4">
                        <div className="flex items-start gap-3">
                          <div className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-surface bg-surface">
                            {item.icon}
                          </div>
                          <div>
                            <div className="text-base font-bold text-main font-display">{item.name}</div>
                            <div className="mt-1 text-sm text-muted">{item.detail}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="panel border border-accent/20 bg-accent-soft p-5">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="section-kicker">Future integrations</div>
                      <h3 className="mt-1 text-xl font-bold tracking-tight text-main font-display">Not yet implemented</h3>
                    </div>
                    <span className="glass-chip border-accent/30 text-accent font-display">Planned stack</span>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {futureItems.map((item) => (
                      <span key={item} className="rounded-full border border-accent/30 bg-surface-raised px-3 py-2 text-xs font-bold text-main font-display">
                        {item}
                      </span>
                    ))}
                  </div>
                </section>
              </div>
            </motion.div>
          </div>
        </div>
      )}
    </AnimatePresence>
  );
}
