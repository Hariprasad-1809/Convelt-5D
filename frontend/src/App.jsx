import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, DatabaseZap, Info, RefreshCw, Shield, ShieldAlert, Zap } from 'lucide-react';
import ConveyorTwin from './components/ConveyorTwin';
import JointDetailModal from './components/JointDetailModal';
import SensorCards from './components/SensorCards';
import AlertsPanel from './components/AlertsPanel';
import TrendCharts from './components/TrendCharts';
import SimulationControls from './components/SimulationControls';
import AboutModal from './components/AboutModal';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const emptySimStatus = { is_running: false, current_cycle: 0, interval_seconds: 0, active_joints: [] };

const riskLabels = {
  LOW: 'LOW',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
  UNKNOWN: 'UNKNOWN',
};

function riskTone(value) {
  switch (value) {
    case 'LOW':
      return 'text-risk-low border-risk-low bg-risk-low-soft glow-low';
    case 'MEDIUM':
      return 'text-risk-medium border-risk-medium bg-risk-medium-soft glow-med';
    case 'HIGH':
      return 'text-risk-high border-risk-high bg-risk-high-soft glow-high';
    default:
      return 'text-risk-unknown border-risk-unknown bg-risk-unknown-soft';
  }
}

function formatLastSync(date) {
  if (!date) return 'Waiting for telemetry';
  return new Intl.DateTimeFormat([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date);
}

function countRisks(joints) {
  return joints.reduce(
    (accumulator, joint) => {
      const tone = joint.risk_level || 'UNKNOWN';
      accumulator[tone] = (accumulator[tone] || 0) + 1;
      return accumulator;
    },
    { LOW: 0, MEDIUM: 0, HIGH: 0, UNKNOWN: 0 },
  );
}

function statCard({ label, value, tone, icon }) {
  return (
    <div className={`panel panel-strong border p-4 ${tone}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="section-kicker">{label}</div>
          <div className="mt-2 text-2xl font-bold tracking-tight text-main font-display">{value}</div>
        </div>
        <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-accent/20 bg-accent-soft text-accent">
          {icon}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [joints, setJoints] = useState([]);
  const [selectedJointId, setSelectedJointId] = useState('J01');
  const [selectedJointDetail, setSelectedJointDetail] = useState(null);
  const [activeAlerts, setActiveAlerts] = useState([]);
  const [alertHistory, setAlertHistory] = useState([]);
  const [trendHistory, setTrendHistory] = useState([]);
  const [simStatus, setSimStatus] = useState(emptySimStatus);
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  const [apiConnected, setApiConnected] = useState(true);
  const [isLoadingInitial, setIsLoadingInitial] = useState(true);
  const [lastSyncedAt, setLastSyncedAt] = useState(null);

  const fetchAllData = useCallback(async () => {
    try {
      const jointsRes = await fetch(`${API_BASE}/joints`);
      if (jointsRes.ok) {
        const jointsData = await jointsRes.json();
        const enriched = await Promise.all(
          jointsData.map(async (joint) => {
            try {
              const sensorsRes = await fetch(`${API_BASE}/joints/${joint.joint_id}/sensors`);
              const sensors = sensorsRes.ok ? await sensorsRes.json() : null;
              return { ...joint, sensors };
            } catch {
              return { ...joint, sensors: null };
            }
          }),
        );

        setJoints(enriched);
        setApiConnected(true);
      } else {
        setApiConnected(false);
      }

      if (selectedJointId) {
        const detailRes = await fetch(`${API_BASE}/joints/${selectedJointId}`);
        if (detailRes.ok) {
          setSelectedJointDetail(await detailRes.json());
        }

        const historyRes = await fetch(`${API_BASE}/history/${selectedJointId}?limit=30`);
        if (historyRes.ok) {
          setTrendHistory(await historyRes.json());
        }
      }

      const alertsRes = await fetch(`${API_BASE}/alerts`);
      if (alertsRes.ok) {
        setActiveAlerts(await alertsRes.json());
      }

      const alertHistoryRes = await fetch(`${API_BASE}/alerts/history`);
      if (alertHistoryRes.ok) {
        setAlertHistory(await alertHistoryRes.json());
      }

      const simRes = await fetch(`${API_BASE}/simulation/status`);
      if (simRes.ok) {
        setSimStatus(await simRes.json());
      }

      setLastSyncedAt(new Date());
      setApiConnected(true);
    } catch (error) {
      console.error('REST API Connection Error:', error);
      setApiConnected(false);
    } finally {
      setIsLoadingInitial(false);
    }
  }, [selectedJointId]);

  useEffect(() => {
    fetchAllData();
    const intervalId = setInterval(fetchAllData, 2000);
    return () => clearInterval(intervalId);
  }, [fetchAllData]);

  const runSimulationAction = async (request) => {
    await request();
    await fetchAllData();
  };

  const selectedJointSummary = useMemo(() => joints.find((joint) => joint.joint_id === selectedJointId), [joints, selectedJointId]);
  const riskCounts = countRisks(joints);
  const activeCount = activeAlerts.length;
  const highestRiskCount = riskCounts.HIGH + riskCounts.MEDIUM;
  const selectedRisk = selectedJointSummary?.risk_level || 'UNKNOWN';

  return (
    <div className="app-shell topo-bg min-h-screen bg-app pb-12 text-main">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-24 top-0 h-72 w-72 rounded-full bg-accent-soft blur-3xl" />
        <div className="absolute right-0 top-16 h-80 w-80 rounded-full bg-accent-soft blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-risk-low-soft blur-3xl" />
      </div>

      <div className="sticky top-0 z-40 border-b border-accent/20 bg-surface/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-2 text-[11px] sm:px-6">
          <div className="flex flex-wrap items-center gap-3 text-muted">
            <span className="simulated-badge px-3 py-1">
              PHASE 1 REST
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-risk-medium bg-risk-medium-soft px-3 py-1 font-mono font-semibold text-risk-medium">
              <AlertTriangle className="h-3.5 w-3.5" />
              A3144 demo only
            </span>
            <span className="hidden sm:inline text-muted">•</span>
            <span>Rule-based weighted fusion</span>
            <span className="hidden sm:inline text-muted">•</span>
            <span>Vision is simulated or manually injected</span>
          </div>
          <button
            type="button"
            onClick={() => setIsAboutOpen(true)}
            className="button-accent inline-flex items-center gap-2 rounded-full px-3 py-1 font-semibold"
          >
            <Info className="h-3.5 w-3.5" />
            About project
          </button>
        </div>
      </div>

      <header className="mx-auto max-w-7xl px-4 pt-6 sm:px-6">
        <div className="panel panel-strong overflow-hidden px-5 py-5 sm:px-6">
          <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr] xl:items-end">
            <div className="space-y-5">
              <div className="flex flex-wrap items-center gap-4">
                <span className="inline-flex h-14 w-14 items-center justify-center rounded-3xl border border-accent/30 bg-accent-soft text-accent glow-cyan">
                  <Zap className="h-7 w-7" />
                </span>
                <div>
                  <div className="flex flex-wrap items-center gap-3">
                    <h1 className="text-3xl font-bold tracking-tight text-main font-display sm:text-5xl">JointGuard</h1>
                    <span className="glass-chip border-accent/30 text-accent">Deep Ocean Console</span>
                  </div>
                  <p className="mt-2 max-w-3xl text-sm text-muted sm:text-[15px]">
                    Conveyor joint health telemetry, inspection-state tracking, simulated sensing, and alert operations in an industrial dark SCADA dashboard.
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
                <span className="glass-chip inline-flex items-center gap-2">
                  <Shield className="h-3.5 w-3.5 text-risk-low" />
                  LOW {riskCounts.LOW}
                </span>
                <span className="glass-chip inline-flex items-center gap-2">
                  <ShieldAlert className="h-3.5 w-3.5 text-risk-medium" />
                  MEDIUM {riskCounts.MEDIUM}
                </span>
                <span className="glass-chip inline-flex items-center gap-2">
                  <AlertTriangle className="h-3.5 w-3.5 text-risk-high" />
                  HIGH {riskCounts.HIGH}
                </span>
                <span className="glass-chip inline-flex items-center gap-2">
                  <DatabaseZap className="h-3.5 w-3.5 text-muted" />
                  Cycle {simStatus.current_cycle}
                </span>
                <span className={`glass-chip inline-flex items-center gap-2 ${apiConnected ? 'border-risk-low bg-risk-low-soft text-risk-low' : 'border-risk-high bg-risk-high-soft text-risk-high'}`}>
                  <Activity className="h-3.5 w-3.5" />
                  {apiConnected ? 'REST connected' : 'REST offline'}
                </span>
                <span className="glass-chip inline-flex items-center gap-2">
                  <RefreshCw className="h-3.5 w-3.5 text-accent" />
                  Last sync {formatLastSync(lastSyncedAt)}
                </span>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xl:w-[100%]">
              {statCard({
                label: 'Selected joint',
                value: selectedJointSummary?.joint_id || selectedJointId,
                tone: riskTone(selectedRisk),
                icon: <Shield className="h-5 w-5" />,
              })}
              {statCard({
                label: 'Simulation',
                value: simStatus.is_running ? 'RUNNING' : 'PAUSED',
                tone: simStatus.is_running ? 'border-risk-low bg-risk-low-soft glow-low' : 'border-risk-unknown bg-risk-unknown-soft',
                icon: <DatabaseZap className="h-5 w-5" />,
              })}
              {statCard({
                label: 'Active alerts',
                value: String(activeCount),
                tone: activeCount ? 'border-risk-medium bg-risk-medium-soft glow-med' : 'border-risk-unknown bg-risk-unknown-soft',
                icon: <AlertTriangle className="h-5 w-5" />,
              })}
              {statCard({
                label: 'Risk pressure',
                value: String(highestRiskCount),
                tone: highestRiskCount ? 'border-risk-high bg-risk-high-soft glow-high' : 'border-risk-unknown bg-risk-unknown-soft',
                icon: <ShieldAlert className="h-5 w-5" />,
              })}
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-4 pt-6 sm:px-6">
        {isLoadingInitial ? (
          <div className="panel panel-strong p-8 text-center text-sm text-muted">
            <div className="loading-shimmer mx-auto mb-4 h-16 w-16 rounded-full border border-accent/30" />
            <div className="mx-auto max-w-md">
              Initializing JointGuard telemetry and live REST polling...
            </div>
          </div>
        ) : (
          <>
            <SimulationControls
              isRunning={simStatus.is_running}
              onStart={() => runSimulationAction(() => fetch(`${API_BASE}/simulation/start`, { method: 'POST' }))}
              onStop={() => runSimulationAction(() => fetch(`${API_BASE}/simulation/stop`, { method: 'POST' }))}
              onReset={() => runSimulationAction(() => fetch(`${API_BASE}/simulation/reset`, { method: 'POST' }))}
              onStep={() => runSimulationAction(() => fetch(`${API_BASE}/simulation/step`, { method: 'POST' }))}
              onInject={(payload) =>
                runSimulationAction(() => fetch(`${API_BASE}/simulation/inject`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(payload),
                }))
              }
            />

            <ConveyorTwin
              joints={joints}
              selectedJointId={selectedJointId}
              onSelectJoint={setSelectedJointId}
              isRunning={simStatus.is_running}
            />

              <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
                <div className="space-y-6">
                  <JointDetailModal jointDetail={selectedJointDetail} />
                  <SensorCards jointDetail={selectedJointDetail} />
                </div>

                <div className="space-y-6">
                  <AlertsPanel activeAlerts={activeAlerts} alertHistory={alertHistory} />
                  <div className="panel panel-strong border border-accent/20 p-5">
                    <div className="section-kicker">Live summary</div>
                    <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-2">
                      {riskLabels && Object.keys(riskLabels).map((riskKey) => (
                        <div key={riskKey} className={`rounded-2xl border px-4 py-4 ${riskTone(riskKey)}`}>
                          <div className="text-[10px] uppercase tracking-[0.28em] text-muted">{riskKey}</div>
                          <div className="mt-2 text-2xl font-black text-main">{riskCounts[riskKey]}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </section>

              <TrendCharts jointId={selectedJointId} historyData={trendHistory} onRefresh={fetchAllData} />
          </>
        )}
      </main>

      <AboutModal isOpen={isAboutOpen} onClose={() => setIsAboutOpen(false)} />
    </div>
  );
}
