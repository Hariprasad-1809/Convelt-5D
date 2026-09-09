/**
 * JointGuard — useAlerts (Context Wrapper)
 *
 * All alert state is now managed by the centralized SimulationContext,
 * which automatically creates and resolves alerts when sensor status changes.
 * This hook is a thin compatibility shim — every existing consumer continues
 * to work without any changes.
 *
 * To switch to a real backend:
 *   → Replace SimulationContext.jsx alert logic with API calls.
 *   → This hook and all consumers stay exactly as they are.
 */

import { useSimulation } from '../context/SimulationContext';

export function useAlerts() {
  const { alerts, activeAlerts, resolvedAlerts, acknowledge, resolve, summary } = useSimulation();
  return { alerts, activeAlerts, resolvedAlerts, acknowledge, resolve, summary };
}
