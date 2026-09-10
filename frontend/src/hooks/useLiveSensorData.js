/**
 * JointGuard — useLiveSensorData (Context Wrapper)
 *
 * All live sensor data now comes from the centralized SimulationContext.
 * This hook is a thin compatibility shim — every existing consumer continues
 * to work without any changes.
 *
 * To switch from mock simulation to a real FastAPI backend:
 *   → Replace SimulationContext.jsx's interval tick with API polling / WebSocket.
 *   → This hook and all consumers stay exactly as they are.
 */

import { useSimulation } from '../context/SimulationContext';

export function useLiveSensorData() {
  const { joints, lastUpdated, updateCount, hardwareStatus, wsStatus, apiConnected, visionData, cameraStatus } = useSimulation();
  return { joints, lastUpdated, updateCount, hardwareStatus, wsStatus, apiConnected, visionData, cameraStatus };
}


