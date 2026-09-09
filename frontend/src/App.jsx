import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import './App.css';
import { SimulationProvider } from './context/SimulationContext';
import AppLayout from './components/layout/AppLayout';

// Pages
import Dashboard       from './pages/Dashboard';
import JointMonitoring from './pages/JointMonitoring';
import SensorData      from './pages/SensorData';
import Alerts          from './pages/Alerts';
import HistoryReports  from './pages/HistoryReports';
import AboutProject    from './pages/AboutProject';
import Settings        from './pages/Settings';

/**
 * AppRoutes — lives inside SimulationProvider so that
 * all pages share the same real backend telemetry and state.
 */
function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index             element={<Dashboard />} />
        <Route path="/joints"    element={<JointMonitoring />} />
        <Route path="/sensors"   element={<SensorData />} />
        <Route path="/alerts"    element={<Alerts />} />
        <Route path="/history"   element={<HistoryReports />} />
        <Route path="/about"     element={<AboutProject />} />
        <Route path="/settings"  element={<Settings />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <SimulationProvider>
        <AppRoutes />
      </SimulationProvider>
    </BrowserRouter>
  );
}
