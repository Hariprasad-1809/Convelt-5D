import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';

export default function AppLayout() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="app-shell">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(c => !c)}
        mobileOpen={mobileOpen}
        onMobileClose={() => setMobileOpen(false)}
      />

      <div className={`main-area ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        <Header
          onMobileToggle={() => setMobileOpen(o => !o)}
        />
        <main className="page-wrapper" id="main-content" role="main">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
