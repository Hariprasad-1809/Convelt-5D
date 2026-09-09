import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';

/**
 * Generic accessible modal with overlay, focus trap, and Escape key handling.
 * @param {boolean}   isOpen
 * @param {Function}  onClose
 * @param {string}    title
 * @param {ReactNode} children
 * @param {string}    [size] - 'sm' | 'md' | 'lg' | 'xl'
 */
export default function Modal({ isOpen, onClose, title, children, size = 'md' }) {
  const overlayRef = useRef(null);
  const panelRef   = useRef(null);

  useEffect(() => {
    if (!isOpen) return;
    const prev = document.activeElement;
    panelRef.current?.focus();

    const handleKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKey);
    document.body.style.overflow = 'hidden';

    return () => {
      document.removeEventListener('keydown', handleKey);
      document.body.style.overflow = '';
      prev?.focus();
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const maxWidth = { sm: 480, md: 600, lg: 800, xl: 1000 }[size] ?? 600;

  return (
    <div
      ref={overlayRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 200,
        padding: '16px',
      }}
      onClick={(e) => { if (e.target === overlayRef.current) onClose(); }}
    >
      <div
        ref={panelRef}
        tabIndex={-1}
        style={{
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border-default)',
          borderRadius: 'var(--radius-lg)',
          width: '100%',
          maxWidth,
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          outline: 'none',
          boxShadow: 'var(--shadow-lg)',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '16px 20px',
          borderBottom: '1px solid var(--border-subtle)',
          flexShrink: 0,
        }}>
          <h2 id="modal-title" style={{ fontSize: 'var(--text-base)', fontWeight: 600 }}>
            {title}
          </h2>
          <button
            onClick={onClose}
            aria-label="Close modal"
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: '4px',
              borderRadius: 'var(--radius-sm)',
              display: 'flex',
              transition: 'color var(--transition)',
            }}
            onMouseOver={e => e.currentTarget.style.color = 'var(--text-primary)'}
            onMouseOut={e => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '20px', overflowY: 'auto', flex: 1 }}>
          {children}
        </div>
      </div>
    </div>
  );
}
