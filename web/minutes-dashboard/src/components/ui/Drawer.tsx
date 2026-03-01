import { type ReactNode, useEffect } from 'react'

interface DrawerProps {
  open: boolean
  onClose: () => void
  children: ReactNode
  title?: string
  position?: 'left' | 'right' | 'bottom'
}

export function Drawer({
  open,
  onClose,
  children,
  title,
  position = 'right',
}: DrawerProps) {
  // Prevent body scroll when drawer is open
  useEffect(() => {
    if (open) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [open])

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    if (open) {
      document.addEventListener('keydown', handleEscape)
    }
    return () => document.removeEventListener('keydown', handleEscape)
  }, [open, onClose])

  const positionStyles = {
    left: {
      container: 'left-0 top-0 h-full',
      panel: 'w-80 max-w-[85vw] h-full',
      translate: open ? 'translate-x-0' : '-translate-x-full',
    },
    right: {
      container: 'right-0 top-0 h-full',
      panel: 'w-80 max-w-[85vw] h-full',
      translate: open ? 'translate-x-0' : 'translate-x-full',
    },
    bottom: {
      container: 'bottom-0 left-0 w-full',
      panel: 'w-full max-h-[85vh] rounded-t-2xl',
      translate: open ? 'translate-y-0' : 'translate-y-full',
    },
  }

  const styles = positionStyles[position]

  return (
    <>
      {/* Backdrop */}
      <div
        className={`
          fixed inset-0 z-40
          bg-black/60 backdrop-blur-sm
          transition-opacity duration-300
          ${open ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        `}
        onClick={onClose}
      />

      {/* Drawer Panel */}
      <div
        className={`
          fixed z-50
          ${styles.container}
        `}
      >
        <div
          className={`
            ${styles.panel}
            bg-slate-800
            border-slate-700
            ${position === 'left' ? 'border-r' : ''}
            ${position === 'right' ? 'border-l' : ''}
            ${position === 'bottom' ? 'border-t' : ''}
            flex flex-col
            transform transition-transform duration-300 ease-out
            ${styles.translate}
          `}
        >
          {/* Header */}
          {title && (
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
              <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
              <button
                onClick={onClose}
                className="
                  p-2 -mr-2
                  text-slate-500 hover:text-slate-100
                  hover:bg-slate-700
                  rounded-lg
                  transition-colors
                "
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}

          {/* Bottom sheet drag handle */}
          {position === 'bottom' && (
            <div className="flex justify-center py-2">
              <div className="w-12 h-1 rounded-full bg-slate-700" />
            </div>
          )}

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4">
            {children}
          </div>
        </div>
      </div>
    </>
  )
}
