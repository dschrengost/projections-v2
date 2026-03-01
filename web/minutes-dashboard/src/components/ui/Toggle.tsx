import { type InputHTMLAttributes } from 'react'

interface ToggleProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  label?: string
  description?: string
  onChange?: (checked: boolean) => void
}

export function Toggle({
  label,
  description,
  checked,
  onChange,
  disabled,
  className = '',
  ...props
}: ToggleProps) {
  return (
    <label
      className={`
        flex items-start gap-3 cursor-pointer
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
    >
      <div className="relative flex-shrink-0 mt-0.5">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange?.(e.target.checked)}
          disabled={disabled}
          className="sr-only peer"
          {...props}
        />
        <div
          className={`
            w-10 h-6 rounded-full
            bg-slate-700
            peer-checked:bg-indigo-600
            peer-focus:ring-2 peer-focus:ring-indigo-500 peer-focus:ring-offset-2 peer-focus:ring-offset-slate-900
            transition-colors duration-200
          `}
        />
        <div
          className={`
            absolute top-1 left-1
            w-4 h-4 rounded-full
            bg-white
            peer-checked:translate-x-4
            transition-transform duration-200
          `}
        />
      </div>
      {(label || description) && (
        <div className="flex flex-col">
          {label && (
            <span className="text-sm font-medium text-slate-100">
              {label}
            </span>
          )}
          {description && (
            <span className="text-xs text-slate-500 mt-0.5">
              {description}
            </span>
          )}
        </div>
      )}
    </label>
  )
}
