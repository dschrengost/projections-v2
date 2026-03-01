import { forwardRef, type InputHTMLAttributes } from 'react'

interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
  variant?: 'default' | 'lock' | 'ban'
}

const variantStyles = {
  default: `
    border-slate-700
    checked:bg-indigo-600 checked:border-indigo-500
    focus:ring-indigo-500
  `,
  lock: `
    border-green-500/50
    checked:bg-green-500 checked:border-green-500
    focus:ring-green-500
  `,
  ban: `
    border-red-500/50
    checked:bg-red-500 checked:border-red-500
    focus:ring-red-500
  `,
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  ({ label, variant = 'default', className = '', ...props }, ref) => {
    return (
      <label className="inline-flex items-center gap-2 cursor-pointer">
        <input
          ref={ref}
          type="checkbox"
          className={`
            w-4 h-4
            rounded
            bg-slate-900
            border-2
            cursor-pointer
            transition-colors duration-150
            focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900
            disabled:opacity-50 disabled:cursor-not-allowed
            ${variantStyles[variant]}
            ${className}
          `.trim().replace(/\s+/g, ' ')}
          {...props}
        />
        {label && (
          <span className="text-sm text-slate-100">{label}</span>
        )}
      </label>
    )
  }
)

Checkbox.displayName = 'Checkbox'
