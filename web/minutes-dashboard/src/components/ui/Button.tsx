import { forwardRef, type ButtonHTMLAttributes } from 'react'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'success'
type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ButtonSize
  fullWidth?: boolean
  loading?: boolean
}

const variantStyles: Record<ButtonVariant, string> = {
  primary: `
    bg-gradient-to-r from-indigo-600 to-purple-600
    text-white font-semibold
    hover:opacity-90
    focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-900
  `,
  secondary: `
    bg-slate-800 border border-slate-700
    text-slate-100
    hover:bg-slate-700 hover:border-indigo-500
    focus:ring-2 focus:ring-border-accent focus:ring-offset-2 focus:ring-offset-slate-900
  `,
  ghost: `
    bg-transparent
    text-slate-400
    hover:bg-slate-800 hover:text-slate-100
    focus:ring-2 focus:ring-border-default focus:ring-offset-2 focus:ring-offset-slate-900
  `,
  danger: `
    bg-red-500/20 border border-red-500/50
    text-red-500
    hover:bg-red-500/30
    focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-slate-900
  `,
  success: `
    bg-green-500/20 border border-green-500/50
    text-green-500
    hover:bg-green-500/30
    focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-slate-900
  `,
}

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-2.5 py-1.5 text-xs rounded-md',
  md: 'px-4 py-2 text-sm rounded-lg',
  lg: 'px-6 py-3 text-base rounded-lg',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      fullWidth = false,
      loading = false,
      disabled,
      className = '',
      children,
      ...props
    },
    ref
  ) => {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={`
          inline-flex items-center justify-center gap-2
          font-medium
          transition-all duration-150 ease-out
          disabled:opacity-50 disabled:cursor-not-allowed
          ${variantStyles[variant]}
          ${sizeStyles[size]}
          ${fullWidth ? 'w-full' : ''}
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      >
        {loading && (
          <svg
            className="animate-spin h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
        )}
        {children}
      </button>
    )
  }
)

Button.displayName = 'Button'
