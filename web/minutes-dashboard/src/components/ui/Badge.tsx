import { type HTMLAttributes, type ReactNode } from 'react'

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info' | 'salary' | 'projection' | 'ownership' | 'ceiling'
type BadgeSize = 'sm' | 'md'

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  children: ReactNode
  variant?: BadgeVariant
  size?: BadgeSize
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-slate-700 text-slate-400',
  success: 'bg-green-500/20 text-green-500 border border-green-500/30',
  warning: 'bg-amber-500/20 text-amber-500 border border-amber-500/30',
  danger: 'bg-red-500/20 text-red-500 border border-red-500/30',
  info: 'bg-blue-500/20 text-blue-500 border border-blue-500/30',
  salary: 'bg-amber-400/20 text-amber-400',
  projection: 'bg-green-400/20 text-green-400',
  ownership: 'bg-violet-400/20 text-violet-400',
  ceiling: 'bg-cyan-400/20 text-cyan-400',
}

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'px-1.5 py-0.5 text-[10px]',
  md: 'px-2 py-1 text-xs',
}

export function Badge({
  children,
  variant = 'default',
  size = 'md',
  className = '',
  ...props
}: BadgeProps) {
  return (
    <span
      className={`
        inline-flex items-center justify-center
        font-semibold
        rounded-full
        whitespace-nowrap
        ${variantStyles[variant]}
        ${sizeStyles[size]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {children}
    </span>
  )
}

interface StatBadgeProps extends HTMLAttributes<HTMLDivElement> {
  label: string
  value: string | number
  variant?: 'salary' | 'projection' | 'ownership' | 'ceiling'
}

export function StatBadge({
  label,
  value,
  variant = 'projection',
  className = '',
  ...props
}: StatBadgeProps) {
  const colorMap = {
    salary: 'text-amber-400',
    projection: 'text-green-400',
    ownership: 'text-violet-400',
    ceiling: 'text-cyan-400',
  }

  return (
    <div
      className={`
        flex items-center gap-1
        text-xs
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      <span className="text-slate-500">{label}</span>
      <span className={`font-semibold ${colorMap[variant]}`}>{value}</span>
    </div>
  )
}
