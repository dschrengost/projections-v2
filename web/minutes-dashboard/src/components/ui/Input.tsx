import { forwardRef, type InputHTMLAttributes } from 'react'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  hint?: string
  error?: string
  fullWidth?: boolean
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, hint, error, fullWidth = false, className = '', ...props }, ref) => {
    return (
      <div className={`flex flex-col gap-1.5 ${fullWidth ? 'w-full' : ''}`}>
        {label && (
          <label className="text-sm font-medium text-slate-400">
            {label}
          </label>
        )}
        <input
          ref={ref}
          className={`
            px-3 py-2
            bg-slate-900
            border border-slate-700
            rounded-lg
            text-sm text-slate-100
            placeholder:text-slate-500
            focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-colors duration-150
            ${error ? 'border-red-500 focus:ring-red-500' : ''}
            ${fullWidth ? 'w-full' : ''}
            ${className}
          `.trim().replace(/\s+/g, ' ')}
          {...props}
        />
        {hint && !error && (
          <span className="text-xs text-slate-500 italic">{hint}</span>
        )}
        {error && (
          <span className="text-xs text-red-500">{error}</span>
        )}
      </div>
    )
  }
)

Input.displayName = 'Input'

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
  hint?: string
  showValue?: boolean
  valueFormatter?: (value: number) => string
}

export const Slider = forwardRef<HTMLInputElement, SliderProps>(
  ({ label, hint, showValue = true, valueFormatter, className = '', value, ...props }, ref) => {
    const displayValue = valueFormatter
      ? valueFormatter(Number(value))
      : String(value)

    return (
      <div className="flex flex-col gap-2">
        {(label || showValue) && (
          <div className="flex items-center justify-between">
            {label && (
              <label className="text-sm font-medium text-slate-400">
                {label}
              </label>
            )}
            {showValue && (
              <span className="text-sm font-semibold text-slate-100">
                {displayValue}
              </span>
            )}
          </div>
        )}
        <input
          ref={ref}
          type="range"
          value={value}
          className={`
            w-full h-2
            bg-slate-700
            rounded-full
            appearance-none
            cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-gradient-to-r
            [&::-webkit-slider-thumb]:from-indigo-600
            [&::-webkit-slider-thumb]:to-purple-600
            [&::-webkit-slider-thumb]:border-2
            [&::-webkit-slider-thumb]:border-slate-800
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:transition-transform
            [&::-webkit-slider-thumb]:hover:scale-110
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:bg-gradient-to-r
            [&::-moz-range-thumb]:from-indigo-600
            [&::-moz-range-thumb]:to-purple-600
            [&::-moz-range-thumb]:border-2
            [&::-moz-range-thumb]:border-slate-800
            [&::-moz-range-thumb]:cursor-pointer
            disabled:opacity-50 disabled:cursor-not-allowed
            ${className}
          `.trim().replace(/\s+/g, ' ')}
          {...props}
        />
        {hint && (
          <span className="text-xs text-slate-500 italic">{hint}</span>
        )}
      </div>
    )
  }
)

Slider.displayName = 'Slider'

interface SelectProps extends InputHTMLAttributes<HTMLSelectElement> {
  label?: string
  options: Array<{ value: string | number; label: string }>
  fullWidth?: boolean
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, options, fullWidth = false, className = '', ...props }, ref) => {
    return (
      <div className={`flex flex-col gap-1.5 ${fullWidth ? 'w-full' : ''}`}>
        {label && (
          <label className="text-sm font-medium text-slate-400">
            {label}
          </label>
        )}
        <select
          ref={ref}
          className={`
            px-3 py-2
            bg-slate-900
            border border-slate-700
            rounded-lg
            text-sm text-slate-100
            focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
            disabled:opacity-50 disabled:cursor-not-allowed
            cursor-pointer
            transition-colors duration-150
            ${fullWidth ? 'w-full' : ''}
            ${className}
          `.trim().replace(/\s+/g, ' ')}
          {...props}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
    )
  }
)

Select.displayName = 'Select'
