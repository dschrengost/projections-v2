import { useEffect, useMemo, useState } from 'react'

interface NumericTextInputProps {
    value: number | null
    onChangeValue: (value: number | null) => void
    min?: number
    max?: number
    step?: number
    placeholder?: string
    className?: string
    allowNull?: boolean
    integerOnly?: boolean
    inputMode?: 'numeric' | 'decimal'
}

function clamp(value: number, min?: number, max?: number): number {
    let next = value
    if (typeof min === 'number') next = Math.max(min, next)
    if (typeof max === 'number') next = Math.min(max, next)
    return next
}

function isTransientDraft(raw: string): boolean {
    const trimmed = raw.trim()
    return trimmed === '' || trimmed === '-' || trimmed === '.' || trimmed === '-.'
}

export default function NumericTextInput({
    value,
    onChangeValue,
    min,
    max,
    step,
    placeholder,
    className,
    allowNull = false,
    integerOnly = false,
    inputMode = 'numeric',
}: NumericTextInputProps) {
    const externalValue = useMemo(() => (value === null || value === undefined ? '' : String(value)), [value])
    const [draft, setDraft] = useState(externalValue)
    const [isFocused, setIsFocused] = useState(false)

    useEffect(() => {
        if (!isFocused) {
            setDraft(externalValue)
        }
    }, [externalValue, isFocused])

    const commit = (raw: string) => {
        const trimmed = raw.trim()
        if (trimmed === '') {
            if (allowNull) {
                onChangeValue(null)
                setDraft('')
                return
            }
            const fallback = clamp(value ?? min ?? 0, min, max)
            onChangeValue(integerOnly ? Math.round(fallback) : fallback)
            setDraft(String(integerOnly ? Math.round(fallback) : fallback))
            return
        }

        const parsed = integerOnly ? parseInt(trimmed, 10) : parseFloat(trimmed)
        if (!Number.isFinite(parsed)) {
            setDraft(externalValue)
            return
        }

        let next = clamp(parsed, min, max)
        if (integerOnly) {
            next = Math.round(next)
        }
        onChangeValue(next)
        setDraft(String(next))
    }

    return (
        <input
            type="text"
            inputMode={inputMode}
            value={draft}
            placeholder={placeholder}
            className={className}
            onFocus={() => setIsFocused(true)}
            onBlur={() => {
                setIsFocused(false)
                commit(draft)
            }}
            onChange={e => {
                const next = e.target.value
                setDraft(next)
                if (!isTransientDraft(next)) {
                    const parsed = integerOnly ? parseInt(next, 10) : parseFloat(next)
                    if (Number.isFinite(parsed)) {
                        let committed = clamp(parsed, min, max)
                        if (integerOnly) {
                            committed = Math.round(committed)
                        }
                        onChangeValue(committed)
                    }
                } else if (allowNull && next.trim() === '') {
                    onChangeValue(null)
                }
            }}
            onKeyDown={e => {
                if (e.key === 'Enter') {
                    e.preventDefault()
                    commit(draft)
                    ;(e.currentTarget as HTMLInputElement).blur()
                }
                if (e.key === 'Escape') {
                    e.preventDefault()
                    setDraft(externalValue)
                    ;(e.currentTarget as HTMLInputElement).blur()
                }
            }}
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={value ?? undefined}
            data-step={step}
        />
    )
}
