import * as React from 'react'
import { cn } from '@/lib/utils'

const Card = React.forwardRef<
    React.ElementRef<'div'>,
    React.ComponentPropsWithoutRef<'div'>
>(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn(
            'rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] text-[hsl(var(--card-foreground))] shadow-sm',
            className,
        )}
        {...props}
    />
))
Card.displayName = 'Card'

const CardHeader = React.forwardRef<
    React.ElementRef<'div'>,
    React.ComponentPropsWithoutRef<'div'>
>(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn('flex flex-col space-y-1.5 rounded-t-lg border-b border-[hsl(var(--border))] p-3', className)}
        {...props}
    />
))
CardHeader.displayName = 'CardHeader'

const CardContent = React.forwardRef<
    React.ElementRef<'div'>,
    React.ComponentPropsWithoutRef<'div'>
>(({ className, ...props }, ref) => (
    <div ref={ref} className={cn('p-3 pt-2.5', className)} {...props} />
))
CardContent.displayName = 'CardContent'

const CardFooter = React.forwardRef<
    React.ElementRef<'div'>,
    React.ComponentPropsWithoutRef<'div'>
>(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn('flex items-center p-3 pt-2', className)}
        {...props}
    />
))
CardFooter.displayName = 'CardFooter'

const CardTitle = React.forwardRef<
    React.ElementRef<'h3'>,
    React.ComponentPropsWithoutRef<'h3'>
>(({ className, ...props }, ref) => (
    <h3
        ref={ref}
        className={cn('text-sm font-semibold leading-none tracking-tight', className)}
        {...props}
    />
))
CardTitle.displayName = 'CardTitle'

const CardDescription = React.forwardRef<
    React.ElementRef<'p'>,
    React.ComponentPropsWithoutRef<'p'>
>(({ className, ...props }, ref) => (
    <p
        ref={ref}
        className={cn('text-sm text-[hsl(var(--muted-foreground))]', className)}
        {...props}
    />
))
CardDescription.displayName = 'CardDescription'

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent }
