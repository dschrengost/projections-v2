const API_BASE = (import.meta.env.VITE_MINUTES_API || '').replace(/\/$/, '')

export const apiUrl = (path: string) => `${API_BASE}${path}`
