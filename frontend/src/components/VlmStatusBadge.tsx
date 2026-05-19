import React, { useEffect, useState } from 'react'
import { CheckCircle2, Loader2 } from 'lucide-react'

import { fetchVlmStatus, VlmStatus } from '../api/vectorize'


export function VlmStatusBadge() {
  const [status, setStatus] = useState<VlmStatus | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    async function loadStatus() {
      try {
        const nextStatus = await fetchVlmStatus()
        if (cancelled) return

        setStatus(nextStatus)
        if (nextStatus.enabled && !nextStatus.ready && !nextStatus.load_failed) {
          timer = window.setTimeout(loadStatus, 5000)
        }
      } catch {
        if (cancelled) return
        setStatus(null)
      }
    }

    loadStatus()

    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  if (!status?.enabled || status.load_failed) return null

  if (status.ready) {
    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-800">
        <CheckCircle2 size={14} />
        <span className="h-2 w-2 rounded-full bg-emerald-500" />
        AI-іменування активне
      </div>
    )
  }

  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-gray-50 px-3 py-1.5 text-xs font-semibold text-gray-600">
      <Loader2 size={14} className="animate-spin" />
      Завантаження AI-іменування шарів...
    </div>
  )
}
