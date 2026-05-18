import React, { useMemo } from 'react'
import {
  Activity,
  BarChart3,
  Clock3,
  Database,
  FileText,
  Gauge,
  Layers,
  LineChart,
  Microscope,
  ShieldCheck,
} from 'lucide-react'
import type { AnalysisRun, Metrics } from '../types'

interface Props {
  metrics?: Metrics
  history: AnalysisRun[]
}

interface ScoreCard {
  label: string
  value: string
  detail: string
  score: number | null
  accent: string
}

function isValidNumber(value?: number | null): value is number {
  return value !== null && value !== undefined && Number.isFinite(value)
}

function average(values: Array<number | null | undefined>): number | null {
  const validValues = values.filter(isValidNumber)
  if (!validValues.length) return null
  return validValues.reduce((sum, value) => sum + value, 0) / validValues.length
}

function formatNumber(value?: number | null, digits = 2): string {
  if (!isValidNumber(value)) return 'N/A'
  return value.toFixed(digits)
}

function formatInteger(value?: number | null): string {
  if (!isValidNumber(value)) return 'N/A'
  return Math.round(value).toLocaleString('uk-UA')
}

function formatPercent(value?: number | null): string {
  if (!isValidNumber(value)) return 'N/A'
  return `${value.toFixed(2)}%`
}

function formatFileSize(bytes?: number | null): string {
  if (!isValidNumber(bytes)) return 'N/A'
  if (bytes < 1024) return `${Math.round(bytes)} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function formatTime(seconds?: number | null): string {
  if (!isValidNumber(seconds)) return 'N/A'
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`
  return `${seconds.toFixed(2)} s`
}

function clampScore(value: number | null, min: number, max: number): number | null {
  if (!isValidNumber(value)) return null
  return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100))
}

function buildPolyline(values: number[], width: number, height: number): string {
  if (values.length === 1) return `0,${height - values[0] * height} ${width},${height - values[0] * height}`

  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width
      const y = height - value * height
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
}

export const AnalyticsDashboard: React.FC<Props> = ({ metrics, history }) => {
  const lastRuns = history.slice(-12)

  const aggregate = useMemo(() => {
    const allMetrics = history.map((run) => run.metrics || {})

    return {
      totalRuns: history.length,
      avgPsnr: average(allMetrics.map((item) => item.psnr)),
      avgSsim: average(allMetrics.map((item) => item.ssim)),
      avgCad: average(allMetrics.map((item) => item.cad_readiness_score)),
      avgGapless: average(allMetrics.map((item) => item.gapless_coverage)),
      avgNodes: average(allMetrics.map((item) => item.node_count)),
      avgTime: average(allMetrics.map((item) => item.processing_time)),
    }
  }, [history])

  const cards: ScoreCard[] = [
    {
      label: 'Pixel Fidelity',
      value: isValidNumber(metrics?.psnr) ? `${formatNumber(metrics?.psnr, 2)} dB` : 'N/A',
      detail: 'PSNR показує точність відтворення пікселів після SVG-рендера.',
      score: clampScore(metrics?.psnr ?? null, 12, 32),
      accent: 'bg-emerald-500',
    },
    {
      label: 'Structure Match',
      value: formatNumber(metrics?.ssim, 3),
      detail: 'SSIM вимірює структурну схожість з оригіналом.',
      score: clampScore(metrics?.ssim ?? null, 0.25, 0.95),
      accent: 'bg-blue-500',
    },
    {
      label: 'Gapless Coverage',
      value: formatPercent(metrics?.gapless_coverage),
      detail: 'Частка полотна без прозорих проміжків між шарами.',
      score: clampScore(metrics?.gapless_coverage ?? null, 80, 100),
      accent: 'bg-violet-500',
    },
    {
      label: 'CAD Readiness',
      value: formatNumber(metrics?.cad_readiness_score, 1),
      detail: 'Індекс готовності до CAD: щільність, редагованість і відсутність gaps.',
      score: clampScore(metrics?.cad_readiness_score ?? null, 35, 100),
      accent: 'bg-amber-500',
    },
  ]

  const trendValues = lastRuns
    .map((run) => run.metrics?.cad_readiness_score)
    .filter(isValidNumber)
    .map((value) => Math.max(0, Math.min(1, value / 100)))

  const polyline = trendValues.length ? buildPolyline(trendValues, 480, 120) : ''

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {cards.map((card) => (
          <div key={card.label} className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-bold text-gray-400 uppercase tracking-wide">{card.label}</p>
                <p className="text-3xl font-extrabold text-gray-900 mt-2">{card.value}</p>
              </div>

              <div className="w-10 h-10 rounded-xl bg-violet-50 text-violet-700 flex items-center justify-center">
                <Gauge size={18} />
              </div>
            </div>

            <div className="h-2 bg-gray-100 rounded-full overflow-hidden mt-5">
              <div
                className={`h-full ${card.accent}`}
                style={{ width: `${card.score ?? 0}%` }}
              />
            </div>

            <p className="text-xs leading-relaxed text-gray-500 mt-3">{card.detail}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1.2fr_0.8fr] gap-5">
        <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
          <div className="flex items-center justify-between gap-4 mb-4">
            <div>
              <div className="flex items-center gap-2">
                <LineChart size={18} className="text-violet-600" />
                <h3 className="font-bold text-gray-900">Тренд CAD Readiness</h3>
              </div>
              <p className="text-sm text-gray-500 mt-1">Останні {lastRuns.length} векторизацій у цьому браузері</p>
            </div>

            <span className="px-3 py-1 rounded-full bg-violet-50 text-violet-700 text-xs font-bold">
              Avg {formatNumber(aggregate.avgCad, 1)}
            </span>
          </div>

          <div className="h-44 rounded-xl bg-gray-50 border border-gray-100 p-4">
            {polyline ? (
              <svg viewBox="0 0 480 120" className="w-full h-full overflow-visible" role="img">
                <line x1="0" y1="20" x2="480" y2="20" stroke="#e5e7eb" strokeDasharray="4 4" />
                <line x1="0" y1="60" x2="480" y2="60" stroke="#e5e7eb" strokeDasharray="4 4" />
                <line x1="0" y1="100" x2="480" y2="100" stroke="#e5e7eb" strokeDasharray="4 4" />
                <polyline points={polyline} fill="none" stroke="#7c3aed" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
                {trendValues.map((value, index) => {
                  const x = trendValues.length === 1 ? 240 : (index / (trendValues.length - 1)) * 480
                  const y = 120 - value * 120
                  return <circle key={`${index}-${value}`} cx={x} cy={y} r="5" fill="#111827" />
                })}
              </svg>
            ) : (
              <div className="h-full flex items-center justify-center text-sm text-gray-400">
                Дані тренду з'являться після кількох векторизацій
              </div>
            )}
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Database size={18} className="text-violet-600" />
            <h3 className="font-bold text-gray-900">Загальна статистика</h3>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl bg-gray-50 p-4">
              <p className="text-xs text-gray-400">Фото</p>
              <p className="text-2xl font-extrabold text-gray-900">{aggregate.totalRuns}</p>
            </div>
            <div className="rounded-xl bg-gray-50 p-4">
              <p className="text-xs text-gray-400">Avg SSIM</p>
              <p className="text-2xl font-extrabold text-gray-900">{formatNumber(aggregate.avgSsim, 3)}</p>
            </div>
            <div className="rounded-xl bg-gray-50 p-4">
              <p className="text-xs text-gray-400">Avg gaps</p>
              <p className="text-2xl font-extrabold text-gray-900">{formatPercent(aggregate.avgGapless)}</p>
            </div>
            <div className="rounded-xl bg-gray-50 p-4">
              <p className="text-xs text-gray-400">Avg time</p>
              <p className="text-2xl font-extrabold text-gray-900">{formatTime(aggregate.avgTime)}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <Microscope size={18} className="text-violet-600" />
            <h3 className="font-bold text-gray-900">Науковий паспорт</h3>
          </div>
          <div className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-gray-500">PSNR baseline</span>
              <span className="font-bold text-gray-900">{formatNumber(aggregate.avgPsnr, 2)} dB</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-500">Gap pixels</span>
              <span className="font-bold text-gray-900">{formatInteger(metrics?.gap_pixels)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-500">Node budget</span>
              <span className="font-bold text-gray-900">{formatInteger(metrics?.node_count)}</span>
            </div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <Layers size={18} className="text-violet-600" />
            <h3 className="font-bold text-gray-900">Редагованість</h3>
          </div>
          <div className="mt-4 space-y-3 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-gray-500">Layers</span>
              <span className="font-bold text-gray-900">{formatInteger(metrics?.layer_count)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-500">Editability</span>
              <span className="font-bold text-gray-900">{formatNumber(metrics?.editability_score, 1)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-500">SVG size</span>
              <span className="font-bold text-gray-900">{formatFileSize(metrics?.file_size)}</span>
            </div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
          <div className="flex items-center gap-2">
            <ShieldCheck size={18} className="text-violet-600" />
            <h3 className="font-bold text-gray-900">Висновок</h3>
          </div>
          <p className="text-sm text-gray-500 leading-relaxed mt-4">
            Найважливіші KPI для доказу цінності: високий CAD Readiness, gapless coverage близький до 100%,
            стабільний SSIM і контрольована кількість вузлів. Це прямо показує, чи SVG придатний для дизайну,
            ілюстрації та CAD-редагування.
          </p>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <Activity size={18} className="text-violet-600" />
          <h3 className="font-bold text-gray-900">Останні вимірювання</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-400 uppercase border-b border-gray-100">
                <th className="py-3 pr-4">Дата</th>
                <th className="py-3 pr-4">Mode</th>
                <th className="py-3 pr-4">PSNR</th>
                <th className="py-3 pr-4">SSIM</th>
                <th className="py-3 pr-4">CAD</th>
                <th className="py-3 pr-4">Вузли</th>
                <th className="py-3 pr-4">Час</th>
              </tr>
            </thead>
            <tbody>
              {lastRuns.slice().reverse().map((run) => (
                <tr key={run.id} className="border-b border-gray-50">
                  <td className="py-3 pr-4 text-gray-500">
                    {new Date(run.createdAt).toLocaleString('uk-UA', {
                      day: '2-digit',
                      month: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </td>
                  <td className="py-3 pr-4 font-semibold text-gray-800">{run.mode || run.requestedMode || 'auto'}</td>
                  <td className="py-3 pr-4">{formatNumber(run.metrics?.psnr, 2)}</td>
                  <td className="py-3 pr-4">{formatNumber(run.metrics?.ssim, 3)}</td>
                  <td className="py-3 pr-4">{formatNumber(run.metrics?.cad_readiness_score, 1)}</td>
                  <td className="py-3 pr-4">{formatInteger(run.metrics?.node_count)}</td>
                  <td className="py-3 pr-4">{formatTime(run.metrics?.processing_time)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
