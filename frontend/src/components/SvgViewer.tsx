import React, { useEffect, useMemo, useState } from 'react'
import { Download, FileText, ZoomIn, ZoomOut } from 'lucide-react'
import type { EditableLayer } from '../types'
import { getSvgDownloadUrl } from '../api/vectorize'

interface Props {
  svgUrl: string
  taskId: string
  layers?: EditableLayer[]
  selectedLayerId?: string | null
}

function findSvgGroup(svg: SVGElement, layer: EditableLayer): Element | null {
  return svg.querySelector(`g[id^="${layer.id}-"]`) || (
    layer.id === 'layer-1' ? svg.querySelector('g#layer-logo') : null
  )
}

function applyLayerState(svgContent: string, layers?: EditableLayer[], selectedLayerId?: string | null): string {
  if (!svgContent) return ''

  try {
    const parser = new DOMParser()
    const doc = parser.parseFromString(svgContent, 'image/svg+xml')
    const svg = doc.querySelector('svg')

    if (!svg) return svgContent

    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet')
    svg.setAttribute(
      'style',
      'max-width:100%;max-height:100%;width:auto;height:auto;display:block;'
    )

    if (layers?.length) {
      const orderedGroups: Element[] = []
      const activePrefixes = new Set(layers.map((layer) => `${layer.id}-`))

      Array.from(svg.querySelectorAll('g[id^="layer-"]')).forEach((group) => {
        const id = group.getAttribute('id') || ''
        const stillExists =
          Array.from(activePrefixes).some((prefix) => id.startsWith(prefix)) ||
          (id === 'layer-logo' && layers.some((layer) => layer.id === 'layer-1'))

        if (!stillExists) {
          group.remove()
        }
      })

      layers.forEach((layer) => {
        const group = findSvgGroup(svg, layer)
        if (!group) return

        if (!layer.visible) {
          group.setAttribute('display', 'none')
        } else {
          group.removeAttribute('display')
        }

        group.setAttribute('data-layer-name', layer.displayName)

        Array.from(group.querySelectorAll('path')).forEach((path) => {
          path.removeAttribute('filter')
          path.removeAttribute('vector-effect')

          if (selectedLayerId === layer.id) {
            path.setAttribute('stroke', '#7C3AED')
            path.setAttribute('stroke-width', '4')
            path.setAttribute('vector-effect', 'non-scaling-stroke')
          }
        })

        orderedGroups.push(group)
      })

      orderedGroups.forEach((group) => {
        svg.appendChild(group)
      })
    }

    return new XMLSerializer().serializeToString(svg)
  } catch (error) {
    console.error(error)
    return svgContent
  }
}

export const SvgViewer: React.FC<Props> = ({
  svgUrl,
  taskId,
  layers,
  selectedLayerId,
}) => {
  const [svgContent, setSvgContent] = useState('')
  const [zoom, setZoom] = useState(1)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    async function loadSvg() {
      try {
        setLoadError(null)

        const response = await fetch(svgUrl)
        if (!response.ok) throw new Error('Не вдалося завантажити SVG')

        const text = await response.text()
        if (isMounted) setSvgContent(text)
      } catch (error) {
        console.error(error)
        if (isMounted) setLoadError('Не вдалося завантажити SVG')
      }
    }

    loadSvg()

    return () => {
      isMounted = false
    }
  }, [svgUrl])

  const modifiedSvgContent = useMemo(
    () => applyLayerState(svgContent, layers, selectedLayerId),
    [svgContent, layers, selectedLayerId]
  )

  const downloadUrl = useMemo(() => {
    if (!modifiedSvgContent) return getSvgDownloadUrl(taskId)

    const blob = new Blob([modifiedSvgContent], { type: 'image/svg+xml' })
    return URL.createObjectURL(blob)
  }, [modifiedSvgContent, taskId])

  useEffect(() => {
    return () => {
      if (downloadUrl.startsWith('blob:')) URL.revokeObjectURL(downloadUrl)
    }
  }, [downloadUrl])

  return (
    <div className="space-y-5">
      <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden shadow-sm">
        <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between gap-4">
          <div>
            <h2 className="font-bold text-gray-900">Результат векторизації</h2>
            <p className="text-sm text-gray-500 mt-1">SVG-перегляд із масштабуванням</p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setZoom((current) => Math.max(0.25, current - 0.25))}
              className="w-9 h-9 rounded-lg border border-gray-200 flex items-center justify-center text-gray-600 hover:bg-gray-50"
              title="Зменшити"
            >
              <ZoomOut size={17} />
            </button>

            <span className="text-sm font-semibold text-gray-500 w-14 text-center">
              {Math.round(zoom * 100)}%
            </span>

            <button
              type="button"
              onClick={() => setZoom((current) => Math.min(3, current + 0.25))}
              className="w-9 h-9 rounded-lg border border-gray-200 flex items-center justify-center text-gray-600 hover:bg-gray-50"
              title="Збільшити"
            >
              <ZoomIn size={17} />
            </button>
          </div>
        </div>

        <div className="bg-[linear-gradient(45deg,#e5e7eb_25%,transparent_25%),linear-gradient(-45deg,#e5e7eb_25%,transparent_25%),linear-gradient(45deg,transparent_75%,#e5e7eb_75%),linear-gradient(-45deg,transparent_75%,#e5e7eb_75%)] bg-[length:20px_20px] bg-[position:0_0,0_10px,10px_-10px,-10px_0px] h-[min(72vh,760px)] min-h-[420px] flex items-center justify-center overflow-auto p-4 sm:p-6">
          {loadError && (
            <div className="text-center text-red-600">
              <FileText size={32} className="mx-auto mb-2" />
              <p className="font-semibold">{loadError}</p>
            </div>
          )}

          {!loadError && !svgContent && (
            <div className="text-gray-400 text-sm">Завантажуємо SVG...</div>
          )}

          {!loadError && svgContent && (
            <div
              style={{
                transform: `scale(${zoom})`,
                transformOrigin: 'center',
              }}
              className="w-full h-full flex items-center justify-center transition-transform [&_svg]:max-w-full [&_svg]:max-h-full [&_svg]:w-auto [&_svg]:h-auto [&_svg]:block"
              dangerouslySetInnerHTML={{ __html: modifiedSvgContent }}
            />
          )}
        </div>
      </div>

      <a
        href={downloadUrl}
        download={`${taskId}.svg`}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-violet-600 hover:bg-violet-700 text-white font-semibold transition-colors"
      >
        <Download size={18} />
        Завантажити SVG
      </a>
    </div>
  )
}
