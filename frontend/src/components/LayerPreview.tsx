import React from 'react'
import {
  ChevronDown,
  ChevronUp,
  Eye,
  EyeOff,
  Layers,
  Pencil,
  Trash2,
} from 'lucide-react'
import type { EditableLayer } from '../types'

interface Props {
  layers: EditableLayer[]
  selectedLayerId: string | null
  onSelectLayer: (id: string | null) => void
  onToggleVisibility: (id: string) => void
  onRenameLayer: (id: string, name: string) => void
  onDeleteLayer: (id: string) => void
  onMoveLayer: (id: string, direction: -1 | 1) => void
}

function isDarkColor(hexColor: string): boolean {
  const hex = hexColor.replace('#', '')

  if (hex.length !== 6) return false

  const r = parseInt(hex.slice(0, 2), 16)
  const g = parseInt(hex.slice(2, 4), 16)
  const b = parseInt(hex.slice(4, 6), 16)
  const luminance = 0.299 * r + 0.587 * g + 0.114 * b

  return luminance < 90
}

export const LayerPreview: React.FC<Props> = ({
  layers,
  selectedLayerId,
  onSelectLayer,
  onToggleVisibility,
  onRenameLayer,
  onDeleteLayer,
  onMoveLayer,
}) => (
  <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden w-full max-w-full">
    <div className="px-4 py-4 border-b border-gray-100 flex items-center justify-between gap-3">
      <div className="flex items-center gap-2">
        <Layers size={18} className="text-violet-600" />
        <h3 className="font-bold text-gray-900">Шари SVG</h3>
      </div>

      <span className="text-xs text-gray-400">{layers.length} шарів</span>
    </div>

    <div className="max-h-[min(72vh,760px)] overflow-auto p-3 space-y-2">
      {layers.map((layer, index) => {
        const selected = selectedLayerId === layer.id

        return (
          <div
            key={layer.id}
            className={`
              flex items-center gap-2 p-2 rounded-xl border transition-colors
              ${selected ? 'border-violet-400 bg-violet-50' : 'border-transparent bg-gray-50'}
              ${layer.visible ? '' : 'opacity-55'}
            `}
          >
            <button
              type="button"
              onClick={() => onToggleVisibility(layer.id)}
              className="w-8 h-8 rounded-lg flex items-center justify-center text-gray-500 hover:bg-white"
              title={layer.visible ? 'Сховати шар' : 'Показати шар'}
            >
              {layer.visible ? <Eye size={17} /> : <EyeOff size={17} />}
            </button>

            <button
              type="button"
              onClick={() => onSelectLayer(selected ? null : layer.id)}
              className={`
                w-7 h-7 rounded-lg border border-gray-300 shrink-0
                ${selected ? 'ring-2 ring-violet-500 ring-offset-2' : ''}
              `}
              style={{ backgroundColor: layer.color }}
              title="Виділити шар на SVG"
            />

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <Pencil size={13} className="text-gray-400 shrink-0" />
                <input
                  value={layer.displayName}
                  onChange={(event) => onRenameLayer(layer.id, event.target.value)}
                  onFocus={() => onSelectLayer(layer.id)}
                  className="w-full bg-transparent text-sm font-semibold text-gray-800 outline-none focus:text-violet-700"
                  aria-label="Назва шару"
                />
              </div>

              <div className="flex flex-wrap items-center gap-2 mt-1">
                <span className="text-xs text-gray-400">{layer.node_count} вузлів</span>

                <span
                  className={`
                    text-[11px] px-2 py-0.5 rounded-full font-mono
                    ${isDarkColor(layer.color) ? 'bg-gray-800 text-white' : 'bg-violet-100 text-violet-700'}
                  `}
                >
                  {layer.originalName}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              <button
                type="button"
                onClick={() => onMoveLayer(layer.id, -1)}
                disabled={index === 0}
                className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-500 hover:bg-white disabled:opacity-30"
                title="Підняти шар"
              >
                <ChevronUp size={15} />
              </button>

              <button
                type="button"
                onClick={() => onMoveLayer(layer.id, 1)}
                disabled={index === layers.length - 1}
                className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-500 hover:bg-white disabled:opacity-30"
                title="Опустити шар"
              >
                <ChevronDown size={15} />
              </button>

              <button
                type="button"
                onClick={() => onDeleteLayer(layer.id)}
                className="w-7 h-7 rounded-lg flex items-center justify-center text-red-500 hover:bg-red-50"
                title="Видалити шар"
              >
                <Trash2 size={15} />
              </button>
            </div>
          </div>
        )
      })}
    </div>
  </div>
)
