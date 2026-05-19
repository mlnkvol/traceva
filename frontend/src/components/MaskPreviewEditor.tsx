import React from 'react'
import {
  Check,
  ChevronDown,
  ChevronUp,
  Eye,
  EyeOff,
  GitMerge,
  Scissors,
  Trash2,
} from 'lucide-react'
import { EditableMaskLayer } from '../types'
import { getPreviewImageUrl, getPreviewMaskOverlayUrl } from '../api/vectorize'

interface Props {
  taskId: string
  layers: EditableMaskLayer[]
  selectedLayerIds: string[]
  onSelectLayer: (id: string, multi?: boolean) => void
  onRenameLayer: (id: string, name: string) => void
  onToggleLayer: (id: string) => void
  onDeleteLayer: (id: string) => void
  onMoveLayer: (id: string, direction: -1 | 1) => void
  onMergeSelected: () => void
  onSplitSelected: () => void
  onConfirm: () => void
  disabled?: boolean
}

export const MaskPreviewEditor: React.FC<Props> = ({
  taskId,
  layers,
  selectedLayerIds,
  onSelectLayer,
  onRenameLayer,
  onToggleLayer,
  onDeleteLayer,
  onMoveLayer,
  onMergeSelected,
  onSplitSelected,
  onConfirm,
  disabled,
}) => {
  const selectedLayers = layers.filter((layer) => selectedLayerIds.includes(layer.id))
  const canMerge = selectedLayers.length >= 2
  const canSplit = selectedLayers.length === 1 && selectedLayers[0].sourceMaskIds.length === 1

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[420px_minmax(0,1fr)] gap-5 items-start">
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-4 py-4 border-b border-gray-100 flex items-center justify-between gap-3">
          <div>
            <h3 className="font-bold text-gray-900">Етап 1: перевірка масок</h3>
            <p className="text-xs text-gray-500 mt-1">{layers.length} масок. SVG ще не створено.</p>
          </div>

          <button
            type="button"
            onClick={onConfirm}
            disabled={disabled || layers.filter((layer) => layer.visible).length === 0}
            className="h-9 px-3 rounded-lg bg-violet-600 text-white text-sm font-bold flex items-center gap-2 disabled:bg-gray-300"
          >
            <Check size={16} />
            Створити SVG
          </button>
        </div>

        <div className="p-3 border-b border-gray-100 grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={onMergeSelected}
            disabled={disabled || !canMerge}
            className="h-9 rounded-lg border border-gray-200 text-sm font-semibold text-gray-700 flex items-center justify-center gap-2 hover:bg-gray-50 disabled:opacity-40"
          >
            <GitMerge size={15} />
            Об’єднати
          </button>

          <button
            type="button"
            onClick={onSplitSelected}
            disabled={disabled || !canSplit}
            className="h-9 rounded-lg border border-gray-200 text-sm font-semibold text-gray-700 flex items-center justify-center gap-2 hover:bg-gray-50 disabled:opacity-40"
          >
            <Scissors size={15} />
            Розділити
          </button>
        </div>

        <div className="max-h-[min(72vh,760px)] overflow-auto p-3 space-y-2">
          {layers.map((layer, index) => {
            const selected = selectedLayerIds.includes(layer.id)

            return (
              <div
                key={layer.id}
                className={`p-2 rounded-xl border transition-colors ${
                  selected ? 'border-violet-400 bg-violet-50' : 'border-transparent bg-gray-50'
                } ${layer.visible ? '' : 'opacity-55'}`}
              >
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onToggleLayer(layer.id)}
                    className="w-8 h-8 rounded-lg flex items-center justify-center text-gray-500 hover:bg-white"
                    title={layer.visible ? 'Сховати шар' : 'Показати шар'}
                  >
                    {layer.visible ? <Eye size={17} /> : <EyeOff size={17} />}
                  </button>

                  <button
                    type="button"
                    onClick={(event) => onSelectLayer(layer.id, event.shiftKey || event.metaKey || event.ctrlKey)}
                    className={`w-7 h-7 rounded-lg border border-gray-300 shrink-0 ${
                      selected ? 'ring-2 ring-violet-500 ring-offset-2' : ''
                    }`}
                    style={{ backgroundColor: layer.color }}
                    title="Виділити маску"
                  />

                  <input
                    value={layer.name}
                    onChange={(event) => onRenameLayer(layer.id, event.target.value)}
                    onFocus={() => onSelectLayer(layer.id)}
                    className="min-w-0 flex-1 bg-transparent text-sm font-semibold text-gray-800 outline-none focus:text-violet-700"
                    aria-label="Назва маски"
                  />

                  <button
                    type="button"
                    onClick={() => onMoveLayer(layer.id, -1)}
                    disabled={index === 0}
                    className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-500 hover:bg-white disabled:opacity-30"
                    title="Підняти в Z-стеку"
                  >
                    <ChevronUp size={15} />
                  </button>

                  <button
                    type="button"
                    onClick={() => onMoveLayer(layer.id, 1)}
                    disabled={index === layers.length - 1}
                    className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-500 hover:bg-white disabled:opacity-30"
                    title="Опустити в Z-стеку"
                  >
                    <ChevronDown size={15} />
                  </button>

                  <button
                    type="button"
                    onClick={() => onDeleteLayer(layer.id)}
                    className="w-7 h-7 rounded-lg flex items-center justify-center text-red-500 hover:bg-red-50"
                    title="Видалити"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>

                <div className="pl-[74px] mt-1 text-[11px] text-gray-400">
                  {layer.sourceMaskIds.length} source mask{layer.sourceMaskIds.length === 1 ? '' : 's'}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-100">
          <h2 className="font-bold text-gray-900">Маски перед векторизацією</h2>
          <p className="text-sm text-gray-500 mt-1">Клікни шар у списку, щоб підсвітити його на фото. Shift/Ctrl додає кілька шарів до вибору.</p>
        </div>

        <div className="p-5 bg-gray-50 flex items-center justify-center min-h-[520px] overflow-auto">
          <div className="relative inline-block max-w-full">
            <img
              src={getPreviewImageUrl(taskId)}
              alt="Preview source"
              className="block max-w-full max-h-[72vh] object-contain"
            />

            {layers.map((layer) =>
              layer.visible && selectedLayerIds.includes(layer.id)
                ? layer.sourceMaskIds.map((maskId) => (
                    <img
                      key={`${layer.id}-${maskId}`}
                      src={getPreviewMaskOverlayUrl(taskId, maskId)}
                      alt=""
                      className="absolute inset-0 w-full h-full pointer-events-none object-fill"
                    />
                  ))
                : null
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
