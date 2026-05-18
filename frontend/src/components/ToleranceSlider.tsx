import React from 'react'

interface Props {
  tolerance: number
  onChange: (value: number) => void
  maxLayers: number
  onMaxLayersChange: (value: number) => void
}

export const ToleranceSlider: React.FC<Props> = ({
  tolerance,
  onChange,
  maxLayers,
  onMaxLayersChange,
}) => (
  <div className="space-y-4 p-4 bg-white rounded-xl border border-gray-200">
    <h3 className="font-semibold text-gray-800 text-sm uppercase tracking-wide">
      Параметри
    </h3>

    <div>
      <div className="flex justify-between text-sm text-gray-600 mb-1">
        <span>Точність (Tolerance)</span>
        <span className="font-mono font-bold text-violet-600">{tolerance.toFixed(1)}</span>
      </div>

      <input
        type="range"
        min={0.5}
        max={5}
        step={0.1}
        value={tolerance}
        onChange={(event) => onChange(parseFloat(event.target.value))}
        className="w-full accent-violet-600"
      />

      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>Точно (більше вузлів)</span>
        <span>Грубо (менше вузлів)</span>
      </div>
    </div>

    <div>
      <div className="flex justify-between text-sm text-gray-600 mb-1">
        <span>Макс. шарів</span>
        <span className="font-mono font-bold text-violet-600">{maxLayers}</span>
      </div>

      <input
        type="range"
        min={1}
        max={96}
        step={1}
        value={maxLayers}
        onChange={(event) => onMaxLayersChange(parseInt(event.target.value))}
        className="w-full accent-violet-600"
      />
    </div>
  </div>
)
