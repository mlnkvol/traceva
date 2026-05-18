import React from 'react'
import { Image, Sparkles, Wand2 } from 'lucide-react'
import type { VectorizeMode } from '../types'

interface Props {
  value: VectorizeMode
  onChange: (mode: VectorizeMode) => void
  disabled?: boolean
}

const modes: {
  value: VectorizeMode
  title: string
  description: string
  icon: React.ReactNode
}[] = [
  {
    value: 'auto',
    title: 'Auto',
    description: 'Система сама обере режим',
    icon: <Wand2 size={18} />,
  },
  {
    value: 'logo',
    title: 'Logo',
    description: 'Логотипи, іконки, line-art',
    icon: <Sparkles size={18} />,
  },
  {
    value: 'semantic',
    title: 'Semantic',
    description: 'Фото та складні зображення',
    icon: <Image size={18} />,
  },
]

export const ModeSelector: React.FC<Props> = ({ value, onChange, disabled }) => (
  <div className="space-y-3">
    <h3 className="font-semibold text-gray-800 text-sm uppercase tracking-wide">
      Режим векторизації
    </h3>

    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
      {modes.map((mode) => {
        const active = value === mode.value

        return (
          <button
            key={mode.value}
            type="button"
            disabled={disabled}
            onClick={() => onChange(mode.value)}
            className={`
              text-left p-3 rounded-xl border transition-all
              disabled:opacity-50 disabled:cursor-not-allowed
              ${
                active
                  ? 'border-violet-500 bg-violet-50 text-violet-700 shadow-sm'
                  : 'border-gray-200 bg-white text-gray-600 hover:border-violet-300 hover:bg-violet-50/50'
              }
            `}
          >
            <div className="flex items-center gap-2 font-semibold text-sm">
              {mode.icon}
              {mode.title}
            </div>

            <p className="text-xs mt-1 text-gray-400">
              {mode.description}
            </p>
          </button>
        )
      })}
    </div>
  </div>
)
