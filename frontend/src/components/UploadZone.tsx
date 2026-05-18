import React, { useCallback, useState } from 'react'
import { Upload, ImageIcon } from 'lucide-react'

interface Props {
  onFile: (file: File) => void
  disabled?: boolean
}

export const UploadZone: React.FC<Props> = ({ onFile, disabled }) => {
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setDragging(false)

    const file = event.dataTransfer.files[0]
    if (file) onFile(file)
  }, [onFile])

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) onFile(file)
  }

  return (
    <label
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`
        flex flex-col items-center justify-center gap-3
        w-full h-56 rounded-2xl border-2 border-dashed cursor-pointer
        transition-all duration-200
        ${dragging ? 'border-violet-500 bg-violet-50' : 'border-gray-300 bg-gray-50 hover:border-violet-400'}
        ${disabled ? 'opacity-50 pointer-events-none' : ''}
      `}
    >
      <div className="p-4 bg-violet-100 rounded-full">
        {dragging
          ? <ImageIcon size={32} className="text-violet-600" />
          : <Upload size={32} className="text-violet-500" />}
      </div>

      <div className="text-center">
        <p className="font-semibold text-gray-700">Перетягни зображення сюди</p>
        <p className="text-sm text-gray-400 mt-1">або натисни для вибору · PNG, JPG, WebP · до 2048px</p>
      </div>

      <input
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={handleChange}
        disabled={disabled}
      />
    </label>
  )
}
