import React, { useEffect, useState } from 'react'
import { AlertCircle, BarChart3, ImageIcon, Lightbulb, Loader2, Sparkles, Wand2 } from 'lucide-react'

import { fetchResult, pollStatus, uploadImage } from '../api/vectorize'
import type { AnalysisRun, EditableLayer, LayerInfo, VectorizeMode, VectorizeResult } from '../types'

import { AnalyticsDashboard } from '../components/AnalyticsDashboard'
import { LayerPreview } from '../components/LayerPreview'
import { ModeSelector } from '../components/ModeSelector'
import { SvgViewer } from '../components/SvgViewer'
import { ToleranceSlider } from '../components/ToleranceSlider'
import { UploadZone } from '../components/UploadZone'

interface ImageAnalysis {
  width: number
  height: number
  megapixels: number
  uniqueColorBuckets: number
  avgSaturation: number
  grayStd: number
  midtoneRatio: number
  edgeRatio: number
}

interface ParameterRecommendation {
  mode: VectorizeMode
  tolerance: number
  maxLayers: number
  title: string
  description: string
  reason: string
}

const PRODUCT_NAME = 'Traceva'
const PRODUCT_TAGLINE = 'Gapless SVG Studio for design and CAD'
const ANALYSIS_HISTORY_KEY = 'traceva-analysis-history'

function getModeTitle(mode?: VectorizeMode | null): string {
  if (mode === 'logo') return 'Logo Mode'
  if (mode === 'semantic') return 'Semantic Mode'
  if (mode === 'auto') return 'Auto Mode'
  return 'Невідомо'
}

function getModeDescription(mode?: VectorizeMode | null): string {
  if (mode === 'logo') return 'Логотипи, іконки та line-art графіка'
  if (mode === 'semantic') return 'Фото та складні зображення через семантичну сегментацію'
  if (mode === 'auto') return 'Система автоматично обирає оптимальний режим'
  return 'Режим ще не визначено'
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function calculateSaturation(r: number, g: number, b: number): number {
  const rn = r / 255
  const gn = g / 255
  const bn = b / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)

  if (max === 0) return 0
  return (max - min) / max
}

function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file)
    const image = new Image()

    image.onload = () => {
      URL.revokeObjectURL(objectUrl)
      resolve(image)
    }

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error('Не вдалося проаналізувати зображення'))
    }

    image.src = objectUrl
  })
}

async function analyzeImage(file: File): Promise<ImageAnalysis> {
  const image = await loadImageFromFile(file)
  const maxSide = 160
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height))
  const width = Math.max(1, Math.round(image.width * scale))
  const height = Math.max(1, Math.round(image.height * scale))

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height

  const context = canvas.getContext('2d')
  if (!context) throw new Error('Не вдалося створити canvas для аналізу')

  context.drawImage(image, 0, 0, width, height)

  const data = context.getImageData(0, 0, width, height).data
  const grayValues: number[] = []
  const colorBuckets = new Set<string>()

  let saturationSum = 0
  let graySum = 0
  let graySquaredSum = 0
  let darkPixels = 0
  let lightPixels = 0
  let validPixels = 0

  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3]
    if (alpha < 20) continue

    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    const gray = 0.299 * r + 0.587 * g + 0.114 * b
    const saturation = calculateSaturation(r, g, b)

    colorBuckets.add(`${Math.floor(r / 32)}-${Math.floor(g / 32)}-${Math.floor(b / 32)}`)
    grayValues.push(gray)
    graySum += gray
    graySquaredSum += gray * gray
    saturationSum += saturation

    if (gray < 70) darkPixels += 1
    if (gray > 185) lightPixels += 1
    validPixels += 1
  }

  if (validPixels === 0) {
    return {
      width: image.width,
      height: image.height,
      megapixels: (image.width * image.height) / 1_000_000,
      uniqueColorBuckets: 0,
      avgSaturation: 0,
      grayStd: 0,
      midtoneRatio: 0,
      edgeRatio: 0,
    }
  }

  const grayMean = graySum / validPixels
  const grayVariance = graySquaredSum / validPixels - grayMean * grayMean
  const grayStd = Math.sqrt(Math.max(0, grayVariance))
  const avgSaturation = saturationSum / validPixels
  const midtoneRatio = 1 - (darkPixels + lightPixels) / validPixels

  let edgeCount = 0
  let edgeTotal = 0

  for (let y = 0; y < height - 1; y += 1) {
    for (let x = 0; x < width - 1; x += 1) {
      const current = grayValues[y * width + x]
      const right = grayValues[y * width + x + 1]
      const bottom = grayValues[(y + 1) * width + x]
      if (current === undefined || right === undefined || bottom === undefined) continue

      const diff = Math.max(Math.abs(current - right), Math.abs(current - bottom))
      if (diff > 35) edgeCount += 1
      edgeTotal += 1
    }
  }

  return {
    width: image.width,
    height: image.height,
    megapixels: (image.width * image.height) / 1_000_000,
    uniqueColorBuckets: colorBuckets.size,
    avgSaturation,
    grayStd,
    midtoneRatio,
    edgeRatio: edgeTotal > 0 ? edgeCount / edgeTotal : 0,
  }
}

function buildRecommendation(analysis: ImageAnalysis): ParameterRecommendation {
  const looksLikeLogo =
    analysis.uniqueColorBuckets <= 45 &&
    analysis.midtoneRatio <= 0.36 &&
    analysis.grayStd >= 35 &&
    analysis.edgeRatio >= 0.015

  const looksLikePhoto =
    analysis.uniqueColorBuckets > 55 ||
    analysis.midtoneRatio > 0.38 ||
    analysis.avgSaturation > 0.16

  if (looksLikeLogo && !looksLikePhoto) {
    return {
      mode: 'logo',
      tolerance: 0.6,
      maxLayers: 1,
      title: 'Схоже на логотип або line-art',
      description: 'Рекомендовано Logo Mode',
      reason: 'У зображенні небагато кольорів, тому краще використати чітку контурну векторизацію.',
    }
  }

  if (looksLikePhoto) {
    return {
      mode: 'semantic',
      tolerance: 0.5,
      maxLayers: analysis.edgeRatio > 0.08 ? 84 : 72,
      title: 'Схоже на фото або складне зображення',
      description: 'Рекомендовано Semantic Mode',
      reason: 'Для фото застосовується detail-first gapless SVG pipeline: 72-84 кольорові шари, tolerance 0.5 і перевірка CAD Readiness після рендера.',
    }
  }

  return {
    mode: 'auto',
    tolerance: 0.7,
    maxLayers: 10,
    title: 'Зображення середньої складності',
    description: 'Рекомендовано Auto Mode',
    reason: 'Система може самостійно обрати режим. Почни з помірних параметрів, а потім підкоригуй за результатом.',
  }
}

function hexToRgb(hexColor: string): { r: number; g: number; b: number } {
  const hex = hexColor.replace('#', '')
  if (hex.length !== 6) return { r: 128, g: 128, b: 128 }

  return {
    r: parseInt(hex.slice(0, 2), 16),
    g: parseInt(hex.slice(2, 4), 16),
    b: parseInt(hex.slice(4, 6), 16),
  }
}

function suggestLayerName(layer: LayerInfo, index: number): string {
  const { r, g, b } = hexToRgb(layer.color)
  const luminance = 0.299 * r + 0.587 * g + 0.114 * b
  const saturation = calculateSaturation(r, g, b)

  if (layer.name === 'background') return 'Фон'
  if (layer.name.startsWith('object_')) return `Об'єкт ${index + 1}`
  if (layer.name.startsWith('detail_')) return `Деталь ${index + 1}`
  if (b > r + 25 && b > g + 10 && luminance > 120) return 'Небо'
  if (g > r + 12 && g > b - 8 && saturation > 0.18) return 'Зелень'
  if (luminance < 55) return 'Тіні'
  if (luminance > 205 && saturation < 0.16) return 'Світлі ділянки'
  if (r > g + 16 && r > b + 16 && saturation > 0.16) return 'Теплі тони'
  if (saturation < 0.12) return 'Нейтральні тони'

  return `Шар ${index + 1}`
}

function createEditableLayers(layers: LayerInfo[]): EditableLayer[] {
  const nameCounts = new Map<string, number>()

  return layers.map((layer, index) => {
    const baseName = suggestLayerName(layer, index)
    const count = (nameCounts.get(baseName) || 0) + 1
    nameCounts.set(baseName, count)

    return {
      ...layer,
      id: `layer-${index + 1}`,
      originalName: layer.name,
      displayName: count > 1 ? `${baseName} ${count}` : baseName,
      visible: true,
    }
  })
}

function loadAnalysisHistory(): AnalysisRun[] {
  try {
    const rawHistory = window.localStorage.getItem(ANALYSIS_HISTORY_KEY)
    if (!rawHistory) return []
    const parsedHistory = JSON.parse(rawHistory)
    return Array.isArray(parsedHistory) ? parsedHistory.slice(-100) : []
  } catch {
    return []
  }
}

function saveAnalysisHistory(history: AnalysisRun[]) {
  window.localStorage.setItem(ANALYSIS_HISTORY_KEY, JSON.stringify(history.slice(-100)))
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [tolerance, setTolerance] = useState(0.7)
  const [maxLayers, setMaxLayers] = useState(10)
  const [simplify] = useState(true)
  const [mode, setMode] = useState<VectorizeMode>('auto')
  const [recommendation, setRecommendation] = useState<ParameterRecommendation | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [result, setResult] = useState<VectorizeResult | null>(null)
  const [editableLayers, setEditableLayers] = useState<EditableLayer[]>([])
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null)
  const [activeResultTab, setActiveResultTab] = useState<'editor' | 'analytics'>('editor')
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisRun[]>(() => loadAnalysisHistory())
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const applyRecommendation = (rec: ParameterRecommendation) => {
    setMode(rec.mode)
    setTolerance(clamp(rec.tolerance, 0.5, 5))
    setMaxLayers(clamp(rec.maxLayers, 1, 96))
  }

  const resetResultState = () => {
    setTaskId(null)
    setResult(null)
    setEditableLayers([])
    setSelectedLayerId(null)
    setActiveResultTab('editor')
    setProgress(0)
    setError(null)
  }

  const handleFileSelect = async (file: File) => {
    setSelectedFile(file)
    setPreviewUrl((currentPreviewUrl) => {
      if (currentPreviewUrl) URL.revokeObjectURL(currentPreviewUrl)
      return URL.createObjectURL(file)
    })
    resetResultState()
    setRecommendation(null)

    try {
      setIsAnalyzing(true)
      const analysis = await analyzeImage(file)
      const rec = buildRecommendation(analysis)
      setRecommendation(rec)
      applyRecommendation(rec)
    } catch (err) {
      console.error(err)
      const fallbackRecommendation: ParameterRecommendation = {
        mode: 'auto',
        tolerance: 0.8,
        maxLayers: 10,
        title: 'Не вдалося точно проаналізувати зображення',
        description: 'Рекомендовано Auto Mode',
        reason: 'Можеш почати з базових параметрів: Tolerance 0.7 і Max Layers 10.',
      }

      setRecommendation(fallbackRecommendation)
      applyRecommendation(fallbackRecommendation)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleVectorize = async () => {
    if (!selectedFile) {
      setError('Спочатку обери зображення')
      return
    }

    try {
      setIsProcessing(true)
      setError(null)
      setProgress(0)
      setResult(null)
      setEditableLayers([])
      setSelectedLayerId(null)
      setActiveResultTab('editor')

      const initialResult = await uploadImage(selectedFile, tolerance, maxLayers, simplify, mode)
      setTaskId(initialResult.task_id)

      let currentProgress = 0
      let isDone = false

      while (!isDone) {
        await new Promise((resolve) => setTimeout(resolve, 800))
        const status = await pollStatus(initialResult.task_id)

        currentProgress = status.progress ?? currentProgress
        setProgress(currentProgress)

        if (status.status === 'done') {
          isDone = true
          const finalResult = await fetchResult(initialResult.task_id)
          const finalLayers = createEditableLayers(finalResult.layers || [])

          setResult(finalResult)
          setEditableLayers(finalLayers)
          setSelectedLayerId(null)
          setActiveResultTab('editor')
          setProgress(100)

          setAnalysisHistory((currentHistory) => {
            const nextHistory = [
              ...currentHistory,
              {
                id: finalResult.task_id,
                createdAt: new Date().toISOString(),
                requestedMode: finalResult.requested_mode,
                mode: finalResult.mode,
                metrics: finalResult.metrics,
                layerCount: finalLayers.length,
              },
            ].slice(-100)

            saveAnalysisHistory(nextHistory)
            return nextHistory
          })
        }

        if (status.status === 'error') {
          throw new Error(status.error || 'Не вдалося виконати векторизацію')
        }
      }
    } catch (err) {
      console.error(err)
      setError(err instanceof Error && err.message ? err.message : 'Не вдалося векторизувати зображення')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleToggleLayerVisibility = (id: string) => {
    setEditableLayers((currentLayers) =>
      currentLayers.map((layer) =>
        layer.id === id ? { ...layer, visible: !layer.visible } : layer
      )
    )
  }

  const handleRenameLayer = (id: string, displayName: string) => {
    setEditableLayers((currentLayers) =>
      currentLayers.map((layer) =>
        layer.id === id ? { ...layer, displayName } : layer
      )
    )
  }

  const handleDeleteLayer = (id: string) => {
    setEditableLayers((currentLayers) => currentLayers.filter((layer) => layer.id !== id))
    setSelectedLayerId((currentId) => (currentId === id ? null : currentId))
  }

  const handleMoveLayer = (id: string, direction: -1 | 1) => {
    setEditableLayers((currentLayers) => {
      const index = currentLayers.findIndex((layer) => layer.id === id)
      const nextIndex = index + direction
      if (index < 0 || nextIndex < 0 || nextIndex >= currentLayers.length) return currentLayers

      const nextLayers = [...currentLayers]
      const [layer] = nextLayers.splice(index, 1)
      nextLayers.splice(nextIndex, 0, layer)
      return nextLayers
    })
  }

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const requestedMode = result?.requested_mode
  const usedMode = result?.mode

  return (
    <div className="min-h-screen bg-[#f5f3ff]">
      <header className="bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6 py-5 flex items-center gap-3">
          <div className="w-11 h-11 rounded-2xl bg-violet-600 flex items-center justify-center text-white">
            <Sparkles size={22} />
          </div>

          <div>
            <h1 className="text-xl font-bold text-gray-900">{PRODUCT_NAME}</h1>
            <p className="text-sm text-gray-500">{PRODUCT_TAGLINE}</p>
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-6 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 items-start">
          <section className="space-y-6">
            <UploadZone onFile={handleFileSelect} disabled={isProcessing} />

            <div className="bg-white rounded-2xl border border-gray-200 p-5 shadow-sm space-y-5">
              <ModeSelector value={mode} onChange={setMode} disabled={isProcessing} />

              {selectedFile && (
                <div className="rounded-2xl border border-violet-100 bg-violet-50 p-4">
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-xl bg-violet-100 text-violet-700 flex items-center justify-center shrink-0">
                      {isAnalyzing ? <Loader2 size={18} className="animate-spin" /> : <Lightbulb size={18} />}
                    </div>

                    <div className="min-w-0">
                      <p className="text-xs font-bold text-violet-500 uppercase tracking-wide">
                        Рекомендовані параметри
                      </p>

                      <h3 className="text-sm font-bold text-gray-900 mt-1">
                        {isAnalyzing
                          ? 'Аналізуємо зображення...'
                          : recommendation?.title || 'Параметри підібрано'}
                      </h3>

                      {recommendation && !isAnalyzing && (
                        <>
                          <p className="text-sm text-gray-600 mt-1">{recommendation.description}</p>

                          <div className="flex flex-wrap gap-2 mt-3">
                            <span className="px-3 py-1 rounded-full bg-white text-violet-700 text-xs font-bold border border-violet-100">
                              Mode: {recommendation.mode}
                            </span>
                            <span className="px-3 py-1 rounded-full bg-white text-violet-700 text-xs font-bold border border-violet-100">
                              Tolerance: {recommendation.tolerance}
                            </span>
                            <span className="px-3 py-1 rounded-full bg-white text-violet-700 text-xs font-bold border border-violet-100">
                              Max Layers: {recommendation.maxLayers}
                            </span>
                          </div>

                          <p className="text-xs text-gray-500 leading-relaxed mt-3">{recommendation.reason}</p>

                          <button
                            type="button"
                            onClick={() => applyRecommendation(recommendation)}
                            disabled={isProcessing}
                            className="mt-3 text-xs font-bold text-violet-700 hover:text-violet-900 disabled:text-gray-400"
                          >
                            Застосувати рекомендацію ще раз
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}

              <ToleranceSlider
                tolerance={tolerance}
                onChange={setTolerance}
                maxLayers={maxLayers}
                onMaxLayersChange={setMaxLayers}
              />

              <button
                type="button"
                onClick={handleVectorize}
                disabled={!selectedFile || isProcessing || isAnalyzing}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-violet-600 hover:bg-violet-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-semibold transition-colors"
              >
                {isProcessing ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    Векторизуємо... {progress > 0 ? `${progress}%` : ''}
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    Векторизувати зображення
                  </>
                )}
              </button>
            </div>

            {false && result && previewUrl && (
              <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden shadow-sm">
                <div className="px-5 py-3 border-b border-gray-100">
                  <h2 className="text-sm font-bold text-gray-500 uppercase">Оригінал</h2>
                </div>

                <div className="p-6 bg-white flex items-center justify-center min-h-[260px]">
                  <img
                    src={previewUrl || ''}
                    alt="Оригінальне зображення"
                    className="max-w-full max-h-[360px] object-contain"
                  />
                </div>
              </div>
            )}
          </section>

          <section className="space-y-6 min-w-0">
            {error && (
              <div className="flex items-center gap-3 p-4 rounded-2xl border border-red-200 bg-red-50 text-red-600">
                <AlertCircle size={18} />
                <p className="text-sm font-medium">{error}</p>
              </div>
            )}

            {previewUrl && (
              <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden shadow-sm">
                <div className="px-5 py-4 border-b border-gray-100">
                  <h2 className="font-bold text-gray-900">Оригінал фото</h2>
                  <p className="text-sm text-gray-500 mt-1">
                    Порівнюй параметри з реальним зображенням перед векторизацією
                  </p>
                </div>

                <div className="p-6 bg-white flex items-center justify-center min-h-[520px]">
                  <img
                    src={previewUrl}
                    alt="Оригінальне зображення"
                    className="max-w-full max-h-[70vh] object-contain rounded-xl"
                  />
                </div>
              </div>
            )}

            {!previewUrl && !isProcessing && (
              <div className="bg-white/70 rounded-2xl border border-dashed border-violet-200 min-h-[420px] flex items-center justify-center text-center p-8">
                <div>
                  <div className="w-14 h-14 mx-auto rounded-2xl bg-violet-100 text-violet-600 flex items-center justify-center mb-4">
                    <Sparkles size={24} />
                  </div>
                  <h2 className="text-lg font-bold text-gray-800">Результат з'явиться тут</h2>
                  <p className="text-sm text-gray-500 mt-2 max-w-sm">
                    Обери зображення, налаштуй параметри й натисни кнопку “Векторизувати зображення”.
                  </p>
                </div>
              </div>
            )}

            {isProcessing && (
              <div className="bg-white rounded-2xl border border-gray-200 p-8 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <Loader2 size={22} className="animate-spin text-violet-600" />
                  <div>
                    <h2 className="font-bold text-gray-900">Виконується векторизація</h2>
                    <p className="text-sm text-gray-500">Обробляємо зображення та будуємо SVG-шари.</p>
                  </div>
                </div>

                <div className="w-full h-3 bg-gray-100 rounded-full overflow-hidden">
                  <div className="h-full bg-violet-600 transition-all" style={{ width: `${progress}%` }} />
                </div>

                <p className="text-sm text-gray-500 mt-3">{progress}%</p>
              </div>
            )}

            {false && result && taskId && (
              <>
                <div className="bg-white rounded-2xl border border-gray-200 p-5 shadow-sm">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 rounded-xl bg-violet-100 text-violet-700 flex items-center justify-center shrink-0">
                        <Wand2 size={19} />
                      </div>

                      <div>
                        <p className="text-xs font-bold text-gray-400 uppercase tracking-wide">Режим обробки</p>
                        <h2 className="text-base font-bold text-gray-900 mt-1">
                          {requestedMode === 'auto' && usedMode
                            ? `Auto -> ${getModeTitle(usedMode)}`
                            : getModeTitle(usedMode || requestedMode)}
                        </h2>
                        <p className="text-sm text-gray-500 mt-1">
                          {requestedMode === 'auto' && usedMode
                            ? `Система автоматично визначила: ${getModeDescription(usedMode)}.`
                            : getModeDescription(usedMode || requestedMode)}
                        </p>
                      </div>
                    </div>

                    <div className="flex flex-col items-end gap-2">
                      {requestedMode && (
                        <span className="px-3 py-1 rounded-full bg-gray-100 text-gray-600 text-xs font-bold uppercase">
                          Обрано: {requestedMode}
                        </span>
                      )}
                      {usedMode && (
                        <span className="px-3 py-1 rounded-full bg-violet-100 text-violet-700 text-xs font-bold uppercase">
                          Використано: {usedMode}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-2xl border border-gray-200 p-2 shadow-sm">
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setActiveResultTab('editor')}
                      className={`
                        flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-bold transition-colors
                        ${activeResultTab === 'editor' ? 'bg-violet-600 text-white' : 'text-gray-500 hover:bg-gray-50'}
                      `}
                    >
                      <ImageIcon size={17} />
                      SVG редактор
                    </button>

                    <button
                      type="button"
                      onClick={() => setActiveResultTab('analytics')}
                      className={`
                        flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-bold transition-colors
                        ${activeResultTab === 'analytics' ? 'bg-violet-600 text-white' : 'text-gray-500 hover:bg-gray-50'}
                      `}
                    >
                      <BarChart3 size={17} />
                      Аналітика
                    </button>
                  </div>
                </div>

                {activeResultTab === 'editor' && result?.svg_url && (
                  <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-5 items-start">
                    {editableLayers.length > 0 && (
                      <LayerPreview
                        layers={editableLayers}
                        selectedLayerId={selectedLayerId}
                        onSelectLayer={setSelectedLayerId}
                        onToggleVisibility={handleToggleLayerVisibility}
                        onRenameLayer={handleRenameLayer}
                        onDeleteLayer={handleDeleteLayer}
                        onMoveLayer={handleMoveLayer}
                      />
                    )}

                    <SvgViewer
                      svgUrl={result?.svg_url || ''}
                      taskId={taskId || ''}
                      layers={editableLayers}
                      selectedLayerId={selectedLayerId}
                    />
                  </div>
                )}

                {activeResultTab === 'analytics' && (
                  <AnalyticsDashboard metrics={result?.metrics} history={analysisHistory} />
                )}
              </>
            )}
          </section>
        </div>

        {result && taskId && (
          <section className="mt-8 space-y-6">
            <div className="bg-white rounded-2xl border border-gray-200 p-5 shadow-sm">
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-xl bg-violet-100 text-violet-700 flex items-center justify-center shrink-0">
                    <Wand2 size={19} />
                  </div>

                  <div>
                    <p className="text-xs font-bold text-gray-400 uppercase tracking-wide">Режим обробки</p>
                    <h2 className="text-base font-bold text-gray-900 mt-1">
                      {requestedMode === 'auto' && usedMode
                        ? `Auto -> ${getModeTitle(usedMode)}`
                        : getModeTitle(usedMode || requestedMode)}
                    </h2>
                    <p className="text-sm text-gray-500 mt-1">
                      {requestedMode === 'auto' && usedMode
                        ? `Система автоматично визначила: ${getModeDescription(usedMode)}.`
                        : getModeDescription(usedMode || requestedMode)}
                    </p>
                  </div>
                </div>

                <div className="flex flex-col items-end gap-2">
                  {requestedMode && (
                    <span className="px-3 py-1 rounded-full bg-gray-100 text-gray-600 text-xs font-bold uppercase">
                      Обрано: {requestedMode}
                    </span>
                  )}
                  {usedMode && (
                    <span className="px-3 py-1 rounded-full bg-violet-100 text-violet-700 text-xs font-bold uppercase">
                      Використано: {usedMode}
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl border border-gray-200 p-2 shadow-sm">
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setActiveResultTab('editor')}
                  className={`
                    flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-bold transition-colors
                    ${activeResultTab === 'editor' ? 'bg-violet-600 text-white' : 'text-gray-500 hover:bg-gray-50'}
                  `}
                >
                  <ImageIcon size={17} />
                  SVG редактор
                </button>

                <button
                  type="button"
                  onClick={() => setActiveResultTab('analytics')}
                  className={`
                    flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-bold transition-colors
                    ${activeResultTab === 'analytics' ? 'bg-violet-600 text-white' : 'text-gray-500 hover:bg-gray-50'}
                  `}
                >
                  <BarChart3 size={17} />
                  Аналітика
                </button>
              </div>
            </div>

            {activeResultTab === 'editor' && result.svg_url && (
              <div className="grid grid-cols-1 lg:grid-cols-[420px_minmax(0,1fr)] gap-5 items-start">
                {editableLayers.length > 0 && (
                  <LayerPreview
                    layers={editableLayers}
                    selectedLayerId={selectedLayerId}
                    onSelectLayer={setSelectedLayerId}
                    onToggleVisibility={handleToggleLayerVisibility}
                    onRenameLayer={handleRenameLayer}
                    onDeleteLayer={handleDeleteLayer}
                    onMoveLayer={handleMoveLayer}
                  />
                )}

                <div className="min-w-0">
                  <SvgViewer
                    svgUrl={result.svg_url}
                    taskId={taskId}
                    layers={editableLayers}
                    selectedLayerId={selectedLayerId}
                  />
                </div>
              </div>
            )}

            {activeResultTab === 'analytics' && (
              <AnalyticsDashboard metrics={result.metrics} history={analysisHistory} />
            )}
          </section>
        )}
      </main>
    </div>
  )
}
