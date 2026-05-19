import axios from 'axios'
import {
  EditableMaskLayer,
  MaskPreviewResult,
  TaskStatusResponse,
  VectorizeMode,
  VectorizeResult,
} from '../types'

const BASE = 'http://localhost:8000/api'

export interface VlmStatus {
  enabled: boolean
  model: string
  device: string
  ready: boolean
  load_failed: boolean
}

export async function fetchVlmStatus(): Promise<VlmStatus> {
  const response = await fetch(`${BASE}/vlm/status`)
  if (!response.ok) throw new Error('VLM status request failed')
  return response.json()
}

export async function uploadImage(
  file: File,
  tolerance: number,
  maxLayers: number,
  simplify: boolean,
  mode: VectorizeMode
): Promise<VectorizeResult> {
  const form = new FormData()

  form.append('file', file)
  form.append('tolerance', String(tolerance))
  form.append('max_layers', String(maxLayers))
  form.append('simplify', String(simplify))
  form.append('mode', mode)

  const { data } = await axios.post<VectorizeResult>(`${BASE}/vectorize`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })

  return data
}

export async function pollStatus(taskId: string): Promise<TaskStatusResponse> {
  const { data } = await axios.get<TaskStatusResponse>(`${BASE}/status/${taskId}`)
  return data
}

export async function fetchResult(taskId: string): Promise<VectorizeResult> {
  const { data } = await axios.get<VectorizeResult>(`${BASE}/result/${taskId}`)
  return data
}

export function getSvgDownloadUrl(taskId: string): string {
  return `${BASE}/download/${taskId}`
}

export async function analyzeImage(file: File) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await axios.post('/api/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return data  // { mode, tolerance, max_layers, reason, tips }
}

export async function prepareMaskPreview(
  file: File,
  maxLayers: number,
  mode: VectorizeMode
): Promise<MaskPreviewResult> {
  const form = new FormData()

  form.append('file', file)
  form.append('max_layers', String(maxLayers))
  form.append('mode', mode)

  const { data } = await axios.post<MaskPreviewResult>(`${BASE}/vectorize/prepare`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })

  return data
}

export async function splitPreviewMask(taskId: string, maskId: string): Promise<MaskPreviewResult> {
  const { data } = await axios.post<MaskPreviewResult>(`${BASE}/preview/${taskId}/split/${maskId}`)
  return data
}

export async function finalizeMaskPreview(
  taskId: string,
  tolerance: number,
  simplify: boolean,
  layers: EditableMaskLayer[]
): Promise<VectorizeResult> {
  const { data } = await axios.post<VectorizeResult>(`${BASE}/vectorize/finalize/${taskId}`, {
    tolerance,
    simplify,
    layers: layers
      .filter((layer) => layer.visible && layer.sourceMaskIds.length > 0)
      .map((layer) => ({
        id: layer.id,
        name: layer.name,
        color: layer.color,
        source_mask_ids: layer.sourceMaskIds,
      })),
  })

  return data
}

export function getPreviewImageUrl(taskId: string): string {
  return `/api/preview/${taskId}/image`
}

export function getPreviewMaskUrl(taskId: string, maskId: string): string {
  return `/api/preview/${taskId}/mask/${maskId}`
}

export function getPreviewMaskOverlayUrl(taskId: string, maskId: string): string {
  return `/api/preview/${taskId}/mask/${maskId}/overlay`
}
