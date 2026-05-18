import axios from 'axios'
import { VectorizeResult, TaskStatusResponse, VectorizeMode } from '../types'

const BASE = 'http://localhost:8000/api'

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