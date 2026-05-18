export type TaskStatus = 'pending' | 'processing' | 'done' | 'error'

export type VectorizeMode = 'auto' | 'logo' | 'semantic'

export interface LayerInfo {
  name: string
  node_count: number
  color: string
}

export interface EditableLayer extends LayerInfo {
  id: string
  originalName: string
  displayName: string
  visible: boolean
}

export interface Metrics {
  psnr?: number | null
  ssim?: number | null
  node_count?: number | null
  layer_count?: number | null
  path_count?: number | null
  file_size?: number | null
  editability_score?: number | null
  iou?: number | null
  processing_time?: number | null
  gapless_coverage?: number | null
  gap_pixels?: number | null
  cad_readiness_score?: number | null
}

export interface AnalysisRun {
  id: string
  createdAt: string
  mode?: VectorizeMode
  requestedMode?: VectorizeMode
  metrics?: Metrics
  layerCount: number
}

export interface VectorizeResult {
  task_id: string
  status: TaskStatus
  svg_url?: string
  layers?: LayerInfo[]
  metrics?: Metrics
  requested_mode?: VectorizeMode
  mode?: VectorizeMode
  error?: string
}

export interface TaskStatusResponse {
  task_id: string
  status: TaskStatus
  progress: number
  requested_mode?: VectorizeMode
  mode?: VectorizeMode
  error?: string
}
