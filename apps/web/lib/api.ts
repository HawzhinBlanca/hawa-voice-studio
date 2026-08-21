/**
 * Hawa Sorani Voice Studio API Client.
 * 
 * All methods call the live FastAPI backend. No mock fallbacks.
 * If the backend is unreachable, errors propagate to the UI layer
 * for proper loading/error state handling.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ==========================================
// Types (from OpenAPI schema)
// ==========================================

export interface Speaker {
  speaker_id: string;
  name: string;
  kurdish_name: string;
  dialect: string;
  gender: string;
  status: string;
  age_bracket: string;
  voice_description: string;
  naturalness_score: number;
  similarity_score: number;
  pronunciation_score: number;
  consent?: {
    consent_type: string;
    commercial_use_permitted: boolean;
    derivative_model_permitted: boolean;
  };
  references: Array<{
    reference_id: string;
    style_name: string;
    duration_seconds: number;
    exact_transcript_raw: string;
  }>;
  styles: Array<{
    style_id: string;
    name: string;
    instruction_prompt: string;
    recommended_speed: number;
  }>;
}

export interface Dataset {
  dataset_id: string;
  name: string;
  description: string;
  source: string;
  license: string;
  total_hours: number;
  approved_hours: number;
  utterance_count: number;
  is_frozen: boolean;
  current_version?: string;
}

export interface TrainingRun {
  run_id: string;
  run_name: string;
  preset: string;
  base_model: string;
  dataset_version: string;
  status: string;
  current_step: number;
  total_steps: number;
  current_loss: number;
  best_loss: number;
  gpu_type: string;
  estimated_cost_spent: number;
}

export interface EvaluationRun {
  evaluation_id: string;
  title: string;
  challenger_model_id: string;
  avg_naturalness: number;
  avg_pronunciation: number;
  avg_similarity: number;
  avg_cer: number;
  win_rate_vs_baseline: number;
  is_approved_for_production: boolean;
}

export interface Deployment {
  deployment_id: string;
  model_name: string;
  state: string;
  traffic_percentage: number;
  p95_latency_ms: number;
  rtf_score: number;
  active_adapters: string[];
}

export interface DashboardStats {
  speakers: number;
  datasets: number;
  trainingRuns: number;
  deployments: number;
}

// ==========================================
// Error Handling
// ==========================================

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || body.message || res.statusText);
  }
  return await res.json();
}

// ==========================================
// API Client
// ==========================================

export const api = {
  // Speakers
  async getSpeakers(): Promise<Speaker[]> {
    const res = await fetch(`${API_BASE}/v1/speakers`, { cache: 'no-store' });
    return handleResponse<Speaker[]>(res);
  },

  async createSpeaker(data: {
    name: string;
    kurdish_name: string;
    dialect: string;
    gender: string;
    age_bracket?: string;
    voice_description?: string;
  }): Promise<Speaker> {
    const res = await fetch(`${API_BASE}/v1/speakers`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return handleResponse<Speaker>(res);
  },

  async revokeSpeaker(speakerId: string, reason: string = 'Owner requested revocation'): Promise<void> {
    const res = await fetch(`${API_BASE}/v1/speakers/${speakerId}/revoke?reason=${encodeURIComponent(reason)}`, {
      method: 'POST',
    });
    await handleResponse(res);
  },

  // Speech Synthesis
  async synthesizeSpeech(payload: {
    input: string;
    voice: string;
    style?: string;
    speed?: number;
    stream?: boolean;
    watermark_enabled?: boolean;
  }): Promise<Blob> {
    const res = await fetch(`${API_BASE}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, format: 'wav', stream: false }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Speech synthesis failed' }));
      throw new ApiError(res.status, err.detail || 'Speech synthesis failed');
    }
    return await res.blob();
  },

  // Kurdish Normalization
  async normalizeText(text: string): Promise<{ raw_text: string; normalized_text: string; phoneme_qa: any }> {
    const formData = new FormData();
    formData.append('text', text);
    const res = await fetch(`${API_BASE}/v1/audio/normalize`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(res);
  },

  // Datasets
  async getDatasets(): Promise<Dataset[]> {
    const res = await fetch(`${API_BASE}/v1/datasets`, { cache: 'no-store' });
    return handleResponse<Dataset[]>(res);
  },

  // Training
  async getTrainingRuns(): Promise<TrainingRun[]> {
    const res = await fetch(`${API_BASE}/v1/training-runs`, { cache: 'no-store' });
    return handleResponse<TrainingRun[]>(res);
  },

  // Evaluations
  async getEvaluations(): Promise<EvaluationRun[]> {
    const res = await fetch(`${API_BASE}/v1/evaluations`, { cache: 'no-store' });
    return handleResponse<EvaluationRun[]>(res);
  },

  // Deployments
  async getDeployments(): Promise<Deployment[]> {
    const res = await fetch(`${API_BASE}/v1/deployments`, { cache: 'no-store' });
    return handleResponse<Deployment[]>(res);
  },

  // Dashboard aggregate stats
  async getDashboardStats(): Promise<DashboardStats> {
    const [speakers, datasets, runs, deployments] = await Promise.allSettled([
      this.getSpeakers(),
      this.getDatasets(),
      this.getTrainingRuns(),
      this.getDeployments(),
    ]);
    return {
      speakers: speakers.status === 'fulfilled' ? speakers.value.length : 0,
      datasets: datasets.status === 'fulfilled' ? datasets.value.length : 0,
      trainingRuns: runs.status === 'fulfilled' ? runs.value.length : 0,
      deployments: deployments.status === 'fulfilled' ? deployments.value.length : 0,
    };
  },

  // Models
  async getModels(): Promise<any[]> {
    const res = await fetch(`${API_BASE}/v1/models`, { cache: 'no-store' });
    return handleResponse(res);
  },

  async transitionModelState(modelId: string, newState: string, reason?: string): Promise<any> {
    const params = new URLSearchParams({ new_state: newState });
    if (reason) params.append('reason', reason);
    const res = await fetch(`${API_BASE}/v1/models/${modelId}/state?${params}`, {
      method: 'PATCH',
    });
    return handleResponse(res);
  },

  // Health
  async getHealth(): Promise<{ status: string; version: string; environment: string }> {
    const res = await fetch(`${API_BASE}/v1/health`, { cache: 'no-store' });
    return handleResponse(res);
  },
};
