"""
Canonical Contracts and Domain DTOs for Sorani Voice Studio.
Used across FastAPI, Temporal Workflows, Inference Gateway, and TypeScript client generation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ==========================================
# Enums
# ==========================================

class SpeakerStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class Dialect(str, Enum):
    SLEMANI = "slemani"
    ERBIL = "erbil"
    DUHOK_BADINI = "badini"
    MUKRI = "mukri"
    STANDARD_CKB = "standard_ckb"


class ConsentType(str, Enum):
    FULL_COMMERCIAL_EXCLUSIVE = "full_commercial_exclusive"
    COMMERCIAL_NON_EXCLUSIVE = "commercial_non_exclusive"
    RESEARCH_ONLY = "research_only"
    RESTRICTED = "restricted"


class UtteranceQualityStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETAKE_REQUESTED = "retake_requested"


class TrainingPreset(str, Enum):
    SORANI_PILOT_LORA = "sorani_pilot_lora"
    FULL_SORANI_FOUNDATION_SFT = "full_sorani_foundation_sft"
    PREMIUM_SPEAKER_LORA = "premium_speaker_lora"
    STYLE_DOMAIN_ADAPTATION = "style_domain_adaptation"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PREPARING_DATA = "preparing_data"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioFormat(str, Enum):
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"
    PCM = "pcm"


class DeploymentState(str, Enum):
    INACTIVE = "inactive"
    CANARY = "canary"
    ACTIVE_PRODUCTION = "active_production"
    DEPRECATED = "deprecated"
    ROLLBACK = "rollback"


# ==========================================
# Speaker Contracts
# ==========================================

class SpeakerConsentDTO(BaseModel):
    consent_id: str
    consent_type: ConsentType
    contract_signed_date: datetime
    expiry_date: Optional[datetime] = None
    commercial_use_permitted: bool = True
    derivative_model_permitted: bool = True
    prohibited_contexts: List[str] = Field(default_factory=list)
    consent_document_uri: Optional[str] = None
    revoked_at: Optional[datetime] = None


class SpeakerReferenceClipDTO(BaseModel):
    reference_id: str
    speaker_id: str
    style_name: str
    audio_uri: str
    exact_transcript_raw: str
    exact_transcript_normalized: str
    duration_seconds: float
    is_canonical: bool = False
    embedding_vector_cached: bool = True


class SpeakerStyleDTO(BaseModel):
    style_id: str
    speaker_id: str
    name: str  # e.g., warm_documentary, energetic, serious, whisper
    instruction_prompt: str
    recommended_speed: float = 1.0
    canonical_reference_id: Optional[str] = None


class SpeakerProfileDTO(BaseModel):
    speaker_id: str
    name: str
    kurdish_name: str
    dialect: Dialect
    gender: str
    status: SpeakerStatus
    age_bracket: str
    voice_description: str
    consent: Optional[SpeakerConsentDTO] = None
    references: List[SpeakerReferenceClipDTO] = Field(default_factory=list)
    styles: List[SpeakerStyleDTO] = Field(default_factory=list)
    naturalness_score: float = 0.0
    similarity_score: float = 0.0
    pronunciation_score: float = 0.0
    created_at: datetime
    updated_at: datetime


class CreateSpeakerRequest(BaseModel):
    name: str
    kurdish_name: str
    dialect: Dialect = Dialect.SLEMANI
    gender: str
    age_bracket: str = "adult"
    voice_description: str
    consent_type: ConsentType = ConsentType.COMMERCIAL_NON_EXCLUSIVE
    commercial_use_permitted: bool = True
    derivative_model_permitted: bool = True


# ==========================================
# Dataset Contracts
# ==========================================

class UtteranceDTO(BaseModel):
    utterance_id: str
    dataset_id: str
    speaker_id: Optional[str] = None
    raw_transcript: str
    normalized_transcript: str
    audio_uri: str
    duration_seconds: float
    style_label: str = "neutral"
    quality_status: UtteranceQualityStatus
    snr_db: float
    silence_ratio: float
    rejection_reason: Optional[str] = None
    created_at: datetime


class DatasetDTO(BaseModel):
    dataset_id: str
    name: str
    description: str
    source: str  # e.g. studio_recording, mozilla_cv_ckb, central_kurdish_tts
    license: str
    total_hours: float
    approved_hours: float
    utterance_count: int
    is_frozen: bool = False
    current_version: Optional[str] = None
    created_at: datetime


class FreezeDatasetRequest(BaseModel):
    dataset_id: str
    version_tag: str
    notes: str


# ==========================================
# Training Contracts
# ==========================================

class TrainingRunRequest(BaseModel):
    run_name: str
    preset: TrainingPreset
    base_model: str = "openbmb/VoxCPM2"
    dataset_id: str
    dataset_version: str
    speaker_id: Optional[str] = None  # for speaker-specific LoRA
    max_steps: int = 20000
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    batch_size_per_gpu: int = 8
    cost_guardrail_dollars: float = 250.0
    target_gpu_type: str = "1x L40S 48GB"


class TrainingCheckpointDTO(BaseModel):
    checkpoint_id: str
    step: int
    validation_loss: float
    cer_score: float
    checkpoint_s3_uri: str
    created_at: datetime


class TrainingRunDTO(BaseModel):
    run_id: str
    run_name: str
    preset: TrainingPreset
    base_model: str
    dataset_version: str
    status: TrainingStatus
    current_step: int
    total_steps: int
    current_loss: float
    best_loss: float
    gpu_type: str
    estimated_cost_spent: float
    checkpoints: List[TrainingCheckpointDTO] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


# ==========================================
# Evaluation Contracts
# ==========================================

class EvaluationScoreDTO(BaseModel):
    evaluation_id: str
    model_version_id: str
    baseline_model_id: str  # e.g. F5-TTS, CosyVoice3, RegaLabs
    sentence_id: str
    naturalness_mos: float  # 1-5
    pronunciation_accuracy: float  # 1-5
    speaker_similarity: float  # 1-5
    emotion_authenticity: float  # 1-5
    cer_score: float
    evaluator_type: str  # native_linguist, automated_asr, crowd_listener
    comments: Optional[str] = None


class EvaluationRunDTO(BaseModel):
    evaluation_id: str
    title: str
    model_version_id: str
    challenger_model_id: str
    test_suite_tag: str  # core_sorani, production_text, expressive, long_form
    sample_count: int
    avg_naturalness: float
    avg_pronunciation: float
    avg_similarity: float
    avg_cer: float
    win_rate_vs_baseline: float
    is_approved_for_production: bool
    created_at: datetime


# ==========================================
# Speech Synthesis & Inference Contracts
# ==========================================

class SpeechRequest(BaseModel):
    model: str = "sorani-pro-v1"
    input: str = Field(..., description="Kurdish Sorani text to synthesize")
    voice: str = Field(..., description="Speaker profile ID or slug (e.g. lamo, heja, shakar)")
    style: str = "warm_documentary"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    seed: Optional[int] = 42
    format: AudioFormat = AudioFormat.WAV
    stream: bool = True
    watermark_enabled: bool = True


class PreviewCloneRequest(BaseModel):
    target_text: str
    reference_audio_base64: Optional[str] = None
    reference_audio_uri: Optional[str] = None
    reference_transcript: Optional[str] = None
    style: str = "neutral"
    speed: float = 1.0


class SpeechJobResponse(BaseModel):
    job_id: str
    status: str
    audio_url: Optional[str] = None
    duration_seconds: float
    characters_processed: int
    ttfb_ms: float
    watermark_payload_id: int
    created_at: datetime


# ==========================================
# Deployment Contracts
# ==========================================

class DeploymentDTO(BaseModel):
    deployment_id: str
    model_version_id: str
    model_name: str
    state: DeploymentState
    traffic_percentage: int = 100
    p95_latency_ms: float
    rtf_score: float
    active_adapters: List[str] = Field(default_factory=list)
    deployed_at: datetime
    updated_at: datetime
