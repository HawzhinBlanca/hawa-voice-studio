"""
SQLAlchemy 2.0 Database Schema for Hawa Sorani Voice Studio.
Defines all 20+ entities with relational constraints, JSONB attributes, and audit hooks.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from packages.shared_config.database import Base, TimestampMixin


def generate_uuid() -> str:
    return str(uuid.uuid4())


# ==========================================
# 1. Organization & Identity
# ==========================================

class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    speakers: Mapped[List["Speaker"]] = relationship("Speaker", back_populates="organization")
    api_keys: Mapped[List["ApiKey"]] = relationship("ApiKey", back_populates="organization")
    datasets: Mapped[List["Dataset"]] = relationship("Dataset", back_populates="organization")


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="linguist_reviewer")  # admin, engineer, linguist_reviewer, listener
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class ApiKey(Base, TimestampMixin):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    rate_limit_rpm: Mapped[int] = mapped_column(Integer, default=120)

    organization: Mapped["Organization"] = relationship("Organization", back_populates="api_keys")


# ==========================================
# 2. Speaker Profiles & Legal Governance
# ==========================================

class Speaker(Base, TimestampMixin):
    __tablename__ = "speakers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    org_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("organizations.id"), nullable=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kurdish_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dialect: Mapped[str] = mapped_column(String(50), default="slemani")  # slemani, erbil, badini, mukri
    gender: Mapped[str] = mapped_column(String(20), nullable=False)
    age_bracket: Mapped[str] = mapped_column(String(30), default="adult")
    voice_description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="active")  # draft, active, suspended, revoked
    
    # Benchmarked scores
    naturalness_score: Mapped[float] = mapped_column(Float, default=4.5)
    similarity_score: Mapped[float] = mapped_column(Float, default=4.7)
    pronunciation_score: Mapped[float] = mapped_column(Float, default=4.8)

    # Relationships
    organization: Mapped[Optional["Organization"]] = relationship("Organization", back_populates="speakers")
    consents: Mapped[List["SpeakerConsent"]] = relationship("SpeakerConsent", back_populates="speaker", cascade="all, delete-orphan")
    references: Mapped[List["SpeakerReference"]] = relationship("SpeakerReference", back_populates="speaker", cascade="all, delete-orphan")
    styles: Mapped[List["SpeakerStyle"]] = relationship("SpeakerStyle", back_populates="speaker", cascade="all, delete-orphan")
    pronunciations: Mapped[List["SpeakerPronunciation"]] = relationship("SpeakerPronunciation", back_populates="speaker", cascade="all, delete-orphan")
    voice_adapters: Mapped[List["VoiceAdapter"]] = relationship("VoiceAdapter", back_populates="speaker")


class SpeakerConsent(Base, TimestampMixin):
    __tablename__ = "speaker_consents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    speaker_id: Mapped[str] = mapped_column(String(36), ForeignKey("speakers.id"), nullable=False)
    consent_type: Mapped[str] = mapped_column(String(50), default="commercial_non_exclusive")
    signed_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expiry_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    commercial_use_permitted: Mapped[bool] = mapped_column(Boolean, default=True)
    derivative_model_permitted: Mapped[bool] = mapped_column(Boolean, default=True)
    prohibited_contexts: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    document_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    speaker: Mapped["Speaker"] = relationship("Speaker", back_populates="consents")


class SpeakerReference(Base, TimestampMixin):
    __tablename__ = "speaker_references"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    speaker_id: Mapped[str] = mapped_column(String(36), ForeignKey("speakers.id"), nullable=False)
    style_name: Mapped[str] = mapped_column(String(100), default="neutral")
    audio_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    raw_transcript: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_transcript: Mapped[str] = mapped_column(Text, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    is_canonical: Mapped[bool] = mapped_column(Boolean, default=False)
    embedding_cache_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    speaker: Mapped["Speaker"] = relationship("Speaker", back_populates="references")


class SpeakerStyle(Base, TimestampMixin):
    __tablename__ = "speaker_styles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    speaker_id: Mapped[str] = mapped_column(String(36), ForeignKey("speakers.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., warm_documentary, energetic
    instruction_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_speed: Mapped[float] = mapped_column(Float, default=1.0)
    canonical_reference_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    speaker: Mapped["Speaker"] = relationship("Speaker", back_populates="styles")


class SpeakerPronunciation(Base, TimestampMixin):
    __tablename__ = "speaker_pronunciations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    speaker_id: Mapped[str] = mapped_column(String(36), ForeignKey("speakers.id"), nullable=False)
    grapheme: Mapped[str] = mapped_column(String(100), nullable=False)
    spoken_form: Mapped[str] = mapped_column(String(255), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    speaker: Mapped["Speaker"] = relationship("Speaker", back_populates="pronunciations")


# ==========================================
# 3. Datasets & Utterance Curation
# ==========================================

class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    org_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("organizations.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(100), default="studio_recording")
    license: Mapped[str] = mapped_column(String(100), default="Proprietary / Cleared")
    total_hours: Mapped[float] = mapped_column(Float, default=0.0)
    approved_hours: Mapped[float] = mapped_column(Float, default=0.0)
    utterance_count: Mapped[int] = mapped_column(Integer, default=0)
    is_frozen: Mapped[bool] = mapped_column(Boolean, default=False)
    current_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    organization: Mapped[Optional["Organization"]] = relationship("Organization", back_populates="datasets")
    versions: Mapped[List["DatasetVersion"]] = relationship("DatasetVersion", back_populates="dataset")
    utterances: Mapped[List["Utterance"]] = relationship("Utterance", back_populates="dataset")


class DatasetVersion(Base, TimestampMixin):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., v1.0-frozen
    manifest_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    approved_utterances: Mapped[int] = mapped_column(Integer, default=0)
    duration_hours: Mapped[float] = mapped_column(Float, default=0.0)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="versions")


class Utterance(Base, TimestampMixin):
    __tablename__ = "utterances"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), nullable=False)
    speaker_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("speakers.id"), nullable=True)
    raw_transcript: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_transcript: Mapped[str] = mapped_column(Text, nullable=False)
    archive_audio_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    derivative_audio_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    style_label: Mapped[str] = mapped_column(String(50), default="neutral")
    quality_status: Mapped[str] = mapped_column(String(30), default="pending_review")  # pending_review, approved, rejected, retake_requested
    snr_db: Mapped[float] = mapped_column(Float, default=25.0)
    silence_ratio: Mapped[float] = mapped_column(Float, default=0.05)
    rejection_reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="utterances")
    reviews: Mapped[List["UtteranceReview"]] = relationship("UtteranceReview", back_populates="utterance")


class UtteranceReview(Base, TimestampMixin):
    __tablename__ = "utterance_reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    utterance_id: Mapped[str] = mapped_column(String(36), ForeignKey("utterances.id"), nullable=False)
    reviewer_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    decision: Mapped[str] = mapped_column(String(30), nullable=False)  # approved, rejected, retake
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    utterance: Mapped["Utterance"] = relationship("Utterance", back_populates="reviews")


# ==========================================
# 4. Training Orchestration & Models
# ==========================================

class TrainingRun(Base, TimestampMixin):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    run_name: Mapped[str] = mapped_column(String(255), nullable=False)
    preset: Mapped[str] = mapped_column(String(50), nullable=False)
    base_model: Mapped[str] = mapped_column(String(100), default="openbmb/VoxCPM2")
    dataset_version_id: Mapped[str] = mapped_column(String(36), nullable=False)
    speaker_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("speakers.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="running")
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=20000)
    current_loss: Mapped[float] = mapped_column(Float, default=1.5)
    best_loss: Mapped[float] = mapped_column(Float, default=1.5)
    gpu_type: Mapped[str] = mapped_column(String(100), default="1x L40S 48GB")
    estimated_cost_spent: Mapped[float] = mapped_column(Float, default=0.0)
    wandb_run_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    checkpoints: Mapped[List["TrainingCheckpoint"]] = relationship("TrainingCheckpoint", back_populates="training_run")


class TrainingCheckpoint(Base, TimestampMixin):
    __tablename__ = "training_checkpoints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    training_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.id"), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_loss: Mapped[float] = mapped_column(Float, nullable=False)
    cer_score: Mapped[float] = mapped_column(Float, default=0.04)
    checkpoint_s3_uri: Mapped[str] = mapped_column(String(500), nullable=False)

    training_run: Mapped["TrainingRun"] = relationship("TrainingRun", back_populates="checkpoints")


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"

    # Valid state transitions (§13)
    VALID_TRANSITIONS = {
        "draft": {"training"},
        "training": {"evaluating", "failed"},
        "evaluating": {"approved", "rejected"},
        "rejected": {"training"},  # can re-train
        "approved": {"canary", "deprecated"},
        "canary": {"production", "rejected", "deprecated"},
        "production": {"deprecated", "revoked"},
        "deprecated": set(),  # terminal
        "revoked": set(),  # terminal
        "failed": {"training"},  # can retry
    }

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version_tag: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # e.g. voxcpm2-ckb-foundation-v1
    architecture: Mapped[str] = mapped_column(String(50), default="VoxCPM2")  # VoxCPM2, CosyVoice3
    state: Mapped[str] = mapped_column(String(30), default="draft")  # draft, training, evaluating, rejected, approved, canary, production, deprecated, revoked
    weights_s3_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    is_foundation: Mapped[bool] = mapped_column(Boolean, default=False)
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    naturalness_mos: Mapped[float] = mapped_column(Float, default=0.0)
    cer_score: Mapped[float] = mapped_column(Float, default=1.0)

    # Reproducibility metadata (§13)
    training_run_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("training_runs.id"), nullable=True)
    base_model_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    git_commit: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    docker_image_digest: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    dataset_version_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    normalizer_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    random_seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gpu_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    voice_adapters: Mapped[List["VoiceAdapter"]] = relationship("VoiceAdapter", back_populates="model_version")

    def transition_state(self, new_state: str) -> None:
        """Enforce valid state transitions. Raises ValueError on invalid transition."""
        allowed = self.VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid model state transition: '{self.state}' → '{new_state}'. "
                f"Allowed transitions from '{self.state}': {allowed or 'none (terminal state)'}"
            )
        self.state = new_state
        if new_state == "approved":
            self.is_approved = True
        elif new_state == "revoked":
            self.is_approved = False


class VoiceAdapter(Base, TimestampMixin):
    __tablename__ = "voice_adapters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    speaker_id: Mapped[str] = mapped_column(String(36), ForeignKey("speakers.id"), nullable=False)
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=False)
    adapter_name: Mapped[str] = mapped_column(String(100), nullable=False)
    adapter_weights_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    lora_rank: Mapped[int] = mapped_column(Integer, default=16)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    speaker: Mapped["Speaker"] = relationship("Speaker", back_populates="voice_adapters")
    model_version: Mapped["ModelVersion"] = relationship("ModelVersion", back_populates="voice_adapters")


# ==========================================
# 5. Evaluation Lab
# ==========================================

class EvaluationRun(Base, TimestampMixin):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=False)
    challenger_model_id: Mapped[str] = mapped_column(String(100), default="F5-TTS-baseline")
    test_suite_tag: Mapped[str] = mapped_column(String(100), default="core_sorani")
    sample_count: Mapped[int] = mapped_column(Integer, default=50)
    avg_naturalness: Mapped[float] = mapped_column(Float, default=4.65)
    avg_pronunciation: Mapped[float] = mapped_column(Float, default=4.82)
    avg_similarity: Mapped[float] = mapped_column(Float, default=4.75)
    avg_cer: Mapped[float] = mapped_column(Float, default=0.032)
    win_rate_vs_baseline: Mapped[float] = mapped_column(Float, default=74.5)
    is_approved_for_production: Mapped[bool] = mapped_column(Boolean, default=True)

    scores: Mapped[List["EvaluationScore"]] = relationship("EvaluationScore", back_populates="evaluation_run")


class EvaluationScore(Base, TimestampMixin):
    __tablename__ = "evaluation_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    evaluation_id: Mapped[str] = mapped_column(String(36), ForeignKey("evaluation_runs.id"), nullable=False)
    sentence_id: Mapped[str] = mapped_column(String(100), nullable=False)
    naturalness_mos: Mapped[float] = mapped_column(Float, nullable=False)
    pronunciation_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    speaker_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    emotion_authenticity: Mapped[float] = mapped_column(Float, nullable=False)
    cer_score: Mapped[float] = mapped_column(Float, nullable=False)
    evaluator_type: Mapped[str] = mapped_column(String(50), default="native_linguist")

    evaluation_run: Mapped["EvaluationRun"] = relationship("EvaluationRun", back_populates="scores")


# ==========================================
# 6. Deployments, Synthesis & Auditing
# ==========================================

class Deployment(Base, TimestampMixin):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=False)
    state: Mapped[str] = mapped_column(String(30), default="active_production")  # canary, active_production, deprecated, rollback
    traffic_percentage: Mapped[int] = mapped_column(Integer, default=100)
    p95_latency_ms: Mapped[float] = mapped_column(Float, default=320.0)
    rtf_score: Mapped[float] = mapped_column(Float, default=0.28)
    endpoint_url: Mapped[str] = mapped_column(String(500), default="http://vllm-omni:8000/v1")


class SynthesisJob(Base, TimestampMixin):
    __tablename__ = "synthesis_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    speaker_id: Mapped[str] = mapped_column(String(36), ForeignKey("speakers.id"), nullable=False)
    model_id: Mapped[str] = mapped_column(String(100), default="sorani-pro-v1")
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    style: Mapped[str] = mapped_column(String(50), default="warm_documentary")
    output_audio_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    audio_output_uri: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    characters_processed: Mapped[int] = mapped_column(Integer, default=0)
    ttfb_ms: Mapped[float] = mapped_column(Float, default=0.0)
    watermark_payload_id: Mapped[int] = mapped_column(Integer, default=42)
    status: Mapped[str] = mapped_column(String(30), default="completed")


class AuditEvent(Base, TimestampMixin):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    actor_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., speaker_consent_granted, voice_revoked, model_promoted
    target_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_id: Mapped[str] = mapped_column(String(100), nullable=False)
    payload_json: Mapped[Optional[dict]] = mapped_column(JSON, default=dict)


# ==========================================
# 9. Organization Members (§18 gap)
# ==========================================

class OrganizationMember(Base, TimestampMixin):
    __tablename__ = "organization_members"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="member")  # admin, engineer, linguist, listener
    invited_by: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


# ==========================================
# 10. Training Metrics (§18 gap)
# ==========================================

class TrainingMetric(Base, TimestampMixin):
    __tablename__ = "training_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    training_run_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_runs.id"), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)  # train_loss, val_loss, val_cer, learning_rate, gpu_vram_gb
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    gpu_utilization_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wall_clock_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


# ==========================================
# 11. Generated Assets (§18 gap)
# ==========================================

class GeneratedAsset(Base, TimestampMixin):
    __tablename__ = "generated_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    synthesis_job_id: Mapped[str] = mapped_column(String(36), ForeignKey("synthesis_jobs.id"), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(50), nullable=False)  # wav, flac, mp3, pcm
    storage_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    sample_rate: Mapped[int] = mapped_column(Integer, default=48000)
    channels: Mapped[int] = mapped_column(Integer, default=1)
    watermark_payload_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    checksum_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

