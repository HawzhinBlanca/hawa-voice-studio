"""
Configuration settings for Sorani Voice Studio.
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global system configuration."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    APP_NAME: str = "Hawa Sorani Voice Studio"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", description="development, staging, production")
    DEBUG: bool = True
    API_V1_STR: str = "/v1"
    SECRET_KEY: str = Field(default="dev-secret-key-kurdish-tts-super-secure-32chars-min", description="JWT & signing key")
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"]

    # Database
    DATABASE_URL: str = Field(
        default=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/hawa_sorani_voice.db"),
        description="PostgreSQL async or SQLite connection string"
    )
    DATABASE_SYNC_URL: str = Field(
        default=os.getenv("DATABASE_SYNC_URL", "sqlite:///./data/hawa_sorani_voice.db"),
        description="Sync connection string for migrations"
    )

    # Object Storage (S3 / Cloudflare R2)
    S3_ENDPOINT_URL: Optional[str] = None
    S3_ACCESS_KEY_ID: str = "mock-access-key"
    S3_SECRET_ACCESS_KEY: str = "mock-secret-key"
    S3_BUCKET_NAME: str = "hawa-sorani-voice-assets"
    S3_REGION_NAME: str = "auto"

    # Temporal Workflow Engine
    TEMPORAL_HOST: str = "localhost:7233"
    TEMPORAL_NAMESPACE: str = "default"
    TEMPORAL_TASK_QUEUE: str = "hawa-sorani-queue"

    # Model & Inference Workers
    VLLM_OMNI_URL: str = "http://localhost:8001"
    VOXCPM_LORA_WORKER_URL: str = "http://localhost:8002"
    COSYVOICE_CHALLENGER_URL: str = "http://localhost:8003"
    DEFAULT_FOUNDATION_MODEL: str = "openbmb/VoxCPM2"

    # Watermarking
    WATERMARK_SALT: str = "hawa-kurdish-tts-audioseal-salt"
    WATERMARK_ENABLED: bool = True

    # Telemetry
    ENABLE_PROMETHEUS: bool = True
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
    SENTRY_DSN: Optional[str] = None


settings = Settings()
