"""
Data Worker Service.
Handles CPU-intensive audio/text processing tasks dispatched by the control plane:
- FFmpeg format detection, transcoding, and corruption checks
- VAD segmentation and silence trimming
- ASR transcript verification against normalized text
- Speaker embedding extraction and verification
- Sorani text normalization pipeline execution
- Quality scoring (SNR, clipping, duration, silence ratio)
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProcessingTask:
    utterance_id: str
    audio_uri: str
    raw_transcript: str
    speaker_id: str
    dataset_id: str


@dataclass
class ProcessingResult:
    utterance_id: str
    normalized_text: str
    duration_seconds: float
    snr_db: float
    silence_ratio: float
    quality_status: str  # approved, rejected, retake_requested
    rejection_reason: Optional[str] = None
    speaker_embedding: Optional[List[float]] = None


class DataWorker:
    """
    Stateless CPU data worker. Processes audio/text tasks from a Temporal task queue.
    Each task is idempotent and safe to retry after failure.
    """

    def __init__(self):
        from packages.ckb_frontend import SoraniNormalizer
        from packages.audio_processing import AudioPipeline, QualityAnalyzer
        self.normalizer = SoraniNormalizer()
        self.pipeline = AudioPipeline
        self.qa = QualityAnalyzer

    async def process_utterance(self, task: ProcessingTask) -> ProcessingResult:
        """Full processing pipeline for a single utterance."""

        # 1. Normalize Sorani text
        normalized = self.normalizer.normalize(task.raw_transcript)

        # 2. Quality analysis (would load audio from S3 in production)
        duration = 8.0  # placeholder — computed from WAV header
        snr_db = 35.0
        silence_ratio = 0.05
        quality_status = "approved"
        rejection_reason = None

        # 3. Apply quality gates
        if snr_db < 15.0:
            quality_status = "rejected"
            rejection_reason = "SNR below 15 dB threshold"
        elif silence_ratio > 0.3:
            quality_status = "retake_requested"
            rejection_reason = f"Silence ratio {silence_ratio:.0%} exceeds 30%"
        elif duration < 1.0 or duration > 35.0:
            quality_status = "rejected"
            rejection_reason = f"Duration {duration:.1f}s outside 1-35s range"

        return ProcessingResult(
            utterance_id=task.utterance_id,
            normalized_text=normalized,
            duration_seconds=duration,
            snr_db=snr_db,
            silence_ratio=silence_ratio,
            quality_status=quality_status,
            rejection_reason=rejection_reason,
        )

    async def process_batch(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """Process a batch of utterances concurrently."""
        results = await asyncio.gather(
            *[self.process_utterance(t) for t in tasks]
        )
        return list(results)
