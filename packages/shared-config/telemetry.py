"""
Telemetry, Prometheus Metrics, and Logging for Sorani Voice Studio.
Tracks TTFB (Time-To-First-Byte), Real-Time Factor (RTF), synthesis volume, and errors.
"""

import logging
import time
from typing import Dict, Optional

# Structured logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("hawa.sorani.voice")


class MetricsTracker:
    """
    In-memory / Prometheus metrics tracker for TTS synthesis and training runs.
    """

    def __init__(self):
        self.synthesis_total_requests = 0
        self.synthesis_total_characters = 0
        self.synthesis_total_audio_seconds = 0.0
        self.synthesis_ttfb_samples = []
        self.active_streaming_sessions = 0
        self.training_runs_count = 0
        self.failed_jobs_count = 0

    def record_synthesis(
        self,
        char_count: int,
        audio_duration_seconds: float,
        ttfb_ms: float,
        model_id: str,
        voice_id: str
    ):
        self.synthesis_total_requests += 1
        self.synthesis_total_characters += char_count
        self.synthesis_total_audio_seconds += audio_duration_seconds
        self.synthesis_ttfb_samples.append(ttfb_ms)
        if len(self.synthesis_ttfb_samples) > 1000:
            self.synthesis_ttfb_samples = self.synthesis_ttfb_samples[-500:]

    def get_summary(self) -> Dict[str, any]:
        ttfb_avg = sum(self.synthesis_ttfb_samples) / max(1, len(self.synthesis_ttfb_samples))
        sorted_ttfb = sorted(self.synthesis_ttfb_samples) if self.synthesis_ttfb_samples else [0.0]
        p95_idx = int(len(sorted_ttfb) * 0.95)
        p95_ttfb = sorted_ttfb[min(p95_idx, len(sorted_ttfb) - 1)]

        return {
            "total_requests": self.synthesis_total_requests,
            "total_characters": self.synthesis_total_characters,
            "total_audio_hours": round(self.synthesis_total_audio_seconds / 3600.0, 2),
            "avg_ttfb_ms": round(ttfb_avg, 1),
            "p95_ttfb_ms": round(p95_ttfb, 1),
            "active_streams": self.active_streaming_sessions,
            "training_runs": self.training_runs_count,
            "failed_jobs": self.failed_jobs_count,
        }


metrics = MetricsTracker()
