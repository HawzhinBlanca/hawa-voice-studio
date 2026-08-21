"""
Inference Gateway Service.
Secure proxy between the FastAPI control plane and GPU inference backends.

Architecture (§14):
- vLLM-Omni for foundation and registered-reference voices
- Official VoxCPM workers for LoRA voices initially
- One dedicated deployment per approved flagship adapter if necessary

Security (§14):
- Never expose vLLM directly — API-key protection does not cover every endpoint
- Keep vLLM on a private network behind this gateway
- All requests must pass through the FastAPI control plane first
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional


class InferenceBackend(str, Enum):
    VLLM_OMNI = "vllm_omni"
    VOXCPM_LORA = "voxcpm_lora"
    COSYVOICE3 = "cosyvoice3"


@dataclass
class InferenceRequest:
    request_id: str
    text: str
    normalized_text: str
    speaker_id: str
    style: str
    speed: float
    seed: Optional[int]
    reference_audio_uri: Optional[str]
    reference_transcript: Optional[str]
    adapter_id: Optional[str]
    backend: InferenceBackend
    stream: bool = True
    watermark_payload_id: int = 0


@dataclass
class InferenceResponse:
    request_id: str
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    ttfb_ms: float
    backend_used: InferenceBackend
    watermark_applied: bool


class InferenceGateway:
    """
    Routes TTS requests to the appropriate GPU inference backend based on
    voice type (foundation, registered reference, or premium LoRA adapter).

    Production serving constraints (§14):
    - P95 TTFB < 500ms
    - Real-time factor < 0.5
    - No audible gaps in streaming
    - Deterministic cancellation
    """

    def __init__(
        self,
        vllm_omni_url: str = "http://localhost:8001",
        voxcpm_lora_url: str = "http://localhost:8002",
        cosyvoice_url: str = "http://localhost:8003",
    ):
        self.backends: Dict[InferenceBackend, str] = {
            InferenceBackend.VLLM_OMNI: vllm_omni_url,
            InferenceBackend.VOXCPM_LORA: voxcpm_lora_url,
            InferenceBackend.COSYVOICE3: cosyvoice_url,
        }
        self._health_cache: Dict[InferenceBackend, bool] = {}

    def resolve_backend(self, adapter_id: Optional[str] = None) -> InferenceBackend:
        """
        Route to the correct backend:
        - If adapter_id is set → VoxCPM LoRA worker (until hot-swapping is validated)
        - Otherwise → vLLM-Omni for foundation/registered voices
        """
        if adapter_id:
            return InferenceBackend.VOXCPM_LORA
        return InferenceBackend.VLLM_OMNI

    async def synthesize(self, req: InferenceRequest) -> InferenceResponse:
        """Send a synthesis request to the appropriate backend."""
        backend = req.backend
        base_url = self.backends.get(backend, self.backends[InferenceBackend.VLLM_OMNI])

        start_time = time.monotonic()

        # In production: HTTP POST to backend's /v1/audio/speech endpoint
        # For now, return a placeholder response
        ttfb = (time.monotonic() - start_time) * 1000

        return InferenceResponse(
            request_id=req.request_id,
            audio_bytes=b"",  # Real audio bytes from backend
            sample_rate=48000,
            duration_seconds=0.0,
            ttfb_ms=round(ttfb, 1),
            backend_used=backend,
            watermark_applied=req.watermark_payload_id > 0,
        )

    async def stream_synthesis(self, req: InferenceRequest) -> AsyncIterator[bytes]:
        """Stream PCM chunks from the inference backend to the client."""
        backend = req.backend
        base_url = self.backends.get(backend, self.backends[InferenceBackend.VLLM_OMNI])

        # In production: stream PCM chunks via SSE/WebSocket from backend
        # Gapless streaming using AudioWorklet on the client side
        chunk_size = 4800  # 100ms at 48kHz mono 16-bit
        for _ in range(10):
            yield b"\x00" * chunk_size * 2  # silence placeholder
            await asyncio.sleep(0.1)

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all inference backends."""
        results = {}
        for backend, url in self.backends.items():
            try:
                # In production: HTTP GET to {url}/health
                results[backend.value] = True
            except Exception:
                results[backend.value] = False
        self._health_cache = {k: v for k, v in zip(self.backends.keys(), results.values())}
        return results
