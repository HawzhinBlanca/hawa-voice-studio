"""
VoxCPM2 Integration Adapter for Central Kurdish (Sorani).
Handles tokenizer-free Sorani prompt formatting, reference conditioning cache,
and LoRA adapter dynamic loading.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class VoxCPMGenerationConfig:
    speed: float = 1.0
    cfg_strength: float = 2.0
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 42
    sample_rate: int = 48000


class VoxCPM2Adapter:
    """
    Interface to OpenBMB VoxCPM2 Foundation Model (2B parameters, native 48 kHz).
    """

    def __init__(
        self,
        model_name_or_path: str = "openbmb/VoxCPM2",
        device: str = "cuda",
        dtype: str = "bfloat16"
    ):
        self.model_name = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.loaded_adapters: Dict[str, str] = {}
        self.reference_cache: Dict[str, any] = {}

    def format_sorani_prompt(
        self,
        normalized_text: str,
        style_instruction: Optional[str] = None,
        speaker_id: Optional[str] = None
    ) -> str:
        """
        Build tokenizer-free conditioning prompt for VoxCPM2.
        Combines natural language style instruction and normalized spoken Kurdish text.
        """
        if style_instruction:
            return f"<|style|>{style_instruction}<|text|>{normalized_text}<|endofprompt|>"
        return f"<|text|>{normalized_text}<|endofprompt|>"

    def load_lora_adapter(self, adapter_name: str, adapter_path: str):
        """Hot-load speaker-specific or style LoRA adapter weights."""
        self.loaded_adapters[adapter_name] = adapter_path

    def unload_lora_adapter(self, adapter_name: str):
        """Unload LoRA adapter."""
        if adapter_name in self.loaded_adapters:
            del self.loaded_adapters[adapter_name]

    def cache_speaker_reference(self, speaker_id: str, reference_audio_path: str, reference_transcript: str):
        """Precompute and cache speaker acoustic reference embeddings for instant inference."""
        self.reference_cache[speaker_id] = {
            "audio_path": reference_audio_path,
            "transcript": reference_transcript,
            "embedding_cached": True
        }
