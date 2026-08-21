"""
Audio Pipeline, VAD, and AudioSeal Watermark Unit Tests.
"""

import math
import pytest
from packages.audio_processing import AudioPipeline, AudioSealWatermark, QualityAnalyzer


def generate_sine_wave(freq: float = 440.0, duration: float = 1.0, sample_rate: int = 48000) -> list[float]:
    num_samples = int(duration * sample_rate)
    return [0.8 * math.sin(2.0 * math.pi * freq * (i / sample_rate)) for i in range(num_samples)]


def test_wav_write_and_read():
    """Test lossless WAV sample conversion roundtrip."""
    samples = generate_sine_wave(440.0, 0.5, 48000)
    wav_bytes = AudioPipeline.write_wav_bytes(samples, 48000, sample_width=2)
    
    assert len(wav_bytes) > 44  # Valid WAV header
    read_samples, sr, channels = AudioPipeline.read_wav_bytes(wav_bytes)
    
    assert sr == 48000
    assert channels == 1
    assert len(read_samples) == len(samples)
    # Check sample precision within 16-bit quantization threshold
    diff = max(abs(a - b) for a, b in zip(samples, read_samples))
    assert diff < 0.001


def test_vad_silence_trimming():
    """Test energy-based VAD trims leading and excessive trailing silence."""
    tone = generate_sine_wave(440.0, 1.0, 16000)
    silence_start = [0.0] * int(16000 * 0.5)
    silence_end = [0.0] * int(16000 * 0.8)
    
    padded_audio = silence_start + tone + silence_end
    trimmed = AudioPipeline.trim_silence_vad(padded_audio, 16000, trailing_silence_keep_ms=100)
    
    # Trimmed duration should be roughly 1.0s + ~100ms
    trimmed_sec = len(trimmed) / 16000.0
    assert 0.9 <= trimmed_sec <= 1.25


def test_audioseal_watermarking():
    """Test AudioSeal 16-bit payload embedding and detection."""
    samples = generate_sine_wave(220.0, 2.0, 48000)
    payload_id = 42981

    # Embed watermark
    watermarked = AudioSealWatermark.embed_watermark(samples, payload_id, sample_rate=48000)
    assert len(watermarked) == len(samples)

    # Detect exact payload
    result = AudioSealWatermark.detect_watermark(watermarked, payload_id, sample_rate=48000)
    assert result.detected is True
    assert result.payload_id == payload_id
    assert result.confidence > 0.5

    # Detect incorrect candidate payload
    wrong_result = AudioSealWatermark.detect_watermark(watermarked, 12345, sample_rate=48000)
    assert wrong_result.detected is False


def test_quality_analyzer():
    """Test quality analyzer detects clipping and silence anomalies."""
    clean_samples = generate_sine_wave(440.0, 1.5, 48000)
    report = QualityAnalyzer.evaluate(clean_samples, 48000)
    
    assert report.is_acceptable is True
    assert report.clipping_rate == 0.0
    assert report.stop_failure_detected is False
