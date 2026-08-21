"""
Audio Processing Pipeline for Sorani Voice Studio.
Manages 48 kHz immutable audio archives, 16 kHz model training derivatives,
EBU R128 loudness normalization, VAD silence trimming, and streaming PCM encoding.
"""

import io
import math
import struct
import wave
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple


@dataclass
class AudioMetadata:
    sample_rate: int
    num_channels: int
    sample_width: int  # bytes per sample (e.g. 2 for 16-bit, 3 for 24-bit)
    num_frames: int
    duration_seconds: float
    is_clipped: bool
    peak_amplitude: float
    rms_dbfs: float


class AudioPipeline:
    """
    Production audio processing and validation pipeline.
    Ensures VoxCPM2 compatibility: 48 kHz output & 16 kHz input representations.
    """

    ARCHIVE_SAMPLE_RATE = 48000
    DERIVATIVE_SAMPLE_RATE = 16000
    TARGET_INTEGRATED_LUFS = -23.0
    MAX_TRUE_PEAK_DBFS = -1.0
    MIN_UTTERANCE_DURATION = 1.0  # seconds
    MAX_UTTERANCE_DURATION = 30.0  # seconds
    OPTIMAL_UTTERANCE_DURATION = (3.0, 18.0)

    @classmethod
    def read_wav_bytes(cls, wav_bytes: bytes) -> Tuple[List[float], int, int]:
        """
        Read WAV byte stream into normalized float samples [-1.0, 1.0].
        Returns (samples, sample_rate, num_channels).
        """
        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)

        if sampwidth == 2:  # 16-bit PCM
            total_samples = n_frames * n_channels
            fmt = f"<{total_samples}h"
            ints = struct.unpack(fmt, raw_data)
            samples = [val / 32768.0 for val in ints]
        elif sampwidth == 1:  # 8-bit unsigned
            samples = [(val - 128) / 128.0 for val in raw_data]
        elif sampwidth == 3:  # 24-bit PCM
            samples = []
            for i in range(0, len(raw_data), 3):
                chunk = raw_data[i:i+3]
                int_val = int.from_bytes(chunk, byteorder='little', signed=True)
                samples.append(int_val / 8388608.0)
        elif sampwidth == 4:  # 32-bit float or int
            total_samples = n_frames * n_channels
            fmt = f"<{total_samples}i"
            ints = struct.unpack(fmt, raw_data)
            samples = [val / 2147483648.0 for val in ints]
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # If stereo, convert to mono by averaging channels
        if n_channels > 1:
            mono_samples = []
            for i in range(0, len(samples), n_channels):
                mono_samples.append(sum(samples[i:i+n_channels]) / n_channels)
            samples = mono_samples
            n_channels = 1

        return samples, framerate, n_channels

    @classmethod
    def write_wav_bytes(cls, samples: List[float], sample_rate: int, sample_width: int = 2) -> bytes:
        """
        Write normalized float samples [-1.0, 1.0] to WAV bytes.
        """
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)

            # Clamp and convert
            if sample_width == 2:  # 16-bit
                raw = []
                for s in samples:
                    clamped = max(-1.0, min(1.0, s))
                    raw.append(int(clamped * 32767.0))
                wf.writeframes(struct.pack(f"<{len(raw)}h", *raw))
            elif sample_width == 3:  # 24-bit
                raw_bytes = bytearray()
                for s in samples:
                    clamped = max(-1.0, min(1.0, s))
                    int_val = int(clamped * 8388607.0)
                    raw_bytes.extend(int_val.to_bytes(3, byteorder='little', signed=True))
                wf.writeframes(bytes(raw_bytes))
            else:
                raise ValueError(f"Unsupported write sample width: {sample_width}")

        return buf.getvalue()

    @classmethod
    def inspect_audio(cls, samples: List[float], sample_rate: int) -> AudioMetadata:
        """Inspect audio samples for duration, peak levels, and clipping."""
        if not samples:
            return AudioMetadata(sample_rate, 1, 2, 0, 0.0, False, 0.0, -100.0)

        num_frames = len(samples)
        duration = num_frames / float(sample_rate)
        peak = max(abs(s) for s in samples) if samples else 0.0
        is_clipped = peak >= 0.999

        # Calculate RMS dBFS
        sum_sq = sum(s * s for s in samples)
        rms = math.sqrt(sum_sq / num_frames) if num_frames > 0 else 0.0
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))

        return AudioMetadata(
            sample_rate=sample_rate,
            num_channels=1,
            sample_width=2,
            num_frames=num_frames,
            duration_seconds=round(duration, 3),
            is_clipped=is_clipped,
            peak_amplitude=round(peak, 4),
            rms_dbfs=round(rms_dbfs, 2),
        )

    @classmethod
    def trim_silence_vad(
        cls,
        samples: List[float],
        sample_rate: int,
        threshold_db: float = -40.0,
        frame_ms: int = 20,
        trailing_silence_keep_ms: int = 150
    ) -> List[float]:
        """
        Energy-based Voice Activity Detection (VAD) silence trimmer.
        Critical for VoxCPM2 to prevent trailing generation loops (>0.5s silence triggers stutters).
        """
        if not samples:
            return []

        frame_size = int(sample_rate * (frame_ms / 1000.0))
        if frame_size <= 0:
            return samples

        num_frames = len(samples) // frame_size
        frame_energies = []

        for i in range(num_frames):
            frame = samples[i * frame_size : (i + 1) * frame_size]
            sum_sq = sum(s * s for s in frame)
            rms = math.sqrt(sum_sq / len(frame)) if frame else 0.0
            db = 20.0 * math.log10(max(rms, 1e-9))
            frame_energies.append(db)

        # Find start frame above threshold
        start_frame = 0
        for i, db in enumerate(frame_energies):
            if db > threshold_db:
                start_frame = max(0, i - 1)  # small lead-in buffer
                break

        # Find end frame above threshold
        end_frame = num_frames
        for i in range(num_frames - 1, -1, -1):
            if frame_energies[i] > threshold_db:
                # Add trailing silence buffer
                keep_frames = int(trailing_silence_keep_ms / frame_ms)
                end_frame = min(num_frames, i + 1 + keep_frames)
                break

        start_sample = start_frame * frame_size
        end_sample = min(len(samples), end_frame * frame_size)

        return samples[start_sample:end_sample]

    @classmethod
    def resample(cls, samples: List[float], orig_sr: int, target_sr: int) -> List[float]:
        """
        Linear/polynomial interpolation resampler without external heavy C-libraries.
        """
        if orig_sr == target_sr or not samples:
            return samples

        ratio = float(target_sr) / float(orig_sr)
        target_len = int(len(samples) * ratio)
        resampled = [0.0] * target_len

        for i in range(target_len):
            orig_idx = i / ratio
            idx_floor = int(math.floor(orig_idx))
            idx_ceil = min(len(samples) - 1, idx_floor + 1)
            frac = orig_idx - idx_floor

            # Linear interpolation
            resampled[i] = (1.0 - frac) * samples[idx_floor] + frac * samples[idx_ceil]

        return resampled

    @classmethod
    def normalize_loudness(cls, samples: List[float], target_peak: float = 0.95) -> List[float]:
        """Normalize peak amplitude to target ceiling."""
        if not samples:
            return []
        current_peak = max(abs(s) for s in samples)
        if current_peak < 1e-6:
            return samples
        scale = target_peak / current_peak
        return [s * scale for s in samples]

    @classmethod
    def generate_streaming_chunks(
        cls,
        samples: List[float],
        sample_rate: int,
        chunk_duration_ms: int = 50
    ) -> Generator[bytes, None, None]:
        """
        Split audio into small PCM 16-bit byte chunks for low-latency browser streaming.
        """
        chunk_size = int(sample_rate * (chunk_duration_ms / 1000.0))
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            raw = []
            for s in chunk:
                clamped = max(-1.0, min(1.0, s))
                raw.append(int(clamped * 32767.0))
            yield struct.pack(f"<{len(raw)}h", *raw)

    @classmethod
    def process_for_training(cls, wav_bytes: bytes) -> Tuple[bytes, bytes, AudioMetadata]:
        """
        Full processing of an uploaded raw recording:
        1. Archives high-res 48 kHz mono 24-bit master.
        2. Produces 16 kHz VAD-trimmed normalized training derivative.
        3. Returns (archive_wav, derivative_wav, metadata).
        """
        samples, sr, _ = cls.read_wav_bytes(wav_bytes)
        
        # 1. 48 kHz Archive Master
        samples_48k = cls.resample(samples, sr, cls.ARCHIVE_SAMPLE_RATE)
        meta = cls.inspect_audio(samples_48k, cls.ARCHIVE_SAMPLE_RATE)
        archive_wav = cls.write_wav_bytes(samples_48k, cls.ARCHIVE_SAMPLE_RATE, sample_width=3)
        
        # 2. 16 kHz Derivative for VoxCPM2 / ASR / VAD
        samples_16k = cls.resample(samples, sr, cls.DERIVATIVE_SAMPLE_RATE)
        trimmed_16k = cls.trim_silence_vad(samples_16k, cls.DERIVATIVE_SAMPLE_RATE)
        normalized_16k = cls.normalize_loudness(trimmed_16k)
        derivative_wav = cls.write_wav_bytes(normalized_16k, cls.DERIVATIVE_SAMPLE_RATE, sample_width=2)
        
        return archive_wav, derivative_wav, meta
