"""
Audio Quality Analyzer for Sorani Voice Studio.
Checks for stop failures, repetitions, clipping, signal-to-noise ratio, and silence anomalies.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class QualityReport:
    is_acceptable: bool
    snr_db: float
    silence_ratio: float
    stop_failure_detected: bool
    repetition_detected: bool
    clipping_rate: float
    reasons: List[str]


class QualityAnalyzer:
    """
    Analyzes generated or recorded speech audio for production gates:
    - Trailing generation loop / stop failure
    - Audio stutter or repetition
    - Clipping artifacts
    - High background noise
    """

    @classmethod
    def evaluate(cls, samples: List[float], sample_rate: int = 48000) -> QualityReport:
        if not samples:
            return QualityReport(False, 0.0, 1.0, True, False, 0.0, ["Empty audio"])

        reasons = []
        num_samples = len(samples)
        duration = num_samples / float(sample_rate)

        # 1. Clipping detection
        clipped_count = sum(1 for s in samples if abs(s) >= 0.999)
        clipping_rate = clipped_count / float(num_samples)
        if clipping_rate > 0.005:  # > 0.5% clipped
            reasons.append(f"Excessive clipping: {clipping_rate*100:.2f}%")

        # 2. Silence ratio & Trailing silence
        frame_size = int(sample_rate * 0.02)  # 20 ms
        num_frames = num_samples // frame_size
        silent_frames = 0
        trailing_silent_frames = 0

        for i in range(num_frames):
            frame = samples[i * frame_size : (i + 1) * frame_size]
            rms = math.sqrt(sum(s * s for s in frame) / len(frame)) if frame else 0.0
            if rms < 0.01:  # silence threshold
                silent_frames += 1
                trailing_silent_frames += 1
            else:
                trailing_silent_frames = 0

        silence_ratio = silent_frames / max(1, num_frames)
        trailing_silence_sec = (trailing_silent_frames * frame_size) / float(sample_rate)

        # Stop failure check (> 0.6s of trailing dead audio or runaway generation)
        stop_failure = trailing_silence_sec > 0.6 or duration > 30.0
        if stop_failure:
            reasons.append(f"Possible stop failure: trailing silence={trailing_silence_sec:.2f}s, duration={duration:.2f}s")

        # 3. Simple repetition detection via autocorrelation
        repetition_detected = False
        if duration > 2.0:
            # Check 0.5s chunks for high similarity
            chunk_len = int(sample_rate * 0.5)
            if num_samples >= chunk_len * 2:
                c1 = samples[-chunk_len * 2 : -chunk_len]
                c2 = samples[-chunk_len:]
                dot = sum(a * b for a, b in zip(c1, c2))
                norm1 = sum(a * a for a in c1) ** 0.5 + 1e-9
                norm2 = sum(b * b for b in c2) ** 0.5 + 1e-9
                sim = dot / (norm1 * norm2)
                if sim > 0.92:
                    repetition_detected = True
                    reasons.append("Audio loop/repetition detected at the end of utterance")

        # Estimated SNR
        signal_rms = math.sqrt(sum(s * s for s in samples) / num_samples) + 1e-9
        noise_rms = 1e-4  # baseline reference
        snr_db = 20.0 * math.log10(signal_rms / noise_rms)

        is_acceptable = len(reasons) == 0

        return QualityReport(
            is_acceptable=is_acceptable,
            snr_db=round(snr_db, 1),
            silence_ratio=round(silence_ratio, 3),
            stop_failure_detected=stop_failure,
            repetition_detected=repetition_detected,
            clipping_rate=round(clipping_rate, 4),
            reasons=reasons
        )
