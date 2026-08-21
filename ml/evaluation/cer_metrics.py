"""
Central Kurdish CER (Character Error Rate) & WER Evaluation Engine.
Computes normalized Levenshtein edit distance for Kurdish ASR & TTS quality QA.
"""

from typing import Dict, List, Tuple
from packages.ckb_frontend import SoraniNormalizer

normalizer = SoraniNormalizer()


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Standard dynamic programming Levenshtein distance."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1]   # Substitution
                )

    return dp[m][n]


def compute_cer(reference_text: str, hypothesis_text: str) -> float:
    """
    Compute Character Error Rate (CER) after Kurdish Unicode normalization.
    """
    ref_norm = normalizer.normalize(reference_text).replace(" ", "")
    hyp_norm = normalizer.normalize(hypothesis_text).replace(" ", "")

    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    dist = levenshtein_distance(list(ref_norm), list(hyp_norm))
    return round(dist / float(len(ref_norm)), 4)


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    """
    Compute Word Error Rate (WER) after Kurdish normalization.
    """
    ref_words = normalizer.normalize(reference_text).split()
    hyp_words = normalizer.normalize(hypothesis_text).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return round(dist / float(len(ref_words)), 4)
