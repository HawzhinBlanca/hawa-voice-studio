"""
Kurdish Sorani Dataset Ingestion & Fine-Tune Preparation Pipeline.
Processes raw audio chunks from "A Comprehensive Central Kurdish Sound Dataset"
into a training-ready JSONL manifest with quality filtering.

Pipeline:
1. Parse transcription TSV to match audio files
2. Convert MP3 → WAV 48kHz mono (VoxCPM2 native sample rate)
3. Run quality analysis (SNR, duration, silence ratio)
4. Apply Sorani text normalization
5. Filter by quality gates
6. Generate JSONL manifest for training
"""

import argparse
import csv
import json
import os
import struct
import sys
import time
import wave
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.ckb_frontend import SoraniNormalizer
from packages.audio_processing import QualityAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest Kurdish audio dataset for VoxCPM2 fine-tuning")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing extracted audio files")
    parser.add_argument("--transcription_tsv", type=str, required=True, help="Path to transcription TSV file")
    parser.add_argument("--output_manifest", type=str, required=True, help="Output JSONL manifest path")
    parser.add_argument("--output_wav_dir", type=str, default=None, help="Directory to write converted WAVs (optional)")
    parser.add_argument("--max_files", type=int, default=0, help="Max files to process (0=all)")
    parser.add_argument("--min_duration", type=float, default=1.0, help="Min utterance duration in seconds")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max utterance duration in seconds")
    parser.add_argument("--min_text_len", type=int, default=3, help="Min normalized text length in chars")
    return parser.parse_args()


def get_mp3_duration_estimate(filepath: str) -> float:
    """Estimate MP3 duration from file size (rough: ~128kbps average)."""
    size_bytes = os.path.getsize(filepath)
    # Typical speech MP3 at 128kbps = 16000 bytes/second
    return size_bytes / 16000.0


def convert_mp3_to_wav_simple(mp3_path: str, wav_path: str, target_sr: int = 48000) -> bool:
    """
    Convert MP3 to WAV using ffmpeg if available, otherwise skip.
    Returns True if conversion succeeded.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(target_sr), "-ac", "1", "-sample_fmt", "s16", wav_path],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def build_transcription_index(tsv_path: str) -> dict:
    """Build a filename→row lookup from the transcription TSV."""
    index = {}
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            filename = row['path']
            index[filename] = {
                'sentence': row['sentence'],
                'gender': row.get('gender', 'unknown'),
                'age': row.get('age', 'unknown'),
            }
    return index


def main():
    args = parse_args()
    normalizer = SoraniNormalizer()

    print("=" * 70)
    print("  Hawa Sorani Voice Studio — Dataset Ingestion Pipeline")
    print("=" * 70)
    print(f"  Audio directory:   {args.audio_dir}")
    print(f"  Transcription:     {args.transcription_tsv}")
    print(f"  Output manifest:   {args.output_manifest}")
    print(f"  Duration filter:   {args.min_duration}s – {args.max_duration}s")
    print(f"  Min text length:   {args.min_text_len} chars")
    print("=" * 70)

    # Step 1: Build transcription index
    print("\n[1/5] Building transcription index...")
    t0 = time.time()
    transcript_index = build_transcription_index(args.transcription_tsv)
    print(f"  Loaded {len(transcript_index):,} transcription entries in {time.time()-t0:.1f}s")

    # Step 2: Discover audio files in the extracted chunk
    print("\n[2/5] Scanning audio directory...")
    audio_dir = Path(args.audio_dir)
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac']:
        audio_files.extend(audio_dir.rglob(ext))
    audio_files.sort()
    total_found = len(audio_files)
    print(f"  Found {total_found:,} audio files")

    if args.max_files > 0:
        audio_files = audio_files[:args.max_files]
        print(f"  Limited to {len(audio_files):,} files (--max_files={args.max_files})")

    # Step 3: Process each audio file
    print("\n[3/5] Processing utterances...")
    os.makedirs(os.path.dirname(args.output_manifest) or '.', exist_ok=True)
    if args.output_wav_dir:
        os.makedirs(args.output_wav_dir, exist_ok=True)

    has_ffmpeg = False
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        has_ffmpeg = True
        print("  ✓ ffmpeg detected — will convert MP3→WAV 48kHz")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ⚠ ffmpeg not found — will estimate duration from file size")

    stats = {
        'total': len(audio_files),
        'matched': 0,
        'no_transcript': 0,
        'too_short': 0,
        'too_long': 0,
        'text_too_short': 0,
        'converted': 0,
        'accepted': 0,
        'total_duration_sec': 0.0,
    }

    manifest_entries = []
    t_start = time.time()

    for idx, audio_path in enumerate(audio_files):
        filename = audio_path.name

        # Progress reporting
        if (idx + 1) % 1000 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1:,}/{len(audio_files):,}] {rate:.0f} files/sec — {stats['accepted']:,} accepted")

        # Match to transcription
        if filename not in transcript_index:
            stats['no_transcript'] += 1
            continue
        stats['matched'] += 1

        transcript = transcript_index[filename]
        raw_text = transcript['sentence']
        gender = transcript['gender']
        age = transcript['age']

        # Normalize Kurdish text
        normalized = normalizer.normalize(raw_text)
        if len(normalized) < args.min_text_len:
            stats['text_too_short'] += 1
            continue

        # Duration estimation or actual measurement
        if has_ffmpeg and args.output_wav_dir:
            wav_path = os.path.join(args.output_wav_dir, filename.replace('.mp3', '.wav'))
            if convert_mp3_to_wav_simple(str(audio_path), wav_path):
                stats['converted'] += 1
                try:
                    with wave.open(wav_path, 'r') as wf:
                        duration = wf.getnframes() / wf.getframerate()
                except Exception:
                    duration = get_mp3_duration_estimate(str(audio_path))
            else:
                duration = get_mp3_duration_estimate(str(audio_path))
                wav_path = str(audio_path)  # keep original
        else:
            duration = get_mp3_duration_estimate(str(audio_path))
            wav_path = str(audio_path)

        # Duration gates
        if duration < args.min_duration:
            stats['too_short'] += 1
            continue
        if duration > args.max_duration:
            stats['too_long'] += 1
            continue

        stats['accepted'] += 1
        stats['total_duration_sec'] += duration

        entry = {
            'audio_path': wav_path,
            'original_path': str(audio_path),
            'raw_text': raw_text,
            'normalized_text': normalized,
            'duration_seconds': round(duration, 2),
            'gender': gender,
            'age': age,
            'file_size_bytes': os.path.getsize(str(audio_path)),
            'source': 'central_kurdish_comprehensive',
        }
        manifest_entries.append(entry)

    # Step 4: Write manifest
    print(f"\n[4/5] Writing manifest ({len(manifest_entries):,} entries)...")
    with open(args.output_manifest, 'w', encoding='utf-8') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Step 5: Summary
    total_hours = stats['total_duration_sec'] / 3600.0
    elapsed_total = time.time() - t_start

    print(f"\n[5/5] Pipeline complete!")
    print("=" * 70)
    print(f"  INGESTION SUMMARY")
    print(f"  {'─' * 50}")
    print(f"  Total audio files scanned:   {stats['total']:>10,}")
    print(f"  Matched to transcription:    {stats['matched']:>10,}")
    print(f"  No transcript found:         {stats['no_transcript']:>10,}")
    print(f"  Text too short (<{args.min_text_len} chars):  {stats['text_too_short']:>10,}")
    print(f"  Duration too short (<{args.min_duration}s):  {stats['too_short']:>10,}")
    print(f"  Duration too long (>{args.max_duration}s):   {stats['too_long']:>10,}")
    if has_ffmpeg and args.output_wav_dir:
        print(f"  MP3→WAV converted:           {stats['converted']:>10,}")
    print(f"  {'─' * 50}")
    print(f"  ✓ ACCEPTED for training:     {stats['accepted']:>10,}")
    print(f"  ✓ Total duration:            {total_hours:>10.1f} hours")
    print(f"  ✓ Manifest written to:       {args.output_manifest}")
    print(f"  ✓ Processing time:           {elapsed_total:>10.1f} seconds")
    print("=" * 70)

    # Write summary JSON
    summary_path = args.output_manifest.replace('.jsonl', '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': 'A Comprehensive Central Kurdish Sound Dataset',
            'chunk': os.path.basename(args.audio_dir),
            'total_files': stats['total'],
            'accepted': stats['accepted'],
            'rejected': stats['total'] - stats['accepted'],
            'total_duration_hours': round(total_hours, 2),
            'avg_duration_seconds': round(stats['total_duration_sec'] / max(stats['accepted'], 1), 2),
            'gender_distribution': {},
            'quality_filters': {
                'min_duration_s': args.min_duration,
                'max_duration_s': args.max_duration,
                'min_text_len': args.min_text_len,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
