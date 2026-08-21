"""
Prepare Stage 1 Architecture Pilot Dataset (25,000 Utterances ~ 40 Hours)
from chunk_7 for VoxCPM2 LoRA Fine-Tuning.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SOURCE_MANIFEST = os.path.join("data", "finetune", "chunk_7_manifest.jsonl")
OUTPUT_MANIFEST = os.path.join("data", "finetune", "pilot_25k_manifest.jsonl")
WAV_DIR = os.path.join("data", "finetune", "pilot_25k_wavs")
SAMPLE_COUNT = 25000


def convert_file(args):
    src_mp3, dst_wav = args
    if os.path.exists(dst_wav) and os.path.getsize(dst_wav) > 1000:
        return True
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-i", src_mp3, "-ar", "48000", "-ac", "1", "-sample_fmt", "s16", dst_wav],
            capture_output=True, timeout=10
        )
        return res.returncode == 0
    except Exception:
        return False


def main():
    print("=" * 70)
    print(" Hawa Sorani Voice Studio — Preparing Stage 1 Pilot (25,000 Utterances)")
    print("=" * 70)

    if not os.path.exists(SOURCE_MANIFEST):
        print(f"Error: {SOURCE_MANIFEST} not found!")
        sys.exit(1)

    os.makedirs(WAV_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_MANIFEST), exist_ok=True)

    # Read 25,000 candidate entries
    entries = []
    with open(SOURCE_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
                if len(entries) >= SAMPLE_COUNT:
                    break

    print(f"Loaded {len(entries):,} candidate utterances from chunk_7.")

    # Prepare conversion tasks
    conversion_tasks = []
    converted_entries = []

    for e in entries:
        src_audio = e["original_path"]
        fname = Path(src_audio).stem + ".wav"
        dst_wav = os.path.join(WAV_DIR, fname)

        conversion_tasks.append((src_audio, dst_wav))

        new_entry = dict(e)
        new_entry["audio_path"] = dst_wav
        new_entry["sample_rate"] = 48000
        new_entry["channels"] = 1
        converted_entries.append(new_entry)

    print(f"Converting {len(conversion_tasks):,} MP3s to 48kHz mono WAV in parallel (16 threads)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(convert_file, conversion_tasks))

    success_count = sum(1 for r in results if r)
    elapsed = time.time() - t0
    print(f"Converted {success_count:,}/{len(conversion_tasks):,} files in {elapsed:.1f}s ({success_count/max(elapsed, 0.001):.1f} files/sec).")

    # Filter only successfully converted
    final_entries = [entry for entry, ok in zip(converted_entries, results) if ok]

    # Write sliced manifest
    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        for e in final_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    total_duration_sec = sum(e["duration_seconds"] for e in final_entries)
    total_hours = total_duration_sec / 3600.0

    print(f"\n✓ Stage 1 Pilot Manifest saved to: {OUTPUT_MANIFEST}")
    print(f"  Utterances: {len(final_entries):,}")
    print(f"  Total Duration: {total_hours:.2f} hours ({total_duration_sec:,.1f} seconds)")
    print(f"  Avg Utterance Duration: {total_duration_sec/len(final_entries):.2f}s")
    print(f"  Audio directory: {WAV_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
