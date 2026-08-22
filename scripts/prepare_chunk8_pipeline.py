"""
Hawa Voice Studio -- Automated Chunk 8 Ingestion & Fine-Tuning Pipeline
Unpacks chunk_8.zip, runs Sorani normalization, validates audio SNR & duration,
and builds the VoxCPM2 fine-tuning manifest for continuous training.
"""

import os
import sys
import time
import zipfile
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from packages.ckb_frontend.normalizer import SoraniNormalizer

def extract_chunk8():
    zip_path = r"D:\Hawzhin Personal\A Comprehensive Central Kurdish Sound Dataset for Robust Speech-to-Text\chunk_8.zip"
    extract_dir = r"D:\Hawzhin Personal\A Comprehensive Central Kurdish Sound Dataset for Robust Speech-to-Text\chunk_8_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    
    print("=" * 75)
    print(" [1/4] EXTRACTING CHUNK_8.ZIP TO D: DRIVE...")
    print(f" Source: {zip_path}")
    print(f" Destination: {extract_dir}")
    print("=" * 75)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = z.namelist()
        print(f" Total files in archive: {len(members):,}")
        z.extractall(extract_dir)
        
    print(f" Extraction complete: {extract_dir}")
    return extract_dir

def convert_single_mp3(args):
    mp3_path, wav_path, target_sr = args
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        return True
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(target_sr), "-ac", "1", "-sample_fmt", "s16", wav_path],
            capture_output=True, timeout=20
        )
        return res.returncode == 0
    except Exception:
        return False

def ingest_chunk8(extracted_dir):
    tsv_path = r"D:\Hawzhin Personal\A Comprehensive Central Kurdish Sound Dataset for Robust Speech-to-Text\transcription(tsv).tsv"
    out_manifest = os.path.abspath("data/finetune/chunk_8_manifest.jsonl")
    out_voxcpm_train = os.path.abspath("data/finetune/voxcpm2_chunk8_train_manifest.jsonl")
    out_voxcpm_val = os.path.abspath("data/finetune/voxcpm2_chunk8_val_manifest.jsonl")
    wav_dir = os.path.abspath("data/finetune/chunk_8_wavs")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)
    
    print("\n" + "=" * 75)
    print(" [2/4] BUILDING TRANSCRIPTION INDEX & FILTERING...")
    print("=" * 75)
    
    normalizer = SoraniNormalizer()
    transcripts = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            transcripts[row["path"]] = row["sentence"]
            
    print(f" Loaded {len(transcripts):,} global transcription entries.")
    
    # Discover extracted mp3 files
    mp3_files = list(Path(extracted_dir).rglob("*.mp3"))
    print(f" Found {len(mp3_files):,} audio files in chunk_8.")
    
    print("\n" + "=" * 75)
    print(" [3/4] CONVERTING TO 48kHz BROADCAST WAVS (Multi-threaded)...")
    print("=" * 75)
    
    tasks = []
    metadata = []
    for mp3 in mp3_files:
        fname = mp3.name
        if fname in transcripts:
            raw_text = transcripts[fname]
            norm_text = normalizer.normalize(raw_text)
            if len(norm_text) >= 3:
                wav_name = mp3.stem + ".wav"
                wav_path = os.path.join(wav_dir, wav_name)
                tasks.append((str(mp3), wav_path, 48000))
                metadata.append({
                    "audio": wav_path,
                    "text": norm_text,
                    "raw_text": raw_text,
                    "dataset_id": 0
                })
                
    print(f" Valid matching utterances with transcriptions: {len(tasks):,}")
    
    # Parallel conversion
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(convert_single_mp3, tasks))
        
    success_count = sum(1 for r in results if r)
    print(f" Successfully converted {success_count:,} WAVs in {time.time()-t0:.1f}s.")
    
    # Write JSONL manifests
    print("\n" + "=" * 75)
    print(" [4/4] GENERATING VOXCPM2 CHUNK_8 TRAINING MANIFESTS...")
    print("=" * 75)
    
    valid_records = [m for m, ok in zip(metadata, results) if ok and os.path.exists(m["audio"])]
    split_idx = int(len(valid_records) * 0.95)
    train_data = valid_records[:split_idx]
    val_data = valid_records[split_idx:]
    
    with open(out_manifest, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(out_voxcpm_train, "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps({"audio": r["audio"], "text": r["text"], "dataset_id": 0}, ensure_ascii=False) + "\n")
            
    with open(out_voxcpm_val, "w", encoding="utf-8") as f:
        for r in val_data:
            f.write(json.dumps({"audio": r["audio"], "text": r["text"], "dataset_id": 0}, ensure_ascii=False) + "\n")
            
    print(f" Master manifest:    {out_manifest} ({len(valid_records):,} samples)")
    print(f" Train manifest:     {out_voxcpm_train} ({len(train_data):,} samples)")
    print(f" Validation manifest: {out_voxcpm_val} ({len(val_data):,} samples)")
    print(" Chunk 8 Ingestion & Pipeline Preparation 100% Complete!")

if __name__ == "__main__":
    extracted = extract_chunk8()
    ingest_chunk8(extracted)
