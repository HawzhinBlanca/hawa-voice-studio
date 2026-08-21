"""
Prepares Kurdish Sorani fine-tuning dataset manifests for official OpenBMB VoxCPM2 training.
Converts manifest into JSONL format with exact 'audio' and 'text' keys.
"""

import json
import os
import sys

def prepare_voxcpm2_manifests():
    input_manifest = "data/finetune/pilot_25k_manifest.jsonl"
    train_output = "data/finetune/voxcpm2_train_manifest.jsonl"
    val_output = "data/finetune/voxcpm2_val_manifest.jsonl"
    
    entries = []
    with open(input_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
                
    print(f"Total entries loaded: {len(entries):,}")
    
    # Format entries
    formatted = []
    for e in entries:
        wav_path = os.path.abspath(e["audio_path"])
        text = e.get("normalized_text", e.get("raw_text", "")).strip()
        if os.path.exists(wav_path) and len(text) >= 3:
            formatted.append({
                "audio": wav_path,
                "text": text,
                "dataset_id": 0
            })
            
    print(f"Valid formatted samples: {len(formatted):,}")
    
    # 95% Train / 5% Val Split
    split_idx = int(len(formatted) * 0.95)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]
    
    with open(train_output, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(val_output, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Train manifest written: {train_output} ({len(train_data):,} samples)")
    print(f"Validation manifest written: {val_output} ({len(val_data):,} samples)")

if __name__ == "__main__":
    prepare_voxcpm2_manifests()
