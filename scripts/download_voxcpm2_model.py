"""
Parallel high-speed downloader for OpenBMB VoxCPM2 model weights (4.58 GB safetensors + 376 MB audiovae).
Downloads directly into data/models/VoxCPM2/.
"""

import os
import sys
import time
from huggingface_hub import hf_hub_download

def download_voxcpm2():
    repo_id = "openbmb/VoxCPM2"
    local_dir = os.path.abspath(r"data\models\VoxCPM2")
    os.makedirs(local_dir, exist_ok=True)
    
    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenization_voxcpm2.py",
        "audiovae.pth",
        "model.safetensors"
    ]
    
    print("=" * 70)
    print(f" DOWNLOADING OpenBMB VoxCPM2 FOUNDATION WEIGHTS TO: {local_dir}")
    print("=" * 70)
    
    for filename in files:
        dest_path = os.path.join(local_dir, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f" [SKIP] {filename} already exists ({os.path.getsize(dest_path) / (1024**2):.1f} MB)")
            continue
            
        print(f" [DOWNLOAD] Downloading {filename} from {repo_id}...")
        t0 = time.time()
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        elapsed = time.time() - t0
        size_mb = os.path.getsize(downloaded) / (1024**2)
        print(f" [DONE] {filename} downloaded: {size_mb:.1f} MB in {elapsed:.1f}s ({size_mb/max(elapsed, 0.1):.1f} MB/s)")
        
    print("\nAll VoxCPM2 foundation weights downloaded and ready!")

if __name__ == "__main__":
    download_voxcpm2()
