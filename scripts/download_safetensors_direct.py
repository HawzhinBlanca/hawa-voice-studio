"""
Direct chunked stream downloader with resume support and live speed monitoring for VoxCPM2 model.safetensors (4.58 GB).
"""

import os
import sys
import time
import requests

def download_model_safetensors():
    url = "https://huggingface.co/openbmb/VoxCPM2/resolve/main/model.safetensors"
    dest_path = os.path.abspath(r"data\models\VoxCPM2\model.safetensors")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    headers = {}
    downloaded_bytes = 0
    
    if os.path.exists(dest_path):
        downloaded_bytes = os.path.getsize(dest_path)
        if downloaded_bytes > 4_500_000_000:
            print(f"[EXISTS] model.safetensors already complete ({downloaded_bytes / (1024**3):.2f} GB)")
            return
        elif downloaded_bytes > 0:
            headers["Range"] = f"bytes={downloaded_bytes}-"
            print(f"[RESUME] Resuming model.safetensors from {downloaded_bytes / (1024**2):.1f} MB...")
            
    print(f"Connecting to: {url}")
    t0 = time.time()
    last_print = t0
    
    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0)) + downloaded_bytes
        total_gb = total_size / (1024**3)
        print(f"Total Size: {total_gb:.2f} GB ({total_size:,} bytes)")
        
        mode = "ab" if downloaded_bytes > 0 else "wb"
        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    
                    now = time.time()
                    if now - last_print >= 5.0:
                        elapsed = now - t0
                        speed_mbs = (downloaded_bytes / (1024**2)) / max(elapsed, 0.1)
                        pct = (downloaded_bytes / total_size) * 100.0 if total_size > 0 else 0
                        eta_sec = (total_size - downloaded_bytes) / max(speed_mbs * 1024 * 1024, 1)
                        print(
                            f" [PROGRESS] {downloaded_bytes / (1024**3):.2f} / {total_gb:.2f} GB "
                            f"({pct:.1f}%) | Speed: {speed_mbs:.1f} MB/s | ETA: {eta_sec:.0f}s"
                        )
                        last_print = now
                        
    total_time = time.time() - t0
    print(f"\n[COMPLETE] model.safetensors downloaded in {total_time:.1f}s ({os.path.getsize(dest_path)/(1024**3):.2f} GB)")

if __name__ == "__main__":
    download_model_safetensors()
