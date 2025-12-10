"""
Download SEA-AD Data Script
---------------------------
Downloads the SEA-AD dataset from a public S3 bucket.
Usage:
    python scripts/download_sea_ad.py
"""

import os
import argparse
import sys
import requests
from tqdm import tqdm

# Hardcoded key from S3 listing (MTG dataset)
# Size: ~30GB
MTG_KEY = "DREAM/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad"
BUCKET_URL = "https://sea-ad-single-cell-profiling.s3.amazonaws.com/"

def download_file(key, dest_dir="data/raw"):
    url = f"{BUCKET_URL}{key}"
    filename = os.path.basename(key)
    dest_path = os.path.join(dest_dir, filename)
    
    os.makedirs(dest_dir, exist_ok=True)
    
    if os.path.exists(dest_path):
        print(f"File {dest_path} already exists. Skipping download.")
        return dest_path

    print(f"WARNING: You are about to download a very large file (~30GB).")
    print(f"Target: {url}")
    print(f"Destination: {dest_path}")
    
    confirm = input("Do you want to proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return None

    print(f"Starting download...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                    size = f.write(chunk)
                    bar.update(size)
        print("Download complete.")
        return dest_path
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download SEA-AD MTG Data")
    # No args needed really for this specific task, but good practice
    args = parser.parse_args()
    
    download_file(MTG_KEY)

if __name__ == "__main__":
    main()
