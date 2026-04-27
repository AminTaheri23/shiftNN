"""Download WikiText-2 raw to artifacts/wikitext-2/.

Run from project root:  python3 training/download_wikitext2.py
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


URLS = [
    # Original Salesforce / Smerity mirror.
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
]

DEST_DIR = Path("artifacts")


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DEST_DIR / "wikitext-2-v1.zip"

    if not zip_path.exists():
        ok = False
        for url in URLS:
            try:
                print(f"Downloading {url} ...")
                urllib.request.urlretrieve(url, zip_path)
                print(f"  saved to {zip_path} ({zip_path.stat().st_size:,} bytes)")
                ok = True
                break
            except Exception as e:
                print(f"  failed: {e}")
        if not ok:
            print("\nAll download URLs failed. Manual download options:")
            print("  curl -L -o artifacts/wikitext-2-v1.zip "
                  "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip")
            print("  # or fetch from HuggingFace mirror via `datasets` library")
            sys.exit(1)

    extracted = DEST_DIR / "wikitext-2"
    if not extracted.exists():
        print(f"Extracting to {DEST_DIR}/...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DEST_DIR)

    if not extracted.exists():
        print(f"Expected {extracted} after unzip; got:")
        for p in DEST_DIR.iterdir():
            print(f"  {p}")
        sys.exit(1)

    print(f"\nFiles in {extracted}:")
    for f in sorted(extracted.iterdir()):
        print(f"  {f.name:<25} {f.stat().st_size:>12,} bytes")


if __name__ == "__main__":
    main()
