"""
download_models.py — Download the GorillaIdentifier models from the HuggingFace Hub.

The weights are not committed to the Git repository (too large). They live at
https://huggingface.co/tit0000/GorillaIdentifier

Usage:
    python models/download_models.py                # detector only (needed to extract crops)
    python models/download_models.py --version all  # detector + trained model + app assets
"""

import sys
import argparse
from pathlib import Path

# Bootstrap config_loader so the HF/Torch cache is read from config.yaml
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.config_loader import apply_cache_env, REPO_ROOT
apply_cache_env()  # sets HF_HOME / TORCH_HOME before huggingface_hub import

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERR] huggingface_hub not installed")
    print("      pip install huggingface_hub")
    sys.exit(1)

# ==============================================================================
# MODEL CATALOGUE
# ==============================================================================
REPO_ID = "tit0000/GorillaIdentifier"

MODELS = {
    "yolo_gorilla": {
        "file":    "yolo_gorilla.pt",
        "dest":    "models/yolo_gorilla.pt",
        "desc":    "Gorilla face detector (YOLOv8), required to extract crops",
        "size_mb": 18,
    },
    "identifier": {
        "file":    "gorilla_v1_best.pt",
        "dest":    "output/v1_gorilla/best.pt",
        "desc":    "Trained V1 identifier checkpoint (skip retraining, or re-export)",
        "size_mb": 105,
    },
    "gallery": {
        "file":    "gallery.json",
        "dest":    "output/v1_gorilla/gallery.json",
        "desc":    "Identity gallery, 66 individuals",
        "size_mb": 30,
    },
    "backbone_tflite": {
        "file":    "megadesc_T_arcface_backbone.tflite",
        "dest":    "output/v1_gorilla/tflite/megadesc_T_arcface_backbone.tflite",
        "desc":    "Identifier backbone in TFLite (for the Android app)",
        "size_mb": 107,
    },
    "detector_tflite": {
        "file":    "yolo_v2_detector.tflite",
        "dest":    "output/v1_gorilla/tflite/yolo_v2_detector.tflite",
        "desc":    "Gorilla face detector in TFLite (for the Android app)",
        "size_mb": 6,
    },
}

VERSION_MAP = {
    "detector": ["yolo_gorilla"],                 # default: the minimum needed to extract crops
    "all":      list(MODELS.keys()),
}


def download(key: str, dry: bool = False):
    m    = MODELS[key]
    dest = REPO_ROOT / m["dest"]
    if dest.exists():
        print(f"  [OK] {m['dest']} already exists")
        return
    print(f"  Downloading {m['desc']} (~{m['size_mb']} MB)...")
    if dry:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id   = REPO_ID,
            filename  = m["file"],
            local_dir = str(dest.parent),
        )
        # hf_hub_download keeps the original filename; rename if dest differs
        got = Path(path)
        if got.name != dest.name:
            got.rename(dest)
        print(f"  [OK] {dest}")
    except Exception as e:
        print(f"  [ERR] {e}")
        print(f"  Download manually from:")
        print(f"  https://huggingface.co/{REPO_ID}/resolve/main/{m['file']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="detector",
                        choices=list(VERSION_MAP.keys()),
                        help="Which set to download (default: detector)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keys = VERSION_MAP[args.version]
    total_mb = sum(MODELS[k]["size_mb"] for k in keys)
    print(f"  Models to download: {len(keys)} ({total_mb} MB estimated)")
    print(f"  Repo : {REPO_ID}")
    print()

    for k in keys:
        download(k, dry=args.dry_run)

    print()
    print("  Direct links if a download fails:")
    print(f"  https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
