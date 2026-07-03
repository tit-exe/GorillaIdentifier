"""
config_loader.py — Single source of truth for all paths and settings.
======================================================================
Every script in the pipeline imports from here instead of hardcoding paths.

Usage in any script:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.config_loader import (
        REPO_ROOT, PHOTOS_DIR, CROPS_JSON, MODELS_DIR, apply_cache_env
    )
    apply_cache_env()   # sets HF_HOME and TORCH_HOME — call before heavy imports

Design notes:
- No side effects at import time (no os.environ calls).
- apply_cache_env() must be called explicitly before importing timm / huggingface_hub.
- All paths are resolved as absolute at import time so scripts can run from any cwd.
- config.yaml is the only file a user needs to edit to adapt to a new machine.
"""

import os
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root — always the directory containing this file's parent (common/)
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).parent.parent.resolve()

# ---------------------------------------------------------------------------
# Load config.yaml
# ---------------------------------------------------------------------------
_cfg_path = REPO_ROOT / "config.yaml"
if not _cfg_path.exists():
    raise FileNotFoundError(
        f"config.yaml not found at {_cfg_path}\n"
        "Make sure you run scripts from the repo root or any subdirectory."
    )

with open(_cfg_path, encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Cache directories
# HF/Torch caches default to OS standard locations if not set in config.yaml.
# Override by uncommenting hf_cache_dir / torch_cache_dir in config.yaml.
# ---------------------------------------------------------------------------
HF_CACHE: Path = Path(
    _cfg.get("hf_cache_dir", "~/.cache/huggingface")
).expanduser().resolve()

TORCH_CACHE: Path = Path(
    _cfg.get("torch_cache_dir", "~/.cache/torch")
).expanduser().resolve()


def apply_cache_env() -> None:
    """
    Set HF_HOME and TORCH_HOME environment variables.

    Call this BEFORE importing timm, huggingface_hub, transformers, or ultralytics.
    Must be explicit — no silent side effects at import time.

    Example:
        from common.config_loader import apply_cache_env
        apply_cache_env()
        import timm  # will now use the configured cache
    """
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["TORCH_HOME"] = str(TORCH_CACHE)
    os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")


# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
SPECIES: str      = _cfg.get("species", "Unknown species")
PROJECT_NAME: str = _cfg.get("project_name", "project")

# ---------------------------------------------------------------------------
# Data paths (all relative to REPO_ROOT, resolved to absolute here)
# ---------------------------------------------------------------------------

# Input: raw photos organized by individual name
PHOTOS_DIR: Path = REPO_ROOT / _cfg.get("raw_photos_dir", "data/photos")

# Input: unlabeled internet images for background/wild class
WILD_IMAGES_DIR: Path = REPO_ROOT / _cfg.get("wild_images_dir", "data/wild_images/raw")

# Output root for training results, exports, etc.
OUTPUT_DIR: Path = REPO_ROOT / _cfg.get("output_dir", "output")

# Crops — 224x224 face crops extracted by YOLO
CROPS_DIR: Path       = REPO_ROOT / "data/crops"
CROPS_KNOWN_DIR: Path = CROPS_DIR / "known"   # labeled individuals (zoo, BOS, field)
CROPS_WILD_DIR: Path  = CROPS_DIR / "wild"    # unlabeled wild crops (background class)

# Unified JSON tracking all crops from all pipeline versions
CROPS_JSON: Path = REPO_ROOT / "data/crops.json"

# ---------------------------------------------------------------------------
# Model paths (not in git — download from Zenodo / see models/download_models.py)
# ---------------------------------------------------------------------------
MODELS_DIR: Path = REPO_ROOT / "models"

YOLO_GORILLA_PT: Path = MODELS_DIR / "yolo_gorilla.pt"

# ---------------------------------------------------------------------------
# YOLO parameters
# ---------------------------------------------------------------------------
YOLO_CONFIDENCE: float  = float(_cfg.get("yolo_confidence", 0.30))
YOLO_FACE_MARGIN: float = float(_cfg.get("yolo_face_margin", 0.05))

# ---------------------------------------------------------------------------
# ArcFace / MegaDescriptor training parameters
# ---------------------------------------------------------------------------
ARC_SCALE: int        = int(_cfg.get("arc_scale", 64))
ARC_MARGIN: float     = float(_cfg.get("arc_margin", 0.50))
MAX_EPOCHS: int       = int(_cfg.get("max_epochs", 100))
PATIENCE: int         = int(_cfg.get("patience", 25))
PATIENCE_START: int   = int(_cfg.get("patience_start", 35))
LR_BACKBONE: float    = float(_cfg.get("lr_backbone", 1e-5))
LR_HEAD: float        = float(_cfg.get("lr_head", 5e-4))
BATCH_SIZE: int       = int(_cfg.get("batch_size", 32))
DEVICE: str           = _cfg.get("device", "auto")

# ---------------------------------------------------------------------------
# Open-set gallery
# ---------------------------------------------------------------------------
UNKNOWN_THRESHOLD: float = float(_cfg.get("unknown_threshold", 0.22))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*dirs: Path) -> None:
    """Create directories if they don't exist. Safe to call multiple times."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def resolve_path(p: str | Path, base: Path = REPO_ROOT) -> Path:
    """
    Resolve a path that may be relative (stored in JSON) or absolute.
    Relative paths are resolved against REPO_ROOT (or a given base).
    Returns an absolute Path.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def to_relative(p: str | Path, base: Path = REPO_ROOT) -> str:
    """
    Convert an absolute path to a POSIX-style relative path from REPO_ROOT.
    Used when writing paths into crops.json so the file is portable.
    Returns a forward-slash string (works on Windows and Linux).
    """
    p = Path(p).resolve()
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        # Path is outside the repo (e.g. absolute path on a different drive)
        return str(p)


# ---------------------------------------------------------------------------
# Quick self-test (python common/config_loader.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"REPO_ROOT        : {REPO_ROOT}")
    print(f"PHOTOS_DIR       : {PHOTOS_DIR}")
    print(f"WILD_IMAGES_DIR  : {WILD_IMAGES_DIR}")
    print(f"CROPS_KNOWN_DIR  : {CROPS_KNOWN_DIR}")
    print(f"CROPS_WILD_DIR   : {CROPS_WILD_DIR}")
    print(f"CROPS_JSON       : {CROPS_JSON}")
    print(f"MODELS_DIR       : {MODELS_DIR}")
    print(f"YOLO_GORILLA_PT  : {YOLO_GORILLA_PT}")
    print(f"HF_CACHE         : {HF_CACHE}")
    print(f"TORCH_CACHE      : {TORCH_CACHE}")
    print(f"SPECIES          : {SPECIES}")
    print(f"PROJECT_NAME     : {PROJECT_NAME}")
    print()
    print("apply_cache_env() preview:")
    print(f"  HF_HOME        -> {HF_CACHE}")
    print(f"  TORCH_HOME     -> {TORCH_CACHE}")
    print()
    print("config_loader OK")
