"""
check_env.py — Environment check for GorillaIdentifier pipeline
===============================================================
Run this first on any new machine to verify the environment.

    python check_env.py

Note: this is NOT a pip setup.py. It only checks the environment.

=== QUICK SETUP (Anaconda recommended) ===

  1. Install Anaconda: https://www.anaconda.com/download
  2. Open "Anaconda Prompt" (NOT regular cmd/PowerShell)
  3. Create the environment:

       conda create -n gorilla_id python=3.10 -y
       conda activate gorilla_id

  4. Install PyTorch WITH CUDA (must come first — PyPI has CPU-only builds):

       pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124

  5. Install all other dependencies:

       pip install timm==0.9.16 ultralytics==8.2.0 Pillow==10.3.0 opencv-python==4.9.0.80 numpy==1.26.4 scikit-learn==1.4.2 umap-learn==0.5.6 matplotlib==3.9.0 PyQt5==5.15.10 tqdm==4.66.4 huggingface_hub==0.23.2

  6. Run this script to verify:

       python check_env.py

  NOTE: Always run with "conda activate gorilla_id" first.
  NOTE: Do NOT use "pip install -r requirements.txt" directly — the torch+CUDA
        package needs the special --index-url above; plain pip installs CPU-only.
"""

import sys
import subprocess
import platform
from pathlib import Path

# ------------------------------------------------------------------
# Config loader (reads config.yaml for cache paths)
# ------------------------------------------------------------------
try:
    from common.config_loader import HF_CACHE, TORCH_CACHE, MODELS_DIR, CROPS_JSON
    _cfg_ok = True
except Exception as e:
    _cfg_ok = False
    _cfg_err = str(e)


def check(condition, msg_ok, msg_fail=""):
    if condition:
        print(f"  [OK] {msg_ok}")
    else:
        print(f"  [KO] {msg_fail or msg_ok}")
    return condition


def main():
    print("=" * 60)
    print("  GorillaIdentifier — Environment check")
    print("=" * 60)

    ok = True

    # Python version
    major, minor = sys.version_info.major, sys.version_info.minor
    ok &= check(major == 3 and minor >= 9,
                f"Python {major}.{minor}",
                f"Python {major}.{minor} — need 3.9+")

    # config.yaml
    cfg = Path("config.yaml")
    ok &= check(cfg.exists(), "config.yaml found", "config.yaml missing — are you in the repo root?")

    # config_loader import
    if _cfg_ok:
        check(True, "config_loader.py imports cleanly")
    else:
        check(False, "", f"config_loader.py failed: {_cfg_err}")
        ok = False

    # PyTorch + CUDA
    try:
        import torch
        cuda = torch.cuda.is_available()
        check(True, f"PyTorch {torch.__version__}")
        if cuda:
            props = torch.cuda.get_device_properties(0)
            check(True, f"CUDA {torch.version.cuda} — {props.name} "
                        f"({props.total_memory/1e9:.1f} GB VRAM)")
        else:
            # Distinguish between "no GPU" and "GPU present but PyTorch is CPU-only"
            nvidia_present = False
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0 and r.stdout.strip():
                    nvidia_present = True
                    gpu_line = r.stdout.strip().splitlines()[0]
                    print(f"  [!!] nvidia-smi detects: {gpu_line.strip()}")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            if nvidia_present:
                print("  [!!] GPU is present but PyTorch has NO CUDA support.")
                print("  [!!] You installed the CPU-only build of PyTorch.")
                print("  [!!] Training will run on CPU — 50-100x slower than GPU.")
                print("  [!!]")
                print("  [!!] Fix: reinstall PyTorch with CUDA in Anaconda Prompt:")
                print("  [!!]   conda activate gorilla_id")
                print("  [!!]   pip uninstall torch torchvision -y")
                print("  [!!]   pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 \\")
                print("  [!!]          --index-url https://download.pytorch.org/whl/cu124")
                ok = False
            else:
                print("  [--] No NVIDIA GPU detected — training will run on CPU (slow)")
    except ImportError:
        check(False, "", "PyTorch not installed — see setup instructions at top of this file")
        ok = False

    # timm
    try:
        import timm
        check(True, f"timm {timm.__version__}")
    except ImportError:
        check(False, "", "timm not installed — pip install timm")
        ok = False

    # ultralytics
    try:
        import ultralytics
        check(True, f"ultralytics {ultralytics.__version__}")
    except ImportError:
        check(False, "", "ultralytics not installed — pip install ultralytics")
        ok = False

    # PyQt5 (reviewer only, not required for training)
    try:
        import PyQt5
        check(True, "PyQt5 available (crop reviewer)")
    except ImportError:
        print("  [--] PyQt5 absent — reviewer unavailable (pip install PyQt5)")

    # huggingface_hub
    try:
        import huggingface_hub
        check(True, f"huggingface_hub {huggingface_hub.__version__}")
    except ImportError:
        check(False, "", "huggingface_hub not installed — pip install huggingface_hub")
        ok = False

    # Cache paths
    if _cfg_ok:
        print()
        print("  Cache paths (from config.yaml):")
        print(f"    HF_HOME    : {HF_CACHE}")
        print(f"    TORCH_HOME : {TORCH_CACHE}")
        print()
        print("  To redirect to a large drive, uncomment and edit in config.yaml:")
        print("    hf_cache_dir:    \"D:/HuggingFaceCache\"")
        print("    torch_cache_dir: \"D:/TorchCache\"")

    # Data structure
    print()
    print("  Data structure:")
    for d in ["data/photos", "data/wild_images/raw", "data/crops/known", "data/crops/wild"]:
        p = Path(d)
        status = "exists" if p.exists() else "will be created by pipeline"
        print(f"    {d:<30} — {status}")

    # Models
    print()
    print("  Models:")
    if _cfg_ok:
        for pt in sorted(MODELS_DIR.glob("*.pt")):
            print(f"    {pt.name:<50} {pt.stat().st_size/1e6:.0f} MB")
        if not any(MODELS_DIR.glob("*.pt")):
            print("    (no model found — yolo_gorilla.pt must be placed in models/, see README)")

    print()
    if ok:
        print("  Environment OK — ready to run the pipeline.")
        print()
        print("  Quick start:")
        print("    (put photos in data/photos/<PrefixName>/ e.g. 'SB Humba')")
        print("    python v1_megadesc_arcface/train.py --extract   # extract face crops")
        print("    python common/review_crops.py                   # review crops")
        print("    python v1_megadesc_arcface/train.py             # train")
        print()
        print("  See README.md for the full step-by-step guide.")
    else:
        print("  Issues detected — fix the [KO] / [!!] items above.")
        print()
        print("  === SETUP REMINDER (Anaconda Prompt) ===")
        print()
        print("  conda create -n gorilla_id python=3.10 -y")
        print("  conda activate gorilla_id")
        print()
        print("  # PyTorch + CUDA (must use special index — PyPI has CPU-only builds)")
        print("  pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 \\")
        print("         --index-url https://download.pytorch.org/whl/cu124")
        print()
        print("  # All other dependencies")
        print("  pip install timm==0.9.16 ultralytics==8.2.0 Pillow==10.3.0 \\")
        print("         opencv-python==4.9.0.80 numpy==1.26.4 scikit-learn==1.4.2 \\")
        print("         umap-learn==0.5.6 matplotlib==3.9.0 PyQt5==5.15.10 \\")
        print("         tqdm==4.66.4 huggingface_hub==0.23.2")
        print()


if __name__ == "__main__":
    main()
