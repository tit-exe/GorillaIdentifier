"""
export_tflite_gorilla.py — Export the trained models to TFLite for the Android app.

Exports two files into output/v1_gorilla/tflite/:
  - megadesc_T_arcface_backbone.tflite   the MegaDescriptor-T identifier backbone
  - yolo_v2_detector.tflite              the gorilla YOLO face detector

Both are then copied by hand into the Android app, under app/src/main/assets/
(https://github.com/tit-exe/GorillaIdentifier_AndroidApp). The backbone is also
hosted on HuggingFace (tit0000/GorillaIdentifier) because it exceeds GitHub's size limit.

The TFLite converter for a Swin Transformer runs on Linux only, so this script must be
launched from WSL2 with a Python environment that has the converter installed. Example:

    wsl -d Ubuntu -- bash -c "conda run -n export_env --no-capture-output \\
        python /mnt/<drive>/<path-to-repo>/export_tflite_gorilla.py"

All paths are derived from the script location, so the repository can live anywhere.
"""

import sys, os, subprocess, importlib, importlib.util
from pathlib import Path
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------
# Pre-install the TFLite conversion dependencies (ultralytics needs these for the
# ONNX -> TFLite step). If any are missing, install them all then re-exec so the
# freshly installed modules become importable.
# --------------------------------------------------------------------------------
_tflite_deps = [
    ("tf_keras",          "tf_keras<=2.19.0"),
    ("sng4onnx",          "sng4onnx>=1.0.1"),
    ("onnx_graphsurgeon", "onnx_graphsurgeon>=0.3.26"),
    ("onnx2tf",           "onnx2tf>=1.26.3,<1.29.0"),
]
_missing = [pkg for mod, pkg in _tflite_deps if importlib.util.find_spec(mod) is None]
if _missing:
    log(f"Installing {len(_missing)} missing TFLite dependencies (tensorflow is large, be patient)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--root-user-action=ignore"] + _missing,
        check=True,
    )
    log("Dependencies installed, restarting so the imports are visible...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# --------------------------------------------------------------------------------
# Paths (all relative to this script's location)
# --------------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent
CHECKPOINT_PT = REPO_ROOT / "output/v1_gorilla/best.pt"
BACKBONE_PT   = REPO_ROOT / "models/gorilla_v1_backbone_only.pt"
YOLO_PT       = REPO_ROOT / "models/yolo_gorilla.pt"

EXPORT_DIR    = REPO_ROOT / "output/v1_gorilla/tflite"
OUT_BACKBONE  = EXPORT_DIR / "megadesc_T_arcface_backbone.tflite"
OUT_DETECTOR  = EXPORT_DIR / "yolo_v2_detector.tflite"
WORK_DIR      = REPO_ROOT / "models/yolo_export_tmp"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------
# 1. Backbone -> TFLite
# --------------------------------------------------------------------------------
log("Importing torch / timm...")
# -- Dependency guard: on a missing package, show the exact install command ------
_missing = [m for m in ("torch", "timm") if importlib.util.find_spec(m) is None]
if _missing:
    log("Missing Python package(s): " + ", ".join(_missing), "STOP")
    log("This script runs inside WSL, in the export environment. Install by hand:")
    log("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    log("  pip install timm litert-torch pillow numpy huggingface_hub")
    sys.exit(1)
import torch
import timm
log(f"torch: {torch.__version__}")

converter = None
for name in ("litert_torch", "ai_edge_torch"):
    try:
        mod = importlib.import_module(name)
        if hasattr(mod, "convert"):
            converter = mod
            log(f"Using {name}")
            break
    except ImportError:
        continue
if converter is None:
    log("[ERR] No TFLite converter found. Run: pip install litert-torch")
    sys.exit(1)

log("Creating MegaDescriptor-T-224 architecture...")
backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=False, num_classes=0)

if BACKBONE_PT.exists():
    log(f"Loading cached backbone-only weights from {BACKBONE_PT.name}...")
    state = torch.load(str(BACKBONE_PT), map_location="cpu", weights_only=False)
    backbone.load_state_dict(state["backbone_state"] if "backbone_state" in state else state)
elif CHECKPOINT_PT.exists():
    log(f"Loading backbone from full checkpoint {CHECKPOINT_PT.name}...")
    ck = torch.load(str(CHECKPOINT_PT), map_location="cpu", weights_only=False)
    if "backbone_state" not in ck:
        log(f"[ERR] 'backbone_state' key not found. Keys: {list(ck.keys())}")
        sys.exit(1)
    backbone.load_state_dict(ck["backbone_state"])
    torch.save({"backbone_state": backbone.state_dict()}, str(BACKBONE_PT))
    log(f"Cached backbone-only weights -> {BACKBONE_PT.name}")
else:
    log(f"[ERR] No checkpoint found at {CHECKPOINT_PT}. Train first, or download the model.")
    sys.exit(1)

backbone = backbone.eval().cpu()
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224)
    out = backbone(dummy)
    log(f"Backbone output shape: {tuple(out.shape)} (expected (1, 768))")
    if out.shape != (1, 768):
        log("[ERR] Unexpected output shape, check the architecture.")
        sys.exit(1)

log("Converting the backbone to TFLite float32 (2 to 5 min on CPU)...")
try:
    edge_model = converter.convert(backbone, (dummy,))
    edge_model.export(str(OUT_BACKBONE))
    log(f"Saved : {OUT_BACKBONE}  ({OUT_BACKBONE.stat().st_size / 1e6:.1f} MB)")
except Exception as e:
    log(f"[ERR] Backbone conversion failed: {e}")
    sys.exit(1)

with open(OUT_BACKBONE, "rb") as f:
    f.seek(4); magic = f.read(4)
if magic != b"TFL3":
    log(f"[WARN] Unexpected magic bytes: {magic}, the backbone may be corrupt.")
    sys.exit(1)
log("Backbone TFLite magic bytes OK.")

# --------------------------------------------------------------------------------
# 2. YOLO detector -> TFLite
# --------------------------------------------------------------------------------
log("")
log("-- YOLO detector export --")
if not YOLO_PT.exists():
    log(f"[ERR] YOLO model not found at {YOLO_PT}. Download it with models/download_models.py.")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    log("[ERR] ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

log(f"Loading {YOLO_PT.name} ({YOLO_PT.stat().st_size / 1e6:.1f} MB)...")
yolo = YOLO(str(YOLO_PT))
log("Exporting the detector to TFLite float32...")
export_path = yolo.export(format="tflite", imgsz=640, half=False, int8=False,
                          project=str(WORK_DIR), name="yolo_gorilla")

ep = Path(str(export_path))
if ep.suffix == ".tflite":
    tflite_path = ep
else:
    candidates = list(ep.rglob("*.tflite")) if ep.is_dir() else list(WORK_DIR.rglob("*.tflite"))
    if not candidates:
        log("[ERR] Could not locate the exported .tflite file.")
        sys.exit(1)
    tflite_path = candidates[0]

import shutil
shutil.copy2(str(tflite_path), str(OUT_DETECTOR))
log(f"Saved : {OUT_DETECTOR}  ({OUT_DETECTOR.stat().st_size / 1e6:.1f} MB)")

with open(OUT_DETECTOR, "rb") as f:
    f.seek(4); magic = f.read(4)
if magic != b"TFL3":
    log(f"[WARN] Unexpected magic bytes: {magic}, the detector may be corrupt.")
    sys.exit(1)
log("Detector TFLite magic bytes OK.")

log("")
log(f"Done. Copy both files from {EXPORT_DIR} into the Android app under app/src/main/assets/,")
log("then rebuild the app in Android Studio.")
