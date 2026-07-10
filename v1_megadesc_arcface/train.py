"""
train.py — GorillaIdentifier V1
================================
MegaDescriptor-T-224 + Sub-center ArcFace
Virunga gorillas — individual identification

This script does EVERYTHING:
  1. Data scan       (stats, exclusion recommendations)
  2. Crop extraction (--extract, requires the gorilla YOLO model)
  3. Training         (4-phase degradation curriculum)
  4. Gallery          (filtered exemplars + calibrated threshold)
  5. Final benchmark  (per-individual F1, confusion matrix, separability)

KEY DESIGN CHOICES:
  [A] SupCon (T=0.07)  : DIRECTLY maximizes inter-individual separability
  [B] ArcFace m=0.50   : margin proven for animal re-ID (not 0.35)
  [C] L_invariance     : forces stable embeddings clean/degraded
  [D] Adaptive K       : K=1 adults, K=2 JUV/BB (fast ontogeny)
  [E] Gorilla augm.    : no vertical flip, ±20°, vegetation erasing
  [F] VRAM auto        : batch_size adapts to any GPU
  [G] Crash-safety     : saves every epoch + Win32 window-close handler

CRASH-SAFETY:
  Closing the window / power loss / Ctrl+C → automatic save.
  Relaunching resumes exactly where it stopped.

USAGE:
  conda activate gorilla_id
  python v1_megadesc_arcface/train.py                  # from existing crops
  python v1_megadesc_arcface/train.py --extract        # extract crops then train
  python v1_megadesc_arcface/train.py --min-photos 5   # individual exclusion threshold
  python v1_megadesc_arcface/train.py --dry-run        # 3-epoch smoke test
  python v1_megadesc_arcface/train.py --benchmark-only # benchmark from an existing gallery

EXPECTED LAYOUT:
  data/photos/{CAT} {Name}/  e.g. "SB Humba/", "ADF Anangana/", "JUV Bakunzi/"
  data/crops/known/          auto-created by --extract
  data/wild_images/raw/      optional — unidentified background images
  models/yolo_gorilla.pt     needed only with --extract
"""

# ── HuggingFace / Torch cache setup, before any other import ─────────────────
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.config_loader import apply_cache_env
apply_cache_env()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import io, json, math, time, signal, random, shutil, warnings, argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# -- Dependency guard: on a missing package, show the exact install command ------
import importlib.util as _ilu
_missing = [m for m in ("numpy", "torch") if _ilu.find_spec(m) is None]
if _missing:
    print("\n[STOP] Missing Python package(s): " + ", ".join(_missing))
    print("Install in the 'gorilla_id' environment (Anaconda Prompt):\n")
    print("  conda activate gorilla_id")
    print("  pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124")
    print("  pip install -r requirements.txt")
    raise SystemExit(1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
except ImportError:
    print("[ERROR] scikit-learn missing. pip install scikit-learn")
    sys.exit(1)

try:
    import timm
except ImportError:
    print("[ERROR] timm missing. pip install timm==0.9.16")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    RICH = True
except ImportError:
    RICH = False
    print("[INFO] rich not found — using basic display (pip install rich)")

# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="GorillaIdentifier V1 — all-in-one script")
parser.add_argument("--extract",        action="store_true", help="Extract crops from data/photos/ via YOLO")
parser.add_argument("--regen-json",     action="store_true", help="Regenerate data/crops.json from existing crops (requires YOLO)")
parser.add_argument("--min-photos",     type=int, default=5,   help="Exclude individuals with < N photos (default: 5)")
parser.add_argument("--dry-run",        action="store_true",   help="3 epochs, for a quick test")
parser.add_argument("--benchmark-only", action="store_true",   help="Benchmark only (gallery must already exist)")
parser.add_argument("--no-wild",        action="store_true",   help="Ignore data/wild_images/ (background class)")
parser.add_argument("--reset",          action="store_true",   help="Start from scratch — deletes resume.pt and best.pt")
ARGS = parser.parse_args()
DRY  = ARGS.dry_run

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
REPO          = Path(__file__).parent.parent.resolve()
PHOTOS_DIR    = REPO / "data" / "photos"
CROPS_DIR     = REPO / "data" / "crops" / "known"
CROPS_JSON    = REPO / "data" / "crops.json"
WILD_DIR      = REPO / "data" / "wild_images" / "raw"
MODELS_DIR    = REPO / "models"
OUT           = REPO / "output" / "v1_gorilla"
CKPT_RESUME      = OUT / "resume.pt"
CKPT_BEST        = OUT / "best.pt"
GALLERY_JSON     = OUT / "gallery_gorilla.json"
REPORT_JSON      = OUT / "report.json"          # legacy, replaced by diagnostics.json
DIAGNOSTICS_JSON = OUT / "diagnostics.json"     # full JSON, all values
CURVES_PNG       = OUT / "curves.png"
BENCHMARK_PNG    = OUT / "benchmark.png"
DATA_STATS_PNG   = OUT / "data_stats.png"
CONFUSION_PNG    = OUT / "confusion.png"
LOG_FILE         = OUT / "train.log"
YOLO_MODEL    = MODELS_DIR / "yolo_gorilla.pt"

for d in [OUT, MODELS_DIR]: d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# AGE-SEX CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════
ADULT_CATS = {"SB", "ADF", "AD", "SAF", "SAM"}   # stable appearance → K=1
JUV_CATS   = {"JUV", "BB", "Baby"}                # fast ontogeny → K=2
ALL_CATS   = ADULT_CATS | JUV_CATS
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def parse_folder(name: str):
    """'ADF Anangana' → ('ADF', 'Anangana')  |  'Anangana' → (None, 'Anangana')"""
    parts = name.strip().split(" ", 1)
    if len(parts) == 2 and parts[0] in ALL_CATS:
        return parts[0], parts[1]
    return None, name

def k_subcenters(category) -> int:
    return 2 if category in JUV_CATS else 1

# ══════════════════════════════════════════════════════════════════════════════
# VRAM AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_optimal_batch() -> int:
    if not torch.cuda.is_available():
        print("[INFO] No GPU detected → CPU, batch=8")
        return 8
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    name = torch.cuda.get_device_name(0)
    if vram >= 16:   bs = 64
    elif vram >= 8:  bs = 32
    elif vram >= 4:  bs = 16   # RTX 3050 4 GB
    else:            bs = 8
    print(f"[INFO] GPU: {name} ({vram:.1f} GB) → batch_size={bs}")
    return bs

BATCH = detect_optimal_batch()
IMG_SIZE = 224
SEED     = 42
MEAN = STD = [0.5, 0.5, 0.5]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
ARC_SCALE   = 64
ARC_MARGIN  = 0.50   # value proven for animal re-ID — DO NOT reduce
K_WILD      = 5
QUALITY_THR = 0.62   # gallery exemplar quality threshold
K_EXEMPLARS = 25     # max exemplars per individual in the gallery
VAL_RATIO   = 0.15
N_HOLDOUT   = 3      # individuals held out as pseudo-unknowns (0 = disabled)

# 4 phases (shortened durations if --dry-run)
PHASES = [
    dict(name="A — Init (head only)",    epochs=3  if not DRY else 1,
         freeze=True,  lr_bb=0.0,   lr_h=1e-3,
         lam_inv=0.00, lam_sup=0.00, severity=0.00),
    dict(name="B — Warmup",              epochs=15 if not DRY else 1,
         freeze=False, lr_bb=5e-6,  lr_h=5e-4,
         lam_inv=0.15, lam_sup=0.10, severity=0.40),
    dict(name="C — Learning",            epochs=20 if not DRY else 1,
         freeze=False, lr_bb=3e-6,  lr_h=2e-4,
         lam_inv=0.25, lam_sup=0.15, severity=0.70),
    dict(name="D — Consolidation",       epochs=15 if not DRY else 1,
         freeze=False, lr_bb=1e-6,  lr_h=5e-5,
         lam_inv=0.25, lam_sup=0.15, severity=1.00,
         early_stop=True, patience=12),
]
TOTAL_EPOCHS = sum(p["epochs"] for p in PHASES)

# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════
_log_fh  = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
_console = Console() if RICH else None

def log(msg="", level="INFO", always=False):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}][{level}] {msg}"
    _log_fh.write(line + "\n"); _log_fh.flush()
    if always or not RICH or level in ("ERROR", "WARN"):
        print(line)

def section(title):
    bar = "─" * 70
    log("", always=True); log(bar, always=True)
    log(f"  {title}", always=True); log(bar, always=True)

# ══════════════════════════════════════════════════════════════════════════════
# CRASH-SAFETY
# ══════════════════════════════════════════════════════════════════════════════
_interrupt = False
_save_fn   = None

def _emergency_save():
    if _save_fn:
        try:   _save_fn("interrupt")
        except Exception as e: log(f"Emergency save failed: {e}", "ERROR")

def _sigint(sig, frame):
    global _interrupt
    if not _interrupt:
        log("Ctrl+C detected — stopping cleanly after this batch...", "WARN")
        _interrupt = True
    else:
        _emergency_save(); _log_fh.close(); sys.exit(1)

signal.signal(signal.SIGINT, _sigint)

try:
    import win32api
    def _win32(t):
        if t in (2, 5, 6):
            global _interrupt; _interrupt = True
            log(f"Window closed (event {t}) — saving...", "WARN")
            _emergency_save(); time.sleep(1.5)
        return False
    win32api.SetConsoleCtrlHandler(_win32, True)
    log("Win32 handler installed — closing the window triggers an auto-save")
except ImportError:
    log("pywin32 not found — save on epoch boundaries only (pip install pywin32)", "WARN")

# ══════════════════════════════════════════════════════════════════════════════
# GORILLA-SPECIFIC AUGMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════
class _LowRes:
    def __init__(self, mn, mx, p): self.mn=mn; self.mx=mx; self.p=p
    def __call__(self, img):
        if random.random() > self.p: return img
        w, h = img.size; f = random.uniform(self.mn, self.mx)
        s = max(int(w*f), 8), max(int(h*f), 8)
        return img.resize(s, Image.BILINEAR).resize((w, h), Image.BICUBIC)

class _JPEG:
    def __init__(self, mn, mx, p): self.mn=mn; self.mx=mx; self.p=p
    def __call__(self, img):
        if random.random() > self.p: return img
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(self.mn, self.mx))
        buf.seek(0); return Image.open(buf).copy()

class _CropJitter:
    def __init__(self, px=2): self.px=px
    def __call__(self, t):
        p = T.functional.pad(t, self.px, padding_mode="reflect")
        i = random.randint(0, 2*self.px); j = random.randint(0, 2*self.px)
        return p[:, i:i+t.shape[1], j:j+t.shape[2]]

class _ForestShadow:
    """
    Simulates dappled forest light (Virunga).
    Darkens a random band across the face — branch / foliage occlusion effect.
    Gorillas ≠ orangutans: very dark conditions, dense undergrowth.
    Operates on a C×H×W tensor already in [0,1].
    """
    def __init__(self, p=0.35): self.p = p
    def __call__(self, t):
        if random.random() > self.p: return t
        t = t.clone()
        _, h, w = t.shape
        if random.random() < 0.6:
            y1 = random.randint(0, h - h//4)
            y2 = random.randint(y1 + h//8, min(h, y1 + h//2))
            alpha = random.uniform(0.20, 0.60)
            t[:, y1:y2, :] = t[:, y1:y2, :] * alpha
        else:
            x1 = random.randint(0, w - w//4)
            x2 = random.randint(x1 + w//8, min(w, x1 + w//2))
            alpha = random.uniform(0.20, 0.60)
            t[:, :, x1:x2] = t[:, :, x1:x2] * alpha
        return t.clamp(0, 1)


class _ColorTemp:
    """
    Simulates the color-temperature variation of forest lighting.

    Field observation (Virunga) — same individual, same photo session:
      #726f6f → warm gray   (R≈G≈B, direct warm light)
      #8a8c9d → blue-gray   (B > R by ~+14%, partial shade)
      #63727e → deep blue-gray (B > R by ~+27%, dense undergrowth)

    This is NOT an HSV hue rotation (saturation stays low) but a temperature
    imbalance: warm light ↑R ↓B, cold/forest light ↑B ↓R, with green
    slightly correlated.

    The model MUST be invariant to this so it doesn't confuse the same
    gray face under two different lighting conditions.

    Operates on a C×H×W tensor in [0,1] (after ToTensor, before _norm).
    strength: max amplitude of the R↔B shift (0.18 covers observed cases).
    """
    def __init__(self, p: float = 0.55, strength: float = 0.18):
        self.p = p
        self.strength = strength   # max shift on the R and B channels

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return t
        t = t.clone()
        # shift ∈ [-strength, +strength]
        # positive = warm light (red), negative = cold/forest light (blue)
        shift = random.uniform(-self.strength, self.strength)
        t[0] = (t[0] * (1.0 + shift)).clamp(0, 1)          # R ↑ if warm
        t[1] = (t[1] * (1.0 + shift * 0.25)).clamp(0, 1)   # G follows weakly
        t[2] = (t[2] * (1.0 - shift)).clamp(0, 1)           # B ↑ if cold
        return t


_norm = T.Normalize(MEAN, STD)

def gorilla_clean_tf():
    """
    Base augmentation for gorillas — Virunga forest.

    Settings tuned for gorilla fur/skin (near-monochrome black):
    - high brightness/contrast : very variable under forest canopy
    - saturation ≤ 0.15         : skin turns cold blue-gray (#8a8c9d)
                                   but stays low-saturation
    - hue ≤ 0.05                 : slight (≠ ColorTemp, which handles the real R↔B shift)
    - _ColorTemp                 : warm↔cold shift observed in the field
    """
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.45, contrast=0.50,
                      saturation=0.15, hue=0.05),   # saturation raised for blue-gray tones
        T.RandomGrayscale(p=0.15),
        T.ToTensor(),
        T.Lambda(_ColorTemp(p=0.50, strength=0.15)),  # warm↔cool shift
        _norm,
    ])

def gorilla_degraded_tf(severity: float):
    """
    Degradation curriculum — Virunga forest.

    At severity=1.0 the augmentations cover the worst field cases:
    - R↔B shift up to ±0.26   (undergrowth + indirect flash)
    - saturation up to 0.30   (wet reflections on skin/fur)
    - hue up to ±0.09         (rare but possible — sky gap in the canopy)
    - ForestShadow + Erasing  : occlusive vegetation
    """
    s = severity
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(max(0.50, 0.75 - 0.25*s), 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(int(20 + 15*s)),
        T.ColorJitter(brightness=0.45 + 0.40*s,
                      contrast=0.50  + 0.40*s,
                      saturation=0.15 + 0.15*s,   # max 0.30 at severity=1
                      hue=0.05 + 0.04*s),          # max 0.09 at severity=1
        T.RandomGrayscale(p=0.20 + 0.15*s),
        T.RandomApply([T.GaussianBlur(11, sigma=(0.5, 0.5+5.5*s))], p=0.10+0.35*s),
        T.Lambda(_LowRes(max(0.08, 0.50-0.42*s), 0.95, p=0.10+0.30*s)),
        T.Lambda(_JPEG(max(10, int(45-35*s)), max(45, int(80-35*s)), p=0.10+0.25*s)),
        T.ToTensor(),
        T.RandomErasing(p=0.15+0.20*s, scale=(0.02, 0.05+0.18*s),
                        ratio=(0.3, 3.0), value="random"),
        T.Lambda(_CropJitter(px=2)),
        T.Lambda(_ForestShadow(p=0.15 + 0.25*s)),
        # Color temperature: warm↔cool, growing intensity with severity
        # Covers the #726f6f → #63727e range observed in the field
        T.Lambda(_ColorTemp(p=0.50 + 0.20*s, strength=0.15 + 0.11*s)),
        _norm,
    ])

def gorilla_val_tf():
    return T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), _norm])

# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════
class PairDataset(Dataset):
    """Returns (clean, degraded, label) — double forward pass for L_invariance.

    Transforms are created ONCE in __init__ and reused.
    Creating a T.Compose in every __getitem__ costs ~200 µs/image.
    """
    def __init__(self, paths, labels, severity=1.0):
        self.paths = paths; self.labels = labels
        self.clean_tf  = gorilla_clean_tf()
        self.degrad_tf = gorilla_degraded_tf(severity)
    def __len__(self): return len(self.paths)
    def _load(self, idx):
        try:   return Image.open(self.paths[idx]).convert("RGB")
        except: return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (80, 80, 80))
    def __getitem__(self, idx):
        img = self._load(idx)
        return self.clean_tf(img), self.degrad_tf(img), int(self.labels[idx])

class PlainDataset(Dataset):
    def __init__(self, paths, labels, tf=None):
        self.paths = paths; self.labels = labels
        self.tf = tf or gorilla_val_tf()
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:   img = Image.open(self.paths[idx]).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (80, 80, 80))
        return self.tf(img), int(self.labels[idx])

class WildDataset(Dataset):
    def __init__(self, wild_dir, n, lbl):
        self.all = sorted([f for f in wild_dir.iterdir() if f.suffix in IMG_EXTS])
        self.lbl = lbl; self.files = self._sample(n)
        self.clean_tf  = gorilla_clean_tf()
        self.degrad_tf = gorilla_degraded_tf(1.0)
    def _sample(self, n):
        n = min(n, len(self.all))
        return random.sample(self.all, n) if n > 0 else []
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:   img = Image.open(self.files[idx]).convert("RGB")
        except: img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (60, 50, 40))
        return self.clean_tf(img), self.degrad_tf(img), self.lbl

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_crops(crops_dir, min_photos=0):
    """Loads data/crops/known/ — returns paths, labels, names."""
    folders = sorted([d for d in crops_dir.iterdir()
                      if d.is_dir() and not d.name.startswith("_")])
    paths, labels, names = [], [], []
    excluded = []
    for d in folders:
        imgs = sorted([f for f in d.iterdir() if f.suffix in IMG_EXTS])
        if len(imgs) < max(min_photos, 2):
            excluded.append((d.name, len(imgs)))
            continue
        i = len(names)
        for f in imgs:
            paths.append(f); labels.append(i)
        names.append(d.name)
    return paths, labels, names, excluded

# ══════════════════════════════════════════════════════════════════════════════
# MODEL: SUB-CENTER ARCFACE
# ══════════════════════════════════════════════════════════════════════════════
class SubCenterArcFace(nn.Module):
    def __init__(self, emb_dim, n_classes, k_list, scale=64.0, margin=0.50):
        super().__init__()
        self.scale = scale; self.margin = margin
        self.n_classes = n_classes
        total_k = sum(k_list)
        self.weight = nn.Parameter(torch.FloatTensor(total_k, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("cls_start",
            torch.tensor([sum(k_list[:i]) for i in range(n_classes)], dtype=torch.long))
        self.register_buffer("cls_k", torch.tensor(k_list, dtype=torch.long))
        self.cos_m = math.cos(margin); self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, emb, labels):
        w = F.normalize(self.weight, dim=1)
        ca = emb @ w.T                                           # [B, total_K]
        logits = torch.zeros(emb.size(0), self.n_classes, device=emb.device)
        for c in range(self.n_classes):
            s = self.cls_start[c].item(); k = self.cls_k[c].item()
            logits[:, c] = ca[:, s:s+k].max(dim=1).values
        cos = logits.clamp(-1.0, 1.0)
        sin = (1.0 - cos**2).clamp(0.0).sqrt()
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        oh  = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
        out = (oh * phi + (1 - oh) * logits) * self.scale
        return F.cross_entropy(out, labels, label_smoothing=0.05)

# ══════════════════════════════════════════════════════════════════════════════
# AUXILIARY LOSSES
# ══════════════════════════════════════════════════════════════════════════════
def loss_supcon(emb, labels, temp=0.07):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).
    Directly maximizes the intra-class / inter-class similarity ratio
    in embedding space.

    Difference vs ArcFace: ArcFace imposes a fixed angular margin,
    SupCon optimizes the separation of ALL pairs simultaneously.
    """
    N = emb.size(0)
    if N < 4:
        return torch.tensor(0.0, device=emb.device)
    sim = torch.matmul(emb, emb.T) / temp                  # [N, N]
    mask_diag = ~torch.eye(N, dtype=torch.bool, device=emb.device)
    lc = labels.unsqueeze(1); lr = labels.unsqueeze(0)
    mask_pos = (lc == lr) & mask_diag                       # pairs of the same individual
    if not mask_pos.any():
        return torch.tensor(0.0, device=emb.device)
    sim_exp   = torch.exp(sim) * mask_diag.float()
    log_denom = torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-8)
    log_pos   = sim - log_denom                              # [N, N]
    n_pos     = mask_pos.float().sum(dim=1).clamp(min=1)
    loss_per  = -(log_pos * mask_pos.float()).sum(dim=1) / n_pos
    return loss_per.mean()

def loss_invariance(emb_clean, emb_deg):
    """Forces embedding(clean_image) ≈ embedding(degraded_image).
    Directly addresses the "screenshot" problem: same gorilla at poor quality
    → same identity recognized."""
    return 1.0 - (emb_clean * emb_deg).sum(dim=1).mean()

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE
# ══════════════════════════════════════════════════════════════════════════════
def load_backbone():
    """Loads MegaDescriptor-T-224 (Swin Transformer, 768-dim).
    Pretrained on multi-species animal re-identification."""
    log("Loading MegaDescriptor-T-224 (Swin Transformer)...")
    bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
    with torch.no_grad():
        dim = bb(torch.randn(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
    log(f"  Backbone loaded — {dim}D — {sum(p.numel() for p in bb.parameters())/1e6:.1f}M params")
    return bb, dim

# ══════════════════════════════════════════════════════════════════════════════
# ATOMIC CHECKPOINT SAVES
# ══════════════════════════════════════════════════════════════════════════════
def _atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp); tmp.replace(path)

def save_resume(bb, arc, opt, sched, phase, ep, g_ep, best_val, best_ep, hist, names):
    _atomic_save({
        "backbone_state": bb.state_dict(), "arc_state": arc.state_dict(),
        "opt_state": opt.state_dict(),
        "sched_state": sched.state_dict() if sched else None,
        "phase": phase, "ep": ep, "g_ep": g_ep,
        "best_val": best_val, "best_ep": best_ep,
        "history": hist, "names": names,
        "ts": datetime.now().isoformat(),
    }, CKPT_RESUME)

def save_best(bb, arc, names, dim, ep, val):
    _atomic_save({
        "backbone_state": bb.state_dict(), "arc_state": arc.state_dict(),
        "names": names, "emb_dim": dim, "epoch": ep, "val": val,
        "version": "v1-gorilla", "mean": MEAN, "std": STD,
    }, CKPT_BEST)

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(bb, tr_p, tr_l, va_p, va_l, n_classes, dim, unk_p=None):
    """
    unk_p: list of pseudo-unknown crops (held-out individuals).
    Returns (acc_clean, acc_degraded, unk_rejection_rate or None).
    """
    bb.eval()
    # Prototypes from the training crops
    proto = torch.zeros(n_classes, dim)
    cnt   = torch.zeros(n_classes)
    dl    = DataLoader(PlainDataset(tr_p, tr_l), 64, num_workers=0)
    for imgs, lbs in dl:
        e = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
        for ei, li in zip(e, lbs.tolist()):
            if li < n_classes: proto[li] += ei; cnt[li] += 1
    for c in range(n_classes):
        if cnt[c] > 0: proto[c] = F.normalize(proto[c], dim=0)

    # Clean accuracy + distribution of positive similarities
    ok = tot = 0
    pos_sims_val = []
    vdl = DataLoader(PlainDataset(va_p, va_l), 64, num_workers=0)
    for imgs, lbs in vdl:
        e = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
        sims = e @ proto.T
        pred = sims.argmax(1)
        ok += (pred == lbs).sum().item(); tot += lbs.size(0)
        for i, lb in enumerate(lbs.tolist()):
            pos_sims_val.append(float(sims[i, lb]))
    acc_c = ok / max(tot, 1)

    # Degraded accuracy
    ok = tot = 0
    vdl2 = DataLoader(PlainDataset(va_p, va_l, gorilla_degraded_tf(0.7)), 64, num_workers=0)
    for imgs, lbs in vdl2:
        e = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
        pred = (e @ proto.T).argmax(1)
        ok += (pred == lbs).sum().item(); tot += lbs.size(0)
    acc_d = ok / max(tot, 1)

    # Unknown rejection rate (held-out pseudo-unknowns)
    unk_rate = None
    if unk_p and len(unk_p) > 0:
        # Dynamic threshold: pos_mean - 2*std (adapts as training progresses)
        # Interpretation: positive embeddings well above → reliable threshold
        pm = np.mean(pos_sims_val) if pos_sims_val else 0.70
        ps = np.std(pos_sims_val)  if pos_sims_val else 0.10
        dyn_thr = float(np.clip(pm - 2.0 * ps, 0.45, 0.80))
        unk_dl = DataLoader(PlainDataset(unk_p, [0]*len(unk_p)), 64, num_workers=0)
        n_rej = 0
        for imgs, _ in unk_dl:
            e = F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu()
            max_sims = (e @ proto.T).max(1).values
            n_rej += (max_sims < dyn_thr).sum().item()
        unk_rate = n_rej / len(unk_p)

    return acc_c, acc_d, unk_rate

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(bb, arc, loaders, opt, sched, pc, live_state, refresh_fn=None):
    bb.train(); arc.train()
    lam_inv = pc["lam_inv"]; lam_sup = pc["lam_sup"]
    tot = defaultdict(float); tot_n = 0
    all_loaders = [l for l in loaders if l is not None]
    total_b = sum(len(l) for l in all_loaders)
    t0 = time.time(); b_idx = 0

    for loader in all_loaders:
        for clean, deg, labels in loader:
            if _interrupt: break
            clean  = clean.to(DEVICE, non_blocking=True).float()
            deg    = deg.to(DEVICE,   non_blocking=True).float()
            labels = labels.to(DEVICE, non_blocking=True).long()

            # Double forward pass (clean + degraded)
            both    = torch.cat([clean, deg], dim=0)
            emb_all = F.normalize(bb(both), dim=1)
            B       = clean.size(0)
            ec, ed  = emb_all[:B], emb_all[B:]

            l_arc = arc(ec, labels) + arc(ed, labels)
            l_inv = loss_invariance(ec, ed)
            # SupCon on clean embeddings only (more stable)
            l_sup = loss_supcon(ec, labels)

            loss = l_arc + lam_inv * l_inv + lam_sup * l_sup
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(bb.parameters()) + list(arc.parameters()), 1.0)
            opt.step()
            if sched: sched.step()

            bs = labels.size(0)
            tot["arc"] += l_arc.item()*bs; tot["inv"] += l_inv.item()*bs
            tot["sup"] += l_sup.item()*bs; tot_n += bs
            b_idx += 1
            eta = (time.time()-t0)/b_idx * (total_b - b_idx)
            live_state.update({
                "batch": b_idx, "total_b": total_b,
                "l_arc": tot["arc"]/tot_n, "l_inv": tot["inv"]/tot_n,
                "l_sup": tot["sup"]/tot_n, "eta_b": int(eta),
            })
            if refresh_fn: refresh_fn()
        if _interrupt: break

    N = max(tot_n, 1)
    return tot["arc"]/N, tot["inv"]/N, tot["sup"]/N

# ══════════════════════════════════════════════════════════════════════════════
# GALLERY
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_gallery(bb, paths, labels, names, dim):
    section("Building the gallery")
    bb.eval()
    ds = PlainDataset(paths, labels)
    dl = DataLoader(ds, 64, num_workers=0)
    embs, labs = [], []
    for imgs, lbs in dl:
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy())
        labs.extend(lbs.tolist())
    embs = np.concatenate(embs).astype(np.float32)
    labs = np.array(labs)
    n_cls = len(names)

    individuals = {}
    pos_sims, neg_sims = [], []
    proto_mat = np.zeros((n_cls, dim), dtype=np.float32)

    for i, name in enumerate(names):
        mask  = labs == i
        ei    = embs[mask]
        if len(ei) == 0:
            log(f"  WARN {name}: 0 crops", "WARN"); continue
        centroid = ei.mean(0); centroid /= (np.linalg.norm(centroid) + 1e-8)
        proto_mat[i] = centroid

        # Quality filter — keep crops close to the centroid
        sims   = ei @ centroid
        pos_sims.extend(sims.tolist())
        good   = ei[sims >= QUALITY_THR]
        if len(good) < 3:
            good = ei[np.argsort(-sims)[:max(3, K_EXEMPLARS//3)]]

        # Top-K exemplars
        gs    = good @ centroid
        top_k = min(K_EXEMPLARS, len(good))
        best  = good[np.argsort(-gs)[:top_k]]
        norms = np.linalg.norm(best, axis=1, keepdims=True)
        best  = best / np.where(norms > 1e-8, norms, 1)

        cat, ind_name = parse_folder(name)
        individuals[name] = {
            "class_index": i, "category": cat or "?", "name": ind_name,
            "n_crops": int(len(ei)), "n_exemplars": len(best),
            "mean_intra": round(float(np.mean(sims)), 4),
            "prototype": centroid.tolist(),
            "exemplars": best.tolist(),
        }
        log(f"  {name:<20}: {len(ei):3d} crops → {len(best):2d} exemplars  intra={np.mean(sims):.3f}")

    # Symmetric inter-class similarities (every individual vs all others)
    for i in range(n_cls):
        ei_neg = embs[labs == i]
        if len(ei_neg) == 0: continue
        others = np.delete(proto_mat, i, axis=0)
        if others.shape[0] > 0:
            neg_sims.extend((ei_neg @ others.T).max(1).tolist())

    # Separability
    pos = np.array(pos_sims)
    neg = np.array(neg_sims) if neg_sims else np.zeros(1)
    gap = float(pos.mean() - neg.mean())
    log(f"\n  Positive : {pos.mean():.4f} ± {pos.std():.4f}")
    log(f"  Negative : {neg.mean():.4f} ± {neg.std():.4f}")
    log(f"  Gap      : {gap:.4f}  {'[EXCELLENT]' if gap > 0.7 else '[GOOD]' if gap > 0.5 else '[AVERAGE]'}")

    # Optimal threshold (maximizes F1)
    thresholds = np.linspace(0, 1, 500)
    f1s = []
    for t in thresholds:
        tp = int((pos >= t).sum()); fp = int((neg >= t).sum()); fn = int((pos < t).sum())
        p = tp / (tp+fp+1e-9); r = tp / (tp+fn+1e-9)
        f1s.append(2*p*r/(p+r+1e-9))
    opt_t = float(thresholds[np.argmax(f1s)])
    log(f"  Optimal threshold : {opt_t:.4f}  (F1={max(f1s):.4f})")

    gallery = {
        "version": "1.0", "species": "Gorilla beringei beringei",
        "project": "gorilla_virunga", "created": datetime.now().isoformat(),
        "model": "MegaDescriptor-T-224 + SubCenterArcFace V1",
        "embedding_dim": dim, "similarity_metric": "cosine",
        "unknown_threshold": round(opt_t, 4),
        "separability_gap": round(gap, 4),
        "n_individuals": len(individuals),
        "inference_note": "score = max cosine similarity over all exemplars",
        "individuals": individuals,
    }
    (OUT / "embeddings").mkdir(exist_ok=True)
    GALLERY_JSON.write_text(json.dumps(gallery, separators=(",", ":"), ensure_ascii=False))
    log(f"  Gallery: {GALLERY_JSON.name} ({GALLERY_JSON.stat().st_size/1024:.0f} KB)")
    return opt_t, gap

# ══════════════════════════════════════════════════════════════════════════════
# FINAL BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def benchmark(bb, paths, labels, names, dim, threshold):
    section("Final benchmark")
    bb.eval()
    n_cls = len(names)

    # Load the gallery
    gallery_data = json.loads(GALLERY_JSON.read_text())
    proto = np.zeros((n_cls, dim), dtype=np.float32)
    for name, ind in gallery_data["individuals"].items():
        i = ind["class_index"]
        exemplars = np.array(ind["exemplars"], dtype=np.float32)
        proto[i] = exemplars.mean(0)
        proto[i] /= (np.linalg.norm(proto[i]) + 1e-8)

    # Compute validation embeddings
    dl    = DataLoader(PlainDataset(paths, labels), 64, num_workers=0)
    embs, labs = [], []
    for imgs, lbs in dl:
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy())
        labs.extend(lbs.tolist())
    embs = np.concatenate(embs).astype(np.float32)
    labs = np.array(labs)

    sims = embs @ proto.T                      # [N, n_cls]
    preds = sims.argmax(1)
    max_sims = sims.max(1)

    # Global metrics
    acc_top1 = float((preds == labs).mean())
    top3_ok  = sum(1 for i, lb in enumerate(labs)
                   if lb in np.argsort(-sims[i])[:3])
    acc_top3 = top3_ok / len(labs)
    unk_mask = max_sims < threshold
    unk_rate = float(unk_mask.mean())
    log(f"  Top-1 accuracy      : {acc_top1*100:.2f}%")
    log(f"  Top-3 accuracy      : {acc_top3*100:.2f}%")
    log(f"  Unknown rejection   : {unk_rate*100:.1f}% of predictions below threshold ({threshold:.3f})")

    # Per-individual F1
    known_mask = ~unk_mask
    if known_mask.sum() > 0:
        p_arr, r_arr, f1_arr, _ = precision_recall_fscore_support(
            labs[known_mask], preds[known_mask], labels=list(range(n_cls)),
            average=None, zero_division=0)
    else:
        p_arr = r_arr = f1_arr = np.zeros(n_cls)

    log("\n  Per-individual F1:")
    for i, name in enumerate(names):
        log(f"    {name:<22} P={p_arr[i]:.3f}  R={r_arr[i]:.3f}  F1={f1_arr[i]:.3f}")

    mean_f1 = f1_arr.mean()
    log(f"\n  Mean F1: {mean_f1:.4f}")

    # ── Most confused pairs ────────────────────────────────────────────────────
    cm_abs = confusion_matrix(labs[known_mask], preds[known_mask], labels=list(range(n_cls)))
    np.fill_diagonal(cm_abs, 0)   # ignore the diagonal
    worst_pairs = []
    for _ in range(min(10, n_cls)):
        idx = np.unravel_index(cm_abs.argmax(), cm_abs.shape)
        cnt = int(cm_abs[idx])
        if cnt == 0: break
        worst_pairs.append({
            "real": names[idx[0]], "predicted": names[idx[1]],
            "count": cnt
        })
        cm_abs[idx] = 0   # mask to find the next one

    log("\n  Top confusions (real → predicted, excluding diagonal):")
    for wp in worst_pairs[:5]:
        log(f"    {wp['real']:<22} → {wp['predicted']:<22} : {wp['count']} times")

    # ── F1 by category ────────────────────────────────────────────────────────
    by_cat = defaultdict(list)
    for i, name in enumerate(names):
        cat, _ = parse_folder(name)
        by_cat[cat or "?"].append(float(f1_arr[i]))
    per_cat = {
        cat: {"mean_f1": round(float(np.mean(vs)), 4), "n": len(vs)}
        for cat, vs in sorted(by_cat.items())
    }
    log("\n  Mean F1 by category:")
    for cat, d in per_cat.items():
        log(f"    {cat:<6} : {d['mean_f1']:.3f}  ({d['n']} individuals)")

    # ── Similarity values for threshold diagnostics ───────────────────────────
    sim_pos = sims[np.arange(len(labs)), labs]
    sims_neg = sims.copy()
    sims_neg[np.arange(len(labs)), labs] = -1.0
    sim_neg = sims_neg.max(1)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_benchmark(names, f1_arr, p_arr, r_arr, labs, preds, sims,
                    max_sims, known_mask, acc_top1, acc_top3, mean_f1, threshold)
    plot_confusion_matrices(names, labs, preds, known_mask)

    # ── Structured results for diagnostics.json ───────────────────────────────
    bm_results = {
        "threshold":       round(float(threshold), 5),
        "acc_top1_pct":    round(acc_top1 * 100, 3),
        "acc_top3_pct":    round(acc_top3 * 100, 3),
        "mean_f1":         round(float(mean_f1), 5),
        "unk_rate_pct":    round(unk_rate * 100, 2),
        "n_val_samples":   int(len(labs)),
        "sim_pos_mean":    round(float(sim_pos.mean()), 5),
        "sim_pos_std":     round(float(sim_pos.std()), 5),
        "sim_neg_mean":    round(float(sim_neg.mean()), 5),
        "per_individual":  {
            names[i]: {
                "precision":    round(float(p_arr[i]), 5),
                "recall":       round(float(r_arr[i]), 5),
                "f1":           round(float(f1_arr[i]), 5),
            }
            for i in range(n_cls)
        },
        "per_category":    per_cat,
        "worst_confusions": worst_pairs,
    }

    return acc_top1, mean_f1, bm_results

def _plot_benchmark(names, f1_arr, p_arr, r_arr, labs, preds, sims,
                    max_sims, known_mask, acc1, acc3, mf1, threshold):
    """
    6 benchmark plots on a 3×2 grid:
      [0,:] Per-individual F1 (colored bars)
      [1,0] Normalized confusion matrix
      [1,1] Precision vs Recall per individual (scatter)
      [2,0] Positive vs negative similarity distribution + threshold
      [2,1] F1-score distribution (histogram)
    """
    n_cls = len(names)
    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)
    fig.suptitle(
        f"GorillaIdentifier V1 — Final benchmark\n"
        f"Top-1: {acc1*100:.1f}%  Top-3: {acc3*100:.1f}%  "
        f"Mean F1: {mf1:.3f}  Threshold: {threshold:.3f}",
        fontsize=13, fontweight="bold"
    )

    # ── 1. Per-individual F1 (full-width row) ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    colors = ["#2ecc71" if f >= 0.80 else "#f39c12" if f >= 0.60 else "#e74c3c"
              for f in f1_arr]
    bars = ax1.barh(names, f1_arr, color=colors, edgecolor="none")
    ax1.axvline(mf1, color="white", linestyle="--", lw=1.5, label=f"Mean F1: {mf1:.3f}")
    ax1.axvline(0.80, color="#2ecc71", linestyle=":", lw=1, alpha=0.5, label="Target 0.80")
    # Value on each bar
    for bar, f in zip(bars, f1_arr):
        ax1.text(min(f + 0.01, 1.0), bar.get_y() + bar.get_height()/2,
                 f"{f:.2f}", va="center", fontsize=6, color="white")
    ax1.set_xlabel("F1-score"); ax1.set_xlim(0, 1.05)
    ax1.set_title("Per-individual F1  (green ≥0.80 | orange ≥0.60 | red <0.60)")
    ax1.legend(fontsize=9)

    # ── 2. Confusion matrix ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if known_mask.sum() > 0:
        cm      = confusion_matrix(labs[known_mask], preds[known_mask],
                                   labels=list(range(n_cls)))
        cm_norm = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-8)
        im = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        if n_cls <= 35:
            ax2.set_xticks(range(n_cls))
            ax2.set_xticklabels(names, rotation=90, fontsize=5)
            ax2.set_yticks(range(n_cls))
            ax2.set_yticklabels(names, fontsize=5)
        else:
            ax2.set_xticks([]); ax2.set_yticks([])
        plt.colorbar(im, ax=ax2, fraction=0.046, label="Recall rate")
        ax2.set_title("Confusion matrix (row-normalized)")
        ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    else:
        ax2.text(0.5, 0.5, "No known predictions", ha="center", va="center",
                 transform=ax2.transAxes, color="gray")

    # ── 3. Precision vs Recall scatter ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    sc = ax3.scatter(r_arr, p_arr, c=f1_arr, cmap="RdYlGn", vmin=0, vmax=1,
                     s=60, edgecolors="white", lw=0.5, zorder=3)
    plt.colorbar(sc, ax=ax3, label="F1-score")
    # Annotate the weakest individuals
    bad_idx = np.argsort(f1_arr)[:5]
    for i in bad_idx:
        ax3.annotate(names[i], (r_arr[i], p_arr[i]),
                     fontsize=6, color="white",
                     xytext=(5, 5), textcoords="offset points")
    ax3.plot([0, 1], [0, 1], "w:", lw=1, alpha=0.4)   # P=R diagonal
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
    ax3.set_xlim(-0.05, 1.05); ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Precision vs Recall (color = F1)\n5 weakest individuals annotated")
    ax3.grid(alpha=0.3)

    # ── 4. Similarity distribution (threshold diagnostics) ───────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    # positive sim = similarity to the true prototype of each sample
    sim_pos = sims[np.arange(len(labs)), labs]
    # negative sim = max similarity to the other prototypes
    sims_neg_mat = sims.copy()
    sims_neg_mat[np.arange(len(labs)), labs] = -1.0
    sim_neg = sims_neg_mat.max(1)
    ax4.hist(sim_pos, bins=50, color="#2ecc71", alpha=0.65, label="Positive sim. (true individual)")
    ax4.hist(sim_neg, bins=50, color="#e74c3c", alpha=0.65, label="Negative sim. (best impostor)")
    ax4.axvline(threshold, color="#f1c40f", lw=2, linestyle="--",
                label=f"Gallery threshold ({threshold:.3f})")
    ax4.set_xlabel("Cosine similarity"); ax4.set_ylabel("Number of samples")
    ax4.set_title("Similarity distribution — rejection threshold diagnostic\n"
                  "(good = green and red well separated, threshold between them)")
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # ── 5. F1-score distribution ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    bins = np.linspace(0, 1, 21)
    ax5.hist(f1_arr, bins=bins, color="#3498db", edgecolor="white", alpha=0.85)
    ax5.axvline(mf1, color="#e74c3c", lw=2, linestyle="--", label=f"Mean F1={mf1:.3f}")
    ax5.axvline(0.80, color="#2ecc71", lw=1.5, linestyle=":", label="Target 0.80")
    n_above = int((f1_arr >= 0.80).sum())
    ax5.set_xlabel("F1-score"); ax5.set_ylabel("Number of individuals")
    ax5.set_title(f"F1-score distribution — {n_above}/{n_cls} individuals ≥ 0.80")
    ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

    plt.savefig(BENCHMARK_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Benchmark saved: {BENCHMARK_PNG.name}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA DISTRIBUTION (generated at the start of training)
# ══════════════════════════════════════════════════════════════════════════════
def plot_data_stats(names, labels, excl, ho_names):
    """
    4-panel data distribution — one dedicated file, data_stats.png.
    Generated as soon as crops are loaded, before any training starts.
    """
    counts   = Counter(labels)
    idx_sort = sorted(range(len(names)), key=lambda i: counts[i], reverse=True)
    snames   = [names[i]  for i in idx_sort]
    scounts  = [counts[i] for i in idx_sort]

    # Color by age-sex category
    _CAT_COLOR = {
        "SB": "#e74c3c", "ADF": "#e67e22", "AD": "#e67e22",
        "SAF": "#f39c12", "SAM": "#f1c40f",
        "JUV": "#2ecc71", "BB": "#3498db", "Baby": "#3498db",
        None: "#95a5a6"
    }
    def _color(name):
        cat, _ = parse_folder(name)
        c = _CAT_COLOR.get(cat, "#95a5a6")
        if name in ho_names: c = "#e056fd"  # purple = held-out
        return c

    colors = [_color(n) for n in snames]
    total  = sum(scounts)
    mean_c = float(np.mean(scounts))
    med_c  = float(np.median(scounts))

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e"); ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_color("#333366")
    fig.suptitle(
        f"GorillaIdentifier V1 — Data distribution\n"
        f"{len(names)} training individuals  |  {len(ho_names)} held-out  |  "
        f"{len(excl)} excluded  |  {total:,} total crops",
        fontsize=13, fontweight="bold", color="white"
    )

    # ── 1. Top 20 individuals ─────────────────────────────────────────────────
    ax = axes[0, 0]
    top_n  = min(20, len(snames))
    tn, tc, tcol = snames[:top_n], scounts[:top_n], colors[:top_n]
    bars = ax.barh(tn, tc, color=tcol, edgecolor="none")
    for bar, cnt in zip(bars, tc):
        ax.text(cnt + 1, bar.get_y() + bar.get_height()/2,
                str(cnt), va="center", fontsize=8, color="white")
    ax.invert_yaxis()
    ax.set_xlabel("Number of crops", color="white")
    ax.set_title(f"Top {top_n} individuals", color="white", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8, colors="white")

    # ── 2. Full sorted distribution ───────────────────────────────────────────
    ax = axes[0, 1]
    ax.bar(range(len(snames)), scounts, color=colors, width=1.0, edgecolor="none")
    ax.set_xlabel("Individuals (sorted, descending)", color="white")
    ax.set_ylabel("Number of crops", color="white")
    ax.set_title("Full distribution", color="white", fontweight="bold")
    ax.set_xticks([])
    ax.axhline(mean_c, color="#e74c3c", lw=1.5, linestyle="--",
               label=f"Mean: {mean_c:.1f}")
    ax.axhline(med_c,  color="#3498db", lw=1.5, linestyle="--",
               label=f"Median: {med_c:.1f}")
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")

    # ── 3. Histogram ───────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.hist(scounts, bins=25, color="#2ecc71", edgecolor="#1a1a2e", alpha=0.85)
    ax.axvline(mean_c, color="#e74c3c", lw=2, linestyle="--",
               label=f"Mean: {mean_c:.1f}")
    ax.axvline(med_c,  color="#3498db", lw=2, linestyle="--",
               label=f"Median: {med_c:.1f}")
    ax.set_xlabel("Crops per individual", color="white")
    ax.set_ylabel("Frequency",            color="white")
    ax.set_title("Distribution histogram", color="white", fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")

    # ── 4. Boxplot ─────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    bp = ax.boxplot(scounts, vert=True, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor="#3498db", alpha=0.7, color="white"),
                    whiskerprops=dict(color="white"), capprops=dict(color="white"),
                    medianprops=dict(color="#e74c3c", lw=2),
                    flierprops=dict(marker="o", color="#f39c12", markersize=5, alpha=0.7))
    ax.set_ylabel("Number of crops", color="white")
    ax.set_title("Distribution boxplot",  color="white", fontweight="bold")
    ax.set_xticks([])

    # Category legend (bottom)
    legend_patches = [
        plt.Rectangle((0,0),1,1, color=c, label=cat)
        for cat, c in _CAT_COLOR.items()
        if cat is not None and any(parse_folder(n)[0] == cat for n in names)
    ]
    if ho_names:
        legend_patches.append(plt.Rectangle((0,0),1,1, color="#e056fd", label="Held-out"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches)+1,
               fontsize=10, facecolor="#1a1a2e", labelcolor="white",
               title="Age-sex category", title_fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(DATA_STATS_PNG, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    log(f"  Data stats: {DATA_STATS_PNG.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRICES (large separate figure)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(names, labs, preds, known_mask):
    """
    2 large matrices side by side:
      - Left  : absolute values (number of predictions)
      - Right : row-normalized values (recall percentage)
    Large format (suited for 66 individuals).
    """
    n      = len(names)
    cm_abs = confusion_matrix(labs[known_mask], preds[known_mask], labels=list(range(n)))
    cm_pct = cm_abs.astype(float) / (cm_abs.sum(1, keepdims=True).clip(1) )

    cell = max(0.28, 14.0 / n)   # cell size scaled to the number of individuals
    fig_w = cell * n * 2 + 4
    fig_h = cell * n + 3
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle(
        f"GorillaIdentifier V1 — Confusion matrices  ({n} individuals)\n"
        f"Rows = actual  |  Columns = predicted",
        fontsize=12, fontweight="bold", color="white"
    )

    for ax, cm, title, cmap, fmt in [
        (axes[0], cm_abs, "Absolute values (count)",   "Blues", "d"),
        (axes[1], cm_pct, "Normalized (recall per row)", "RdYlGn", ".0%"),
    ]:
        ax.set_facecolor("#0d0d1a")
        im = ax.imshow(cm, cmap=cmap,
                       vmin=0, vmax=(None if fmt == "d" else 1),
                       aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        # Cell annotations (only if the matrix isn't too large)
        if n <= 40:
            thresh = cm.max() / 2.0 if fmt == "d" else 0.5
            for i in range(n):
                for j in range(n):
                    val  = cm[i, j]
                    txt  = (f"{val:{fmt}}" if fmt == "d" else f"{val:.0%}")
                    col  = "white" if val < thresh else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=max(4, 8 - n//10), color=col)

        fs = max(4, 9 - n//12)
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=fs, color="white")
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=fs, color="white")
        ax.set_xlabel("Predicted class", color="white"); ax.set_ylabel("Actual class", color="white")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(CONFUSION_PNG, dpi=120, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
    log(f"  Confusion matrices: {CONFUSION_PNG.name}  ({CONFUSION_PNG.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# FULL DIAGNOSTICS JSON (every value, atomic write)
# ══════════════════════════════════════════════════════════════════════════════
def save_diagnostics_json(*, meta, data_info, history, bm_results, hyperparams):
    """
    Atomic save: writes to a .tmp file then renames it.
    Crash-resistant mid-write — the previous file stays intact until the end.

    Structure:
      meta         — version, date, duration
      data         — individuals, crops, holdout, excluded
      training     — full epoch-by-epoch history
      benchmark    — F1/P/R per individual, per category, worst pairs, threshold
      hyperparams  — every setting
    """
    diag = {
        "meta": meta,
        "data": data_info,
        "training": {
            "phases": [
                {"name": ph["name"], "epochs": ph["epochs"],
                 "lam_inv": ph["lam_inv"], "lam_sup": ph["lam_sup"],
                 "severity": ph["severity"]}
                for ph in PHASES
            ],
            "best_epoch":  history.get("best_epoch"),
            "best_score":  history.get("best_score"),
            "history_per_epoch": {
                "epoch":          list(range(1, len(history["l_arc"]) + 1)),
                "l_arcface":      [round(v, 5) for v in history["l_arc"]],
                "l_invariance":   [round(v, 5) for v in history["l_inv"]],
                "l_supcon":       [round(v, 5) for v in history["l_sup"]],
                "acc_clean_pct":  [round(v*100, 2) for v in history["acc_c"]],
                "acc_degraded_pct": [round(v*100, 2) for v in history["acc_d"]],
                "separability":   [round(v, 5) for v in history["sep"]],
                "composite":      [round(v, 5) for v in history["comp"]],
                "unk_rate_pct":   [round(v*100, 2) for v in history.get("unk_rate", [])],
            },
        },
        "benchmark": bm_results,
        "hyperparams": hyperparams,
    }

    tmp = DIAGNOSTICS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(DIAGNOSTICS_JSON)   # atomic on every modern OS
    log(f"  Diagnostics JSON: {DIAGNOSTICS_JSON.name}  "
        f"({DIAGNOSTICS_JSON.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
def plot_curves(history):
    """
    6 training plots on a 3×2 grid.
    All metrics per epoch since the start (crash-safe: full history).
    """
    eps = list(range(1, len(history["l_arc"]) + 1))
    # Add phase-boundary markers (vertical lines)
    phase_ends = []
    for ph in PHASES:
        prev = phase_ends[-1] if phase_ends else 0
        phase_ends.append(prev + ph["epochs"])

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("GorillaIdentifier V1 — Training curves", fontsize=14, fontweight="bold")

    def _phase_lines(ax):
        for i, pe in enumerate(phase_ends[:-1]):
            ax.axvline(pe + 0.5, color="#aaaaaa", linestyle=":", lw=1, alpha=0.6)
        ax.set_xlim(0.5, max(eps) + 0.5)

    # ── 1. ArcFace loss ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(eps, history["l_arc"], color="#e74c3c", lw=1.5, label="L_ArcFace")
    _phase_lines(ax)
    ax.set_title("ArcFace loss (should decrease steadily)")
    ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend()

    # ── 2. Auxiliary losses ────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(eps, history["l_sup"], color="#9b59b6", lw=1.5, label="L_SupCon (separability)")
    ax.plot(eps, history["l_inv"], color="#3498db", lw=1.5, label="L_invariance (clean≈degraded)")
    _phase_lines(ax)
    ax.set_title("Auxiliary losses")
    ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend()

    # ── 3. Clean vs degraded accuracy ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(eps, [v*100 for v in history["acc_c"]], color="#2ecc71", lw=2, label="Clean")
    ax.plot(eps, [v*100 for v in history["acc_d"]], color="#f39c12", lw=2, label="Degraded (severity 0.7)")
    _phase_lines(ax)
    ax.axhline(90, color="#2ecc71", linestyle="--", lw=1, alpha=0.4)
    ax.axhline(70, color="#f39c12", linestyle="--", lw=1, alpha=0.4)
    ax.set_title("Identification accuracy (Top-1)")
    ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend()

    # ── 4. Separability gap ────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(eps, history["sep"], color="#1abc9c", lw=2, label="Separability gap")
    _phase_lines(ax)
    ax.axhline(0.7, color="#1abc9c", linestyle="--", lw=1, alpha=0.5, label="Excellent threshold (0.7)")
    ax.axhline(0.5, color="#f39c12", linestyle="--", lw=1, alpha=0.5, label="Good threshold (0.5)")
    ax.set_title("Inter-individual separability (intra/inter gap)")
    ax.set_ylabel("gap"); ax.set_ylim(0, None); ax.grid(alpha=0.3); ax.legend()

    # ── 5. Unknown rejection (unk_rate) ───────────────────────────────────────
    ax = axes[2, 0]
    ur = [v*100 for v in history.get("unk_rate", [])]
    if ur:
        ax.plot(eps[:len(ur)], ur, color="#e67e22", lw=2, label="% held-out rejected")
        _phase_lines(ax)
        ax.axhline(80, color="#e67e22", linestyle="--", lw=1, alpha=0.5, label="Target 80%")
        ax.set_ylim(0, 105)
    else:
        ax.text(0.5, 0.5, "No held-out data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title("Pseudo-unknown rejection rate (held-out)")
    ax.set_ylabel("%"); ax.grid(alpha=0.3); ax.legend()

    # ── 6. Composite score ─────────────────────────────────────────────────────
    ax = axes[2, 1]
    ax.plot(eps, history["comp"], color="#f1c40f", lw=2, label="Composite score")
    # Mark the best epoch
    best_idx = int(np.argmax(history["comp"]))
    ax.scatter([eps[best_idx]], [history["comp"][best_idx]],
               color="#f1c40f", s=80, zorder=5, label=f"Best (ep {eps[best_idx]})")
    _phase_lines(ax)
    ax.set_title("Composite score (0.25×clean + 0.35×degraded + 0.25×sep + 0.15×rejection)")
    ax.set_ylabel("score"); ax.grid(alpha=0.3); ax.legend()

    # Phase legend at the bottom
    phase_labels = "  |  ".join(
        f"{'ABCD'[i]} = {ph['name'].split('—')[0].strip()}"
        for i, ph in enumerate(PHASES)
    )
    fig.text(0.5, 0.01, f"Phases: {phase_labels}  (dotted lines = end of phase)",
             ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(CURVES_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Curves saved: {CURVES_PNG.name}  ({len(eps)} epochs)")

# ══════════════════════════════════════════════════════════════════════════════
# DATA SCAN
# ══════════════════════════════════════════════════════════════════════════════
def scan_photos(min_photos):
    section("Data scan")
    source = CROPS_DIR if CROPS_DIR.exists() and any(CROPS_DIR.iterdir()) else PHOTOS_DIR
    label  = "crops" if source == CROPS_DIR else "photos"
    log(f"  Source: {source} ({label})")

    if not source.exists() or not any(source.iterdir()):
        log("  ERROR: no folder found. Check data/photos/ or data/crops/known/", "ERROR")
        return {}, {}

    included = {}; excluded = {}
    for folder in sorted(source.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"): continue
        imgs = [f for f in folder.iterdir() if f.suffix in IMG_EXTS]
        cat, name = parse_folder(folder.name)
        entry = {"folder": folder, "n": len(imgs), "cat": cat, "name": name}
        if len(imgs) < min_photos:
            excluded[folder.name] = entry
        else:
            included[folder.name] = entry

    total = sum(e["n"] for e in included.values())
    log(f"\n  Included individuals : {len(included)}", always=True)
    log(f"  Total {label:<8}      : {total}", always=True)
    log(f"  Median                : {np.median([e['n'] for e in included.values()]):.0f}", always=True)
    log(f"  Mean                  : {np.mean([e['n'] for e in included.values()]):.1f}", always=True)
    log(f"\n  Excluded individuals (< {min_photos} {label}):", always=True)
    if excluded:
        for n, e in sorted(excluded.items(), key=lambda x: x[1]["n"]):
            log(f"    {n:<25} {e['n']} {label}", always=True)
    else:
        log("    (none)", always=True)

    by_cat = defaultdict(list)
    for n, e in included.items():
        by_cat[e["cat"] or "?"].append(e["n"])
    log("\n  By category:", always=True)
    for cat in sorted(by_cat):
        ns = by_cat[cat]
        log(f"    {cat:<6} : {len(ns):3d} individuals  total={sum(ns)}"
            f"  median={np.median(ns):.0f}", always=True)

    return included, excluded

# ══════════════════════════════════════════════════════════════════════════════
# CROP EXTRACTION (optional — --extract)
# ══════════════════════════════════════════════════════════════════════════════
def extract_crops(min_photos):
    section("Crop extraction (YOLO)")
    if not YOLO_MODEL.exists():
        log(f"  ERROR: YOLO model not found at {YOLO_MODEL}", "ERROR")
        log("  Download the gorilla model with: python models/download_models.py", "WARN")
        log("  Place it at models/yolo_gorilla.pt and rerun with --extract", "WARN")
        return False

    try:
        from ultralytics import YOLO
    except ImportError:
        log("  ERROR: ultralytics missing. pip install ultralytics==8.2.0", "ERROR")
        return False

    log(f"  Loading YOLO: {YOLO_MODEL.name}...", always=True)
    model = YOLO(str(YOLO_MODEL))
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    MARGIN = 0.05

    folders = sorted([f for f in PHOTOS_DIR.iterdir() if f.is_dir()])
    log(f"  {len(folders)} folders to process...", always=True)

    # Load existing JSON so manual reviews aren't overwritten
    existing_json = {}
    if CROPS_JSON.exists():
        try:
            existing_json = json.loads(CROPS_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    new_entries = dict(existing_json)

    for i, folder in enumerate(folders, 1):
        imgs = [f for f in folder.iterdir() if f.suffix in IMG_EXTS]
        if len(imgs) < min_photos:
            log(f"  [{i:2d}/{len(folders)}] Skipped: {folder.name} ({len(imgs)} photos)", always=True)
            continue
        individu = folder.name
        out_dir = CROPS_DIR / individu
        # Clean the existing folder to avoid orphan crops from a previous extraction
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        n_ok = 0
        for img_path in sorted(imgs):
            try:
                img = Image.open(img_path).convert("RGB")
                W, H = img.size
                results = model.predict(str(img_path), conf=0.25, verbose=False)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf_val = float(box.conf[0])
                    mx = MARGIN * (x2-x1); my = MARGIN * (y2-y1)
                    x1c = max(0, int(x1-mx)); y1c = max(0, int(y1-my))
                    x2c = min(W, int(x2+mx)); y2c = min(H, int(y2+my))
                    crop = img.crop((x1c, y1c, x2c, y2c)).resize((224, 224), Image.BICUBIC)
                    stem = img_path.stem
                    crop_name = f"{stem}_crop{n_ok:02d}.jpg"
                    crop_path = out_dir / crop_name
                    crop.save(crop_path)
                    # Write JSON entry (key: "IndividualName/stem_cropXX")
                    jkey = f"{individu}/{crop_path.stem}"
                    if jkey not in new_entries:  # don't overwrite a manual review
                        try:
                            photo_rel = img_path.relative_to(REPO).as_posix()
                        except ValueError:
                            photo_rel = str(img_path)
                        try:
                            crop_rel = crop_path.relative_to(REPO).as_posix()
                        except ValueError:
                            crop_rel = str(crop_path)
                        new_entries[jkey] = {
                            "individu":     individu,
                            "stem":         crop_path.stem,
                            "photo_source": photo_rel,
                            "crop_file":    crop_rel,
                            "crop_x1": x1c, "crop_y1": y1c,
                            "crop_x2": x2c, "crop_y2": y2c,
                            "yolo_conf":    round(conf_val, 3),
                            "statut":       "auto",
                            "source_type":  "known",
                        }
                    n_ok += 1
            except Exception as e:
                log(f"    Error {img_path.name}: {e}", "WARN")
        log(f"  [{i:2d}/{len(folders)}] {individu:<25} : {len(imgs)} photos → {n_ok} crops", always=True)

    # Save the JSON
    CROPS_JSON.parent.mkdir(parents=True, exist_ok=True)
    CROPS_JSON.write_text(json.dumps(new_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"  JSON saved: {len(new_entries)} entries → {CROPS_JSON.name}", always=True)
    log("  Extraction complete. Run python common/review_crops.py to review.", always=True)
    return True

# ══════════════════════════════════════════════════════════════════════════════
# REGEN JSON — rebuilds data/crops.json from existing crops
# ══════════════════════════════════════════════════════════════════════════════
def regen_crops_json():
    """
    Regenerates data/crops.json from crops already extracted in data/crops/known/.
    Re-runs YOLO on the original photos to recover the bounding boxes.

    NOTE: _crop{N:02d} is a GLOBAL counter for the folder, not the box index
    within the photo. Crops are grouped by original photo, then each crop is
    matched to a YOLO box in order (box 0 → the photo's first crop, etc.)
    """
    import re as _re
    from collections import defaultdict
    section("Regenerating data/crops.json")

    if not CROPS_DIR.exists():
        log("  data/crops/known/ not found. Run --extract first.", "ERROR")
        return False
    if not YOLO_MODEL.exists():
        log(f"  ERROR: YOLO model not found at {YOLO_MODEL}", "ERROR")
        return False
    try:
        from ultralytics import YOLO
    except ImportError:
        log("  ERROR: ultralytics missing. pip install ultralytics==8.2.0", "ERROR")
        return False

    log(f"  Loading YOLO: {YOLO_MODEL.name}...", always=True)
    model   = YOLO(str(YOLO_MODEL))
    MARGIN  = 0.05
    entries = {}
    folders = sorted([f for f in CROPS_DIR.iterdir() if f.is_dir()])
    log(f"  {len(folders)} folders to process...", always=True)

    for i, crop_folder in enumerate(folders, 1):
        individu     = crop_folder.name
        photo_folder = PHOTOS_DIR / individu
        crop_files   = sorted(
            p for p in crop_folder.iterdir() if p.suffix in IMG_EXTS
        )

        # Group crops by original photo (same orig_stem),
        # sorted by their numeric suffix to preserve extraction order
        photo_groups = defaultdict(list)
        for crop_path in crop_files:
            m = _re.match(r'^(.+)_crop(\d+)$', crop_path.stem)
            if not m:
                continue
            orig_stem = m.group(1)
            global_idx = int(m.group(2))
            photo_groups[orig_stem].append((global_idx, crop_path))

        # Sort each group by global_idx
        for orig_stem in photo_groups:
            photo_groups[orig_stem].sort(key=lambda x: x[0])

        n_ok = 0
        for orig_stem, crops_for_photo in sorted(photo_groups.items()):
            # Find the original photo
            orig_photo = None
            for ext in IMG_EXTS:
                cand = photo_folder / f"{orig_stem}{ext}"
                if cand.exists():
                    orig_photo = cand
                    break
            if orig_photo is None:
                continue

            # Run YOLO once per photo
            try:
                img_pil = Image.open(orig_photo).convert("RGB")
                W, H    = img_pil.size
                results = model.predict(str(orig_photo), conf=0.25, verbose=False)
                boxes   = results[0].boxes
            except Exception as e:
                log(f"    Error {orig_stem}: {e}", "WARN")
                continue

            # Match each crop to a box, in order
            for box_i, (_, crop_path) in enumerate(crops_for_photo):
                if len(boxes) == 0:
                    continue
                # Use the corresponding box, or box 0 if too few were detected
                box = boxes[min(box_i, len(boxes) - 1)]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0])
                mx = MARGIN * (x2-x1); my = MARGIN * (y2-y1)
                x1c = max(0, int(x1-mx)); y1c = max(0, int(y1-my))
                x2c = min(W, int(x2+mx)); y2c = min(H, int(y2+my))

                try: photo_rel = orig_photo.relative_to(REPO).as_posix()
                except ValueError: photo_rel = str(orig_photo)
                try: crop_rel = crop_path.relative_to(REPO).as_posix()
                except ValueError: crop_rel = str(crop_path)

                jkey = f"{individu}/{crop_path.stem}"
                entries[jkey] = {
                    "individu":     individu,
                    "stem":         crop_path.stem,
                    "photo_source": photo_rel,
                    "crop_file":    crop_rel,
                    "crop_x1": x1c, "crop_y1": y1c,
                    "crop_x2": x2c, "crop_y2": y2c,
                    "yolo_conf":    round(conf_val, 3),
                    "statut":       "auto",
                    "source_type":  "known",
                }
                n_ok += 1

        log(f"  [{i:2d}/{len(folders)}] {individu:<25} : {n_ok} entries", always=True)

    CROPS_JSON.parent.mkdir(parents=True, exist_ok=True)
    CROPS_JSON.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"  JSON saved: {len(entries)} entries → {CROPS_JSON.name}", always=True)
    log("  Run python common/review_crops.py to review.", always=True)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# RICH LIVE TABLE
# ══════════════════════════════════════════════════════════════════════════════
def make_table(st, pc, g_ep, best_ep, best_val):
    t = Table.grid(padding=1)
    t.add_column(style="bold cyan"); t.add_column()
    t.add_row("Phase",       pc["name"])
    t.add_row("Epoch",       f"{st.get('ep_in','?')}/{pc['epochs']}  (global {g_ep}/{TOTAL_EPOCHS})")
    t.add_row("Batch",       f"{st.get('batch',0)}/{st.get('total_b',1)}")
    t.add_row("L_arcface",   f"{st.get('l_arc',0):.4f}")
    t.add_row("L_SupCon",    f"{st.get('l_sup',0):.4f}  (λ={pc['lam_sup']:.2f})")
    t.add_row("L_invariance",f"{st.get('l_inv',0):.4f}  (λ={pc['lam_inv']:.2f})")
    t.add_row("─"*20, "─"*28)
    t.add_row("Acc clean",   f"{st.get('acc_c',0)*100:.2f}%")
    t.add_row("Acc degraded",f"{st.get('acc_d',0)*100:.2f}%")
    t.add_row("Separability",f"{st.get('sep',0):.4f}  {'[EXCELLENT]' if st.get('sep',0)>0.7 else ''}")
    unk = st.get('unk_rate')
    t.add_row("Unknown rejection", f"{unk*100:.1f}%  {'[HIGH]' if unk and unk>0.8 else ''}" if unk is not None else "N/A")
    t.add_row("Composite",   f"{st.get('comp',0):.4f}" + (" [BEST]" if st.get("is_best") else ""))
    t.add_row("─"*20, "─"*28)
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated()/1e9
        total = torch.cuda.get_device_properties(0).total_memory/1e9
        t.add_row("GPU VRAM",  f"{used:.1f}/{total:.1f} GB")
    t.add_row("ETA batch",   str(timedelta(seconds=st.get("eta_b",0))))
    t.add_row("Best",        f"epoch {best_ep}  score {best_val:.4f}")
    return Panel(t, title="[bold green]GorillaIdentifier V1[/bold green]", border_style="green")

# ══════════════════════════════════════════════════════════════════════════════
# IN-TRAINING SEPARABILITY (fast, computed on 200 crops)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def quick_separability(bb, paths, labels, n_classes, dim, n_sample=200):
    bb.eval()
    idx = np.random.choice(len(paths), min(n_sample, len(paths)), replace=False)
    sp = [paths[i] for i in idx]; sl = [labels[i] for i in idx]
    ds = PlainDataset(sp, sl)
    dl = DataLoader(ds, 64, num_workers=0)
    embs, labs = [], []
    for imgs, lbs in dl:
        embs.append(F.normalize(bb(imgs.to(DEVICE).float()), dim=1).cpu().numpy())
        labs.extend(lbs.tolist())
    embs = np.concatenate(embs).astype(np.float32)
    labs = np.array(labs)
    # Gap = mean(intra) - mean(inter)
    proto = np.zeros((n_classes, dim), dtype=np.float32)
    cnt   = np.zeros(n_classes)
    for e, l in zip(embs, labs):
        proto[l] += e; cnt[l] += 1
    for c in range(n_classes):
        if cnt[c] > 0: proto[c] /= (np.linalg.norm(proto[c]) + 1e-8)
    sims = embs @ proto.T
    pos = np.array([sims[i, labs[i]] for i in range(len(labs))])
    neg = np.array([np.delete(sims[i], labs[i]).max() for i in range(len(labs))])
    return float(pos.mean() - neg.mean())

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    section(f"GorillaIdentifier V1 — {'DRY RUN ' if DRY else ''}starting {datetime.now():%Y-%m-%d %H:%M}")
    log(f"  Device : {DEVICE}")
    log(f"  Batch  : {BATCH}")
    log(f"  Output : {OUT}")

    # ── Reset (--reset) ───────────────────────────────────────────────────────
    if ARGS.reset:
        for f in [CKPT_RESUME, CKPT_BEST]:
            if f.exists():
                f.unlink()
                log(f"  Deleted: {f.name}", always=True)
        log("  Clean restart — all checkpoints erased.", always=True)

    # ── Scan ──────────────────────────────────────────────────────────────────
    included, excluded = scan_photos(ARGS.min_photos)
    if not included:
        log("No valid data. Stopping.", "ERROR"); return

    # ── Extraction (--extract) ────────────────────────────────────────────────
    if ARGS.extract:
        ok = extract_crops(ARGS.min_photos)
        if not ok: return
        log("", always=True)
        log("  Crops extracted. Optional review: python common/review_crops.py", always=True)
        log("  Rerun without --extract to train: python v1_megadesc_arcface/train.py", always=True)
        return

    # ── Regen JSON (--regen-json) ─────────────────────────────────────────────
    if ARGS.regen_json:
        regen_crops_json()
        return

    # ── Load crops ────────────────────────────────────────────────────────────
    source = CROPS_DIR if (CROPS_DIR.exists() and any(
        d for d in CROPS_DIR.iterdir() if d.is_dir())) else None
    if source is None:
        log("data/crops/known/ is empty. Run --extract first.", "ERROR"); return

    section("Loading crops")
    all_paths, all_labels, names, excl = load_crops(CROPS_DIR, ARGS.min_photos)
    n_classes = len(names)
    log(f"  {n_classes} individuals — {len(all_paths):,} crops")
    if excl:
        log(f"  Excluded (< {ARGS.min_photos} crops):")
        for n, k in sorted(excl, key=lambda x: x[1]):
            log(f"    {n:<25} {k} crops")

    if n_classes < 2:
        log("Need at least 2 individuals to train.", "ERROR"); return

    # ── Pseudo-unknowns held out (simulates unknown rejection without wild images) ─
    # N individuals with the most crops are removed: the model NEVER sees them
    # during training → a genuinely clean rejection test.
    n_ho = min(N_HOLDOUT, max(0, n_classes - 15))   # keep at least 15 known individuals in training
    unk_crops_p = []
    ho_names    = []
    if n_ho > 0:
        crops_per_ind = Counter(all_labels)
        # Prefer individuals with the most crops (more reliable test)
        sorted_by_crops = sorted(range(n_classes),
                                 key=lambda i: crops_per_ind[i], reverse=True)
        holdout_set = set(sorted_by_crops[:n_ho])
        ho_names    = [names[i] for i in sorted(holdout_set)]

        # Split known vs held-out, remap labels
        kp = [p for p, l in zip(all_paths, all_labels) if l not in holdout_set]
        kl_orig = [l for l in all_labels if l not in holdout_set]
        old2new = {old: new for new, old in enumerate(sorted(set(kl_orig)))}
        kl      = [old2new[l] for l in kl_orig]
        known_names = [names[i] for i in sorted(set(kl_orig))]
        n_known = len(known_names)
        unk_crops_p = [p for p, l in zip(all_paths, all_labels) if l in holdout_set]

        log(f"  Held-out pseudo-unknowns ({n_ho}): {ho_names}", always=True)
        log(f"    → {len(unk_crops_p)} crops excluded from training, used to test rejection", always=True)
        log(f"  Training on: {n_known} individuals  {len(kp)} crops", always=True)
    else:
        kp = all_paths; kl = all_labels; known_names = names; n_known = n_classes
        log("  No held-out set (too few individuals to remove any)")

    # ── Data stats (generated before training, no need to wait until the end) ─
    plot_data_stats(names, all_labels, excl, ho_names)

    # Check consistency with an existing checkpoint
    if CKPT_RESUME.exists():
        try:
            ck_names = torch.load(str(CKPT_RESUME), map_location="cpu",
                                  weights_only=False).get("names", [])
            if ck_names != known_names:
                log("  ! Incompatible checkpoint (individuals changed) — clean restart", "WARN")
                CKPT_RESUME.unlink()
                if CKPT_BEST.exists(): CKPT_BEST.unlink()
        except Exception:
            pass

    # Train/val split (stratified on known individuals only)
    idx = list(range(len(kp)))
    tr_idx, va_idx = train_test_split(idx, test_size=VAL_RATIO,
                                      stratify=kl, random_state=SEED)
    tr_p = [kp[i] for i in tr_idx]; tr_l = [kl[i] for i in tr_idx]
    va_p = [kp[i] for i in va_idx]; va_l = [kl[i] for i in va_idx]

    # Wild (background class — optional, unused if the folder is absent)
    has_wild = (not ARGS.no_wild and WILD_DIR.exists()
                and any(f for f in WILD_DIR.iterdir() if f.suffix in IMG_EXTS))
    n_all_classes = n_known + (1 if has_wild else 0)
    if has_wild:
        wild_files = [f for f in WILD_DIR.iterdir() if f.suffix in IMG_EXTS]
        log(f"  Background class: {len(wild_files):,} wild images")
    else:
        log(f"  Background class: absent (unknown rejection via held-out pseudo-unknowns)")

    # ── Benchmark-only ────────────────────────────────────────────────────────
    if ARGS.benchmark_only:
        if not GALLERY_JSON.exists():
            log("Gallery not found. Train the model first.", "ERROR"); return
        if not CKPT_BEST.exists():
            log("Model not found (best.pt). Train first.", "ERROR"); return
        bb, dim = load_backbone()
        ck = torch.load(str(CKPT_BEST), map_location=DEVICE, weights_only=False)
        bb.load_state_dict(ck["backbone_state"]); bb = bb.to(DEVICE)
        threshold = json.loads(GALLERY_JSON.read_text())["unknown_threshold"]
        _, _, _ = benchmark(bb, va_p, va_l, known_names, dim, threshold)
        return

    # ── Backbone + ArcFace ────────────────────────────────────────────────────
    section("Backbone + ArcFace")
    backbone, emb_dim = load_backbone()
    backbone = backbone.to(DEVICE)

    k_list = []
    for name in known_names:
        cat, _ = parse_folder(name)
        k_list.append(k_subcenters(cat))
    if has_wild:
        k_list.append(K_WILD)
    arc = SubCenterArcFace(emb_dim, n_all_classes, k_list, ARC_SCALE, ARC_MARGIN).to(DEVICE)
    log(f"  Classes: {n_known} known individuals + {'1 wild' if has_wild else '0 wild'} = {n_all_classes}")
    log(f"  K adults={sum(1 for k in k_list[:n_known] if k==1)}"
        f"  K JUV/BB={sum(1 for k in k_list[:n_known] if k==2)}"
        f"{'  K wild='+str(K_WILD) if has_wild else ''}")
    log(f"  ArcFace margin={ARC_MARGIN}  scale={ARC_SCALE}")

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    start_phase = 0; start_ep = 0; g_ep = 0
    best_val = -999.0; best_ep = 0
    history = {"l_arc":[], "l_inv":[], "l_sup":[], "acc_c":[], "acc_d":[], "sep":[], "comp":[], "unk_rate":[]}

    if CKPT_RESUME.exists():
        section("Resuming from checkpoint")
        ck = torch.load(str(CKPT_RESUME), map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(ck["backbone_state"])
        arc.load_state_dict(ck["arc_state"])
        start_phase = ck["phase"]; start_ep = ck["ep"] + 1; g_ep = ck["g_ep"]
        best_val = ck["best_val"]; best_ep = ck["best_ep"]
        history = ck.get("history", history)
        # Compatibility: add keys missing from older checkpoints
        for _k in ["l_arc","l_inv","l_sup","acc_c","acc_d","sep","comp","unk_rate"]:
            history.setdefault(_k, [])
        log(f"  Resuming phase {start_phase}, epoch {start_ep}, best={best_val:.4f}")
        if start_ep >= PHASES[start_phase]["epochs"]:
            start_phase += 1; start_ep = 0
            if start_phase >= len(PHASES):
                log("  All phases complete — gallery + benchmark")
                backbone.load_state_dict(
                    torch.load(str(CKPT_BEST), map_location=DEVICE, weights_only=False)["backbone_state"])
                thr, gap = build_gallery(backbone, kp, kl, known_names, emb_dim)
                _, _, _ = benchmark(backbone, va_p, va_l, known_names, emb_dim, thr)
                return

    # Shared state for crash-safety
    _cur = {"bb": backbone, "arc": arc, "opt": None, "sched": None,
            "phase": start_phase, "ep": start_ep, "g_ep": g_ep}

    def _emrg(reason="interrupt"):
        log(f"[EMERGENCY] phase={_cur['phase']} ep={_cur['ep']} — saving...", "WARN")
        save_resume(_cur["bb"], _cur["arc"],
                    _cur["opt"] or _mk_opt(PHASES[_cur["phase"]]), _cur["sched"],
                    _cur["phase"], _cur["ep"]-1, _cur["g_ep"]-1,
                    best_val, best_ep, history, known_names)  # known_names, not names
        log("[EMERGENCY] saved OK", "WARN")
    global _save_fn; _save_fn = _emrg

    def _mk_opt(pc):
        pg = [{"params": arc.parameters(), "lr": pc["lr_h"]}]
        if not pc["freeze"]:
            pg.insert(0, {"params": backbone.parameters(), "lr": pc["lr_bb"]})
        return optim.AdamW(pg, weight_decay=1e-4)

    def _mk_sched(opt, pc, done):
        total = pc["epochs"]; warmup = min(3, total//4)
        def _lr(step):
            if step < warmup: return (step+1)/max(warmup, 1)
            prog = (step-warmup)/max(total-warmup, 1)
            return max(0.01, 0.5*(1+math.cos(math.pi*prog)))
        s = optim.lr_scheduler.LambdaLR(opt, _lr)
        for _ in range(done): s.step()
        return s

    # ── Phase loop ────────────────────────────────────────────────────────────
    live_state = {}
    patience_c = 0

    for phase_idx in range(start_phase, len(PHASES)):
        pc      = PHASES[phase_idx]
        ep_from = start_ep if phase_idx == start_phase else 0
        patience_c = 0
        section(pc["name"])

        for p in backbone.parameters():
            p.requires_grad_(not pc["freeze"])

        opt   = _mk_opt(pc)
        sched = _mk_sched(opt, pc, ep_from)
        _cur["opt"] = opt; _cur["sched"] = sched

        # Restore optimizer if resuming within this phase
        if phase_idx == start_phase and CKPT_RESUME.exists() and ep_from > 0:
            ck2 = torch.load(str(CKPT_RESUME), map_location="cpu", weights_only=False)
            try:
                opt.load_state_dict(ck2["opt_state"])
                if ck2.get("sched_state") and sched:
                    sched.load_state_dict(ck2["sched_state"])
                log("  Optimizer restored")
            except Exception as e:
                log(f"  Optimizer not restored ({e}) — starting fresh", "WARN")

        live_ctx = Live(console=_console, refresh_per_second=4) if RICH else None
        if live_ctx: live_ctx.start()

        _best_box = [best_val]; _bep_box = [best_ep]; _gep_box = [g_ep]
        refresh_fn = None
        if live_ctx:
            def _ref():
                live_ctx.update(make_table(live_state, pc, _gep_box[0], _bep_box[0], _best_box[0]))
            refresh_fn = _ref

        for ep in range(ep_from, pc["epochs"]):
            if _interrupt: break
            g_ep += 1
            _cur["phase"] = phase_idx; _cur["ep"] = ep; _cur["g_ep"] = g_ep
            _gep_box[0] = g_ep; _bep_box[0] = best_ep; _best_box[0] = best_val
            live_state["ep_in"] = ep+1

            # Build loaders for this epoch
            tr_ds  = PairDataset(tr_p, tr_l, pc["severity"])
            tr_w   = [1.0/Counter(tr_l)[l] for l in tr_l]
            tr_dl  = DataLoader(tr_ds, BATCH,
                                sampler=WeightedRandomSampler(tr_w, len(tr_w), True),
                                num_workers=0, pin_memory=False)
            wild_dl = None
            if has_wild and pc.get("wild_n", 0) > 0:
                wds = WildDataset(WILD_DIR, pc["wild_n"], n_known)  # wild label = n_known
                if len(wds) > 0:
                    wild_dl = DataLoader(wds, BATCH, shuffle=True,
                                         num_workers=0, pin_memory=False)

            loaders = [tr_dl, wild_dl]

            # Train
            t0ep = time.time()
            la, li, ls = train_epoch(backbone, arc, loaders, opt, sched, pc,
                                     live_state, refresh_fn)
            ep_t = time.time() - t0ep

            # Validation (held-out pseudo-unknowns passed in to test rejection)
            acc_c, acc_d, unk_rate = validate(backbone, tr_p, tr_l, va_p, va_l,
                                              n_known, emb_dim, unk_p=unk_crops_p)
            sep = quick_separability(backbone, tr_p, tr_l, n_known, emb_dim)

            # Composite score (held-out rejection included when available)
            # acc_c: clean accuracy  | acc_d: degraded accuracy
            # sep  : separability    | unk_rate: pseudo-unknown rejection rate
            _ur = unk_rate if unk_rate is not None else 0.5  # 0.5 = neutral if no held-out set
            comp = (0.25 * acc_c
                  + 0.35 * acc_d
                  + 0.25 * min(max(sep / 1.0, 0), 1)
                  + 0.15 * _ur)
            is_best = comp > best_val
            if is_best:
                best_val = comp; best_ep = g_ep
                save_best(backbone, arc, known_names, emb_dim, g_ep, comp)

            for k, v in zip(["l_arc","l_inv","l_sup","acc_c","acc_d","sep","comp","unk_rate"],
                            [la, li, ls, acc_c, acc_d, sep, comp, _ur]):
                history[k].append(v)

            live_state.update({
                "acc_c": acc_c, "acc_d": acc_d, "sep": sep,
                "unk_rate": unk_rate, "comp": comp, "is_best": is_best,
            })

            # Save checkpoint every epoch (crash-safe)
            save_resume(backbone, arc, opt, sched, phase_idx, ep, g_ep,
                        best_val, best_ep, history, known_names)

            if RICH and live_ctx:
                live_ctx.update(make_table(live_state, pc, g_ep, best_ep, best_val))
            else:
                star = " *" if is_best else ""
                _ur_str = f" unk={_ur*100:.0f}%" if unk_rate is not None else ""
                log(f"  Ep {g_ep:3d} | arc={la:.3f} sup={ls:.3f} inv={li:.3f} "
                    f"| clean={acc_c*100:.1f}% deg={acc_d*100:.1f}%{_ur_str} "
                    f"| sep={sep:.3f} | comp={comp:.4f}{star} | {ep_t:.0f}s")

            # Early stopping in phase D
            if pc.get("early_stop"):
                if is_best: patience_c = 0
                else:
                    patience_c += 1
                    if patience_c >= pc["patience"]:
                        log(f"  Early stopping at epoch {g_ep}"); break

        if live_ctx: live_ctx.stop()
        if _interrupt: break

    # ── Post-training ─────────────────────────────────────────────────────────
    section("Post-training")
    if CKPT_BEST.exists():
        ck = torch.load(str(CKPT_BEST), map_location=DEVICE, weights_only=False)
        backbone.load_state_dict(ck["backbone_state"])
        log(f"  Best model loaded — epoch {ck['epoch']}  score {ck['val']:.4f}")

    if history["l_arc"]: plot_curves(history)

    thr, gap = build_gallery(backbone, kp, kl, known_names, emb_dim)
    _, _, bm_results = benchmark(backbone, va_p, va_l, known_names, emb_dim, thr)

    # ── Full diagnostics JSON (atomic) ────────────────────────────────────────
    total_t = time.time() - t_start
    crops_per_ind = Counter(all_labels)

    data_info = {
        "total_individuals_raw": len(names) + len(excl),
        "n_known":     n_known,
        "n_holdout":   len(ho_names),
        "n_excluded":  len(excl),
        "holdout_names":  ho_names,
        "excluded_names": [n for n, _ in excl],
        "total_crops":    len(all_paths),
        "train_crops":    len(tr_p),
        "val_crops":      len(va_p),
        "unk_crops":      len(unk_crops_p),
        "per_individual": {
            names[i]: {
                "category": parse_folder(names[i])[0] or "?",
                "n_crops":  int(crops_per_ind[i]),
                "in_training": names[i] in known_names,
                "is_holdout":  names[i] in ho_names,
            }
            for i in range(len(names))
        },
    }

    history_ext = dict(history)
    history_ext["best_epoch"] = best_ep
    history_ext["best_score"] = round(best_val, 5)

    hyperparams = {
        "arc_margin": ARC_MARGIN, "arc_scale": ARC_SCALE,
        "batch_size": BATCH, "img_size": IMG_SIZE,
        "val_ratio": VAL_RATIO, "seed": SEED,
        "n_holdout": N_HOLDOUT,
        "total_epochs": g_ep,
        "phases": [
            {"name": ph["name"], "epochs": ph["epochs"],
             "lam_inv": ph["lam_inv"], "lam_sup": ph["lam_sup"],
             "severity": ph["severity"]}
            for ph in PHASES
        ],
    }

    meta = {
        "version": "v1-gorilla",
        "generated": datetime.now().isoformat(),
        "training_min": round(total_t / 60, 1),
        "dry_run": DRY,
        "separability_gap": round(gap, 5),
    }

    bm_results["separability_gap"] = round(gap, 5)

    save_diagnostics_json(
        meta=meta,
        data_info=data_info,
        history=history_ext,
        bm_results=bm_results,
        hyperparams=hyperparams,
    )

    # Legacy report (short, for backward compatibility)
    REPORT_JSON.write_text(json.dumps({
        "version": "v1-gorilla", "generated": meta["generated"],
        "training_min": meta["training_min"], "dry_run": DRY,
        "best_epoch": best_ep, "best_composite": round(best_val, 4),
        "gallery_threshold": round(thr, 4), "separability_gap": round(gap, 4),
        "n_individuals": n_known, "n_holdout": len(ho_names),
        "acc_top1_pct": bm_results["acc_top1_pct"],
        "mean_f1": bm_results["mean_f1"],
    }, indent=2, ensure_ascii=False))

    section("DONE")
    _ho_line = f"  Held-out unknowns    : {ho_names}\n" if ho_names else ""
    log(f"""
  Separability gap     : {gap:.4f}  {'(EXCELLENT > 0.7)' if gap > 0.7 else '(GOOD > 0.5)' if gap > 0.5 else '(AVERAGE)'}
  Composite score      : {best_val:.4f}  (epoch {best_ep})
  Gallery threshold    : {thr:.4f}
  Top-1 accuracy       : {bm_results['acc_top1_pct']:.1f}%
  Mean F1              : {bm_results['mean_f1']:.4f}
  Known individuals    : {n_known}
{_ho_line}  Total duration       : {total_t/60:.1f} min
  Model                : {CKPT_BEST.name}
  Gallery              : {GALLERY_JSON.name}
  Output files:
    {BENCHMARK_PNG.name}   {CONFUSION_PNG.name}
    {CURVES_PNG.name}      {DATA_STATS_PNG.name}
    {DIAGNOSTICS_JSON.name}
""")
    _log_fh.close()

if __name__ == "__main__":
    main()
