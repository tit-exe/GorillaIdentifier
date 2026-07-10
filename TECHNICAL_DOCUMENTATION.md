# GorillaIdentifier — Technical Documentation

Individual identification of mountain gorillas (*Gorilla beringei beringei*) by facial
recognition. This document covers the entire V1 training pipeline: data, architecture,
loss functions, curriculum, gallery, validation, and results.

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Data — acquisition and structure](#2-data--acquisition-and-structure)
3. [Crop extraction (YOLO)](#3-crop-extraction-yolo)
4. [V1 model architecture](#4-v1-model-architecture)
5. [Loss functions](#5-loss-functions)
6. [4-phase training curriculum](#6-4-phase-training-curriculum)
7. [Augmentation pipeline](#7-augmentation-pipeline)
8. [Gallery and threshold calibration](#8-gallery-and-threshold-calibration)
9. [Validation and composite score](#9-validation-and-composite-score)
10. [Inference](#10-inference)
11. [Output files](#11-output-files)

---

## 1. System overview

```
Field photos
    ↓
[YOLO yolo_gorilla.pt]        face detection, 640×640 input
    ↓ 224×224 crops
[Optional manual review]
    ↓
[MegaDescriptor-T-224 training]
    ↓
[Filtered exemplar gallery]
    ↓
[Inference: cosine similarity → identity or "Unknown"]
```

The whole pipeline is implemented in a single script, `v1_megadesc_arcface/train.py`,
which chains extraction, training, gallery construction, and benchmarking.

---

## 2. Data — acquisition and structure

### 2.1 Folder convention

One folder per individual in `data/photos/`, named `{CATEGORY} {Name}`:

```
data/photos/
    SB Humba/          ← silverback
    SB Kibumba/
    ADF Anangana/      ← adult female
    SAF Ndakasi/       ← subadult female
    SAM Mafunzo/       ← subadult male
    JUV Bakunzi/       ← juvenile
    BB Amani/          ← baby / blackback
```

### 2.2 Categories and their role

| Prefix | Category | K sub-centers (ArcFace) | Rationale |
|---|---|---|---|
| `SB` | Silverback | 1 | Stable adult morphology |
| `ADF` | Adult Female | 1 | Same |
| `SAF` | Subadult Female | 1 | Same |
| `SAM` | Subadult Male | 1 | Same |
| `JUV` | Juvenile | 2 | Appearance changes fast — rapid ontogeny |
| `BB` | Baby / Blackback | 2 | Same |
| *(none)* | Unspecified | 1 | Fallback |

The number K of sub-centers per class is adjusted per category to capture the
intra-individual variability of young individuals (see §4.2).

### 2.3 Exclusion thresholds

An individual is excluded from training if it doesn't have enough photos/crops:

| Parameter | Default value |
|---|---|
| `--min-photos` (scan/exclusion) | 5 |
| Hard threshold (mobile gallery) | 10 crops after extraction |
| Warning threshold | 15 crops |

Practical recommendation: **≥ 30 photos per individual**, varied angles and conditions.

### 2.4 Held-out pseudo-unknowns

To measure the unknown-rejection rate without needing dedicated images of unknown
gorillas, the script automatically removes `N_HOLDOUT = 3` individuals from training.
These individuals are chosen among those with the **highest crop count** (more reliable
test). Their crops are used only to measure the rejection rate during validation, never
for training.

```python
N_HOLDOUT = 3
# Constraint: keep at least 15 individuals in training
n_ho = min(N_HOLDOUT, max(0, n_classes - 15))
# Selection: individuals with the most crops
sorted_by_crops = sorted(range(n_classes), key=lambda i: crops_per_ind[i], reverse=True)
holdout_set = set(sorted_by_crops[:n_ho])
```

### 2.5 Train / validation split

Stratified split (per individual): 85% training / 15% validation (`VAL_RATIO = 0.15`).
`SEED = 42` guarantees reproducibility.

---

## 3. Crop extraction (YOLO)

### 3.1 Detection model

Custom YOLO detector (`yolo_gorilla.pt`, ~7 MB) trained specifically for mountain
gorilla faces in the Virungas. Downloadable from HuggingFace (see README.md).

- Input: original image, resized to 640×640
- Confidence threshold: 0.30 (configurable via `config.yaml`)
- Margin around the box: 5% of box size

### 3.2 Post-processing

For each detection:
1. Compute the bounding box + margin
2. Crop and resize to **224×224 px** (MegaDescriptor's input size)
3. Save to `data/crops/known/{individual_folder}/`

### 3.3 crops.json file

Each extraction creates or updates `data/crops.json`: a dictionary mapping
photo → list of crops with bbox, YOLO confidence, and review status.
It allows tracing the origin of every crop and identifying photos with no detection.

### 3.4 Manual review

The `common/review_crops.py` tool (PyQt5 drag-and-drop interface) supports:
- `Enter`: validate a crop
- `Del`: physically delete the crop
- `R`: reject (moves to a rejected/ folder)
- `Q`: quit

Recommended step before training, to eliminate false positives (branches,
gorillas seen from behind, partial detections).

---

## 4. V1 model architecture

### 4.1 Backbone — MegaDescriptor-T-224

| Attribute | Value |
|---|---|
| Architecture | Swin Transformer Tiny |
| Parameters | 27.5 M |
| Source | `timm`, HuggingFace `BVRA/MegaDescriptor-T-224` |
| Pretraining | Multi-species re-identification (WildlifeDatasets — 37 species) |
| Input | 224×224 px, 3 channels |
| Output | 768D vector (before classification) |
| Output normalization | L2 → unit vector |

The backbone is loaded with `num_classes=0` (classification head ignored) — only
the 768D feature vector is used.

The critical advantage of MegaDescriptor's multi-species pretraining is that it has
already learned discriminant descriptors for individual animal appearance, unlike a
classic ImageNet backbone which specializes in object categories.

### 4.2 Classification head — Sub-center ArcFace

The ArcFace head projects the 768D embeddings into angular space and computes the
margin loss.

#### Sub-centers

Instead of a single prototype per class, Sub-center ArcFace maintains **K prototypes
(sub-centers)** per individual. A class's logit is the **maximum** over its K
sub-centers:

```python
# For each class c with K[c] sub-centers
logits[:, c] = ca[:, s : s+k].max(dim=1).values
```

This lets each individual have multiple "modes" in embedding space — useful for
juveniles whose appearance changes quickly.

#### Full ArcFace forward pass

```python
w = F.normalize(self.weight, dim=1)           # L2-normalize the prototypes
cos = (emb @ w.T).clamp(-1.0, 1.0)           # cosine similarity
phi = cos * cos_m - sin * sin_m               # angular margin shift m
phi = where(cos > th, phi, cos - mm)          # numerical protection for cos<0
out = (one_hot * phi + (1-one_hot) * cos) * scale
return cross_entropy(out, labels, label_smoothing=0.05)
```

| Hyperparameter | Value | Role |
|---|---|---|
| `scale` | 64 | Logit temperature (sharpens decision boundaries) |
| `margin` | 0.50 | Angular margin in radians (strict — proven for animal re-ID) |
| `label_smoothing` | 0.05 | Reduces overconfidence on underrepresented individuals |

---

## 5. Loss functions

The total loss is a linear combination of three terms:

```
L_total = L_ArcFace + λ_inv × L_invariance + λ_sup × L_SupCon
```

At every forward pass, each batch is processed **twice** in parallel: a clean version
and a degraded version. Both versions contribute to L_ArcFace, and their mutual
relationship is exploited by L_invariance.

```python
both    = torch.cat([clean, deg], dim=0)         # [2B, 3, 224, 224]
emb_all = F.normalize(bb(both), dim=1)           # [2B, 768] — L2-normalized
ec, ed  = emb_all[:B], emb_all[B:]               # split clean / degraded

l_arc = arc(ec, labels) + arc(ed, labels)        # ArcFace on both
l_inv = loss_invariance(ec, ed)                  # clean↔degraded invariance
l_sup = loss_supcon(ec, labels)                  # SupCon on clean only
loss  = l_arc + lam_inv * l_inv + lam_sup * l_sup
```

### 5.1 L_ArcFace — Sub-center ArcFace

Imposes an angular margin between individuals in embedding space. Computes the
cross-entropy after applying the margin to the correct class's logit.

**Role**: angularly separate individuals, train the classification head.

### 5.2 L_invariance

```python
def loss_invariance(emb_clean, emb_deg):
    return 1.0 - (emb_clean * emb_deg).sum(dim=1).mean()
```

Equivalent to `1 - cosine_similarity(emb_clean, emb_degraded)`. Minimizing this loss
forces the backbone to produce the **same embedding** regardless of degradation.

**Role**: robustness to field conditions (blur, JPEG compression, shadows).

### 5.3 L_SupCon — Supervised Contrastive Loss

```python
def loss_supcon(emb, labels, temp=0.07):
    sim = torch.matmul(emb, emb.T) / temp            # temperature-scaled similarity matrix
    mask_pos = (labels_col == labels_row) & mask_diag # positives = same individual, ≠ self
    sim_exp  = torch.exp(sim) * mask_diag.float()    # exclude the diagonal (self-similarity)
    log_denom = torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-8)
    log_pos   = sim - log_denom
    # Average over all positives available in the batch
    loss_per = -(log_pos * mask_pos.float()).sum(dim=1) / n_pos
    return loss_per.mean()
```

Temperature `T = 0.07` (low value = very sharp boundaries, gradients concentrated
on hard pairs). Computed only on clean embeddings (not the degraded ones).

**Role**: directly maximize intra/inter-individual separability within the batch,
independent of the classification head.

---

## 6. 4-phase training curriculum

Training follows a progressive curriculum: the backbone starts frozen, augmentation
severity ramps up gradually, and auxiliary loss weights increase progressively.

```python
PHASES = [
    dict(name="A — Init",          epochs=3,  freeze=True,  lr_bb=0.0,  lr_h=1e-3,
         lam_inv=0.00, lam_sup=0.00, severity=0.00),
    dict(name="B — Warmup",        epochs=15, freeze=False, lr_bb=5e-6, lr_h=5e-4,
         lam_inv=0.15, lam_sup=0.10, severity=0.40),
    dict(name="C — Learning",      epochs=20, freeze=False, lr_bb=3e-6, lr_h=2e-4,
         lam_inv=0.25, lam_sup=0.15, severity=0.70),
    dict(name="D — Consolidation", epochs=15, freeze=False, lr_bb=1e-6, lr_h=5e-5,
         lam_inv=0.25, lam_sup=0.15, severity=1.00, early_stop=True, patience=12),
]
```

| Phase | Epochs | Backbone | lr_backbone | lr_head | λ_inv | λ_sup | Augm. severity |
|---|---|---|---|---|---|---|---|
| A — Init | 3 | **Frozen** | 0 | 1e-3 | 0.00 | 0.00 | 0% |
| B — Warmup | 15 | Unfrozen | 5e-6 | 5e-4 | 0.15 | 0.10 | 40% |
| C — Learning | 20 | Unfrozen | 3e-6 | 2e-4 | 0.25 | 0.15 | 70% |
| D — Consolidation | 15 | Unfrozen | 1e-6 | 5e-5 | 0.25 | 0.15 | 100% |

Total: 53 epochs at most (early stopping possible in phase D).

### 6.1 Curriculum rationale

**Phase A**: only the ArcFace head trains (backbone frozen). This initializes the
class prototypes without disturbing MegaDescriptor's pretrained features.

**Phase B**: progressive backbone unfreezing at a very low learning rate. Moderate
augmentation severity (40%) — the model starts adapting its features to Virunga
gorillas. Auxiliary losses (SupCon, invariance) enter at reduced weights.

**Phase C**: main learning phase. Severity at 70%, auxiliary weights at their final
values. The backbone adapts in depth.

**Phase D**: very low learning rate, maximum severity (100%). Early stopping with
patience=12: if the composite score doesn't improve for 12 consecutive epochs,
training stops.

### 6.2 Learning rate scheduler

Within each phase, a cosine scheduler with a 3-epoch linear warmup (or 25% of the
phase duration):

```python
def _lr(step):
    if step < warmup:
        return (step+1) / max(warmup, 1)               # linear ramp-up
    prog = (step - warmup) / max(total - warmup, 1)
    return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))  # cosine decay
```

### 6.3 Optimizer

AdamW with `weight_decay=1e-4`. Two parameter groups with distinct learning rates:
backbone (lr_bb) and ArcFace head (lr_h).

### 6.4 Dynamic threshold during training

For inter-phase validation (computing the composite score), a dynamic rejection
threshold is used instead of the final calibrated threshold:

```
threshold_dynamic = pos_mean - 2 × pos_std
threshold_dynamic = clip(threshold_dynamic, 0.45, 0.80)
```

This adaptive threshold tracks the evolving similarity distributions during
training and is never used at inference (replaced by the F1-calibrated threshold).

### 6.5 Crash safety

The script saves a `resume.pt` checkpoint **after every epoch** (atomic save:
write to a `.tmp` file, then `rename` to avoid corruption). On interruption
(Ctrl+C, window closed, power loss), relaunching the script resumes exactly at
the last completed epoch.

Handlers implemented:
- `signal.SIGINT` (Ctrl+C on Linux/Mac)
- `win32api.SetConsoleCtrlHandler` (window close on Windows, SIGBREAK)

### 6.6 VRAM auto-detection

Batch size is automatically adjusted based on available VRAM:

```python
if vram >= 16: BATCH = 64
elif vram >= 8: BATCH = 32
elif vram >= 4: BATCH = 16   # RTX 3050 4GB → bs=16
else:           BATCH = 8
```

---

## 7. Augmentation pipeline

### 7.1 General architecture

Every image is augmented **twice** for each batch:
- **Clean** version: light augmentations
- **Degraded** version: same augmentations + gorilla-specific degradations

The overall degradation severity is controlled by the phase's `severity` parameter
(0.0 → 1.0). At `severity=0`, only the base augmentations are applied.

### 7.2 Base augmentations (clean + degraded)

```python
transforms_clean = T.Compose([
    T.RandomResizedCrop(224, scale=(0.75, 1.0)),
    T.RandomHorizontalFlip(p=0.5),         # horizontal flip only (never vertical)
    T.RandomRotation(degrees=20),           # ±20°
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
```

**No vertical flip**: gorillas are always upright in field photos. Flipping vertically
would create training examples that don't exist in practice.

### 7.3 Gorilla-specific degradations

These augmentations reproduce the real acquisition conditions in the Virungas:

#### `_LowRes` — Low resolution

```python
# Downscale to p% of original size, then bilinear upscale
# p ∈ [8%, 95%] — simulates photos taken from far away (200m+ in forest)
factor = random.uniform(0.08, 0.95)
small  = img.resize((max(1, int(w*factor)), max(1, int(h*factor))), Image.BILINEAR)
img    = small.resize((w, h), Image.BILINEAR)
```

#### `_JPEG` — Compression artifacts

```python
# Random JPEG quality between 10 and 80
# Reproduces WhatsApp photos transmitted over 2G by rangers
quality = random.randint(10, 80)
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=quality)
img = Image.open(io.BytesIO(buf.getvalue()))
```

#### `_CropJitter` — Crop offset

```python
# Random jitter of the crop's center and scale
# Reproduces the YOLO detector's imprecision under real conditions
scale = random.uniform(0.85, 1.15)
dx, dy = random.randint(-20, 20), random.randint(-20, 20)
```

#### `_ForestShadow` — Forest shadows (Virunga-specific)

```python
class _ForestShadow:
    # Darkens a horizontal OR vertical band of the image
    # Proportion 60%/40% (horizontal more frequent)
    # alpha ∈ [0.20, 0.60] — 20% to 60% darkening
    # Simulates branches and foliage in the foreground
    # Very common in the dense vegetation of the Virungas (2000-3000m)
```

#### `_ColorTemp` — Color temperature

```python
class _ColorTemp:
    # Warm↔cold color temperature shift
    # Based on field observations (Virunga palettes):
    #   #726f6f → warm gray   (R ≈ G ≈ B)
    #   #8a8c9d → cold gray   (B > R by ~14%)
    #   #63727e → dense forest (B > R by ~27%)
    #
    # shift ∈ [-strength, +strength]
    t[0] = (t[0] * (1.0 + shift)).clamp(0, 1)        # R increases if warm
    t[1] = (t[1] * (1.0 + shift * 0.25)).clamp(0, 1) # G follows weakly
    t[2] = (t[2] * (1.0 - shift)).clamp(0, 1)        # B increases if cold
```

### 7.4 Augmentation summary

| Augmentation | Type | Main parameter | Field rationale |
|---|---|---|---|
| `RandomResizedCrop` | Base | scale 75-100% | Variable framing |
| `HorizontalFlip` | Base | p=0.5 | Gorillas photographed from both sides |
| `RandomRotation` | Base | ±20° | Camera held at an angle |
| `ColorJitter` | Base | ±20% br/ct/sat | Lighting variability |
| `_LowRes` | Gorilla | 8-95% resolution | Distant forest photos |
| `_JPEG` | Gorilla | quality 10-80 | WhatsApp/2G transmission |
| `_CropJitter` | Gorilla | ±20px, ±15% scale | YOLO imprecision |
| `_ForestShadow` | Gorilla | alpha 0.20-0.60 | Virunga foliage |
| `_ColorTemp` | Gorilla | adaptive shift | Under-canopy forest light |

---

## 8. Gallery and threshold calibration

### 8.1 Gallery construction

After training, the best checkpoint is loaded and the gallery is built over
the entire training crop set (no validation crops):

1. **Embedding extraction**: every crop → backbone → L2-normalized 768D vectors
2. **Centroid computation** per individual: L2-normalized mean of all its embeddings
3. **Quality filtering**: only crops with `cosine_sim(emb, centroid) ≥ QUALITY_THR`
   are kept as exemplars:

```python
QUALITY_THR = 0.62   # intra-individual quality threshold
K_EXEMPLARS = 25     # maximum number of exemplars per individual
```

4. **Selection**: among the qualifying crops, up to `K_EXEMPLARS=25` exemplars are
   kept (the K closest to the centroid if more than K are available).

### 8.2 Role of multiple exemplars

Inference computes an individual's score as the **maximum** over all its exemplars:

```
score_i = max{ cosine_sim(query, exemplar_j) : j ∈ [1..K_i] }
```

This strategy is robust to appearance variability: a single "matching" exemplar
is enough for a positive identification, even if the other exemplars correspond
to different angles or conditions.

### 8.3 Rejection threshold calibration

The optimal threshold is found by maximizing the F1-score over the positive and
negative similarity distributions measured on the validation set:

```python
pos = []  # true-positive scores (same individual)
neg = []  # negative scores (different individuals or unknowns)

# For each validation image:
#   pos.append(max cosine_sim to the correct individual's exemplars)
#   neg.append(max cosine_sim to each OTHER individual's exemplars)

thresholds = np.linspace(0, 1, 500)
for t in thresholds:
    tp = int((pos >= t).sum())
    fp = int((neg >= t).sum())
    fn = int((pos < t).sum())
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1s.append(2*p*r / (p+r+1e-9))

opt_t = float(thresholds[np.argmax(f1s)])   # → 0.4689 for V1
```

**V1 calibrated threshold: 0.4689**

### 8.4 Separability gap

Main model quality metric:

```
gap = mean(pos_sims) − mean(neg_sims)
```

The higher the gap, the better individuals are separated in embedding space.
Rough interpretation: gap > 0.7 = excellent, > 0.5 = good, < 0.3 = insufficient.

**V1: gap = 0.4351**

### 8.5 Gallery JSON format

```json
{
  "version": "gorilla-v1",
  "species": "Gorilla beringei beringei",
  "embedding_dim": 768,
  "normalization": "megadescriptor",
  "similarity_metric": "cosine",
  "unknown_threshold": 0.4689,
  "separability_gap": 0.4351,
  "n_individuals": 66,
  "individuals": {
    "SB Mastaki": {
      "class_index": 61,
      "category": "SB",
      "embeddings": [[0.032, -0.011, ...], ...],   // up to 25 exemplars, 768 floats each
      "mean_intra": 0.7783
    }
  }
}
```

The `mean_intra` field is the average cosine similarity between all pairs of
an individual's exemplars, taken from the actual V1 gallery in the example above.
A low value (below roughly 0.65) indicates inconsistent exemplars, which is a
possible sign of a labeling error or overly diverse photos.

---

## 9. Validation and composite score

### 9.1 Composite score

Metric used to track progress during training and select the best checkpoint:

```
composite_score = 0.25 × acc_clean
               + 0.35 × acc_degraded
               + 0.25 × min(sep_gap / 1.0, 1)
               + 0.15 × unk_rejection_rate
```

| Component | Weight | Description |
|---|---|---|
| `acc_clean` | 25% | Top-1 accuracy on clean images |
| `acc_degraded` | 35% | Top-1 accuracy on degraded images — dominant weight |
| `sep_gap` | 25% | Intra/inter separability (normalized to [0,1]) |
| `unk_rejection_rate` | 15% | Rejection rate of held-out pseudo-unknowns |

The emphasis on `acc_degraded` (35%) reflects the main objective: working under
real, degraded field conditions, not just on high-quality images.

If no held-out individual is available (< 15 individuals total), `unk_rate` is
replaced with 0.5 (neutral).

### 9.2 Fast inter-epoch separability

To avoid computing the full gallery at every epoch, a fast approximation is
computed on 200 randomly drawn crops:

```python
# Normalized centroids per class
proto[c] = mean(embs_c) / ||mean(embs_c)||

# For each crop i:
pos[i] = cosine_sim(emb_i, proto[label_i])           # similarity to its own prototype
neg[i] = max{ cosine_sim(emb_i, proto[c]) : c ≠ label_i }  # best rival prototype

sep = mean(pos) - mean(neg)
```

### 9.3 Final benchmark

The final benchmark (run after training) computes, over the validation set:

- **Top-1 accuracy**: correct identification rate at rank 1
- **Top-3 accuracy**: correct identification rate within the top 3 ranks
- **Per-individual F1**: precision, recall, F1 for each individual (with rejection)
- **Mean F1**: average F1 across all individuals
- **Confusion matrix**: `confusion.png`
- **Similarity distribution**: pos/neg histogram, calibrated threshold: `benchmark.png`

**V1 results (published model):**

| Metric | Value |
|---|---|
| Recognized individuals | **66** |
| Top-1 accuracy | **93.0 %** |
| Top-3 accuracy | **96.1 %** |
| Mean F1 | **0.981** |
| Composite score | **0.808** |
| Calibrated rejection threshold | **0.4689** |
| Separability gap | **0.4351** |
| Training time | ~66 min (RTX 3050 4 GB) |

---

## 10. Inference

Inference relies on the V1 backbone and the `gallery.json` produced by
`prepare_gorilla_gallery.py` (see README.md).

### 10.1 Preprocessing

```
normalized_pixel = (pixel_value / 255 − 0.5) / 0.5
```

For each channel (R, G, B). Input format: `float32[1, 3, 224, 224]` (NCHW).

### 10.2 Decision

Two conditions must both hold for a positive identification:

| Condition | Threshold | Purpose |
|---|---|---|
| `best_score ≥ unknown_threshold` | 0.4689 (V1) | Rejects absolute unknowns |
| `margin ≥ 0.08` | fixed | Rejects ambiguous identifications (top-1 ≈ top-2) |

If either fails → **"Unknown gorilla"**.

The margin is defined as `score_top1 − score_top2`. It flags cases where two
individuals have close scores, indicating identification uncertainty.

---

## 11. Output files

All output files live in `output/v1_gorilla/` (not tracked by git, see .gitignore).

| File | Content |
|---|---|
| `best.pt` | Best checkpoint (backbone_state + arc_state + metadata) |
| `resume.pt` | Resume checkpoint (last epoch) — for crash recovery |
| `gallery_gorilla.json` | Gallery: exemplars, threshold, gap, metadata |
| `gallery.json` | Simplified gallery, generated by `prepare_gorilla_gallery.py` |
| `diagnostics.json` | Full JSON: epoch-by-epoch metric history, benchmark, hyperparameters, data_info |
| `report.json` | Short legacy JSON (10-field summary) |
| `curves.png` | Training curves: losses and metrics per epoch |
| `benchmark.png` | Pos/neg similarity distribution + calibrated threshold + per-individual F1 |
| `confusion.png` | Confusion matrix over the validation set |
| `data_stats.png` | Crops-per-individual histogram + category breakdown |
| `train.log` | Full text training log |

### 11.1 diagnostics.json structure

```json
{
  "meta": {
    "version": "v1-gorilla",
    "generated": "2026-06-12T13:44:27",
    "training_min": 65.9,
    "separability_gap": 0.4351
  },
  "data_info": {
    "total_individuals_raw": 71,
    "n_known": 66,
    "n_holdout": 3,
    "n_excluded": 2,
    "total_crops": 2809,
    "train_crops": 2023,
    "val_crops": 358,
    "unk_crops": 428,
    "per_individual": {
      "SB Humba": { "category": "SB", "n_crops": 14, "in_training": true, "is_holdout": false }
    }
  },
  "history": {
    "l_arc": [...],
    "l_inv": [...],
    "l_sup": [...],
    "acc_c": [...],
    "acc_d": [...],
    "sep":   [...],
    "comp":  [...],
    "unk_rate": [...],
    "best_epoch": 42,
    "best_score": 0.8083
  },
  "bm_results": {
    "acc_top1_pct": 93.0,
    "acc_top3_pct": 96.1,
    "mean_f1": 0.981,
    "gallery_threshold": 0.4689,
    "separability_gap": 0.4351,
    "per_class": { "SB Humba": { "f1": 1.0, "precision": 1.0, "recall": 1.0 } }
  },
  "hyperparams": {
    "arc_margin": 0.50,
    "arc_scale": 64,
    "batch_size": 16,
    "img_size": 224,
    "val_ratio": 0.15,
    "n_holdout": 3,
    "total_epochs": 47,
    "phases": [...]
  }
}
```

---

## References

- Čermák et al. (2024). *WildlifeDatasets: A Comprehensive Library for Animal Re-Identification.* WACV 2024.
- Deng et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.
- Deng et al. (2020). *Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces.* ECCV 2020.
- Khosla et al. (2020). *Supervised Contrastive Learning.* NeurIPS 2020.
- Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
- Bain et al. (2022). *BVRA/MegaDescriptor — Multi-species animal re-identification backbone.* HuggingFace Hub.
