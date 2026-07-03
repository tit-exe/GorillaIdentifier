# GorillaIdentifier

Individual identification of mountain gorillas (*Gorilla beringei beringei*) by facial recognition.

**Pipeline:** photo, then YOLO for face detection, then MegaDescriptor-T-224 for a 768-dimensional embedding, then cosine similarity against a gallery to return an identity or "Unknown".

This repository contains only the V1 training pipeline: extraction, training, and gallery construction. It does not include photos, crops, or model weights. Those are either regenerated locally or downloaded from Zenodo.

---

## Results (V1, 66 individuals, Virunga 2025)

| Metric | Value |
|---|---|
| Recognized individuals | 66 |
| Top-1 accuracy | **93.0%** |
| Top-3 accuracy | 96.1% |
| Mean F1 | 0.981 |
| Composite score | 0.808 |
| Rejection threshold | 0.469 |
| Separability gap | 0.435 |
| Backbone | MegaDescriptor-T-224 (Swin Transformer Tiny, 27.5M parameters) |
| Training time | about 66 minutes on an RTX 3050 4 GB |

These numbers come from the benchmark run on the held-out validation set after training (see `output/v1_gorilla/diagnostics.json` once you've trained your own model). Top-1 and top-3 accuracy measure how often the correct individual appears at rank 1 or in the top 3 matches. The rejection threshold is the cosine-similarity cutoff below which a face is reported as unknown, calibrated by maximizing F1 on the validation set. The separability gap is the average similarity gap between an individual's own exemplars and its closest rival; a higher gap means less confusion between individuals.

---

## Installation

Open Anaconda Prompt (not PowerShell or cmd):

```bash
conda create -n gorilla_id python=3.10 -y
conda activate gorilla_id

# PyTorch with CUDA must be installed before requirements.txt.
# PyPI serves a CPU-only build by default.
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Remaining dependencies
pip install -r requirements.txt

# Check the installation
python check_env.py
```

Every line should print `[OK]`. If you see `[!!] GPU present but PyTorch has NO CUDA`, uninstall PyTorch and reinstall it with the `--index-url` shown above.

Each new terminal session needs `conda activate gorilla_id`, then a move to the repository root. Every path used by the pipeline is relative to that root, defined in `config.yaml`, so the repository can be cloned anywhere on any drive.

---

## YOLO model

The gorilla face detector, `models/yolo_gorilla.pt`, is hosted on Zenodo:

```
https://doi.org/10.5281/zenodo.18757935
```

Download it and place it at `models/yolo_gorilla.pt` before running any crop extraction. The MegaDescriptor-T-224 backbone does not need a manual download: it is fetched automatically from the HuggingFace Hub the first time `v1_megadesc_arcface/train.py` runs, and cached locally afterward (see `config.yaml` to change the cache location).

---

## Training pipeline

### 1. Organize the photos

Create one subfolder per individual inside `data/photos/`:

```
data/photos/
    SB Humba/
        IMG_001.jpg
        IMG_002.jpg
        ...
    ADF Anangana/
        ...
    JUV Bakunzi/
        ...
```

The folder name's prefix sets how many ArcFace sub-centers that individual gets during training. More sub-centers give the model more tolerance for appearance variation, which matters for young individuals whose look changes quickly.

| Prefix | Category | Sub-centers |
|---|---|---|
| `SB` | Silverback | 1 |
| `ADF` | Adult female | 1 |
| `AD` | Adult | 1 |
| `SAF` / `SAM` | Subadult female / male | 1 |
| `JUV` | Juvenile | 2 |
| `BB` | Baby / blackback | 2 |
| *(none)* | Unspecified | 1 |

Aim for at least 30 photos per individual, with varied angles and conditions. Photos do not need to be pre-cropped; YOLO detects faces automatically. See `data/photos/README.md` and `data/wild_images/raw/README.md` for more detail on both folders.

### 2. Scan the data (optional)

```bash
python v1_megadesc_arcface/train.py --min-photos 5
```

Prints a report listing which individuals are included or excluded, and how many crops are available for each.

### 3. Extract the crops

```bash
python v1_megadesc_arcface/train.py --extract
```

YOLO detects faces in every photo and saves 224x224 crops to `data/crops/known/`.

### 4. Review the crops (recommended)

```bash
python common/review_crops.py
```

A drag-and-drop interface for checking each crop. Press Enter to validate, Del to delete, R to reject, and Q to quit.

### 5. Train

```bash
python v1_megadesc_arcface/train.py
```

Training resumes automatically if interrupted. It writes its results to `output/v1_gorilla/`. `best.pt` is the best checkpoint found during training, `gallery_gorilla.json` is the gallery of exemplars kept after filtering for intra-individual quality, and `diagnostics.json` holds the complete metrics, training curves, and confusion matrix.

### 6. Benchmark only

If a model is already trained and you only want to recompute the metrics:

```bash
python v1_megadesc_arcface/train.py --benchmark-only
```

### 7. Prepare the final gallery

```bash
python prepare_gorilla_gallery.py
```

This converts `output/v1_gorilla/gallery_gorilla.json`, the internal training format, into `output/v1_gorilla/gallery.json`, a simplified format that any downstream client can consume directly: a mobile app, a batch inference script, a web demo, and so on.

---

## Repository structure

```
GorillaIdentifier/
├── v1_megadesc_arcface/
│   └── train.py                       All-in-one script: extract, train, build the gallery, benchmark
├── common/
│   ├── config_loader.py               Centralizes every path, reads config.yaml
│   └── review_crops.py                PyQt5 tool for reviewing crops
├── data/                              Not tracked by git, see .gitignore
│   ├── photos/                        Field photos per individual, fill this yourself
│   └── wild_images/raw/               Background images for the "unknown" class, fill this yourself
├── models/                            Not tracked by git, download from Zenodo
│   ├── README.md                      Explains what goes here and where to get it
│   └── yolo_gorilla.pt
├── output/v1_gorilla/                 Not tracked by git, generated by the pipeline
│   ├── best.pt
│   ├── gallery_gorilla.json
│   ├── gallery.json                   generated by prepare_gorilla_gallery.py
│   └── diagnostics.json
├── check_env.py                       Environment diagnostic
├── prepare_gorilla_gallery.py         Converts the internal gallery to the simplified format
├── config.yaml                        Centralized settings: paths and hyperparameters
├── requirements.txt                   Python dependencies
├── README.md                          This file
└── TECHNICAL_DOCUMENTATION.md         Architecture, loss functions, curriculum, and augmentation details
```

---

## AI architecture, in brief

### Inference pipeline

```
Field photo
    |
    v
[YOLO yolo_gorilla.pt]              face detection, 640x640 input
    | 224x224 crop
    v
[MegaDescriptor-T-224]              Swin Transformer Tiny, pretrained for multi-species re-identification
    | L2-normalized 768D vector
    v
[gallery.json]                      max cosine similarity over up to 25 exemplars per individual
    |
    v
similarity >= 0.469 and margin >= 0.08   ->   individual identified, with a score
similarity < 0.469 or margin < 0.08      ->   "Unknown"
```

### Loss functions (training only)

The total training loss combines three terms: `loss = L_ArcFace + lambda_inv * L_invariance + lambda_sup * L_SupCon`.

| Term | Role |
|---|---|
| L_ArcFace (sub-center, scale 64, margin 0.50) | Angular separation between individuals |
| L_invariance | Pushes the embedding of a clean image close to that of its degraded version, for robustness in the field |
| L_SupCon (temperature 0.07) | Maximizes intra- and inter-individual separability within each batch |

### Four-phase curriculum

| Phase | Epochs | Backbone | lambda_inv | lambda_sup |
|---|---|---|---|---|
| A: init | 3 | Frozen | 0 | 0 |
| B: warmup | 15 | Unfrozen | 0.15 | 0.10 |
| C: learning | 20 | Unfrozen | 0.25 | 0.15 |
| D: consolidation | 15 | Unfrozen | 0.25 | 0.15 |

Early stopping applies during phase D, with a patience of 12 epochs measured on the composite score.

### Gorilla-specific augmentations

| Augmentation | What it does |
|---|---|
| `_LowRes` | Downscales to 8 to 95% of original resolution then upscales, simulating a distant photo |
| `_JPEG` | Adds compression artifacts at quality 10 to 80, matching phone or WhatsApp photos |
| `_ForestShadow` | Darkens horizontal or vertical bands, simulating foliage-filtered sunlight |
| `_ColorTemp` | Shifts the color temperature between warm and cold |
| Standard set | RandomResizedCrop, horizontal flip at 0.5, rotation up to 20 degrees, color jitter |

There is no vertical flip, since gorillas are always upright in field photos.

`TECHNICAL_DOCUMENTATION.md` covers the full detail: equations, hyperparameters, and the reasoning behind each design choice.

---

## Adapting the pipeline to another species

1. Train a YOLO face-detection model for the target species.
2. Replace `models/yolo_gorilla.pt` with that model.
3. Organize photos in `data/photos/<Individual>/`, following the same one-folder-per-individual convention.
4. Edit `species` and `project_name` in `config.yaml`.
5. Run the pipeline as described above.

MegaDescriptor-T is pretrained across many species, so it performs reasonably well even with a limited dataset.

---

## Limitations

The YOLO detector misses most faces photographed at more than roughly 60 degrees off-axis. Juveniles and babies change appearance quickly, so their gallery entries should be refreshed every three to six months. More generally, gradual changes in individual appearance over time mean the model benefits from retraining roughly once a year.

---

## References

Čermák et al. (2024). *WildlifeDatasets: A Comprehensive Library for Animal Re-Identification.* WACV 2024.

Deng et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.

Deng et al. (2020). *Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces.* ECCV 2020.

Khosla et al. (2020). *Supervised Contrastive Learning.* NeurIPS 2020.
