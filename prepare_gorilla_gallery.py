# prepare_gorilla_gallery.py
# Converts the training gallery (output/v1_gorilla/gallery_gorilla.json) into a
# simplified, app-ready gallery format (output/v1_gorilla/gallery.json).
#
# Run from the repo root (no WSL needed):
#   python prepare_gorilla_gallery.py

import json
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent
SRC  = REPO / "output/v1_gorilla/gallery_gorilla.json"
DST  = REPO / "output/v1_gorilla/gallery.json"

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"Reading {SRC.name}...")
src = json.loads(SRC.read_text(encoding="latin-1"))

# Simplified gallery format, consumed by any downstream client
# (mobile app, web demo, batch-inference script, etc.):
#   - "unknown_threshold"         float
#   - "embedding_dim"             int
#   - "normalization"             str  (dim=768 → MegaDescriptor preprocessing, mean=std=0.5)
#   - individuals.<name>.class_index   int
#   - individuals.<name>.category      str  (SB/ADF/SAF/SAM/JUV/BB/?)
#   - individuals.<name>.embeddings    list[list[float]]  ← up to 25 exemplars

out_individuals = {}
n_exemplars_total = 0
n_skipped = 0

for name, ind in src["individuals"].items():
    exemplars = ind.get("exemplars", [])

    if not exemplars:
        # Fall back to prototype if no exemplars (should not happen for training-built gallery)
        proto = ind.get("prototype")
        if proto:
            exemplars = [proto]
        else:
            log(f"  [SKIP] {name}: no exemplars and no prototype")
            n_skipped += 1
            continue

    entry = {
        "class_index": ind.get("class_index", 0),
        "category":    ind.get("category", ""),
        "embeddings":  exemplars,       # list of up to 25 float[768] arrays
    }
    # Preserve mean_intra as documentation field (useful for debugging)
    if "mean_intra" in ind:
        entry["mean_intra"] = round(ind["mean_intra"], 4)

    out_individuals[name] = entry
    n_exemplars_total += len(exemplars)

gallery = {
    "version":           src.get("version", "gorilla-v1"),
    "species":           src.get("species", "Gorilla beringei beringei"),
    "project":           src.get("project", "GorillaIdentifier"),
    "created":           src.get("created", datetime.now().isoformat()),
    "model":             src.get("model", "MegaDescriptor-T-224 + ArcFace"),
    "embedding_dim":     src.get("embedding_dim", 768),
    # "normalization" tells downstream consumers which image preprocessing to use.
    # "megadescriptor" → mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] (correct for this model).
    "normalization":     "megadescriptor",
    "similarity_metric": src.get("similarity_metric", "cosine"),
    "unknown_threshold": src.get("unknown_threshold", 0.469),
    "separability_gap":  src.get("separability_gap"),
    "n_individuals":     len(out_individuals),
    "individuals":       out_individuals,
}

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(gallery, indent=2, ensure_ascii=False), encoding="utf-8")

log(f"Written -> {DST}")
log(f"  {len(out_individuals)} individuals  |  {n_exemplars_total} total exemplars  |  {n_skipped} skipped")
log(f"  threshold = {gallery['unknown_threshold']}")
log(f"  embedding_dim = {gallery['embedding_dim']}")
avg_ex = n_exemplars_total / max(len(out_individuals), 1)
log(f"  avg exemplars/individual = {avg_ex:.1f}")
log("Done.")
