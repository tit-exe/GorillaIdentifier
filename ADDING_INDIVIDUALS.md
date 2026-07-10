# Adding Individuals to the GorillaIdentifier Gallery

This document explains how the AI works, what the gallery contains, and how to safely add new individuals without degrading recognition quality. Read it fully before implementing any "add individual" feature.

---

## 1. How the AI works (pipeline overview)

```
Photo
  └── YOLO detector (yolo_v2_detector.tflite)
        └── Gorilla face crop [224×224 px]
              └── MegaDescriptor-T-224 backbone (megadesc_T_arcface_backbone.tflite)
                    └── 768-dim embedding vector  ──── L2-normalize ────►  unit vector
                          └── Compare to gallery via cosine similarity
                                └── max over all exemplars of the best match
                                      └── score ≥ threshold AND margin ≥ 0.08 → identified
                                          score < threshold OR margin < 0.08  → "Unknown"
```

### Embedding

The backbone converts a 224×224 gorilla face crop into a **768-dimensional float vector**. After L2-normalization, this vector lives on the surface of a unit sphere. Two crops of the **same** individual should produce vectors close to each other (high cosine similarity ≈ dot product); crops of **different** individuals should produce distant vectors.

- **Image preprocessing**: `pixel = (pixel_value/255 − 0.5) / 0.5` for each channel (mean=std=0.5).  
  Input format: NCHW float32, shape `[1, 3, 224, 224]`.
- **Output**: raw 768-dim float vector → L2-normalize in the app → unit vector.
- **Similarity metric**: cosine similarity = dot product of two unit vectors → ∈ [−1, 1].

### Gallery lookup

For each individual *i*, the gallery stores up to 25 **exemplar** embeddings (selected training crops). During inference:

```
score_i = max{ dot(query, exemplar_j) : j ∈ exemplars_i }
          + optional field_embedding check
```

The `max` is the key design choice: **a single matching exemplar is enough**. This makes the system robust to appearance variation (lighting, angle, age progression) — as long as at least one gallery exemplar looks similar to the query.

### Rejection conditions (both must pass for a positive ID)

| Condition | Threshold | Purpose |
|---|---|---|
| `best_score ≥ unknown_threshold` | 0.469 (V1) | Absolute — rejects gorillas not in gallery |
| `margin ≥ 0.08` | fixed | Conviction — rejects ambiguous matches (top-1 ≈ top-2) |

If either condition fails → result is **"Unknown gorilla"**.

---

## 2. Gallery JSON format

File: `gallery.json` (in app assets or user's model directory)

```json
{
  "version": "gorilla-v1",
  "species": "Gorilla beringei beringei",
  "embedding_dim": 768,
  "normalization": "megadescriptor",
  "similarity_metric": "cosine",
  "unknown_threshold": 0.469,
  "separability_gap": 0.435,
  "n_individuals": 66,
  "individuals": {
    "SB Mastaki": {
      "class_index": 0,
      "category": "SB",
      "embeddings": [
        [0.032, -0.011, ...],   // exemplar 1 — 768 floats, L2-normalised
        [0.041, -0.008, ...],   // exemplar 2
        ...                     // up to 25 exemplars
      ],
      "mean_intra": 0.812       // avg pairwise similarity among exemplars (quality indicator)
    },
    "JUV Kasuki": {
      "class_index": 1,
      "category": "JUV",
      "embeddings": [ [...], [...], ... ],
      "mean_intra": 0.764
    }
  }
}
```

### Fields per individual

| Field | Type | Description |
|---|---|---|
| `class_index` | int | Internal index (0-based, assigned at add time) |
| `category` | string | Age-sex class: `SB` (silverback), `ADF` (adult female), `SAF`/`SAM` (sub-adult), `JUV` (juvenile), `BB` (baby). `?` = unclassified. |
| `embeddings` | list[list[float]] | Up to 25 training exemplars. App uses MAX similarity. |
| `embedding` | list[float] | Alternative: single prototype (used for user-added individuals). The app supports both formats. |
| `mean_intra` | float | Average cosine similarity between all pairs of `embeddings`. Range ~0.65–0.95. Low value = inconsistent crops or mixed-identity error. |
| `field_embedding` | list[float] | Optional. Weighted average of user-added field photos. Never overwrites `embeddings`. |
| `field_crops` | int | How many field photos have been merged into `field_embedding`. Capped at 50. |

---

## 3. Adding a new individual (app flow)

Implemented in `AddIndividualViewModel.kt` + `GalleryManager.kt`.

### Step-by-step

1. User enters a name and selects 1–N photos (camera or gallery picker).
2. YOLO runs on each photo → detected gorilla faces → saved as JPEG crops.
3. Backbone runs on each crop → 768-dim embedding vector.
4. App averages all embeddings → **prototype vector** → L2-normalizes.
5. **Duplicate check**: if prototype similarity ≥ 0.82 to any existing individual → warning shown. User can proceed anyway, but this is a strong signal of a duplicate or mislabeled individual.
6. `GalleryManager.addIndividual(name, embeddings)`:
   - Computes prototype = average of embeddings
   - Stores as `"embedding": [...]` in gallery JSON
   - Does NOT store the individual exemplars (only prototype)
7. Gallery JSON is overwritten; previous version is backed up automatically.

### Hard limits and warnings

| Constraint | Value | Enforced by |
|---|---|---|
| Minimum crops (hard) | 10 | `AddIndividualViewModel.MIN_CROPS_HARD` — blocked |
| Minimum crops (warning) | 15 | `AddIndividualViewModel.MIN_CROPS_WARN` — shown to user |
| Duplicate detection threshold | 0.82 | `GalleryManager.findSimilarIndividuals` |
| Max field crops | 50 | `GalleryManager.MAX_FIELD_CROPS` |

---

## 4. Adding field crops to an existing individual

When the user runs "Add photos" for an individual already in the gallery, `GalleryManager.addFieldCrops()` is called. This **never touches the original `embeddings`** — it only updates `field_embedding`.

### Quality gate

Before any field crop is accepted, it is checked against the individual's **anchor** (the first training exemplar, or the stored `embedding` if user-added):

```
single photo (< 3 submitted):  cosine_sim(crop_emb, anchor) ≥ threshold × 0.65  ≈ 0.305
batch (≥ 3 submitted):         cosine_sim(avg_emb, anchor)  ≥ threshold × 0.45  ≈ 0.211
```

Crops that fail are **silently rejected**. If the whole batch fails → `QualityGateRejection` state is shown.

### Post-merge validation

After merging new embeddings into `field_embedding`, two checks run before writing to disk:

1. **Self-similarity**: `cosine_sim(merged, anchor) ≥ threshold` (0.469)  
   If not → the merged prototype has drifted too far from the individual's identity. Rejected.

2. **No false positives**: `max{ cosine_sim(merged, other_anchor) } < threshold`  
   If not → the merged prototype would cause another individual to be misidentified as this one. Rejected.

If either check fails, the field update is discarded entirely (no partial writes).

---

## 5. The dominance problem — and how to avoid it

**What it is**: if a new individual has low-quality, blurry, or generic photos, their embedding prototype sits in the "generic gorilla" zone — a region of the embedding space that is close to many individuals. During inference, gorillas that don't match anyone well will get classified as this individual instead of "Unknown". One dominant individual can quietly absorb all unrecognized gorillas.

**Concrete example**: 5 crops of a gorilla photographed from behind (face not visible). The backbone produces embeddings that are noisy and generic. Any unknown gorilla has a small chance of landing near this prototype, and because inference takes the max over all individuals, this individual "wins" many low-confidence queries.

### Safeguards already in the app

| Safeguard | Where | What it does |
|---|---|---|
| Minimum 10 crops (hard) | `AddIndividualViewModel` | Ensures the average is representative; single photos cannot be added |
| Duplicate warning at 0.82 | `GalleryManager.findSimilarIndividuals` | Catches identical or very similar existing individuals |
| Quality gate for field crops | `GalleryManager.addFieldCrops` | Rejects crops too different from the anchor |
| Post-merge self-similarity | `GalleryManager.addFieldCrops` | Ensures field prototype stays close to anchor |
| Post-merge false-positive check | `GalleryManager.addFieldCrops` | Prevents field prototype from matching another individual |
| Margin threshold 0.08 | `MegaDescriptorBackbone` | Rejects low-conviction identifications (top-2 scores too close) |

### What is NOT protected (gaps to fill)

1. **New individual quality gate**: `addIndividual` (for brand-new individuals) does **not** check whether the prototype is distinguishable from all existing individuals. If the user submits 10 photos from a bad angle and confirms despite the similarity warning, the bad individual gets added. The only protection is the duplicate warning at 0.82 — but a bad prototype may score 0.5–0.7 against many individuals without triggering it.

2. **Multiple individuals per photo**: YOLO detects one gorilla face per photo. If two gorillas are visible, YOLO picks the largest/most confident detection. If the user submits photos of the wrong gorilla by accident, the prototype is contaminated.

### Recommended minimum quality requirements for an AI implementing this feature

Before calling `addIndividual` or `addFieldCrops`, validate:

| Check | Minimum | How to verify |
|---|---|---|
| Face clearly visible | ≥80% of crops | YOLO confidence score ≥ 0.5 |
| Photo count | ≥ 15 crops (ideally) | Hard block at 10, warning below 15 |
| Intra-batch consistency | mean pairwise similarity ≥ 0.50 | Compute before adding |
| Max similarity to ANY existing individual | < 0.75 | `findSimilarIndividuals` at 0.75 threshold |
| Crop resolution after YOLO crop | ≥ 64×64 px (before resize to 224) | Check bbox area |

#### Computing intra-batch consistency

```kotlin
fun meanIntraSimilarity(embeddings: List<FloatArray>): Float {
    if (embeddings.size < 2) return 1.0f
    var sum = 0.0f
    var count = 0
    for (i in embeddings.indices) {
        for (j in i + 1 until embeddings.size) {
            sum += EmbeddingUtils.dotProduct(embeddings[i], embeddings[j])
            count++
        }
    }
    return sum / count
}
```

If `meanIntraSimilarity < 0.50` → the batch is inconsistent. Likely causes:
- Multiple different individuals mixed in (wrong labeling)
- Photos from very different conditions (some clear, some blurry/occluded)
- The gorilla is not a clear individual (e.g., a completely new animal never seen by the model)

A consistent batch with `mean_intra ≥ 0.70` typically produces a reliable prototype.

---

## 6. Patch sharing (multi-device sync)

Rangers can add new individuals on their device and share them with other devices via a **patch file** (JSON export).

### Export

```
GalleryManager.exportPatch(listOf("Name")) → File (patch_ranger_YYYYMMDD_HHMMSS.json)
```

Patch format:
```json
{
  "patch_version": "1.0",
  "device_name": "ranger_alice",
  "created_at": "2026-06-18T10:00:00",
  "embedding_dim": 768,
  "added_individuals": {
    "New Gorilla": {
      "embedding": [...],
      "num_crops": 12,
      "added_at": "2026-06-18T10:00:00"
    }
  }
}
```

### Import

`GalleryManager.importPatch(patchBytes)`:
- New individuals are added directly.
- Existing individuals: the patch embedding is validated against the local anchor (quality gate: `cosine_sim ≥ threshold × 0.5`), then merged into `field_embedding` via weighted average.
- Invalid names, wrong dimensions, or failed quality gates are silently skipped.
- A backup is always created before writing.

---

## 7. Backup and undo

Every gallery write is automatically backed up to `files/models/backups/gallery_YYYYMMDD_HHMMSS.json`. Up to 20 backups are kept (oldest deleted automatically).

```kotlin
galleryManager.undoLastGalleryChange()  // restores most recent backup
galleryManager.listBackupsWithDiff()    // returns list with diff vs current gallery
```

---

## 8. Export checklist (for deploying a new model version)

1. Run `prepare_gorilla_gallery.py` → converts training gallery to Android format, writes to `assets/gallery.json`.
2. Run `export_tflite_gorilla.py` (in WSL2, see the script header) which writes `megadesc_T_arcface_backbone.tflite` and `yolo_v2_detector.tflite` to `output/v1_gorilla/tflite/`; copy them into the app `assets/`.
3. Verify `unknown_threshold` in `gallery.json` matches the benchmark threshold (V1: **0.469**).
4. Verify `embedding_dim: 768` in `gallery.json`.
5. Verify `normalization: "megadescriptor"` in `gallery.json`.
6. Rebuild the Android app in Android Studio.

---

## 9. Key constants summary

| Constant | Value | File |
|---|---|---|
| `unknown_threshold` | 0.469 | `gallery.json` (V1 benchmark) |
| `MARGIN_THRESHOLD` | 0.08 | `MegaDescriptorBackbone.kt` |
| `MAX_FIELD_CROPS` | 50 | `GalleryManager.kt` |
| `SINGLE_GATE_RATIO` | 0.65 | `GalleryManager.kt` (single photo gate) |
| `BATCH_GATE_RATIO` | 0.45 | `GalleryManager.kt` (batch gate) |
| `BATCH_MIN_SIZE` | 3 | `GalleryManager.kt` |
| `MIN_CROPS_HARD` | 10 | `AddIndividualViewModel.kt` |
| `MIN_CROPS_WARN` | 15 | `AddIndividualViewModel.kt` |
| Duplicate warning | 0.82 | `GalleryManager.findSimilarIndividuals` |
| Embedding dim | 768 | MegaDescriptor-T-224 backbone |
| Input size | 224×224 px | MegaDescriptor-T-224 |
| Image mean / std | 0.5 / 0.5 | All channels |
