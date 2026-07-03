# Wild / background images

Put unlabeled images of your target species here, downloaded from the internet.

These are used as a **background class** during training. They teach the model what the species looks like in general, so it can tell apart "I recognise this individual" from "this is an individual I've never seen."

## How many?

| | Result |
|---|---|
| Minimum | **1,000 images** (training will work but rejection may be weak) |
| Recommended | **3,000 – 5,000 images** |
| This project used | images sources diverses (iNaturalist, GBIF, recherche web) |

## Where to find images

- [iNaturalist](https://www.inaturalist.org) (filter by species, export observations)
- [GBIF](https://www.gbif.org) (species occurrence data with photos)
- Google Images / Bing Images
- Any field photo database for your species

## Rules

- Images do **not** need labels. The folder name is enough (they all count as "background")
- Mix of angles, distances, lighting, quality → the more diverse the better
- Remove obvious non-faces (landscape shots with no animal visible)
- Any `.jpg`, `.jpeg`, `.png` is accepted

## After filling this folder

```
python v1_megadesc_arcface/train.py --extract
```

YOLO will detect faces in each image and save 224×224 crops to `data/crops/wild/`.
