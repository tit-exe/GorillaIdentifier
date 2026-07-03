# Field photos

Put your own field photos here — **one subfolder per individual**. This folder is empty by design (see `.gitignore`): nobody's field photos belong in a public git history.

## Structure

```
data/photos/
    SB Humba/
        IMG_001.jpg
        IMG_002.jpg
        ...
    ADF Anangana/
        IMG_001.jpg
        ...
    JUV Bakunzi/
        ...
```

- One folder per individual, folder name = `{PREFIX} {Name}` (prefix optional, see below)
- Any number of photos per file name — the exact filename doesn't matter
- Photos do **not** need to be pre-cropped — YOLO detects and crops faces automatically
- Accepted formats: `.jpg`, `.jpeg`, `.png`

## Naming convention (prefix)

The prefix controls how many ArcFace "sub-centers" the individual gets during training (more sub-centers = more tolerance for appearance variation, useful for young individuals whose look changes fast):

| Prefix | Category | Sub-centers |
|---|---|---|
| `SB` | Silverback | 1 |
| `ADF` | Adult Female | 1 |
| `AD` | Adult | 1 |
| `SAF` / `SAM` | Subadult Female / Male | 1 |
| `JUV` | Juvenile | 2 |
| `BB` | Baby / Blackback | 2 |
| *(none)* | Unspecified | 1 |

If you're adapting this pipeline to another species, prefixes are optional — just keep one folder per individual.

## How many photos?

**≥ 30 photos per individual**, varied angles and conditions, is the practical target used for the V1 model (66 individuals, 93.0% top-1 accuracy). Fewer photos still work but recognition quality drops.

## Next step

Once this folder is filled:

```
python v1_megadesc_arcface/train.py --min-photos 5   # optional: scan report
python v1_megadesc_arcface/train.py --extract         # extract face crops
python common/review_crops.py                         # review crops (recommended)
python v1_megadesc_arcface/train.py                    # train
```

See the main `README.md` for the full pipeline.
