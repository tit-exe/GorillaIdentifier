# models/

This folder is empty by design and not tracked by git (see `.gitignore`). It holds two things, and only one of them needs a manual download.

## yolo_gorilla.pt (download required)

The gorilla face detector, a custom YOLOv8 model trained specifically to find gorilla faces in field photos. It is hosted on HuggingFace at [tit0000/GorillaIdentifier](https://huggingface.co/tit0000/GorillaIdentifier). Fetch it with:

```bash
python models/download_models.py
```

File: `models/yolo_gorilla.pt`, about 18 MB. Required before running `v1_megadesc_arcface/train.py --extract`.

## MegaDescriptor-T-224 (downloaded automatically, no action needed)

The re-identification backbone that turns a face crop into a 768-dimensional embedding. It is not stored in this folder. `timm` downloads it automatically from the HuggingFace Hub (`BVRA/MegaDescriptor-T-224`) the first time training runs, and caches it locally for reuse.

By default the cache goes to the standard OS location, `~/.cache/huggingface`. To redirect it, for example to a larger drive, uncomment `hf_cache_dir` in `config.yaml`.
