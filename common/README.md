# common/

## review_crops.py

Graphical tool for manual review of face crops.

```bash
python common/review_crops.py
```

Drag and drop crops from your file explorer onto the window.

**Controls:**
- `Enter`: validate the crop
- `S`: skip
- `Del` / `X`: delete (file + JSON entry)
- `R`: reject (blurry, bad angle, partial face)
- `A` / `←`: previous
- `Q`: quit

## config_loader.py

Centralizes all paths and parameters read from `config.yaml`.

```python
from common.config_loader import REPO_ROOT, PHOTOS_DIR, CROPS_JSON, MODELS_DIR, apply_cache_env
apply_cache_env()  # call before importing timm / huggingface_hub
```
