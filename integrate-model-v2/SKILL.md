---
name: integrate-model
description: Integrate a newly ported ML model into the cable segmentation codebase. Walks through discovery, channel adaptation, Hydra configs, dataset conversion, training with WandB logging, inference, overfit test, and unit tests. Use when a new model has been pasted/vendored and needs to be wired into the project.
---

# Integrate Model

Wire a newly ported/vendored model into the cable segmentation pipeline.

## Prerequisites

Before starting, confirm with the user:
1. The model source code is already vendored under `src/cable_segmentation/models/<model>/`
2. The user has a name for the model (used in configs, entrypoints, filenames)

Use `<MODEL>` as placeholder below. Replace with the actual model name (e.g., `lstr`, `clrnet`).

## Integration checklist

Work through each phase sequentially. Mark items done as you complete them.

---

### Phase 0: Discovery

Classify the model and understand its data expectations before writing any code.

#### 0a. Output type classification

Read the model's forward pass and determine which category it falls into:

- **Segmentation** — outputs pixel-level masks (binary or multi-class)
- **Detection** — outputs per-object bounding boxes + masks + confidence scores
- **Polyline/keypoint** — outputs coordinate sequences (polylines, polynomials, heatmaps decoded to points)

Record the output type. It determines:
- Which **model-native eval metric** to use (Phase 5)
- What the **loss review** should focus on (Phase 5)
- How the **inference module** converts output to the common pipeline format (Phase 6)

#### 0b. Model data format inspection

The source dataset is always **COCO format** (`_annotations.coco.json` per split). Read the vendored model's dataset/dataloader code and document:

- [ ] What file format does the model expect? (JSON-lines, txt, image+mask pairs, etc.)
- [ ] What coordinate system? (normalized 0-1, pixel coords, image-relative)
- [ ] What directory structure? (single dir, split dirs, nested structure)
- [ ] What image format/dtype does it expect? (PNG uint8, float32 tensors, etc.)

These answers drive the dataset converter in Phase 4.

#### 0c. Input stem identification

Find the model's first convolutional layer (the one that accepts the raw image input). Record:

- [ ] Layer name/path (e.g., `backbone.conv1`, `patch_embed.proj`)
- [ ] Current shape: `Conv2d(3, C, kernel_size=K, ...)`
- [ ] Whether it's a simple Conv2d or something more complex (patch embedding, stem block)

This drives the channel adaptation in Phase 2.

---

### Phase 1: Dependencies & entrypoints

**File:** `pyproject.toml`

- [ ] Add `[project.optional-dependencies.<MODEL>]` group with model-specific packages
- [ ] Add CLI entrypoints under `[project.scripts]`:
  - `cable-seg-<MODEL>-train` — training
  - `cable-seg-<MODEL>-train-test` — overfit test
  - `cable-seg-<MODEL>-convert` — dataset conversion (if needed)
- [ ] Run `pip3 install -e ".[<MODEL>]"` to verify

Reference pattern — see existing `lstr` and `rfdetr` groups in pyproject.toml.

---

### Phase 2: Input channel adaptation

Modify the model to accept single-channel thermal input instead of 3-channel RGB.

**Strategy:** Replace the first conv layer identified in Phase 0c. Always modify the architecture — never replicate grayscale 3x.

- [ ] Replace `Conv2d(3, C, ...)` with `Conv2d(1, C, ...)` in the vendored model code
- [ ] For pretrained weight loading: use `load_state_dict(checkpoint, strict=False)` (RF-DETR pattern)
  - The modified first-conv weights will be skipped automatically due to shape mismatch
  - The new 1-channel conv initializes with default PyTorch init
  - All other pretrained weights load normally
- [ ] Add `input_channels: 1` to the model's Hydra config (Phase 3) so this is configurable
- [ ] Verify the model's forward pass completes with a `(1, 1, H, W)` dummy tensor

**Normalization:** Use ImageNet stats for the single thermal channel, matching RF-DETR's approach. The pretrained backbone expects ImageNet-scale activations:
```python
# ImageNet stats applied to single-channel thermal
mean = [0.485]
std = [0.229]
```

---

### Phase 3: Hydra configs

**Directory:** `src/cable_segmentation/configs/`

#### 3a. Root config

- [ ] Create root config `<MODEL>.yaml`:
  ```yaml
  defaults:
    - <MODEL>_model@model_config: default
    - <MODEL>_train@train_config: default
    - _self_

  dataset: THERMAL_CABLE
  input_size: [360, 640]
  input_channels: 1

  data_dir: ${hydra:runtime.cwd}/assets/<MODEL>_dataset
  cache_dir: ${hydra:runtime.cwd}/cache/<MODEL>
  result_dir: ${hydra:runtime.cwd}/results/<MODEL>

  wandb:
    enabled: true
    project: cable-<MODEL>
    entity: ${oc.env:WANDB_ENTITY,null}
    name: null
    tags: []
  ```

#### 3b. Model config

- [ ] Create `<MODEL>_model/default.yaml` with architecture params (`# @package model_config`)
- [ ] Include `input_channels: 1` in the model config

#### 3c. Train config

- [ ] Create `<MODEL>_train/default.yaml` with training hyperparams (`# @package train_config`)

#### 3d. Augmentation config

- [ ] Create `<MODEL>_aug/default.yaml` with Albumentations augmentations (`# @package train_config`):
  ```yaml
  # @package train_config
  aug_config:
    # Photometric (do not affect geometry — safe for all annotation types)
    RandomBrightnessContrast:
      brightness_limit: 0.2
      contrast_limit: 0.2
      p: 0.5
    GaussianBlur:
      blur_limit: 3
      p: 0.3
    GaussNoise:
      std_range: [0.01, 0.05]
      p: 0.3
    CLAHE:
      clip_limit: 4.0
      tile_grid_size: [8, 8]
      p: 0.3
    # Geometric (must transform annotations alongside images)
    HorizontalFlip:
      p: 0.5
    Rotate:
      limit: 15
      border_mode: 0
      p: 0.3
    Affine:
      scale: [0.9, 1.1]
      translate_percent: 0.05
      p: 0.3
  ```
- [ ] Wire the augmentation config into the root config defaults list

#### 3e. Optional configs

- [ ] Optionally create `<MODEL>_convert.yaml` for the dataset converter
- [ ] Optionally create `<MODEL>_overfit.yaml` for overfit testing

---

### Phase 4: Dataset converter

**File:** `src/cable_segmentation/<MODEL>_convert.py`

Converts COCO-format annotations to the model's expected format (determined in Phase 0b).

- [ ] Hydra entrypoint: `@hydra.main(version_base="1.3", config_path="configs", config_name="<MODEL>_convert")`
- [ ] Input: COCO-format annotations (`_annotations.coco.json` per split)
- [ ] Output: model-specific format (as documented in Phase 0b)
- [ ] Handle train/valid/test splits
- [ ] Use `hydra.utils.to_absolute_path()` for path resolution
- [ ] Use `logging` module, not `print()`

---

### Phase 5: Training script with WandB

**File:** `src/cable_segmentation/<MODEL>_train.py`

See [WANDB.md](WANDB.md) for the complete WandB logging specification.

#### 5a. Core training setup

- [ ] Hydra entrypoint: `@hydra.main(version_base="1.3", config_path="configs", config_name="<MODEL>")`
- [ ] Load WandB API key from `.env` file using `dotenv`: `from dotenv import load_dotenv; load_dotenv()`
- [ ] WandB project name must be `cable-<MODEL>` (set in `configs/<MODEL>.yaml`, not hardcoded in Python)
- [ ] WandB init with full resolved config
- [ ] Always fine-tune from pretrained weights — load with `strict=False` (RF-DETR pattern)
- [ ] LR schedule: linear warmup + cosine decay (see `warmup_epochs`, `min_lr` in train config)
- [ ] Checkpoint saving at `snapshot_interval` epochs
- [ ] All metrics use `step=global_step` (not epoch)
- [ ] Use `logging` module, not `print()`

#### 5b. Loss function review

Before writing the training loop, read the model's loss function and understand its components:

- [ ] List all loss components and their weights
- [ ] Flag components that may need rebalancing for single-class cable detection (e.g., classification losses weighted for many classes, auxiliary losses for multi-class scenarios)
- [ ] Expose all loss weights as configurable values in `<MODEL>_train/default.yaml` so they can be tuned without code changes
- [ ] Add comments noting which weights may need adjustment

Do not tune loss weights now — defer to after seeing initial training runs.

#### 5c. Metric logging

**Train metrics** (every `display` steps, `train/` namespace):
- [ ] loss, lr, epoch, per-component losses

**Validation metrics** (every `val_interval` epochs, `val/` namespace):
- [ ] loss, epoch, per-component losses

**Model-native eval metric** (every `image_log_interval` epochs, `eval/` namespace):

Depends on output type classified in Phase 0a:
- **Segmentation models:** `eval/iou`, `eval/dice`
- **Detection models:** `eval/mAP`, `eval/iou`
- **Polyline models:** `eval/centerline_err_px`, `eval/matched`, `eval/total_gt`, `eval/total_pred`

- [ ] Log the appropriate model-native metric for the output type

**Pipeline centerline metric** (every `image_log_interval` epochs, `eval/` namespace):

Required for all model types to enable cross-model comparison:
- [ ] Convert model output to polyline format (using the inference module's conversion logic)
- [ ] Compute `eval/pipeline_centerline_err_px` via Hungarian matching
- [ ] Log `eval/pipeline_matched`, `eval/pipeline_total_gt`, `eval/pipeline_total_pred`

**Visual overlays** (every `image_log_interval` epochs, `val/predictions`):
- [ ] Side-by-side GT vs Pred panels as `wandb.Image`

---

### Phase 6: Inference module

**File:** `src/cable_segmentation/<MODEL>_inference.py`

- [ ] `load_<MODEL>_model(checkpoint, *, device="cuda", **kwargs) -> torch.nn.Module`
  - Load with `strict=False` (RF-DETR pattern)
  - Strip DataParallel prefixes if present
  - Return model in eval mode
- [ ] `preprocess_frame(frame, input_size=(360, 640)) -> torch.Tensor`
  - Handle uint16 (raw thermal), uint8, float inputs
  - Normalize with ImageNet stats: mean=0.485, std=0.229 (single channel)
  - Return `(1, 1, H, W)` tensor
- [ ] `infer_frame(model, frame, *, input_size, confidence_threshold=0.5, device="cuda") -> list[dict]`
  - Each track dict must contain:
    ```python
    {
        "polyline": [[col, row], ...],          # for frontend rendering
        "coordinates": np.ndarray (N, 2),       # (row, col) for temperature extraction
        "confidence": float,
        "class_name": "cable",                  # or model-specific class
    }
    ```
  - The `coordinates` field is `(row, col)` order; `polyline` is `[col, row]` order
  - For non-polyline models (segmentation, detection), convert the native output to this common format

---

### Phase 7: Overfit training test

**Files:** `src/cable_segmentation/<MODEL>_train_test.py`, `configs/<MODEL>_overfit.yaml`

A standalone script that trains on N images and validates on the **same** images. Proves the model can memorize a tiny dataset before investing in full training runs.

#### Overfit config (`configs/<MODEL>_overfit.yaml`)

- [ ] Create config with defaults composing `<MODEL>_model@model_config: default` (no train config — overfit uses its own params)
- [ ] Include these settings:
  ```yaml
  # Overfit test settings
  num_images: 8          # number of images to memorize
  max_epochs: 50         # iterations/epochs to run
  batch_size: 1          # typically 1 for overfit
  learning_rate: 0.0005  # may need tuning per model
  overlay_interval: 100  # how often to log visual overlays
  log_interval: 10       # how often to log scalar metrics
  ```
- [ ] Disable dropout in model config override: `model_config: { drop_out: 0.0 }`
- [ ] WandB config with `name: <MODEL>-overfit-test` and `tags: [overfit-test]`

#### Overfit script (`<MODEL>_train_test.py`)

- [ ] Hydra entrypoint: `@hydra.main(version_base="1.3", config_path="configs", config_name="<MODEL>_overfit")`
- [ ] Load `.env` for WandB API key (use `_load_dotenv()` pattern — walk parent dirs to find `.env`)
- [ ] Load the dataset and truncate to `num_images` samples
- [ ] Pre-build the full batch so every gradient step sees **all** images (prevents catastrophic interference from per-image updates)
- [ ] Simple single-process training loop (no queues, no prefetching)
- [ ] **Scalar logging** (every `log_interval` steps): `train/loss`, `train/iter`, per-component losses
- [ ] **Overlay + eval logging** (every `overlay_interval` steps):
  - Build side-by-side GT vs Pred panels (same format as Phase 5 overlays)
  - Compute model-native metric + pipeline centerline error
  - Log metrics and `train/predictions` as list of `wandb.Image`
  - Save overlay PNGs locally to `result_dir/overlays/`
- [ ] **Final evaluation** after training completes:
  - Log `eval/final_centerline_err_px`
  - Log `train/predictions_final` as separate wandb images for post-training review
  - Print per-image error breakdown to logger
- [ ] `wandb_run.finish()` at end

---

### Phase 8: Tests

**File:** `tests/test_<MODEL>.py`

- [ ] **Forward pass**: verify output tensor shapes for single and batch inputs
- [ ] **Preprocessing**: test uint16, uint8, float, and RGB→grayscale conversion
- [ ] **Inference**: verify `infer_frame()` returns correct track dict structure (polyline, coordinates, confidence, class_name)
- [ ] **Pipeline integration**: test `build_frame_data_from_polylines()` with model output
- [ ] **Temperature extraction**: verify sampling along polyline coordinates
- [ ] Run `pytest tests/test_<MODEL>.py` to confirm all pass

---

### Phase 9: Final verification

- [ ] `pip3 install -e ".[<MODEL>]"` succeeds
- [ ] `cable-seg-<MODEL>-train --help` shows Hydra config options
- [ ] `cable-seg-<MODEL>-train-test --help` shows overfit config options
- [ ] `ruff check src/cable_segmentation/<MODEL>_*.py tests/test_<MODEL>.py`
- [ ] `ruff format src/cable_segmentation/<MODEL>_*.py tests/test_<MODEL>.py`
