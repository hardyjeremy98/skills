---
name: integrate-model
description: Integrate a newly ported ML model into the cable segmentation codebase. Walks through pyproject.toml setup, Hydra configs, training script with WandB logging, inference module, dataset converter, overfit test, and unit tests. Use when a new model has been pasted/vendored and needs to be wired into the project.
---

# Integrate Model

Wire a newly ported/vendored model into the cable centerline prediction pipeline.

## Prerequisites

Before starting, confirm with the user:
1. The model source code is already vendored under `src/cable_segmentation/models/<model>/`
2. The user knows what the model's output format is (polylines, polynomials, heatmaps, etc.)
3. The user has a name for the model (used in configs, entrypoints, filenames)

Use `<MODEL>` as placeholder below. Replace with the actual model name (e.g., `lstr`, `clrnet`).

## Integration checklist

Work through each phase sequentially. Mark items done as you complete them.

### Phase 1: Dependencies & entrypoints

**File:** `pyproject.toml`

- [ ] Add `[project.optional-dependencies.<MODEL>]` group with model-specific packages
- [ ] Add CLI entrypoints under `[project.scripts]`:
  - `cable-seg-<MODEL>-train` — training
  - `cable-seg-<MODEL>-train-test` — overfit test
  - `cable-seg-<MODEL>-convert` — dataset conversion (if needed)
- [ ] Run `pip3 install -e ".[<MODEL>]"` to verify

Reference pattern — see existing `lstr` and `rfdetr` groups in pyproject.toml.

### Phase 2: Hydra configs

**Directory:** `src/cable_segmentation/configs/`

- [ ] Create root config `<MODEL>.yaml` with:
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
- [ ] Create `<MODEL>_model/default.yaml` with architecture params (`# @package model_config`)
- [ ] Create `<MODEL>_train/default.yaml` with training hyperparams (`# @package train_config`)
- [ ] Optionally create `<MODEL>_convert.yaml` for the dataset converter
- [ ] Optionally create `<MODEL>_overfit.yaml` for overfit testing

### Phase 3: Dataset converter

**File:** `src/cable_segmentation/<MODEL>_convert.py`

- [ ] Hydra entrypoint: `@hydra.main(version_base="1.3", config_path="configs", config_name="<MODEL>_convert")`
- [ ] Input: COCO-format annotations from Roboflow (`_annotations.coco.json` per split)
- [ ] Output: model-specific format (e.g., JSON-lines with lane coords)
- [ ] Handle train/valid/test splits
- [ ] Use `hydra.utils.to_absolute_path()` for path resolution
- [ ] Use `logging` module, not `print()`

### Phase 4: Training script with WandB

**File:** `src/cable_segmentation/<MODEL>_train.py`

See [WANDB.md](WANDB.md) for the complete WandB logging specification.

- [ ] Hydra entrypoint: `@hydra.main(version_base="1.3", config_path="configs", config_name="<MODEL>")`
- [ ] Load WandB API key from `.env` file using `dotenv`: `from dotenv import load_dotenv; load_dotenv()`
- [ ] WandB project name must be `cable-<MODEL>` (set in `configs/<MODEL>.yaml`, not hardcoded in Python)
- [ ] WandB init with full resolved config
- [ ] **Train metrics** (every `display` steps, `train/` namespace): loss, lr, epoch, per-component losses
- [ ] **Validation metrics** (every `val_interval` epochs, `val/` namespace): loss, epoch, per-component losses
- [ ] **Eval metrics** (every `image_log_interval` epochs, `eval/` namespace): centerline_err_px, matched, total_gt, total_pred
- [ ] **Visual overlays** (every `image_log_interval` epochs, `val/predictions`): side-by-side GT vs Pred panels as `wandb.Image`
- [ ] LR schedule: linear warmup + cosine decay (see `warmup_epochs`, `min_lr` in train config)
- [ ] Checkpoint saving at `snapshot_interval` epochs
- [ ] All metrics use `step=global_step` (not epoch)
- [ ] Use `logging` module, not `print()`

### Phase 5: Inference module

**File:** `src/cable_segmentation/<MODEL>_inference.py`

- [ ] `load_<MODEL>_model(checkpoint, *, device="cuda", **kwargs) -> torch.nn.Module`
  - Strip DataParallel prefixes if present
  - Return model in eval mode
- [ ] `preprocess_frame(frame, input_size=(360, 640)) -> torch.Tensor`
  - Handle uint16 (raw thermal), uint8, float inputs
  - Normalize: `(img - 0.5) / 0.25` (thermal stats)
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

### Phase 6: Overfit training test

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
  - Build side-by-side GT vs Pred panels (same format as Phase 4 overlays)
  - Compute centerline error via Hungarian matching
  - Log `eval/centerline_err_px`, `eval/matched`, `eval/total_gt`, `eval/total_pred`
  - Log `train/predictions` as list of `wandb.Image`
  - Save overlay PNGs locally to `result_dir/overlays/`
- [ ] **Final evaluation** after training completes:
  - Log `eval/final_centerline_err_px`
  - Log `train/predictions_final` as separate wandb images for post-training review
  - Print per-image error breakdown to logger
- [ ] `wandb_run.finish()` at end

### Phase 7: Tests

**File:** `tests/test_<MODEL>.py`

- [ ] **Forward pass**: verify output tensor shapes for single and batch inputs
- [ ] **Preprocessing**: test uint16, uint8, float, and RGB→grayscale conversion
- [ ] **Inference**: verify `infer_frame()` returns correct track dict structure
- [ ] **Pipeline integration**: test `build_frame_data_from_polylines()` with model output
- [ ] **Temperature extraction**: verify sampling along polyline coordinates
- [ ] Run `pytest tests/test_<MODEL>.py` to confirm all pass

### Phase 8: Final verification

- [ ] `pip3 install -e ".[<MODEL>]"` succeeds
- [ ] `cable-seg-<MODEL>-train --help` shows Hydra config options
- [ ] `cable-seg-<MODEL>-train-test --help` shows overfit config options
- [ ] `ruff check src/cable_segmentation/<MODEL>_*.py tests/test_<MODEL>.py`
- [ ] `ruff format src/cable_segmentation/<MODEL>_*.py tests/test_<MODEL>.py`
