# WandB Logging Specification

Complete specification for WandB integration in training scripts. All new models must follow this pattern.

## Config block

Add to the root Hydra YAML (`configs/<MODEL>.yaml`):

```yaml
wandb:
  enabled: true
  project: cable-<MODEL>
  entity: ${oc.env:WANDB_ENTITY,null}
  name: null
  tags: []
```

## Environment setup

Load the WandB API key from `.env` before initializing. The `.env` file contains `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT`:

```python
from dotenv import load_dotenv
load_dotenv()
```

This must run before `wandb.init()` so the API key is available. The project name `cable-<MODEL>` is set in the Hydra YAML config (not hardcoded in Python).

## Initialization

```python
wandb_run = None
if cfg.wandb.enabled:
    import wandb

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=OmegaConf.select(cfg, "wandb.entity"),
        name=OmegaConf.select(cfg, "wandb.name"),
        tags=list(cfg.wandb.get("tags", [])),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
```

Pass the full resolved Hydra config so every experiment is reproducible from its WandB run page.

## Metric namespaces

All metrics use `step=global_step` (not epoch) so curves align on a single x-axis.

### train/ (every `display` steps)

| Key | Type | Description |
|-----|------|-------------|
| `train/loss` | float | Total loss |
| `train/lr` | float | Current learning rate |
| `train/epoch` | int | Current epoch |
| `train/{component}_scaled` | float | Each scaled loss component (e.g., `train/loss_ce_scaled`) |
| `train/{component}_unscaled` | float | Each unscaled loss component |

Log every `train_config.display` optimizer steps:

```python
if wandb_run and global_step % tc.display == 0:
    log_dict = {
        "train/loss": loss_value,
        "train/lr": current_lr,
        "train/epoch": epoch,
    }
    for k, v in loss_dict_reduced_scaled.items():
        log_dict[f"train/{k}"] = v.item() if hasattr(v, "item") else v
    for k, v in loss_dict_reduced_unscaled.items():
        log_dict[f"train/{k}"] = v.item() if hasattr(v, "item") else v
    wandb_run.log(log_dict, step=global_step)
```

### val/ (every `val_interval` epochs)

| Key | Type | Description |
|-----|------|-------------|
| `val/loss` | float | Total validation loss |
| `val/epoch` | int | Current epoch |
| `val/{component}_scaled` | float | Each scaled loss component |

```python
if wandb_run:
    val_log = {"val/loss": val_loss_value, "val/epoch": epoch}
    for k, v in val_reduced_scaled.items():
        val_log[f"val/{k}"] = v.item() if hasattr(v, "item") else v
    wandb_run.log(val_log, step=global_step)
```

### eval/ (every `image_log_interval` epochs)

#### Model-native metrics

Depends on the model's output type (classified in Phase 0a of SKILL.md):

**Segmentation models:**

| Key | Type | Description |
|-----|------|-------------|
| `eval/iou` | float | Intersection over union |
| `eval/dice` | float | Dice coefficient |

**Detection models:**

| Key | Type | Description |
|-----|------|-------------|
| `eval/mAP` | float | Mean average precision |
| `eval/iou` | float | Mask IoU |

**Polyline models:**

| Key | Type | Description |
|-----|------|-------------|
| `eval/centerline_err_px` | float | Mean centerline error in pixels (Hungarian-matched GT/pred pairs) |
| `eval/matched` | int | Number of GT lanes matched to predictions |
| `eval/total_gt` | int | Total GT lanes in eval set |
| `eval/total_pred` | int | Total predicted lanes |

#### Pipeline centerline metric (all model types)

Required for cross-model comparison. Converts model output to polyline format, then computes centerline error:

| Key | Type | Description |
|-----|------|-------------|
| `eval/pipeline_centerline_err_px` | float | Mean centerline error after converting to common polyline format |
| `eval/pipeline_matched` | int | Number of matched GT/pred pairs |
| `eval/pipeline_total_gt` | int | Total GT in eval set |
| `eval/pipeline_total_pred` | int | Total predictions |

Compute via `_centerline_error(gt_lanes, pred_lanes, img_w)` which returns:
```python
{"mean_px": float, "max_px": float, "num_gt": int, "num_pred": int, "num_matched": int, "per_lane": list[float]}
```

### val/predictions (every `image_log_interval` epochs)

Log a list of `wandb.Image` objects — one per sample — each showing a side-by-side GT vs Pred panel:

```python
if wandb_run and epoch % tc.image_log_interval == 0:
    overlays = _build_overlay(validation_db, nnet, val_overlay_indices, img_h, img_w)
    if overlays:
        wandb_run.log(
            {"val/predictions": [wandb.Image(rgb, caption=cap) for cap, rgb in overlays]},
            step=global_step,
        )
```

**Panel layout:**
- **Left (GT):** Thermal frame with GT lane points drawn as colored circles, labeled "GT"
- **Right (Pred):** Thermal frame with predicted curves drawn as green polylines, labeled with error stats (e.g., `Pred err=4.2px (3/3)`)
- **Caption:** includes the sample index

### train/predictions_final (final epoch only)

For overfit scripts, log final-epoch overlays separately for post-training review:

```python
if wandb_run and epoch == tc.max_epochs - 1:
    wandb_run.log(
        {"train/predictions_final": [wandb.Image(rgb, caption=cap) for cap, rgb in overlays]},
        step=global_step,
    )
```

## Summary

At training end, optionally log summary scalars:

```python
if wandb_run:
    wandb_run.summary["best_val_loss"] = best_val_loss
    wandb_run.summary["final_centerline_err_px"] = final_err
    wandb_run.finish()
```
